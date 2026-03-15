"""
Ensembling techniques: Rank averaging, recall-first, and optimization.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from mallorn.config import cfg
from mallorn.training.thresholding import find_optimal_threshold

def ensemble_recall_first(oof_preds, test_preds, y_train):
    """
    Recall-first ensemble: Use NN as recall gatekeeper.
    
    If NN says positive with moderate confidence, trust it even if trees are uncertain.
    This exploits NN's superior recall while maintaining precision from trees.
    
    Returns (oof_ensemble, test_ensemble)
    """
    # Get NN predictions (highest F1 model based on diagnostics)
    nn_oof = oof_preds.get('nn', np.zeros(len(y_train)))
    nn_test = test_preds.get('nn', np.zeros(len(list(test_preds.values())[0])) if test_preds else np.zeros(0))
    
    # Get tree model average (LGBM, XGB, RF)
    tree_models = ['lgbm', 'xgb', 'rf']
    active_trees = [m for m in tree_models if m in oof_preds and np.any(oof_preds[m] != 0)]
    
    if active_trees:
        tree_oof = np.mean([oof_preds[m] for m in active_trees], axis=0)
        tree_test = np.mean([test_preds[m] for m in active_trees], axis=0)
    else:
        tree_oof = nn_oof
        tree_test = nn_test
    
    # Recall gate: If NN > 0.4 AND trees > 0.25, boost to max(current, 0.6)
    # This captures TDEs that trees are uncertain about but NN detects
    boost_mask_oof = (nn_oof > 0.4) & (tree_oof > 0.25)
    boost_mask_test = (nn_test > 0.4) & (tree_test > 0.25)
    
    # Base ensemble: weight NN higher (0.45 vs 0.35 for trees, 0.20 for XGB)
    xgb_oof = oof_preds.get('xgb', tree_oof)
    xgb_test = test_preds.get('xgb', tree_test)
    
    final_oof = 0.45 * nn_oof + 0.35 * tree_oof + 0.20 * xgb_oof
    final_test = 0.45 * nn_test + 0.35 * tree_test + 0.20 * xgb_test
    
    # Apply recall boost - floor probabilities at 0.6 where gate is triggered
    final_oof[boost_mask_oof] = np.maximum(final_oof[boost_mask_oof], 0.55)
    if len(final_test) > 0:
        final_test[boost_mask_test] = np.maximum(final_test[boost_mask_test], 0.55)
    
    # Report stats
    n_boosted_oof = np.sum(boost_mask_oof)
    n_boosted_test = np.sum(boost_mask_test) if len(boost_mask_test) > 0 else 0
    print(f"\n[TARGET] Recall-first ensemble:")
    print(f"   OOF samples boosted: {n_boosted_oof} ({100*n_boosted_oof/len(final_oof):.1f}%)")
    if len(final_test) > 0:
        print(f"   Test samples boosted: {n_boosted_test} ({100*n_boosted_test/len(final_test):.1f}%)")
    
    return final_oof, final_test


def optimize_ensemble_weights(oof_preds, y_true):
    """Learn optimal ensemble weights via scipy optimization.
    
    Uses Nelder-Mead to find model weights that maximize F1 score on OOF predictions,
    replacing simple averaging with learned weights for better ensemble performance.
    """
    from scipy.optimize import minimize
    
    active_models = [k for k, v in oof_preds.items() if np.any(v != 0)]
    
    # Phase 6: Prune weak models (Anti-Drag)
    model_f1s = {}
    for m in active_models:
        _, f1 = find_optimal_threshold(y_true, oof_preds[m])
        model_f1s[m] = f1
    
    if model_f1s:
        best_f1 = max(model_f1s.values())
        original_count = len(active_models)
        # Prune models > 5% worse than best
        active_models = [m for m in active_models if model_f1s[m] >= best_f1 - 0.05]
        if len(active_models) < original_count:
            print(f" [CUT]️  Pruned {original_count - len(active_models)} weak models. Kept: {active_models}")

    if len(active_models) <= 1:
        return {active_models[0]: 1.0} if active_models else {}
    
    pred_matrix = np.column_stack([oof_preds[k] for k in active_models])
    
    def neg_f1(weights):
        weights = np.abs(weights) / (np.abs(weights).sum() + 1e-10)
        ensemble = pred_matrix @ weights
        _, f1 = find_optimal_threshold(y_true, ensemble)
        return -f1
    
    n_models = len(active_models)
    result = minimize(neg_f1, np.ones(n_models) / n_models, 
                     method='Nelder-Mead', options={'maxiter': 500})
    
    optimal_weights = np.abs(result.x) / (np.abs(result.x).sum() + 1e-10)
    weight_dict = dict(zip(active_models, optimal_weights))
    
    print(f"\n[TARGET] Optimized ensemble weights:")
    for model, weight in sorted(weight_dict.items(), key=lambda x: -x[1]):
        print(f"   {model}: {weight:.3f}")
    
    return weight_dict


def ensemble_rank_based(oof_preds, test_preds, active_models=None):
    """
    Robust Rank-Based Ensemble.
    
    Averages prediction RANKS instead of raw probabilities.
    Critical for models with different calibration (e.g., LGBM vs NN).
    
    Returns (oof_ranks, test_ranks) normalized 0-1.
    """
    if active_models is None:
        active_models = [k for k, v in oof_preds.items() if np.any(v != 0)]
    
    if not active_models:
        return np.zeros(len(list(oof_preds.values())[0])), np.zeros(len(list(test_preds.values())[0]))
    
    # OOF Ranks
    oof_ranks = np.zeros(len(list(oof_preds.values())[0]))
    for m in active_models:
        oof_ranks += rankdata(oof_preds[m]) / len(oof_preds[m])
    oof_ranks /= len(active_models)
    
    # Test Ranks
    test_ranks = np.zeros(len(list(test_preds.values())[0]))
    for m in active_models:
        test_ranks += rankdata(test_preds[m]) / len(test_preds[m])
    test_ranks /= len(active_models)
    
    print(f"\n[STATS] Rank-Based Ensemble combining: {active_models}")
    return oof_ranks, test_ranks


def train_stacking_with_calibration(oof_preds, y, test_preds, groups=None):
    """Stacking with probability calibration."""
  
    # Identical model set for Train/Test to prevent feature mismatch
    # Only include models that have non-zero OOF predictions AND exist in test set
    active_models = sorted([
        k for k in oof_preds.keys() 
        if k in test_preds and np.any(oof_preds[k] != 0)
    ])
    
    if not active_models:
        print("[WARN] No valid models for stacking! Returning zeros.")
        return np.zeros(len(y)), np.zeros(len(test_preds[list(test_preds.keys())[0]]))

    meta_train = np.column_stack([oof_preds[k] for k in active_models])
    meta_test = np.column_stack([test_preds[k] for k in active_models])
  
    print(f"\n{'='*70}")
    print("STACKING & CALIBRATION")
    print(f"{'='*70}")
    print(f"Stacking Models: {active_models}")
    print(f"Meta-features: {meta_train.shape}")
  
    # Meta-learners (Phase 6: Added XGB)
    meta_learners = {
        'logistic': LogisticRegression(
            class_weight='balanced', C=0.1, max_iter=1000, random_state=cfg.SEED
        ),
        'lgbm': lgb.LGBMClassifier(
            num_leaves=15, max_depth=5, learning_rate=0.05,
            n_estimators=100, class_weight='balanced', random_state=cfg.SEED,
            verbose=-1
        ),
        'xgb': xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=9.0, random_state=cfg.SEED, eval_metric='logloss',
            n_jobs=1
        )
    }
  
    if groups is not None:
         # Use same CV strategy as base models to maintain group integrity
         splits = []
         print(" [SHIELD]  Using Group CV for Stacking")
         for r in range(cfg.N_REPEATS):
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED + r)
            splits.extend(list(sgkf.split(meta_train, y, groups=groups)))
    else:
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg.N_REPEATS, random_state=cfg.SEED)
        splits = list(skf.split(meta_train, y))
        
    results = {}
    
    # Impute missing values (handling potential NaNs from base models)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    meta_train = imputer.fit_transform(meta_train)
    meta_test = imputer.transform(meta_test)
  
    for name, learner in meta_learners.items():
        oof = np.zeros(len(y))
        test = np.zeros(len(meta_test))
      
        for train_idx, val_idx in splits:
            # Calibration if configured
            if getattr(cfg, 'USE_CALIBRATION', False):
                calibrated = CalibratedClassifierCV(learner, method='isotonic', cv=3)
                calibrated.fit(meta_train[train_idx], y[train_idx])
                oof[val_idx] = calibrated.predict_proba(meta_train[val_idx])[:, 1]
                test += calibrated.predict_proba(meta_test)[:, 1] / (5 * getattr(cfg, 'N_REPEATS', 3))
            else:
                learner.fit(meta_train[train_idx], y[train_idx])
                oof[val_idx] = learner.predict_proba(meta_train[val_idx])[:, 1]
                test += learner.predict_proba(meta_test)[:, 1] / (5 * getattr(cfg, 'N_REPEATS', 3))
      
        _, f1 = find_optimal_threshold(y, oof)
        auc = roc_auc_score(y, oof)
      
        results[name] = {'oof': oof, 'test': test, 'f1': f1, 'auc': auc}
        print(f"{name:12s}: F1={f1:.4f}, AUC={auc:.4f}")
  
    # Best meta-learner
    best = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\n[OK] Best: {best[0]}")
  
    return best[1]['oof'], best[1]['test']
