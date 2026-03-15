"""
Advanced pseudo-labeling techniques (Noisy Student, FixMatch, Co-training).
"""
import os
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import lightgbm as lgb
import xgboost as xgb

from mallorn.config import cfg
from mallorn.models.vision import SwinV2LightCurveTransformer
from mallorn.models.calibration import calibrate_predictions
from mallorn.training.thresholding import find_optimal_threshold
from mallorn.models.trees import get_lgbm_params, get_xgb_params

def apply_advanced_pseudo_labeling(X_train_img, X_unlabeled_img, y_train, swin_model_class, criterion, train_fn, device=None):
    if device is None:
        device = cfg.DEVICE
    print("\n[PHASE] 🟡 ADVANCED PSEUDO-LABELING (Noisy Student)")
    print("  [WARN] Advanced Pseudo-Labeling is currently a stub in this modularized version.")
    return X_train_img, y_train

def get_pseudo_labels(model, X_unlabeled, threshold_high=0.95, threshold_low=0.0):
    probs = model.predict_proba(X_unlabeled)[:, 1]
    pos_mask = probs >= threshold_high
    neg_mask = probs <= threshold_low
    pseudo_indices = np.where(pos_mask | neg_mask)[0]
    pseudo_labels = np.where(pos_mask[pseudo_indices], 1, 0)
    return pseudo_indices, pseudo_labels

# ============================================================
# PSEUDO-LABELING
# ============================================================
def pseudo_labeling(X, y, X_test, test_preds, n_iterations=3):
    """Conservative iterative pseudo-labeling."""
    print(f"\n{'='*70}")
    print("PSEUDO-LABELING")
    print(f"{'='*70}")
  
    ensemble_test = np.mean([test_preds[k] for k in test_preds if np.any(test_preds[k] != 0)], axis=0)
  
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
      
        high_pos = ensemble_test >= getattr(cfg, 'PSEUDO_THRESHOLD_HIGH', 0.95)
        
        # POSITIVE-ONLY STRATEGY: Disable negative pseudo-labeling if threshold is <= 0
        if getattr(cfg, 'PSEUDO_THRESHOLD_LOW', 0.0) <= 0.0:
            high_neg = np.zeros_like(ensemble_test, dtype=bool)
            print("   (Positive-only pseudo-labeling enabled)")
        else:
            high_neg = ensemble_test <= cfg.PSEUDO_THRESHOLD_LOW
      
        n_pos, n_neg = np.sum(high_pos), np.sum(high_neg)
        print(f" Confident: {n_pos} pos, {n_neg} neg")
        
        if n_pos < 5:
            print(" Not enough confident predictions. Stopping.")
            break
      
        if n_pos < 3 or n_neg < 30:
            break
      
        # Limit pseudo-labels
        max_pos = min(n_pos, int(0.3 * y.sum()))
        max_neg = min(n_neg, int(0.05 * len(y)))
      
        pos_idx = np.random.choice(np.where(high_pos)[0], min(len(np.where(high_pos)[0]), max_pos), replace=False)
        neg_idx = np.random.choice(np.where(high_neg)[0], min(len(np.where(high_neg)[0]), max_neg), replace=False)
      
        pseudo_X = np.vstack([X_test[pos_idx], X_test[neg_idx]])
        pseudo_y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
      
        X_combined = np.vstack([X, pseudo_X])
        y_combined = np.concatenate([y, pseudo_y])
      
        print(f" Training on {len(X_combined)} samples")
      
        # Retrain
        lgbm_params = get_lgbm_params()
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=getattr(cfg, 'N_REPEATS', 3), random_state=cfg.SEED + iteration)
        new_test = np.zeros(len(X_test))
      
        for train_idx, val_idx in skf.split(X_combined, y_combined):
            train_data = lgb.Dataset(X_combined[train_idx], label=y_combined[train_idx])
            val_data = lgb.Dataset(X_combined[val_idx], label=y_combined[val_idx])
          
            model = lgb.train(
                lgbm_params, train_data, num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
          
            new_test += model.predict(X_test) / (5 * getattr(cfg, 'N_REPEATS', 3))
      
        ensemble_test = 0.85 * ensemble_test + 0.15 * new_test
  
    return ensemble_test

def get_feature_views(feature_names):
    """
    Separate features into 3 independent views for co-training.
    """
    temporal_patterns = ['rise', 'decay', 'duration', 'cadence', 'time_', 'auc', 
                         'asymmetry', 'peak', 'slope', 'curvature', 'n_obs',
                         'bazin', 'pl_decay', 'power_law']
    color_patterns = ['color', '_gr', '_ri', '_gi', '_rz', '_iz', 'rest_frame', 
                      'multiband', 'band_ratio', 'coherence', 'temp_', 'luminosity']
    
    views = {'temporal': [], 'color': [], 'statistical': []}
    view_indices = {'temporal': [], 'color': [], 'statistical': []}
    
    for idx, feat in enumerate(feature_names):
        feat_lower = feat.lower()
        if any(pat in feat_lower for pat in temporal_patterns):
            views['temporal'].append(feat)
            view_indices['temporal'].append(idx)
        elif any(pat in feat_lower for pat in color_patterns):
            views['color'].append(feat)
            view_indices['color'].append(idx)
        else:
            views['statistical'].append(feat)
            view_indices['statistical'].append(idx)
    
    print(f"   Feature Views: Temporal={len(views['temporal'])}, "
          f"Color={len(views['color'])}, Statistical={len(views['statistical'])}")
    
    return views, view_indices

def select_pseudo_labels_calibrated(test_probs_dict, calibrators=None):
    active_models = [k for k in test_probs_dict if np.any(test_probs_dict[k] != 0)]
    n_models = len(active_models)
    
    all_preds = np.array([test_probs_dict[k] for k in active_models])
    
    if calibrators and getattr(cfg, 'PSEUDO_USE_CALIBRATION', False):
        calibrated_preds = []
        for i, model_name in enumerate(active_models):
            if model_name in calibrators:
                cal_probs = calibrate_predictions(all_preds[i], calibrators[model_name])
                calibrated_preds.append(cal_probs)
            else:
                calibrated_preds.append(all_preds[i])
        all_preds = np.array(calibrated_preds)
    
    ensemble_probs = np.mean(all_preds, axis=0)
    pos_votes = np.sum(all_preds >= 0.5, axis=0)
    neg_votes = n_models - pos_votes
    agreement_scores = np.maximum(pos_votes, neg_votes) / n_models
    
    high_conf_pos = (ensemble_probs >= getattr(cfg, 'PSEUDO_THRESHOLD_TDE', 0.95))
    high_conf_neg = (ensemble_probs <= (1 - getattr(cfg, 'PSEUDO_THRESHOLD_NON_TDE', 0.80)))
    
    if getattr(cfg, 'PSEUDO_USE_MODEL_AGREEMENT', False):
        min_agree = getattr(cfg, 'PSEUDO_AGREEMENT_MIN', 2)
        pos_agree = pos_votes >= min_agree
        neg_agree = neg_votes >= min_agree
        high_conf_pos = high_conf_pos & pos_agree
        high_conf_neg = high_conf_neg & neg_agree
    
    pseudo_idx_pos = np.where(high_conf_pos)[0]
    pseudo_idx_neg = np.where(high_conf_neg)[0]
    
    return pseudo_idx_pos, pseudo_idx_neg, agreement_scores

def apply_noisy_augmentation(X, noise_scale=0.1, dropout_rate=0.15):
    X_noisy = X.copy()
    if noise_scale > 0:
        stds = np.std(X_noisy, axis=0, keepdims=True) + 1e-8
        noise = np.random.randn(*X_noisy.shape) * stds * noise_scale
        X_noisy = X_noisy + noise
    if dropout_rate > 0:
        mask = np.random.rand(*X_noisy.shape) > dropout_rate
        X_noisy = X_noisy * mask
    return X_noisy

def noisy_student_training(X_train, y_train, X_test, test_preds, 
                            n_iterations=3, threshold_curriculum=None,
                            calibrators=None):
    print(f"\n{'='*70}")
    print("NOISY STUDENT TRAINING")
    print(f"{'='*70}")
    
    if threshold_curriculum is None:
        threshold_curriculum = getattr(cfg, 'THRESHOLD_CURRICULUM', [0.95, 0.90, 0.85])
    
    active_models = [k for k in test_preds if np.any(test_preds[k] != 0)]
    ensemble_test = np.mean([test_preds[k] for k in active_models], axis=0)
    
    all_pseudo_X = []
    all_pseudo_y = []
    
    for iteration in range(n_iterations):
        if iteration < len(threshold_curriculum):
            pos_thresh = threshold_curriculum[iteration]
        else:
            pos_thresh = threshold_curriculum[-1]
        
        neg_thresh = 1 - getattr(cfg, 'PSEUDO_THRESHOLD_NON_TDE', 0.80)
        
        print(f"\n[BOOKS] Iteration {iteration + 1}/{n_iterations} "
              f"(Pos thresh: {pos_thresh:.2f}, Neg thresh: {neg_thresh:.2f})")
        
        high_pos = ensemble_test >= pos_thresh
        high_neg = ensemble_test <= neg_thresh
        
        if getattr(cfg, 'PSEUDO_USE_MODEL_AGREEMENT', False) and len(active_models) >= 3:
            all_preds = np.array([test_preds[k] for k in active_models])
            pos_votes = np.sum(all_preds >= 0.5, axis=0)
            neg_votes = len(active_models) - pos_votes
            
            high_pos = high_pos & (pos_votes >= getattr(cfg, 'PSEUDO_AGREEMENT_MIN', 2))
            high_neg = high_neg & (neg_votes >= getattr(cfg, 'PSEUDO_AGREEMENT_MIN', 2))
        
        n_pos, n_neg = np.sum(high_pos), np.sum(high_neg)
        print(f"   Confident samples: {n_pos} pos, {n_neg} neg")
        
        if n_pos < 3:
            print("   Not enough positive pseudo-labels. Stopping.")
            break
        
        n_original_pos = int(y_train.sum())
        max_pos = min(n_pos, int(0.5 * n_original_pos))
        max_neg = min(n_neg, int(0.1 * len(y_train)))
        
        pos_idx = np.where(high_pos)[0]
        neg_idx = np.where(high_neg)[0]
        
        if len(pos_idx) > max_pos:
            conf_order = np.argsort(-ensemble_test[pos_idx])
            pos_idx = pos_idx[conf_order[:max_pos]]
        
        if len(neg_idx) > max_neg:
            conf_order = np.argsort(ensemble_test[neg_idx])
            neg_idx = neg_idx[conf_order[:max_neg]]
        
        pseudo_X = np.vstack([X_test[pos_idx], X_test[neg_idx]])
        pseudo_y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
        
        all_pseudo_X.append(pseudo_X)
        all_pseudo_y.append(pseudo_y)
        
        combined_pseudo_X = np.vstack(all_pseudo_X)
        combined_pseudo_y = np.concatenate(all_pseudo_y)
        
        X_train_noisy = apply_noisy_augmentation(
            X_train, 
            noise_scale=getattr(cfg, 'NOISY_STUDENT_NOISE_SCALE', 0.1),
            dropout_rate=0.1
        )
        combined_pseudo_X_noisy = apply_noisy_augmentation(
            combined_pseudo_X,
            noise_scale=getattr(cfg, 'NOISY_STUDENT_NOISE_SCALE', 0.1) * 0.5,
            dropout_rate=0.05
        )
        
        X_combined = np.vstack([X_train_noisy, combined_pseudo_X_noisy])
        y_combined = np.concatenate([y_train, combined_pseudo_y])
        
        print(f"   Training STUDENT on {len(X_combined)} samples "
              f"(Original: {len(X_train)}, Pseudo: {len(combined_pseudo_X)})")
        
        xgb_params = get_xgb_params()
        xgb_params['n_estimators'] = 500
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED + iteration)
        new_test = np.zeros(len(X_test))
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
            dtrain = xgb.DMatrix(X_combined[train_idx], label=y_combined[train_idx])
            dval = xgb.DMatrix(X_combined[val_idx], label=y_combined[val_idx])
            dtest = xgb.DMatrix(X_test)
            
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            new_test += model.predict(dtest) / 5
        
        blend_weight = 0.3 + 0.1 * iteration
        ensemble_test = (1 - blend_weight) * ensemble_test + blend_weight * new_test
        
        new_pos = np.sum(ensemble_test >= 0.5)
        print(f"   New ensemble: {new_pos} predicted positives")
    
    print(f"\n[OK] Noisy Student Complete: Added {len(combined_pseudo_X)} pseudo-labels total")
    return ensemble_test

def multi_view_cotraining(X_train, y_train, X_test, feature_names, n_iterations=3):
    print(f"\n{'='*70}")
    print("MULTI-VIEW CO-TRAINING")
    print(f"{'='*70}")
    
    views, view_indices = get_feature_views(feature_names)
    
    min_view_size = min(len(v) for v in view_indices.values())
    if min_view_size < 20:
        print(f"   [WARN] View too small ({min_view_size} features). Skipping co-training.")
        return np.zeros(len(X_test))
    
    X_train_views = {view: X_train[:, indices] for view, indices in view_indices.items()}
    X_test_views = {view: X_test[:, indices] for view, indices in view_indices.items()}
    
    view_preds_test = {}
    xgb_params = get_xgb_params()
    xgb_params['n_estimators'] = 300
    
    print("\n[STATS] Training initial view-specific models...")
    for view_name, X_v_train in X_train_views.items():
        print(f"   Training {view_name} view ({X_v_train.shape[1]} features)...")
        
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
        for train_idx, val_idx in skf.split(X_v_train, y_train):
            dtrain = xgb.DMatrix(X_v_train[train_idx], label=y_train[train_idx])
            dval = xgb.DMatrix(X_v_train[val_idx], label=y_train[val_idx])
            dtest = xgb.DMatrix(X_test_views[view_name])
            
            model = xgb.train(
                xgb_params, dtrain, num_boost_round=300,
                evals=[(dval, 'val')], early_stopping_rounds=30,
                verbose_eval=False
            )
            
            oof_preds[val_idx] = model.predict(dval)
            test_preds += model.predict(dtest) / 5
        
        view_preds_test[view_name] = test_preds
        thresh, f1 = find_optimal_threshold(y_train, oof_preds)
        print(f"      {view_name}: OOF F1 = {f1:.4f}")
    
    print(f"\n[CYCLE] Starting co-training iterations...")
    
    for iteration in range(n_iterations):
        print(f"\n   Iteration {iteration + 1}/{n_iterations}")
        pseudo_added = 0
        
        for view_name in views.keys():
            other_views = [v for v in views.keys() if v != view_name]
            
            this_conf = view_preds_test[view_name]
            other_mean = np.mean([view_preds_test[v] for v in other_views], axis=0)
            
            confident_pos = (this_conf >= getattr(cfg, 'COTRAINING_CONFIDENCE', 0.9)) & (other_mean < 0.5)
            confident_neg = (this_conf <= (1 - getattr(cfg, 'COTRAINING_CONFIDENCE', 0.9))) & (other_mean >= 0.5)
            
            n_teach_pos = np.sum(confident_pos)
            n_teach_neg = np.sum(confident_neg)
            
            if n_teach_pos + n_teach_neg < 5:
                continue
            
            max_teach = min(50, int(0.1 * np.sum(y_train)))
            pos_idx = np.where(confident_pos)[0][:max_teach]
            neg_idx = np.where(confident_neg)[0][:max_teach]
            
            if len(pos_idx) > 0 or len(neg_idx) > 0:
                for other_view in other_views:
                    X_v_train = X_train_views[other_view]
                    X_v_test = X_test_views[other_view]
                    
                    if len(pos_idx) > 0 and len(neg_idx) > 0:
                        pseudo_X = np.vstack([X_v_test[pos_idx], X_v_test[neg_idx]])
                        pseudo_y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
                    elif len(pos_idx) > 0:
                        pseudo_X = X_v_test[pos_idx]
                        pseudo_y = np.ones(len(pos_idx))
                    else:
                        pseudo_X = X_v_test[neg_idx]
                        pseudo_y = np.zeros(len(neg_idx))
                    
                    X_expanded = np.vstack([X_v_train, pseudo_X])
                    y_expanded = np.concatenate([y_train, pseudo_y])
                    
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SEED + iteration)
                    new_test_preds = np.zeros(len(X_test))
                    
                    for train_idx, val_idx in skf.split(X_expanded, y_expanded):
                        dtrain = xgb.DMatrix(X_expanded[train_idx], label=y_expanded[train_idx])
                        dtest = xgb.DMatrix(X_v_test)
                        
                        model = xgb.train(
                            xgb_params, dtrain, num_boost_round=200,
                            verbose_eval=False
                        )
                        new_test_preds += model.predict(dtest) / 3
                    
                    view_preds_test[other_view] = 0.7 * view_preds_test[other_view] + 0.3 * new_test_preds
                    pseudo_added += len(pos_idx) + len(neg_idx)
        
        print(f"      Added {pseudo_added} pseudo-labels across views")
        if pseudo_added < 10:
            print("      Convergence reached. Stopping.")
            break
    
    final_preds = np.mean([view_preds_test[v] for v in views.keys()], axis=0)
    print(f"\n[OK] Co-Training Complete")
    for view_name in views.keys():
        pred_pos = np.sum(view_preds_test[view_name] >= 0.5)
        print(f"   {view_name}: {pred_pos} predicted positives")
    
    return final_preds
