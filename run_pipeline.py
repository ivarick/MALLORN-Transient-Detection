"""
Main execution pipeline for MALLORN.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from mallorn.config import cfg
from mallorn.data.dataset import load_all_splits
from mallorn.features.extraction import build_features
from mallorn.training.trainer import train_ensemble
from mallorn.utils import report_oof_diagnostics
from mallorn.training.ensembling import train_stacking_with_calibration, ensemble_recall_first, ensemble_rank_based, optimize_ensemble_weights
from mallorn.data.pseudo_labeling import pseudo_labeling, noisy_student_training, multi_view_cotraining
from mallorn.training.thresholding import optimize_threshold_global_oof

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("MALLORN TDE Classifier")
    print(f"Device : {cfg.DEVICE}")
    print("=" * 60)

    # --- Configuration Summary ---
    print("\n[CONFIG] Pipeline configuration:")
    print(f"  Models          : LGBM={cfg.USE_LGBM}, XGB={getattr(cfg, 'USE_XGB', True)}, CatBoost={getattr(cfg, 'USE_CATBOOST', False)}, "
          f"RF={getattr(cfg, 'USE_RF', False)}, HGB={getattr(cfg, 'USE_HGB', False)}, NN={cfg.USE_NN}")
    print(f"  TFT             : {getattr(cfg, 'USE_TEMPORAL_FUSION', False)} (hidden={getattr(cfg, 'TFT_HIDDEN_SIZE', 160)}, heads={getattr(cfg, 'TFT_ATTENTION_HEADS', 4)})")
    print(f"  SMOTE           : enabled={cfg.USE_SMOTE}, ratio={getattr(cfg, 'SMOTE_RATIO', 'auto')}, k={cfg.SMOTE_K_NEIGHBORS}")
    print(f"  SMOTE+ENN       : {getattr(cfg, 'USE_SMOTE_ENN', False)}  |  Tomek Links: {getattr(cfg, 'USE_TOMEK_LINKS', False)}  |  ADASYN: {getattr(cfg, 'USE_ADASYN', False)}")
    print(f"  Augmentation    : photometric={getattr(cfg, 'USE_PHOTOMETRIC_AUGMENTATION', False)}, temporal={getattr(cfg, 'USE_TEMPORAL_AUGMENTATION', False)}")
    print(f"  Feature libs    : wavelets={cfg.USE_WAVELETS}, GP={getattr(cfg, 'USE_GP_FEATURES', False)}, dmdt={getattr(cfg, 'USE_DMDT_FEATURES', False)}")
    print(f"  Focal loss      : gamma={getattr(cfg, 'FOCAL_LOSS_GAMMA', 2.0)}, alpha={getattr(cfg, 'FOCAL_LOSS_ALPHA', 0.25)}")
    print(f"  CV              : {cfg.N_FOLDS} folds x {getattr(cfg, 'N_REPEATS', 3)} repeats (stratified)")
    print(f"  Pseudo-labels   : advanced={getattr(cfg, 'USE_ADVANCED_PSEUDO_LABELS', False)}, noisy-student iters={getattr(cfg, 'NOISY_STUDENT_ITERATIONS', 3)}")
    print(f"  Co-training     : {getattr(cfg, 'USE_MULTI_VIEW_COTRAINING', False)}")
    print("=" * 60)

    # 1. Load Data
    print("\n[DATA] Loading metadata...")
    train_log = pd.read_csv(os.path.join(cfg.BASE_DIR, "train_log.csv"))
    test_log = pd.read_csv(os.path.join(cfg.BASE_DIR, "test_log.csv"))
    print(f"  Train objects : {len(train_log)}")
    print(f"  Test objects  : {len(test_log)}")

    # 2. Load lightcurves
    print("\n[DATA] Loading split light curves...")
    train_splits = load_all_splits("train")
    test_splits = load_all_splits("test")

    # 3. Feature extraction
    print("\n[FEAT] Extracting features...")
    train_features = build_features(train_log, train_splits)
    test_features = build_features(test_log, test_splits)

    # 3b. Prune sparse features (>95% NaN across training set)
    feature_cols = [col for col in train_features.columns if col not in ['object_id', 'target', 'error']]
    nan_counts = train_features[feature_cols].isna().mean()
    to_drop = nan_counts[nan_counts > 0.95].index.tolist()
    if to_drop:
        print(f"  Pruning {len(to_drop)} sparse features (>95% NaN)")
        train_features.drop(columns=to_drop, inplace=True)
        test_features.drop(columns=to_drop, inplace=True, errors='ignore')

    # Re-define feature columns after dropping
    feature_cols = [col for col in train_features.columns if col not in ['object_id', 'target', 'error']]

    # 4. Fill NaN / Inf
    train_features[feature_cols] = train_features[feature_cols].fillna(0)
    test_features[feature_cols] = test_features[feature_cols].fillna(0)
    train_features[feature_cols] = np.nan_to_num(train_features[feature_cols].values, nan=0, posinf=1e9, neginf=-1e9)
    test_features[feature_cols] = np.nan_to_num(test_features[feature_cols].values, nan=0, posinf=1e9, neginf=-1e9)

    # 5. Normalise features with RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_features[feature_cols])
    X_test = scaler.transform(test_features[feature_cols])
    
    y_train = train_log['target'].values
  
    # 6. Optional: supervised contrastive pre-training
    pretrained_weights = None
    if getattr(cfg, 'USE_TRANSFORMER', False) and getattr(cfg, 'USE_CONTRASTIVE_PRETRAIN', False):
        print("\n[PRETRAIN] Loading or computing SupCon encoder weights...")
        weights_path = os.path.join(cfg.BASE_DIR, "simclr_encoder.pth")
        if os.path.exists(weights_path):
            print(f"  Loading weights from {weights_path}")
            pretrained_weights = torch.load(weights_path)
        else:
            print("  No saved weights found \u2014 running supervised contrastive pre-training.")
            try:
                from cnn_model import pretrain_supcon
                pretrained_weights = pretrain_supcon(
                    train_log, train_splits,
                    X_tab=X_train,
                    n_epochs=getattr(cfg, 'PRETRAIN_EPOCHS', 50),
                    batch_size=256,
                    device=cfg.DEVICE,
                    use_atat=(getattr(cfg, 'USE_TRANSFORMER', False))
                )
                torch.save(pretrained_weights, weights_path)
                print(f"  Weights saved to {weights_path}")
            except ImportError:
                print("  [WARN] cnn_model.py not found. Contrastive pre-training skipped.")

    # 7. Ensemble training
    print("\n[TRAIN] Training base models...")
    processed_splits = {}
    if getattr(cfg, 'USE_TRANSFORMER', False):
        # Time-series extraction
        pass

    oof_preds, test_preds, model_scores = train_ensemble(
        X_train, y_train, X_test,
        n_folds=cfg.N_FOLDS,
        n_repeats=getattr(cfg, 'N_REPEATS', 3),
        feature_names=feature_cols,
        groups=train_log['object_id'],
        splits_dict=processed_splits,
        pretrained_weights=pretrained_weights
    )

    print("\n[DIAG] Computing OOF diagnostics...")
    oof_diagnostics = report_oof_diagnostics(oof_preds, y_train, model_scores)

    # 8. Stacking & calibration
    print("\n[STACK] Stacking and calibrating meta-learners...")
    oof_stack, test_stack = train_stacking_with_calibration(
        oof_preds, y_train, test_preds, groups=train_log['object_id']
    )

    # 9. Pseudo-labeling
    if getattr(cfg, 'USE_ADVANCED_PSEUDO_LABELS', False):
        print("\n[PSEUDO] Noisy Student pseudo-labeling...")
        final_test_probs = noisy_student_training(
            X_train, y_train, X_test, test_preds,
            n_iterations=getattr(cfg, 'NOISY_STUDENT_ITERATIONS', 3),
            threshold_curriculum=getattr(cfg, 'THRESHOLD_CURRICULUM', [0.95, 0.90, 0.85])
        )
    elif getattr(cfg, 'USE_PSEUDO_LABELS', False):
        print("\n[PSEUDO] Basic pseudo-labeling...")
        final_test_probs = pseudo_labeling(
            X_train, y_train, X_test, test_preds, getattr(cfg, 'PSEUDO_ITERATIONS', 3)
        )
    else:
        final_test_probs = test_stack
    
    # 9b. Multi-view co-training
    if getattr(cfg, 'USE_MULTI_VIEW_COTRAINING', False):
        print("\n[COTRAIN] Multi-view co-training...")
        cotraining_preds = multi_view_cotraining(
            X_train, y_train, X_test, feature_cols,
            n_iterations=getattr(cfg, 'COTRAINING_ITERATIONS', 3)
        )
        if np.any(cotraining_preds != 0):
            final_test_probs = 0.7 * final_test_probs + 0.3 * cotraining_preds
            print(f"   Blended Noisy Student + Co-Training (70/30)")
        else:
            print(f"   Co-Training skipped, using Noisy Student only")
  
    # 10. Optimise ensemble weights and apply recall-first gating
    print("\n[OPT] Optimising ensemble weights...")
    ensemble_weights = optimize_ensemble_weights(oof_preds, y_train)
    
    active_models = [k for k in oof_preds if np.any(oof_preds[k] != 0)]
    if ensemble_weights:
        oof_weighted = sum(oof_preds[k] * ensemble_weights.get(k, 0) for k in active_models)
        test_weighted = sum(test_preds[k] * ensemble_weights.get(k, 0) for k in active_models)
    else:
        oof_weighted = np.mean([oof_preds[k] for k in active_models], axis=0) if active_models else np.zeros_like(y_train)
        test_weighted = np.mean([test_preds[k] for k in active_models], axis=0) if active_models else np.zeros_like(final_test_probs)
    
    print("\n[OPT] Applying recall-first ensemble gating...")
    oof_recall, test_recall = ensemble_recall_first(oof_preds, test_preds, y_train)
    
    oof_weighted = np.maximum(oof_weighted, oof_recall)
    if len(test_weighted) > 0 and len(test_recall) > 0:
        test_weighted = np.maximum(test_weighted, test_recall)
    
    if not getattr(cfg, 'USE_PSEUDO_LABELS', False) and len(test_weighted) > 0:
        final_test_probs = test_weighted
    
    print("\n[OPT] Computing rank-based ensemble...")
    kept_models = list(ensemble_weights.keys()) if ensemble_weights else None
    oof_rank, test_rank = ensemble_rank_based(oof_preds, test_preds, active_models=kept_models)
    
    # 11. Find optimal threshold on global OOF
    print("\n[THRESH] Optimising classification threshold on global OOF...")
    val_probs = oof_stack.flatten() if oof_stack.ndim > 1 else oof_stack
    optimal_thresh_global, f1_global = optimize_threshold_global_oof(y_train, val_probs)
    optimal_thresh_rank, f1_rank = optimize_threshold_global_oof(y_train, oof_rank)
    print(f"  Global OOF F1 : {f1_global:.4f} (threshold={optimal_thresh_global:.4f})")
    print(f"  Rank OOF F1   : {f1_rank:.4f}  (threshold={optimal_thresh_rank:.4f})")

    # 12. Generate final predictions
    print("\n[OUT] Writing predictions...")
    predictions = pd.DataFrame({
        'object_id': test_log['object_id'],
        'target': (test_rank >= optimal_thresh_rank).astype(int) if len(test_rank) > 0 else np.zeros(len(test_log), dtype=int)
    })
    predictions.to_csv("predictions.csv", index=False)
    n_pos = predictions['target'].sum()
    tde_rate = n_pos / len(predictions) if len(predictions) > 0 else 0
    print(f"  File           : predictions.csv")
    print(f"  Threshold      : {optimal_thresh_rank:.4f} (rank-ensemble, OOF-optimised)")
    print(f"  Predicted TDEs : {n_pos} / {len(predictions)} ({tde_rate*100:.2f}%)")

    # Summary
    print(f"\n{'='*60}")
    print("MALLORN TDE Classifier \u2014 Final Results")
    print(f"{'='*60}")

    if 'oof_diagnostics' in locals() and oof_diagnostics:
        print("\n[RESULTS] OOF performance by model:")
        for model_name, metrics in oof_diagnostics.items():
            print(f"  {model_name.upper():>14}: F1={metrics.get('f1', 0):.4f}  P={metrics.get('precision', 0):.4f}  "
                  f"R={metrics.get('recall', 0):.4f}  |P-R|={metrics.get('pr_gap', 0):.4f}")

    if 'ensemble_weights' in locals() and ensemble_weights:
        print("\n[RESULTS] Optimised ensemble weights:")
        for model_name, weight in sorted(ensemble_weights.items(), key=lambda x: -x[1]):
            print(f"  {model_name.upper():>14}: {weight:.4f}")

    print("\n[RESULTS] Classification thresholds:")
    print(f"  Global OOF stack : {optimal_thresh_global:.4f}  (F1={f1_global:.4f})")
    print(f"  Rank ensemble    : {optimal_thresh_rank:.4f}  (F1={f1_rank:.4f})")

    n_tde_train = int(y_train.sum())
    imbalance = int((len(y_train) - n_tde_train) / max(n_tde_train, 1))
    print("\n[DATA]   Training set overview:")
    print(f"  Total objects   : {len(y_train)}")
    print(f"  TDEs (positive) : {n_tde_train} ({n_tde_train/len(y_train)*100:.2f}%)")
    print(f"  Imbalance ratio : {imbalance}:1")

    print(f"\n[OUT]  Predictions written to: predictions.csv  "
          f"({n_pos} positive predictions, {tde_rate*100:.2f}% TDE rate)")
    print(f"{'='*60}")
    print("Pipeline complete.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        cfg.DEVICE = "cuda"
        torch.cuda.manual_seed_all(cfg.SEED)
    main()
