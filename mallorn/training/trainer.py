"""
Core training loops and orchestration.
"""
import os
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from mallorn.config import cfg
from mallorn.models.trees import get_lgbm_params, get_xgb_params, get_catboost_params
from mallorn.models.neural_nets import TDEClassifierNet, NNDataset, mixup_data, mixup_criterion, SimpleTFT
from mallorn.models.vision import SwinV2LightCurveTransformer, light_curve_to_image
from mallorn.models.calibration import train_calibrator, calibrate_predictions, apply_temperature_scaling, learn_temperature
from mallorn.data.augmentation import (
    apply_smote_resampling, apply_smote_enn, apply_smote_tomek, 
    apply_borderline_smote, apply_svm_smote, apply_adasyn_resampling,
    apply_enn, apply_tomek_links
)
from mallorn.training.thresholding import find_optimal_threshold
from mallorn.features.selection import select_features_ensemble

# ============================================================
# NEURAL NETWORK TRAINING LOOP
# ============================================================

def train_nn(X_train, y_train, X_val, y_val, epochs=400, patience=60, batch_size=128, best_alpha=None):
    """
    Optimized training with:
    - BCEWithLogitsLoss (Stable)
    - OneCycleLR scheduler (Fast convergence)
    - Mixup augmentation
    - SWA for last 20 epochs
    """
    
    print("  Training Neural Network (BCE + OneCycleLR)...")
    model, best_f1 = _train_single_nn(
        X_train, y_train, X_val, y_val,
        epochs=epochs, patience=patience, batch_size=batch_size,
        alpha=None, verbose=True
    )
    
    return model, best_f1, None

def _train_single_nn(X_train, y_train, X_val, y_val, epochs, patience, batch_size, alpha, verbose=True):
    """Helper for single NN training run."""
    device = cfg.DEVICE
    use_mixup = True
    mixup_alpha = 0.2
    swa_start_epoch = max(0, epochs - 20)

    # Balanced sampling via oversampling
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
  
    # Oversample minority class
    n_pos_samples = len(pos_indices) * 4
    pos_samples = np.random.choice(pos_indices, n_pos_samples, replace=True)
  
    # Balanced dataset
    combined_indices = np.concatenate([pos_samples, neg_indices])
    np.random.shuffle(combined_indices)
  
    X_train_balanced = X_train[combined_indices]
    y_train_balanced = y_train[combined_indices]
  
    train_dl = DataLoader(
        NNDataset(X_train_balanced, y_train_balanced),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
  
    val_dl = DataLoader(
        NNDataset(X_val, y_val),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0
    )
  
    model = TDEClassifierNet(X_train.shape[1]).to(device)
  
    # Stable Loss: BCEWithLogitsLoss
    pos_weight = torch.tensor([1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3, # Initial LR for OneCycle
        weight_decay=1e-4
    )
  
    # OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3, 
        epochs=epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0 
    )
  
    best_score = -np.inf
    best_state = None
    wait = 0
    
    # For SWA
    swa_states = []
  
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
      
        for x, y_batch in train_dl:
            x, y_batch = x.to(device), y_batch.to(device)
            
            # Apply Mixup augmentation
            if use_mixup and np.random.random() > 0.5:
                x, y_a, y_b, lam = mixup_data(x, y_batch, mixup_alpha)
                optimizer.zero_grad()
                outputs = model(x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y_batch)
          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
          
            train_loss += loss.item()
            n_batches += 1
      
        check_step = 3 if verbose else 10
        if epoch % check_step == 0 or epoch > epochs - 30:
            model.eval()
            val_probs = []
            with torch.no_grad():
                for batch in val_dl:
                    x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    probs = torch.sigmoid(model(x))
                    val_probs.append(probs.cpu().numpy())
          
            val_probs = np.concatenate(val_probs)
            _, current_f1 = find_optimal_threshold(y_val, val_probs)
          
            if verbose and (epoch % 10 == 0 or current_f1 > best_score):
                print(f"Epoch {epoch+1}: Val F1={current_f1:.4f} | Loss={train_loss/n_batches:.4f} | LR={scheduler.get_last_lr()[0]:.2e}")
          
            # Track best state
            if current_f1 > best_score:
                best_score = current_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            
            # Collect states for SWA
            if epoch >= swa_start_epoch:
                swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
          
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # SWA: Average states
    if len(swa_states) >= 3 and best_state is not None:
        if verbose:
            print(f"Applying SWA averaging of {len(swa_states)} states from last 20 epochs...")
        avg_state = {}
        for key in best_state.keys():
            avg_state[key] = torch.stack([s[key].float() for s in swa_states]).mean(dim=0)
        model.load_state_dict(avg_state)
        
        # Verify SWA didn't hurt performance
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_dl:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                probs = torch.sigmoid(model(x))
                val_probs.append(probs.cpu().numpy())
        val_probs = np.concatenate(val_probs)
        _, swa_f1 = find_optimal_threshold(y_val, val_probs)
        
        if swa_f1 < best_score - 0.02:  # Revert
            if verbose:
                print(f"SWA F1={swa_f1:.4f} < Best F1={best_score:.4f}, reverting to best...")
            model.load_state_dict(best_state)
        else:
            if verbose:
                print(f"SWA F1={swa_f1:.4f} (Best single: {best_score:.4f})")
            best_score = max(best_score, swa_f1)
    elif best_state:
        model.load_state_dict(best_state)
  
    return model, best_score

def predict_nn(model, X, batch_size=256):
    """Efficient batch prediction."""
    device = cfg.DEVICE
    model.eval()
    model.to(device)
  
    ds = NNDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
  
    probs = []
    with torch.no_grad():
        for batch in dl:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            p = torch.sigmoid(model(x))
            probs.append(p.cpu().numpy())
  
    return np.concatenate(probs)

# ============================================================
# TRAIN ENSEMBLE ORCHESTRATOR
# ============================================================

def train_ensemble(X, y, X_test, n_folds=5, n_repeats=3, feature_names=None, groups=None, splits_dict=None, pretrained_weights=None):
    """
    Comprehensive ensemble training with repeated CV.
    """
    oof_preds = collections.defaultdict(lambda: np.zeros(len(y)))
    test_preds = collections.defaultdict(lambda: np.zeros(len(X_test)))
    model_scores = collections.defaultdict(list)
    model_thresholds = collections.defaultdict(list)
    
    # Initialize keys
    model_keys = ['lgbm', 'xgb', 'catboost', 'rf', 'et', 'gbm', 'nn', 'transformer', 'swinv2', 'tft']
    for key in model_keys:
        _ = oof_preds[key]
        _ = test_preds[key]
    
    # Feature selection
    if cfg.AUTO_FEATURE_SELECTION and feature_names is not None:
        print("\n🧠 Selecting features (Ensemble)...")
        _, selected_names_tree, _ = select_features_ensemble(
            X, y, feature_names, n_features=cfg.N_FEATURES_TREE
        )
        selected_indices_tree = [feature_names.index(name) for name in selected_names_tree]
        
        _, selected_names_nn, _ = select_features_ensemble(
            X, y, feature_names, n_features=cfg.N_FEATURES_NN
        )
        selected_indices_nn = [feature_names.index(name) for name in selected_names_nn]
    else:
        selected_indices_tree = list(range(X.shape[1]))
        selected_indices_nn = list(range(X.shape[1]))

    # Cross-Validation Strategy
    splits = []
    if groups is not None:
        print(f" [SHIELD]  Using Object-Level Stratified Group CV (Leakage Protection)")
        for r in range(n_repeats):
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED + r)
            splits.extend(list(sgkf.split(X, y, groups=groups)))
    else:
        kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=cfg.SEED)
        splits = list(kf.split(X, y))
    
    fold_scores = []
    total_folds = len(splits)
    
    for i, (train_idx, val_idx) in enumerate(splits):
        fold = i % n_folds
        repeat = i // n_folds
        print(f"\n[CYCLE] Fold {fold+1}/{n_folds} (Repeat {repeat+1}/{n_repeats})")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Leakage Check
        if groups is not None:
             train_groups = set(groups.iloc[train_idx] if hasattr(groups, 'iloc') else groups[train_idx])
             val_groups = set(groups.iloc[val_idx] if hasattr(groups, 'iloc') else groups[val_idx])
             intersection = train_groups.intersection(val_groups)
             assert len(intersection) == 0, f"CRITICAL: OBJECT LEAKAGE DETECTED!"
        
        train_pos, val_pos = y_train.sum(), y_val.sum()
        print(f"   Train Pos: {train_pos}/{len(y_train)} | Val Pos: {val_pos}/{len(y_val)}")
        
        fold_val_preds = []
        
        # --- LightGBM ---
        if cfg.USE_LGBM:
            print("  Training LightGBM...", end="")
            X_train_fold = X_train[:, selected_indices_tree]
            if cfg.USE_SMOTE:
                X_train_res, y_train_res = apply_smote_resampling(
                    X_train_fold, y_train,
                    k_neighbors=cfg.SMOTE_K_NEIGHBORS,
                    sampling_strategy=cfg.SMOTE_SAMPLING_STRATEGY
                )
            else:
                X_train_res, y_train_res = X_train_fold, y_train
            
            model = lgb.train(
                get_lgbm_params(),
                lgb.Dataset(X_train_res, label=y_train_res),
                num_boost_round=cfg.N_ESTIMATORS,
                valid_sets=[lgb.Dataset(X_val[:, selected_indices_tree], label=y_val)],
                callbacks=[lgb.early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False)]
            )
            val_pred = model.predict(X_val[:, selected_indices_tree])
            test_pred = model.predict(X_test[:, selected_indices_tree])
                         
            calibrator = train_calibrator(val_pred, y_val)
            val_pred_cal = calibrate_predictions(val_pred, calibrator)
            test_pred_cal = calibrate_predictions(test_pred, calibrator)
            
            oof_preds['lgbm'][val_idx] += val_pred_cal / n_repeats
            test_preds['lgbm'] += test_pred_cal / total_folds
            
            best_thresh, score = find_optimal_threshold(y_val, val_pred_cal)
            print(f" F1={score:.4f}")
            model_scores['lgbm'].append(score)
            model_thresholds['lgbm'].append(best_thresh)
            fold_val_preds.append(val_pred_cal)
            
        # --- XGBoost ---
        if getattr(cfg, 'USE_XGB', True):
            print("  Training XGBoost (Optimized)...", end="")
            
            X_train_fold = X_train[:, selected_indices_tree]
            X_val_fold = X_val[:, selected_indices_tree]
            
            if getattr(cfg, 'USE_SMOTE_ENN', False):
                X_train_res, y_train_res = apply_smote_enn(X_train_fold, y_train, cfg.SMOTE_K_NEIGHBORS)
            elif getattr(cfg, 'USE_SMOTE', True):
                X_train_res, y_train_res = apply_smote_resampling(X_train_fold, y_train, cfg.SMOTE_K_NEIGHBORS)
            else:
                X_train_res, y_train_res = X_train_fold, y_train
            
            dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
            dval = xgb.DMatrix(X_val_fold, label=y_val)
            
            model = xgb.train(
                get_xgb_params(),
                dtrain,
                num_boost_round=cfg.N_ESTIMATORS,
                evals=[(dval, 'eval')],
                early_stopping_rounds=cfg.EARLY_STOPPING_ROUNDS,
                verbose_eval=False
            )
            
            val_pred = model.predict(dval)
            test_pred = model.predict(xgb.DMatrix(X_test[:, selected_indices_tree]))
            
            calibrator = train_calibrator(val_pred, y_val)
            val_pred_cal = calibrate_predictions(val_pred, calibrator)
            test_pred_cal = calibrate_predictions(test_pred, calibrator)
            
            oof_preds['xgb'][val_idx] += val_pred_cal / n_repeats
            test_preds['xgb'] += test_pred_cal / total_folds
            
            best_thresh, score = find_optimal_threshold(y_val, val_pred_cal)
            print(f" F1={score:.4f}")
            model_scores['xgb'].append(score)
            model_thresholds['xgb'].append(best_thresh)
            fold_val_preds.append(val_pred_cal)

        # --- Random Forest ---
        if getattr(cfg, 'USE_RF', False):
            print("  Training Random Forest (Tuned)...", end="")
            
            X_train_fold = X_train[:, selected_indices_tree]
            if getattr(cfg, 'USE_SMOTE', False):
                X_train_res, y_train_res = apply_smote_resampling(X_train_fold, y_train, cfg.SMOTE_K_NEIGHBORS)
            else:
                X_train_res, y_train_res = X_train_fold, y_train
                
            model = RandomForestClassifier(
                n_estimators=500, max_depth=15, min_samples_split=10, 
                min_samples_leaf=4, class_weight='balanced_subsample',
                n_jobs=-1, random_state=cfg.SEED
            )
            model.fit(X_train_res, y_train_res)
            val_pred = model.predict_proba(X_val[:, selected_indices_tree])[:, 1]
            test_pred = model.predict_proba(X_test[:, selected_indices_tree])[:, 1]
            
            calibrator = train_calibrator(val_pred, y_val)
            val_pred_cal = calibrate_predictions(val_pred, calibrator)
            test_pred_cal = calibrate_predictions(test_pred, calibrator)
            
            oof_preds['rf'][val_idx] += val_pred_cal / n_repeats
            test_preds['rf'] += test_pred_cal / total_folds
            
            best_thresh, score = find_optimal_threshold(y_val, val_pred_cal)
            print(f" F1={score:.4f}")
            model_scores['rf'].append(score)
            model_thresholds['rf'].append(best_thresh)
            fold_val_preds.append(val_pred_cal)

        # --- Neural Network ---
        if cfg.USE_NN:
            # Note X_train/X_val mapping based on selected_indices_nn
            X_nn_train = X_train[:, selected_indices_nn]
            X_nn_val = X_val[:, selected_indices_nn]
            X_nn_test = X_test[:, selected_indices_nn]
            
            model, best_f1, _ = train_nn(
                X_nn_train, y_train, X_nn_val, y_val,
                epochs=cfg.N_EPOCHS, patience=cfg.NN_PATIENCE,
                batch_size=cfg.BATCH_SIZE
            )
            
            val_pred = predict_nn(model, X_nn_val)
            test_pred = predict_nn(model, X_nn_test)
            
            # Calibration usually needed for NN probabilities
            calibrator = train_calibrator(val_pred, y_val, method='platt')
            val_pred_cal = calibrate_predictions(val_pred, calibrator)
            test_pred_cal = calibrate_predictions(test_pred, calibrator)
            
            oof_preds['nn'][val_idx] += val_pred_cal / n_repeats
            test_preds['nn'] += test_pred_cal / total_folds
            
            best_thresh, score = find_optimal_threshold(y_val, val_pred_cal)
            print(f"  NN F1={score:.4f}")
            model_scores['nn'].append(score)
            model_thresholds['nn'].append(best_thresh)
            fold_val_preds.append(val_pred_cal)
            
        # Fold ensemble diagnostic
        if fold_val_preds:
            ensemble_val = np.mean(fold_val_preds, axis=0)
            thresh, f1 = find_optimal_threshold(y_val, ensemble_val)
            
            pred_bin = (ensemble_val >= thresh).astype(int)
            pos_count = np.sum(pred_bin)
            prec = np.sum((pred_bin == 1) & (y_val == 1)) / (np.sum(pred_bin == 1) + 1e-10)
            rec = np.sum((pred_bin == 1) & (y_val == 1)) / (np.sum(y_val == 1) + 1e-10)
            
            print(f"[STATS] Fold Ensemble F1: {f1:.4f} | Thresh={thresh:.3f} | P={prec:.3f} | R={rec:.3f}")
            fold_scores.append(f1)

    print("\n[WRENCH] Median Deployment Thresholds:")
    for model, thresholds in model_thresholds.items():
        if len(thresholds) > 0:
            median_thresh = np.median(thresholds)
            print(f"   {model.upper():>12}: {median_thresh:.4f}")
            
    return oof_preds, test_preds, model_scores
