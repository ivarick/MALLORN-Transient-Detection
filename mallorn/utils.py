"""
General pipeline utilities: safe math, formatting, and prediction output validation.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mallorn.config import cfg
from mallorn.training.thresholding import find_optimal_threshold

def safe_divide_scalar(a, b, default=0.0):
    return a / b if b != 0 else default

def safe_divide(a, b, default=0.0):
    """Vectorized safe division."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, default)
        return np.nan_to_num(result, nan=default, posinf=default, neginf=default)

def fast_percentiles(x, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """Fast percentile computation."""
    if len(x) == 0:
        return [0.0] * len(percentiles)
    return np.percentile(x, percentiles).tolist()

def format_split_name(split_val):
    """Normalize split name format."""
    if isinstance(split_val, str):
        match = re.search(r'(\d+)', split_val)
        if match:
            return f"Split_{int(match.group(1)):02d}"
    return f"Split_{int(split_val):02d}"

# ============================================================
# PREDICTION OUTPUT VALIDATION
# ============================================================

def validate_predictions(predictions_df, test_ids, expected_count=7135):
    """
    Validate the integrity of binary classification predictions.

    Checks:
    - Required columns present ('object_id', 'target')
    - Exact test set length
    - All object_ids accounted for
    - Strictly integer 0/1 labels (no floats or NaN)
    - Predicted TDE rate within expected prevalence bounds
    """
    errors = []
    
    if 'object_id' not in predictions_df.columns:
        errors.append("Missing 'object_id' column")
    if 'target' not in predictions_df.columns:
        errors.append("Missing 'target' column")
    
    if errors:
        return False, errors
    
    # Row count
    if len(predictions_df) != expected_count:
        errors.append(f"Wrong number of rows: {len(predictions_df)}, expected {expected_count}")
    
    # Missing object IDs
    missing_ids = set(test_ids) - set(predictions_df['object_id'])
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} object_ids")
    
    # Label dtype
    if not pd.api.types.is_integer_dtype(predictions_df['target']):
        errors.append("'target' column must contain integers (0 or 1), not floats")
    
    unique_values = set(predictions_df['target'].unique())
    if not unique_values.issubset({0, 1}):
        invalid = unique_values - {0, 1}
        errors.append(f"Invalid label values: {invalid}. Must be 0 or 1.")
    
    # NaN check
    if predictions_df['target'].isna().any():
        errors.append("'target' column contains NaN values")
    
    # TDE prevalence check
    if cfg.VALIDATE_PREDICTIONS:
        n_tdes = predictions_df['target'].sum()
        tde_rate = n_tdes / len(predictions_df)
        expected_rate = cfg.PRIOR_TDE_RATE
        tolerance = cfg.TDE_RATE_TOLERANCE
        
        lower = expected_rate * (1 - tolerance)
        upper = expected_rate * (1 + tolerance)
        
        if not (lower <= tde_rate <= upper):
            errors.append(
                f"Predicted TDE rate {tde_rate:.4f} ({n_tdes} objects) outside expected range "
                f"[{lower:.4f}, {upper:.4f}] ({expected_rate * 100:.2f}% ± {tolerance*100:.1f}%)"
            )
    
    is_valid = len(errors) == 0
    return is_valid, errors

def compute_confusion_matrix_analysis(y_true, y_pred, class_names=['Non-TDE', 'TDE']):
    """
    Analyze confusion matrix to identify misclassification patterns.
    """
    from sklearn.metrics import confusion_matrix
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate rates
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'matrix': cm,
            'tpr': tpr,
            'fpr': fpr,
            'false_positives': fp,
            'false_negatives': fn
        }
    except Exception as e:
        print(f"Error in confusion matrix analysis: {e}")
        return None

def report_oof_diagnostics(oof_preds, y_true, model_scores):
    """
    Comprehensive OOF diagnostics: F1, Precision, Recall, P-R Gap for each model.
    
    Fulfills Priority 0 (Report OOF F1/P/R) and Priority 5 (Report Δ = |P - R|).
    """
    print(f"\n{'='*70}")
    print("[STATS] COMPREHENSIVE OOF DIAGNOSTICS")
    print(f"{'='*70}")
    
    results = {}
    
    for model_name, preds in oof_preds.items():
        if not np.any(preds != 0):
            continue
            
        # Find optimal threshold for this model
        thresh, f1 = find_optimal_threshold(y_true, preds)
        
        # Compute predictions at threshold
        pred_binary = (preds >= thresh).astype(int)
        
        # Calculate precision and recall
        tp = np.sum((pred_binary == 1) & (y_true == 1))
        fp = np.sum((pred_binary == 1) & (y_true == 0))
        fn = np.sum((pred_binary == 0) & (y_true == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        pr_gap = abs(precision - recall)
        
        results[model_name] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'pr_gap': pr_gap,
            'threshold': thresh
        }
        
        # Print model diagnostics
        gap_warning = " [WARN]" if pr_gap > 0.05 else ""
        print(f"\n{model_name.upper():>12}: F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | Δ={pr_gap:.4f}{gap_warning}")
        
        # Print CV mean if available
        if model_name in model_scores and len(model_scores[model_name]) > 0:
            cv_mean = np.mean(model_scores[model_name])
            cv_std = np.std(model_scores[model_name])
            print(f"             CV Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Ensemble diagnostics
    active_models = [k for k in oof_preds if np.any(oof_preds[k] != 0)]
    if len(active_models) > 1:
        ensemble_oof = np.mean([oof_preds[k] for k in active_models], axis=0)
        
        thresh, f1 = find_optimal_threshold(y_true, ensemble_oof)
        pred_binary = (ensemble_oof >= thresh).astype(int)
        
        tp = np.sum((pred_binary == 1) & (y_true == 1))
        fp = np.sum((pred_binary == 1) & (y_true == 0))
        fn = np.sum((pred_binary == 0) & (y_true == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        pr_gap = abs(precision - recall)
        
        results['ensemble_avg'] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'pr_gap': pr_gap,
            'threshold': thresh
        }
        
        print(f"\n{'ENSEMBLE':>12}: F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | Δ={pr_gap:.4f}")
    
    # Summary
    print(f"\n{'-'*70}")
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"[OK] Best Model: {best_model[0].upper()} (F1={best_model[1]['f1']:.4f})")
    
    # Check P-R equilibrium status
    min_gap_model = min(results.items(), key=lambda x: x[1]['pr_gap'])
    print(f"[SCALE]️  Most Balanced (P/R): {min_gap_model[0].upper()} (Δ={min_gap_model[1]['pr_gap']:.4f})")
    
    # Alert if gap > 0.05 on ensemble
    if 'ensemble_avg' in results and results['ensemble_avg']['pr_gap'] > 0.05:
        print(f"\n[WARN]  WARNING: Ensemble P-R gap ({results['ensemble_avg']['pr_gap']:.4f}) > 0.05!")
        print(f"   Consider adjusting Focal Loss α to balance Precision/Recall.")
    
    return results
