"""
Threshold optimization strategies for F1 maximization and P-R equilibrium.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import StratifiedKFold

from mallorn.config import cfg

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_optimal_threshold(y_true, y_probs):
    """Find F1-optimal threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
  
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
  
    return best_threshold, f1_scores[best_idx]


def find_robust_threshold(y_true, y_probs, n_bootstrap=200):
    """Bootstrap-based robust threshold."""
    thresholds = []
    n_samples = len(y_true)
  
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        thresh, _ = find_optimal_threshold(y_true[idx], y_probs[idx])
        thresholds.append(thresh)
  
    return np.median(thresholds)


def find_stable_threshold(y_true, y_probs, min_thresh=0.15, max_thresh=0.85, target_percentile=50):
    """STABILIZED threshold selection to prevent wild variations.
    
    Problem: Fold thresholds vary wildly (0.088 to 0.99) indicating model uncertainty.
    Solution: Clip to reasonable range + use percentile instead of max F1.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        min_thresh: Minimum allowed threshold (prevents extreme low values)
        max_thresh: Maximum allowed threshold (prevents extreme high values)
        target_percentile: Percentile to use for threshold (50 = median, more stable)
    
    Returns:
        stable_threshold, f1_at_threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Filter thresholds to reasonable range
    valid_mask = (thresholds >= min_thresh) & (thresholds <= max_thresh)
    if np.sum(valid_mask) == 0:
        # Fallback: use middle of range
        return (min_thresh + max_thresh) / 2, 0.0
    
    valid_thresh = thresholds[valid_mask]
    valid_f1 = f1_scores[valid_mask]
    
    # Use percentile-based selection instead of argmax (more stable)
    if target_percentile == 50:
        # Find threshold closest to median F1 performance
        median_f1 = np.median(valid_f1)
        closest_idx = np.argmin(np.abs(valid_f1 - median_f1))
        stable_thresh = valid_thresh[closest_idx]
    else:
        # Use argmax but within valid range
        best_idx = np.argmax(valid_f1)
        stable_thresh = valid_thresh[best_idx]
    
    # Final clip for safety
    stable_thresh = np.clip(stable_thresh, min_thresh, max_thresh)
    
    # Compute F1 at this threshold
    pred = (y_probs >= stable_thresh).astype(int)
    f1 = f1_score(y_true, pred)
    
    return stable_thresh, f1


def optimize_threshold_global_oof(y_true, y_probs):
    """
    Optimizes threshold on GLOBAL OOF predictions.
    
    Why:
      - Per-fold thresholds fit noise (small N positives)
      - Averaging per-fold thresholds gives invalid decision boundary
      - Global optimization sees full 300+ positives -> stable F1 surface
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    # Validation
    p_at_best = precision[best_idx]
    r_at_best = recall[best_idx]
    
    print(f"\n[SEARCH] Global OOF Optimization:")
    print(f"   Best Threshold: {best_thresh:.4f}")
    print(f"   OOF F1: {best_f1:.4f}")
    print(f"   Precision: {p_at_best:.4f} | Recall: {r_at_best:.4f}")
    
    return best_thresh, best_f1


def find_balanced_pr_threshold(y_true, y_probs, min_f1_pct=0.98):
    """
    Find threshold that minimizes |precision - recall| gap while maintaining F1.
    
    Logic:
    1. Find P-R curve intersection point (Equilibrium).
    2. Constrain: F1 must be within 2% of the absolute maximum F1.
    3. Return the threshold that satisfies this constraint with minimal P-R gap.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        min_f1_pct: Only consider thresholds with F1 >= this % of best F1 (default 0.98)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    # 1. Find absolute max F1
    best_f1 = np.max(f1_scores)
    f1_threshold = best_f1 * min_f1_pct
    
    # 2. Find Equilibrium (Intersection)
    # This is where Precision approx equals Recall
    # Minimizing |P - R|
    
    best_gap = float('inf')
    best_balanced_idx = np.argmax(f1_scores) # Default to max F1
    
    # Iterate over all thresholds
    # Note: precision and recall arrays are len(thresholds) + 1
    # We ignore the last element which is 1.0, 0.0 usually
    
    n_thresh = len(thresholds)
    
    for i in range(n_thresh):
        # Constraint: F1 must be close to max
        if f1_scores[i] >= f1_threshold:
            gap = abs(precision[i] - recall[i])
            if gap < best_gap:
                best_gap = gap
                best_balanced_idx = i
                
    balanced_thresh = thresholds[best_balanced_idx]
    p, r = precision[best_balanced_idx], recall[best_balanced_idx]
    f1 = f1_scores[best_balanced_idx]
    
    print(f"\n[SCALE]️  Balanced P/R Threshold (Constraint: F1 >= {min_f1_pct*100:.1f}% of Max):")
    print(f"   Max F1: {best_f1:.4f}")
    print(f"   Balanced F1: {f1:.4f} (-{(best_f1-f1):.4f})")
    print(f"   Threshold: {balanced_thresh:.4f}")
    print(f"   Precision: {p:.4f}")
    print(f"   Recall:    {r:.4f}")
    print(f"   P-R Gap:   {best_gap:.4f}")
    
    if best_gap > 0.05:
        print(f"   [WARN]  Gap > 0.05! Consider adjusting Focal Loss alpha.")
    
    return balanced_thresh


def find_optimal_threshold_f2(y_true, y_probs, beta=2.0):
    """
    Find threshold that maximizes F-beta score (F2 by default for high recall).
    
    F-beta formula: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    With beta=2, recall is weighted 2x more than precision.
    This prioritizes finding all TDEs (high recall) over precision.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        beta: Beta parameter (default 2.0 for F2 score)
    
    Returns:
        best_threshold, best_f2_score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # F-beta score
    beta_sq = beta ** 2
    f2_scores = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-10)
    
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"\n[TARGET] F{beta} Optimization (Recall-Prioritized):")
    print(f"   Best Threshold: {best_threshold:.4f}")
    print(f"   F{beta} Score: {f2_scores[best_idx]:.4f}")
    print(f"   Precision: {precision[best_idx]:.4f} | Recall: {recall[best_idx]:.4f}")
    
    return best_threshold, f2_scores[best_idx]


def find_high_recall_threshold(y_true, y_probs, min_precision=0.65, target_recall=0.85):
    """
    Find threshold that achieves target recall while maintaining minimum precision.
    
    In astronomical surveys, missing a TDE (false negative) typically costs more
    than a false alarm (false positive).
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        min_precision: Minimum acceptable precision (default 0.65)
        target_recall: Target recall to achieve (default 0.85)
    
    Returns:
        best_threshold, metrics_dict
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Find thresholds that meet minimum precision and target recall
    valid_indices = []
    for i in range(len(thresholds)):
        if precision[i] >= min_precision and recall[i] >= target_recall:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        # No threshold meets both criteria - prioritize precision constraint
        print(f"\n[WARN]  No threshold meets both P>={min_precision} and R>={target_recall}")
        print("   Prioritizing minimum precision constraint...")
        
        valid_indices = [i for i in range(len(thresholds)) if precision[i] >= min_precision]
        
        if len(valid_indices) == 0:
            # Even minimum precision not achievable
            print("   [WARN]  Minimum precision not achievable!")
            best_idx = np.argmax(f1_scores)
        else:
            # Maximize recall among valid options
            best_idx = max(valid_indices, key=lambda i: recall[i])
    else:
        # Among valid options, maximize F1
        best_idx = max(valid_indices, key=lambda i: f1_scores[i])
    
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    metrics = {
        'threshold': best_threshold,
        'precision': precision[best_idx],
        'recall': recall[best_idx],
        'f1': f1_scores[best_idx]
    }
    
    print(f"\n[TARGET] High-Recall Threshold (P>={min_precision}, R>={target_recall}):")
    print(f"   Best Threshold: {best_threshold:.4f}")
    print(f"   F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
    
    return best_threshold, metrics


def find_optimal_threshold_nested(y_true, y_probs, n_inner_folds=3):
    """Nested CV for unbiased threshold estimation.
    
    Prevents threshold overfitting by computing optimal thresholds on held-out
    inner folds, then returning the median. More reliable than full-data threshold.
    """
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=cfg.SEED)
    thresholds = []
    
    for _, val_idx in skf.split(y_probs.reshape(-1, 1), y_true):
        thresh, _ = find_optimal_threshold(y_true[val_idx], y_probs[val_idx])
        thresholds.append(thresh)
    
    return np.median(thresholds)


def create_threshold_optimized_xgb(base_estimator=None, cv=None, scoring='f1'):
    """
    Create TunedThresholdClassifierCV wrapper for XGBoost.
    
    This is the PRIMARY optimization lever for bridging 0.66 -> 0.75 F1.
    Optimizes decision threshold (typically 0.2-0.4) instead of default 0.5.
    
    Expected improvement: +0.05-0.10 F1
    """
    import xgboost as xgb
    
    if base_estimator is None:
        # XGBoost with scale_pos_weight=158 for 159:1 imbalance
        base_estimator = xgb.XGBClassifier(
            scale_pos_weight=getattr(cfg, 'XGB_SCALE_POS_WEIGHT', 158),
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            random_state=cfg.SEED,
            n_estimators=500,
            n_jobs=-1
        )
    
    if cv is None:
        cv = StratifiedKFold(n_splits=getattr(cfg, 'THRESHOLD_CV_FOLDS', 3), 
                            shuffle=True, random_state=cfg.SEED)
    
    # Create TunedThresholdClassifierCV
    # Note: This requires scikit-learn >= 1.5
    try:
        from sklearn.model_selection import TunedThresholdClassifierCV
        
        threshold_optimizer = TunedThresholdClassifierCV(
            base_estimator,
            scoring=scoring,
            cv=cv,
            store_cv_results=True,
            refit=True,
            n_jobs=-1,
            response_method='predict_proba'
        )
        
        return threshold_optimizer
        
    except ImportError:
        print("  [WARN] TunedThresholdClassifierCV not available (sklearn < 1.5)")
        print("  Falling back to base estimator with manual threshold optimization")
        return base_estimator
