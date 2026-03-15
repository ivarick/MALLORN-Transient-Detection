"""
AGGRESSIVE feature selection to combat the curse of dimensionality.
"""
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from mallorn.config import cfg

def select_features_ensemble(X, y, feature_names, n_features=1000):
    """
    AGGRESSIVE feature selection to combat curse of dimensionality.
    
    CRITICAL: 1165 features for 64 positives = 18:1 ratio (extreme overfitting).
    Strategy: Variance filter -> Correlation filter -> Model-based selection
    """
    print(f"\n[SEARCH] AGGRESSIVE Feature Selection ({len(feature_names)} -> {n_features})...")
    print(f"   Class distribution: {np.sum(y==1)} positive, {np.sum(y==0)} negative")
    
    # Stage 1: Aggressive variance filtering
    var_threshold = cfg.VARIANCE_THRESHOLD if hasattr(cfg, 'VARIANCE_THRESHOLD') else 0.001
    variances = np.var(X, axis=0)
    var_mask = variances > var_threshold
    
    n_low_var = np.sum(~var_mask)
    if n_low_var > 0:
        print(f"   Stage 1 (Variance > {var_threshold}): Removed {n_low_var} features")
    
    X_filtered = X[:, var_mask]
    feature_names_filtered = [f for f, m in zip(feature_names, var_mask) if m]
    
    # Stage 2: Correlation filtering (remove highly correlated features)
    if hasattr(cfg, 'USE_CORRELATION_FILTER') and cfg.USE_CORRELATION_FILTER:
        corr_threshold = cfg.CORRELATION_THRESHOLD if hasattr(cfg, 'CORRELATION_THRESHOLD') else 0.95
        if X_filtered.shape[1] > 100:  # Only if still many features
            corr_matrix = np.abs(np.corrcoef(X_filtered, rowvar=False))
            upper = np.triu(corr_matrix > corr_threshold, k=1)
            to_drop = set()
            for i in range(upper.shape[0]):
                if i in to_drop:
                    continue
                correlated = np.where(upper[i, :])[0]
                for j in correlated:
                    if j not in to_drop:
                        to_drop.add(j)
            
            if len(to_drop) > 0:
                keep_mask = np.array([i not in to_drop for i in range(X_filtered.shape[1])])
                X_filtered = X_filtered[:, keep_mask]
                feature_names_filtered = [f for f, m in zip(feature_names_filtered, keep_mask) if m]
                print(f"   Stage 2 (Correlation < {corr_threshold}): Removed {len(to_drop)} features")
    
    # Stage 3: Hard cap on features before model selection
    max_pre_features = cfg.MAX_FEATURES_BEFORE_SELECTION if hasattr(cfg, 'MAX_FEATURES_BEFORE_SELECTION') else 300
    if len(feature_names_filtered) > max_pre_features:
        # Quick MI-based pre-selection
        mi_quick = mutual_info_classif(X_filtered, y, random_state=cfg.SEED, n_neighbors=3)
        top_pre = np.argsort(mi_quick)[::-1][:max_pre_features]
        X_filtered = X_filtered[:, top_pre]
        feature_names_filtered = [feature_names_filtered[i] for i in top_pre]
        print(f"   Stage 3 (Hard cap): Kept top {max_pre_features} by MI")
  
    if len(feature_names_filtered) <= n_features:
        return np.where(var_mask)[0], feature_names_filtered, np.ones(len(feature_names_filtered))
  
    # Feature selection ensemble
    
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X_filtered, y, random_state=cfg.SEED, n_neighbors=5)
  
    # 2. LightGBM importance
    from mallorn.models.trees import get_lgbm_params  # Local import to avoid circular dependency if models import features
    lgbm = lgb.LGBMClassifier(**get_lgbm_params(), n_estimators=200)
    lgbm.fit(X_filtered, y)
    lgbm_importance = lgbm.feature_importances_
  
    # 3. Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=cfg.SEED, n_jobs=-1)
    rf.fit(X_filtered, y)
    rf_importance = rf.feature_importances_
  
    # 4. Recursive Feature Elimination
    estimator = LogisticRegression(class_weight='balanced', max_iter=500, random_state=cfg.SEED)
    rfe = RFE(estimator, n_features_to_select=n_features, step=0.1)
    rfe.fit(X_filtered, y)
    rfe_scores = rfe.ranking_  # Lower rank is better
    rfe_norm = 1 / (rfe_scores + 1e-10)  # Invert for importance
  
    # Normalize and combine
    mi_norm = mi_scores / (mi_scores.max() + 1e-10)
    lgbm_norm = lgbm_importance / (lgbm_importance.max() + 1e-10)
    rf_norm = rf_importance / (rf_importance.max() + 1e-10)
    rfe_norm = rfe_norm / (rfe_norm.max() + 1e-10)
  
    # Weighted average (more weight to LGBM and RFE)
    combined_scores = 0.35 * lgbm_norm + 0.25 * mi_norm + 0.2 * rf_norm + 0.2 * rfe_norm
  
    # Select top features
    top_local_indices = np.argsort(combined_scores)[::-1][:n_features]
    
    # Map back to original indices
    # We need to trace back through the filtering stages
    # This is tricky, so we'll just return the names and final scores, and let the caller handle filtering the dataframe
    selected_features = [feature_names_filtered[i] for i in top_local_indices]
  
    # Final Correlation removal on selected features
    X_selected = X_filtered[:, top_local_indices]
    corr_matrix = np.corrcoef(X_selected, rowvar=False)
    upper = np.triu(np.abs(corr_matrix) > 0.95, k=1)  # High correlation threshold
    to_drop = [column for column in range(X_selected.shape[1]) if any(upper[:, column])]
  
    keep_indices = [i for i in range(X_selected.shape[1]) if i not in to_drop]
    
    final_features = [selected_features[i] for i in keep_indices]
    final_scores = combined_scores[top_local_indices][keep_indices]
  
    print(f" Selected {len(final_features)} features after correlation removal")
    print(f" Top 15: {final_features[:15]}")
    
    # We return the NAEMS of the selected features. The caller should filter their DataFrame columns.
    return None, final_features, final_scores
