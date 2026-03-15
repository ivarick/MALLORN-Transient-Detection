"""
Data augmentation strategies and resampling techniques for imbalance handling.
"""
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight

from mallorn.config import cfg

# ============================================================
# DATA AUGMENTATION STRATEGIES (Section 4.3)
# ============================================================

def apply_photometric_noise(t, f, e, noise_scale=None, heteroscedastic=None):
    """
    Add Gaussian noise to photometric data to simulate varying observational conditions.
    """
    if noise_scale is None:
        noise_scale = cfg.PHOTOMETRIC_NOISE_SCALE if hasattr(cfg, 'PHOTOMETRIC_NOISE_SCALE') else 0.05
    if heteroscedastic is None:
        heteroscedastic = cfg.HETEROSCEDASTIC_NOISE if hasattr(cfg, 'HETEROSCEDASTIC_NOISE') else True

    if heteroscedastic and len(e) > 0:
        # Scale noise by reported photometric errors
        scaled_noise = noise_scale * (1 + e / (np.median(e) + 1e-10))
        noise = np.random.normal(0, scaled_noise, size=len(f))
    else:
        # Homoscedastic noise
        noise = np.random.normal(0, noise_scale, size=len(f))
    
    f_augmented = f + noise
    return f_augmented


def apply_temporal_stretch(t, stretch_factor=None):
    """
    Apply temporal stretching to simulate different redshift/time dilation effects.
    """
    if stretch_factor is None:
        stretch_range = cfg.TEMPORAL_STRETCH_RANGE if hasattr(cfg, 'TEMPORAL_STRETCH_RANGE') else (0.5, 2.0)
        stretch_factor = np.random.uniform(stretch_range[0], stretch_range[1])
    
    t_stretched = t * stretch_factor
    return t_stretched


def apply_magnitude_shift(f, shift=None):
    """
    Apply magnitude/flux shift to simulate distance variations.
    """
    if shift is None:
        shift_range = cfg.MAGNITUDE_SHIFT_RANGE if hasattr(cfg, 'MAGNITUDE_SHIFT_RANGE') else (-0.5, 0.5)
        shift = np.random.uniform(shift_range[0], shift_range[1])
    
    # Convert to magnitude, shift, convert back to flux
    # m = -2.5 * log10(f) + constant, so f ~ 10^(-0.4*m)
    f_shifted = f * 10**(-0.4 * shift)
    return f_shifted


def augment_light_curve(t, f, e, augmentation_prob=0.5):
    """
    Apply random augmentation to a light curve.
    
    Combines photometric noise, temporal stretch, and magnitude shift.
    """
    if np.random.random() > augmentation_prob:
        return t, f, e
    
    use_photometric = cfg.USE_PHOTOMETRIC_AUGMENTATION if hasattr(cfg, 'USE_PHOTOMETRIC_AUGMENTATION') else True
    use_temporal = cfg.USE_TEMPORAL_AUGMENTATION if hasattr(cfg, 'USE_TEMPORAL_AUGMENTATION') else False

    # Photometric noise
    if use_photometric:
        f = apply_photometric_noise(t, f, e)
    
    # Temporal stretch
    if use_temporal:
        t = apply_temporal_stretch(t)
    
    # Magnitude shift
    if use_temporal:
        f = apply_magnitude_shift(f)
    
    return t, f, e

# ============================================================
# ADVANCED SMOTE & IMBALANCE HANDLING (Section 4)
# ============================================================

def get_class_weights(y_train, method='balanced'):
    """
    Calculate inverse frequency class weights for cost-sensitive learning.
    """
    if method == 'balanced':
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
    elif method == 'sqrt_balanced':
        # Square root weighting for less aggressive penalization
        n_samples = len(y_train)
        n_classes = len(np.unique(y_train))
        
        weights = []
        for c in np.unique(y_train):
            n_c = np.sum(y_train == c)
            weight = np.sqrt(n_samples / (n_classes * n_c))
            weights.append(weight)
        weights = np.array(weights)
    else:
        # Default balanced
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
    
    weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"  Class weights: {weight_dict}")
    return weight_dict

def apply_smote_resampling(X_train, y_train, k_neighbors=5, sampling_strategy='auto'):
    """
    Apply SMOTE for minority class oversampling.
    """
    try:
        # Conservative SMOTE with smaller k for extreme imbalance
        effective_k = min(k_neighbors, int(sum(y_train) - 1))
        effective_k = max(effective_k, 1)
        
        if effective_k < 2:
            print(f"  [WARN] Not enough minority samples for SMOTE (k={effective_k}), skipping")
            return X_train, y_train
            
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k,
            random_state=cfg.SEED
        )
        
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        print(f"  [OK] SMOTE: {len(X_train)} -> {len(X_res)} samples")
        print(f"     Positives: {sum(y_train)} -> {sum(y_res)}")
        
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] SMOTE failed: {e}, returning original data")
        return X_train, y_train

def apply_borderline_smote(X_train, y_train, k_neighbors=5, sampling_strategy='auto'):
    """
    Borderline-SMOTE: Focus on samples near decision boundary.
    """
    try:
        effective_k = min(k_neighbors, int(sum(y_train) - 1))
        effective_k = max(effective_k, 2)
        
        smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k,
            random_state=cfg.SEED,
            m_neighbors=10  # Neighbors for identifying borderline
        )
        
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"  [OK] Borderline-SMOTE: {len(X_train)} -> {len(X_res)} samples")
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] Borderline-SMOTE failed: {e}, trying standard SMOTE")
        return apply_smote_resampling(X_train, y_train, k_neighbors, sampling_strategy)


def apply_svm_smote(X_train, y_train, k_neighbors=5):
    """
    SVM-SMOTE: Uses SVM to identify support vectors for synthetic generation.
    """
    try:
        effective_k = min(k_neighbors, int(sum(y_train) - 1))
        effective_k = max(effective_k, 2)
        
        smote = SVMSMOTE(
            k_neighbors=effective_k,
            random_state=cfg.SEED
        )
        
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"  [OK] SVM-SMOTE: {len(X_train)} -> {len(X_res)} samples")
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] SVM-SMOTE failed: {e}, falling back to standard SMOTE")
        return apply_smote_resampling(X_train, y_train, k_neighbors)


def apply_tomek_links(X_train, y_train):
    """
    Tomek Links removal for boundary cleaning.
    """
    try:
        tl = TomekLinks(sampling_strategy='majority')
        X_res, y_res = tl.fit_resample(X_train, y_train)
        
        n_removed = len(X_train) - len(X_res)
        print(f"  [OK] Tomek Links: Removed {n_removed} majority samples")
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] Tomek Links failed: {e}")
        return X_train, y_train


def apply_enn(X_train, y_train, n_neighbors=3):
    """
    Edited Nearest Neighbors (ENN) for cleaning.
    """
    try:
        enn = EditedNearestNeighbours(
            n_neighbors=n_neighbors,
            sampling_strategy='majority'
        )
        X_res, y_res = enn.fit_resample(X_train, y_train)
        
        n_removed = len(X_train) - len(X_res)
        print(f"  [OK] ENN: Removed {n_removed} noisy majority samples")
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] ENN failed: {e}")
        return X_train, y_train


def apply_smote_enn(X_train, y_train, k_neighbors=5, sampling_strategy='auto'):
    """
    Combined SMOTE + ENN approach with controlled sampling.
    """
    try:
        n_pos = int(sum(y_train))
        n_neg = len(y_train) - n_pos
        effective_k = min(k_neighbors, n_pos - 1)
        effective_k = max(effective_k, 2)
        
        # Validate float ratio
        if isinstance(sampling_strategy, (int, float)) and sampling_strategy != 'auto':
            current_ratio = n_pos / n_neg
            target_ratio = float(sampling_strategy)
            if target_ratio <= current_ratio:
                target_ratio = max(current_ratio * 3, 0.1)
                target_ratio = min(target_ratio, 0.5)
                sampling_strategy = target_ratio
        
        smote_enn = SMOTEENN(
            smote=SMOTE(k_neighbors=effective_k, random_state=cfg.SEED, sampling_strategy=sampling_strategy),
            random_state=cfg.SEED
        )
        
        X_res, y_res = smote_enn.fit_resample(X_train, y_train)
        print(f"  [OK] SMOTE+ENN: {len(X_train)} -> {len(X_res)} samples")
        print(f"     Positives: {sum(y_train)} -> {sum(y_res)}")
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] SMOTE+ENN failed: {e}, falling back to SMOTE")
        return apply_smote_resampling(X_train, y_train, k_neighbors)


def apply_smote_tomek(X_train, y_train, k_neighbors=5, sampling_strategy='auto'):
    """
    Combined SMOTE + Tomek Links approach with controlled sampling.
    """
    try:
        n_pos = int(sum(y_train))
        n_neg = len(y_train) - n_pos
        effective_k = min(k_neighbors, n_pos - 1)
        effective_k = max(effective_k, 2)
        
        # Validate float ratio
        if isinstance(sampling_strategy, (int, float)) and sampling_strategy != 'auto':
            current_ratio = n_pos / n_neg
            target_ratio = float(sampling_strategy)
            if target_ratio <= current_ratio:
                target_ratio = max(current_ratio * 3, 0.1)
                target_ratio = min(target_ratio, 0.5)
                sampling_strategy = target_ratio
        
        smote_tomek = SMOTETomek(
            smote=SMOTE(k_neighbors=effective_k, random_state=cfg.SEED, sampling_strategy=sampling_strategy),
            random_state=cfg.SEED
        )
        
        X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
        print(f"  [OK] SMOTE+Tomek: {len(X_train)} -> {len(X_res)} samples")
        print(f"     Positives: {sum(y_train)} -> {sum(y_res)}")
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] SMOTE+Tomek failed: {e}, falling back to SMOTE")
        return apply_smote_resampling(X_train, y_train, k_neighbors)


def apply_adasyn_resampling(X_train, y_train, n_neighbors=5):
    """
    Apply ADASYN for adaptive synthetic sampling.
    
    ADASYN generates more synthetic samples for "harder" minority examples
    near the decision boundary.
    """
    try:
        effective_n = min(n_neighbors, int(sum(y_train) - 1))
        effective_n = max(effective_n, 1)
        
        adasyn = ADASYN(
            n_neighbors=effective_n,
            random_state=cfg.SEED
        )
        
        X_res, y_res = adasyn.fit_resample(X_train, y_train)
        
        print(f"  [OK] ADASYN: {len(X_train)} -> {len(X_res)} samples")
        return X_res, y_res
        
    except Exception as e:
        print(f"  [WARN] ADASYN failed: {e}, returning original data")
        return X_train, y_train
