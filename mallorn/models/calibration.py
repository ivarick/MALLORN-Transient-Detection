"""
Probability calibration techniques.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from mallorn.config import cfg

def apply_temperature_scaling(logits, temperature=1.5):
    """
    Temperature scaling for neural network calibration.
    
    Higher temperature (>1) makes predictions less confident.
    Lower temperature (<1) makes predictions more confident.
    T=1.5 is typically good for reducing overconfidence.
    """
    return logits / temperature

def learn_temperature(val_logits, val_labels, init_temp=1.5):
    """
    Learn optimal temperature on validation set.
    Minimizes negative log likelihood.
    """
    from scipy.optimize import minimize_scalar
    
    def nll_loss(T):
        scaled = val_logits / T
        probs = 1 / (1 + np.exp(-scaled))  # sigmoid
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        loss = -np.mean(val_labels * np.log(probs) + (1 - val_labels) * np.log(1 - probs))
        return loss
    
    result = minimize_scalar(nll_loss, bounds=(0.1, 5.0), method='bounded')
    return result.x

def train_calibrator(probs, y_true, method='auto'):
    """
    Train calibration model with enhanced strategies for imbalanced data.
    
    Methods:
    - 'isotonic': Non-parametric, powerful but needs data (200+ samples)
    - 'platt': Logistic regression, better for small/imbalanced data
    - 'auto': Select based on data size and imbalance ratio
    
    Phase 2 Step 4: Calibrate Base Models.
    """
    n_samples = len(y_true)
    n_pos = np.sum(y_true)
    imbalance_ratio = n_samples / max(n_pos, 1)
    
    # Auto-select method based on data characteristics
    if method == 'auto':
        if n_samples >= 500 and imbalance_ratio < 50:
            method = 'isotonic'
        elif n_samples >= 100:
            method = 'platt'
        else:
            method = 'none'  # Too few samples, skip calibration
    
    if method == 'isotonic' and n_samples >= 200:
        # Isotonic with increased robustness for imbalanced data
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        iso.fit(probs, y_true)
        return iso
    elif method in ['platt', 'logistic']:
        # Platt Scaling with moderate regularization for small data
        probs_reshaped = probs.reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')
        lr.fit(probs_reshaped, y_true)
        return lr
    else:
        # No calibration for very small datasets
        return None

def calibrate_predictions(probs, calibrator):
    """Apply fitted calibrator to predictions."""
    if calibrator is None:
        return probs
        
    # Check type by method presence or class
    if hasattr(calibrator, 'predict_proba'):
        # LogisticRegression (Platt)
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    else:
        # IsotonicRegression
        return calibrator.predict(probs)
