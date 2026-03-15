"""
Physics calculations for extincting and fitting astrophysical formulas.
"""
import numpy as np
import extinction
from scipy.optimize import curve_fit

from mallorn.config import cfg
from mallorn.utils import safe_divide_scalar

def apply_deextinction(flux, ebv, band):
    """
    Correct flux for galactic dust extinction using Fitzpatrick 99 law.
    """
    if ebv <= 0:
        return flux
        
    # Get wavelength for band
    wave = np.array([cfg.FILTER_WAVELENGTHS[band]])
    
    # Calculate extinction in magnitudes (A_v = 3.1 * EBV)
    a_lambda = extinction.fitzpatrick99(wave, float(ebv) * 3.1)[0]
    
    # Convert magnitude extinction to flux correction factor
    ext_factor = 10 ** (a_lambda / 2.5)
    return flux * ext_factor

def power_law_model(t, amplitude, exponent, offset):
    """Simple power-law function."""
    return amplitude * (t ** exponent) + offset

def fit_power_law(t, f, t_peak):
    """
    Fit t^(-5/3) power law to decay phase.
    """
    feats = {
        'pl_tde_distance': 10.0, 'pl_vs_exp_ratio': 10.0,
        'pl_decay_alpha': 0.0, 'pl_fit_success': 0,
        'pl_amplitude': 0.0, 'pl_offset': 0.0
    }
  
    try:
        mask = (t > t_peak) & (f > 0)
        t_decay = t[mask] - t_peak + 0.1
        f_decay = f[mask]
      
        if len(t_decay) < 5:
            return feats
          
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
          
        try:
            popt_exp, _ = curve_fit(exp_decay, t_decay, f_decay,
                                    p0=[np.max(f_decay), 0.05, 0],
                                    bounds=([0, 0, -np.inf], [np.inf, 10, np.inf]),
                                    maxfev=500)
            res_exp = np.sum((f_decay - exp_decay(t_decay, *popt_exp)) ** 2)
        except:
            res_exp = np.inf
            
        try:
            popt_pl, _ = curve_fit(power_law_model, t_decay, f_decay,
                                   p0=[np.max(f_decay), -1.5, 0],
                                   bounds=([0, -5.0, -np.inf], [np.inf, -0.1, np.inf]),
                                   maxfev=500)
            res_pl = np.sum((f_decay - power_law_model(t_decay, *popt_pl)) ** 2)
          
            feats['pl_decay_alpha'] = popt_pl[1]
            feats['pl_amplitude'] = popt_pl[0]
            feats['pl_offset'] = popt_pl[2]
            feats['pl_fit_success'] = 1
            feats['pl_tde_distance'] = abs(popt_pl[1] - cfg.TDE_DECAY_POWER)
          
            if res_exp > 0 and res_pl > 0:
                feats['pl_vs_exp_ratio'] = res_pl / res_exp
              
        except:
            pass
          
    except:
        pass
  
    return feats

def power_law_func(x, a, t0, alpha, c):
    """Standard power-law decay component calculation."""
    return np.where(x > t0, a * ((x - t0 + 1)**alpha) + c, c)

def fit_rise_decay_power_law(t, f, t_peak, e=None):
    """
    Attempts to fit a more robust combined rise/decay profile where the
    decay is explicitly constrained toward -5/3. Extracting the difference
    between actual fall and expected TDE fall.
    """
    feats = {
        'rd_pl_sigma': 100.0,
        'rd_pl_tde_residual': 100.0,
        'rd_pl_amp': 0.0,
        'rd_pl_alpha': 0.0,
        'rd_pl_success': 0
    }
    
    if len(t) < 5:
        return feats
        
    mask = (t > t_peak) & (f > 0)
    t_decay = t[mask]
    f_decay = f[mask]
    
    if len(t_decay) < 4:
        return feats
        
    try:
        sigma = e[mask] if e is not None else None
        
        # P0: amplitude=max, t0=t_peak, alpha=-1.6, C=0
        popt, pcov = curve_fit(
            power_law_func, t_decay, f_decay,
            p0=[np.max(f_decay), t_peak, -1.6, np.min(f_decay)],
            bounds=([0, t_peak-10, -4.0, -np.inf], [np.inf, t_peak+10, -0.1, np.inf]),
            sigma=sigma, absolute_sigma=True,
            maxfev=1000
        )
        
        preds = power_law_func(t_decay, *popt)
        mse = np.mean((f_decay - preds)**2)
        
        # Now fit forcing alpha = -5/3 specifically to check residual
        def forced_tde_func(x, a, t0, c):
            return np.where(x > t0, a * ((x - t0 + 1)**(-5/3)) + c, c)
            
        popt_forced, _ = curve_fit(
            forced_tde_func, t_decay, f_decay,
            p0=[np.max(f_decay), t_peak, np.min(f_decay)],
            bounds=([0, t_peak-10, -np.inf], [np.inf, t_peak+10, np.inf]),
            sigma=sigma, maxfev=1000
        )
        tde_preds = forced_tde_func(t_decay, *popt_forced)
        tde_mse = np.mean((f_decay - tde_preds)**2)
        
        feats['rd_pl_sigma'] = np.sqrt(mse) / (np.mean(f_decay) + 1e-5)
        feats['rd_pl_tde_residual'] = np.sqrt(tde_mse) / (np.mean(f_decay) + 1e-5)
        feats['rd_pl_amp'] = popt[0]
        feats['rd_pl_alpha'] = popt[2]
        feats['rd_pl_success'] = 1
        
    except:
        pass
        
    return feats

def calculate_absolute_magnitudes(band_data, z):
    """
    Calculate absolute magnitudes and luminosity features.
    
    Physics:
    - Distance Modulus (DM): 5 * log10(dL_pc) - 5
    - K-correction: -2.5 * log10(1+z)
    - M = m - DM - K
    """
    feats = {}
    if z <= 0.001:
        return feats
        
    # Hubble Law approximation (H0=70, OmegaM=0.3)
    c = 3e5
    H0 = 70.0
    dL_Mpc = (c / H0) * z * (1 + z/2)
    dL_pc = dL_Mpc * 1e6
    
    dist_mod = 5 * np.log10(dL_pc) - 5
    k_corr = 2.5 * np.log10(1 + z)
    
    for band, data in band_data.items():
        if len(data['f']) > 0:
            f_max = np.max(data['f'])
            if f_max > 0:
                # Apparent Mag (flux in uJy ZP ~ 23.9)
                m_app = -2.5 * np.log10(f_max) + 23.9
                M_abs = m_app - dist_mod - k_corr
                feats[f'{band}_abs_mag_peak'] = M_abs
                
                # Luminosity (erg/s approx)
                feats[f'{band}_log_luminosity'] = np.log10(f_max) + 2 * np.log10(dL_pc)
            else:
                feats[f'{band}_abs_mag_peak'] = 0.0
                feats[f'{band}_log_luminosity'] = 0.0
                
    return feats

def bazin_function(t, A, t0, tau_rise, tau_fall, B):
    """Bazin function for transient light curves."""
    arg_fall = np.clip((t - t0) / tau_fall, -20, 20)
    arg_rise = np.clip(-(t - t0) / tau_rise, -20, 20)
    return A * np.exp(-arg_fall) / (1 + np.exp(arg_rise)) + B

def fit_bazin(t, f):
    """
    Fit Bazin function and extract discriminative parameters.
    """
    feats = {
        'bazin_tau_rise': 0.0, 'bazin_tau_fall': 0.0, 'bazin_asymmetry': 0.0,
        'bazin_amplitude': 0.0, 'bazin_residual': 1e9, 'bazin_t0': 0.0,
        'bazin_fit_success': 0.0
    }
    
    if len(t) < 10:
        return feats
    
    idx = np.argsort(t)
    t, f = t[idx], f[idx]
    
    try:
        idx_peak = np.argmax(f)
        t0_init = t[idx_peak]
        A_init = max(np.max(f) - np.min(f), 1.0)
        B_init = np.percentile(f, 10)
        
        popt, _ = curve_fit(
            bazin_function, t, f,
            p0=[A_init, t0_init, 10.0, 50.0, B_init],
            bounds=([0, t[0]-10, 0.5, 1, -np.inf], [np.inf, t[-1]+10, 200, 500, np.inf]),
            maxfev=1000
        )
        
        A, t0, tau_rise, tau_fall, B = popt
        feats['bazin_amplitude'] = A
        feats['bazin_t0'] = t0
        feats['bazin_tau_rise'] = tau_rise
        feats['bazin_tau_fall'] = tau_fall
        feats['bazin_asymmetry'] = safe_divide_scalar(tau_fall, (tau_rise + 1e-10))
        feats['bazin_fit_success'] = 1.0
        
        pred = bazin_function(t, *popt)
        feats['bazin_residual'] = np.std(f - pred) / (np.std(f) + 1e-10)
        
    except:
        pass
    
    return feats
