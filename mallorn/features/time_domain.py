"""
Feature extraction over the time domain (flux, temporal profiles, frequencies).
"""
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

from mallorn.utils import safe_divide_scalar

def compute_flux_features(t, f, e):
    """Optimized basic flux statistics."""
    feats = {}
  
    if len(f) == 0:
        return {k: 0.0 for k in [
            'n_obs', 'mean_flux', 'median_flux', 'std_flux', 'min_flux',
            'max_flux', 'amplitude', 'snr_mean', 'snr_median', 'skewness',
            'kurtosis'
        ]}
  
    feats['n_obs'] = len(t)
    feats['mean_flux'] = np.mean(f)
    feats['median_flux'] = np.median(f)
    feats['std_flux'] = np.std(f)
    feats['min_flux'] = np.min(f)
    feats['max_flux'] = np.max(f)
    feats['amplitude'] = feats['max_flux'] - feats['min_flux']
  
    snr = f / (e + 1e-10)
    feats['snr_mean'] = np.mean(snr)
    feats['snr_median'] = np.median(snr)
  
    feats['skewness'] = stats.skew(f) if len(f) > 2 else 0.0
    feats['kurtosis'] = stats.kurtosis(f) if len(f) > 3 else 0.0
  
    return feats

def compute_temporal_features(t, f):
    """Optimized temporal features."""
    feats = {}
  
    if len(t) < 2:
        return {k: 0.0 for k in [
            'duration', 'n_obs', 'cadence_mean', 'cadence_std', 'cadence_median',
            'cadence_min', 'cadence_max', 'cadence_iqr', 'irregular_cadence',
            'time_to_peak', 'time_from_peak', 'peak_flux', 'rise_rate', 'decay_rate',
            'rise_time_10_90', 'decay_time_90_10', 'asymmetry',
            'slope', 'slope_err', 'curvature', 'auc', 'auc_rise', 'auc_decay',
            'peak_prominence', 'n_peaks', 'time_frac_rising', 'flux_ratio_early_late',
            'linear_trend_strength', 'mean_rise_rate', 'mean_decay_rate', 'flux_skewness'
        ]}
  
    # Sort by time
    idx = np.argsort(t)
    t, f = t[idx], f[idx]
  
    # Basic temporal
    feats['duration'] = t[-1] - t[0]
    feats['n_obs'] = len(t)
  
    # Cadence statistics (vectorized)
    dt = np.diff(t)
    feats['cadence_mean'] = np.mean(dt)
    feats['cadence_std'] = np.std(dt)
    feats['cadence_median'] = np.median(dt)
    feats['cadence_min'] = np.min(dt)
    feats['cadence_max'] = np.max(dt)
    p25, p75 = np.percentile(dt, [25, 75])
    feats['cadence_iqr'] = p75 - p25
    feats['irregular_cadence'] = feats['cadence_std'] / (feats['cadence_mean'] + 1e-10)
  
    # Peak features
    idx_peak = np.argmax(f)
    t_peak = t[idx_peak]
    f_peak = f[idx_peak]
  
    feats['time_to_peak'] = t_peak - t[0]
    feats['time_from_peak'] = t[-1] - t_peak
    feats['peak_flux'] = f_peak
    feats['time_frac_rising'] = feats['time_to_peak'] / (feats['duration'] + 1e-10)
  
    # Rise and decay rates
    if idx_peak > 0:
        feats['rise_rate'] = (f_peak - f[0]) / (t_peak - t[0] + 1e-10)
        # Mean rise rate
        feats['mean_rise_rate'] = np.mean(np.diff(f[:idx_peak+1]) / (np.diff(t[:idx_peak+1]) + 1e-10))
    else:
        feats['rise_rate'] = 0.0
        feats['mean_rise_rate'] = 0.0
  
    if idx_peak < len(f) - 1:
        feats['decay_rate'] = (f[-1] - f_peak) / (t[-1] - t_peak + 1e-10)
        # Mean decay rate
        feats['mean_decay_rate'] = np.mean(np.diff(f[idx_peak:]) / (np.diff(t[idx_peak:]) + 1e-10))
    else:
        feats['decay_rate'] = 0.0
        feats['mean_decay_rate'] = 0.0
  
    # Rise/decay times
    threshold_low = 0.1 * f_peak
    threshold_high = 0.9 * f_peak
  
    # Rise time calculation
    pre_peak_mask = t <= t_peak
    if np.sum(pre_peak_mask) > 1:
        t_pre = t[pre_peak_mask]
        f_pre = f[pre_peak_mask]
        above_low = np.where(f_pre > threshold_low)[0]
        above_high = np.where(f_pre > threshold_high)[0]
      
        if len(above_low) > 0 and len(above_high) > 0:
            feats['rise_time_10_90'] = t_pre[above_high[0]] - t_pre[above_low[0]]
        else:
            feats['rise_time_10_90'] = 0.0
    else:
        feats['rise_time_10_90'] = 0.0
  
    # Decay time calculation
    post_peak_mask = t >= t_peak
    if np.sum(post_peak_mask) > 1:
        t_post = t[post_peak_mask]
        f_post = f[post_peak_mask]
        below_high = np.where(f_post < threshold_high)[0]
        below_low = np.where(f_post < threshold_low)[0]
      
        if len(below_high) > 0 and len(below_low) > 0:
            feats['decay_time_90_10'] = t_post[below_low[0]] - t_post[below_high[0]]
        else:
            feats['decay_time_90_10'] = 0.0
    else:
        feats['decay_time_90_10'] = 0.0
  
    # Asymmetry
    feats['asymmetry'] = safe_divide_scalar(
        float(feats['decay_time_90_10']),
        float(feats['rise_time_10_90']),
        1.0
    )
    
    # Skewness — TDE light curves are typically right-skewed
    feats['flux_skewness'] = stats.skew(f) if len(f) > 2 else 0.0
  
    # Peak analysis
    if len(f) > 3:
        peaks, properties = find_peaks(f, prominence=np.std(f)/2)
        feats['n_peaks'] = len(peaks)
        feats['peak_prominence'] = np.max(properties['prominences']) if len(peaks) > 0 else 0.0
    else:
        feats['n_peaks'] = 0
        feats['peak_prominence'] = 0.0
  
    # Linear fit
    try:
        t_centered = t - t.mean()
        A = np.vstack([t_centered, np.ones(len(t))]).T
        result = np.linalg.lstsq(A, f, rcond=None)
        feats['slope'] = result[0][0]
      
        residuals = f - (result[0][0] * t_centered + result[0][1])
        feats['slope_err'] = np.std(residuals)
      
        # R² score
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((f - np.mean(f))**2)
        feats['linear_trend_strength'] = 1 - ss_res / (ss_tot + 1e-10)
    except:
        feats['slope'] = 0.0
        feats['slope_err'] = 0.0
        feats['linear_trend_strength'] = 0.0
  
    # Curvature
    feats['curvature'] = np.mean(np.diff(f, 2)) if len(f) >= 3 else 0.0
  
    # Area under curve
    feats['auc'] = np.trapezoid(f, t)
    feats['auc_rise'] = np.trapezoid(f[:idx_peak+1], t[:idx_peak+1]) if idx_peak > 0 else 0.0
    feats['auc_decay'] = np.trapezoid(f[idx_peak:], t[idx_peak:]) if idx_peak < len(f) - 1 else 0.0
  
    # Early vs late flux
    mid_idx = len(f) // 2
    feats['flux_ratio_early_late'] = safe_divide_scalar(
        float(np.mean(f[:mid_idx])),
        float(np.mean(f[mid_idx:])),
        1.0
    ) if mid_idx > 0 else 1.0
  
    return feats

def compute_per_band_temporal_features(t, f, e):
    """
    Per-band rise/decay/asymmetry features.
    
    Critical for TDE detection:
    - TDEs: fast rise (~10-30 days), slow decay (~60-180 days)
    - SNe Ia: symmetric rise/decay (~15-20 days each)
    - AGN: irregular, no clear peak
    """
    feats = {}
    
    if len(t) < 5:
        return {
            'rise_time': 0.0, 'decay_time': 0.0, 'temporal_asymmetry': 1.0,
            'peak_median_ratio': 1.0, 'flux_variability_rate': 0.0,
            'time_above_half_max': 0.0, 'fwhm_time': 0.0
        }
    
    idx = np.argsort(t)
    t, f, e = t[idx], f[idx], e[idx]
    
    idx_peak = np.argmax(f)
    t_peak = t[idx_peak]
    f_peak = f[idx_peak]
    
    t_first = t[0]
    feats['rise_time'] = t_peak - t_first
    
    t_last = t[-1]
    feats['decay_time'] = t_last - t_peak
    
    feats['temporal_asymmetry'] = safe_divide_scalar(
        float(feats['decay_time']), float(feats['rise_time']), 1.0
    )
    
    f_median = np.median(f)
    feats['peak_median_ratio'] = safe_divide_scalar(float(f_peak), float(f_median), 1.0)
    
    dt = np.diff(t)
    df = np.abs(np.diff(f))
    feats['flux_variability_rate'] = np.mean(df / (dt + 1e-10)) if len(dt) > 0 else 0.0
    
    half_max = f_peak / 2
    above_half = f > half_max
    if np.sum(above_half) > 1:
        t_above = t[above_half]
        feats['time_above_half_max'] = t_above[-1] - t_above[0]
        feats['fwhm_time'] = feats['time_above_half_max']
    else:
        feats['time_above_half_max'] = 0.0
        feats['fwhm_time'] = 0.0
    
    return feats

def compute_frequency_features(t, f):
    """Optimized frequency domain features."""
    feats = {}
  
    if len(t) < 10:
        return {k: 0.0 for k in [
            'dominant_freq', 'spectral_entropy', 'spectral_centroid',
            'spectral_rolloff', 'power_ratio_low_high', 'spectral_spread',
            'spectral_flatness'
        ]}
  
    idx = np.argsort(t)
    t, f = t[idx], f[idx]
  
    n_samples = len(t)
    t_uniform = np.linspace(t[0], t[-1], n_samples)
    f_interp = np.interp(t_uniform, t, f)
  
    fft_vals = np.abs(fft(f_interp))
    freqs = fftfreq(n_samples, d=np.mean(np.diff(t_uniform)))
  
    pos_mask = freqs > 0
    fft_vals = fft_vals[pos_mask]
    freqs = freqs[pos_mask]
  
    if len(fft_vals) == 0:
        return {k: 0.0 for k in feats.keys()}
  
    psd = fft_vals**2
    psd_norm = psd / (np.sum(psd) + 1e-10)
  
    feats['dominant_freq'] = freqs[np.argmax(fft_vals)]
    feats['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    feats['spectral_centroid'] = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    feats['spectral_spread'] = np.sqrt(
        np.sum(((freqs - feats['spectral_centroid'])**2) * psd) / (np.sum(psd) + 1e-10)
    )
  
    cumsum = np.cumsum(psd)
    rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
    feats['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
  
    mid_idx = len(freqs) // 2
    low_power = np.sum(psd[:mid_idx])
    high_power = np.sum(psd[mid_idx:])
    feats['power_ratio_low_high'] = safe_divide_scalar(float(low_power), float(high_power), 1.0)
  
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
    arithmetic_mean = np.mean(psd)
    feats['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
  
    return feats

def compute_variability_features(t, f, e):
    """Optimized variability features."""
    feats = {}
  
    if len(f) < 3:
        return {k: 0.0 for k in [
            'reduced_chi2', 'stetson_j', 'stetson_k', 'von_neumann',
            'excess_variance', 'rms', 'mad', 'iqr', 'autocorr_1',
            'autocorr_2', 'beyond_1std', 'beyond_2std', 'beyond_3std',
            'welch_stetson', 'con', 'kurtosis_residual', 'eta_e'
        ]}
  
    w = 1.0 / (e**2 + 1e-10)
    wmean = np.sum(w * f) / np.sum(w)
  
    chi2 = np.sum(w * (f - wmean)**2)
    feats['reduced_chi2'] = chi2 / (len(f) - 1)
  
    resid = (f - wmean) / (e + 1e-10)
  
    if len(resid) > 1:
        pairs = resid[:-1] * resid[1:]
        feats['stetson_j'] = np.sum(np.sign(pairs) * np.sqrt(np.abs(pairs))) / len(resid)
    else:
        feats['stetson_j'] = 0.0
  
    feats['stetson_k'] = np.sum(np.abs(resid)) / (np.sqrt(np.sum(resid**2)) * np.sqrt(len(resid)) + 1e-10)
  
    delta = np.sqrt(len(f) / (len(f) - 1)) * (f - wmean) / (e + 1e-10)
    if len(delta) > 1:
        feats['welch_stetson'] = np.sum(delta[:-1] * delta[1:]) / len(delta)
    else:
        feats['welch_stetson'] = 0.0
  
    feats['von_neumann'] = np.sum((f[1:] - f[:-1])**2) / (np.sum((f - wmean)**2) + 1e-10)
  
    mean_err_sq = np.mean(e**2)
    feats['excess_variance'] = max(0, (np.var(f) - mean_err_sq) / (wmean**2 + 1e-10))
  
    feats['eta_e'] = np.sum((f[1:] - f[:-1])**2) / ((len(f) - 1) * np.var(f) + 1e-10)
  
    sorted_f = np.sort(f)
    n_third = max(1, len(sorted_f) // 3)
    feats['con'] = np.sum(sorted_f[-n_third:]) / (np.sum(sorted_f) + 1e-10)
  
    feats['rms'] = np.sqrt(np.mean(f**2))
    feats['mad'] = np.median(np.abs(f - np.median(f)))
    feats['iqr'] = np.percentile(f, 75) - np.percentile(f, 25)
  
    if len(f) > 2:
        try:
            corr_matrix = np.corrcoef(f[:-1], f[1:])
            feats['autocorr_1'] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        except:
            feats['autocorr_1'] = 0.0
    else:
        feats['autocorr_1'] = 0.0
  
    if len(f) > 3:
        try:
            corr_matrix = np.corrcoef(f[:-2], f[2:])
            feats['autocorr_2'] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        except:
            feats['autocorr_2'] = 0.0
    else:
        feats['autocorr_2'] = 0.0
  
    std = np.std(f)
    mean = np.mean(f)
    abs_dev = np.abs(f - mean)
  
    feats['beyond_1std'] = np.sum(abs_dev > 1 * std) / len(f)
    feats['beyond_2std'] = np.sum(abs_dev > 2 * std) / len(f)
    feats['beyond_3std'] = np.sum(abs_dev > 3 * std) / len(f)
  
    feats['kurtosis_residual'] = stats.kurtosis(resid) if len(resid) > 3 else 0.0
  
    return feats

def compute_tde_physics_features(t, f, e, band='r'):
    from scipy.optimize import curve_fit
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    import warnings
    
    feats = {}
    prefix = f'{band}_tde_'
    
    default_keys = [
        'power_law_D', 'power_law_N', 'power_law_td', 'power_law_chi2',
        'decay_consistency', 'match_score',
        'gp_length_scale', 'gp_amplitude', 'smoothness_ratio',
        'flux_30d_ratio', 'flux_60d_ratio', 'late_time_slope',
        'color_stability', 'timescale'
    ]
    for k in default_keys:
        feats[f'{prefix}{k}'] = 0.0
    feats[f'{prefix}power_law_chi2'] = 1e6
    feats[f'{prefix}flux_30d_ratio'] = 1.0
    feats[f'{prefix}flux_60d_ratio'] = 1.0

    if len(t) < 5:
        return feats
    
    idx = np.argsort(t)
    t, f, e = t[idx], f[idx], e[idx]
    
    idx_peak = np.argmax(f)
    t_peak = t[idx_peak]
    f_peak = f[idx_peak]
    
    post_peak_mask = (t > t_peak) & (f > 0)
    if np.sum(post_peak_mask) >= 5:
        t_decay = t[post_peak_mask]
        f_decay = f[post_peak_mask]
        e_decay = e[post_peak_mask]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            m_decay = -2.5 * np.log10(np.maximum(f_decay, 1e-10))
            w_decay = 1.0 / (2.5 * e_decay / (f_decay * np.log(10)) + 1e-5)**2
        
        valid = np.isfinite(m_decay) & np.isfinite(w_decay)
        
        if np.sum(valid) >= 5:
            def power_law_func(t_obs, N, D, td):
                return N + 2.5 * D * np.log10(np.maximum(t_obs - td + 40, 1.0))
            
            try:
                p0 = [np.mean(m_decay), 1.67, t_peak - 20]
                bounds = ([-np.inf, 0, t_peak - 200], [np.inf, 5, t_peak + 10])
                popt, _ = curve_fit(
                    power_law_func, t_decay[valid], m_decay[valid], 
                    p0=p0, bounds=bounds, sigma=1/np.sqrt(w_decay[valid]), absolute_sigma=True,
                    maxfev=1000
                )
                N_fit, D_fit, td_fit = popt
                m_pred = power_law_func(t_decay[valid], *popt)
                chi2 = np.sum((m_decay[valid] - m_pred)**2 * w_decay[valid]) / (np.sum(valid) - 3)
                
                feats[f'{prefix}power_law_N'] = N_fit
                feats[f'{prefix}power_law_D'] = D_fit
                feats[f'{prefix}power_law_td'] = td_fit
                feats[f'{prefix}power_law_chi2'] = chi2
                feats[f'{prefix}decay_consistency'] = 1.0 / (1.0 + chi2)
                feats[f'{prefix}match_score'] = np.exp(-1.5 * abs(D_fit - (5/3)))
            except:
                try:
                    log_t = np.log10(np.maximum(t_decay[valid] - (t_peak - 20) + 40, 1.0))
                    coeffs = np.polyfit(log_t, m_decay[valid], 1)
                    feats[f'{prefix}power_law_D'] = coeffs[0] / 2.5
                    feats[f'{prefix}power_law_N'] = coeffs[1]
                except:
                    pass

    if len(t) >= 5:
        t_scaled = (t - t[0]).reshape(-1, 1) / 100.0
        f_scaled = (f - np.mean(f)) / (np.std(f) + 1e-10)
        e_scaled = e / (np.std(f) + 1e-10)
        
        if len(t) > 25:
            idx_sub = np.linspace(0, len(t)-1, 25, dtype=int)
            t_gp, f_gp, e_gp = t_scaled[idx_sub], f_scaled[idx_sub], e_scaled[idx_sub]
        else:
            t_gp, f_gp = t_scaled, f_scaled

        try:
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.05, 10.0)) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=(e_gp**2 + 1e-6), n_restarts_optimizer=0)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(t_gp, f_gp)
            
            if hasattr(gp.kernel_, 'k1'):
                rbf_kernel = gp.kernel_.k1.k2
                if isinstance(rbf_kernel, RBF):
                    l_scale = rbf_kernel.length_scale
                else:
                    l_scale = gp.kernel_.k1.get_params().get('k2__length_scale', 0)
            else:
                l_scale = 0
            
            feats[f'{prefix}gp_length_scale'] = l_scale * 100.0
            feats[f'{prefix}gp_amplitude'] = gp.kernel_.k1.k1.constant_value if hasattr(gp.kernel_, 'k1') else 0
        except:
            pass

    for days in [30, 60]:
        target_t = t_peak + days
        if target_t <= t[-1]:
            closest_idx = np.argmin(np.abs(t - target_t))
            feats[f'{prefix}flux_{days}d_ratio'] = f[closest_idx] / (f_peak + 1e-10)
    
    if len(f) >= 3:
        d2f = np.diff(f, 2)
        feats[f'{prefix}smoothness_ratio'] = np.std(f) / (np.std(d2f) + 1e-10)

    late_mask = t > (t_peak + 50)
    if np.sum(late_mask) >= 3:
        try:
            slope, _ = np.polyfit(t[late_mask], f[late_mask], 1)
            feats[f'{prefix}late_time_slope'] = slope
        except: pass
        
    mid_t = t_peak + (t[-1] - t_peak) / 2
    early_mask = (t >= t_peak) & (t < mid_t)
    late_mask = t >= mid_t
    if np.sum(early_mask) >= 2 and np.sum(late_mask) >= 2:
        feats[f'{prefix}color_stability'] = np.var(f[late_mask]) / (np.var(f[early_mask]) + 1e-10)
        
    feats[f'{prefix}timescale'] = t[-1] - t[0]
    
    return feats

def compute_fvar_features(t, f, e):
    """Fractional Variability Amplitude (F_var)."""
    feats = {}
    
    if len(f) < 5:
        return {'fvar': 0.0, 'fvar_err': 1.0, 'excess_variance_norm': 0.0}
    
    mean_flux = np.mean(f)
    variance = np.var(f, ddof=1)
    mean_err_sq = np.mean(e**2)
    
    excess_var = max(0, variance - mean_err_sq)
    fvar = np.sqrt(excess_var) / (np.abs(mean_flux) + 1e-10)
    
    n = len(f)
    if excess_var > 0 and mean_flux != 0:
        term1 = (1/(2*n)) * (mean_err_sq / (mean_flux**2 + 1e-10))**2
        term2 = (mean_err_sq / n) * (2 * fvar / (np.abs(mean_flux) + 1e-10))**2
        fvar_err = np.sqrt(term1 + term2)
    else:
        fvar_err = 1.0
    
    feats['fvar'] = fvar
    feats['fvar_err'] = min(fvar_err, 10.0)
    feats['excess_variance_norm'] = excess_var / (mean_flux**2 + 1e-10)
    
    return feats

def compute_structure_function(t, f):
    """First-order Structure Function SF(τ)."""
    feats = {'sf_10d': 0.0, 'sf_30d': 0.0, 'sf_60d': 0.0, 
             'sf_slope': 0.0, 'sf_ratio_60_10': 1.0}
    
    if len(t) < 10:
        return feats
    
    idx = np.argsort(t)
    t, f = t[idx], f[idx]
    
    lag_bins = [10, 30, 60]
    lag_tolerance = 7
    sf_values = []
    
    for tau in lag_bins:
        diffs = []
        for i in range(len(t) - 1):
            lags = t[i+1:] - t[i]
            mask = np.abs(lags - tau) < lag_tolerance
            if np.any(mask):
                flux_diffs = (f[i+1:][mask] - f[i])**2
                diffs.extend(flux_diffs)
        
        if len(diffs) >= 3:
            sf = np.sqrt(np.mean(diffs))
        else:
            sf = 0.0
        
        sf_values.append(sf)
        feats[f'sf_{tau}d'] = sf
    
    valid_sf = [(tau, sf) for tau, sf in zip(lag_bins, sf_values) if sf > 0]
    if len(valid_sf) >= 2:
        log_tau = np.log10([v[0] for v in valid_sf])
        log_sf = np.log10([v[1] for v in valid_sf])
        slope = np.polyfit(log_tau, log_sf, 1)[0]
        feats['sf_slope'] = slope
    
    if sf_values[0] > 0:
        feats['sf_ratio_60_10'] = sf_values[2] / (sf_values[0] + 1e-10)
    
    return feats

def compute_agn_discriminator_features(f):
    """AGN-specific features that distinguish stochastic from transient behavior."""
    feats = {'negative_flux_frac': 0.0, 'flux_asymmetry': 0.0, 
             'baseline_crossings': 0, 'amplitude_ratio': 1.0}
    
    if len(f) < 3:
        return feats
    
    feats['negative_flux_frac'] = np.sum(f < 0) / len(f)
    
    std = np.std(f)
    if std > 0:
        feats['flux_asymmetry'] = (np.mean(f) - np.median(f)) / std
    
    crossings = np.sum(np.diff(np.sign(f)) != 0)
    feats['baseline_crossings'] = crossings
    
    median_f = np.median(f)
    if median_f > 0:
        feats['amplitude_ratio'] = np.max(f) / median_f
    elif np.max(f) > 0:
        feats['amplitude_ratio'] = 10.0
    
    return feats

def compute_wavelet_features(t, f):
    """Extract wavelet transforms for multi-scale analysis."""
    import pywt
    feats = {}
  
    try:
        t_uniform = np.linspace(t[0], t[-1], 64)
        f_interp = np.interp(t_uniform, t, f)
      
        coeffs = pywt.wavedec(f_interp, 'db4', level=3)
        cA3, cD3, cD2, cD1 = coeffs
      
        feats['wav_approx_power'] = np.sum(cA3**2)
        feats['wav_detail3_power'] = np.sum(cD3**2)
        feats['wav_detail2_power'] = np.sum(cD2**2)
        feats['wav_detail1_power'] = np.sum(cD1**2)
      
        total_power = sum([feats[k] for k in feats]) + 1e-10
        feats['wav_approx_ratio'] = feats['wav_approx_power'] / total_power
        feats['wav_detail2_ratio'] = feats['wav_detail2_power'] / total_power
      
    except:
        return {k: 0.0 for k in [
            'wav_approx_power', 'wav_detail3_power', 'wav_detail2_power',
            'wav_detail1_power', 'wav_approx_ratio', 'wav_detail2_ratio'
        ]}
  
    return feats

def compute_gp_features(t, f, e):
    """Fit a lightweight Gaussian Process to measure intrinsic smoothness."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    import warnings
    feats = {'gp_length_scale': 1.0, 'gp_amplitude': 1.0, 'gp_noise': 0.1, 'gp_marginal_likelihood': -100.0}
  
    try:
        t_scaled = (t - t.min()).reshape(-1, 1) / (t.max() - t.min() + 1e-10)
        f_scaled = (f - f.min()) / (f.max() - f.min() + 1e-10)
        alpha = (e / (f.max() - f.min() + 1e-10))**2
      
        if len(t) > 30:
            idx = np.linspace(0, len(t)-1, 30, dtype=int)
            t_gp, f_gp, a_gp = t_scaled[idx], f_scaled[idx], alpha[idx]
        else:
            t_gp, f_gp, a_gp = t_scaled, f_scaled, alpha
      
        kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(0.01, 10.0)) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=a_gp, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0)
      
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(t_gp, f_gp)
          
        if hasattr(gp.kernel_, 'k1'):
            if hasattr(gp.kernel_.k1, 'k2'):
                feats['gp_length_scale'] = gp.kernel_.k1.k2.length_scale
            if hasattr(gp.kernel_.k1, 'k1'):
                feats['gp_amplitude'] = gp.kernel_.k1.k1.constant_value
            feats['gp_noise'] = gp.kernel_.k2.noise_level
            feats['gp_marginal_likelihood'] = gp.log_marginal_likelihood_value_
    except:
        pass
          
    return feats
