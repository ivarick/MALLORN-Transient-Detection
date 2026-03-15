"""
Multi-band color relationship, interaction, and stability features.
"""
import numpy as np

from mallorn.config import cfg
from mallorn.utils import safe_divide_scalar

def compute_color_features(band_data):
    """Optimized color features with comprehensive metrics using CORRECTED flux/time."""
    colors = {}
    color_pairs = [("g", "r"), ("r", "i"), ("u", "g"), ("i", "z"), ("g", "i"), ("u", "r"), ("u", "i")]
  
    for b1, b2 in color_pairs:
        prefix = f"color_{b1}_{b2}"
        default_keys = ['_mean', '_std', '_slope', '_at_peak', '_min', '_max', '_range',
                       '_early', '_late', '_evolution', '_median', '_iqr']
      
        if b1 not in band_data or b2 not in band_data:
            for suffix in default_keys:
                colors[f"{prefix}{suffix}"] = 0.0
            continue
            
        data1 = band_data[b1]
        data2 = band_data[b2]
      
        if len(data1['t']) < 2 or len(data2['t']) < 2:
            for suffix in default_keys:
                colors[f"{prefix}{suffix}"] = 0.0
            continue
      
        t1, f1 = data1['t'], data1['f']
        t2, f2 = data2['t'], data2['f']
      
        t_min = max(t1.min(), t2.min())
        t_max = min(t1.max(), t2.max())
      
        if t_max <= t_min:
            for suffix in default_keys:
                colors[f"{prefix}{suffix}"] = 0.0
            continue
      
        # Common time grid
        n_points = min(50, len(t1), len(t2))
        t_common = np.linspace(t_min, t_max, n_points)
      
        try:
            # Interpolate fluxes
            f1_interp = np.interp(t_common, t1, f1)
            f2_interp = np.interp(t_common, t2, f2)
          
            valid = (f1_interp > 0) & (f2_interp > 0)
            if np.sum(valid) < 2:
                for suffix in default_keys:
                    colors[f"{prefix}{suffix}"] = 0.0
                continue
          
            # Compute color
            color = -2.5 * np.log10(f1_interp[valid] / f2_interp[valid])
            t_valid = t_common[valid]
          
            # Statistics
            colors[f"{prefix}_mean"] = np.mean(color)
            colors[f"{prefix}_median"] = np.median(color)
            colors[f"{prefix}_std"] = np.std(color)
            colors[f"{prefix}_min"] = np.min(color)
            colors[f"{prefix}_max"] = np.max(color)
            colors[f"{prefix}_range"] = colors[f"{prefix}_max"] - colors[f"{prefix}_min"]
            colors[f"{prefix}_iqr"] = np.percentile(color, 75) - np.percentile(color, 25)
          
            # Color evolution
            if len(t_valid) > 1:
                colors[f"{prefix}_slope"] = np.polyfit(t_valid, color, 1)[0]
            else:
                colors[f"{prefix}_slope"] = 0.0
          
            # Early vs late
            mid_idx = len(color) // 2
            if mid_idx > 0:
                colors[f"{prefix}_early"] = np.mean(color[:mid_idx])
                colors[f"{prefix}_late"] = np.mean(color[mid_idx:])
                colors[f"{prefix}_evolution"] = colors[f"{prefix}_late"] - colors[f"{prefix}_early"]
            else:
                colors[f"{prefix}_early"] = colors[f"{prefix}_mean"]
                colors[f"{prefix}_late"] = colors[f"{prefix}_mean"]
                colors[f"{prefix}_evolution"] = 0.0
          
            # Color at peak
            idx_peak = np.argmax(f1_interp)
            if idx_peak < len(color):
                colors[f"{prefix}_at_peak"] = color[min(idx_peak, len(color)-1)]
            else:
                colors[f"{prefix}_at_peak"] = colors[f"{prefix}_mean"]
              
        except:
            for suffix in default_keys:
                colors[f"{prefix}{suffix}"] = 0.0
  
    return colors

def compute_cross_band_features(lc, band_data):
    """
    Cross-band interaction features.
    
    Many astronomical classes are defined by color evolution:
    - TDEs: blue at peak, redden slowly
    - SNe Ia: red at peak, bluen then redden
    - AGN: variable colors, no consistent pattern
    """
    feats = {}
    bands = list(band_data.keys())
    
    # Peak time differences between bands (chromatic lags)
    peak_times = {}
    peak_fluxes = {}
    
    for band in bands:
        data = band_data[band]
        if len(data['f']) > 0:
            idx_peak = np.argmax(data['f'])
            peak_times[band] = data['t'][idx_peak]
            peak_fluxes[band] = data['f'][idx_peak]
    
    # Peak flux ratios between bands
    ref_band = 'r' if 'r' in peak_fluxes else (bands[0] if bands else None)
    if ref_band and ref_band in peak_fluxes:
        for band in cfg.FILTERS:
            if band in peak_fluxes:
                feats[f'peak_flux_ratio_{band}_r'] = safe_divide_scalar(
                    float(peak_fluxes[band]), float(peak_fluxes[ref_band]), 1.0
                )
            else:
                feats[f'peak_flux_ratio_{band}_r'] = 0.0
    else:
        for band in cfg.FILTERS:
            feats[f'peak_flux_ratio_{band}_r'] = 0.0
    
    # Peak time differences (chromatic delays)
    band_pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('u', 'r'), ('g', 'i')]
    for b1, b2 in band_pairs:
        if b1 in peak_times and b2 in peak_times:
            feats[f'peak_time_delta_{b1}_{b2}'] = peak_times[b1] - peak_times[b2]
        else:
            feats[f'peak_time_delta_{b1}_{b2}'] = 0.0
    
    # Color evolution slopes (change in color over time)
    # This captures reddening/bluening behavior
    color_pairs = [('g', 'r'), ('u', 'g'), ('r', 'i')]
    for b1, b2 in color_pairs:
        if b1 in band_data and b2 in band_data:
            df1 = band_data[b1]
            df2 = band_data[b2]
            
            if len(df1['t']) >= 3 and len(df2['t']) >= 3:
                # Interpolate to common times
                t_min = max(df1['t'].min(), df2['t'].min())
                t_max = min(df1['t'].max(), df2['t'].max())
                
                if t_max > t_min:
                    n_points = min(20, len(df1['t']), len(df2['t']))
                    t_common = np.linspace(t_min, t_max, n_points)
                    
                    f1_interp = np.interp(t_common, df1['t'], df1['f'])
                    f2_interp = np.interp(t_common, df2['t'], df2['f'])
                    
                    # Compute color where both are positive
                    valid = (f1_interp > 0) & (f2_interp > 0)
                    if np.sum(valid) >= 3:
                        color = -2.5 * np.log10(f1_interp[valid] / f2_interp[valid])
                        t_valid = t_common[valid]
                        
                        # Color evolution slope
                        slope, _ = np.polyfit(t_valid, color, 1)
                        feats[f'color_slope_{b1}_{b2}'] = slope
                        
                        # Color at early vs late time
                        mid = len(color) // 2
                        if mid > 0:
                            feats[f'color_change_{b1}_{b2}'] = np.mean(color[mid:]) - np.mean(color[:mid])
                        else:
                            feats[f'color_change_{b1}_{b2}'] = 0.0
                    else:
                        feats[f'color_slope_{b1}_{b2}'] = 0.0
                        feats[f'color_change_{b1}_{b2}'] = 0.0
                else:
                    feats[f'color_slope_{b1}_{b2}'] = 0.0
                    feats[f'color_change_{b1}_{b2}'] = 0.0
            else:
                feats[f'color_slope_{b1}_{b2}'] = 0.0
                feats[f'color_change_{b1}_{b2}'] = 0.0
        else:
            feats[f'color_slope_{b1}_{b2}'] = 0.0
            feats[f'color_change_{b1}_{b2}'] = 0.0
    
    return feats

def compute_multiband_tde_features(band_data):
    """
    Complex multiband behavior specific to TDEs vs AGN vs SNe.
    """
    feats = {}
    
    # Calculate colors safely
    def get_color(b1, b2, idx_type='peak'):
        try:
            if b1 in band_data and b2 in band_data:
                f1_data = band_data[b1]['f']
                f2_data = band_data[b2]['f']
                if len(f1_data) == 0 or len(f2_data) == 0:
                    return np.nan
                    
                if idx_type == 'peak':
                    f1 = np.max(f1_data)
                    f2 = np.max(f2_data)
                elif idx_type == 'mean':
                    f1 = np.mean(f1_data)
                    f2 = np.mean(f2_data)
                else:
                    return np.nan
                    
                if f1 > 0 and f2 > 0:
                    return -2.5 * np.log10(f1/f2)
            return np.nan
        except:
            return np.nan

    g_r_peak = get_color('g', 'r', 'peak')
    g_r_mean = get_color('g', 'r', 'mean')
    u_g_peak = get_color('u', 'g', 'peak')
    
    # 1. Color stability: TDEs keep stable temperature for months
    if not np.isnan(g_r_peak) and not np.isnan(g_r_mean):
        feats['color_stability_g_r'] = abs(g_r_peak - g_r_mean)
    else:
        feats['color_stability_g_r'] = 0.0
        
    # 2. Extreme blueness: TDEs are hotter than most SNe
    if not np.isnan(u_g_peak):
        feats['is_extremely_blue'] = 1.0 if u_g_peak < -0.5 else 0.0
    else:
        feats['is_extremely_blue'] = 0.0
        
    # 3. Peak bolometric proxy (sum of fluxes at peak)
    peak_fluxes = [np.max(d['f']) for b, d in band_data.items() if len(d['f']) > 0]
    if peak_fluxes:
        feats['pseudo_bolometric_peak'] = np.sum(peak_fluxes)
    else:
        feats['pseudo_bolometric_peak'] = 0.0
        
    return feats

def compute_multiband_features(band_data):
    """Enhanced multiband features using CORRECTED flux/time."""
    feats = {}
    n_bands = len(band_data)
    feats['n_bands'] = n_bands
    feats['n_bands_squared'] = n_bands ** 2
  
    # Peak flux statistics
    peak_fluxes = {band: np.max(data['f']) for band, data in band_data.items() if len(data['f']) > 0}
  
    if peak_fluxes:
        ref_band = 'r' if 'r' in peak_fluxes else list(peak_fluxes.keys())[0]
        ref_flux = peak_fluxes[ref_band]
      
        for band in cfg.FILTERS:
            feats[f'peak_ratio_{band}_to_ref'] = safe_divide_scalar(
                float(peak_fluxes.get(band, 0)),
                float(ref_flux),
                0.0
            )
      
        peak_values = list(peak_fluxes.values())
        feats['peak_flux_mean_bands'] = np.mean(peak_values)
        feats['peak_flux_std_bands'] = np.std(peak_values)
        feats['peak_flux_range_bands'] = np.max(peak_values) - np.min(peak_values)
        feats['peak_flux_ratio_min_max'] = safe_divide_scalar(
            float(np.min(peak_values)), 
            float(np.max(peak_values)), 
            0.0
        )
    else:
        for band in cfg.FILTERS:
            feats[f'peak_ratio_{band}_to_ref'] = 0.0
        feats['peak_flux_mean_bands'] = 0.0
        feats['peak_flux_std_bands'] = 0.0
        feats['peak_flux_range_bands'] = 0.0
        feats['peak_flux_ratio_min_max'] = 0.0
  
    # Time of peak in each band
    peak_times = {}
    for band, data in band_data.items():
        if len(data['f']) > 0:
            idx = np.argmax(data['f'])
            peak_times[band] = data['t'][idx]
  
    # Peak time analysis
    if len(peak_times) >= 2:
        times = list(peak_times.values())
        feats['peak_time_spread'] = np.max(times) - np.min(times)
        feats['peak_time_std'] = np.std(times)
        feats['peak_time_mean'] = np.mean(times)
      
        # Chromatic lags (important for TDE identification)
        if 'u' in peak_times and 'i' in peak_times:
            feats['lag_u_i'] = peak_times['u'] - peak_times['i']
        else:
            feats['lag_u_i'] = 0.0
          
        if 'g' in peak_times and 'r' in peak_times:
            feats['lag_g_r'] = peak_times['g'] - peak_times['r']
        else:
            feats['lag_g_r'] = 0.0
      
        if 'u' in peak_times and 'z' in peak_times:
            feats['lag_u_z'] = peak_times['u'] - peak_times['z']
        else:
            feats['lag_u_z'] = 0.0
            
        if 'g' in peak_times and 'i' in peak_times:
            feats['lag_g_i'] = peak_times['g'] - peak_times['i']
        else:
            feats['lag_g_i'] = 0.0
            
        if 'r' in peak_times and 'z' in peak_times:
            feats['lag_r_z'] = peak_times['r'] - peak_times['z']
        else:
            feats['lag_r_z'] = 0.0
    else:
        for k in ['peak_time_spread', 'peak_time_std', 'peak_time_mean',
                  'lag_u_i', 'lag_g_r', 'lag_u_z', 'lag_g_i', 'lag_r_z']:
            feats[k] = 0.0
  
    feats['total_obs'] = sum(len(data['f']) for data in band_data.values())
  
    # Observation density
    if n_bands > 0:
        all_t = np.concatenate([data['t'] for data in band_data.values()])
        duration = np.max(all_t) - np.min(all_t) if len(all_t) > 1 else 1.0
        feats['obs_density'] = len(all_t) / (duration + 1e-10)
        obs_per_band = [len(data['f']) for data in band_data.values()]
        feats['obs_uniformity'] = np.std(obs_per_band) / (np.mean(obs_per_band) + 1e-10)
    else:
        feats['obs_density'] = 0.0
        feats['obs_uniformity'] = 0.0
  
    return feats

def compute_interaction_features(all_feats):
    """Physics-motivated interaction features."""
    interactions = {}
  
    # Power-law × TDE match
    if 'global_power_law_index' in all_feats and 'global_tde_match' in all_feats:
        interactions['powerlaw_tde_interaction'] = (
            all_feats['global_power_law_index'] * all_feats['global_tde_match']
        )
  
    # Blue color × slow decay (strong TDE signal)
    if 'color_g_r_mean' in all_feats and 'global_decay_time_90_10' in all_feats:
        is_blue = float(all_feats['color_g_r_mean'] < -0.2)
        is_slow = float(all_feats['global_decay_time_90_10'] > 30)
        interactions['blue_slow_decay'] = is_blue * is_slow
        interactions['blue_times_decay'] = all_feats['color_g_r_mean'] * all_feats['global_decay_time_90_10']
  
    # Redshift × duration (time dilation)
    if 'Z' in all_feats and 'global_duration' in all_feats:
        interactions['z_duration'] = all_feats['Z'] * all_feats['global_duration']
        interactions['z_squared_duration'] = (all_feats['Z'] ** 2) * all_feats['global_duration']
  
    # Peak flux × SNR
    if 'global_peak_flux' in all_feats and 'global_snr_mean' in all_feats:
        interactions['peak_snr'] = all_feats['global_peak_flux'] * all_feats['global_snr_mean']
  
    # Asymmetry × power-law residual
    if 'global_asymmetry' in all_feats and 'global_power_law_residual' in all_feats:
        interactions['asym_powerlaw'] = (
            all_feats['global_asymmetry'] * all_feats['global_power_law_residual']
        )
  
    # Coverage quality
    if 'n_bands' in all_feats and 'obs_density' in all_feats:
        interactions['coverage_quality'] = all_feats['n_bands'] * all_feats['obs_density']
  
    # Smoothness score
    if 'global_von_neumann' in all_feats and 'global_excess_variance' in all_feats:
        interactions['smoothness_score'] = (
            (2.0 - all_feats['global_von_neumann']) * (1.0 - all_feats['global_excess_variance'])
        )
  
    # Color evolution × duration
    if 'color_g_r_evolution' in all_feats and 'global_duration' in all_feats:
        interactions['color_evolution_duration'] = (
            all_feats['color_g_r_evolution'] * all_feats['global_duration']
        )
  
    # Rise rate × decay rate ratio
    if 'global_rise_rate' in all_feats and 'global_decay_rate' in all_feats:
        interactions['rise_decay_ratio'] = safe_divide_scalar(
            float(abs(all_feats['global_rise_rate'])),
            float(abs(all_feats['global_decay_rate']) + 1e-10),
            1.0
        )
        
    # Multi-band peak synchronization
    if 'peak_time_std' in all_feats and 'n_bands' in all_feats:
        interactions['peak_synchronization'] = all_feats['n_bands'] / (all_feats['peak_time_std'] + 1.0)
  
    return interactions
