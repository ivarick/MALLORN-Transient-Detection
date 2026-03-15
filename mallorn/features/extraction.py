"""
Core feature extraction orchestration for the MALLORN pipeline.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from mallorn.config import cfg
from mallorn.utils import format_split_name
from mallorn.features.physics import (
    apply_deextinction, fit_power_law, fit_rise_decay_power_law,
    calculate_absolute_magnitudes, fit_bazin
)
from mallorn.features.time_domain import (
    compute_flux_features, compute_temporal_features, compute_frequency_features,
    compute_variability_features, compute_wavelet_features, compute_gp_features,
    compute_per_band_temporal_features, compute_tde_physics_features,
    compute_fvar_features, compute_structure_function,
    compute_agn_discriminator_features
)
from mallorn.features.color import (
    compute_color_features, compute_cross_band_features,
    compute_multiband_tde_features, compute_multiband_features,
    compute_interaction_features
)

def extract_single_object_features(obj_lc, meta):
    """Extract features for a single object (for parallel processing)."""
    try:
        return extract_all_features(obj_lc, meta)
    except Exception as e:
        return {'error': str(e)}

def extract_all_features(lc, meta):
    """Complete optimized feature extraction."""
    all_feats = {}
  
    # Metadata features
    z = float(meta.get("Z", 0.0) or 0.0)
    ebv = float(meta.get("EBV", 0.0) or 0.0)
  
    all_feats['Z'] = z
    all_feats['Z_squared'] = z ** 2
    all_feats['Z_cubed'] = z ** 3
    all_feats['Z_log1p'] = np.log1p(z)
    all_feats['Z_sqrt'] = np.sqrt(max(0, z))
    all_feats['Z_inv'] = 1.0 / (z + 0.01)
    all_feats['EBV'] = ebv
    all_feats['EBV_squared'] = ebv ** 2
    all_feats['EBV_log1p'] = np.log1p(ebv)
    all_feats['Z_EBV_interaction'] = z * ebv
  
    # Per-band features
    all_flux = []
    all_time = []
    all_err = []
    band_data = {}  # Store per-band data for cross-band features
  
    for band in cfg.FILTERS:
        df = lc[lc["Filter"] == band].sort_values("Time (MJD)")
        t_obs = df["Time (MJD)"].values
        f_obs = df["Flux"].values
        e_obs = df["Flux_err"].values
        
        # 1. Physics: De-extinction (Correct for Milky Way Dust)
        f = apply_deextinction(f_obs, ebv, band)
        e = apply_deextinction(e_obs, ebv, band)
        
        # 2. Physics: Time Dilation Correction (Rest-Frame Time)
        t = t_obs 
        z_factor = 1.0 / (1.0 + max(float(z), 0.0))
        
        if len(t) > 0:
            all_flux.extend(f)
            # Use Observed Time for global features to maintain consistency with other obs-frame features
            all_time.extend(t) 
            all_err.extend(e)
      
        # Flux features
        flux_feats = compute_flux_features(t, f, e)
        for k, v in flux_feats.items():
            all_feats[f"{band}_{k}"] = v
      
        # Temporal features
        temp_feats = compute_temporal_features(t, f)
        for k, v in temp_feats.items():
            all_feats[f"{band}_{k}"] = v
      
        # Frequency features
        freq_feats = compute_frequency_features(t, f)
        for k, v in freq_feats.items():
            all_feats[f"{band}_{k}"] = v
      
        # Variability features
        var_feats = compute_variability_features(t, f, e)
        for k, v in var_feats.items():
            all_feats[f"{band}_{k}"] = v
      
        # Wavelet features
        if cfg.USE_WAVELETS and len(t) >= 16:
            wav_feats = compute_wavelet_features(t, f)
            for k, v in wav_feats.items():
                all_feats[f"{band}_{k}"] = v
      
        # GP features
        if cfg.USE_GP_FEATURES and len(t) >= 5:
            gp_feats = compute_gp_features(t, f, e)
            for k, v in gp_feats.items():
                all_feats[f"{band}_{k}"] = v
      
        # Power-law fit
        if len(t) > 0:
            idx_peak = np.argmax(f)
            t_peak = t[idx_peak]
            pl_feats = fit_power_law(t, f, t_peak)
            for k, v in pl_feats.items():
                all_feats[f"{band}_{k}"] = v
            
            # Robust Per-band Power-Law (TDE t^-5/3 signature)
            if band in ['r', 'g', 'i'] and len(t) >= 5:
                rd_pl_feats = fit_rise_decay_power_law(t, f, t_peak, e)  # Pass errors
                for k, v in rd_pl_feats.items():
                    all_feats[f"{band}_{k}"] = v
        
        # Bazin curve fitting (PLAsTiCC-proven)
        if len(t) >= 10:
            bazin_feats = fit_bazin(t, f)
            for k, v in bazin_feats.items():
                all_feats[f"{band}_{k}"] = v
        
        # Per-band temporal features (rise/decay/asymmetry)
        if len(t) >= 5:
            temporal_feats = compute_per_band_temporal_features(t, f, e)
            for k, v in temporal_feats.items():
                all_feats[f"{band}_{k}"] = v
        
        # TDE-specific physics features (power-law D, GP length scale, etc.)
        # Optimized: Only compute for priority bands to save time
        if len(t) >= 5 and band in ['g', 'r', 'i']:
            tde_feats = compute_tde_physics_features(t, f, e, band=band)
            for k, v in tde_feats.items():
                all_feats[k] = v  # Already prefixed with band name
        
        # F_var (THE AGN vs TDE discriminator - ALeRCE proven)
        if len(t) >= 5:
            fvar_feats = compute_fvar_features(t, f, e)
            for k, v in fvar_feats.items():
                all_feats[f"{band}_{k}"] = v
        
        # Structure Function (timescale fingerprint)
        if len(t) >= 10:
            sf_feats = compute_structure_function(t, f)
            for k, v in sf_feats.items():
                all_feats[f"{band}_{k}"] = v
        
        # AGN discriminator features
        agn_feats = compute_agn_discriminator_features(f)
        for k, v in agn_feats.items():
            all_feats[f"{band}_{k}"] = v
        
        # Store band data for cross-band features
        if len(t) > 0:
            band_data[band] = {'t': t, 'f': f, 'e': e}
            
        # Add Rest-Frame Variants (Physics)
        # We look for time-domain features we just added for this band and scale them
        keys_to_check = [k for k in all_feats.keys() if k.startswith(f"{band}_")]
        for k in keys_to_check:
             feat_name = k.replace(f"{band}_", "")
             
             # Time features (t' = t / (1+z))
             if any(tk in feat_name for tk in ['time', 'duration', 'period', 'tau']):
                 all_feats[f"{k}_rest"] = all_feats[k] * z_factor
             
             # Rate features (r' = r * (1+z))
             elif 'rate' in feat_name:
                 all_feats[f"{k}_rest"] = all_feats[k] / z_factor
  
    # Aggregate Power-Law Features (Global TDE Signatures)
    pl_distances = []
    pl_ratios = []
    pl_alphas = []
    pl_success_count = 0
    
    for b in ['g', 'r', 'i']:
        # Check if fit succeeded for this band
        if f"{b}_pl_fit_success" in all_feats and all_feats[f"{b}_pl_fit_success"] == 1:
            pl_success_count += 1
            
            if f"{b}_pl_tde_distance" in all_feats:
                pl_distances.append(all_feats[f"{b}_pl_tde_distance"])
            
            if f"{b}_pl_vs_exp_ratio" in all_feats:
                pl_ratios.append(all_feats[f"{b}_pl_vs_exp_ratio"])
                
            if f"{b}_pl_decay_alpha" in all_feats:
                pl_alphas.append(all_feats[f"{b}_pl_decay_alpha"])
    
    # Aggregate Features
    all_feats['global_pl_success_count'] = pl_success_count
    
    if pl_distances:
        all_feats['global_pl_tde_distance_min'] = np.min(pl_distances)  # Best match
        all_feats['global_pl_tde_distance_median'] = np.median(pl_distances)
    else:
        all_feats['global_pl_tde_distance_min'] = np.nan
        all_feats['global_pl_tde_distance_median'] = np.nan
        
    if pl_ratios:
        all_feats['global_pl_vs_exp_ratio_min'] = np.min(pl_ratios)  # Best discriminator
        all_feats['global_pl_vs_exp_ratio_median'] = np.median(pl_ratios)
    else:
        all_feats['global_pl_vs_exp_ratio_min'] = np.nan
        all_feats['global_pl_vs_exp_ratio_median'] = np.nan
        
    if pl_alphas:
        all_feats['global_pl_decay_alpha_mean'] = np.mean(pl_alphas)
    else:
        all_feats['global_pl_decay_alpha_mean'] = np.nan

    # Color features
    color_feats = compute_color_features(band_data)
    all_feats.update(color_feats)
  
    # Cross-band features (peak time deltas, color evolution)
    if len(band_data) >= 2:
        cross_band_feats = compute_cross_band_features(lc, band_data)
        all_feats.update(cross_band_feats)
        
        # Multiband TDE features (color stability, chromatic behavior)
        tde_multiband_feats = compute_multiband_tde_features(band_data)
        all_feats.update(tde_multiband_feats)

        # Rest-Frame Metrics (Absolute Magnitude, Luminosity)
        abs_mag_feats = calculate_absolute_magnitudes(band_data, z)
        all_feats.update(abs_mag_feats)
  
    # Multiband features
    mb_feats = compute_multiband_features(band_data)
    all_feats.update(mb_feats)
  
    # Global features
    if len(all_time) > 0:
        all_time = np.array(all_time)
        all_flux = np.array(all_flux)
        all_err = np.array(all_err)
      
        idx = np.argsort(all_time)
        all_time, all_flux, all_err = all_time[idx], all_flux[idx], all_err[idx]
      
        # All global feature sets
        for prefix, func, args in [
            ('global', compute_flux_features, (all_time, all_flux, all_err)),
            ('global', compute_temporal_features, (all_time, all_flux)),
            ('global', compute_frequency_features, (all_time, all_flux)),
            ('global', compute_variability_features, (all_time, all_flux, all_err))
        ]:
            feats = func(*args)
            for k, v in feats.items():
                all_feats[f"{prefix}_{k}"] = v
      
        # Global wavelets
        if cfg.USE_WAVELETS and len(all_time) >= 16:
            wav_feats = compute_wavelet_features(all_time, all_flux)
            for k, v in wav_feats.items():
                all_feats[f"global_{k}"] = v
      
        # Global GP
        if cfg.USE_GP_FEATURES and len(all_time) >= 5:
            gp_feats = compute_gp_features(all_time, all_flux, all_err)
            for k, v in gp_feats.items():
                all_feats[f"global_{k}"] = v
      
        # Global power-law
        idx_peak = np.argmax(all_flux)
        t_peak = all_time[idx_peak]
        global_pl = fit_power_law(all_time, all_flux, t_peak)
        for k, v in global_pl.items():
            all_feats[f"global_{k}"] = v
        
        # Rise/Decay Power-Law (TDE Signature)
        global_rd_pl = fit_rise_decay_power_law(all_time, all_flux, t_peak)
        for k, v in global_rd_pl.items():
            all_feats[f"global_{k}"] = v
        
        # Global F_var (critical for AGN vs TDE discrimination)
        if len(all_time) >= 5:
            global_fvar = compute_fvar_features(all_time, all_flux, all_err)
            for k, v in global_fvar.items():
                all_feats[f"global_{k}"] = v
        
        # Global Structure Function
        if len(all_time) >= 10:
            global_sf = compute_structure_function(all_time, all_flux)
            for k, v in global_sf.items():
                all_feats[f"global_{k}"] = v
        
        # Global AGN discriminators
        global_agn = compute_agn_discriminator_features(all_flux)
        for k, v in global_agn.items():
            all_feats[f"global_{k}"] = v
  
    # Interaction features
    interaction_feats = compute_interaction_features(all_feats)
    all_feats.update(interaction_feats)
  
    # Enhanced TDE heuristic score
    tde_score = 0.0
    total_weight = 0.0
  
    # Blue color
    if all_feats.get('color_g_r_mean', 0) < -0.4:
        tde_score += 3.0
        total_weight += 3.0
    elif all_feats.get('color_g_r_mean', 0) < -0.2:
        tde_score += 1.5
        total_weight += 3.0
    else:
        total_weight += 3.0
  
    # Slow decay
    decay_time = all_feats.get('global_decay_time_90_10', 0)
    if decay_time > 60:
        tde_score += 3.0
    elif decay_time > 30:
        tde_score += 1.5
    total_weight += 3.0
  
    # Power-law match
    tde_match = all_feats.get('global_tde_match', 0)
    if tde_match > 0.7:
        tde_score += 3.0
    elif tde_match > 0.5:
        tde_score += 1.5
    total_weight += 3.0
  
    # Smooth lightcurve
    vn = all_feats.get('global_von_neumann', 2)
    if vn < 0.5:
        tde_score += 2.0
    elif vn < 1.0:
        tde_score += 1.0
    total_weight += 2.0
  
    # Long duration
    duration = all_feats.get('global_duration', 0)
    if duration > 150:
        tde_score += 2.0
    elif duration > 80:
        tde_score += 1.0
    total_weight += 2.0
  
    # High SNR
    if all_feats.get('global_snr_mean', 0) > 15:
        tde_score += 1.5
    elif all_feats.get('global_snr_mean', 0) > 8:
        tde_score += 0.75
    total_weight += 1.5
  
    # Low chromatic lag (achromatic = TDE-like)
    if all_feats.get('peak_time_std', 10) < 5:
        tde_score += 1.5
    elif all_feats.get('peak_time_std', 10) < 10:
        tde_score += 0.75
    total_weight += 1.5
  
    all_feats['tde_heuristic_score'] = tde_score / total_weight if total_weight > 0 else 0.0
    
    # Final Sanitization to prevent Stacker crashes
    for k, v in all_feats.items():
        if not np.isfinite(v):
            all_feats[k] = 0.0
            
    return all_feats

def build_features(log_df, split_lcs):
    """Build feature matrix with optional parallelization."""
    if cfg.PARALLEL_FEATURE_EXTRACTION and cfg.N_JOBS != 1:
        print(f" Using parallel processing with {cfg.N_JOBS} jobs...")
      
        # Prepare data for parallel processing
        tasks = []
        for _, row in log_df.iterrows():
            oid = row["object_id"]
            try:
                split = format_split_name(row["split"])
                lc_df = split_lcs[split]
                obj_lc = lc_df[lc_df["object_id"] == oid]
                tasks.append((obj_lc, row, oid))
            except:
                tasks.append((None, None, oid))
      
        # Parallel feature extraction
        def process_task(task):
            obj_lc, meta, oid = task
            if obj_lc is None:
                return {'object_id': oid}
            try:
                feats = extract_all_features(obj_lc, meta)
                feats['object_id'] = oid
                return feats
            except Exception as e:
                return {'object_id': oid, 'error': str(e)}
      
        features_list = Parallel(n_jobs=cfg.N_JOBS)(
            delayed(process_task)(task) for task in tqdm(tasks, desc="Extracting features")
        )
    else:
        # Sequential processing
        features_list = []
        for _, row in tqdm(log_df.iterrows(), total=len(log_df), desc="Extracting features"):
            oid = row["object_id"]
            try:
                split = format_split_name(row["split"])
                lc_df = split_lcs[split]
                obj_lc = lc_df[lc_df["object_id"] == oid]
              
                feats = extract_all_features(obj_lc, row)
                feats['object_id'] = oid
                features_list.append(feats)
            except Exception as e:
                print(f"Warning: {oid}: {e}")
                features_list.append({'object_id': oid})
  
    feat_df = pd.DataFrame(features_list)
    return log_df[['object_id']].merge(feat_df, on='object_id', how='left')
