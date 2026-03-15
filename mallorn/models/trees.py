"""
Tree-based model parameters (LightGBM, XGBoost, CatBoost).
"""
from mallorn.config import cfg

def get_lgbm_params():
    """
    High-performance LightGBM parameters optimized for GPU and class imbalance.
    Includes DARTS-inspired regularization and refined leaf-wise growth.
    """
    return {
        # Hardware Acceleration
        'device': 'gpu',                # Enable GPU training
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'n_jobs': -1,

        # Core Task
        'objective': 'binary',
        'metric': 'binary_logloss',     # Most stable for early stopping
        'boosting_type': 'gbdt',
        'seed': cfg.SEED,

        # Tree Structure (Leaf-wise)
        'num_leaves': 31,               # Reduced from 64 to prevent overfitting (Phase 6.5)
        'max_depth': 6,                 # Shallow-to-medium depth to prevent overfitting
        'min_child_samples': 50,        # Increased significantly for stability
        'min_child_weight': 1e-3,
        'min_split_gain': 0.05,         # More conservative split requirement

        # Learning Rate & Speed
        'learning_rate': 0.015,         # Lowered for better convergence on GPU
        'num_iterations': 10000,        # Use with early_stopping_rounds in train loop

        # Regularization (The "Mallorn" Special)
        'reg_alpha': 1.5,               # L1 regularization to zero out noisy features
        'reg_lambda': 2.5,              # L2 regularization to stabilize weights
        'path_smooth': 0.2,             # Increased smoothing for deeper trees
        'extra_trees': True,            # Set to True: adds randomness to splits (excellent for noise)
        
        # Subsampling & Diversity (sklearn-compatible names)
        'colsample_bytree': 0.6,        # Subsample features per iteration
        'subsample': 0.75,              # Subsample data rows
        'subsample_freq': 5,            # Perform bagging every 5 iterations
        'colsample_bynode': 0.8,        # Added: Subsample features per node split

        # Imbalance Strategy
        'scale_pos_weight': 15,         # Reduced from 19 to balance Precision/Recall (Phase 6.5)
        'is_unbalance': False,          # Use either scale_pos_weight OR is_unbalance (not both)
        
        # Verbosity & Misc
        'verbose': -1,
        'max_bin': 255                  # Standard GPU binning
    }

def get_xgb_params():
    """
    GPU-accelerated XGBoost parameters with advanced gradient-based sampling.
    Optimized for heavy class imbalance (1:19) and high-speed training.
    """
    return {
        # Hardware Acceleration
        'device': 'cuda',               # Trigger GPU acceleration
        'tree_method': 'hist',          # Optimized histogram method for GPU
        
        # Core Configuration
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',       # Stable metric for early stopping
        'seed': cfg.SEED,
        'n_jobs': -1,

        # Tree Structure
        'max_depth': 7,                 # Slightly shallower for better GPU memory fit
        'min_child_weight': 10,         # Robustness against noise
        'learning_rate': 0.015,         # Slower, more precise updates
        
        # Sampling (GPU-Exclusive Optimizations)
        'sampling_method': 'gradient_based', # Only works with GPU 'hist'. Much faster/accurate.
        'subsample': 0.7,               
        'colsample_bytree': 0.6,
        'colsample_bylevel': 0.6,
        'colsample_bynode': 0.6,
        
        # Regularization
        'gamma': 0.5,
        'reg_alpha': 1.5,
        'reg_lambda': 3.0,
        
        # Imbalance Strategy
        # Imbalance Strategy - Using scale_pos_weight=158 for 159:1 ratio
        'scale_pos_weight': cfg.XGB_SCALE_POS_WEIGHT if hasattr(cfg, 'XGB_SCALE_POS_WEIGHT') else 158,
        'max_delta_step': 1,            # Stabilizes updates for the 1:159 extreme imbalance
        
        # Memory Efficiency
        'max_bin': 256,
        'grow_policy': 'depthwise'
    }

def get_catboost_params():
    """Optimized CatBoost parameters with GPU memory management."""
    base_params = {
        'iterations': cfg.N_ESTIMATORS,
        'depth': 6,  # Reduced to save GPU memory
        'learning_rate': 0.02,
        'eval_metric': 'Logloss',       # Explicitly set for stability
        'l2_leaf_reg': 5,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'auto_class_weights': 'Balanced',
        'random_seed': cfg.SEED,
        'early_stopping_rounds': cfg.EARLY_STOPPING_ROUNDS,  # Keep this one
        'verbose': 0,
        'leaf_estimation_iterations': 5,
        # Removed od_type and od_wait - they conflict with early_stopping_rounds
    }
    
    # GPU-specific settings to prevent OOM
    device = cfg.DEVICE if hasattr(cfg, 'DEVICE') else 'cpu'
    if device == 'cuda' or cfg.USE_GPU:
        base_params.update({
            'task_type': 'GPU',
            'devices': '0',
            'gpu_ram_part': 0.5,  # Use only 50% of GPU RAM
            'max_ctr_complexity': 2,
            'depth': 5,  # Further reduced for GPU
            'border_count': 128
        })
    else:
        base_params.update({
            'task_type': 'CPU',
            'thread_count': -1
        })
    
    return base_params
