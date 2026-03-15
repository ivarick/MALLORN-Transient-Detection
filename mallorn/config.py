"""
Central configuration for the MALLORN astronomical classification pipeline.
"""
class Config:
    BASE_DIR = "./MALLORN"
    SEED = 1771
    N_FOLDS = 5  # Increased for robust CV
    N_REPEATS = 2  # More repeats for threshold stability
    FILTERS = ["u", "g", "r", "i", "z", "y"]
    # LSST Effective Wavelengths (Angstroms)
    FILTER_WAVELENGTHS = {
        'u': 3641, 'g': 4704, 'r': 6155, 
        'i': 7504, 'z': 8695, 'y': 10056
    }
  
    # TDE Physics
    TDE_DECAY_POWER = -5/3
    TDE_TYPICAL_DURATION = 120
  
    # Model configs
    USE_LGBM = True
    USE_XGB = False
    USE_CATBOOST = False  # Ordered boosting for small imbalanced data (disabled)
    USE_NN = True
    USE_RF = True
    USE_ET = False  # Extreme randomization with heavy constraints (disabled)
    USE_GBM = False  # Sklearn GradientBoosting (disabled)
    USE_HGB = True   # HistGradientBoosting with heavy regularization
    USE_RIDGE = False  # Ridge classifier baseline (disabled)
    USE_LOGREG = False  # Logistic Regression with elasticnet (disabled)
    USE_KNN = False   # KNN with cosine distance (disabled)
    USE_SGD = False  # Too weak (F1~0.15), disabled
    USE_ADA = False  # Too weak (F1~0.18), disabled
    USE_TABNET = False  # pytorch_tabnet not installed, disabled
    USE_TRANSFORMER = False  # 0.20 F1, severe model pollution, disabled
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_HEADS = 8
    
    # Temporal Fusion Transformer (TFT) for time-series
    USE_TEMPORAL_FUSION = True 
    TFT_HIDDEN_SIZE = 160
    TFT_ATTENTION_HEADS = 4
    TFT_DROPOUT = 0.1
    
    # ============================================================
    # THRESHOLD CALIBRATION
    # ============================================================
    # TunedThresholdClassifierCV for OOF-optimised decision boundary
    USE_THRESHOLD_CALIBRATION = True
    THRESHOLD_SCORING = 'f1'
    THRESHOLD_CV_FOLDS = 10
    
    # Vision Transformers (SwinV2)
    USE_SWINV2 = False  # Disabled — tensor shape issue pending fix
    SWINV2_PATCH_SIZE = 4
    SWINV2_EMBED_DIM = 48
    SWINV2_WINDOW_SIZE = 8
    SWINV2_IMAGE_SIZE = 128
    SWINV2_BATCH_SIZE = 8
    SWINV2_USE_AMP = True    # Automatic Mixed Precision
    SWINV2_GRADIENT_CHECKPOINTING = False

    @classmethod
    def configure_for_gpu_tier(cls, tier='12GB'):
        """Configure SwinV2 settings based on GPU VRAM tier."""
        if tier == '16GB':
            cls.SWINV2_IMAGE_SIZE = 256
            cls.SWINV2_EMBED_DIM = 96
            cls.SWINV2_WINDOW_SIZE = 16
            cls.SWINV2_BATCH_SIZE = 16
            print(f"Configured for 16GB VRAM: {cls.SWINV2_IMAGE_SIZE}x{cls.SWINV2_IMAGE_SIZE}, dim={cls.SWINV2_EMBED_DIM}")
        elif tier == '12GB':
            cls.SWINV2_IMAGE_SIZE = 128
            cls.SWINV2_EMBED_DIM = 48
            cls.SWINV2_WINDOW_SIZE = 8
            cls.SWINV2_BATCH_SIZE = 8
            print(f"Configured for 12GB VRAM: {cls.SWINV2_IMAGE_SIZE}x{cls.SWINV2_IMAGE_SIZE}, dim={cls.SWINV2_EMBED_DIM}")
        elif tier == '8GB':
            cls.SWINV2_IMAGE_SIZE = 64
            cls.SWINV2_EMBED_DIM = 32
            cls.SWINV2_WINDOW_SIZE = 8
            cls.SWINV2_BATCH_SIZE = 4
            print(f"Configured for 8GB VRAM: {cls.SWINV2_IMAGE_SIZE}x{cls.SWINV2_IMAGE_SIZE}, dim={cls.SWINV2_EMBED_DIM}")
        elif tier == '6GB':
            cls.USE_SWINV2 = False
            cls.USE_TEMPORAL_FUSION = False
            print(f"Configured for 6GB VRAM: Vision transformers disabled")
        else:
            print(f"Unknown tier {tier}, using default 12GB config")
    
    # Data-level Imbalance Handling
    USE_SMOTE = True  # SMOTE for minority oversampling
    USE_ADASYN = True  # ADASYN enabled for adaptive minority oversampling
    SMOTE_K_NEIGHBORS = 5  # k_neighbors for SMOTE (careful with small TDE class)
    SMOTE_SAMPLING_STRATEGY = 'auto'  # or float for specific ratio
    
    # Class imbalance — XGBoost positive class re-weighting
    XGB_SCALE_POS_WEIGHT = 50
    
    # Focal Loss Configuration
    FOCAL_LOSS_GAMMA = 2.0  # Focusing parameter
    FOCAL_LOSS_ALPHA = 0.5  # Class weight for positive class (balanced for P/R equilibrium)
    
    # ============================================================
    # DATA-LEVEL CLASS IMBALANCE MITIGATION
    # ============================================================
    # Conservative SMOTE to avoid distorting decision boundaries
    USE_BORDERLINE_SMOTE = False  # too aggressive; creates noisy synthetic samples
    USE_SVMSMOTE = False  # SVM boundaries too complex for this imbalance ratio
    SMOTE_RATIO = 0.05  # 1:20 minority-to-majority ratio
    
    # Under-sampling Majority Classes
    USE_TOMEK_LINKS = True  # Tomek Links removal for boundary cleaning
    USE_ENN = True  # Edited Nearest Neighbors for cleaning
    USE_ALLKNN = True  # AllKNN for progressive cleaning
    UNDER_SAMPLING_STRATEGY = 'auto'
    
    # Combined SMOTE + ENN / SMOTE + Tomek
    USE_SMOTE_ENN = True  # Combine SMOTE with ENN cleaning
    USE_SMOTE_TOMEK = False  # Combine SMOTE with Tomek links
    
    # Cost-Sensitive Learning
    USE_CLASS_WEIGHTS = True  # Inverse frequency class weights
    CLASS_WEIGHT_METHOD = 'balanced'  # 'balanced' or 'sqrt_balanced'
    
    # ============================================================
    # DATA AUGMENTATION
    # ============================================================
    USE_PHOTOMETRIC_AUGMENTATION = True  # Add Gaussian noise to photometry
    PHOTOMETRIC_NOISE_SCALE = 0.05  # 0.01-0.1 magnitudes
    HETEROSCEDASTIC_NOISE = True  # Band-specific error patterns
    
    # ============================================================
    # LIGHT-CURVE FEATURE LIBRARY
    # ============================================================
    USE_LIGHT_CURVE_FEATURES = True  # light-curve library features
    USE_DMDT_FEATURES = True  # dmdt (Difference Magnitude vs Time) representations
    DMDT_BINS = 16  # 16x16 dmdt histogram
    USE_LSST_FEATURES = True  # LSST-specific feature extractors
    
    # ============================================================
    # PRECISION-RECALL OPTIMIZATION
    # ============================================================
    OPTIMIZE_FOR_F2 = True  # F2 score optimization enabled (β=2 weights recall 2x)
    MIN_PRECISION_THRESHOLD = 0.65  # Minimum acceptable precision
    RECALL_PRIORITY = 0.7  # Weight for recall in optimization (0.5 = balanced)
    
    # Prediction output validation
    VALIDATE_PREDICTIONS = True  # Sanity-check binary output before writing to disk
    PRIOR_TDE_RATE = 0.0063     # Expected TDE prevalence (64/10178 in labelled set)
    TDE_RATE_TOLERANCE = 0.05   # Acceptable relative deviation from prior (±5%)
    
    # Advanced Representation Learning
    USE_CONTRASTIVE_PRETRAIN = True  # SupCon pre-training
    USE_SUPCON = True  # Supervised Contrastive (better for imbalance)
    USE_ARCFACE = True  # ArcFace margin
    ARCFACE_MARGIN_TDE = 0.7
    ARCFACE_MARGIN_OTHER = 0.3
    PRETRAIN_EPOCHS = 50

    # Advanced features
    USE_WAVELETS = True
    USE_GP_FEATURES = True
    PARALLEL_FEATURE_EXTRACTION = True
    N_JOBS = -1
  
    # Feature selection — dimensionality control
    AUTO_FEATURE_SELECTION = True
    N_FEATURES_TREE = 80   # max features for tree models
    N_FEATURES_NN = 200    # max features for neural networks
    N_FEATURES_TO_SELECT = 200
    
    # Feature pre-filtering
    USE_VARIANCE_THRESHOLD = True   # Remove near-zero variance features
    VARIANCE_THRESHOLD = 0.001
    USE_CORRELATION_FILTER = True   # Remove highly correlated features
    CORRELATION_THRESHOLD = 0.95
    MAX_FEATURES_BEFORE_SELECTION = 300

    # Pseudo-labeling
    USE_PSEUDO_LABELS = False
    PSEUDO_THRESHOLD_HIGH = 0.95  # Confidence threshold for positive pseudo-labels
    PSEUDO_THRESHOLD_LOW = 0.00   # Set to 0.0 to disable negative pseudo-labeling
    PSEUDO_ITERATIONS = 3
    
    # Advanced Pseudo-Labeling
    USE_ADVANCED_PSEUDO_LABELS = True
    PSEUDO_THRESHOLD_TDE = 0.95      # High threshold for minority class
    PSEUDO_THRESHOLD_NON_TDE = 0.80  # Lower threshold for majority
    PSEUDO_USE_CALIBRATION = True    # Calibrate probabilities before selection
    PSEUDO_USE_MODEL_AGREEMENT = True
    PSEUDO_AGREEMENT_MIN = 3         # Require 3/4 models to agree
    NOISY_STUDENT_ITERATIONS = 3
    NOISY_STUDENT_NOISE_SCALE = 0.1  # Gaussian noise scale
    THRESHOLD_CURRICULUM = [0.95, 0.90, 0.85]  # Descending thresholds
    
    # FixMatch Consistency Regularization
    USE_FIXMATCH = True  # Disabled by default (expensive)
    FIXMATCH_WEAK_SHIFT = 0.05     # 5% time shift
    FIXMATCH_STRONG_NOISE = 0.2    # 20% noise
    
    # Multi-view co-training
    USE_MULTI_VIEW_COTRAINING = True
    COTRAINING_ITERATIONS = 3
    COTRAINING_CONFIDENCE = 0.90
  
    # Training
    EARLY_STOPPING_ROUNDS = 200
    N_ESTIMATORS = 5000
    N_EPOCHS = 400
    NN_PATIENCE = 60
    BATCH_SIZE = 128
  
    # Calibration
    USE_CALIBRATION = True
    DEVICE = "cpu"
    USE_GPU = True

cfg = Config()
