import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import platform
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from typing import Dict, Tuple, Optional, Union, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


# ============================================================
# PERFORMANCE FIX: Enable cuDNN benchmarking
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# ============================================================
# TIER 5: FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for extremely imbalanced classification.
    Down-weights easy examples, focuses on hard-to-classify samples.
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Balancing parameter (default: 0.75)
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# TIER 5: LIGHTCURVE AUGMENTATION
# ============================================================

class LightCurveAugmenter:
    """Time-series augmentation for astronomical lightcurves."""
    def __init__(self, noise_std=0.05, time_jitter=0.02, flux_scale_range=(0.9, 1.1), p=0.5):
        self.noise_std = noise_std
        self.time_jitter = time_jitter
        self.flux_scale_range = flux_scale_range
        self.p = p
    
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        
        x = x.clone()
        
        # Add Gaussian noise to flux values
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        
        # Apply random flux scaling
        scale = np.random.uniform(*self.flux_scale_range)
        x = x * scale
            
        return x


class ContrastiveAugmenter:
    """
    Augmentation pipeline for SimCLR pre-training.
    Returns two augmented views of the same lightcurve.
    """
    def __init__(self, seq_len=100):
        self.seq_len = seq_len
        self.base_augmenter = LightCurveAugmenter(p=1.0)  # Always augment
        
    def __call__(self, x):
        """
        x: [Channels=6, SeqLen]
        Returns (x_i, x_j)
        """
        x_i = self.augment(x.clone())
        x_j = self.augment(x.clone())
        return x_i, x_j
        
    def augment(self, x):
        # 1. Base Augmentation (Flux scale, noise)
        x = self.base_augmenter(x)
        
        # 2. Random masking (Cutout) - simulate observational gaps
        if np.random.random() < 0.5:
            mask_len = int(0.2 * self.seq_len)
            start = np.random.randint(0, max(1, self.seq_len - mask_len))
            x[:, start:start+mask_len] = 0.0
            
        # 3. Channel Dropout (simulate missing band)
        if np.random.random() < 0.3 and x.shape[0] >= 6:
            band_idx = np.random.randint(0, min(6, x.shape[0]))
            x[band_idx, :] = 0.0
            
        # 4. Photometric redshift perturbation
        if np.random.random() < 0.3:
            z_err = np.random.normal(0, 0.1)
            factor = 1.0 + z_err
            x = x / (factor**2)  # Flux ~ 1/d_L^2
            
        return x


# Config
SEQ_LEN = 100
CHANNELS = 2  # Flux, Flux_err


# ============================================================
# DATASET - PERFORMANCE OPTIMIZED
# ============================================================

class LightCurveDataset(Dataset):
    """
    Dataset for lightcurve data with optional tabular features.
    OPTIMIZED: Disabled GP by default, uses fast linear interpolation.
    """
    def __init__(self, splits, object_ids, split_map, labels=None, seq_len=100, 
                 tabular_features=None, use_gp=False):  # CHANGED: GP disabled by default
        self.splits = splits
        self.object_ids = object_ids if isinstance(object_ids, (list, np.ndarray)) else object_ids.values
        self.split_map = split_map
        self.labels = labels
        self.seq_len = seq_len
        self.tabular_features = tabular_features
        self.use_gp = use_gp
        
        self.band_map = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'Y': 5}
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']
        
    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        oid = self.object_ids[idx]
        
        # Determine split and get DF
        split_name = self.split_map.get(oid)
        if split_name and split_name in self.splits:
            full_df = self.splits[split_name]
            df = full_df[full_df['object_id'] == oid]
        else:
            df = self.splits.get(oid, pd.DataFrame())

        # Prepare 6-channel input: [6 Bands, Seq_Len]
        tensor = np.zeros((6, self.seq_len), dtype=np.float32)
        
        if not df.empty:
            # Normalize column names
            col_map = {
                'Filter': 'passband', 'band': 'passband',
                'Flux': 'flux', 'flux': 'flux',
                'Flux_err': 'flux_err', 'flux_err': 'flux_err',
                'Time (MJD)': 'mjd', 'mjd': 'mjd', 'Time': 'mjd'
            }
            df = df.rename(columns=col_map)
            
            # Global Time Normalization for this object
            t_min = df['mjd'].min()
            t_max = df['mjd'].max()
            duration = t_max - t_min
            if duration == 0: 
                duration = 1.0
            
            # Define target uniform time grid
            t_grid = np.linspace(t_min, t_max, self.seq_len)
            
            for band_name, grp in df.groupby('passband'):
                if band_name not in self.band_map: 
                    continue
                b_idx = self.band_map[band_name]
                
                if len(grp) < 2:  # Need at least 2 points for interpolation
                    continue
                
                t = grp['mjd'].values
                f = grp['flux'].values
                e = grp['flux_err'].values
                
                # Normalize Flux (Critical for NN)
                f_mean = np.nanmean(f)
                f_std = np.nanstd(f) + 1e-6
                
                # Nan check for scalar stats
                if np.isnan(f_mean) or np.isnan(f_std):
                    f_mean = 0.0
                    f_std = 1.0
                    
                f_norm = (f - f_mean) / f_std
                e_norm = e / f_std
                
                # Clean potential NaNs in normalized arrays
                f_norm = np.nan_to_num(f_norm, nan=0.0)
                e_norm = np.nan_to_num(e_norm, nan=0.0)
                
                # PERFORMANCE FIX: Use linear interpolation by default
                # GP is 100x slower and often not worth it
                if self.use_gp and len(grp) >= 3:
                    try:
                        # GP Regression (SLOW - only use if explicitly requested)
                        kernel = 1.0 * RBF(length_scale=20.0, length_scale_bounds=(1e-1, 100.0)) + \
                                 WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 10.0))
                        
                        alpha = e_norm**2 + 1e-6
                        
                        gp = GaussianProcessRegressor(
                            kernel=kernel, 
                            alpha=alpha, 
                            normalize_y=False, 
                            n_restarts_optimizer=0
                        )
                        gp.fit(t.reshape(-1, 1), f_norm)
                        f_pred = gp.predict(t_grid.reshape(-1, 1))
                        tensor[b_idx, :] = f_pred
                        
                    except Exception:
                        # Fallback to linear interpolation
                        f_pred = np.interp(t_grid, t, f_norm)
                        tensor[b_idx, :] = f_pred
                else:
                    # Fast Linear Interpolation (DEFAULT)
                    f_pred = np.interp(t_grid, t, f_norm)
                    tensor[b_idx, :] = f_pred
                    
        x = torch.tensor(tensor, dtype=torch.float32)
        
        # Handle tabular features if present
        if self.tabular_features is not None:
            if isinstance(self.tabular_features, pd.DataFrame):
                try:
                    tab_vec = self.tabular_features.loc[oid].values.astype(np.float32)
                except (KeyError, AttributeError):
                    tab_vec = np.zeros(300, dtype=np.float32)
            else:
                # Assume it's a dict {oid: array}
                tab_vec = self.tabular_features.get(oid, np.zeros(300, dtype=np.float32))
            
            tab_tensor = torch.tensor(tab_vec, dtype=torch.float32)
            
            if self.labels is not None:
                y = torch.tensor(self.labels[idx], dtype=torch.float32)
                return (x, tab_tensor), y
            return (x, tab_tensor)
            
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        return x


def normalize_split_name(s):
    """Normalize split names to consistent format."""
    if '_' in s:
        p1, p2 = s.split('_')
        return f"Split_{int(p2):02d}"
    return s


# ============================================================
# MODELS
# ============================================================

class TemporalPositionalEncoding(nn.Module):
    """
    Time-aware positional encoding for irregular time series.
    
    Unlike standard positional encoding which assumes uniform spacing,
    this encodes the actual time values, making it suitable for
    astronomical light curves with irregular cadence.
    """
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.d_model = d_model
        # Learnable time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )
    
    def forward(self, x, time_values):
        """
        x: [batch, seq_len, d_model]
        time_values: [batch, seq_len] - actual MJD values
        """
        # Normalize time to [0, 1] range per sample
        t_min = time_values.min(dim=1, keepdim=True)[0]
        t_max = time_values.max(dim=1, keepdim=True)[0]
        t_norm = (time_values - t_min) / (t_max - t_min + 1e-8)
        
        # Encode time
        time_enc = self.time_embed(t_norm.unsqueeze(-1))  # [batch, seq, d_model]
        
        return x + time_enc


class InceptionBlock(nn.Module):
    """
    Inception block for multi-scale feature extraction.
    
    Uses parallel convolutions with different kernel sizes to capture
    both short-term and long-term patterns in lightcurves.
    
    Based on InceptionTime architecture which achieves SOTA on UCR archive.
    """
    def __init__(self, in_channels, n_filters=32, bottleneck_size=32):
        super().__init__()
        
        # Bottleneck for dimensionality reduction
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
        
        # Multi-scale convolutions (kernel sizes: 9, 19, 39)
        self.conv_small = nn.Conv1d(bottleneck_size, n_filters, kernel_size=9, padding=4, bias=False)
        self.conv_medium = nn.Conv1d(bottleneck_size, n_filters, kernel_size=19, padding=9, bias=False)
        self.conv_large = nn.Conv1d(bottleneck_size, n_filters, kernel_size=39, padding=19, bias=False)
        
        # MaxPool path for residual-like connection
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_maxpool = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        
        # Batch norm for concatenated output (4 paths * n_filters)
        self.bn = nn.BatchNorm1d(n_filters * 4)
        
    def forward(self, x):
        # Bottleneck
        bottleneck = self.bottleneck(x)
        
        # Multi-scale convolutions
        conv_small = self.conv_small(bottleneck)
        conv_medium = self.conv_medium(bottleneck)
        conv_large = self.conv_large(bottleneck)
        
        # MaxPool path
        maxpool = self.maxpool(x)
        maxpool = self.conv_maxpool(maxpool)
        
        # Concatenate all paths
        out = torch.cat([conv_small, conv_medium, conv_large, maxpool], dim=1)
        out = self.bn(out)
        out = F.relu(out)
        
        return out


class InceptionTime(nn.Module):
    """
    InceptionTime: State-of-the-art deep learning for time series classification.
    
    Architecture features:
    - Multiple Inception blocks for multi-scale pattern extraction
    - Residual connections for deep network training
    - Global Average Pooling for sequence aggregation
    
    Achieves "on par with the state-of-art" on UCR archive while being scalable.
    Reference: Ismail Fawaz et al., 2019
    """
    def __init__(self, input_channels=6, n_classes=1, n_filters=32, n_blocks=3, 
                 bottleneck_size=32, embedding_dim=128, dropout=0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Initial Inception block
        self.inception1 = InceptionBlock(input_channels, n_filters, bottleneck_size)
        
        # Additional Inception blocks with residual connections
        self.inception_blocks = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        
        current_channels = n_filters * 4  # 4 paths in inception block
        
        for i in range(n_blocks - 1):
            self.inception_blocks.append(
                InceptionBlock(current_channels, n_filters, bottleneck_size)
            )
            # Residual connection (optional skip)
            if i % 2 == 1:
                self.residual_convs.append(
                    nn.Sequential(
                        nn.Conv1d(current_channels, n_filters * 4, kernel_size=1, bias=False),
                        nn.BatchNorm1d(n_filters * 4)
                    )
                )
            else:
                self.residual_convs.append(None)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(n_filters * 4, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, n_classes)
        
    def forward(self, x):
        """
        x: [batch, channels=6, seq_len=100]
        """
        # Initial inception
        out = self.inception1(x)
        identity = out
        
        # Additional inception blocks with residual
        for i, (inception, residual) in enumerate(zip(self.inception_blocks, self.residual_convs)):
            out = inception(out)
            
            if residual is not None:
                out = out + residual(identity)
                identity = out
        
        # Global average pooling
        out = self.gap(out)  # [batch, channels, 1]
        out = out.squeeze(-1)  # [batch, channels]
        
        # Embedding
        emb = self.embedding_head(out)
        
        # Classification
        logits = self.classifier(emb)
        
        return logits, emb


class TDE_Transformer(nn.Module):
    """
    Tier 3: Transformer-based TDE Classifier (T2 Architecture).
    Ref: "Time-Series Transformer Implementation (HIGH IMPACT)"
    
    Architecture:
    1. Input Embedding: 1D Conv per band (captures local patterns)
    2. Positional Encoding: Temporal ordering
    3. Transformer Encoder: 3-6 layers, 4-8 heads
    4. GAP + Classification
    """
    def __init__(self, input_bands=6, seq_len=100, embedding_dim=128, n_heads=8, n_layers=4, dropout=0.2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input embedding using 1D convolution
        self.input_conv = nn.Conv1d(input_bands, embedding_dim, kernel_size=3, padding=1)
        
        # Position encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, embedding_dim, seq_len) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Global pooling + Head
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        x: [batch, bands=6, seq_len=100]
        """
        # Embedding: [Batch, 6, L] -> [Batch, D, L]
        x = self.input_conv(x)
        
        # Add Positional Encoding
        x = x + self.pos_encoder[:, :, :x.shape[2]]
        
        # Permute for Transformer: [Batch, L, D] (since batch_first=True)
        x = x.permute(0, 2, 1)
        
        # Transformer
        x = self.transformer(x)
        
        # GAP: [Batch, L, D] -> [Batch, D, L] -> [Batch, D, 1]
        x = x.permute(0, 2, 1)
        x = self.gap(x).squeeze(-1)
        
        # Classification
        out = self.classifier(x)
        
        return out, x


class TabularTransformer(nn.Module):
    """
    Transformer branch for Tabular Metadata (ATAT-style).
    
    Instead of a simple MLP, we project features into a sequence of embeddings
    to allow the Transformer to learn complex feature interactions via self-attention.
    
    Ref: "ATAT: processes metadata (redshift, extinction, derived features)"
    """
    def __init__(self, input_dim, d_model=32, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        
        # Project input into sequence of tokens
        self.n_tokens = 8 
        self.tokenizer = nn.Linear(input_dim, self.n_tokens * d_model)
        
        self.pos_encoder = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output aggregation
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        x: [Batch, n_features] (e.g. 300)
        """
        bs = x.shape[0]
        
        # Tokenize: [Batch, n_features] -> [Batch, n_tokens * d_model]
        tokens = self.tokenizer(x)
        
        # Reshape: [Batch, n_tokens, d_model]
        tokens = tokens.view(bs, self.n_tokens, -1)
        
        # Positional Encoding
        tokens = tokens + self.pos_encoder
        
        # Transformer
        out = self.transformer(tokens)
        
        # Aggregate: [Batch, n_tokens, d_model] -> [Batch, d_model]
        out = out.permute(0, 2, 1)
        emb = self.gap(out).squeeze(-1)
        
        return emb


class ATAT_Model(nn.Module):
    """
    ATAT: Astronomical Transformer (Dual Branch).
    Combines Lightcurve Transformer (Tlc) and Tabular Transformer (Ttab).
    
    Ref: "Concatenate outputs from both transformers... Final classification through 2-layer MLP"
    """
    def __init__(self, 
                 lc_input_bands=6, lc_seq_len=100, lc_embed=128,
                 tab_input_dim=300, tab_embed=32,
                 n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        
        # Branch 1: Lightcurve Transformer (Tlc)
        self.tlc = TDE_Transformer(
            input_bands=lc_input_bands,
            seq_len=lc_seq_len,
            embedding_dim=lc_embed,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        # Remove classifier to get embedding directly
        self.tlc.classifier = nn.Identity()
        
        # Branch 2: Tabular Transformer (Ttab)
        self.ttab = TabularTransformer(
            input_dim=tab_input_dim,
            d_model=tab_embed,
            n_heads=4,
            n_layers=3,
            dropout=dropout
        )
        
        # Fusion Head
        fusion_dim = lc_embed + tab_embed
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1)
        )
        
    def forward(self, lc_data, tab_data):
        """
        lc_data: [Batch, 6, 100]
        tab_data: [Batch, n_features]
        """
        # Get embeddings
        _, lc_emb = self.tlc(lc_data)
        tab_emb = self.ttab(tab_data)
        
        # Concatenate
        combined = torch.cat([lc_emb, tab_emb], dim=1)
        
        # Final classification
        logits = self.head(combined)
        
        return logits, combined


class TDE_CNN1D(nn.Module):
    """Legacy 1D CNN - kept for compatibility. Consider using TDE_Transformer instead."""
    def __init__(self, input_channels=6, embedding_dim=64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2 (Dilated)
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3 (Dilated)
            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        self.embedding_head = nn.Sequential(
            nn.Linear(1536, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.features(x)
        emb = self.embedding_head(x)
        out = self.classifier(emb)
        return out, emb


# ============================================================
# CONTRASTIVE & METRIC LEARNING - NaN FIXES
# ============================================================

class SimCLR(nn.Module):
    """
    SimCLR Framework for Self-Supervised Learning.
    Encoder: Base Transformer (Tlc or ATAT)
    Projector: MLP to embedding space
    """
    def __init__(self, encoder, input_dim=128, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )
        
    def forward(self, x):
        # x is a batch of lightcurves OR tuple (lc, tab) for ATAT
        if isinstance(x, (list, tuple)):
            # ATAT expects (lc, tab) as separate arguments
            lc, tab = x
            _, h = self.encoder(lc, tab)
        else:
            _, h = self.encoder(x)
        z = self.projector(h)
        return h, z


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss - NaN FIXED VERSION.
    
    FIX 1: Increased default temperature from 0.1 to 0.2 for numerical stability
    FIX 2: Added gradient clipping in normalization
    FIX 3: Better handling of edge cases
    """
    def __init__(self, temperature=0.2, base_temperature=0.07):  # CHANGED: temperature 0.1 -> 0.2
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        """
        features: [Batch, Dim] (normalized)
        labels: [Batch]
        """
        device = features.device
        batch_size = features.shape[0]
        
        # FIX: Check for NaN in inputs
        if torch.isnan(features).any():
            print("WARNING: NaN detected in features input to SupConLoss")
            features = torch.nan_to_num(features, nan=0.0)
        
        # Labels reshape
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num labels must match num features')
            
        # Mask of same-class samples (positives)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # FIX: L2 normalize with explicit eps to prevent division by zero
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # Compute logits (cosine similarity)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # FIX: Clip logits to prevent overflow
        anchor_dot_contrast = torch.clamp(anchor_dot_contrast, min=-50, max=50)
        
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        
        # FIX: Check if mask has any positives
        mask_sum = mask.sum(1)
        if (mask_sum == 0).any():
            # Some samples have no positives - add small epsilon
            mask_sum = torch.clamp(mask_sum, min=1e-7)
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)
        
        # Mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-7)
        
        # FIX: Check for NaN before loss computation
        if torch.isnan(mean_log_prob_pos).any():
            print("WARNING: NaN in mean_log_prob_pos, replacing with zeros")
            mean_log_prob_pos = torch.nan_to_num(mean_log_prob_pos, nan=0.0)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        
        # FIX: Final NaN check
        if torch.isnan(loss):
            print("WARNING: NaN loss detected, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


class ImbalancedArcFaceLoss(nn.Module):
    """
    ArcFace with Class-Specific Margins for Rare Classes.
    Margin TDE >> Margin Non-TDE.
    """
    def __init__(self, in_features, out_features, s=30.0, margin_tde=0.7, margin_non_tde=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.margins = {1: margin_tde, 0: margin_non_tde}
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # input: [Batch, features]
        # label: [Batch] (long)
        
        # Ensure weight is on same device as input
        weight = self.weight.to(input.device)
        
        # Normalize weights and features
        cosine = F.linear(F.normalize(input), F.normalize(weight))
        
        # Get dynamic margins based on label
        m_vec = torch.tensor([self.margins[int(y)] for y in label.cpu().numpy()], device=input.device)
        
        # One-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Add angular margin: theta + m
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + m_vec.view(-1, 1))
        
        # Scale
        output = one_hot * target_logits + (1.0 - one_hot) * cosine
        output *= self.s
        
        return output


# ============================================================
# BALANCED BATCH SAMPLER
# ============================================================

class BalancedBatchSampler:
    """
    Ensures each batch has a minimum number of positive samples.
    Critical for extreme class imbalance (4.8% positive).
    """
    def __init__(self, labels, batch_size, min_positives=4):
        self.labels = labels
        self.batch_size = batch_size
        self.min_positives = min_positives
        
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]
        
        self.n_batches = len(labels) // batch_size
        
    def __iter__(self):
        for _ in range(self.n_batches):
            # Sample positives
            pos_sample = np.random.choice(
                self.pos_indices, 
                size=min(self.min_positives, len(self.pos_indices)),
                replace=False
            )
            
            # Sample negatives to fill batch
            n_neg = self.batch_size - len(pos_sample)
            neg_sample = np.random.choice(
                self.neg_indices,
                size=n_neg,
                replace=False
            )
            
            # Combine and shuffle
            batch = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(batch)
            
            yield batch.tolist()
    
    def __len__(self):
        return self.n_batches


# ============================================================
# TRAINING FUNCTIONS - PERFORMANCE OPTIMIZED
# ============================================================

def train_transformer_fold(
    train_ids, val_ids, y_train, y_val, 
    processed_splits, 
    X_train_tab=None, X_val_tab=None,
    n_epochs=30,  # INCREASED from 15
    batch_size=64,  # DECREASED from 128 for stability
    device='cuda', 
    use_atat=True,
    pretrained_weights=None, 
    use_arcface=False, 
    num_workers=2,
    # Architecture Params
    n_heads=4,
    n_layers=3,
    embedding_dim=128,
    dropout=0.3,
    patience=8  # Default patience
):
    """
    FIXED VERSION with:
    1. Gradient clipping (CRITICAL)
    2. Better learning rate schedule
    3. Class balancing via sampler
    4. Early stopping with patience
    5. Warmup phase
    6. Focal Loss instead of BCE
    """
    # Imports removed: classes are defined in this file
    
    # ============================================================
    # FIX 1: CREATE BALANCED SAMPLER
    # ============================================================
    tab_map_train = None
    tab_map_val = None
    
    if use_atat and X_train_tab is not None:
        tab_map_train = {oid: row for oid, row in zip(train_ids, X_train_tab)}
        tab_map_val = {oid: row for oid, row in zip(val_ids, X_val_tab)}
        
    train_ds = LightCurveDataset(
        processed_splits, train_ids, {}, y_train, 
        tabular_features=tab_map_train,
        use_gp=False
    )
    val_ds = LightCurveDataset(
        processed_splits, val_ids, {}, y_val, 
        tabular_features=tab_map_val,
        use_gp=False
    )
    
    # CRITICAL: Use BalancedBatchSampler for extreme imbalance
    balanced_sampler = BalancedBatchSampler(
        labels=y_train, 
        batch_size=batch_size, 
        min_positives=max(4, int(batch_size * 0.1))  # At least 10% positive per batch
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_sampler=balanced_sampler,  # Use balanced sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # ============================================================
    # FIX 2: MODEL WITH BETTER INITIALIZATION
    # ============================================================
    embedding_dim = 128
    if use_atat and X_train_tab is not None:
        model = ATAT_Model(
            tab_input_dim=X_train_tab.shape[1],
            tab_embed=32,
            dropout=0.3  # INCREASED dropout
        ).to(device)
    else:
        model = TDE_Transformer(
            input_bands=6, 
            seq_len=100, 
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers, 
            dropout=dropout
        ).to(device)
    
    # Load Pre-trained Weights if available
    if pretrained_weights is not None:
        print("  Loading SimCLR weights...", end="")
        missing, unexpected = model.load_state_dict(pretrained_weights, strict=False)
        print(f" Done. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        base_lr = 1e-4  # LOWER for fine-tuning
    else:
        base_lr = 5e-4  # LOWER than original 1e-3
    
    # ============================================================
    # FIX 3: FOCAL LOSS + BETTER OPTIMIZER
    # ============================================================
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr, 
        weight_decay=1e-3,  # INCREASED regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Use Focal Loss for extreme imbalance
    criterion = FocalLoss(
        gamma=2.0,  # Focus on hard examples
        alpha=0.85,  # Weight positive class more (since 4.8% positive)
        label_smoothing=0.05,  # Small smoothing for regularization
        reduction='mean'
    ).to(device)
    
    # ============================================================
    # FIX 4: BETTER LEARNING RATE SCHEDULE WITH WARMUP
    # ============================================================
    warmup_epochs = 3
    total_steps = len(train_loader) * n_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ============================================================
    # FIX 5: AUGMENTATION
    # ============================================================
    # Import removed: LightCurveAugmenter is defined in this file
    augmenter = LightCurveAugmenter(
        noise_std=0.03,  # Reduced noise
        time_jitter=0.01,
        flux_scale_range=(0.95, 1.05),  # Narrower range
        p=0.5
    )
    
    # ============================================================
    # FIX 6: GRADIENT SCALER + ACCUMULATION
    # ============================================================
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    use_amp = scaler is not None
    
    # Gradient accumulation for effective larger batch size
    accumulation_steps = 2  # Effective batch size = 64 * 2 = 128
    
    # ============================================================
    # FIX 7: EARLY STOPPING
    # ============================================================
    best_f1 = 0
    best_probs = np.zeros(len(y_val))
    best_emb = np.zeros((len(y_val), embedding_dim + (32 if use_atat else 0)))
    best_emb = np.zeros((len(y_val), embedding_dim + (32 if use_atat else 0)))
    # patience = patience (passed as arg)
    patience_counter = 0
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if use_atat:
                x_tab, y = batch
                if isinstance(x_tab, (list, tuple)):
                    x, tab = x_tab
                else:
                    x, tab, y = batch[0], batch[1], batch[-1]
                x = augmenter(x)
                x = torch.nan_to_num(x, nan=0.0)
                tab = torch.nan_to_num(tab, nan=0.0)
                x, tab, y = x.to(device, non_blocking=True), tab.to(device, non_blocking=True), y.to(device, non_blocking=True).float().unsqueeze(1)
            else:
                x, y = batch
                x = augmenter(x)
                x = torch.nan_to_num(x, nan=0.0)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float().unsqueeze(1)
            
            # Forward pass with AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    if use_atat:
                        logits, emb = model(x, tab)
                    else:
                        logits, emb = model(x)
                    
                    loss = criterion(logits, y)
                    loss = loss / accumulation_steps  # Normalize for accumulation
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # CRITICAL: Gradient clipping BEFORE optimizer step
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            else:
                if use_atat:
                    logits, emb = model(x, tab)
                else:
                    logits, emb = model(x)
                
                loss = criterion(logits, y)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            
            epoch_loss += loss.item() * accumulation_steps
            n_batches += 1
            
            # Get current LR
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = epoch_loss / n_batches
        
        # ============================================================
        # VALIDATION
        # ============================================================
        model.eval()
        val_probs = []
        val_targets = []
        val_embs = []
        
        with torch.no_grad():
            for batch in val_loader:
                if use_atat:
                    x_tab, y = batch
                    if isinstance(x_tab, (list, tuple)):
                        x, tab = x_tab
                    else:
                        x, tab, y = batch[0], batch[1], batch[-1]
                    x = torch.nan_to_num(x, nan=0.0)
                    tab = torch.nan_to_num(tab, nan=0.0)
                    x, tab = x.to(device, non_blocking=True), tab.to(device, non_blocking=True)
                    logits, emb = model(x, tab)
                else:
                    x, y = batch
                    x = torch.nan_to_num(x, nan=0.0)
                    x = x.to(device, non_blocking=True)
                    logits, emb = model(x)
                
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy().flatten())
                val_targets.extend(y.cpu().numpy())
                val_embs.extend(emb.cpu().numpy())
        
        # Calculate F1
        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)
        
        best_thresh = 0.5
        curr_f1 = 0
        for t in np.linspace(0.1, 0.9, 17):  # More threshold candidates
            sc = f1_score(val_targets, (val_probs >= t).astype(int), zero_division=0)
            if sc > curr_f1:
                curr_f1 = sc
                best_thresh = t
        
        # Print detailed metrics
        n_pred_pos = (val_probs >= best_thresh).sum()
        n_true_pos = val_targets.sum()
        print(f"  Epoch {epoch+1} - Loss: {avg_loss:.4f} | Val F1: {curr_f1:.4f} (t={best_thresh:.2f}) | "
              f"Pred: {n_pred_pos} | True: {n_true_pos} | Best: {best_f1:.4f}")
        
        # Early stopping check
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_probs = val_probs
            best_emb = np.array(val_embs)
            patience_counter = 0
            print(f"  ✓ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    return best_probs, best_emb, best_f1


# ============================================================
# ADDITIONAL DIAGNOSTIC FUNCTION
# ============================================================

def diagnose_transformer_training(model, train_loader, device, use_atat=False):
    """
    Diagnostic tool to check for common issues:
    - NaN in outputs
    - Dead ReLUs
    - Gradient magnitudes
    - Prediction distribution
    """
    model.eval()
    
    print("\n" + "="*60)
    print("TRANSFORMER DIAGNOSTIC CHECK")
    print("="*60)
    
    all_probs = []
    all_grads = []
    
    # Check first batch
    batch = next(iter(train_loader))
    
    if use_atat:
        x_tab, y = batch
        if isinstance(x_tab, (list, tuple)):
            x, tab = x_tab
        else:
            x, tab, y = batch[0], batch[1], batch[-1]
        x = x.to(device)
        tab = tab.to(device)
        y = y.to(device).float().unsqueeze(1)
        
        model.train()
        logits, emb = model(x, tab)
    else:
        x, y = batch
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)
        
        model.train()
        logits, emb = model(x)
    
    # Check 1: NaN in outputs
    print(f"\n1. NaN Check:")
    print(f"   Logits NaN: {torch.isnan(logits).any().item()}")
    print(f"   Embeddings NaN: {torch.isnan(emb).any().item()}")
    
    # Check 2: Output distribution
    probs = torch.sigmoid(logits)
    print(f"\n2. Prediction Distribution:")
    print(f"   Mean prob: {probs.mean().item():.4f}")
    print(f"   Std prob: {probs.std().item():.4f}")
    print(f"   Min prob: {probs.min().item():.4f}")
    print(f"   Max prob: {probs.max().item():.4f}")
    print(f"   % > 0.5: {(probs > 0.5).float().mean().item()*100:.1f}%")
    
    # Check 3: Gradient flow
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    
    print(f"\n3. Gradient Magnitudes:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 10 or grad_norm < 1e-6:
                print(f"   {name}: {grad_norm:.2e} ⚠️")
    
    # Check 4: Embedding statistics
    print(f"\n4. Embedding Statistics:")
    print(f"   Mean: {emb.mean().item():.4f}")
    print(f"   Std: {emb.std().item():.4f}")
    print(f"   L2 Norm: {emb.norm(dim=1).mean().item():.4f}")
    
    print("="*60 + "\n")


# ============================================================
# HYPERPARAMETER RECOMMENDATIONS
# ============================================================

RECOMMENDED_HYPERPARAMS = {
    "standard": {
        "n_epochs": 30,
        "batch_size": 64,
        "base_lr": 5e-4,
        "weight_decay": 1e-3,
        "dropout": 0.3,
        "n_layers": 3,
        "focal_gamma": 2.0,
        "focal_alpha": 0.85,
        "warmup_epochs": 3,
        "patience": 8
    },
    
    "aggressive": {
        "n_epochs": 50,
        "batch_size": 32,  # Smaller for more gradient updates
        "base_lr": 3e-4,
        "weight_decay": 5e-3,  # Strong regularization
        "dropout": 0.4,
        "n_layers": 2,  # Simpler model
        "focal_gamma": 3.0,  # More focus on hard examples
        "focal_alpha": 0.9,
        "warmup_epochs": 5,
        "patience": 12
    },
    
    "fast_debug": {
        "n_epochs": 10,
        "batch_size": 128,
        "base_lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "n_layers": 2,
        "focal_gamma": 2.0,
        "focal_alpha": 0.75,
        "warmup_epochs": 1,
        "patience": 5
    }
}


def pretrain_supcon(
    train_log, splits, 
    X_tab=None,
    n_epochs=50, batch_size=512, device='cuda', use_atat=True, num_workers=4
):
    """
    Supervised Contrastive Pre-training (SupCon) - NaN FIXED VERSION.
    
    FIX 1: Increased temperature in SupConLoss
    FIX 2: Added gradient clipping and NaN checks
    FIX 3: Better normalization handling
    """
    print(f"\n🔄 Starting SupCon Pre-training ({n_epochs} epochs)...")
    
    # Dataset (Labeled)
    processed_splits = {}
    for k, v in splits.items():
        processed_splits.update({grp_name: grp_df for grp_name, grp_df in v.groupby('object_id')})
        
    tab_map = None
    if use_atat and X_tab is not None:
        tab_map = {oid: row for oid, row in zip(train_log['object_id'].values, X_tab)}
    
    # Custom Dataset returns (view1, view2), label
    class SupConDataset(LightCurveDataset):
        def __init__(self, *args, **kwargs):
            # Disable GP for pre-training speed
            kwargs['use_gp'] = False 
            super().__init__(*args, **kwargs)
            self.augmenter = ContrastiveAugmenter(seq_len=self.seq_len)
            
        def __getitem__(self, idx):
            parent_out = super().__getitem__(idx)
            
            if self.labels is not None:
                if isinstance(parent_out, tuple):
                    data, label = parent_out
                    if isinstance(data, tuple):  # ATAT (lc, tab)
                        lc, tab = data
                        x_i, x_j = self.augmenter(lc)
                        return ((x_i, tab), (x_j, tab)), label
                    else:  # Standard LC
                        x_i, x_j = self.augmenter(data)
                        return (x_i, x_j), label
                else:
                    # Just lightcurve, no label
                    x_i, x_j = self.augmenter(parent_out)
                    return (x_i, x_j), torch.tensor(0.0)
            else:
                x_i, x_j = self.augmenter(parent_out)
                return (x_i, x_j), torch.tensor(0.0)

    ds = SupConDataset(
        processed_splits, 
        train_log['object_id'].values, 
        {},
        labels=train_log['target'].values,
        tabular_features=tab_map
    )
    
    # PERFORMANCE FIX: Multi-worker loading
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Model Setup
    if use_atat and X_tab is not None:
        base_encoder = ATAT_Model(
            tab_input_dim=X_tab.shape[1],
            tab_embed=32
        )
    else:
        base_encoder = TDE_Transformer(
            input_bands=6, seq_len=100, embedding_dim=128
        )
        
    model = SimCLR(base_encoder, input_dim=128+32 if use_atat else 128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # FIX: Use updated SupConLoss with higher temperature
    criterion = SupConLoss(temperature=0.2)  # Increased from 0.1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # PERFORMANCE FIX: Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    use_amp = scaler is not None
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        nan_count = 0
        
        pbar = tqdm(loader, desc=f"SupCon Epoch {epoch+1}/{n_epochs}")
        
        for batch in pbar:
            # batch: (v1, v2), label
            views, labels = batch
            v1, v2 = views
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # FIX: NaN check in labels
            if torch.isnan(labels).any():
                print("WARNING: NaN in labels, skipping batch")
                continue
            
            # Forward with AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    if isinstance(v1, (list, tuple)) and len(v1) == 2:
                        # ATAT case
                        lc1, tab1 = v1
                        lc2, tab2 = v2
                        lc1 = torch.nan_to_num(lc1, nan=0.0)
                        tab1 = torch.nan_to_num(tab1, nan=0.0)
                        lc2 = torch.nan_to_num(lc2, nan=0.0)
                        tab2 = torch.nan_to_num(tab2, nan=0.0)
                        lc1, tab1 = lc1.to(device, non_blocking=True), tab1.to(device, non_blocking=True)
                        lc2, tab2 = lc2.to(device, non_blocking=True), tab2.to(device, non_blocking=True)
                        
                        _, z_i = model((lc1, tab1))
                        _, z_j = model((lc2, tab2))
                    else:
                        v1 = torch.nan_to_num(v1, nan=0.0)
                        v2 = torch.nan_to_num(v2, nan=0.0)
                        v1 = v1.to(device, non_blocking=True)
                        v2 = v2.to(device, non_blocking=True)
                        _, z_i = model(v1)
                        _, z_j = model(v2)
                    
                    # FIX: Check for NaN in embeddings
                    if torch.isnan(z_i).any() or torch.isnan(z_j).any():
                        nan_count += 1
                        continue
                    
                    # Stack features: [2*Batch, Dim]
                    features = torch.cat([z_i, z_j], dim=0)
                    targets = torch.cat([labels, labels], dim=0)
                    
                    loss = criterion(features, targets)
                
                # FIX: Check for NaN loss
                if torch.isnan(loss):
                    nan_count += 1
                    print(f"NaN loss detected, skipping batch (NaN count: {nan_count})")
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without AMP
                if isinstance(v1, (list, tuple)) and len(v1) == 2:
                    # ATAT case
                    lc1, tab1 = v1
                    lc2, tab2 = v2
                    lc1 = torch.nan_to_num(lc1, nan=0.0)
                    tab1 = torch.nan_to_num(tab1, nan=0.0)
                    lc2 = torch.nan_to_num(lc2, nan=0.0)
                    tab2 = torch.nan_to_num(tab2, nan=0.0)
                    lc1, tab1 = lc1.to(device, non_blocking=True), tab1.to(device, non_blocking=True)
                    lc2, tab2 = lc2.to(device, non_blocking=True), tab2.to(device, non_blocking=True)
                    
                    _, z_i = model((lc1, tab1))
                    _, z_j = model((lc2, tab2))
                else:
                    v1 = torch.nan_to_num(v1, nan=0.0)
                    v2 = torch.nan_to_num(v2, nan=0.0)
                    v1 = v1.to(device, non_blocking=True)
                    v2 = v2.to(device, non_blocking=True)
                    _, z_i = model(v1)
                    _, z_j = model(v2)
                
                # FIX: Check for NaN in embeddings
                if torch.isnan(z_i).any() or torch.isnan(z_j).any():
                    nan_count += 1
                    continue
                
                # Stack features: [2*Batch, Dim]
                features = torch.cat([z_i, z_j], dim=0)
                targets = torch.cat([labels, labels], dim=0)
                
                loss = criterion(features, targets)
                
                # FIX: Check for NaN loss
                if torch.isnan(loss):
                    nan_count += 1
                    print(f"NaN loss detected, skipping batch (NaN count: {nan_count})")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'nan_batches': nan_count})
            
        scheduler.step()
        avg_loss = total_loss / max(1, len(loader) - nan_count)
        print(f" Epoch {epoch+1}: SupCon Loss = {avg_loss:.4f} (NaN batches: {nan_count})")
            
    print("✅ SupCon Pre-training Complete.")
    return base_encoder.state_dict()


def train_cnn_model(train_log, splits, n_epochs=20, batch_size=64, device='cuda', 
                      use_transformer=True, test_log=None, test_splits=None,
                      use_atat=True, X_train_tab=None, X_test_tab=None, num_workers=4):
    """
    TIER 5 INTEGRATED: Full lightcurve feature extraction with all improvements.
    PERFORMANCE OPTIMIZED VERSION.
    """
    print("🧠 Pre-processing for Lightcurve Feature Extraction...")
    
    # Pre-group splits for speed
    processed_splits = {}
    print("  Indexing lightcurves by object ID (this may take a moment)...")
    for k, v in splits.items():
        processed_splits.update({grp_name: grp_df for grp_name, grp_df in v.groupby('object_id')})
    
    # Create augmenter
    augmenter = LightCurveAugmenter(
        noise_std=0.05,
        time_jitter=0.02,
        flux_scale_range=(0.9, 1.1),
        p=0.5
    )
    
    # Prepare Tabular Data Map if ATAT
    tab_map_train = None
    if use_atat and X_train_tab is not None:
        print("  Mapping tabular features for ATAT training...")
        tab_map_train = {oid: row for oid, row in zip(train_log['object_id'].values, X_train_tab)}
    
    dataset = LightCurveDataset(
        processed_splits, 
        train_log['object_id'].values, 
        {}, 
        train_log['target'].values,
        tabular_features=tab_map_train,
        use_gp=False  # PERFORMANCE: Disabled GP
    )
    
    # Prepare Test Loader if provided
    test_loader = None
    test_preds = None
    if test_log is not None and test_splits is not None:
        print("  Indexing TEST lightcurves...")
        processed_test = {} 
        for k, v in test_splits.items():
            processed_test.update({grp_name: grp_df for grp_name, grp_df in v.groupby('object_id')})
        
        tab_map_test = None
        if use_atat and X_test_tab is not None:
            tab_map_test = {oid: row for oid, row in zip(test_log['object_id'].values, X_test_tab)}
        
        test_dataset = LightCurveDataset(
            processed_test, 
            test_log['object_id'].values, 
            {},
            tabular_features=tab_map_test,
            use_gp=False  # PERFORMANCE: Disabled GP
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_preds = np.zeros(len(test_log))
    
    # 5-Fold Stratified
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    embedding_dim = 128 if use_transformer else 64
    oof_preds = np.zeros(len(train_log))
    embeddings = np.zeros((len(train_log), embedding_dim))
    
    # PERFORMANCE FIX: Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    use_amp = scaler is not None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_log, train_log['target'])):
        model_name = "Transformer" if use_transformer else "CNN"
        print(f"\n═══ {model_name} Fold {fold+1}/5 ═══")
        
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        
        # PERFORMANCE FIX: Multi-worker data loading
        train_loader = DataLoader(
            train_sub, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_sub, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Use Transformer or legacy CNN
        if use_transformer:
            if use_atat and X_train_tab is not None:
                model = ATAT_Model(
                    lc_input_bands=6,
                    lc_seq_len=100,
                    lc_embed=128,
                    tab_input_dim=X_train_tab.shape[1],
                    tab_embed=32,
                    n_heads=4,
                    n_layers=3,
                    dropout=0.2
                ).to(device)
            else:
                model = TDE_Transformer(
                    input_bands=6, 
                    seq_len=100,
                    embedding_dim=embedding_dim,
                    n_heads=4, 
                    n_layers=3, 
                    dropout=0.2
                ).to(device)
        else:
            model = TDE_CNN1D(embedding_dim=embedding_dim).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # FIX: Calculate class weights properly
        fold_y_train = train_log['target'].iloc[train_idx].values
        n_pos = np.sum(fold_y_train == 1)
        n_neg = len(train_idx) - n_pos
        pos_weight = n_neg / max(n_pos, 1)  # Prevent division by zero
        
        # Use stable BCEWithLogitsLoss instead of Focal Loss
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=1e-3,
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        best_f1 = 0
        best_model = None
        
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            
            for batch in pbar:
                # Handle batch structure
                if use_atat and X_train_tab is not None:
                    try:
                        if len(batch) == 2:
                            x_data, y = batch
                            if isinstance(x_data, (list, tuple)) and len(x_data) == 2:
                                x, tab = x_data
                            else:
                                x, y = batch
                                tab = None
                        else:
                            x, y = batch
                            tab = None
                        
                        if tab is not None:
                            x = augmenter(x)
                            x = torch.nan_to_num(x, nan=0.0)
                            tab = torch.nan_to_num(tab, nan=0.0)
                            x, tab, y = x.to(device, non_blocking=True), tab.to(device, non_blocking=True), y.to(device, non_blocking=True).float().unsqueeze(1)
                        else:
                            x = augmenter(x)
                            x = torch.nan_to_num(x, nan=0.0)
                            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float().unsqueeze(1)
                    except Exception as e:
                        x, y = batch
                        x = augmenter(x)
                        x = torch.nan_to_num(x, nan=0.0)
                        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float().unsqueeze(1)
                        tab = None
                else:
                    x, y = batch
                    x = augmenter(x)
                    x = torch.nan_to_num(x, nan=0.0)
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float().unsqueeze(1)
                    tab = None
                
                optimizer.zero_grad(set_to_none=True)
                
                # PERFORMANCE FIX: Mixed Precision Training
                if use_amp:
                    with torch.cuda.amp.autocast():
                        if tab is not None:
                            logits, _ = model(x, tab)
                        else:
                            logits, _ = model(x)
                        loss = criterion(logits, y)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if tab is not None:
                        logits, _ = model(x, tab)
                    else:
                        logits, _ = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            # Validation
            model.eval()
            val_probs = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Handle batch structure similarly
                    if use_atat and X_train_tab is not None:
                        try:
                            if len(batch) == 2:
                                x_data, y = batch
                                if isinstance(x_data, (list, tuple)) and len(x_data) == 2:
                                    x, tab = x_data
                                else:
                                    x, y = batch
                                    tab = None
                            else:
                                x, y = batch
                                tab = None
                            
                            if tab is not None:
                                x = torch.nan_to_num(x, nan=0.0)
                                tab = torch.nan_to_num(tab, nan=0.0)
                                x, tab = x.to(device, non_blocking=True), tab.to(device, non_blocking=True)
                                logits, _ = model(x, tab)
                            else:
                                x = torch.nan_to_num(x, nan=0.0)
                                x = x.to(device, non_blocking=True)
                                logits, _ = model(x)
                        except Exception:
                            x, y = batch
                            x = torch.nan_to_num(x, nan=0.0)
                            x = x.to(device, non_blocking=True)
                            logits, _ = model(x)
                    else:
                        x, y = batch
                        x = torch.nan_to_num(x, nan=0.0)
                        x = x.to(device, non_blocking=True)
                        logits, _ = model(x)
                        
                    probs = torch.sigmoid(logits) 
                    val_probs.extend(probs.cpu().numpy())
                    val_targets.extend(y.cpu().numpy())
            
            val_probs = np.array(val_probs).flatten()
            val_targets = np.array(val_targets).flatten()
            
            # Calculate F1
            f1 = 0
            for t in np.linspace(0.1, 0.9, 9):
                sc = f1_score(val_targets, (val_probs >= t).astype(int), zero_division=0)
                if sc > f1: 
                    f1 = sc
            
            if f1 >= best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model.state_dict())
            
            print(f"  Epoch {epoch+1} - Val F1: {f1:.4f} (Best: {best_f1:.4f})")
                
        # Load best and extract embeddings
        if best_model is not None:
            model.load_state_dict(best_model)
        model.eval()
        
        # OOF Prediction
        with torch.no_grad():
            fold_probs = []
            fold_embs = []
            
            for batch in val_loader:
                if use_atat and X_train_tab is not None:
                    try:
                        if len(batch) == 2:
                            x_data, y = batch
                            if isinstance(x_data, (list, tuple)) and len(x_data) == 2:
                                x, tab = x_data
                                x = torch.nan_to_num(x, nan=0.0)
                                tab = torch.nan_to_num(tab, nan=0.0)
                                x, tab = x.to(device, non_blocking=True), tab.to(device, non_blocking=True)
                                logits, emb = model(x, tab)
                            else:
                                x, y = batch
                                x = torch.nan_to_num(x, nan=0.0)
                                x = x.to(device, non_blocking=True)
                                logits, emb = model(x)
                        else:
                            x, y = batch
                            x = torch.nan_to_num(x, nan=0.0)
                            x = x.to(device, non_blocking=True)
                            logits, emb = model(x)
                    except Exception:
                        x, y = batch
                        x = torch.nan_to_num(x, nan=0.0)
                        x = x.to(device, non_blocking=True)
                        logits, emb = model(x)
                else:
                    x, y = batch
                    x = torch.nan_to_num(x, nan=0.0)
                    x = x.to(device, non_blocking=True)
                    logits, emb = model(x)
                
                probs = torch.sigmoid(logits)
                fold_probs.extend(probs.cpu().numpy().flatten())
                fold_embs.extend(emb.cpu().numpy())
        
        oof_preds[val_idx] = np.array(fold_probs)
        embeddings[val_idx] = np.array(fold_embs)
        print(f"  ✓ Fold {fold+1} Best Val F1: {best_f1:.4f}")

        # Test Inference
        if test_loader is not None:
            model.eval()
            fold_test_probs = []
            
            with torch.no_grad():
                for batch in test_loader:
                    if use_atat and X_test_tab is not None:
                        try:
                            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                                x, tab = batch
                                x = torch.nan_to_num(x, nan=0.0)
                                tab = torch.nan_to_num(tab, nan=0.0)
                                x, tab = x.to(device, non_blocking=True), tab.to(device, non_blocking=True)
                                logits, _ = model(x, tab)
                            else:
                                x = batch if not isinstance(batch, (list, tuple)) else batch[0]
                                x = torch.nan_to_num(x, nan=0.0)
                                x = x.to(device, non_blocking=True)
                                logits, _ = model(x)
                        except Exception:
                            x = batch if not isinstance(batch, (list, tuple)) else batch[0]
                            x = torch.nan_to_num(x, nan=0.0)
                            x = x.to(device, non_blocking=True)
                            logits, _ = model(x)
                    else:
                        x = batch if not isinstance(batch, (list, tuple)) else batch[0]
                        x = torch.nan_to_num(x, nan=0.0)
                        x = x.to(device, non_blocking=True)
                        logits, _ = model(x)
                        
                    probs = torch.sigmoid(logits)
                    fold_test_probs.extend(probs.cpu().numpy().flatten())
            
            # Aggregate predictions (Bagging)
            test_preds += np.array(fold_test_probs) / 5.0

    return embeddings, oof_preds, test_preds


def extract_features_cnn(train_log, test_log, train_splits, test_splits, num_workers=4):
    """
    Extract CNN features for train and test sets - PERFORMANCE OPTIMIZED.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training CNN Feature Extractor on {device}...")
    
    # Train with cross-validation
    train_emb, train_oof, _ = train_cnn_model(
        train_log, train_splits, device=device, 
        use_transformer=False, n_epochs=20, num_workers=num_workers
    )
    
    print("\nTraining CNN on full dataset for Test inference...")
    
    # Process Train Splits
    processed_train = {}
    for k, v in train_splits.items():
        processed_train.update({grp_name: grp_df for grp_name, grp_df in v.groupby('object_id')})
        
    full_dataset = LightCurveDataset(
        processed_train, 
        train_log['object_id'].values, 
        {}, 
        train_log['target'].values,
        use_gp=False  # PERFORMANCE
    )
    full_loader = DataLoader(
        full_dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Train full model
    embedding_dim = 64
    model = TDE_CNN1D(embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # FIX: Calculate pos_weight properly
    n_pos = np.sum(train_log['target'].values == 1)
    n_neg = len(train_log) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    for epoch in range(12):
        model.train()
        for x, y in full_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            
    # Infer Test
    processed_test = {}
    print("  Indexing TEST lightcurves...")
    for k, v in test_splits.items():
        processed_test.update({grp_name: grp_df for grp_name, grp_df in v.groupby('object_id')})
        
    test_dataset = LightCurveDataset(
        processed_test, 
        test_log['object_id'].values, 
        {},
        use_gp=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    model.eval()
    test_embs = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device, non_blocking=True)
            _, emb = model(x)
            test_embs.extend(emb.cpu().numpy())
            
    test_emb = np.array(test_embs)
    
    # Create DataFrames
    embedding_dim = train_emb.shape[1] if len(train_emb) > 0 else 64
    cols = [f'cnn_emb_{i}' for i in range(embedding_dim)]
    
    df_train = pd.DataFrame(train_emb, columns=cols)
    df_train['object_id'] = train_log['object_id'].values
    
    df_test = pd.DataFrame(test_emb, columns=cols)
    df_test['object_id'] = test_log['object_id'].values
    
    return df_train, df_test


def evaluate_with_metrics(model, val_loader, device, use_atat=False):
    """Enhanced validation with per-class metrics"""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            if use_atat:
                x_tab, y = batch
                if isinstance(x_tab, (list, tuple)):
                    x, tab = x_tab
                else:
                    x, tab, y = batch[0], batch[1], batch[-1]
                x = torch.nan_to_num(x, nan=0.0)
                tab = torch.nan_to_num(tab, nan=0.0)
                x, tab = x.to(device, non_blocking=True), tab.to(device, non_blocking=True)
                logits, _ = model(x, tab)
            else:
                x, y = batch
                x = torch.nan_to_num(x, nan=0.0)
                x = x.to(device, non_blocking=True)
                logits, _ = model(x)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    best_preds = None
    
    for thresh in np.linspace(0.1, 0.9, 17):
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_preds = preds
    
    # Calculate per-class accuracy
    pos_mask = all_targets == 1
    neg_mask = all_targets == 0
    
    pos_acc = (best_preds[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0
    neg_acc = (best_preds[neg_mask] == 0).mean() if neg_mask.sum() > 0 else 0
    
    print(f"    Thresh={best_thresh:.3f} | F1={best_f1:.4f} | "
          f"PosAcc={pos_acc:.3f} | NegAcc={neg_acc:.3f} | "
          f"AvgProb={all_probs.mean():.3f}")
    
    return best_f1, all_probs