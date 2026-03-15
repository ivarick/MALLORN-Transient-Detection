"""
Neural network architectures and components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from mallorn.config import cfg

# ============================================================
# ARCHITECTURE BLOCKS
# ============================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block with improved initialization."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
  
    def forward(self, x):
        return x * self.fc(x)

class ResidualBlock(nn.Module):
    """
    Enhanced Pre-activation Residual block with GLU gating and SE attention.
    Pre-activation (Norm -> Act -> Linear) improves gradient flow.
    """
    def __init__(self, in_dim, out_dim, dropout=0.3, use_se=True):
        super().__init__()
        
        # Gated Linear Unit (GLU) influence: splitting the output for gating
        self.linear1 = nn.Linear(in_dim, out_dim * 2) 
        self.bn1 = nn.BatchNorm1d(in_dim)
        
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
      
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
      
        self.projection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.se = SEBlock(out_dim) if use_se else nn.Identity()
      
    def forward(self, x):
        identity = self.projection(x)
      
        # Pre-activation bottleneck
        out = self.bn1(x)
        out = self.linear1(out)
        
        # GLU-style gating: split in half and multiply
        out, gate = out.chunk(2, dim=-1)
        out = out * torch.sigmoid(gate)
        
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.linear2(self.activation(out))
        out = self.se(out)
      
        return out + identity

# ============================================================
# MAIN MODELS
# ============================================================

class TDEClassifierNet(nn.Module):
    """
    Simplified architecture optimized for TDE classification.
    
    Architecture: BatchNorm -> Encoder(512) -> 3 ResBlocks -> Attention -> Decoder(128->1)
    
    Key improvements:
    - Simpler 3-block design (prevents overfitting on small dataset)
    - Single encoder bottleneck (512) for feature compression
    - Self-attention for cross-feature interactions
    - Focused decoder for classification
    - Increased Dropout (0.4) for regularization
    """
    def __init__(self, input_dim, encoder_dim=512, hidden_dim=256, num_blocks=3, dropout=0.4):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Encoder: compress features to encoder_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks (encoder_dim -> hidden_dim -> hidden_dim)
        self.res_blocks = nn.ModuleList()
        dims = [encoder_dim] + [hidden_dim] * num_blocks
        for i in range(num_blocks):
            self.res_blocks.append(
                ResidualBlock(dims[i], dims[i+1], dropout=dropout * (1 - i * 0.1), use_se=True)
            )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Decoder: hidden_dim -> 128 -> 1
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # Encode
        x = self.encoder(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Self-attention (add residual)
        x_attn = x.unsqueeze(1)
        # Use simple attention without mask
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + attn_out.squeeze(1)
        x = self.attn_norm(x)
        
        # Decode to output
        return self.decoder(x).squeeze(-1)

# Keep old class for backward compatibility
class ImprovedNeuralNet(TDEClassifierNet):
    """Alias for backward compatibility."""
    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        # Ignore old parameters, use new simplified architecture
        super().__init__(input_dim, encoder_dim=512, hidden_dim=256, num_blocks=3, dropout=0.40)


class SimpleTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer for astronomical time-series.
    
    Removes complex VSN/GRN components that cause dimension issues.
    Uses standard LSTM + Attention for robust sequence modeling.
    """
    def __init__(self, input_dim, hidden_size=128, num_heads=4, dropout=0.2, 
                 num_lstm_layers=2, num_attention_layers=2):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size, num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2  # bidirectional
        
        # Multi-head self attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(lstm_output_size, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_attention_layers)
        ])
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(lstm_output_size) for _ in range(num_attention_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Handle NaN by replacing with zeros
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # LSTM encoding
        x, _ = self.lstm_encoder(x)  # (batch, seq_len, hidden_size*2)
        
        # Multi-head self attention with residual connections
        for attn, norm in zip(self.attention_layers, self.attention_norms):
            attn_out, _ = attn(x, x, x)  # Self-attention
            x = norm(x + attn_out)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, hidden_size*2)
        
        # Output
        return self.output_layers(x).squeeze(-1)  # (batch,)

# ============================================================
# LOSS & TRAINING UTILITIES
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for extremely imbalanced classification.
    
    Down-weights easy negatives, focuses on hard-to-classify positives.
    This directly targets the F1 loss from false negatives.
    
    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: class weight for positive class (higher = more recall)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
    """
    def __init__(self, gamma=None, alpha=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma if gamma is not None else getattr(cfg, 'FOCAL_LOSS_GAMMA', 2.0)
        self.alpha = alpha if alpha is not None else getattr(cfg, 'FOCAL_LOSS_ALPHA', 0.25)
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs are logits, targets are 0/1
        probs = torch.sigmoid(inputs)
        
        # Apply label smoothing
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute focal weights
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        
        # p_t = p for positive, 1-p for negative (use original targets for focal weight)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight: higher for positives
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for tabular data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class NNDataset(Dataset):
    """
    Dataset wrapper with optional Swap Noise regularization.
    """
    def __init__(self, X, y=None, swap_noise_rate=0.0):
        # Swap Noise regularization: randomly swap feature values between samples
        # to encourage the model to learn robust, positional invariance.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.swap_noise_rate = swap_noise_rate
  
    def __len__(self):
        return len(self.X)
  
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        
        # Apply Swap Noise during training only
        if self.swap_noise_rate > 0 and self.y is not None:
            mask = torch.rand(x.shape) < self.swap_noise_rate
            random_idx = torch.randint(0, len(self.X), (1,)).item()
            x[mask] = self.X[random_idx][mask]

        if self.y is not None:
            return x, self.y[idx]
        return x
