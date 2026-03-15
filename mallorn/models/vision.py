"""
Vision transformer components for light curve imaging classification.
"""
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# VISION TRANSFORMER (SwinV2) FOR LIGHT CURVE CLASSIFICATION
# ============================================================

class SimpleTransformerBlock(nn.Module):
    """
    Simplified Vision Transformer Block (no window partitioning).
    
    More memory efficient and stable than window-based attention.
    Uses global attention on patch embeddings.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """x: (B, N, C) - batch, num_patches, channels"""
        # Self attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for light curve images."""
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = img_size // patch_size
        self.num_patches = self.patches_resolution ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x


class SwinV2LightCurveTransformer(nn.Module):
    """
    Simplified Vision Transformer for Light Curve Classification.
    
    Uses global attention on patch embeddings - more stable than window-based.
    Optimized for 12GB VRAM (RTX 3060).
    
    # SwinV2 Vision Transformer for image-based light curve classification
    """
    def __init__(self, img_size=128, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=48, num_layers=4, num_heads=4, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Absolute position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks (no window partitioning, no hierarchical merging)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        
    def forward(self, x):
        # Patch embedding: (B, C, H, W) -> (B, N, embed_dim)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling and classification
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, embed_dim)
        x = self.head(x)   # (B, num_classes)
        return x.squeeze(-1)


class PatchMerging(nn.Module):
    """Patch Merging Layer for hierarchical architecture."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        x = x.view(B, H, W, C)
        
        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


def light_curve_to_image(lc_data, band_data, img_size=256):
    """
    Convert light curve to 2D image representation for SwinV2.
    
    Two methods:
    - Overlay: 66.9% F1 (small datasets)
    - Grid: 81.7% F1 (large datasets)
    
    Returns 3-channel image (g, r, i bands)
    """
    img = np.zeros((3, img_size, img_size), dtype=np.float32)
    bands = ['g', 'r', 'i']
    
    for ch, band in enumerate(bands):
        if band in band_data:
            t = band_data[band]['t']
            f = band_data[band]['f']
            
            if len(t) < 2:
                continue
                
            # Normalize to image coordinates
            t_norm = (t - t.min()) / (t.max() - t.min() + 1e-10)
            f_norm = (f - f.min()) / (f.max() - f.min() + 1e-10)
            
            # Grid method: map to 2D grid
            x_coords = (t_norm * (img_size - 1)).astype(int)
            y_coords = ((1 - f_norm) * (img_size - 1)).astype(int)
            
            # Clip to bounds
            x_coords = np.clip(x_coords, 0, img_size - 1)
            y_coords = np.clip(y_coords, 0, img_size - 1)
            
            # Draw points with Gaussian blur
            for x, y in zip(x_coords, y_coords):
                if 0 <= x < img_size and 0 <= y < img_size:
                    img[ch, y, x] = 1.0
                    
    return img
