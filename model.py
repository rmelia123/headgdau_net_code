"""
HeadGDAU-Net: A Lightweight Hybrid Network with Group-Dilated-Attention and BiLSTM
for Prostate mp-MRI Segmentation

Complete PyTorch implementation with modular components:
- HeadNet: BiLSTM-based cross-slice context aggregation
- GDA Block: Grouped-Dilated-Attention with W-MSA
- GDAU-Net: U-Net backbone with GDA blocks
- Feature Fusion: 1x1 convolution channel reduction + additive fusion

Author: Meili Ren et al.
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ==================== HeadNet: Cross-Slice Context Aggregation ====================

class HeadNet(nn.Module):
    """
    HeadNet: Bidirectional LSTM for cross-slice anatomical sequence modeling.

    Captures directed morphological progression (apex->base) through asymmetric
    forward/backward processing, modeling how radiologists integrate adjacent
    slice information to resolve ambiguous boundaries.

    Args:
        input_channels: Number of input channels (default: 64 for preprocessed features)
        hidden_dim: BiLSTM hidden dimension (default: 128, empirically optimal)
        num_layers: Number of BiLSTM layers (default: 2)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(self, input_channels: int = 64, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(HeadNet, self).__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection: map input features to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Spatial downsampling: 256->128
        )

        # Bidirectional LSTM for sequence modeling
        # Input: (B, C, H, W) -> flatten spatial -> (B, H*W, C)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection: map back to input_channels for fusion
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, neighbor_slices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of HeadNet.

        Args:
            x: Central slice features (B, C, H, W)
            neighbor_slices: Adjacent slices (B, T, C, H, W) where T=2 (t-1, t+1)

        Returns:
            Temporal context features (B, C, H, W) for fusion with spatial features
        """
        B, C, H, W = x.shape
        T = neighbor_slices.shape[1]

        # Process each neighbor slice through input projection
        # neighbor_slices: (B, T, C, H, W) -> (B*T, C, H, W)
        neighbor_flat = neighbor_slices.view(B * T, C, H, W)
        neighbor_proj = self.input_projection(neighbor_flat)  # (B*T, hidden_dim, H/2, W/2)

        _, C_proj, H_proj, W_proj = neighbor_proj.shape

        # Flatten spatial dimensions for LSTM: (B*T, C_proj, H_proj, W_proj) -> (B, T, H_proj*W_proj, C_proj)
        neighbor_seq = neighbor_proj.view(B, T, C_proj, H_proj * W_proj)
        neighbor_seq = neighbor_seq.permute(0, 1, 3, 2)  # (B, T, H*W, C)
        neighbor_seq = neighbor_seq.reshape(B, T * H_proj * W_proj, C_proj)

        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(neighbor_seq)  # (B, T*H*W, hidden_dim)

        # Reshape back to spatial: (B, T*H*W, hidden_dim) -> (B, T, H*W, hidden_dim)
        lstm_out = lstm_out.view(B, T, H_proj * W_proj, self.hidden_dim)

        # Average across time dimension (t-1 and t+1)
        context = lstm_out.mean(dim=1)  # (B, H*W, hidden_dim)

        # Reshape to spatial: (B, H*W, hidden_dim) -> (B, hidden_dim, H_proj, W_proj)
        context = context.permute(0, 2, 1).view(B, self.hidden_dim, H_proj, W_proj)

        # Upsample back to original resolution
        context = F.interpolate(context, size=(H, W), mode='bilinear', align_corners=False)

        # Output projection to match input channels
        context = self.output_projection(context)
        context = self.dropout(context)

        return context


# ==================== GDA Block: Group-Dilated-Attention ====================

class GroupedDilatedConv(nn.Module):
    """
    Grouped Dilated Convolution with learnable multi-scale fusion.

    Three parallel branches with dilation rates 1, 2, 3 capturing:
    - Rate 1 (3x3 RF): Fine-grained local details (cellular structures)
    - Rate 2 (5x5 RF): Regional edge features and local shape variations  
    - Rate 3 (7x7 RF): Global prostate morphology and adjacent tissue relationships

    Learnable weights ω₁, ω₂, ω₃ are initialized to 1.0 and updated via
    backpropagation with softmax gating for adaptive multi-scale fusion.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8,
                 dilation_rates: Tuple[int, ...] = (1, 2, 3)):
        super(GroupedDilatedConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.dilation_rates = dilation_rates
        self.num_branches = len(dilation_rates)

        # Ensure channels divisible by groups
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"

        # Multi-scale dilated convolution branches
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d,
                         dilation=d, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for d in dilation_rates
        ])

        # Learnable fusion weights (initialized to 1.0, softmax-gated)
        self.fusion_weights = nn.Parameter(torch.ones(self.num_branches))

        # Channel shuffle for cross-group information flow
        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)

        Returns:
            Fused multi-scale features (B, C_out, H, W)
        """
        # Compute dilated features for each branch
        dilated_features = [conv(x) for conv in self.dilated_convs]

        # Softmax gating: ω̂ᵢ = 3·e^(ωᵢ) / Σⱼ e^(ωⱼ)  (Eq. 13 in paper)
        # Scale by num_branches to maintain magnitude
        weights = F.softmax(self.fusion_weights, dim=0) * self.num_branches

        # Weighted fusion: F_dilated = Σ ω̂ᵢ · DConv_ri(x)  (Eq. 14)
        fused = sum(w * feat for w, feat in zip(weights, dilated_features))

        # Channel shuffle for cross-group communication
        fused = self.channel_shuffle(fused)

        return fused


class ChannelShuffle(nn.Module):
    """Channel shuffle operation for cross-group information flow."""

    def __init__(self, groups: int):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.groups

        # Reshape: (B, C, H, W) -> (B, g, C/g, H, W)
        x = x.view(B, g, C // g, H, W)

        # Transpose: (B, g, C/g, H, W) -> (B, C/g, g, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Flatten back: (B, C/g, g, H, W) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class WindowedMSA(nn.Module):
    """
    Windowed Multi-Head Self-Attention (W-MSA) with relative position bias.

    4 attention heads with 7×7 windows. Each head operates on C/4 channels,
    maintaining computational efficiency while capturing diverse feature
    relationships. Relative position encoding provides spatial sensitivity.

    Args:
        dim: Input dimension (default: 64, must be divisible by num_heads)
        num_heads: Number of attention heads (default: 4)
        window_size: Window size (default: 7)
        qkv_bias: Whether to use bias in QKV projection (default: True)
        attn_drop: Attention dropout rate (default: 0.0)
        proj_drop: Projection dropout rate (default: 0.0)
    """

    def __init__(self, dim: int = 64, num_heads: int = 4, window_size: int = 7,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super(WindowedMSA, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = dim // num_heads

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias (learned, Eq. 15 in paper)
        # B ∈ R^(49×49) for 7×7 windows
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

        # Relative position index (fixed, not learned)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, 7, 7)
        coords_flatten = torch.flatten(coords, 1)  # (2, 49)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, 49, 49)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (49, 49, 2)

        # Shift to start from 0
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        relative_position_index = relative_coords.sum(-1)  # (49, 49)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features (B, H, W, C) - already window-partitioned
            mask: Attention mask for shifted windows (optional)

        Returns:
            Attention-refined features (B, H, W, C)
        """
        B, N, C = x.shape  # N = window_size^2 = 49

        # QKV projection: (B, N, C) -> (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention with relative position bias (Eq. 15)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GDABlock(nn.Module):
    """
    Group-Dilated-Attention (GDA) Block - core computational unit.

    Integrates three mutually compensatory components:
    1. Grouped Convolution: Reduces parameters but causes inter-group isolation
    2. Dilated Convolution: Expands receptive field per group, mitigating isolation
    3. Windowed MSA: Compensates for reduced long-range interaction via cross-window aggregation

    Each component's limitation is explicitly mitigated by another, yielding
    synergistic efficiency-accuracy balance (validated in Table 5 of paper).
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8,
                 num_heads: int = 4, window_size: int = 7, 
                 dilation_rates: Tuple[int, ...] = (1, 2, 3)):
        super(GDABlock, self).__init__()

        # Layer normalization (pre-normalization, as in VT/W-Swin)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # Grouped Dilated Convolution
        self.gdc = GroupedDilatedConv(in_channels, out_channels, groups, dilation_rates)

        # Windowed Multi-Head Self-Attention
        self.w_msa = WindowedMSA(dim=out_channels, num_heads=num_heads, 
                                  window_size=window_size)

        # Window partition/merge helpers
        self.window_size = window_size

        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(0.1)
        )

        # Residual connections
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)                             if in_channels != out_channels else nn.Identity()

    def window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Partition into non-overlapping windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """Reverse window partition."""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C_in, H, W)

        Returns:
            Output features (B, C_out, H, W)
        """
        B, C, H, W = x.shape

        # Residual shortcut
        shortcut = self.residual_conv(x)

        # Pre-normalization (LayerNorm on channel dimension)
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm1(x_norm)

        # Grouped Dilated Convolution
        x_gdc = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_gdc = self.gdc(x_gdc)
        x_gdc = x_gdc.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Windowed MSA
        # Partition into windows
        x_windows = self.window_partition(x_gdc, self.window_size)  # (B*num_windows, 7, 7, C)
        x_windows = x_windows.view(-1, self.window_size ** 2, x_gdc.shape[-1])  # (B*num_windows, 49, C)

        # Apply W-MSA
        attn_windows = self.w_msa(x_windows)  # (B*num_windows, 49, C)

        # Reverse windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, x_gdc.shape[-1])
        x_msa = self.window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        # Post-MSA normalization and MLP
        x_msa = self.norm2(x_msa)
        x_mlp = self.mlp(x_msa)

        # Residual connection
        x_out = x_msa + x_mlp
        x_out = x_out.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Final residual addition
        x_out = x_out + shortcut

        return x_out


# ==================== GDAU-Net: U-Net with GDA Blocks ====================

class GDAEncoder(nn.Module):
    """GDA Encoder with downsampling."""

    def __init__(self, in_channels: int, features: Tuple[int, ...] = (64, 128, 256)):
        super(GDAEncoder, self).__init__()

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i, feature in enumerate(features):
            in_ch = in_channels if i == 0 else features[i-1]
            self.blocks.append(GDABlock(in_ch, feature))
            if i < len(features) - 1:
                self.downs.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        skip_connections = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            skip_connections.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        return x, skip_connections


class GDADecoder(nn.Module):
    """GDA Decoder with upsampling and skip connections."""

    def __init__(self, features: Tuple[int, ...] = (256, 128, 64)):
        super(GDADecoder, self).__init__()

        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i, feature in enumerate(features):
            in_ch = features[i-1] if i > 0 else features[0]
            skip_ch = features[i] if i < len(features) - 1 else features[-1]

            self.ups.append(
                nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2)
            )
            self.blocks.append(GDABlock(feature + skip_ch, feature))

    def forward(self, x: torch.Tensor, skip_connections: list) -> torch.Tensor:
        for i, (up, block) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)

            # Handle size mismatch due to odd dimensions
            skip = skip_connections[-(i+1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return x


class GDAUNet(nn.Module):
    """
    GDAU-Net: Lightweight U-Net with GDA blocks for multi-scale feature extraction.

    Three downsampling encoding stages + three upsampling decoding stages.
    Skip connections preserve fine spatial details lost during downsampling.
    """

    def __init__(self, in_channels: int = 64, num_classes: int = 1):
        super(GDAUNet, self).__init__()

        features = (64, 128, 256)

        self.encoder = GDAEncoder(in_channels, features)
        self.bottleneck = GDABlock(features[-1], features[-1] * 2)
        self.decoder = GDADecoder((features[-1] * 2, features[-1], features[-2], features[-3]))

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(features[-3], num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding
        x, skip_connections = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding with skip connections
        x = self.decoder(x, skip_connections)

        # Segmentation
        out = self.seg_head(x)

        return out


# ==================== HeadGDAU-Net: Complete Model ====================

class HeadGDAUNet(nn.Module):
    """
    HeadGDAU-Net: Dual-stream architecture integrating cross-slice anatomical priors
    with lightweight multi-scale spatial encoding.

    Architecture:
    1. HeadNet extracts boundary priors from adjacent slices via BiLSTM
    2. 1x1 convolution reduces temporal features from 128->64 channels
    3. Additive fusion: F_fus = F_temp' + F_spa
    4. GDAU-Net performs efficient multi-scale segmentation
    5. Hybrid loss (sBFDloss) balances boundary precision vs. regional accuracy

    Args:
        in_channels: Input channels (3 for T2WI+DWI+DCE-MRI fusion)
        num_classes: Number of output classes (1 for binary prostate segmentation)
        headnet_hidden_dim: BiLSTM hidden dimension (default: 128)
        headnet_num_layers: BiLSTM layers (default: 2)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1,
                 headnet_hidden_dim: int = 128, headnet_num_layers: int = 2):
        super(HeadGDAUNet, self).__init__()

        # Input preprocessing: 3-channel mp-MRI -> 64-channel feature maps
        self.input_preprocess = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # HeadNet: Cross-slice context aggregation
        self.headnet = HeadNet(
            input_channels=64,
            hidden_dim=headnet_hidden_dim,
            num_layers=headnet_num_layers
        )

        # Channel reduction: 128 -> 64 via 1x1 convolution (Eq. 14-15 in paper)
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # GDAU-Net: Lightweight segmentation backbone
        self.gdau_net = GDAUNet(in_channels=64, num_classes=num_classes)

    def forward(self, x: torch.Tensor, neighbor_slices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of HeadGDAU-Net.

        Args:
            x: Central slice (B, C, H, W) - C=3 for multi-sequence fusion
            neighbor_slices: Adjacent slices (B, T, C, H, W) - T=2 (t-1, t+1)

        Returns:
            Segmentation probability map (B, 1, H, W)
        """
        # Preprocess central slice
        f_spa = self.input_preprocess(x)  # (B, 64, H, W)

        # HeadNet: Extract temporal context from neighbors
        f_temp = self.headnet(f_spa, neighbor_slices)  # (B, 64, H, W)

        # Channel reduction (already 64 channels, but applying for consistency)
        f_temp = self.channel_reduction(f_temp)

        # Feature fusion: additive (Eq. 15: F_fus = F_temp' + F_spa)
        f_fus = f_temp + f_spa

        # GDAU-Net segmentation
        out = self.gdau_net(f_fus)

        return out

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Quick test to verify model construction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = HeadGDAUNet(in_channels=3, num_classes=1).to(device)

    # Dummy input
    B, C, H, W = 2, 3, 256, 256
    T = 2  # Two neighbor slices (t-1, t+1)

    x = torch.randn(B, C, H, W).to(device)
    neighbor_slices = torch.randn(B, T, C, H, W).to(device)

    # Forward pass
    with torch.no_grad():
        out = model(x, neighbor_slices)

    print(f"Input shape: {x.shape}")
    print(f"Neighbor slices shape: {neighbor_slices.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {model.get_parameter_count():,}")
    print(f"Expected: ~26.2M")

    assert out.shape == (B, 1, H, W), f"Output shape mismatch: {out.shape}"
    print("\n✓ Model test passed!")


if __name__ == "__main__":
    test_model()
