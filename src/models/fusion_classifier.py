"""
Fusion classifier: combines spatial and frequency features for deepfake detection.

Supports three modes:
- 'spatial':   Uses only EfficientNet-B0 spatial features (1280-D)
- 'frequency': Uses only DWT frequency features (128-D)
- 'hybrid':    Concatenates both (1408-D) → MLP → binary logit

Author: Simran Chaudhary
"""

import torch
import torch.nn as nn
from src.models.spatial_efficientnet import SpatialEfficientNet
from src.models.frequency_dwt_branch import FrequencyDWTBranch


class FusionClassifier(nn.Module):
    """
    MLP classification head for fused spatial + frequency features.

    Input:  Concatenated feature vector (spatial_dim + freq_dim).
    Output: Single logit (use sigmoid for probability).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)  # (B,)


class HybridDeepfakeClassifier(nn.Module):
    """
    Complete hybrid deepfake classifier.

    Combines:
    - Spatial branch:   EfficientNet-B0 → 1280-D
    - Frequency branch: DWT CNN → 128-D
    - Fusion:           Concatenate → MLP → logit

    Supports 'spatial', 'frequency', or 'hybrid' modes.

    Args:
        mode: 'spatial', 'frequency', or 'hybrid'.
        pretrained_spatial: Whether to load ImageNet weights for EfficientNet.
        freq_feature_dim: Output dim of the frequency branch (default 128).
        hidden_dim: Hidden dim of the fusion MLP.
        dropout: Dropout rate in fusion MLP.
    """

    VALID_MODES = ("spatial", "frequency", "hybrid")

    def __init__(
        self,
        mode: str = "hybrid",
        pretrained_spatial: bool = True,
        freq_feature_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.VALID_MODES}")

        self.mode = mode

        # ── Spatial branch ──
        if mode in ("spatial", "hybrid"):
            self.spatial = SpatialEfficientNet(pretrained=pretrained_spatial)
            spatial_dim = self.spatial.get_feature_dim()  # 1280
        else:
            self.spatial = None
            spatial_dim = 0

        # ── Frequency branch ──
        if mode in ("frequency", "hybrid"):
            self.frequency = FrequencyDWTBranch(
                in_channels=4, feature_dim=freq_feature_dim
            )
            freq_dim = self.frequency.get_feature_dim()  # 128
        else:
            self.frequency = None
            freq_dim = 0

        # ── Fusion classifier ──
        total_dim = spatial_dim + freq_dim
        self.classifier = FusionClassifier(
            input_dim=total_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self._total_feature_dim = total_dim

    def forward(
        self,
        rgb_input: torch.Tensor = None,
        dwt_input: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            rgb_input: (B, 3, 224, 224) tensor for spatial branch.
            dwt_input: (B, 4, 112, 112) tensor for frequency branch.

        Returns:
            (B,) logits — apply sigmoid for probabilities.
        """
        features = []

        if self.mode in ("spatial", "hybrid"):
            if rgb_input is None:
                raise ValueError("rgb_input required for spatial/hybrid mode")
            spatial_feat = self.spatial(rgb_input)           # (B, 1280)
            features.append(spatial_feat)

        if self.mode in ("frequency", "hybrid"):
            if dwt_input is None:
                raise ValueError("dwt_input required for frequency/hybrid mode")
            freq_feat = self.frequency(dwt_input)            # (B, 128)
            features.append(freq_feat)

        fused = torch.cat(features, dim=1)                   # (B, 1280+128) or (B, 1280) or (B, 128)
        logits = self.classifier(fused)                      # (B,)
        return logits

    def get_feature_dim(self) -> int:
        return self._total_feature_dim


# ── Quick test ──
if __name__ == "__main__":
    for mode in ["spatial", "frequency", "hybrid"]:
        print(f"\n{'='*40}")
        print(f"Mode: {mode}")
        print(f"{'='*40}")

        model = HybridDeepfakeClassifier(mode=mode, pretrained_spatial=False)
        model.eval()

        rgb = torch.randn(2, 3, 224, 224) if mode != "frequency" else None
        dwt = torch.randn(2, 4, 112, 112) if mode != "spatial" else None

        with torch.no_grad():
            logits = model(rgb_input=rgb, dwt_input=dwt)
        print(f"Logits shape: {logits.shape}")       # Expected: (2,)
        print(f"Feature dim:  {model.get_feature_dim()}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters:   {n_params:,}")
