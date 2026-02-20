"""
Frequency branch: DWT-based CNN for frequency-domain feature extraction.

Takes the 4-channel DWT tensor (cA, cH, cV, cD subbands) from dwt_utils
and processes it through a small CNN to produce a 128-D feature vector.

Architecture:
    Input (4, 112, 112) → Conv layers → AdaptiveAvgPool → FC → 128-D output

Author: Simran Chaudhary
"""

import torch
import torch.nn as nn


class FrequencyDWTBranch(nn.Module):
    """
    Small CNN for frequency-domain feature extraction from DWT subbands.

    Input:  (B, 4, 112, 112) — 4-channel DWT tensor.
    Output: (B, 128) — frequency feature vector.
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 128):
        super().__init__()

        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            # Block 1: (4, 112, 112) → (32, 56, 56)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (32, 56, 56) → (64, 28, 28)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (64, 28, 28) → (128, 14, 14)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: (128, 14, 14) → (128, 7, 7)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global average pooling → (128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency features from DWT subbands.

        Args:
            x: (B, 4, 112, 112) DWT tensor.

        Returns:
            (B, 128) feature tensor.
        """
        x = self.features(x)
        x = self.fc(x)
        return x

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim


# ── Quick test ──
if __name__ == "__main__":
    model = FrequencyDWTBranch(in_channels=4, feature_dim=128)
    model.eval()

    dummy = torch.randn(2, 4, 112, 112)
    with torch.no_grad():
        features = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {features.shape}")  # Expected: (2, 128)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
