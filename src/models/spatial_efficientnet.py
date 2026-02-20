"""
Spatial branch: EfficientNet-B0 feature extractor.

Wraps torchvision.models.efficientnet_b0 with ImageNet pretrained weights.
Removes the classification head, outputting a 1280-D feature vector per image.

Author: Simran Chaudhary
"""

import torch
import torch.nn as nn
from torchvision import models


class SpatialEfficientNet(nn.Module):
    """
    EfficientNet-B0 spatial feature extractor.

    Input:  (B, 3, 224, 224) — RGB face crops, ImageNet-normalized.
    Output: (B, 1280) — spatial feature vector.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load EfficientNet-B0 with ImageNet weights
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Feature dimension before the classifier
        self.feature_dim = self.backbone.classifier[1].in_features  # 1280

        # Remove the classification head
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features.

        Args:
            x: (B, 3, 224, 224) tensor, ImageNet-normalized.

        Returns:
            (B, 1280) feature tensor.
        """
        return self.backbone(x)

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim


# ── Quick test ──
if __name__ == "__main__":
    model = SpatialEfficientNet(pretrained=True)
    model.eval()

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {features.shape}")  # Expected: (2, 1280)
    print(f"Feature dim: {model.get_feature_dim()}")
