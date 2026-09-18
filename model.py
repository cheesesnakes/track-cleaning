"""
model.py

Hierarchical multi-head classifier. One ResNet18 backbone feeds three
linear heads (species / genus / family). Warm-starting from a pretrained
backbone is handled at construction time — the fc weights from the
pretraining checkpoint are simply discarded.
"""

import torch
import torch.nn as nn
from torchvision import models


class TaxonomicMultiHead(nn.Module):
    def __init__(
        self,
        backbone_path=None,
        num_species=2,
        num_genera=2,
        num_families=2,
    ):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if backbone_path is not None:
            state = torch.load(backbone_path, map_location="cpu")
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            # missing == ['fc.weight', 'fc.bias'] is expected; anything else is not
            extra_missing = [k for k in missing if not k.startswith("fc.")]
            if extra_missing:
                print(f"   ⚠ backbone missing keys: {extra_missing}")
            if unexpected:
                print(f"   ⚠ backbone unexpected keys: {unexpected}")

        self.feature_dim = backbone.fc.in_features
        # Strip the final fc; keep avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.fc_species = nn.Linear(self.feature_dim, num_species)
        self.fc_genus = nn.Linear(self.feature_dim, num_genera)
        self.fc_family = nn.Linear(self.feature_dim, num_families)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.fc_species(f), self.fc_genus(f), self.fc_family(f)

    def update_heads(self, num_species, num_genera, num_families):
        """
        Rebuild only the classification heads, preserving the (trained)
        backbone. Called whenever the label set grows.
        """
        device = next(self.parameters()).device
        self.fc_species = nn.Linear(self.feature_dim, max(num_species, 2)).to(device)
        self.fc_genus = nn.Linear(self.feature_dim, max(num_genera, 2)).to(device)
        self.fc_family = nn.Linear(self.feature_dim, max(num_families, 2)).to(device)

    @property
    def num_species(self):
        return self.fc_species.out_features

    @property
    def num_genera(self):
        return self.fc_genus.out_features

    @property
    def num_families(self):
        return self.fc_family.out_features
