"""
model.py

Hierarchical multi-head classifier. One ResNet18 backbone feeds three
linear heads (species / genus / family).

Warm-starting from a pretrained ResNet18 state_dict is handled at
construction time: the fc weights are discarded, the conv/bn weights are
loaded into the trunk, and the three heads start from their default init.

update_heads() grows or shrinks the heads while preserving the rows that
already exist, so retraining during active learning keeps everything the
model has already learned.  A shrink is technically supported for
robustness, but it is *warned about* because in this pipeline it almost
always means a caller passed an empty prev_* map — which is exactly the
failure mode that silently truncated the checkpoint before the sidecar
label-map fix.
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
        verbose=True,
    ):
        super().__init__()

        self.verbose = verbose

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if backbone_path is not None:
            state = torch.load(backbone_path, map_location="cpu")
            # Accept both raw resnet18 state_dicts (with 'fc.*') and our own
            # multihead checkpoints (with 'fc_species.*' etc.) — strip anything
            # that is clearly a classifier head so the trunk load is clean.
            state = {
                k: v
                for k, v in state.items()
                if not (
                    k.startswith("fc.")
                    or k.startswith("fc_species.")
                    or k.startswith("fc_genus.")
                    or k.startswith("fc_family.")
                )
            }
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            # The only expected missing keys are the original 'fc.*' pair.
            extra_missing = [k for k in missing if not k.startswith("fc.")]
            if extra_missing:
                print(
                    f"   ⚠ backbone missing keys: {extra_missing[:8]}"
                    f"{' …' if len(extra_missing) > 8 else ''}"
                )
            if unexpected:
                print(
                    f"   ⚠ backbone unexpected keys: {unexpected[:8]}"
                    f"{' …' if len(unexpected) > 8 else ''}"
                )

        self.feature_dim = backbone.fc.in_features
        # Drop fc, keep avgpool.  nn.Sequential indices map to:
        #   0 conv1, 1 bn1, 2 relu, 3 maxpool, 4-7 layer1-4, 8 avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.fc_species = nn.Linear(self.feature_dim, max(num_species, 2))
        self.fc_genus = nn.Linear(self.feature_dim, max(num_genera, 2))
        self.fc_family = nn.Linear(self.feature_dim, max(num_families, 2))

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.fc_species(f), self.fc_genus(f), self.fc_family(f)

    # ------------------------------------------------------------------
    # Head management
    # ------------------------------------------------------------------
    @staticmethod
    def _grow_head(
        old_head: nn.Linear, new_n: int, name: str = "head", verbose: bool = True
    ) -> nn.Linear:
        """
        Return a Linear with `new_n` outputs whose first min(old, new) rows
        are copied from `old_head`.  Any newly added rows keep the default
        initialisation.  If `new_n == old_n` the original module is
        returned unchanged so its optimizer state (if any) survives.

        Shrinking is supported but logged loudly: in this pipeline it
        almost always means the caller passed an empty prev_* map, which
        silently discarded learned head weights before the sidecar
        label-map fix.  The first `new_n` rows are preserved either way,
        matching the index-stable contract of build_label_maps().
        """
        new_n = max(int(new_n), 2)
        old_n, feat_dim = old_head.weight.shape
        if new_n == old_n:
            return old_head

        new_head = nn.Linear(feat_dim, new_n).to(
            device=old_head.weight.device, dtype=old_head.weight.dtype
        )
        copy_n = min(old_n, new_n)
        with torch.no_grad():
            new_head.weight[:copy_n] = old_head.weight[:copy_n].to(
                new_head.weight.dtype
            )
            new_head.bias[:copy_n] = old_head.bias[:copy_n].to(new_head.bias.dtype)

        if verbose:
            if new_n > old_n:
                print(
                    f"   ⬆ {name}: {old_n} → {new_n} (+{new_n - old_n} rows, "
                    f"{copy_n} preserved)"
                )
            else:
                print(
                    f"   ⚠ {name} SHRANK: {old_n} → {new_n} "
                    f"({old_n - new_n} rows discarded). This usually means "
                    f"the caller passed an empty prev_* map — check that "
                    f"the label-map sidecar loaded correctly."
                )
        return new_head

    def update_heads(self, num_species, num_genera, num_families):
        """
        Resize the three classifier heads in place, preserving every row
        that already exists.  Called whenever the label set grows during
        active learning.  The backbone is never touched.

        Returns a dict of the form:
            {"species": (old_n, new_n),
             "genus":   (old_n, new_n),
             "family":  (old_n, new_n)}
        so the caller can log / assert against unexpected resizes.  An
        entry is `(n, n)` when the head was already the requested size.
        """
        before = (
            self.fc_species.out_features,
            self.fc_genus.out_features,
            self.fc_family.out_features,
        )

        self.fc_species = self._grow_head(
            self.fc_species, num_species, "species head", self.verbose
        )
        self.fc_genus = self._grow_head(
            self.fc_genus, num_genera, "genus head", self.verbose
        )
        self.fc_family = self._grow_head(
            self.fc_family, num_families, "family head", self.verbose
        )

        after = (
            self.fc_species.out_features,
            self.fc_genus.out_features,
            self.fc_family.out_features,
        )
        return {
            "species": (before[0], after[0]),
            "genus": (before[1], after[1]),
            "family": (before[2], after[2]),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def num_species(self):
        return self.fc_species.out_features

    @property
    def num_genera(self):
        return self.fc_genus.out_features

    @property
    def num_families(self):
        return self.fc_family.out_features
