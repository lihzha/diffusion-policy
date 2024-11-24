from guided_dc.diffusion.model.common.modules import SpatialEmb, RandomShiftsAug, CropRandomizer
import torch.nn as nn
import einops


class VisionEncoder(nn.Module):
    def __init__(self,
                 model_name,
                 feature_aggregation, # - type: 'spatial_emb', 'compress', 'none', 'cls', 'concat', 'mean'
                 aug=None,
                 ):

        

        # 1. Augmentation
        self.aug = aug
        
        # 2. Feature aggregation
        self.feature_aggregation = feature_aggregation.type
        if self.feature_aggregation == 'spatial_emb' > 0:
            assert feature_aggregation.spatial_emb > 1, "this is the patch dimension"
            self.compress = SpatialEmb(
                num_patch=self.num_patch,
                patch_dim=self.patch_repr_dim,
                prop_dim=feature_aggregation.cond_dim,
                proj_dim=feature_aggregation.spatial_emb,
                dropout=feature_aggregation.dropout,
            )
        elif self.feature_aggregation == 'compress':
            self.compress = nn.Sequential(
                nn.Linear(self.repr_dim, feature_aggregation.visual_feature_dim),
                nn.LayerNorm(feature_aggregation.visual_feature_dim),
                nn.Dropout(feature_aggregation.dropout),
                nn.ReLU(),
            )
        else:
            self.compress = None

    def forward(self, rgb: dict, state):
        for k in rgb.keys():
            rgb[k] = einops.rearrange(rgb[k], "b t c h w -> (b t) c h w")
            if self.aug:
                rgb[k] = self.aug(rgb[k])
        
        
    
    def aggregate_features(self, x, state):
        assert len(x.shape) == 3  # (batch, patch_nums, embed_dim)
        if self.feature_aggregation == 'spatial_emb':
            return self.compress.forward(x, state)
        elif self.feature_aggregation == 'compress':
            x = x.flatten(1, -1)
            return self.compress(x)
        elif self.feature_aggregation == 'cls':
            return x[:, 0]
        elif self.feature_aggregation == 'mean':
            return x.mean(dim=1)
        elif self.feature_aggregation == 'concat':
            return x.flatten(1)
        else:
            raise NotImplementedError
    
    @property
    def repr_dim(self):
        raise NotImplementedError