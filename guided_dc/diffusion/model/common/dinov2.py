import einops
import torch
from torch import nn
import torchvision
from typing import Callable
import timm
from guided_dc.diffusion.model.common.vis_encoder import VisionEncoder
   
   
class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        # aug,
        # feature_aggregation,
        model_name='vit_base_patch14_dinov2.lvd142m',  # 'resnet18
        # weights='IMAGENET1K_V1',
        pretrained=False,
        share_rgb_model=False,
        num_views=3,
        img_cond_steps=1,
        frozen=False,
        use_lora=False,
        drop_path_rate=0.0,
        img_size=96,
        lora_rank=8,
        feature_aggregation='cls', # 'cls', 'mean', 'concat'
    ):
        # super().__init__(aug=aug, feature_aggregation=feature_aggregation)
        
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.model_name = model_name
        self.feature_aggregation = feature_aggregation
        self.share_rgb_model = share_rgb_model
        
        if not share_rgb_model:
            self.forward = self.forward_loop
            
            self.model = nn.ModuleList([timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool='',    # '' means no pooling
                num_classes=0,              # remove classification layer
                img_size=img_size,
                drop_path_rate=drop_path_rate,  # stochastic depth
            ) for _ in range(num_views)]
            )
    
        else:
            self.forward = self.forward_batch
            self.model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool='',    # '' means no pooling
                num_classes=0,              # remove classification layer
                img_size=img_size,   
                drop_path_rate=drop_path_rate,  # stochastic depth
            )
    
        if frozen:
            assert pretrained
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_lora:
            import peft
            assert pretrained and not frozen
            lora_config = peft.LoraConfig(
                r=lora_rank,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=["qkv"],
            )
            model = peft.get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        self.get_repr_dim(img_size)
            
    def get_repr_dim(self, img_size):
        if not self.share_rgb_model:
            x = self.model[0].forward(torch.randn(1, 3, img_size, img_size))
            self.patch_repr_dim = x.shape[-1]
            self.num_patch = x.shape[-2]
            x = self.aggregate_features(x)
            self.repr_dim = x.shape[-1] * self.img_cond_steps * self.num_views
        else:
            x = self.model.forward(torch.randn(1, 3, img_size, img_size))
            self.patch_repr_dim = x.shape[-1]
            self.num_patch = x.shape[-2]
            x = self.aggregate_features(x)
            self.repr_dim = x.shape[-1] * self.img_cond_steps * self.num_views
        
        
    def aggregate_features(self, x):
        if self.feature_aggregation == 'cls':
            return x[:, 0]
        elif self.feature_aggregation == 'mean':
            return x.mean(dim=1)
        elif self.feature_aggregation == 'concat':
            return x.flatten(1)
        else:
            raise NotImplementedError
    
    
    def forward_loop(self, x: dict):
        outputs = []
        for v, f in zip(x.values(), self.model):
            y = f(v)
            y = self.aggregate_features(y)
            y = einops.rearrange(y, "(b cs) d -> b (cs d)", cs=self.img_cond_steps)
            outputs.append(y)
        return torch.cat(outputs, dim=-1)  # bs, img_cond_steps*embed_dim*num_views
    
    def forward_batch(self, x: dict):
        # x is a dict with keys as view names and values as images. Concatenate the images along the batch dimension and reshape to (batch, patch_nums, embed_dim) after passing through the embedding layer.
        x = torch.cat(list(x.values()), dim=0)
        y = self.model(x)
        y = self.aggregate_features(y)
        y = einops.rearrange(y, "(bcs v) d -> bcs (v d)", v=self.num_views)
        y = einops.rearrange(y, "(b cs) vd -> b (cs vd)", cs=self.img_cond_steps)
        return y