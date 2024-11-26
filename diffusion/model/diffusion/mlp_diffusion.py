"""
MLP models for diffusion policies.

"""

import torch
import torch.nn as nn
import logging

from diffusion.model.common.mlp import MLP, ResidualMLP
from diffusion.model.diffusion.modules import SinusoidalPosEmb

log = logging.getLogger(__name__)


class VisionDiffusionMLP(nn.Module):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()

        # vision
        self.backbone = backbone
        visual_feature_dim = backbone.repr_dim

        # diffusion
        input_dim = (
            time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        )
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond: dict,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = x.shape
        # if isinstance(cond["rgb"], dict):
        #     _, T_rgb, C, H, W = next(iter(cond["rgb"].values())).shape
        # else:
        #     _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten chunk
        x = x.view(B, -1)

        # flatten history
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"].copy()

        # rgb = cond["rgb"][:, -self.img_cond_steps :]

        ## concatenate images in cond by channels
        # if self.num_img > 1:
        #     rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
        #     rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        # else:
        # rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        ## convert rgb to float32 for augmentation
        # rgb = rgb.float()

        # # get vit output - pass in two images separately
        # if self.num_img > 1:  # TODO: properly handle multiple images
        #     rgb1 = rgb[:, 0]
        #     rgb2 = rgb[:, 1]
        #     if self.augment:
        #         rgb1 = self.aug(rgb1)
        #         rgb2 = self.aug(rgb2)
        #     feat1 = self.backbone(rgb1)
        #     feat2 = self.backbone(rgb2)
        #     feat1 = self.compress1.forward(feat1, state)
        #     feat2 = self.compress2.forward(feat2, state)
        #     feat = torch.cat([feat1, feat2], dim=-1)
        # else:  # single image
        # if self.augment:
        #     rgb = self.aug(rgb)
        feat = self.backbone(rgb, state)  # [batch, num_patch, embed_dim]
        cond_encoded = torch.cat([feat, state], dim=-1)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, cond_encoded], dim=-1)

        # mlp
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, Ta, Da = x.shape

        # flatten chunk
        x = x.view(B, -1)

        # flatten history
        state = cond["state"].view(B, -1)

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # append time and cond
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, state], dim=-1)

        # mlp head
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)
