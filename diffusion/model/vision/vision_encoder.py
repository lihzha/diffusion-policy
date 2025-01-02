import einops
import torch
import torch.nn as nn

from diffusion.model.vision.modules import SpatialEmb


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        feature_aggregation,  # - type: 'spatial_emb', 'compress', 'avgpool', 'cls', 'concat', 'mean', 'max'
        aug=None,
        num_views=3,
        img_cond_steps=1,
        share_rgb_model=False,
        img_size=(3, 96, 96),
        # For timm models
        pretrained=False,
        frozen=False,
        use_lora=False,
        drop_path_rate=0.0,
        lora_rank=8,
        use_group_norm=True,
        # For custom vit model
        patch_size=16,
        depth=1,
        embed_dim=128,
        num_heads=4,
        embed_style="embed2",
        embed_norm=0,
        share_embed_head=False,
        use_large_patch=False,
    ):
        super().__init__()

        # 1. Augmentation
        if aug is not None:
            self.aug = torch.nn.Sequential(*aug)
            if "custom_vit" in model_name:
                from diffusion.model.vision.modules import CropRandomizer

                for i in range(len(self.aug)):
                    if isinstance(self.aug[i], CropRandomizer):
                        if self.aug[i].crop_height % patch_size != 0:
                            nearest_crop_height_pct = (
                                self.aug[i].crop_height // patch_size * patch_size
                            ) / img_size[-1]
                            raise ValueError(
                                f"crop_height should be divisible by patch_size. Nearest crop_height percentage: {nearest_crop_height_pct}"
                            )
                        if self.aug[i].crop_width % patch_size != 0:
                            nearest_crop_width_pct = (
                                self.aug[i].crop_width // patch_size * patch_size
                            ) / img_size[-2]
                            raise ValueError(
                                f"crop_width should be divisible by patch_size. Nearest crop_width percentage: {nearest_crop_width_pct}"
                            )
                        break
            self.raw_img_size = img_size.copy()
            img_size = self.aug(torch.randint(0, 256, (1, *img_size))).shape[-3:]
        else:
            self.raw_img_size = img_size.copy()

        # 2. Load model
        self.model_name = model_name
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.share_rgb_model = share_rgb_model
        self.use_large_patch = use_large_patch

        if "dino" in model_name:
            from diffusion.model.vision.timm_models import DINOv2Encoder

            self.model = DINOv2Encoder(
                model_name=model_name,
                pretrained=pretrained,
                share_rgb_model=share_rgb_model,
                num_views=num_views,
                img_cond_steps=img_cond_steps,
                frozen=frozen,
                use_lora=use_lora,
                drop_path_rate=drop_path_rate,
                img_size=img_size,
                lora_rank=lora_rank,
            )
        elif "resnet" in model_name:
            from diffusion.model.vision.timm_models import ResNetEncoder

            self.model = ResNetEncoder(
                model_name=model_name,
                pretrained=pretrained,
                share_rgb_model=share_rgb_model,
                num_views=num_views,
                img_cond_steps=img_cond_steps,
                use_group_norm=use_group_norm,
                frozen=frozen,
                use_lora=use_lora,
                lora_rank=lora_rank,
            )
        elif "custom_vit" in model_name:
            from diffusion.model.vision.vit import VitEncoder

            self.model = VitEncoder(
                img_size=img_size,
                img_cond_steps=img_cond_steps,
                patch_size=patch_size,
                depth=depth,
                embed_dim=embed_dim,
                num_heads=num_heads,
                embed_style=embed_style,
                embed_norm=embed_norm,
                share_embed_head=share_embed_head,
                num_views=num_views,
                use_large_patch=use_large_patch,
            )
        else:
            raise NotImplementedError

        # 3. Feature aggregation
        self.feature_aggregation = feature_aggregation.type
        if "vit" in model_name:
            assert self.feature_aggregation in [
                "spatial_emb",
                "compress",
                "cls",
                "concat",
                "mean",
            ], f"feature_aggregation type '{self.feature_aggregation}' not supported"
        elif "resnet" in model_name:
            assert self.feature_aggregation in [
                "avgpool",
                "compress",
                "max",
            ], f"feature_aggregation type '{self.feature_aggregation}' not supported"
        self.get_repr_dim()
        if self.feature_aggregation == "spatial_emb":
            assert "vit" in self.model_name, "only support vit model"
            assert feature_aggregation.spatial_emb > 1, "this is the patch dimension"
            self.nn_compress = SpatialEmb(
                num_patch=self.num_patch,
                patch_dim=self.patch_repr_dim,
                prop_dim=feature_aggregation.cond_dim,
                proj_dim=feature_aggregation.spatial_emb,
                dropout=feature_aggregation.dropout,
            )
            self.repr_dim = feature_aggregation.spatial_emb
        elif self.feature_aggregation == "compress":
            self.nn_compress = nn.Sequential(
                nn.Linear(self.raw_repr_dim, feature_aggregation.visual_feature_dim),
                nn.LayerNorm(feature_aggregation.visual_feature_dim),
                nn.Dropout(feature_aggregation.dropout),
                nn.ReLU(),
            )
            self.repr_dim = feature_aggregation.visual_feature_dim
        elif self.feature_aggregation == "avgpool":
            assert "resnet" in self.model_name, "only support resnet model"
            self.nn_compress = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        else:
            self.nn_compress = None

    def get_repr_dim(self):
        x = {
            f"{i}": torch.randint(0, 256, (1, self.img_cond_steps, *self.raw_img_size))
            for i in range(self.num_views)
        }
        x = self.forward(x, aggregate=False)

        if "resnet" in self.model_name:
            assert len(x.shape) == 4
            self.patch_repr_dim = x.shape[1]
            if self.feature_aggregation == "compress":
                self.raw_repr_dim = (
                    self.patch_repr_dim * self.img_cond_steps * self.num_views
                )
            else:
                self.repr_dim = (
                    self.patch_repr_dim * self.img_cond_steps * self.num_views
                )
            self.num_patch = None
        elif "vit" in self.model_name:
            self.num_patch = x.shape[1] * x.shape[0]
            self.patch_repr_dim = x.shape[-1]
            if self.feature_aggregation == "spatial_emb":
                pass
            elif self.feature_aggregation == "compress":
                self.raw_repr_dim = self.patch_repr_dim * self.num_patch
            else:
                x = self.aggregate_features(x)
                self.repr_dim = x.shape[-1]

    def forward(self, rgb: dict, state=None, aggregate=True):
        for k in rgb.keys():
            rgb[k] = einops.rearrange(rgb[k], "b cs c h w -> (b cs) c h w").contiguous()
            if self.aug:
                rgb[k] = self.aug(rgb[k])
        # if aggregate:
        #     cv2.imwrite(
        #         "i0.png",
        #         cv2.cvtColor(
        #             (rgb["0"].cpu().numpy()[0].transpose(1, 2, 0) + 0.5) * 255,
        #             cv2.COLOR_BGR2RGB,
        #         ),
        #     )
        # cv2.imwrite(
        #     "i2.png",
        #     cv2.cvtColor(
        #         (rgb["2"].cpu().numpy()[0].transpose(1, 2, 0) + 0.5) * 255,
        #         cv2.COLOR_BGR2RGB,
        #     ),
        # )
        feat = self.model.forward(
            rgb
        )  # (bs * img_cond_steps * num_views, patch_nums, embed_dim)
        if aggregate:
            feat = self.aggregate_features(feat, state)
        return feat

    def aggregate_features(self, x, state=None):
        if "resnet" in self.model_name:
            assert len(x.shape) == 4  # (batch*img_cond_steps*num_views, emb_dim, h, w)
            if self.feature_aggregation == "avgpool":
                x = self.nn_compress(x)
                x = einops.rearrange(
                    x,
                    "(v b cs) d-> b (cs v d)",
                    cs=self.img_cond_steps,
                    v=self.num_views,
                ).contiguous()
                return x
            x = torch.flatten(
                x, start_dim=-2
            )  # (batch*img_cond_steps*num_views, emb_dim, h*w)
            x = torch.transpose(
                x, 1, 2
            ).contiguous()  # (batch*img_cond_steps*num_views, h*w, emb_dim)
            x = einops.rearrange(
                x,
                "(v b cs) hw d-> b hw (cs v d)",
                cs=self.img_cond_steps,
                v=self.num_views,
            ).contiguous()
            if self.feature_aggregation == "compress":
                x = x.mean(dim=1)
                x = self.nn_compress(x)
            elif self.feature_aggregation == "max":
                x = x.max(dim=1)
            return x

        # vit
        assert "vit" in self.model_name
        assert (
            len(x.shape) == 3
        )  # (batch*img_cond_steps*num_views, patch_nums, embed_dim)
        if self.model_name == "custom_vit":
            if not self.use_large_patch:
                x = einops.rearrange(
                    x,
                    "(v b cs) p d -> b (cs v p) d",
                    cs=self.img_cond_steps,
                    v=self.num_views,
                ).contiguous()
        else:
            x = einops.rearrange(
                x,
                "(v b cs) p d -> b (cs v p) d",
                cs=self.img_cond_steps,
                v=self.num_views,
            ).contiguous()

        if self.feature_aggregation == "spatial_emb":
            x = self.nn_compress.forward(x, state)
            assert len(x.shape) == 2
        elif self.feature_aggregation == "compress":
            x = x.flatten(1, -1)
            x = self.nn_compress(x)
        elif self.feature_aggregation == "cls":
            # Get the cls token for each view, given that views and img_cond_steps and patches are concatenated
            cls_token_idx_list = [
                self.num_patch // self.num_views // self.img_cond_steps * i
                for i in range(self.num_views * self.img_cond_steps)
            ]
            x = x[:, cls_token_idx_list, :]
            x = x.flatten(1)
        elif self.feature_aggregation == "mean":
            x = x.mean(dim=1)
        elif self.feature_aggregation == "concat":
            x = x.flatten(1)
        else:
            raise NotImplementedError
        return x


if __name__ == "__main__":
    import math

    from omegaconf import OmegaConf

    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil)
    OmegaConf.register_new_resolver("round_down", math.floor)

    cfg = OmegaConf.load("guided_dc/cfg/real/pick_and_place/diffusion_unet_dino.yaml")
