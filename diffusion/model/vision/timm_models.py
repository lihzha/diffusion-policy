import torch
from torch import nn
import timm
from typing import Callable


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


class TimmEncoder(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch14_dinov2.lvd142m",  # 'resnet18
        pretrained=False,
        share_rgb_model=False,
        num_views=3,
        img_cond_steps=1,
        frozen=False,
        use_lora=False,
        lora_rank=8,
        img_size=(3, 96, 96),
        drop_path_rate=0.0,
    ):

        super().__init__()

        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        self.model_name = model_name
        self.share_rgb_model = share_rgb_model

        if "resnet" in model_name:
            if not share_rgb_model:
                self.forward = self.forward_loop
                self.model = nn.ModuleList()
                for _ in range(num_views):
                    model = nn.Sequential(
                        *list(
                            timm.create_model(
                                model_name=model_name,
                                pretrained=pretrained,
                                global_pool="",
                                num_classes=0,
                            ).children()
                        )[:-2]
                    )
                    self.model.append(model)

            else:
                self.forward = self.forward_batch
                self.model = timm.create_model(
                    model_name=model_name,
                    global_pool="",
                    pretrained=pretrained,
                    num_classes=0,
                )
                self.model = nn.Sequential(
                    *list(self.model.children())[:-2],
                )

        elif "dino" in model_name:
            assert img_size[1] == img_size[2]
            if not share_rgb_model:
                self.forward = self.forward_loop

                self.model = nn.ModuleList(
                    [
                        timm.create_model(
                            model_name=model_name,
                            pretrained=pretrained,
                            global_pool="",  # '' means no pooling
                            num_classes=0,  # remove classification layer
                            img_size=img_size[1],
                            drop_path_rate=drop_path_rate,  # stochastic depth
                        )
                        for _ in range(num_views)
                    ]
                )

            else:
                self.forward = self.forward_batch
                self.model = timm.create_model(
                    model_name=model_name,
                    pretrained=pretrained,
                    global_pool="",  # '' means no pooling
                    num_classes=0,  # remove classification layer
                    img_size=img_size[1],
                    drop_path_rate=drop_path_rate,  # stochastic depth
                )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

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

    def forward_loop(self, x: dict):
        outputs = []
        for v, f in zip(x.values(), self.model):
            y = f(
                v
            )  # vit: bs*img_cond_steps, patch_nums, embed_dim; resnet: bs*img_cond_steps, embed_dim
            outputs.append(y)
        return torch.cat(
            outputs, dim=0
        )  # vit: bs*img_cond_steps*num_views, patch_nums, embed_dim;
        # resnet: bs*img_cond_steps*num_views, embed_dim, 7, 7

    def forward_batch(self, x: dict):
        # x is a dict with keys as view names and values as images. Concatenate the images along the batch dimension and reshape to (batch, patch_nums, embed_dim) after passing through the embedding layer.
        x = torch.cat(list(x.values()), dim=0)  # bs*img_cond_steps*num_views, c, h, w
        y = self.model(x)
        return y  # vit: bs*img_cond_steps*num_views, patch_nums, embed_dim;
        # resnet: bs*img_cond_steps*num_views, embed_dim, 7, 7


class DINOv2Encoder(TimmEncoder):
    def __init__(
        self,
        model_name="vit_base_patch14_dinov2.lvd142m",  # 'resnet18
        pretrained=False,
        share_rgb_model=False,
        num_views=3,
        img_cond_steps=1,
        frozen=False,
        use_lora=False,
        drop_path_rate=0.0,
        img_size=(3, 96, 96),
        lora_rank=8,
    ):
        super().__init__(
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


class ResNetEncoder(TimmEncoder):
    def __init__(
        self,
        model_name="resnet18",
        pretrained=False,
        share_rgb_model=False,
        num_views=3,
        img_cond_steps=1,
        use_group_norm=True,
        frozen=False,
        use_lora=False,
        lora_rank=8,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            share_rgb_model=share_rgb_model,
            num_views=num_views,
            img_cond_steps=img_cond_steps,
            frozen=frozen,
            use_lora=use_lora,
            lora_rank=lora_rank,
        )

        if use_group_norm:
            assert not pretrained
            if isinstance(self.model, nn.ModuleList):
                for i in range(num_views):
                    self.model[i] = replace_submodules(
                        root_module=self.model[i],
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features // 16, num_channels=x.num_features
                        ),
                    )
            else:
                self.model = replace_submodules(
                    root_module=self.model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features // 16, num_channels=x.num_features
                    ),
                )
