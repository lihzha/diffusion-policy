"""
Custom ViT image encoder implementation from IBRL, https://github.com/hengyuan-hu/ibrl

"""

import math

import einops
import torch
from torch import nn
from torch.nn.init import trunc_normal_


class VitEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        img_cond_steps=1,
        patch_size=8,
        depth=1,
        embed_dim=128,
        num_heads=4,
        # act_layer=nn.GELU,
        embed_style="embed2",
        embed_norm=0,
        share_embed_head=False,
        num_views=3,
        use_large_patch=False,
    ):
        super().__init__()
        self.vit = MinVit(
            embed_style=embed_style,
            embed_dim=embed_dim,
            embed_norm=embed_norm,
            num_head=num_heads,
            depth=depth,
            num_channel=img_size[0],
            img_h=img_size[1],
            img_w=img_size[2],
            share_embed_head=share_embed_head,
            num_views=num_views,
            img_cond_steps=img_cond_steps,
            patch_size=patch_size,
            use_large_patch=use_large_patch,
        )
        self.img_h = img_size[1]
        self.img_w = img_size[2]
        self.num_patch = self.vit.num_patch
        self.embed_dim = embed_dim

    def forward(self, obs) -> torch.Tensor:
        feats: torch.Tensor = self.vit.forward(obs)  # [batch, num_patch, embed_dim]
        return feats


class PatchEmbed1(nn.Module):
    def __init__(self, embed_dim, num_channel=3, img_h=240, img_w=320, patch_size=8):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channel, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.num_patch = math.ceil(img_h / patch_size) * math.ceil(img_w / patch_size)
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c").contiguous()
        return y


class PatchEmbed2(nn.Module):
    def __init__(
        self, embed_dim, use_norm, num_channel=3, img_h=240, img_w=320, patch_size=8
    ):
        super().__init__()
        coef = patch_size // 8
        ks1 = 8 * coef
        stride1 = 4 * coef
        ks2 = 3
        stride2 = 2
        layers = [
            nn.Conv2d(num_channel, embed_dim, kernel_size=ks1, stride=stride1),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=ks2, stride=stride2),
        ]
        self.embed = nn.Sequential(*layers)
        H1 = math.ceil((img_h - ks1) / stride1) + 1
        W1 = math.ceil((img_w - ks1) / stride1) + 1
        H2 = math.ceil((H1 - ks2) / stride2) + 1
        W2 = math.ceil((W1 - ks2) / stride2) + 1
        self.num_patch = H2 * W2
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor):
        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c").contiguous()
        return y


class MultiViewPatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_channel=3,
        img_h=240,
        img_w=320,
        embed_style="embed2",
        num_views=3,
        use_norm=True,
        share_embed_head=False,
        img_cond_steps=1,
        patch_size=8,
        use_large_patch=False,
    ):
        super().__init__()
        if not share_embed_head:
            self.forward = self.forward_loop
            if embed_style == "embed1":
                self.embed_layers = nn.ModuleList(
                    [
                        PatchEmbed1(
                            embed_dim,
                            num_channel=num_channel,
                            img_h=img_h,
                            img_w=img_w,
                            patch_size=patch_size,
                        )
                        for _ in range(num_views)
                    ]
                )
            elif embed_style == "embed2":
                self.embed_layers = nn.ModuleList(
                    [
                        PatchEmbed2(
                            embed_dim,
                            use_norm=use_norm,
                            num_channel=num_channel,
                            img_h=img_h,
                            img_w=img_w,
                            patch_size=patch_size,
                        )
                        for _ in range(num_views)
                    ]
                )
            else:
                raise ValueError("Invalid patch embedding style.")
        else:
            self.forward = self.forward_batch
            if embed_style == "embed1":
                self.embed_layers = PatchEmbed1(
                    embed_dim,
                    num_channel=num_channel,
                    img_h=img_h,
                    img_w=img_w,
                    patch_size=patch_size,
                )

            elif embed_style == "embed2":
                self.embed_layers = PatchEmbed2(
                    embed_dim,
                    use_norm=use_norm,
                    num_channel=num_channel,
                    img_h=img_h,
                    img_w=img_w,
                    patch_size=patch_size,
                )
            else:
                raise ValueError("Invalid patch embedding style.")
        self.num_views = num_views
        self.img_cond_steps = img_cond_steps
        self.img_h = img_h
        self.img_w = img_w
        self.embed_dim = embed_dim
        self.use_large_patch = use_large_patch
        if use_large_patch:
            self.num_patch = (
                self.embed_layers[0].num_patch * num_views * img_cond_steps
                if isinstance(self.embed_layers, nn.ModuleList)
                else self.embed_layers.num_patch * num_views * img_cond_steps
            )
        else:
            self.num_patch = (
                self.embed_layers[0].num_patch
                if isinstance(self.embed_layers, nn.ModuleList)
                else self.embed_layers.num_patch
            )
        self.test_patch_embed()

    def test_patch_embed(self):
        x = {
            f"{i}": torch.rand(2 * self.img_cond_steps, 3, self.img_h, self.img_w)
            for i in range(self.num_views)
        }
        y = self.forward(x)
        if self.use_large_patch:
            assert y.size() == (
                2,
                self.num_patch,
                self.embed_dim,
            ), f"{y.size()}, {(2, self.num_patch, self.embed_dim)}"
        else:
            assert y.size() == (
                2 * self.img_cond_steps * self.num_views,
                self.num_patch,
                self.embed_dim,
            )

    def forward_loop(self, x: dict):
        patch_embds = []
        for v, embed in zip(x.values(), self.embed_layers):
            y = embed(v)
            patch_embds.append(y)
        if self.use_large_patch:
            ret = torch.cat(patch_embds, dim=0)  # (v b cs) p d
            return einops.rearrange(
                ret,
                "(v b cs) p d -> b (cs v p) d",
                v=self.num_views,
                cs=self.img_cond_steps,
            ).contiguous()
        else:
            return torch.cat(patch_embds, dim=0)  # (v b cs) p d

    def forward_batch(self, x: dict):
        # x is a dict with keys as view names and values as images. Concatenate the images along the batch dimension and reshape to (batch, patch_nums, embed_dim) after passing through the embedding layer.
        x = torch.cat(list(x.values()), dim=0)
        y = self.embed_layers(x)  # (v b cs) p d
        # y = einops.rearrange(y, "(b cs v) p d -> b (cs v p) d", b=b, cs=self.img_cond_steps, v=self.num_views)
        if self.use_large_patch:
            y = einops.rearrange(
                y,
                "(v b cs) p d -> b (cs v p) d",
                v=self.num_views,
                cs=self.img_cond_steps,
            ).contiguous()
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        x: [batch, seq, embed_dim]
        """
        qkv = self.qkv_proj(x)
        q, k, v = (
            einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.num_head)
            .unbind(1)
            .contiguous()
        )
        # force flash/mem-eff attention, it will raise error if flash cannot be applied
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)").contiguous()
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.gelu(self.linear1(x)))
        return x


class MinVit(nn.Module):
    def __init__(
        self,
        embed_style,
        embed_dim,
        embed_norm,
        num_head,
        depth,
        num_channel=3,
        img_h=240,
        img_w=320,
        num_views=3,
        share_embed_head=False,
        img_cond_steps=1,
        patch_size=8,
        use_large_patch=False,
    ):
        super().__init__()

        self.patch_embed = MultiViewPatchEmbed(
            embed_dim,
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
            embed_style=embed_style,
            num_views=num_views,
            share_embed_head=share_embed_head,
            use_norm=embed_norm,
            img_cond_steps=img_cond_steps,
            patch_size=patch_size,
            use_large_patch=use_large_patch,
        )

        self.num_patch = self.patch_embed.num_patch

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch, embed_dim))

        layers = [
            TransformerLayer(embed_dim, num_head, dropout=0) for _ in range(depth)
        ]

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        # weight init
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        x = self.patch_embed(x)  # [batch, num_patch*img_cond_steps]
        x = x + self.pos_embed
        x = self.net(x)  # [batch, num_patch, embed_dim]
        return self.norm(x)  # [batch, num_patch, embed_dim]


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(
    fn, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def test_patch_embed():
    print("embed 1")
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 96, 96)
    y = embed(x)
    print(y.size())

    print("embed 2")
    embed = PatchEmbed2(128, True)
    x = torch.rand(10, 3, 96, 96)
    y = embed(x)
    print(y.size())


def test_transformer_layer():
    embed = PatchEmbed1(128)
    x = torch.rand(10, 3, 96, 96)
    y = embed(x)
    print(y.size())

    transformer = TransformerLayer(128, 4, False, 0)
    z = transformer(y)
    print(z.size())


if __name__ == "__main__":
    # obs_shape = [3, 480, 640]
    # num_views = 1

    # enc = VitEncoder(
    #     obs_shape,
    #     num_channel=obs_shape[0],
    #     img_h=obs_shape[1],
    #     img_w=obs_shape[2],
    # )
    # x = {key: torch.rand(64, *obs_shape) * 255 for key in [f"view{i}" for i in range(num_views)]}

    # print("input size:", x.keys())
    # print("embed_size", enc.vit.patch_embed(x).size())
    # print("output size:", enc(x, flatten=False).size())
    # print("repr dim:", enc.repr_dim, ", real dim:", enc(x, flatten=True).size())
    pm = MultiViewPatchEmbed(
        128,
        num_channel=3,
        img_h=88,
        img_w=88,
        embed_style="embed2",
        num_views=2,
        use_norm=False,
        share_embed_head=False,
        img_cond_steps=2,
        patch_size=8,
        use_large_patch=True,
    )
    # pm = PatchEmbed2(128, use_norm=0, num_channel=3, img_h=240, img_w=320, patch_size=8)
    x = {f"{i}": torch.rand(10, 3, 88, 88) for i in range(3)}
    y = pm(x)
    print(y.size())
    breakpoint()
