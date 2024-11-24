import einops
import torch
from torch import nn
import torchvision
from typing import Callable
import timm

# def get_resnet(name, weights=None, **kwargs):
#     """
#     name: resnet18, resnet34, resnet50
#     weights: "IMAGENET1K_V1", "r3m"
#     """
#     # load r3m weights
#     if (weights == "r3m") or (weights == "R3M"):
#         return get_r3m(name=name, **kwargs)

#     func = getattr(torchvision.models, name)
#     resnet = func(weights=weights, **kwargs)
#     resnet.fc = torch.nn.Identity()
#     return resnet

# def get_r3m(name, **kwargs):
#     """
#     name: resnet18, resnet34, resnet50
#     """
#     import r3m
#     r3m.device = 'cpu'
#     model = r3m.load_r3m(name)
#     r3m_model = model.module
#     resnet_model = r3m_model.convnet
#     resnet_model = resnet_model.to('cpu')
#     return resnet_model

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
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
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module
   
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        name='resnet18',
        # weights='IMAGENET1K_V1',
        pretrained=False,
        share_rgb_model=False,
        num_views=3,
        img_cond_steps=1,
        use_group_norm=True,
        frozen=False,
        use_lora=False,
        lora_rank=8
    ):
        super().__init__()
        
        self.img_cond_steps = img_cond_steps
        self.num_views = num_views
        
        assert 'resnet' in name
        
        if not share_rgb_model:
            self.forward = self.forward_loop
            self.model = nn.ModuleList()
            for _ in range(num_views):
                model = nn.Sequential(*list(timm.create_model(
                    model_name=name,
                    pretrained=pretrained,
                    global_pool='',
                    num_classes=0
                ).children())[:-2]
                )
                model = nn.Sequential(model, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
                
                self.model.append(model)
    
        else:
            self.forward = self.forward_batch
            self.model = timm.create_model(
                model_name=name,
                global_pool='',
                pretrained=pretrained,
                num_classes=0
            )
            
            # Add a AdaptiveAvgPool2d(output_size=(1, 1)) layer to the model
            self.model = nn.Sequential(
                *list(self.model.children())[:-2],
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
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
            
       
        if use_group_norm:
            assert not pretrained
            if isinstance(self.model, nn.ModuleList):
                for i in range(num_views):
                    self.model[i] = replace_submodules(
                        root_module=self.model[i],
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features//16, 
                            num_channels=x.num_features)
                    )
            else:
                self.model = replace_submodules(
                    root_module=self.model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features//16, 
                        num_channels=x.num_features)
                )
            
    @property
    def repr_dim(self):
        return 512 * self.img_cond_steps * self.num_views

    def forward_loop(self, x: dict):
        outputs = []
        for v, f in zip(x.values(), self.model):
            y = f(v)
            y = einops.rearrange(y, "(b cs) d -> b (cs d)", cs=self.img_cond_steps)
            outputs.append(y)
        return torch.cat(outputs, dim=-1)  # bs, img_cond_steps*embed_dim*num_views
    
    def forward_batch(self, x: dict):
        # x is a dict with keys as view names and values as images. Concatenate the images along the batch dimension and reshape to (batch, patch_nums, embed_dim) after passing through the embedding layer.

       
        x = torch.cat(list(x.values()), dim=0)
        
        y = self.model(x)
        y = einops.rearrange(y, "(bcs v) d -> bcs (v d)", v=self.num_views)
        y = einops.rearrange(y, "(b cs) vd -> b (cs vd)", cs=self.img_cond_steps)
        return y