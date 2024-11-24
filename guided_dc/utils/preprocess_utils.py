import torch
from typing import Dict, Callable
from collections import namedtuple

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")

def preprocess_img(img):
    assert len(img.shape) == 3, img.shape
    assert img.max() > 5, img.max() 
    img = img / 255.0 - 0.5
    return img


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def batch_apply(
        x: Batch, 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Batch:
    # Create a list of modified values by applying func to each field
    modified_values = [dict_apply(getattr(x, field), func) if isinstance(getattr(x, field), dict) else func(getattr(x, field)) for field in x._fields]
    
    # Return a new Batch instance with the modified values
    return Batch(*modified_values)