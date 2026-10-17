
import os
import warnings
from typing import List, Optional, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    return list_pretrained_models_by_tag('openai')


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    try:
        
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        
        state_dict = torch.load(model_path, map_location="cpu")

    
    cast_dtype = get_cast_dtype(precision)
    try:
        model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

    
    model = model.to(device)
    
    if precision != 'fp16':
        model.float()
        if precision == 'bf16':
            
            convert_weights_to_lp(model, dtype=torch.bfloat16)

    
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model
