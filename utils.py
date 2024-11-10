import math
import random
from typing import List, Dict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import re


def make_copies(module: nn.Module, num_copies: int) -> List[nn.Module]:
    return [deepcopy(module) for _ in range(num_copies)]


def transform_key(key):
    name_map = {
        'wte': 'tok_embedding',
        'wpe': 'pos_embedding',
        'h': 'blocks',
        'ln_1': 'layer_norm1',
        'ln_2': 'layer_norm2',
        'attn.c_proj': 'attention.out_projection',
        'attn': 'attention',
        'c_attn': 'qkv_projection',
        'mlp.c_fc': 'mlp.in_layer',
        'mlp.c_proj': 'mlp.out_layer',
        'ln_f': 'final_norm'
    }

    # Replace 'transformer.' with 'model.'
    new_key = key.replace('transformer.', 'model.')

    # Replace names using the mapping
    for old, new in name_map.items():
        new_key = new_key.replace(f'.{old}.', f'.{new}.')

    return new_key
def transform_keys(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {transform_key(k): v for k, v in d.items()}


def configure_optimizers(model, weight_decay, learning_rate, device_type):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
    return optimizer

def get_lr(warmup_steps, total_steps, max_lr, min_lr, actual_step):
    if actual_step < warmup_steps:
        # Linear warmup from min_lr to max_lr
        return min_lr + (max_lr - min_lr) * (actual_step / warmup_steps)
    elif actual_step >= total_steps:
        return min_lr
    else:
        # Cosine decay from max_lr to min_lr
        decay_ratio = (actual_step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr-min_lr) * (1 + math.cos(decay_ratio * math.pi))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)




