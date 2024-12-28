import os
import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def make_copies(module: nn.Module, num_copies: int) -> list[nn.Module]:
    return [deepcopy(module) for _ in range(num_copies)]


def transform_key(key: str) -> str:
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
def transform_keys(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {transform_key(k): v for k, v in d.items()}


def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, device_type: str) -> torch.optim.Optimizer:
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

def get_lr(warmup_steps: int, total_steps: int, max_lr: float, min_lr: float, actual_step: int) -> float:
    if actual_step < warmup_steps:
        # Linear warmup from min_lr to max_lr
        return min_lr + (max_lr - min_lr) * (actual_step / warmup_steps)
    elif actual_step >= total_steps:
        return min_lr
    else:
        # Cosine decay from max_lr to min_lr
        decay_ratio = (actual_step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr-min_lr) * (1 + math.cos(decay_ratio * math.pi))

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def find_max_batch_size(model: nn.Module, optimizer: torch.optim.Optimizer, seq_len: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.train()
    model.to(device)

    torch.cuda.empty_cache()
    input_ids = torch.randint(0, 10000, (1, seq_len), device=device)
    target_ids = torch.randint(0, 10000, (1, seq_len), device=device)
    batch_size = 1
    max_successful = 1
    
    # First try powers of 2
    while True:
        try:
            batch_input = input_ids.expand(batch_size, -1)
            _, loss = model(batch_input, target_ids)
            loss.backward()
            optimizer.zero_grad()
            
            max_successful = batch_size
            batch_size *= 2
            
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            break
            
    # Binary search between last successful and failed batch size
    left = max_successful
    right = batch_size
    
    while left < right - 1:
        mid = (left + right) // 2
        try:
            batch_input = input_ids.expand(mid, -1) 
            _, loss = model(batch_input, target_ids)
            loss.backward()
            optimizer.zero_grad()
            
            left = mid  # This size worked
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            right = mid  # This size failed
            torch.cuda.empty_cache()

    return left  # Return largest successful batch size

def measure_throughput(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    seq_len: int, 
    use_compile: bool = False, 
) -> list[float]:
    # Initialize DDP settings
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ.get('RANK'))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
        ddp_world_size = int(os.environ.get('WORLD_SIZE'))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    else:
        ddp_rank = ddp_local_rank = 0
        ddp_world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    master_process = ddp_rank == 0
    
    if master_process:
        print(f"Using device: {device}")
    
    # Setup model
    torch.set_float32_matmul_precision('high')
    model.train().to(device)
    
    if use_compile:
        model = torch.compile(model, mode="max-autotune", dynamic=False, fullgraph=True)
        
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Measure throughput
    max_minibatch_size = find_max_batch_size(model, optimizer, seq_len)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    throughputs = []
    batch_size = 1
    
    while batch_size <= max_minibatch_size:
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        target_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        
        # Warmup run
        for _ in range(10):
            _, loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.zero_grad()
        
        # Timed run
        start.record()
        for _ in range(10):
            optimizer.zero_grad()
            _, loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.step()
        end.record()
        
        torch.cuda.synchronize()

        throughput = batch_size / (end.elapsed_time(start) / 1000) * 10
        throughputs.append(throughput * ddp_world_size if ddp else throughput)
        
        batch_size *= 2
    
    if ddp:
        destroy_process_group()
    
    for throughput in throughputs:
        print(f"Throughput: {throughput:.2f} tokens/s")
    return throughputs





