import numpy as np
import os
import safetensors.torch as safetorch
import torch

def create_test_dataset(output_dir: str, total_tokens: int = 4_000_000_000, tokens_per_shard: int = 100_000_000):
    """
    Create test dataset with consecutive tokens spread across multiple shards
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating {total_tokens} tokens in {output_dir}...")
    
    num_shards = total_tokens // tokens_per_shard
    print(f"Creating {num_shards} shards with {tokens_per_shard} tokens each...")
    
    for shard_idx in range(num_shards):
        # Generate consecutive tokens for this shard using uint32
        start_idx = shard_idx * tokens_per_shard
        tokens = torch.arange(start_idx, start_idx + tokens_per_shard, dtype=torch.int64)
        
        # Save as safetensor file
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:03d}.safetensors")
        safetorch.save_file({'tokens': tokens}, shard_path)
        
        if (shard_idx + 1) % 5 == 0:
            print(f"Created {shard_idx + 1}/{num_shards} shards") 

    print(f"Created {num_shards} shards")

if __name__ == "__main__":
    create_test_dataset("test_data")