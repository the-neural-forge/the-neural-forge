import torch
import numpy as np
from dataset import TextDataLoader

def test_full_dataset_coverage():
    from tqdm import tqdm
    
    B = 4  # batch size
    S = 1024  # sequence length
    num_processes = 8
    
    # Create numpy arrays to track seen tokens for each process (using uint32)
    seen_tokens = [np.zeros(4_000_000_000, dtype=np.bool_) for _ in range(num_processes)]
    
    # Create dataloaders for each process
    loaders = [
        TextDataLoader(
            B=B, 
            S=S, 
            data_dir="test_data",
            seed=42,
            process_rank=i,
            num_processes=num_processes
        ) for i in range(num_processes)
    ]
    
    # Collect all tokens from each process
    for proc_idx, loader in enumerate(loaders):
        tokens_seen = 0
        pbar = tqdm(range(len(loader)), desc=f"Process {proc_idx}")
        for _ in pbar:
            inputs, targets = loader.get_item()
            # Mark tokens as seen in both inputs and the last token of targets
            tokens = inputs.numpy().flatten()
            seen_tokens[proc_idx][tokens] = True
            
            # Get the last token from each sequence in targets
            last_tokens = targets[:, -1].numpy()
            seen_tokens[proc_idx][last_tokens] = True
            
            tokens_seen += len(tokens) + len(last_tokens)
            pbar.set_postfix({'tokens': f'{tokens_seen:,}'})
            
        print(f"Process {proc_idx}: Processed {tokens_seen:,} tokens")
    
    # Combine all seen tokens
    combined_seen = np.zeros(4_000_000_000, dtype=np.bool_)
    for seen in seen_tokens:
        combined_seen |= seen
    
    # Check coverage
    total_seen = np.sum(combined_seen)
    missing_count = 4_000_000_000 - total_seen
    
    print(f"\nResults:")
    print(f"Total tokens seen: {total_seen:,}")
    print(f"Missing tokens: {missing_count:,}")
    
    # Find some missing tokens
    if missing_count > 0:
        missing_indices = np.where(~combined_seen)[0]
        print("\nSample of missing tokens:")
        print(f"First 10 missing tokens: {missing_indices[:10]}")
        if len(missing_indices) > 1000000:
            print(f"Some tokens around 1M mark: {missing_indices[1000000:1000010]}")
        if len(missing_indices) > 2000000:
            print(f"Some tokens around 2M mark: {missing_indices[2000000:2000010]}")
    
    assert missing_count == 0, f"Missing {missing_count:,} tokens"
    
    # Verify distribution across processes
    process_token_counts = [np.sum(tokens) for tokens in seen_tokens]
    token_count_std = np.std(process_token_counts)
    token_count_mean = np.mean(process_token_counts)
    cv = token_count_std / token_count_mean  # coefficient of variation
    
    print(f"\nToken distribution across processes:")
    for i, count in enumerate(process_token_counts):
        print(f"Process {i}: {count:,} tokens")
    print(f"Coefficient of variation: {cv:.4f}")
    
    # Check that token distribution is relatively even (CV < 0.1 means less than 10% variation)
    assert cv < 0.1, f"Uneven token distribution across processes (CV={cv:.4f})"
if __name__ == "__main__":
    test_full_dataset_coverage()