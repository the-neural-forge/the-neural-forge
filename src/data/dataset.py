import tiktoken
import torch
import safetensors.torch as safetorch
import os
import numpy as np


def _load_tokens(filename: str) -> torch.Tensor:
    return safetorch.load_file(filename)['tokens']


class TextDataLoader:
    def __init__(self, B: int, S: int, data_dir: str = None, seed: int = None, single_file: str = None, process_rank: int = 0, num_processes: int = 1):
        """
        Initialize TextDataLoader with either directory of shards or single file mode

        Args:
            B: Batch size
            S: Sequence length
            data_dir: Directory containing data shards (optional if using single file)
            seed: Random seed
            single_file: Path to single text file (optional)
        """
        self.B = B
        self.S = S
        self.batch_size = self.B * self.S + 1
        self.rand_generator = np.random.default_rng(seed)
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Single file mode setup
        self.single_file_mode = single_file is not None
        if self.single_file_mode:
            if num_processes > 1:
                raise ValueError("Single file mode does not support multiprocessing")

            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.tokens = self._load_single_file(single_file)
            self.tokens_total = len(self.tokens)
            self.batches_total = self.tokens_total // self.batch_size
            self.batch_indices = np.arange(self.batches_total)
            self.rand_generator.shuffle(self.batch_indices)
            self.batch_idx = 0

        # Sharded mode setup
        else:
            if data_dir is None:
                raise ValueError("Must provide either data_dir or single_file")
            self.data_dir = data_dir
            self.shards = os.listdir(self.data_dir)
            self.num_shards = len(self.shards)
            self.shard_indices = np.arange(self.num_shards)
            self.rand_generator.shuffle(self.shard_indices)
            self.tokens_per_shard = None
            self.batches_per_shard = None
            self.batch_indices = None
            self.tokens = None
            self.shard_idx = process_rank
            self.batch_idx = 0

            self._load_shard()


    def _load_single_file(self, file_path: str) -> torch.Tensor:
        """Load and tokenize a single text file or load safetensor file"""
        try:
            if file_path.endswith('.safetensors'):
                tokens = _load_tokens(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

            if len(tokens) < self.batch_size:
                raise ValueError(f"File {file_path} is too small for the requested batch size")
            return tokens

        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")
        
    def _load_shard(self):
        """Load next shard in sharded mode"""

        if self.shard_idx >= self.num_shards:
            self.shard_idx = self.process_rank
            self.rand_generator.shuffle(self.shard_indices)

        # Use shard_indices to get the shuffled shard index
        shard_path = os.path.join(self.data_dir, self.shards[self.shard_indices[self.shard_idx]])
        self.tokens = _load_tokens(shard_path)
        self.tokens_per_shard = self.tokens.size(0)
        self.batches_per_shard = self.tokens_per_shard // self.batch_size
        self.batch_indices = np.arange(self.batches_per_shard)

        self.rand_generator.shuffle(self.batch_indices)
        self.shard_idx += self.num_processes

    def get_item(self):
        """Get next batch in sharded mode"""
        if self.single_file_mode:
            raise ValueError("Use get_single_file_item() in single file mode")

        if self.batch_idx == self.batches_per_shard:
            self.batch_idx = 0
            self._load_shard()

        start_idx = self.batch_indices[self.batch_idx] * self.batch_size
        buffer = self.tokens[start_idx: start_idx + self.batch_size]
        inputs = buffer[:-1].view(self.B, self.S)
        targets = buffer[1:].view(self.B, self.S)

        self.batch_idx += 1
        return inputs, targets

    def get_single_file_item(self):
        """Get next batch in single file mode"""
        if not self.single_file_mode:
            raise ValueError("Use get_item() in sharded mode")

        # Each process works on its own subset of batches
        if self.batch_idx >= self.batches_total // self.num_processes:
            self.batch_idx = 0
            # Shuffle indices for this process's subset
            process_indices = self.batch_indices[self.process_rank::self.num_processes]
            self.rand_generator.shuffle(process_indices)
            self.batch_indices[self.process_rank::self.num_processes] = process_indices

        # Calculate global batch index for this process
        global_batch_idx = self.process_rank + (self.batch_idx * self.num_processes)
        start_idx = self.batch_indices[global_batch_idx] * self.batch_size
        buffer = self.tokens[start_idx: start_idx + self.batch_size]
        inputs = buffer[:-1].view(self.B, self.S)
        targets = buffer[1:].view(self.B, self.S)

        self.batch_idx += 1
        return inputs.pin_memory(), targets.pin_memory()
    
    def reset(self):
        if self.single_file_mode:
            self.batch_idx = 0
        else:
            self.batch_idx = 0
            self._load_shard()

    def __len__(self):
        if self.single_file_mode:
            return self.batches_total // self.num_processes
        else:
            return self.num_shards * self.batches_per_shard // self.num_processes


