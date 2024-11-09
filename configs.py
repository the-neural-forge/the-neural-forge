from dataclasses import dataclass


@dataclass
class GPTConfig:
    max_seq_len: int = 1024
    vocab_size: int = 50257
    d_model: int = 768
    num_layers: int = 12
    activation: str = 'gelu'
    hidden_dim: int = 4 * d_model
    num_heads: int = 12
