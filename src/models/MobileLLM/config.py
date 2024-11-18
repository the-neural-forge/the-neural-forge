from dataclasses import dataclass

@dataclass
class MobileLLMConfig:
    vocab_size: int
    d_model: int
    hidden_dim: int
    num_layers: int
    num_repeats: int
    q_heads: int
    kv_heads: int
    head_dim: int
    
