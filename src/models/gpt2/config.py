from dataclasses import dataclass
import os


@dataclass
class GPT2Config:
    max_seq_len: int = int(os.getenv('MAX_SEQ_LEN', 1024))
    vocab_size: int = int(os.getenv('VOCAB_SIZE', 50257))
    d_model: int = int(os.getenv('D_MODEL', 768))
    num_layers: int = int(os.getenv('NUM_LAYERS', 12))
    activation: str = os.getenv('ACTIVATION', 'gelu')
    hidden_dim: int = int(os.getenv('HIDDEN_DIM', 4 * d_model))
    num_heads: int = int(os.getenv('NUM_HEADS', 12))
