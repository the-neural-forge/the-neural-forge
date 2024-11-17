import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal

class MLP(nn.Module):
    def __init__(self, d_model: int, activation: Literal['gelu', 'relu'], hidden_dim, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.in_layer = nn.Linear(d_model, hidden_dim, **kwargs)
        self.activation = nn.GELU(approximate='tanh') if activation == "gelu" else nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, d_model, **kwargs)
        self.out_layer.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor):
        """
        Args:
        :param x: Tensor, shape [batch_size, seq_len, d_model]
        :return: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = self.in_layer(x)
        x = self.out_layer(self.activation(x))
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.max_len = max_len

        pos_S1 = torch.arange(max_len, **kwargs).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, **kwargs) * (-math.log(10000.0) / embed_dim))

        pe_SE = torch.zeros(max_len, embed_dim, **kwargs)
        pe_SE[:, 0::2] = torch.sin(pos_S1 * div_term)
        pe_SE[:, 1::2] = torch.cos(pos_S1 * div_term)

        self.register_buffer('pe_1SE', pe_SE.unsqueeze(0))

    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        return x_BSE + self.pe_1SE[:, :x_BSE.size(1)]



class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.qkv_projection = nn.Linear(d_model, 3 * d_model, **kwargs)
        self.out_projection = nn.Linear(d_model, d_model, **kwargs)
        self.out_projection.NANOGPT_SCALE_INIT = 1

    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        B, S, E = x_BSE.size()

        # todo check speeds with torch.compile
        qkv_BSE3 = self.qkv_projection(x_BSE)
        qkv_BS3ND = qkv_BSE3.reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv_3BHSD = qkv_BS3ND.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, S, head_dim)
        query_BHSD, key_BHSD, value_BHSD = qkv_3BHSD[0], qkv_3BHSD[1], qkv_3BHSD[2]


        out_BHSD = F.scaled_dot_product_attention(query_BHSD, key_BHSD, value_BHSD, is_causal=True)
        out_BSE = out_BHSD.transpose(1, 2).contiguous().view(B, S, E)
        out_BSE = self.out_projection(out_BSE)

        return out_BSE


class Block(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.activation = config.activation
        self.hidden_dim = config.hidden_dim

        self.layer_norm1 = nn.LayerNorm(self.d_model, **kwargs)
        self.attention = MultiHeadedAttention(self.d_model, self.num_heads, **kwargs)
        self.layer_norm2 = nn.LayerNorm(self.d_model, **kwargs)
        self.mlp = MLP(self.d_model, self.activation, self.hidden_dim, **kwargs)

    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        """
        :param x_BSE: torch.Tensor [batch_size, seq_len, d_model]
        :return: torch.Tensor [batch_size, seq_len, d_model]
        """
        x_BSE = x_BSE + self.attention(self.layer_norm1(x_BSE))
        x_BSE = x_BSE + self.mlp(self.layer_norm2(x_BSE))
        return x_BSE    