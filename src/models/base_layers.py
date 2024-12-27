import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal

class FeedForward(nn.Module):
    def __init__(self, d_model: int, activation: Literal['gelu', 'relu'], hidden_dim, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.in_layer = nn.Linear(d_model, hidden_dim, **kwargs)
        self.activation = nn.GELU(approximate='tanh') if activation == "gelu" else nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, d_model, **kwargs)

    def forward(self, x: torch.Tensor):
        """
        Args:
        :param x: Tensor, shape [batch_size, seq_len, d_model]
        :return: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = self.in_layer(x)
        x = self.out_layer(self.activation(x))
        return x

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.in_layer = nn.Linear(d_model, 2 * hidden_dim, **kwargs)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, d_model, **kwargs)

    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        x_BS2H = self.in_layer(x_BSE)
        x_BS2H = x_BS2H.chunk(2, dim=-1)
        x_BSH = x_BS2H[0] * self.activation(x_BS2H[1])
        x_BSE = self.out_layer(x_BSH)
        return x_BSE

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


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, q_heads: int, kv_heads: int, head_dim: int, **kwargs):
        super().__init__(**kwargs)

        assert d_model % q_heads == 0, "d_model must be divisible by q_heads"
        assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads"
        self.d_model = d_model
        self.q_heads = q_heads
        self.kv_heads = kv_heads

        self.head_dim = head_dim

        qkv_dim = (self.q_heads + 2 * self.kv_heads) * self.head_dim
        self.qkv_projection = nn.Linear(d_model, qkv_dim, **kwargs)
        self.out_projection = nn.Linear(self.q_heads * self.head_dim, d_model, **kwargs)


    def _repeat_kv(self, key_BKSH: torch.Tensor, value_BKSH: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_BQSH = key_BKSH.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        value_BQSH = value_BKSH.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        return key_BQSH, value_BQSH



    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        B, S, _ = x_BSE.size()

        q_size = self.q_heads * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        query_BSQ, key_BSK, value_BSK = self.qkv_projection(x_BSE).chunk([q_size, kv_size, kv_size], dim=-1)
        query_BQSH = query_BSQ.view(B, S, self.q_heads, self.head_dim).transpose(1, 2)
        key_BKSH = key_BSK.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        value_BKSH = value_BSK.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

        key_BQSH, value_BQSH = self._repeat_kv(key_BKSH, value_BKSH)

        attn_BQSH = F.scaled_dot_product_attention(query_BQSH, key_BQSH, value_BQSH, is_causal=True)
        attn_BSQH = attn_BQSH.transpose(1, 2).contiguous().view(B, S, self.q_heads * self.head_dim)
        attn_BSE = self.out_projection(attn_BSQH)

        return attn_BSE



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
        self.mlp = FeedForward(self.d_model, self.activation, self.hidden_dim, **kwargs)

    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        """
        :param x_BSE: torch.Tensor [batch_size, seq_len, d_model]
        :return: torch.Tensor [batch_size, seq_len, d_model]
        """
        x_BSE = x_BSE + self.attention(self.layer_norm1(x_BSE))
        x_BSE = x_BSE + self.mlp(self.layer_norm2(x_BSE))
        return x_BSE