from src.models.base_layers import SwiGLUFeedForward, GroupedQueryAttention
from src.models.MobileLLM.config import MobileLLMConfig
from torch import nn
import torch

class MobileLLMBlock(nn.Module):
    def __init__(self, config: MobileLLMConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.layer_norm1 = nn.LayerNorm(config.d_model, **kwargs)
        self.attention = GroupedQueryAttention(config.d_model, config.q_heads, config.kv_heads, config.head_dim, **kwargs)
        self.layer_norm2 = nn.LayerNorm(config.d_model, **kwargs)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.hidden_dim, **kwargs)

    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        x_BSE = x_BSE + self.attention(self.layer_norm1(x_BSE))
        x_BSE = x_BSE + self.feed_forward(self.layer_norm2(x_BSE))
        return x_BSE
    

class MobileLLM(nn.Module):
    def __init__(self, config: MobileLLMConfig, **kwargs):
        super().__init__(**kwargs)


        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **kwargs)
        self.blocks = nn.ModuleList([MobileLLMBlock(config, **kwargs) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model, **kwargs)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, **kwargs)

    def forward(self, x_BSE: torch.Tensor) -> torch.Tensor:
        x_BSE = self.embedding(x_BSE)
        for _ in range(self.config.num_repeats):
            for block in self.blocks:
                x_BSE = block(x_BSE)
        x_BSE = self.final_norm(x_BSE)
        x_BSE = self.lm_head(x_BSE)
        return x_BSE
        

