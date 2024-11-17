import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_layers import Block
from src.models.gpt2.config import GPT2Config
from src.utils import make_copies, transform_keys


""" 
Shape convention:
B: batch size
S: sequence length
E: embedding dimension (d_model)
H: number of attention heads
D: head dimension (E // H)
V: vocabulary size
F: feed-forward hidden dimension
"""

class GPT2Block(Block):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention.out_layer.NANOGPT_SCALE_INIT = 1
        self.mlp.out_layer.NANOGPT_SCALE_INIT = 1

class GPT2(nn.Module):
    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.model = nn.ModuleDict(dict(
            tok_embedding=nn.Embedding(config.vocab_size, config.d_model, **kwargs),
            pos_embedding=nn.Embedding(config.max_seq_len, config.d_model, **kwargs),
            blocks=nn.ModuleList([GPT2Block(config, **kwargs) for _ in range(config.num_layers)]),
            final_norm=nn.LayerNorm(config.d_model, **kwargs),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, **kwargs, bias=False)  # as in gpt2
        # self.lm_head.weight = self.model.tok_embedding.weight
        self.model.tok_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.d_model) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    @classmethod
    def from_pretrained(cls, model_name: str) -> 'GPT2':
        gpt2_configs = {
            'gpt2': dict(num_layers=12, num_heads=12, d_model=768),  # 124M params
            'gpt2-medium': dict(num_layers=24, num_heads=16, d_model=1024),  # 350M params
            'gpt2-large': dict(num_layers=36, num_heads=20, d_model=1280),  # 774M params
            'gpt2-xl': dict(num_layers=48, num_heads=25, d_model=1600),  # 1558M params
        }
        assert model_name in gpt2_configs.keys()
        config = gpt2_configs[model_name]
        config['max_seq_len'] = 1024
        config['vocab_size'] = 50257
        config['activation'] = 'gelu'
        config['hidden_dim'] = config['d_model'] * 4

        config = GPT2Config(**config)

        from transformers import GPT2LMHeadModel
        hf_state_dict = GPT2LMHeadModel.from_pretrained(model_name).state_dict()
        hf_state_dict = {k: v for k, v in hf_state_dict.items() if not k.endswith('.attn.bias')}
        gpt2 = GPT2(config)

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        hf_state_dict = {
            k: v.t() if any(w in k for w in transposed) else v for k, v in hf_state_dict.items()
        }
        hf_state_dict = transform_keys(hf_state_dict)

        gpt2.load_state_dict(hf_state_dict, strict=True)

        return gpt2


    def forward(self, idx_BS: torch.Tensor, targets_BS: torch.Tensor = None):
        """
        :param idx: torch.Tensor [batch_size, seq_len]
        :param targets torch.Tensor [batch_size, seq_len]
        :return: torch.Tensor [batch_size, seq_len, d_model]
        """
        B, S = idx_BS.size()
        pos_S = torch.arange(0, S, device=idx_BS.device, dtype=torch.long)
        
        x_BSE = self.model.pos_embedding(pos_S) + self.model.tok_embedding(idx_BS)
        for block in self.model.blocks:
            x_BSE = block(x_BSE)

        x_BSE = self.model.final_norm(x_BSE)
        logits_BSV = self.lm_head(x_BSE)
        loss = None
        if targets_BS is not None:
            loss = F.cross_entropy(logits_BSV.view(-1, logits_BSV.size(-1)), targets_BS.view(-1))
        return logits_BSV, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
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
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def sampling_loop(self, x_BS: torch.Tensor, max_length: int) -> torch.Tensor:
        while x_BS.size(1) < max_length:
            logits_BSV, _ = self(x_BS)
            logits_BV = logits_BSV[:, -1, :]

            probs_BV = F.softmax(logits_BV, dim=-1)
            top_probs_BK, top_indices_BK = torch.topk(probs_BV, 50, dim=-1)
            result_B1 = torch.multinomial(top_probs_BK, 1)
            next_token_B1 = torch.gather(top_indices_BK, -1, result_B1)
            x_BS = torch.cat([x_BS, next_token_B1], dim=1)

        return x_BS