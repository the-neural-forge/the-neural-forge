import torch
import torch.nn as nn

class HyperConnections(nn.Module):
    def __init__(self, expansion_rate: int, layer_idx: int, block: nn.Module, dynamic: bool, d_model: int, **kwargs):
        super().__init__(**kwargs)

        self.block = block

        self.dynamic = dynamic
        self.expansion_rate = expansion_rate
        self.d_model = d_model

        a_m = torch.zeros((expansion_rate, 1))
        a_m.data[layer_idx % expansion_rate] = 1.0

        a_r = torch.eye(expansion_rate)

        self.static_alpha = torch.cat([a_m, a_r], dim=1)
        self.static_beta = nn.Parameter(torch.ones(expansion_rate))

        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(d_model, expansion_rate + 1))
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d_model))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing hyper-connections.
        
        Args:
            h (torch.Tensor): Input tensor of shape [batch, seq_len, rate, dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, rate, dim]
        """
        # Width connections
        mix_h, beta = self._width_connection(h)
        
        # Extract input for current layer
        h_0 = mix_h[..., 0, :]
        
        # Apply layer transformation (to be implemented by parent module)  
        h_o = self.transform_fn(h_0)
        
        # Depth connections
        out = self._depth_connection(mix_h, h_o, beta)
        
        return out

    def _width_connection(self, h: torch.Tensor):
        """Compute width connections (WC matrix in paper)"""
        if self.dynamic:
            norm_h = self.layer_norm(h)
            
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = torch.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
            
            dc_weight = norm_h @ self.dynamic_beta_fn 
            dc_weight = torch.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]
            beta = self.static_beta[None, None, ...]

        mix_h = alpha.transpose(-1, -2) @ h
        
        return mix_h, beta

    def _depth_connection(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor):
        """Compute depth connections"""
        h = beta[..., None] * h_o[..., None, :] + mix_h[..., 1:, :]
        return h        





    

