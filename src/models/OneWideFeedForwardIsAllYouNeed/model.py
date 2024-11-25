from src.models.base_layers import FeedForward, Block


class OneFFNBlock(Block):
    def __init__(self, config, feed_forward: FeedForward, **kwargs):
        super().__init__(config, **kwargs)

        self.ffn = feed_forward


class OneFFNModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.blocks = nn.ModuleList([OneFFNBlock(config, FeedForward(config.d_model, config.activation, config.hidden_dim), **kwargs) for _ in range(config.num_layers)])