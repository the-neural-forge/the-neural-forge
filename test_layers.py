import torch

from configs import GPTConfig
from layers import PositionalEncoding, MLP, MultiHeadedAttention, Block, GPT2


def test_positional_encoding():
    sequence_length = 128
    batch_size = 16
    embedding_dim = 64

    x = torch.randn((sequence_length, batch_size, embedding_dim))
    pe = PositionalEncoding(embedding_dim)
    y = pe(x)

    assert x.size() == y.size()


def test_mlp():
    sequence_length = 128
    batch_size = 16
    embedding_dim = 64
    hidden_dim = 4 * embedding_dim
    activation = 'gelu'

    x = torch.randn((sequence_length, batch_size, embedding_dim))
    mlp = MLP(embedding_dim, activation, hidden_dim)
    y = mlp(x)

    assert x.size() == y.size()


def test_multi_headed_attention():
    sequence_length = 128
    batch_size = 16
    embedding_dim = 64
    num_heads = 8

    x = torch.randn((sequence_length, batch_size, embedding_dim))
    attention = MultiHeadedAttention(embedding_dim, num_heads)

    y = attention(x)
    assert x.size() == y.size()


def test_block():
    sequence_length = 128
    batch_size = 16
    embedding_dim = 64
    num_heads = 8
    config = GPTConfig(max_seq_len=sequence_length, d_model=embedding_dim, num_heads=num_heads)

    x = torch.randn((sequence_length, batch_size, embedding_dim))
    block = Block(config)

    y = block(x)

    assert x.size() == y.size()

def test_gpt():
    sequence_length = 128
    batch_size = 16
    embedding_dim = 64
    num_heads = 8
    config = GPTConfig(max_seq_len=sequence_length, d_model=embedding_dim, num_heads=num_heads)
    model = GPT2(config)

    x = torch.randint(0, 128, (batch_size, sequence_length))

    y, _ = model(x)
    assert y.size(0) == x.size(0)
    assert y.size(1) == x.size(1)
    assert y.size(2) == model.config.vocab_size




