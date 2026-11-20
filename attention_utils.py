import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, input_length, num_heads, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.input_length = input_length
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_k = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_v = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.scale = torch.sqrt(torch.tensor(self.head_dim))
        self.register_buffer("mask", torch.triu(torch.ones(input_length, input_length), diagonal=1))

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        keys = self.W_k(x)
        values = self.W_v(x)
        queries = self.W_q(x)

        keys = keys.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores.masked_fill_(self.mask.bool()[:sequence_length, :sequence_length], -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.scale, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        context_vector = context_vector.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.output_dim
        )
        return self.out_proj(context_vector)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["embedding_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dim=cfg["embedding_dim"],
            output_dim=cfg["embedding_dim"],
            input_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        return x + shortcut
