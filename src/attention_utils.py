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

class Expert(nn.Module):
    '''
    A single expert

    Lets assume the input is X, of dim (B, N, D)
    U(X) = W_u * X, U(X) of dim (B, N, D) This is the uplink layer.

    G(X) = W_g * X, G(X) of dim (B, N, D) This is the gate layer.
    Then we pass this Gate layer through a SiLU activation function to give us
    G(X) = SiLU( G(X) )

    D(X) = W_d *( G(X) * U(X) ), D(X) of dim (B, N, D) This is the downlink layer.
    '''

    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["embedding_dim"]
        hidden_dim = cfg["expert_hidden_dim"]
        self.W_u = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W_g = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W_d = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.W_d(
            self.silu(
                self.W_g(x)
            ) * self.W_u(x)
        )

class MoeLayer(nn.Module):
    '''
    Mixture of Experts Layer

    This will come after the MultiHeadAttention Layer in the TransformerBlock in MoE models.

    Lets assume there are K experts. Each expert is a linear layer.

    Input: (batch_size, sequence_length, embedding_dim) or just (B, N, D)
    The input first goes to a linear layer called the gate. W_g, (D, K)
    H(X) = W_g * X, H(X) of dim (B, N, K) gives us weights that we softmax to get the probabilities for each expert.
    But to give all the experts a fair chance, during training we add fixed-std gaussian noise to the logits. Hence,
    H(X) = W_g * X + n, n ~ N(0, noise_std^2) of dim (B, N, K). The noise is not used in the load-balancing loss.

    Now let K' be the number of experts that are selected. We take the top K' weights and use them to select the experts.
    Then we pass the input to the softmax, to get the probabilities for each expert in K'
    G(X) = softmax( H'(X) ) where H'(X) has all other values set to -inf
    Then we pass the input to the experts in K' and get the output.
    O(X) = sum( G(X) * O_i(X) ) where O_i(X) is the output of the i-th expert.
    '''

    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["num_experts"]
        self.top_k = cfg["top_k"]
        self.noise_std = cfg["gate_noise_std"]
        self.gate = nn.Linear(cfg["embedding_dim"], self.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(self.num_experts)])
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)  # follows .to(device)
        self.register_buffer("z_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("expert_frac", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("router_entropy", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)  # (T, D)

        clean_logits = self.gate(flat)  # (T, K)
        logits = clean_logits
        if self.training and self.noise_std > 0:
            logits = clean_logits + torch.randn_like(clean_logits) * self.noise_std

        top_vals, top_idx = logits.topk(self.top_k, dim=-1)  # (T, K')
        weights = torch.softmax(top_vals, dim=-1)  # (T, K')

        # Sort (token, slot) assignments by expert so each expert gets one contiguous chunk.
        # A single .tolist() replaces the per-expert nonzero()/numel() host syncs.
        flat_idx = top_idx.flatten()  # (T*K',)
        order = flat_idx.argsort()
        counts = torch.bincount(flat_idx, minlength=self.num_experts)
        token_idx = order // self.top_k
        expert_in = flat[token_idx].split(counts.tolist())
        expert_out = torch.cat([expert(chunk) for expert, chunk in zip(self.experts, expert_in)])
        out = torch.zeros_like(flat).index_add_(
            0, token_idx, expert_out * weights.flatten()[order].unsqueeze(-1)
        )

        # Switch-style load balancing loss: K * sum_i(fraction routed to i * mean router prob of i)
        probs = torch.softmax(clean_logits, dim=-1)  # noise-free so the balancing signal isn't noisy
        frac = counts.float() / flat_idx.numel()
        self.aux_loss = self.num_experts * (frac * probs.mean(0)).sum()
        # Router z-loss (ST-MoE): penalises large router logits for stability
        self.z_loss = torch.logsumexp(clean_logits, dim=-1).pow(2).mean()
        # Detached routing stats for logging only
        self.expert_frac = frac.detach()
        self.router_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean().detach()

        return out.view(batch_size, seq_len, dim)

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

class MOEBlock(nn.Module):
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
        self.moe = MoeLayer(cfg)
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
        x = self.moe(x)
        x = self.drop_shortcut(x)
        return x + shortcut