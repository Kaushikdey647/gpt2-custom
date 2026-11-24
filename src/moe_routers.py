from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RouterOutput:
    """What an MoE layer needs from a router, for T flattened tokens, K experts, K' selected per token."""

    top_idx: torch.Tensor  # (T, K') long, experts chosen per token
    weights: torch.Tensor  # (T, K') combine weights for those experts
    probs: torch.Tensor  # (T, K) noise-free routing probabilities, used for the balancing loss and entropy
    logits: torch.Tensor  # (T, K) noise-free logits, used for the z-loss


class BaseRouter(nn.Module):
    """Base class for MoE routers. Subclasses implement `forward`: tokens -> expert assignment.

    Input: flattened tokens (T, D). The router only decides *which* experts get each token and with what
    weight; dispatching to experts and the aux/z losses stay in the MoE layer.
    T: number of tokens
    D: hidden dimension
    """

    def __init__(self, cfg):
        super().__init__()
        self.embedding_dim = cfg["embedding_dim"]
        self.num_experts = cfg["num_experts"]
        self.top_k = cfg["top_k"]

    def forward(self, flat: torch.Tensor) -> RouterOutput:
        raise NotImplementedError

class LinearRouter(BaseRouter):
    '''
    Linear router uses single linear projection with bias from the hidden dimension to the number of experts.
    Followed by a softmax activation
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.linear = nn.Linear(self.embedding_dim, self.num_experts, bias=True)

    def forward(self, flat: torch.Tensor) -> RouterOutput:
        logits = self.linear(flat)
        probs = torch.softmax(logits, dim=-1)
        return NotImplementedError

class AttentionRouter(BaseRouter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.linear = nn.Linear(self.embedding_dim, self.num_heads * self.head_dim, bias=True)

    def forward(self, flat: torch.Tensor) -> RouterOutput:
        logits = self.linear(flat)
        probs = torch.softmax(logits, dim=-1)
        return NotImplementedError

class MLPRouter(BaseRouter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_layers = cfg["num_layers"]
        self.layer_size = cfg["layer_size"]
        self.linear = nn.Linear(self.embedding_dim, self.layer_size, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.num_experts),
        )

    def forward(self, flat: torch.Tensor) -> RouterOutput:
        logits = self.linear(flat)
        probs = torch.softmax(logits, dim=-1)
        return NotImplementedError

class MLPHadamardRouter(BaseRouter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_layers = cfg["num_layers"]
        self.layer_size = cfg["layer_size"]
        self.linear = nn.Linear(self.embedding_dim, self.layer_size, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.num_experts),
        )

    def forward(self, flat: torch.Tensor) -> RouterOutput:
        logits = self.linear(flat)
        probs = torch.softmax(logits, dim=-1)
        return NotImplementedError

class HashRouter(BaseRouter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_experts = cfg["num_experts"]
        self.hash_size = cfg["hash_size"]
        self.hash_table = nn.ModuleList([nn.Linear(self.embedding_dim, self.hash_size) for _ in range(self.num_experts)])

    def forward(self, flat: torch.Tensor) -> RouterOutput:
        logits = self.hash_table(flat)
        probs = torch.softmax(logits, dim=-1)
        return NotImplementedError