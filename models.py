from attention_utils import LayerNorm, TransformerBlock, MOEBlock
import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

        # Now the transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        self.final_norm = LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=x.device))
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits # [batch_size, seq_len, vocab_size]

class GPT2MoEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

        # Now the transformer blocks
        self.moe_blocks = nn.Sequential(
            *[MOEBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        self.final_norm = LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
        self.aux_loss_coef = cfg["aux_loss_coef"]
        self.z_loss_coef = cfg.get("z_loss_coef", 0.0)  # older checkpoint configs lack it

    @property
    def aux_loss(self):
        # Load-balancing loss from the last forward pass, summed over layers
        return sum(block.moe.aux_loss for block in self.moe_blocks)

    @property
    def z_loss(self):
        # Router z-loss from the last forward pass, summed over layers
        return sum(block.moe.z_loss for block in self.moe_blocks)

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=x.device))
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.moe_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits # [batch_size, seq_len, vocab_size]