"""LoRA / QLoRA

LoRA
--------
freeze W, learn a low-rank update  y = W x + (alpha/r) * B(A(dropout(x))),  A ~ kaiming, B = 0 (so training starts at the base model).


QLoRA
-------------
additionally store the frozen base Linear weights as 4-bit NF4 (blockwise absmax, dequantised on the fly) in pure PyTorch,
so it runs on MPS/CPU; on CUDA, bitsandbytes' Linear4bit is faster and is the standard choice.

The Quantization Algorithm
---------------------
We quantize the float x into q as follows:

q = clamp( round(x/S + Z), 0, 2^b - 1 )

here x is the original weight, S is the scale, Z is the zero point and b is the number of bits.

Then we can dequantize the weight by

x' = S * (q - Z)


"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear layers inside TransformerBlock: attention projections and the two MLP layers (FeedForward.layers.{0,2})
DEFAULT_TARGETS = ("W_q", "W_k", "W_v", "out_proj", "layers.0", "layers.2")
DEFAULT_MODULES_TO_SAVE = ("position_embedding",)

# NF4 codebook: quantiles of a standard normal rescaled to [-1, 1]
NF4_CODE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141506790,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0
    ]
)


class NF4Linear(nn.Module):
    """Frozen Linear whose weight is stored as packed 4-bit NF4 codes + per-block fp32 absmax."""

    def __init__(self, linear: nn.Linear, block_size=64):
        super().__init__()
        self.in_features, self.out_features, self.block_size = linear.in_features, linear.out_features, block_size
        w = linear.weight.detach().float().flatten()
        pad = (-w.numel()) % block_size
        w = F.pad(w, (0, pad)).view(-1, block_size)
        absmax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        code = NF4_CODE.to(w.device)
        idx = torch.bucketize(w / absmax, (code[1:] + code[:-1]) / 2).to(torch.uint8).flatten()  # nearest code
        self.register_buffer("packed", (idx[0::2] << 4) | idx[1::2])  # two 4-bit codes per byte
        self.register_buffer("absmax", absmax.flatten())
        self.register_buffer("code", code)
        self.numel = linear.weight.numel()
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def dequantize(self):
        idx = torch.stack([self.packed >> 4, self.packed & 0xF], dim=1).flatten().long()
        w = (self.code[idx].view(-1, self.block_size) * self.absmax[:, None]).flatten()[: self.numel]
        return w.view(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.dequantize().to(x.dtype), self.bias)


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)


def apply_lora(model, r=8, alpha=16, dropout=0.05, targets=DEFAULT_TARGETS, quantize=False,
               modules_to_save=DEFAULT_MODULES_TO_SAVE):
    """Freeze `model`, wrap matching Linears in LoRA (and NF4-quantise every block Linear if quantize). In place.

    Call after loading base weights. Stores the settings on model.lora_cfg for checkpointing.
    """
    for p in model.parameters():
        p.requires_grad = False
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            name = f"{parent_name}.{child_name}".lstrip(".")
            if not isinstance(child, nn.Linear) or not name.startswith(("transformer_blocks", "moe_blocks")):
                continue  # embeddings / out_head stay full precision and untouched
            if quantize:
                child = NF4Linear(child)
            if any(name.endswith(t) for t in targets):
                child = LoRALinear(child, r, alpha, dropout)
            setattr(parent, child_name, child)
    for name in modules_to_save:
        for p in model.get_submodule(name).parameters():
            p.requires_grad = True
    model.lora_cfg = {"r": r, "alpha": alpha, "dropout": dropout, "targets": list(targets), "quantize": quantize,
                      "modules_to_save": list(modules_to_save)}
    return model


def adapter_state_dict(model):
    keep = tuple(f"{m}." for m in model.lora_cfg["modules_to_save"])
    return {k: v for k, v in model.state_dict().items() if "lora_" in k or k.startswith(keep)}


def load_adapter_state_dict(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("lora_" in k for k in missing):
        raise ValueError(f"Adapter mismatch. unexpected={unexpected} missing_lora={[k for k in missing if 'lora_' in k]}")


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def merge_lora(model):
    """Fold adapters into plain nn.Linear weights (dequantising an NF4 base first) so the model is adapter-free."""
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                if isinstance(base, NF4Linear):
                    w = base.dequantize()
                    merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None, device=w.device)
                    merged.weight.copy_(w)
                    if base.bias is not None:
                        merged.bias.copy_(base.bias)
                    base = merged
                base.weight += child.scale * (child.lora_B @ child.lora_A)
                setattr(parent, child_name, base)
    model.lora_cfg = None
    return model
