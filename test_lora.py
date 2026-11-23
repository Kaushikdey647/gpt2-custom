"""Sanity checks for lora.py: `uv run python test_lora.py`."""
import copy

import torch

from src.lora import NF4Linear, adapter_state_dict, apply_lora, merge_lora, trainable_params
from src.models import GPTModel

CFG = dict(vocab_size=100, context_length=16, embedding_dim=64, num_heads=4, num_layers=2, dropout_rate=0.0, qkv_bias=False)


def _model():
    torch.manual_seed(0)
    return GPTModel(CFG).eval()


def test_starts_identical_and_only_adapters_train():
    base = _model()
    x = torch.randint(0, 100, (2, 16))
    ref = base(x)
    lora = apply_lora(copy.deepcopy(base), r=4, alpha=8, dropout=0.0).eval()
    assert torch.allclose(lora(x), ref, atol=1e-6)  # B = 0 -> no change at init
    names = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n or n.startswith("position_embedding") for n in names)


def test_merge_matches_adapter_output():
    lora = apply_lora(_model(), r=4, alpha=8, dropout=0.0).eval()
    for p in trainable_params(lora):
        torch.nn.init.normal_(p, std=0.1)  # make the adapters non-trivial
    x = torch.randint(0, 100, (2, 16))
    before = lora(x)
    assert torch.allclose(merge_lora(lora)(x), before, atol=1e-4)


def test_nf4_roundtrip_error_and_qlora_trains():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 128)
    q = NF4Linear(lin)
    rel = (q.dequantize() - lin.weight).norm() / lin.weight.norm()
    assert rel < 0.12, rel  # NF4 on Gaussian-ish weights: ~10% relative error
    assert q.packed.numel() * 2 == lin.weight.numel() and q.packed.dtype == torch.uint8
    model = apply_lora(_model(), r=4, alpha=8, quantize=True)
    x = torch.randint(0, 100, (2, 16))
    model(x).sum().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if "lora_B" in n)
    assert set(adapter_state_dict(model)) <= set(model.state_dict())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
