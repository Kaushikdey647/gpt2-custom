"""Sweep context_length x batch_size and report the throughput optimum on this device.

Objective: minimise seconds per token (= maximise tokens/s) of a real train step
(fwd + bwd + clip + AdamW). Pure max over a grid always favours the biggest config
that fits, so we also report the "knee": the smallest tokens/step within --tol of best.
Random tokens are used; throughput does not depend on the data.
"""
import argparse
import gc
import time

import torch

from src.utils import build_model, calc_loss_batch, get_device

BASE_CONFIG = {
    "vocab_size": 50257,
    "embedding_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "dropout_rate": 0.1,
    "qkv_bias": False,
    "use_moe": True,
    "num_experts": 8,
    "top_k": 2,
    "expert_hidden_dim": 768,
    "gate_noise_std": 0.1,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def bench(ctx, bs, device, steps, warmup):
    """Seconds per train step, or None if it doesn't fit in memory."""
    cfg = {**BASE_CONFIG, "context_length": ctx}
    try:
        model = build_model(cfg).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randint(0, cfg["vocab_size"], (bs, ctx), device=device)
        y = torch.randint(0, cfg["vocab_size"], (bs, ctx), device=device)

        def step():
            opt.zero_grad()
            calc_loss_batch(x, y, model, device).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        for _ in range(warmup):
            step()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        sync(device)
        return (time.perf_counter() - t0) / steps
    except RuntimeError as e:  # CUDA raises OutOfMemoryError (a RuntimeError); MPS a plain one
        if "out of memory" not in str(e).lower():
            raise
        return None
    finally:
        model = opt = x = y = None
        free(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--bs", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--tol", type=float, default=0.05, help="knee = smallest tokens/step within this fraction of best")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"{'ctx':>5} {'batch':>6} {'tok/step':>9} {'s/step':>8} {'tok/s':>9}")

    # Group by tokens/step (ascending); within a group go bs-descending (8x64, 4x128, 2x256).
    # Every shape of one product is tried before moving to the next, so a hang/OOM late in the sweep loses little.
    products = sorted({c * b for c in args.ctx for b in args.bs})
    results = []  # (ctx, bs, tokens_per_step, tok_per_s)
    oom_ctx = set()  # ctx values where some batch already OOMed; larger batches there will too
    for prod in products:
        group = sorted(((c, b) for c in args.ctx for b in args.bs if c * b == prod), key=lambda cb: -cb[1])
        group = [(c, b) for c, b in group if c not in oom_ctx]
        if not group:
            break  # every shape at this size is known to OOM; larger products will too
        for ctx, bs in group:
            dt = bench(ctx, bs, device, args.steps, args.warmup)
            if dt is None:
                print(f"{ctx:>5} {bs:>6} {ctx * bs:>9}      OOM", flush=True)
                oom_ctx.add(ctx)
                continue
            tps = ctx * bs / dt
            results.append((ctx, bs, ctx * bs, tps))
            print(f"{ctx:>5} {bs:>6} {ctx * bs:>9} {dt:>8.3f} {tps:>9.0f}", flush=True)

    if not results:
        raise SystemExit("Nothing fit in memory.")

    best = max(results, key=lambda r: r[3])
    print(f"\nMin s/token: ctx={best[0]} batch={best[1]} ({best[3]:.0f} tok/s)")
    knee = min((r for r in results if r[3] >= (1 - args.tol) * best[3]), key=lambda r: r[2])
    print(f"Knee (within {args.tol:.0%} of best, fewest tokens/step): ctx={knee[0]} batch={knee[1]} ({knee[3]:.0f} tok/s)")
    print("Note: ctx also changes the model/task; pick it for modelling reasons, then tune batch for speed.")


if __name__ == "__main__":
    main()
