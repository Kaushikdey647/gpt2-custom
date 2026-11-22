import math
import os
import signal
import sys
import time
import termios

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from matplotlib.ticker import MaxNLocator
import wandb
from tqdm import tqdm

from models import GPT2MoEModel, GPTModel
from text_dataset import IGNORE_INDEX


def build_model(cfg):
    return (GPT2MoEModel if cfg.get("use_moe") else GPTModel)(cfg)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_wandb(project, model_name, config, no_wandb=False, tags=(), extra_config=None):
    """One project per train script; id=model_name so resuming a checkpoint resumes its W&B run.

    Rename the model for a new version (also starts a fresh checkpoint). Mode None honours WANDB_MODE.
    """
    return wandb.init(
        project=project,
        id=model_name,
        name=model_name,
        resume="allow",
        mode="disabled" if no_wandb else None,
        job_type="train",
        config={"model_name": model_name, **config, **(extra_config or {})},
        tags=list(tags),
    )


def eval_every(steps_per_epoch, num_epochs, points=200):
    """Eval interval giving ~`points` curve points over the whole run."""
    return max(1, steps_per_epoch * num_epochs // points)


def _device_mem_gib(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024**3
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024**3
    return None


def _device_memory(device):
    gib = 1024**3
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name}, {props.total_memory / gib:.1f} GiB"
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / gib
    except (ValueError, OSError, AttributeError):
        return device.type
    if device.type == "mps":
        return f"Apple MPS (unified memory), {ram:.1f} GiB system RAM"
    return f"CPU, {ram:.1f} GiB system RAM"


def print_training_summary(model, cfg, device, train_loader, val_loader, model_name):
    """Print model, device, and hyperparameter details before training."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print("=" * 60)
    print(f"Model:            {model_name} ({type(model).__name__})")
    print(f"Parameters:       {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable (~{size_mb:.0f} MiB fp32)")
    # Windows x context length: exact for LM datasets, an upper bound for padded chat datasets
    print(f"Train Tokens:     ~{len(train_loader.dataset) * cfg['context_length']:,}")
    print(f"Val Tokens:       ~{len(val_loader.dataset) * cfg['context_length']:,}")
    if cfg.get("use_moe"):
        expert = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
        active = total - expert * (1 - cfg["top_k"] / cfg["num_experts"])
        print(
            f"MoE:              {cfg['num_experts']} experts, top-{cfg['top_k']}, "
            f"expert_hidden_dim={cfg['expert_hidden_dim']}, ~{active / 1e6:.2f}M active params/token"
        )
        print(
            f"MoE losses:       gate_noise_std={cfg['gate_noise_std']}, "
            f"aux_loss_coef={cfg['aux_loss_coef']}, z_loss_coef={cfg['z_loss_coef']}"
        )
    print(
        f"Architecture:     layers={cfg['num_layers']}, heads={cfg['num_heads']}, emb_dim={cfg['embedding_dim']}, "
        f"ctx={cfg['context_length']}, vocab={cfg['vocab_size']}, dropout={cfg['dropout_rate']}, qkv_bias={cfg['qkv_bias']}"
    )
    print(
        f"Training:         epochs={cfg['num_epochs']}, lr={cfg['learning_rate']}, batch_size={cfg['batch_size']}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )
    print(f"Device:           {device} ({_device_memory(device)})")
    print("=" * 60)


def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cropped = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cropped)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
        ignore_index=IGNORE_INDEX,
    )
    if hasattr(model, "aux_loss_coef") and model.training:
        loss = loss + model.aux_loss_coef * model.aux_loss + model.z_loss_coef * model.z_loss
    return loss


def loss_plot_path(model_name):
    return os.path.join("model-data", f"{model_name}-losses.png")


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path=None):
    """Three panels: train/val loss, train/val perplexity, val - train gap."""
    train = torch.tensor(train_losses)
    val = torch.tensor(val_losses)
    panels = [
        ("Loss", [(train, "Train", "-"), (val, "Val", "-.")]),
        ("Perplexity", [(train.exp(), "Train", "-"), (val.exp(), "Val", "-.")]),
        ("Val - train gap", [(val - train, "Gap", "-")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, (ylabel, series) in zip(axes, panels):
        for values, label, style in series:
            ax.plot(epochs_seen, values, linestyle=style, label=label)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        top = ax.twiny()
        top.plot(tokens_seen, series[0][0], alpha=0)
        top.set_xlabel("Tokens seen")
    axes[1].set_yscale("log")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        tqdm.write(f"  plot      {save_path}")
    plt.close(fig)


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        print("Warning: empty data loader — returning nan loss")
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: GPTModel, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50, context_size=context_size)
    generated = token_ids_to_text(token_ids[:, encoded.shape[1] :], tokenizer)
    generated = generated.split("<|endoftext|>", 1)[0].strip()
    tqdm.write(f"  prompt    {start_context.strip()}\n  sample    {generated or '(empty)'}")
    model.train()
    return generated


def checkpoint_path(model_name):
    os.makedirs("model-data", exist_ok=True)
    return os.path.join("model-data", f"{model_name}.pt")


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    tokens_seen,
    train_loss,
    val_loss,
    train_losses,
    val_losses,
    track_tokens_seen,
    config,
):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
            "config": config,
        },
        path,
    )
    tqdm.write(f"  saved     {path}")


def load_checkpoint(path, model, optimizer, device, config=None):
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if config is not None and "config" in checkpoint:
        for key in ("vocab_size", "context_length", "embedding_dim", "num_heads", "num_layers"):
            if key in checkpoint["config"] and checkpoint["config"][key] != config.get(key):
                raise ValueError(
                    f"Checkpoint config mismatch for {key}: "
                    f"{checkpoint['config'][key]} vs {config.get(key)}"
                )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f"Loaded checkpoint: {path} (epoch={checkpoint['epoch']}, step={checkpoint['global_step']})")
    return checkpoint


def load_pretrained_weights(model, path, device):
    """Load base-model weights into a model that may have a longer context.

    Attention masks are rebuilt by the new model, and position embeddings beyond
    the base context keep their fresh init. Optimizer/epoch state is not loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pretrained checkpoint at {path}")
    state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    state = {k: v for k, v in state.items() if not k.endswith(".mask")}

    base_pos = state.pop("position_embedding.weight")
    new_pos = model.position_embedding.weight
    if base_pos.shape[0] > new_pos.shape[0] or base_pos.shape[1] != new_pos.shape[1]:
        raise ValueError(
            f"Cannot fit base position embedding {tuple(base_pos.shape)} into {tuple(new_pos.shape)}"
        )
    with torch.no_grad():
        new_pos[: base_pos.shape[0]].copy_(base_pos)

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(".mask") and k != "position_embedding.weight"]
    if missing or unexpected:
        raise ValueError(f"Pretrained weight mismatch. missing={missing} unexpected={unexpected}")
    print(f"Loaded pretrained weights: {path} (base context {base_pos.shape[0]} -> {new_pos.shape[0]})")


def load_openai_gpt2_weights(model, device, repo_id="openai-community/gpt2"):
    """Download OpenAI GPT-2 weights from the HF Hub and copy them into GPTModel.

    Model config must match GPT-2: qkv_bias=True, same width/heads/layers, and
    context_length <= 1024 (position embeddings are sliced to fit).
    """
    path = hf_hub_download(repo_id, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("transformer."): v for k, v in sd.items()}

    def copy(param, value):
        if param.shape != value.shape:
            raise ValueError(f"Shape mismatch: {tuple(param.shape)} vs {tuple(value.shape)}")
        with torch.no_grad():
            param.copy_(value)

    ctx = model.position_embedding.weight.shape[0]
    copy(model.token_embedding.weight, sd["wte.weight"])
    copy(model.position_embedding.weight, sd["wpe.weight"][:ctx])
    copy(model.out_head.weight, sd["wte.weight"])
    copy(model.final_norm.scale, sd["ln_f.weight"])
    copy(model.final_norm.shift, sd["ln_f.bias"])

    for i, block in enumerate(model.transformer_blocks):
        p = f"h.{i}."
        q_w, k_w, v_w = sd[p + "attn.c_attn.weight"].T.chunk(3, dim=0)
        q_b, k_b, v_b = sd[p + "attn.c_attn.bias"].chunk(3)
        attn = block.attention
        copy(attn.W_q.weight, q_w)
        copy(attn.W_k.weight, k_w)
        copy(attn.W_v.weight, v_w)
        copy(attn.W_q.bias, q_b)
        copy(attn.W_k.bias, k_b)
        copy(attn.W_v.bias, v_b)
        copy(attn.out_proj.weight, sd[p + "attn.c_proj.weight"].T)
        copy(attn.out_proj.bias, sd[p + "attn.c_proj.bias"])

        ff = block.feed_forward.layers
        copy(ff[0].weight, sd[p + "mlp.c_fc.weight"].T)
        copy(ff[0].bias, sd[p + "mlp.c_fc.bias"])
        copy(ff[2].weight, sd[p + "mlp.c_proj.weight"].T)
        copy(ff[2].bias, sd[p + "mlp.c_proj.bias"])

        copy(block.norm1.scale, sd[p + "ln_1.weight"])
        copy(block.norm1.shift, sd[p + "ln_1.bias"])
        copy(block.norm2.scale, sd[p + "ln_2.weight"])
        copy(block.norm2.shift, sd[p + "ln_2.bias"])

    model.to(device)
    print(f"Loaded OpenAI GPT-2 weights from {repo_id} (context {ctx})")


def _format_loss_metrics(step, train_loss, val_loss, grad_norm):
    return (
        f"eval @ step {step:,}  train {train_loss:.3f} · val {val_loss:.3f} · "
        f"ppl {math.exp(train_loss):.1f}/{math.exp(val_loss):.1f} · "
        f"gap {val_loss - train_loss:+.3f} · gnorm {grad_norm:.2f}"
    )


def _moe_stats(model):
    """Load-balancing loss, router entropy (vs. max ln E) and per-expert load averaged over layers."""
    frac, entropy = model.router_stats()
    return {
        "aux_loss": model.aux_loss.item(),
        "entropy": entropy.item(),
        "max_entropy": math.log(frac.shape[1]),
        "load_pct": frac.mean(0).mul(100).tolist(),
        "max_load_pct": frac.max().item() * 100,
    }


def _format_moe_metrics(m):
    load = "/".join(str(round(x)) for x in m["load_pct"])
    return (
        f"lb {m['aux_loss']:.3f} · H {m['entropy']:.2f}/{m['max_entropy']:.2f} · "
        f"load% {load} (max {m['max_load_pct']:.0f})"
    )


class _PauseOnInterrupt:
    """First Ctrl+C requests a pause (eval + sample, then resume); a second one before it's handled aborts."""

    def __enter__(self):
        self.requested = False
        self._previous = signal.signal(signal.SIGINT, self._handle)
        # Stop the terminal echoing a literal "^C" into the progress bar
        self._tty = None
        if sys.stdin.isatty():
            self._tty = termios.tcgetattr(sys.stdin)
            attrs = termios.tcgetattr(sys.stdin)
            attrs[3] &= ~termios.ECHOCTL
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, attrs)
        return self

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)
        if self._tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._tty)


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    model_name,
    config=None,
    grad_clip=1.0,
    run=None,
):
    """run: optional wandb run; logged against a "step" axis so resumed epochs don't violate wandb's step ordering."""
    model.to(device)
    path = checkpoint_path(model_name)

    train_losses = []
    val_losses = []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = -1
    start_epoch = 0
    train_loss = float("nan")
    val_loss = float("nan")

    checkpoint = load_checkpoint(path, model, optimizer, device, config)
    if checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        tokens_seen = checkpoint["tokens_seen"]
        train_losses = list(checkpoint.get("train_losses", []))
        val_losses = list(checkpoint.get("val_losses", []))
        track_tokens_seen = list(checkpoint.get("track_tokens_seen", []))
        train_loss = checkpoint.get("train_loss", float("nan"))
        val_loss = checkpoint.get("val_loss", float("nan"))
        if start_epoch >= num_epochs:
            print(f"Checkpoint already finished all {num_epochs} epochs — nothing to train.")
            return train_losses, val_losses, track_tokens_seen
        print(f"Resuming from epoch {start_epoch + 1}/{num_epochs}")

    if run is not None:
        run.define_metric("step")
        run.define_metric("tokens_seen", step_metric="step")
        for prefix in ("train/*", "eval/*", "moe/*", "system/*", "samples", "plots/*"):
            run.define_metric(prefix, step_metric="step")
    print("Ctrl+C: eval + sample, then continue · Ctrl+C twice: abort\n")
    steps_per_epoch = len(train_loader)
    bar = tqdm(
        total=num_epochs * steps_per_epoch,
        initial=start_epoch * steps_per_epoch,
        unit="step",
        dynamic_ncols=True,
        bar_format="{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    session_tokens = 0
    start_time = time.time()
    sample_rows = []
    # .item() only when the display refreshes to avoid per-step host syncs
    grad_norm = torch.tensor(float("nan"))

    with _PauseOnInterrupt() as pause:
        try:
            for epoch in range(start_epoch, num_epochs):
                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    # Returns the pre-clip norm; max_norm=inf just measures it
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip if grad_clip is not None else float("inf")
                    )
                    optimizer.step()
                    tokens_seen += input_batch.numel()
                    session_tokens += input_batch.numel()
                    global_step += 1
                    bar.update(1)

                    if global_step % 10 == 0:
                        elapsed = bar.format_dict["elapsed"] or 1e-9
                        bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                        loss_val = loss.item()
                        tok_s = session_tokens / elapsed
                        bar.set_postfix_str(f"loss {loss_val:.3f} · {tok_s:,.0f} tok/s")
                        if run is not None:
                            metrics = {
                                "step": global_step,
                                "tokens_seen": tokens_seen,
                                "train/loss": loss_val,  # total optimised loss (incl. MoE aux terms)
                                "train/grad_norm": grad_norm.item(),
                                "train/grad_clipped": float(grad_clip is not None and grad_norm.item() > grad_clip),
                                "train/param_norm": torch.linalg.vector_norm(
                                    torch.stack([torch.linalg.vector_norm(p.detach()) for p in model.parameters()])
                                ).item(),
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/tokens_per_sec": tok_s,
                                "train/epoch": epoch + (global_step % steps_per_epoch + 1) / steps_per_epoch,
                                "train/elapsed_min": (time.time() - start_time) / 60,
                            }
                            if hasattr(model, "aux_loss_coef"):
                                aux, z = model.aux_loss.item(), model.z_loss.item()
                                metrics["train/ce_loss"] = loss_val - model.aux_loss_coef * aux - model.z_loss_coef * z
                                metrics["moe/aux_loss_step"] = aux
                                metrics["moe/z_loss_step"] = z
                            mem = _device_mem_gib(device)
                            if mem is not None:
                                metrics["system/device_mem_gib"] = mem
                            run.log(metrics)

                    if global_step % eval_freq == 0 or pause.requested:
                        # Read MoE routing stats from this training step before eval forwards overwrite them
                        moe = _moe_stats(model) if hasattr(model, "router_stats") else None
                        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                        line = _format_loss_metrics(global_step, train_loss, val_loss, grad_norm.item())
                        if moe:
                            line += "\n    moe  " + _format_moe_metrics(moe)
                        if run is not None:
                            metrics = {
                                "step": global_step,
                                "tokens_seen": tokens_seen,
                                "eval/train_loss": train_loss,
                                "eval/val_loss": val_loss,
                                "eval/train_ppl": math.exp(train_loss),
                                "eval/val_ppl": math.exp(val_loss),
                                "eval/gap": val_loss - train_loss,
                            }
                            if moe:
                                metrics.update(
                                    {
                                        "moe/aux_loss": moe["aux_loss"],
                                        "moe/router_entropy": moe["entropy"],
                                        "moe/max_expert_load_pct": moe["max_load_pct"],
                                        **{f"moe/expert_load_pct/{i}": v for i, v in enumerate(moe["load_pct"])},
                                    }
                                )
                            run.log(metrics)
                        if not pause.requested:
                            tqdm.write(line)
                        if global_step % eval_freq == 0:
                            train_losses.append(train_loss)
                            val_losses.append(val_loss)
                            track_tokens_seen.append(tokens_seen)

                    if pause.requested:
                        tqdm.write(f"\n⏸  paused\n{line}")
                        generate_and_print_sample(model, tokenizer, device, start_context)
                        tqdm.write("▶  resuming\n")
                        pause.requested = False

                tqdm.write(f"\n── end of epoch {epoch + 1}/{num_epochs} ──")
                sample = generate_and_print_sample(model, tokenizer, device, start_context)
                sample_rows.append([epoch + 1, global_step, start_context, sample])
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    tokens_seen,
                    train_loss,
                    val_loss,
                    train_losses,
                    val_losses,
                    track_tokens_seen,
                    config,
                )
                if train_losses:
                    epochs_seen = torch.linspace(0, epoch + 1, len(train_losses))
                    plot_losses(epochs_seen, track_tokens_seen, train_losses, val_losses, loss_plot_path(model_name))
                if run is not None:
                    media = {
                        "step": global_step,
                        "samples": wandb.Table(columns=["epoch", "step", "prompt", "generated"], data=sample_rows),
                    }
                    if train_losses:
                        media["plots/losses"] = wandb.Image(loss_plot_path(model_name))
                    run.log(media)
        except KeyboardInterrupt:
            # Checkpoints are per-epoch, so progress since the last epoch end is discarded
            tqdm.write(f"Aborted at step {global_step}; last checkpoint: {path}")
        finally:
            bar.close()
            if run is not None:
                run.summary["train/total_minutes"] = (time.time() - start_time) / 60
                run.summary["train/final_tokens_seen"] = tokens_seen
                run.summary["train/final_eval_train_loss"] = train_loss
                run.summary["train/final_eval_val_loss"] = val_loss

    return train_losses, val_losses, track_tokens_seen
