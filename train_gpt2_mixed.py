import argparse
import os

import tiktoken
import torch

from src.text_dataset import create_dataloader_mixed
from src.utils import (
    build_model,
    checkpoint_path,
    eval_every,
    generate_and_print_sample,
    get_device,
    init_wandb,
    print_training_summary,
    train_model_simple,
)

MODEL_NAME = "gpt-2-124m-mixed"
WANDB_PROJECT = "gpt2-mixed"  # one project per corpus; each model/version is a run in it
EVAL_ITER = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--tokens", type=int, default=1_000_000_000, help="Total train token budget across the mix")
    parser.add_argument("--val-tokens", type=int, default=2_000_000, help="Total val token budget across the mix")
    args = parser.parse_args()

    GPT_CONFIG = {
        "vocab_size": 50257,
        "context_length": 128,
        "stride": 64,
        "embedding_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "start_context": "The history of",
    }
    model_name = MODEL_NAME

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = build_model(GPT_CONFIG)
    # 40% FineWeb, 25% FineWeb-Edu, 20% Cosmopedia, 10% books, 2.5% Wikipedia, 2.5% code
    train_loader, val_loader = create_dataloader_mixed(
        GPT_CONFIG, total_tokens=args.tokens, val_tokens=args.val_tokens, seed=123
    )

    device = get_device()
    model.to(device)
    print_training_summary(model, GPT_CONFIG, device, train_loader, val_loader, model_name)
    # fused=True runs the AdamW update as one kernel on cuda/mps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GPT_CONFIG["learning_rate"], fused=device.type in ("cuda", "mps")
    )

    if os.path.exists(checkpoint_path(model_name)):
        print("Resuming from checkpoint — sample after load happens inside training.")
    else:
        print("Untrained model sample:")
        generate_and_print_sample(model, tokenizer, device, GPT_CONFIG["start_context"])

    eval_freq = eval_every(len(train_loader), GPT_CONFIG["num_epochs"])
    run = init_wandb(
        WANDB_PROJECT,
        model_name,
        GPT_CONFIG,
        no_wandb=args.no_wandb,
        tags=["dense", "mixed"],
        extra_config={
            "train_tokens": args.tokens,
            "val_tokens": args.val_tokens,
            "eval_freq": eval_freq,
            "eval_iter": EVAL_ITER,
            "grad_clip": 1.0,
            "seed": 123,
            "device": device.type,
            "params_total": sum(p.numel() for p in model.parameters()),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
        },
    )
    train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=GPT_CONFIG["num_epochs"],
        eval_freq=eval_freq,
        eval_iter=EVAL_ITER,
        start_context=GPT_CONFIG["start_context"],
        tokenizer=tokenizer,
        model_name=model_name,
        config=GPT_CONFIG,
        run=run,
    )
    run.finish()
    # Loss plot is saved next to the checkpoint inside train_model_simple


if __name__ == "__main__":
    main()
