import argparse
import os

import tiktoken
import torch

from text_dataset import create_dataloader_cosmopedia, create_dataloader_cosmopedia_full
from utils import (
    build_model,
    checkpoint_path,
    generate_and_print_sample,
    get_device,
    print_training_summary,
    train_model_simple,
)


# Cosmopedia configs mixed for training (max_samples applies per subset in the non-full run)
SUBSETS = ["stanford", "khanacademy", "openstax", "wikihow", "auto_math_text", "stories"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Train on the entire cosmopedia stanford subset")
    args = parser.parse_args()

    GPT_CONFIG_MOE = {
        "vocab_size": 50257,
        "context_length": 64,
        "stride": 64,
        "embedding_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "num_epochs": 2,
        "start_context": "In this chapter",
        "use_moe": True,
        "num_experts": 8,
        "top_k": 2,
        "expert_hidden_dim": 768,
        "gate_noise_std": 0.1,
        "aux_loss_coef": 0.01,
        "z_loss_coef": 0.001,
    }
    model_name = "gpt-2-moe-8x-mixed"  # new name so the old stanford-only checkpoint isn't resumed
    if args.full:
        model_name += "-full"

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = build_model(GPT_CONFIG_MOE)
    if args.full:
        train_loader, val_loader = create_dataloader_cosmopedia_full(GPT_CONFIG_MOE, subset=SUBSETS)
    else:
        train_loader, val_loader = create_dataloader_cosmopedia(
            GPT_CONFIG_MOE, subset=SUBSETS, max_samples=10000, val_ratio=0.1, seed=123
        )

    device = get_device()
    model.to(device)
    print_training_summary(model, GPT_CONFIG_MOE, device, train_loader, val_loader, model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_MOE["learning_rate"])

    if os.path.exists(checkpoint_path(model_name)):
        print("Resuming from checkpoint — sample after load happens inside training.")
    else:
        print("Untrained model sample:")
        generate_and_print_sample(model, tokenizer, device, GPT_CONFIG_MOE["start_context"])

    train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=GPT_CONFIG_MOE["num_epochs"],
        eval_freq=100,
        eval_iter=10,
        start_context=GPT_CONFIG_MOE["start_context"],
        tokenizer=tokenizer,
        model_name=model_name,
        config=GPT_CONFIG_MOE,
    )
    # Loss plot is saved next to the checkpoint inside train_model_simple


if __name__ == "__main__":
    main()
