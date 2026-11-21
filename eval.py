import argparse
import math
import sys

import tiktoken
import torch

from text_dataset import create_dataloader_cosmopedia, create_dataloader_cosmopedia_full, create_dataloader_gsm8k
from train_gpt2_moe_cosmopedia import SUBSETS
from utils import (
    build_model,
    calc_loss_loader,
    checkpoint_path,
    generate_and_print_sample,
    get_device,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint from model-data/.")
    parser.add_argument("--model", required=True, help="Checkpoint name without .pt")
    parser.add_argument("--dataset", choices=("cosmopedia", "gsm8k"), default="cosmopedia")
    parser.add_argument("--full", action="store_true", help="Use the full-cosmopedia val split")
    parser.add_argument("--num-batches", type=int, default=None, help="Limit val batches (default: all)")
    parser.add_argument("--prompt", help="Also print a greedy completion for this prompt")
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(checkpoint_path(args.model), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    if args.dataset == "gsm8k":
        _, val_loader = create_dataloader_gsm8k(config)
    elif args.full:
        _, val_loader = create_dataloader_cosmopedia_full(config, subset=SUBSETS)
    else:
        # Same subsets/samples/seed as train_gpt2_moe_cosmopedia.py, so val split is held-out
        _, val_loader = create_dataloader_cosmopedia(
            config, subset=SUBSETS, max_samples=10000, val_ratio=0.1, seed=123
        )

    with torch.no_grad():
        val_loss = calc_loss_loader(val_loader, model, device, args.num_batches)
    print(f"{args.model} on {args.dataset}: val loss {val_loss:.4f}, perplexity {math.exp(val_loss):.2f}")

    if args.prompt:
        generate_and_print_sample(model, tiktoken.get_encoding("gpt2"), device, args.prompt)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
