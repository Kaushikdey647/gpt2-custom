import argparse
import time

import tiktoken
import torch

from inference import load_model
from src.text_dataset import _load_cosmopedia
from src.utils import (
    generate_text_nucleus_sampling,
    generate_text_simple,
    generate_text_top_k_sampling,
    get_device,
    text_to_token_ids,
    token_ids_to_text,
)

GENERATORS = {
    "greedy": generate_text_simple,
    "top_k": generate_text_top_k_sampling,
    "nucleus": generate_text_nucleus_sampling,
}


def main():
    parser = argparse.ArgumentParser(
        description="Take a held-out Cosmopedia doc, keep its first tokens, and let every generator in utils.py continue it."
    )
    parser.add_argument("--model", required=True, help="Checkpoint name without .pt")
    parser.add_argument("--subsets", nargs="+", default=["stanford"], help="Cosmopedia subsets the model trained on")
    parser.add_argument("--index", type=int, default=0, help="Which held-out doc to use")
    parser.add_argument("--prefix-tokens", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Same held-out split as create_dataloader_cosmopedia_full (val_ratio=0.001, seed=123)
    held_out = _load_cosmopedia(args.subsets).train_test_split(test_size=0.001, seed=123)["test"]
    text = held_out[args.index]["text"]

    device = get_device()
    model = load_model(args.model, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    context_size = model.position_embedding.weight.shape[0]
    full_ids = text_to_token_ids(text, tokenizer)
    idx = full_ids[:, : args.prefix_tokens].to(device)

    print(f"\n══ held-out doc {args.index}/{len(held_out)} ══\n{text}\n")
    print(f"══ prefix ({args.prefix_tokens} tokens) ══\n{token_ids_to_text(idx, tokenizer)}\n")
    for name, generate in GENERATORS.items():
        torch.manual_seed(args.seed)  # same seed per generator so sampling runs are reproducible
        start = time.time()
        out = generate(model, idx, args.max_new_tokens, context_size, temperature=args.temperature)
        elapsed = time.time() - start
        completion = token_ids_to_text(out[:, idx.shape[1] :], tokenizer).split("<|endoftext|>", 1)[0].strip()
        print(f"── {name} ({elapsed:.1f}s) ──\n{completion or '(empty)'}\n")


if __name__ == "__main__":
    main()
