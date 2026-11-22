"""Shared evaluation: per-subset + overall held-out metrics, logged to W&B (resuming the training run) and a JSON for the model card."""
import argparse
import json
import math
import os
import sys

import tiktoken
import torch
import wandb

from text_dataset import (
    _load_cosmopedia,
    create_dataloader_cosmopedia_full,
    create_dataloader_gsm8k,
    create_dataloader_lm,
)
from utils import (
    build_model,
    calc_loss_batch,
    calc_loss_loader,
    checkpoint_path,
    generate_and_print_sample,
    get_device,
)

# Every Cosmopedia config we evaluate on. Configs a model wasn't trained on are scored on random docs (all unseen).
# web_samples_v1/v2 are left out: tens of GB to download for a small eval.
COSMOPEDIA_SUBSETS = ["stanford", "khanacademy", "openstax", "wikihow", "auto_math_text", "stories"]
HELD_OUT, UNSEEN = "held-out", "unseen subset"


def eval_results_path(model_name):
    return os.path.join("model-data", f"{model_name}-eval.json")


def subset_val_loaders(config, rows_per_subset, train_subsets, train_max_samples, val_ratio=0.1, seed=123):
    """{subset: (split, loader)}. Trained subsets use the train script's held-out split; others use random docs."""
    raw = _load_cosmopedia(train_subsets, train_max_samples, seed, with_subset=True).shuffle(seed=seed)
    val = raw.train_test_split(test_size=val_ratio, seed=seed)["test"]
    loaders = {}
    for name in COSMOPEDIA_SUBSETS:
        if name in train_subsets:
            split, texts = HELD_OUT, val.filter(lambda r: r["subset"] == name)["text"][:rows_per_subset]
        else:
            split, texts = UNSEEN, _load_cosmopedia(name, rows_per_subset, seed)["text"]
        loaders[name] = (split, create_dataloader_lm("<|endoftext|>".join(texts), config, shuffle=False, drop_last=False))
    return loaders


def token_weighted_loss(loader, model, device):
    """Returns (summed loss, token count); calc_loss_batch is a per-batch token mean."""
    total, n = 0.0, 0
    for x, y in loader:
        total += calc_loss_batch(x, y, model, device).item() * y.numel()
        n += y.numel()
    return total, n


def run_eval(description, model_name, wandb_project, train_subsets, train_max_samples):
    """Evaluates the checkpoint the counterpart train script produces (model_name, +"-full" with --full)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", help=f"Override checkpoint name without .pt (default: {model_name})")
    parser.add_argument("--dataset", choices=("cosmopedia", "gsm8k"), default="cosmopedia")
    parser.add_argument("--full", action="store_true", help="Use the full-cosmopedia val split")
    parser.add_argument("--rows-per-subset", type=int, default=50, help="Docs sampled per subset (cosmopedia)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--num-batches", type=int, default=None, help="Limit val batches (--full / gsm8k only)")
    parser.add_argument("--prompt", help="Also print a greedy completion for this prompt")
    args = parser.parse_args()
    args.model = args.model or model_name + ("-full" if args.full else "")

    device = get_device()
    checkpoint = torch.load(checkpoint_path(args.model), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Resume the training run (same id as model name) so final eval sits next to its training curves.
    # Only summary/tables/charts are written, which doesn't touch the training "step" history.
    run = wandb.init(
        project=wandb_project,
        id=args.model,
        resume="allow",
        mode="disabled" if args.no_wandb else None,  # None: honour WANDB_MODE (e.g. offline)
        job_type="eval",
    )

    if args.dataset == "cosmopedia" and not args.full:
        loaders = subset_val_loaders(config, args.rows_per_subset, train_subsets, train_max_samples)
        print(f"{'subset':<16} {'split':<14} {'tokens':>9} {'loss':>7} {'ppl':>8}")
        sums = {"all": [0.0, 0], "in_dist": [0.0, 0]}
        results = {}
        with torch.no_grad():
            for name, (split, loader) in loaders.items():
                total, n = token_weighted_loss(loader, model, device)
                results[name] = {"split": split, "tokens": n, "loss": total / n, "ppl": math.exp(total / n)}
                print(f"{name:<16} {split:<14} {n:>9,} {total / n:>7.3f} {math.exp(total / n):>8.2f}")
                for key in ("all", "in_dist") if split == HELD_OUT else ("all",):
                    sums[key][0] += total
                    sums[key][1] += n
        for key, label in (("in_dist", "overall (held-out)"), ("all", "overall (all)")):
            total, n = sums[key]
            results[f"overall_{key}"] = {"split": label, "tokens": n, "loss": total / n, "ppl": math.exp(total / n)}
            print(f"{label:<31} {n:>9,} {total / n:>7.3f} {math.exp(total / n):>8.2f}")

        with open(eval_results_path(args.model), "w") as f:
            json.dump(
                {
                    "rows_per_subset": args.rows_per_subset,
                    "context_length": config["context_length"],
                    "train_subsets": train_subsets,
                    "results": results,
                },
                f,
                indent=2,
            )
        for name, r in results.items():
            run.summary[f"final_eval/{name}/loss"] = r["loss"]
            run.summary[f"final_eval/{name}/ppl"] = r["ppl"]
            run.summary[f"final_eval/{name}/tokens"] = r["tokens"]
        run.summary["final_eval/loss"] = results["overall_in_dist"]["loss"]  # headline = held-out only
        run.summary["final_eval/ppl"] = results["overall_in_dist"]["ppl"]
        run.summary["final_eval/rows_per_subset"] = args.rows_per_subset
        columns = ["subset", "split", "tokens", "loss", "ppl"]
        rows = [[k, r["split"], r["tokens"], r["loss"], r["ppl"]] for k, r in results.items()]
        per_subset = wandb.Table(columns=columns, data=[r for r in rows if not r[0].startswith("overall")])
        run.log(
            {
                "final_eval/by_subset": wandb.Table(columns=columns, data=rows),
                "final_eval/loss_by_subset": wandb.plot.bar(per_subset, "subset", "loss", title="Eval loss by subset"),
                "final_eval/ppl_by_subset": wandb.plot.bar(per_subset, "subset", "ppl", title="Eval perplexity by subset"),
            }
        )
    else:
        if args.dataset == "gsm8k":
            _, val_loader = create_dataloader_gsm8k(config)
        else:
            _, val_loader = create_dataloader_cosmopedia_full(config, subset=train_subsets)
        with torch.no_grad():
            val_loss = calc_loss_loader(val_loader, model, device, args.num_batches)
        print(f"{args.model} on {args.dataset}: val loss {val_loss:.4f}, perplexity {math.exp(val_loss):.2f}")
        run.summary[f"final_eval/{args.dataset}/loss"] = val_loss
        run.summary[f"final_eval/{args.dataset}/ppl"] = math.exp(val_loss)

    if args.prompt:
        generate_and_print_sample(model, tiktoken.get_encoding("gpt2"), device, args.prompt)
    run.finish()


def main_guard(fn):
    try:
        fn()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
