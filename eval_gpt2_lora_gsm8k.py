import argparse
import copy
import math

import torch
import wandb

from src.lora import apply_lora, load_adapter_state_dict, merge_lora
from src.text_dataset import create_dataloader_gsm8k
from train_gpt2_lora_gsm8k import WANDB_PROJECT, build_base, model_name_for
from src.utils import calc_loss_loader, checkpoint_path, get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate the adapter trained by train_gpt2_lora_gsm8k.py on GSM8K test.")
    parser.add_argument("--qlora", action="store_true", help="Evaluate the QLoRA run instead of LoRA")
    parser.add_argument("--model", help="Override checkpoint name without .pt")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()
    model_name = args.model or model_name_for(args.qlora)

    device = get_device()
    checkpoint = torch.load(checkpoint_path(model_name), map_location=device, weights_only=False)
    config, lcfg = checkpoint["config"], checkpoint["lora_cfg"]
    run = wandb.init(
        project=WANDB_PROJECT, id=model_name, resume="allow", job_type="eval",
        mode="disabled" if args.no_wandb else None,  # None: honour WANDB_MODE
    )
    _, val_loader = create_dataloader_gsm8k(config)

    base = build_base(config, device).to(device).eval()
    keys = ("r", "alpha", "dropout", "targets", "quantize", "modules_to_save")
    adapted = apply_lora(copy.deepcopy(base), **{k: lcfg[k] for k in keys})
    load_adapter_state_dict(adapted, checkpoint["model_state_dict"])
    adapted.to(device).eval()
    merged = merge_lora(copy.deepcopy(adapted)).eval()

    x, _ = next(iter(val_loader))
    with torch.no_grad():
        merge_gap = (adapted(x.to(device)) - merged(x.to(device))).abs().max().item()

    rows = {}
    with torch.no_grad():
        for name, m in (("base", base), ("adapter", adapted), ("merged", merged)):
            rows[name] = calc_loss_loader(val_loader, m, device)
            print(f"{name:<8} loss {rows[name]:.4f}  ppl {math.exp(rows[name]):.2f}")
    print(f"max |adapter - merged| logit gap: {merge_gap:.2e}")

    for name, loss in rows.items():
        run.summary[f"final_eval/gsm8k/{name}/loss"] = loss
        run.summary[f"final_eval/gsm8k/{name}/ppl"] = math.exp(loss)
    run.summary["final_eval/gsm8k/merge_max_logit_gap"] = merge_gap
    run.summary["final_eval/gsm8k/loss"] = rows["adapter"]  # headline = base + adapter
    run.finish()


if __name__ == "__main__":
    main()
