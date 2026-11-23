import argparse

import tiktoken
import torch

from src.lora import apply_lora, trainable_params
from src.models import GPTModel
from src.text_dataset import CHAT_TEMPLATE, create_dataloader_gsm8k
from src.utils import (
    checkpoint_path,
    eval_every,
    generate_and_print_sample,
    get_device,
    init_wandb,
    load_pretrained_weights,
    print_training_summary,
    train_model_simple,
)

WANDB_PROJECT = "gpt2-lora-gsm8k"  # one project per train script; LoRA and QLoRA runs are runs in it
BASE_MODEL_NAME = "gpt-2-124m"
MODEL_NAME = "gpt-2-124m-gsm8k-lora"  # "-qlora" suffix is swapped in with --qlora
EVAL_ITER = 10
LORA = {"r": 16, "alpha": 32, "dropout": 0.05}


def model_name_for(qlora):
    return MODEL_NAME.replace("-lora", "-qlora") if qlora else MODEL_NAME


def build_base(cfg, device):
    """Pretrained GPT-2 (our Cosmopedia checkpoint), extended to the finetune context. Adapters are applied on top."""
    model = GPTModel(cfg)
    load_pretrained_weights(model, checkpoint_path(BASE_MODEL_NAME), device)
    return model


def main():
    parser = argparse.ArgumentParser(description="LoRA / QLoRA finetune of gpt-2-124m on GSM8K.")
    parser.add_argument("--qlora", action="store_true", help="Store the frozen base as 4-bit NF4 (pure PyTorch)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 512,
        "embedding_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False,
        "batch_size": 4,
        "learning_rate": 2e-4,  # LoRA tolerates ~10x the full-finetune LR
        "num_epochs": 3,
        "lora_r": LORA["r"],
        "lora_alpha": LORA["alpha"],
        "lora_dropout": LORA["dropout"],
        "qlora": args.qlora,
        "start_context": CHAT_TEMPLATE.format(
            instruction=(
                "Charlie sold apples to 24 of his friends in June, and then he sold "
                "half as many apples in July. How many clips did Charlie sell "
                "altogether in June and July?"
            ),
            input="None",
            response="",
        ),
    }
    model_name = model_name_for(args.qlora)

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = get_device()
    model = build_base(GPT_CONFIG_124M, device)
    apply_lora(model, quantize=args.qlora, **LORA)
    model.to(device)
    train_loader, val_loader = create_dataloader_gsm8k(GPT_CONFIG_124M)

    n_train = sum(p.numel() for p in trainable_params(model))
    n_all = sum(p.numel() for p in model.parameters())
    print_training_summary(model, GPT_CONFIG_124M, device, train_loader, val_loader, model_name)
    print(f"Adapter:          r={LORA['r']} alpha={LORA['alpha']} dropout={LORA['dropout']} qlora={args.qlora} | "
          f"{n_train / 1e6:.2f}M trainable ({100 * n_train / n_all:.2f}% of params)")
    optimizer = torch.optim.AdamW(trainable_params(model), lr=GPT_CONFIG_124M["learning_rate"])

    eval_freq = eval_every(len(train_loader), GPT_CONFIG_124M["num_epochs"])
    run = init_wandb(
        WANDB_PROJECT,
        model_name,
        GPT_CONFIG_124M,
        no_wandb=args.no_wandb,
        tags=["lora", "qlora" if args.qlora else "lora-fp32", "gsm8k"],
        extra_config={
            "base_model": BASE_MODEL_NAME,
            "lora_targets": model.lora_cfg["targets"],
            "modules_to_save": model.lora_cfg["modules_to_save"],
            "trainable_params": n_train,
            "trainable_pct": 100 * n_train / n_all,
            "eval_freq": eval_freq,
            "eval_iter": EVAL_ITER,
            "grad_clip": 1.0,
            "seed": 123,
            "device": device.type,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
        },
    )
    print("Base + untrained adapter sample:")
    generate_and_print_sample(model, tokenizer, device, GPT_CONFIG_124M["start_context"])

    train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=GPT_CONFIG_124M["num_epochs"],
        eval_freq=eval_freq,
        eval_iter=EVAL_ITER,
        start_context=GPT_CONFIG_124M["start_context"],
        tokenizer=tokenizer,
        model_name=model_name,
        config=GPT_CONFIG_124M,
        run=run,
    )
    run.finish()


if __name__ == "__main__":
    main()
