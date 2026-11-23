import os

import tiktoken
import torch

from src.models import GPTModel
from src.text_dataset import CHAT_TEMPLATE, create_dataloader_gsm8k
from src.utils import (
    checkpoint_path,
    generate_and_print_sample,
    get_device,
    load_pretrained_weights,
    print_training_summary,
    train_model_simple,
)


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 512,
        "embedding_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "num_epochs": 3,
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
    base_model_name = "gpt-2-124m"
    model_name = "gpt-2-124m-gsm8k"

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    train_loader, val_loader = create_dataloader_gsm8k(GPT_CONFIG_124M)

    device = get_device()
    model.to(device)
    print_training_summary(model, GPT_CONFIG_124M, device, train_loader, val_loader, model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_124M["learning_rate"])

    if os.path.exists(checkpoint_path(model_name)):
        print("Resuming fine-tune from checkpoint.")
    else:
        load_pretrained_weights(model, checkpoint_path(base_model_name), device)
        print("Base model sample:")
        generate_and_print_sample(model, tokenizer, device, GPT_CONFIG_124M["start_context"])

    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=GPT_CONFIG_124M["num_epochs"],
        eval_freq=50,
        eval_iter=10,
        start_context=GPT_CONFIG_124M["start_context"],
        tokenizer=tokenizer,
        model_name=model_name,
        config=GPT_CONFIG_124M,
    )
    # Loss plot is saved next to the checkpoint inside train_model_simple


if __name__ == "__main__":
    main()
