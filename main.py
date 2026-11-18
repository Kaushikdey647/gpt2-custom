import os

import tiktoken
import torch
from models import GPTModel
from utils import (
    checkpoint_path,
    create_dataloader_cosmopedia,
    generate_and_print_sample,
    plot_losses,
    train_model_simple,
)

def main():
    GPT_CONFIG_124M = {
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
        "num_epochs": 5,
        "start_context": "In this chapter",
    }
    model_name = "gpt-2-124m"

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    train_loader, val_loader = create_dataloader_cosmopedia(
        GPT_CONFIG_124M, subset="stanford", max_samples=5000, val_ratio=0.1, seed=123
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_124M["learning_rate"])

    if os.path.exists(checkpoint_path(model_name)):
        print("Resuming from checkpoint — sample after load happens inside training.")
    else:
        print("Untrained model sample:")
        generate_and_print_sample(model, tokenizer, device, GPT_CONFIG_124M["start_context"])

    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=GPT_CONFIG_124M["num_epochs"],
        eval_freq=100,
        eval_iter=10,
        start_context=GPT_CONFIG_124M["start_context"],
        tokenizer=tokenizer,
        model_name=model_name,
        config=GPT_CONFIG_124M,
    )
    if train_losses:
        epochs_tensor = torch.linspace(0, GPT_CONFIG_124M["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)

    print("Great, now give a sentence for it to complete: ")
    start_context = input()
    generate_and_print_sample(model, tokenizer, device, start_context)

if __name__ == "__main__":
    main()
