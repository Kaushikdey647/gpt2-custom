import os

import tiktoken
import torch

from models import GPTModel
from text_dataset import CHAT_TEMPLATE, create_dataloader_gsm8k
from utils import (
    checkpoint_path,
    generate_and_print_sample,
    get_device,
    load_openai_gpt2_weights,
    plot_losses,
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
        "qkv_bias": True,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "start_context": CHAT_TEMPLATE.format(
            instruction=(
                "Natalia sold shells to 32 of her friends in April, and then she sold "
                "twice as many shells in May. How many shells did Natalia sell "
                "altogether in April and May?"
            ),
            input="None",
            response="",
        ),
    }
    model_name = "gpt2-openai-gsm8k"

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    train_loader, val_loader = create_dataloader_gsm8k(GPT_CONFIG_124M)

    device = get_device()
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_124M["learning_rate"])

    if os.path.exists(checkpoint_path(model_name)):
        print("Resuming fine-tune from checkpoint.")
    else:
        load_openai_gpt2_weights(model, device)
        print("OpenAI GPT-2 sample before fine-tuning:")
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
    if train_losses:
        epochs_tensor = torch.linspace(0, GPT_CONFIG_124M["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
