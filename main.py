import tiktoken
import torch
from models import GPTModel
from utils import get_text, create_dataloader_v2, train_model_simple, generate_and_print_sample, plot_losses

def main():

    # Set up the configuration for the model
    GPT_CONFIG_124M = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 128, # Context length
        "embedding_dim": 768, # Embedding dimension
        "num_heads": 12, # Number of attention heads
        "num_layers": 12, # Number of layers
        "dropout_rate": 0.1, # Dropout rate
        "qkv_bias": False, # Query-Key-Value bias
        "batch_size": 8,
        "learning_rate": 3e-4,
        "num_epochs": 5,
        "start_context": "Israel is a"
    }

    # Set up the seed for the random number generator
    torch.manual_seed(123)

    # Tokenizer for some manual labour
    tokenizer = tiktoken.get_encoding("gpt2")

    # Model for the GPT-2 architecture
    model = GPTModel(GPT_CONFIG_124M)

    # Get the text data from the file
    text_data = get_text()

    tokenized_text = tokenizer.encode(text_data)
    print("Total tokens: ", len(tokenized_text))

    total_chars = len(text_data)

    # Split the text data into training and validation sets
    train_split = 0.9
    split_idx = int(total_chars * train_split)

    # Create the dataloaders for the training and validation sets
    train_loader = create_dataloader_v2(text_data[:split_idx], GPT_CONFIG_124M)
    val_loader = create_dataloader_v2(text_data[split_idx:], GPT_CONFIG_124M)

    # Set the device for the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_124M["learning_rate"])

    print("Untrained model sample: ")
    generate_and_print_sample(model, tokenizer, device, GPT_CONFIG_124M["start_context"])

    # Train the model
    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=GPT_CONFIG_124M["num_epochs"],
        eval_freq=5,
        eval_iter=10,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )
    epochs_tensor = torch.linspace(0, GPT_CONFIG_124M["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)

    print("Great, now give a sentence for it to complete: ")
    start_context = input()
    generate_and_print_sample(model, tokenizer, device, start_context)

if __name__ == "__main__":
    main()
