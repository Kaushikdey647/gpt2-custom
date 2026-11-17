import torch
import tiktoken
import torch.nn as nn
from torch.utils.backcompat import keepdim_warning
from torch.utils.data import DataLoader
from models import GPTModel
from text_dataset import TextDataset
from basic_tokenizer import Tokenizer
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def start_emb_layer(vocab_size,output_dim):
    emb_layer = torch.nn.Embedding(vocab_size,output_dim)
    return emb_layer

def get_text():
    with open("the-text.txt", "r") as file:
        text = file.read()
    return text

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, num_workers=0, shuffle=True, drop_last=True):
    tokenizer = Tokenizer()
    dataset = TextDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return dataloader

def create_dataloader_v2(text, config: dict):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(text, tokenizer, config["context_length"], config["context_length"] // 2)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=True)

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0):
    # For a simple text generation, lets generate the text token by token
    for _ in range(max_new_tokens):
        # Crop the input to the context size
        idx_cropped = idx[:, -context_size:]
        with torch.no_grad():
            # Forward pass and get the logits
            logits = model(idx_cropped)
            # Divide the logits by the temperature
            logits = logits[:, -1, :] / temperature
            # Apply the softmax function to get the probabilities of each token
            probs = torch.softmax(logits, dim=-1)
            # Argmax chooses the most likely next token
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            # Concatenate the next token to the input
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Creates the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    text = tokenizer.decode(token_ids.squeeze(0).tolist())
    return text

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
    epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if(len(data_loader) == 0):
        return 0.0
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter, start_context, tokenizer):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model: GPTModel, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Generated Text: {decoded_text.replace('<|endoftext|>', '').replace('\n', ' ').strip()}")
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses = []
    val_losses = []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = -1
    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        print("Training...")
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, start_context, tokenizer)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"Tokens Seen: {tokens_seen}")
                print("-" * 10)
            
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen
