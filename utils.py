import torch
import tiktoken
import torch.nn as nn
from torch.utils.backcompat import keepdim_warning
from torch.utils.data import DataLoader
from verdict import VerdictDataset
from basic_tokenizer import Tokenizer

def start_emb_layer(vocab_size,output_dim):
    emb_layer = torch.nn.Embedding(vocab_size,output_dim)
    return emb_layer

def get_verdict_text():
    with open("the-verdict.txt", "r") as file:
        text = file.read()
    return text

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, num_workers=0, shuffle=True, drop_last=True):
    tokenizer = Tokenizer()
    dataset = VerdictDataset(text, tokenizer, max_length, stride)
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
    dataset = VerdictDataset(text, tokenizer, config["context_length"], config["context_length"] // 2)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=True)

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cropped = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cropped)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx