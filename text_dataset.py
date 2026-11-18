import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # tiktoken rejects special tokens unless allowed; basic Tokenizer has no such kwarg
        try:
            token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        except TypeError:
            token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]