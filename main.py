import tiktoken
import torch
from models import GPTModel
from utils import generate_text_simple

def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257, # Vocabulary size
        "context_length": 256, # Context length
        "embedding_dim": 768, # Embedding dimension
        "num_heads": 12, # Number of attention heads
        "num_layers": 12, # Number of layers
        "dropout_rate": 0.1, # Dropout rate
        "qkv_bias": False, # Query-Key-Value bias
        "batch_size": 4
    }
    start_text = "Hello, how are"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_text)
    encoded = torch.tensor(encoded).unsqueeze(0) # Creates the batch dimension
    print("Input text: ", start_text)
    print("Encoded Text: ", encoded)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval() # putting into eval mode
    out = generate_text_simple(model, encoded, max_new_tokens=100, context_size=GPT_CONFIG_124M["context_length"])
    print("Generated Text: ", tokenizer.decode(out[0].tolist()))

if __name__ == "__main__":
    main()
