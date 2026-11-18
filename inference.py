import os
import sys

import tiktoken
import torch
from models import GPTModel
from utils import checkpoint_path, generate_text_simple, text_to_token_ids, token_ids_to_text

MODEL_NAME = "gpt-2-124m"
MAX_NEW_TOKENS = 80

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint(model_name):
    candidates = [
        checkpoint_path(model_name),
        os.path.join("model-data", f"{model_name}-latest.pt"),
        os.path.join("model-data", f"{model_name}-best.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No checkpoint for {model_name!r} in model-data/. "
        f"Tried: {', '.join(candidates)}. Train with main.py first."
    )


def load_model(model_name, device):
    path = resolve_checkpoint(model_name)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"Checkpoint {path} has no config — retrain with current code.")
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"Loaded {path} (epoch={checkpoint.get('epoch')}, "
        f"step={checkpoint.get('global_step')}, device={device})"
    )
    return model, config


def complete(model, tokenizer, device, prompt, max_new_tokens=MAX_NEW_TOKENS):
    context_size = model.position_embedding.weight.shape[0]
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        out = generate_text_simple(
            model, idx, max_new_tokens=max_new_tokens, context_size=context_size
        )
    full = token_ids_to_text(out, tokenizer)
    # Return only the newly generated suffix when possible
    if full.startswith(prompt):
        return full[len(prompt):].replace("<|endoftext|>", "").strip()
    return full.replace("<|endoftext|>", "").strip()


def main():
    device = get_device()
    model, _config = load_model(MODEL_NAME, device)
    tokenizer = tiktoken.get_encoding("gpt2")

    print("Chat (completion). Empty line or 'quit' / 'exit' to stop. 'clear' resets context.\n")
    history = ""

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user or user.lower() in {"quit", "exit", "q"}:
            break
        if user.lower() == "clear":
            history = ""
            print("(context cleared)\n")
            continue

        prompt = f"{history}{user}" if not history else f"{history}\n{user}"
        reply = complete(model, tokenizer, device, prompt)
        print(f"Model: {reply}\n")
        history = f"{prompt}{reply}"


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
