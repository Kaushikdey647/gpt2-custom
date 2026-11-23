import argparse
import glob
import os
import sys

import tiktoken
import torch
from src.text_dataset import CHAT_TEMPLATE
from src.utils import (
    build_model,
    checkpoint_path,
    generate_text_simple,
    get_device,
    text_to_token_ids,
    token_ids_to_text,
)

MAX_NEW_TOKENS = 200
# Checkpoints whose names contain these were instruction-tuned with CHAT_TEMPLATE
CHAT_MODEL_MARKERS = ("gsm8k", "chat", "sft")


def list_models():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("model-data/*.pt"))


def choose_model(models):
    print("Available models:")
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input(f"Pick a model [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        if choice in models:
            return choice
        print("Invalid choice.")


def load_model(model_name, device):
    path = checkpoint_path(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"Checkpoint {path} has no config — retrain with current code.")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"Loaded {path} (epoch={checkpoint.get('epoch')}, "
        f"step={checkpoint.get('global_step')}, context={config['context_length']}, device={device})"
    )
    return model


def generate(model, tokenizer, device, prompt, max_new_tokens):
    context_size = model.position_embedding.weight.shape[0]
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        out = generate_text_simple(model, idx, max_new_tokens=max_new_tokens, context_size=context_size)
    new_tokens = out[:, idx.shape[1]:]
    text = token_ids_to_text(new_tokens, tokenizer)
    return text.split("<|endoftext|>", 1)[0]


def chat_loop(model, tokenizer, device, max_new_tokens):
    print("Chat mode (each question is answered independently). 'quit' or empty line to stop.\n")
    while True:
        user = read_input()
        if user is None:
            break
        prompt = CHAT_TEMPLATE.format(instruction=user, input="None", response="")
        reply = generate(model, tokenizer, device, prompt, max_new_tokens)
        # Stop if the model starts a new instruction block
        reply = reply.split("### Instruction:", 1)[0].strip()
        print(f"Model: {reply}\n")


def completion_loop(model, tokenizer, device, max_new_tokens):
    print("Completion mode. 'clear' resets context, 'quit' or empty line to stop.\n")
    history = ""
    while True:
        user = read_input()
        if user is None:
            break
        if user.lower() == "clear":
            history = ""
            print("(context cleared)\n")
            continue
        prompt = f"{history}\n{user}" if history else user
        reply = generate(model, tokenizer, device, prompt, max_new_tokens).strip()
        print(f"Model: {reply}\n")
        history = f"{prompt} {reply}"


def read_input():
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not user or user.lower() in {"quit", "exit", "q"}:
        return None
    return user


def main():
    parser = argparse.ArgumentParser(description="Chat with a checkpoint from model-data/.")
    parser.add_argument("--model", help="Checkpoint name without .pt (prompts if omitted)")
    parser.add_argument(
        "--mode",
        choices=("auto", "chat", "complete"),
        default="auto",
        help="chat wraps input in the instruction template; auto picks chat for fine-tuned models",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    models = list_models()
    if not models:
        raise FileNotFoundError("No checkpoints in model-data/. Train a model first.")
    if args.list:
        print("\n".join(models))
        return

    model_name = args.model or choose_model(models)
    mode = args.mode
    if mode == "auto":
        mode = "chat" if any(m in model_name for m in CHAT_MODEL_MARKERS) else "complete"

    device = get_device()
    model = load_model(model_name, device)
    tokenizer = tiktoken.get_encoding("gpt2")

    if mode == "chat":
        chat_loop(model, tokenizer, device, args.max_new_tokens)
    else:
        completion_loop(model, tokenizer, device, args.max_new_tokens)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
