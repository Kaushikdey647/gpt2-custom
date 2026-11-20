# GPT2 Custom

A from-scratch GPT-2–style decoder-only transformer in PyTorch, following the *Build a Large Language Model (From Scratch)* path: pretrain on next-token prediction, then instruction fine-tune, then generate.

Tokenizer is **tiktoken GPT-2** (`vocab_size=50257`). Checkpoints land in `model-data/`.

## Setup

```bash
uv sync
# or: pip install -e .
```

Requires Python ≥ 3.13. Training picks CUDA → MPS → CPU automatically.

| Script | Role |
|--------|------|
| `train_gpt2_cosmopedia.py` | Pretrain from scratch on Cosmopedia |
| `train_gpt2_gsm8k.py` | Fine-tune your Cosmopedia checkpoint on GSM8K |
| `train_openai_gpt2_gsm8k.py` | Load OpenAI GPT-2 weights, then fine-tune on GSM8K |
| `inference.py` | Interactive generation / chat from a saved checkpoint |

---

## Architecture

`GPTModel` is a **decoder-only** stack (no encoder, no cross-attention).

```
token ids
   │
   ├─ token embedding      (50257 × d)
   └─ position embedding   (context_length × d)
   │
   └─ dropout
   │
   └─ N × TransformerBlock   (pre-norm)
         LayerNorm → masked multi-head attention → residual
         LayerNorm → GELU MLP (4× width)         → residual
   │
   └─ LayerNorm
   └─ linear head → logits  (seq × vocab)
```

Default **124M-class** shape (same widths as GPT-2 small):

| Hyperparam | Value |
|------------|-------|
| `embedding_dim` | 768 |
| `num_layers` | 12 |
| `num_heads` | 12 |
| `head_dim` | 64 |
| MLP | `Linear → GELU(tanh) → Linear` (4× expand) |
| Norm | custom pre-norm `LayerNorm` |
| Attention | causal multi-head; upper triangle masked to `-inf` |
| `qkv_bias` | `False` for scratch models; `True` when loading OpenAI GPT-2 |

Forward returns logits `[batch, seq, vocab]`. Loss is token-level cross-entropy (with `ignore_index=-100` for instruction SFT so prompt/pad tokens are not trained).

See `TRANSFORMER.md` for a deeper attention walkthrough.

---

## Data

### Pretraining (language modeling)

- **Cosmopedia** (`HuggingFaceTB/cosmopedia`, subset `stanford` by default)
- Documents are shuffled, optionally capped (`max_samples`), split train/val, joined with `<|endoftext|>`, then cut into fixed windows (`TextDataset` with `context_length` / `stride`)
- Targets are next-token shifts of the inputs (standard LM)

### Fine-tuning (instruction SFT)

- **GSM8K** (`openai/gsm8k`) via `ChatDataset`
- Each row is rendered with:

```text
### Instruction:
{instruction}

### Input:
{input or None}

### Response:
{answer}<|endoftext|>
```

- Sequences are padded/truncated to `context_length`
- Loss is **masked**: only response tokens contribute (`IGNORE_INDEX` on prompt + padding)

`create_dataloader_chat` / ConvFinQA-style formatters in `text_dataset.py` support other instruction shapes if you want to plug in custom JSON lists.

---

## Pretrain

Train a small GPT from scratch on Cosmopedia:

```bash
uv run python train_gpt2_cosmopedia.py
```

Defaults in that script:

| Setting | Value | Notes |
|---------|-------|-------|
| Context / stride | 64 / 64 | Short windows for speed; raise for better LM |
| Batch size | 8 | |
| LR | `1e-4` AdamW | |
| Epochs | 5 | |
| Samples | 5000 of Cosmopedia stanford | Bump `max_samples` for a real run |
| Checkpoint | `model-data/gpt-2-124m.pt` | Resumes automatically if present |

**Strategy (pretrain)**

1. Next-token CE on windowed text
2. Periodic eval on train/val loaders (`eval_freq` / `eval_iter`)
3. Sample generation after each epoch
4. Checkpoint after each full epoch (optimizer + losses + config) — restart-safe
5. Grad clip `1.0` by default

Expect weak prose at context 64 and 5k docs; this path is for learning the loop, not matching GPT-2 quality. Scale `context_length`, `max_samples`, and epochs when you have GPU time.

---

## Fine-tune

Two paths:

### A. Fine-tune *your* Cosmopedia weights

```bash
# needs model-data/gpt-2-124m.pt from pretraining
uv run python train_gpt2_gsm8k.py
```

- Loads base weights with `load_pretrained_weights` (can extend context; position embeddings beyond the base length keep init)
- Context bumped to **512**, LR lowered to **`5e-5`**, `qkv_bias=False`
- Writes `model-data/gpt-2-124m-gsm8k.pt`

### B. Fine-tune from OpenAI GPT-2

```bash
uv run python train_openai_gpt2_gsm8k.py
```

- Downloads HF `openai-community/gpt2` and maps weights into `GPTModel` (`qkv_bias=True`)
- Same GSM8K SFT setup → `model-data/gpt2-openai-gsm8k.pt`

**Strategy (fine-tune)**

1. Start from a pretrained LM (scratch Cosmopedia or OpenAI)
2. Instruction template + response-only loss
3. Smaller LR than pretrain; fewer epochs (default 3)
4. Eval prompts use the chat template with an empty `### Response:` so the model continues as the assistant
5. Same checkpoint/resume loop as pretrain

---

## Inference

```bash
uv run python inference.py
uv run python inference.py --list
uv run python inference.py --model gpt-2-124m-gsm8k --mode chat
uv run python inference.py --model gpt-2-124m --mode complete --max-new-tokens 100
```

| Mode | Behavior |
|------|----------|
| `auto` (default) | `chat` if the checkpoint name contains `gsm8k` / `chat` / `sft`, else `complete` |
| `chat` | Wraps each user turn in `CHAT_TEMPLATE`; independent Q&A (no multi-turn history) |
| `complete` | Raw continuation; history accumulates until you type `clear` |

**Strategy (inference)**

- Greedy argmax decoding (`generate_text_simple`) — deterministic, no sampling
- Context cropped to the model’s `context_length`
- Stops visually at `<|endoftext|>`; chat mode also cuts if a new `### Instruction:` appears
- Device: CUDA → MPS → CPU

For creative sampling you’d add temperature / top-p in `generate_text_simple`; the training path currently uses greedy for eval samples too.

---

## Layout

```
attention_utils.py   # MultiHeadAttention, FeedForward, LayerNorm, TransformerBlock
models.py            # GPTModel
text_dataset.py      # TextDataset, ChatDataset, dataloaders
utils.py             # train loop, loss, checkpoints, weight loading, generation helpers
inference.py         # CLI chat / completion
train_gpt2_*.py      # entrypoints
model-data/          # *.pt checkpoints (gitignored contents may vary)
```

---

## Typical workflow

```bash
uv sync
uv run python train_gpt2_cosmopedia.py      # → gpt-2-124m.pt
uv run python train_gpt2_gsm8k.py           # → gpt-2-124m-gsm8k.pt
uv run python inference.py --model gpt-2-124m-gsm8k --mode chat
```

Or skip scratch pretrain and go straight to OpenAI init + GSM8K:

```bash
uv run python train_openai_gpt2_gsm8k.py
uv run python inference.py --model gpt2-openai-gsm8k
```
