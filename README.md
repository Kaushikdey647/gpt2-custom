# GPT2 Custom

A from-scratch GPT-2–style decoder-only transformer in PyTorch, following the *Build a Large Language Model (From Scratch)* path: pretrain on next-token prediction (dense and 8-expert MoE variants), instruction fine-tune (full, LoRA or QLoRA), evaluate, generate, and publish to the Hugging Face Hub. Every training run is tracked in Weights & Biases.

Tokenizer is **tiktoken GPT-2** (`vocab_size=50257`). Checkpoints land in `model-data/` (gitignored).

## Setup

```bash
uv sync
wandb login            # once; or pass --no-wandb to any train/eval script
```

Requires Python ≥ 3.13. Training picks CUDA → MPS → CPU automatically. Hub upload needs `huggingface-cli login` (or `HF_TOKEN`).

All scripts run from the repo root as `uv run python <script>.py [flags]`. `--no-wandb` works on every train/eval script; `--help` lists all flags.

| Script | Role | Flags |
|--------|------|-------|
| `train_gpt2_cosmopedia.py` | Pretrain the dense model from scratch on Cosmopedia (`stanford`) | `--full`, `--no-wandb` |
| `train_gpt2_moe_cosmopedia.py` | Pretrain the 8-expert MoE model on six mixed Cosmopedia subsets | `--full`, `--no-wandb` |
| `train_gpt2_gsm8k.py` | Full fine-tune your dense checkpoint on GSM8K | |
| `train_openai_gpt2_gsm8k.py` | Load OpenAI GPT-2 weights, then fully fine-tune on GSM8K | |
| `train_gpt2_lora_gsm8k.py` | LoRA (or QLoRA) fine-tune of the dense checkpoint on GSM8K | `--qlora`, `--no-wandb` |
| `eval_gpt2_cosmopedia.py` / `eval_gpt2_moe_cosmopedia.py` | Per-subset + overall eval of the matching train script's model | `--model`, `--dataset {cosmopedia,gsm8k}`, `--full`, `--rows-per-subset`, `--num-batches`, `--prompt`, `--no-wandb` |
| `eval_gpt2_lora_gsm8k.py` | GSM8K loss for base vs base+adapter vs merged | `--qlora`, `--model`, `--no-wandb` |
| `inference.py` | Interactive generation / chat from a saved checkpoint | `--list`, `--model`, `--mode {auto,chat,complete}`, `--max-new-tokens` |
| `compare_generators.py` | Continue a held-out Cosmopedia doc with greedy, top-k and nucleus decoding | `--model` (required), `--subsets`, `--index`, `--prefix-tokens`, `--max-new-tokens`, `--temperature`, `--seed` |
| `upload_to_hub.py` | Push a checkpoint + generated model card (with eval results) to the Hub | `--model`, `--repo-id` (both required), `--card-only`, `--public` |
| `tune_batch.py` | Sweep batch size × context length for best throughput on this device | `--bs`, `--ctx`, `--steps`, `--warmup`, `--tol` |
| `test_lora.py` | Sanity checks for LoRA / NF4 / merging | |

Shared code lives in the `src/` package (scripts import it as `from src.utils import ...`; run scripts from the repo root): `src/utils.py` (train loop, checkpoints, loss, generation), `src/eval_common.py` (eval logic), `src/lora.py` (LoRA/QLoRA), `src/text_dataset.py` (datasets), `src/models.py`, `src/attention_utils.py`.

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

### MoE variant (`GPT2MoEModel`)

Same stack, but each block's MLP is replaced by a `MoeLayer`: a linear router picks the **top-k of N experts** per token (gate noise during training); each expert is a gated (SwiGLU-style) MLP. Defaults in `train_gpt2_moe_cosmopedia.py`: 8 experts, top-1, `expert_hidden_dim=768` → **275.6M total params, ~127M active per token**. Two auxiliary losses keep routing healthy: a load-balancing loss (`aux_loss_coef=0.01`) and a router z-loss (`z_loss_coef=0.001`). Router entropy and per-expert load are logged during training.

---

## Data

### Pretraining (language modeling)

- **Cosmopedia** (`HuggingFaceTB/cosmopedia`): the dense model uses `stanford`; the MoE mixes `stanford, khanacademy, openstax, wikihow, auto_math_text, stories` (10,000 docs per subset)
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

```bash
uv run python train_gpt2_cosmopedia.py        # dense  -> model-data/gpt-2-124m.pt
uv run python train_gpt2_moe_cosmopedia.py    # MoE    -> model-data/gpt-2-moe-8x-mixed.pt
# flags: --full (entire dataset, tokenised once into data-cache/), --no-wandb
```

| Setting | Dense | MoE |
|---------|-------|-----|
| Context / stride | 64 / 64 | 64 / 64 |
| Batch size · LR · epochs | 64 · `1e-4` AdamW · 2 | 64 · `1e-4` AdamW · 2 |
| Data | 5,000 `stanford` docs | 10,000 docs × 6 subsets |
| Train/val split | 90/10 of docs, seed 123 | same |
| Checkpoint | `gpt-2-124m.pt` | `gpt-2-moe-8x-mixed.pt` |

**Strategy**

1. Next-token CE on windowed text (MoE adds the aux + z losses)
2. ~200 evals over the run (`eval_every`) on train/val loaders; a loss plot is saved next to the checkpoint
3. Sample generation and a checkpoint (model + optimizer + losses + config) after each epoch — restart-safe; rerunning resumes
4. Grad clip `1.0`; fused AdamW on CUDA/MPS
5. **Ctrl+C once** pauses for an eval + sample then continues; **twice** aborts (progress since the last epoch checkpoint is lost)

Rename the model (`MODEL_NAME` in the script) to start a fresh checkpoint and W&B run for a new version.

---

## Fine-tune

### A. Full fine-tune of *your* Cosmopedia weights

```bash
uv run python train_gpt2_gsm8k.py        # needs model-data/gpt-2-124m.pt -> gpt-2-124m-gsm8k.pt
```

Loads base weights with `load_pretrained_weights` (can extend context; position embeddings beyond the base length keep init), context **512**, LR `5e-5`.

### B. Fine-tune from OpenAI GPT-2

```bash
uv run python train_openai_gpt2_gsm8k.py # -> gpt2-openai-gsm8k.pt  (downloads openai-community/gpt2, qkv_bias=True)
```

Both use the instruction template + response-only loss, a smaller LR than pretraining and 3 epochs.

### C. LoRA / QLoRA

```bash
uv run python train_gpt2_lora_gsm8k.py            # -> gpt-2-124m-gsm8k-lora.pt
uv run python train_gpt2_lora_gsm8k.py --qlora    # -> gpt-2-124m-gsm8k-qlora.pt
uv run python eval_gpt2_lora_gsm8k.py [--qlora]   # base vs base+adapter vs merged on GSM8K test
uv run python test_lora.py                        # sanity checks
```

`src/lora.py` is hand-rolled: `y = W x + (α/r)·B(A(dropout(x)))` with `A` Kaiming-init and `B = 0`, so training starts exactly at the base model. Defaults: `r=16, α=32, dropout=0.05`, applied to `W_q/W_k/W_v/out_proj` and both MLP layers, LR `2e-4`. The base is frozen; the position embeddings stay trainable because the 64→512 context extension leaves the new rows untrained.

- **QLoRA** stores the frozen block weights as 4-bit **NF4** (blockwise absmax, pure PyTorch so it runs on MPS/CPU) and dequantises on the fly. On this 124M model it cuts MPS memory ~680 → ~400 MiB at the same speed; the payoff grows with model size. On CUDA, `bitsandbytes` would be faster.
- Adapter checkpoints hold only the adapter + position embeddings (~12 MiB); the base is reloaded from `gpt-2-124m.pt`.
- `merge_lora()` folds adapters into plain `Linear` weights (dequantising an NF4 base) for deployment; merged and adapter outputs match to ~1e-5.

Not built yet: loading adapter checkpoints in `inference.py` (it only loads full checkpoints) and GGUF/llama.cpp export of the merged model.

---

## Evaluate

```bash
uv run python eval_gpt2_moe_cosmopedia.py     # evaluates gpt-2-moe-8x-mixed (no --model needed)
uv run python eval_gpt2_cosmopedia.py         # evaluates gpt-2-124m
# flags: --rows-per-subset N (default 50), --full, --dataset gsm8k, --model NAME, --prompt TEXT, --no-wandb
```

Each eval script scores its counterpart train script's checkpoint on **all six Cosmopedia subsets**: subsets the model trained on use held-out docs, the rest are random `unseen subset` docs. Output is per-subset and overall loss/perplexity (token-weighted), with `overall (held-out)` as the headline. Results go to `model-data/<model>-eval.json`, the W&B run (summary values, table, bar charts) and the Hub model card.

---

## Inference

```bash
uv run python inference.py --list
uv run python inference.py --model gpt-2-124m-gsm8k --mode chat
uv run python inference.py --model gpt-2-moe-8x-mixed --mode complete --max-new-tokens 100
uv run python compare_generators.py --model gpt-2-moe-8x-mixed --subsets stanford khanacademy openstax wikihow auto_math_text stories
```

`inference.py` is interactive (type `quit` to exit). `compare_generators.py` is non-interactive: it prints a held-out Cosmopedia doc, trims it to the first `--prefix-tokens` (default 10), then prints each generator's continuation (same seed per generator, `--temperature 0.8`). Pick another doc with `--index N`. `--subsets` must match the subsets the model trained on (default `stanford`; the MoE needs all six, as above) so the doc is truly held-out. Add a generator by putting it in the `GENERATORS` dict; it needs the signature `(model, idx, max_new_tokens, context_size, temperature=...)`.


| Mode | Behavior |
|------|----------|
| `auto` (default) | `chat` if the checkpoint name contains `gsm8k` / `chat` / `sft`, else `complete` |
| `chat` | Wraps each user turn in `CHAT_TEMPLATE`; independent Q&A |
| `complete` | Raw continuation; history accumulates until you type `clear` |

`inference.py` and the training-time samples use **greedy** decoding. `utils.py` also provides top-k and nucleus sampling, which `compare_generators.py` runs side by side on a held-out doc prefix.

---

## Experiment tracking (Weights & Biases)

One W&B **project per train script** (`gpt2-cosmopedia`, `gpt2-moe-cosmopedia`, `gpt2-lora-gsm8k`); each model/version is a **run** whose id is the model name, so resuming a checkpoint resumes its run and eval scripts attach their results to the same run.

Logged: full config (+ param counts, subsets, seed, device); every 10 steps train loss, CE loss (MoE), grad norm, clip flag, param norm, LR, tokens/s, epoch, device memory; per eval train/val loss, perplexity and gap; MoE aux/z loss, router entropy and per-expert load; per-epoch sample table and loss plot; final per-subset eval. Curves use a `step` axis so resumed epochs don't clash.

---

## Publish to the Hub

```bash
uv run python upload_to_hub.py --model gpt-2-moe-8x-mixed --repo-id USER/NAME [--public]
uv run python upload_to_hub.py --model gpt-2-moe-8x-mixed --repo-id USER/NAME --card-only   # README only
```

Uploads weights (`pytorch_model.bin`), `config.json`, the loss plot and a generated model card whose `model-index` carries per-subset perplexity/loss as Hub eval results. Run the eval script first so the card has results.

---

## Tools

```bash
uv run python tune_batch.py [--bs 8 16 32] [--ctx 64 128] [--steps 5] [--warmup 2] [--tol 0.05]
# sweeps batch × context in constant-tokens-per-step groups (8×64, 4×128, 2×256, then 16×64, ...)
```

Reports tokens/s per shape and the throughput optimum plus a "knee" (smallest tokens/step within 5% of best).

---

## Layout

```
src/                     # importable package: `from src.<module> import ...`
  attention_utils.py     # MultiHeadAttention, FeedForward, LayerNorm, TransformerBlock, Expert/MoeLayer/MOEBlock
  models.py              # GPTModel, GPT2MoEModel (+ a stale GPT2LoraWrapper superseded by lora.py)
  lora.py                # LoRALinear, NF4Linear, apply_lora, adapter save/load, merge_lora
  text_dataset.py        # TextDataset, ChatDataset, Cosmopedia/GSM8K dataloaders
  utils.py               # train loop, loss, checkpoints, W&B helpers, weight loading, generation
  eval_common.py         # shared per-subset eval; eval_gpt2_*.py are thin wrappers
train_*.py  eval_*.py    # entrypoints (repo root)
inference.py  compare_generators.py  upload_to_hub.py  tune_batch.py  test_lora.py
model-data/              # *.pt checkpoints, *-eval.json, *-losses.png (gitignored)
data-cache/              # tokenised --full datasets
```

---

## Typical workflow

```bash
uv sync && wandb login
uv run python train_gpt2_moe_cosmopedia.py          # pretrain, tracked in W&B
uv run python eval_gpt2_moe_cosmopedia.py           # per-subset eval -> W&B + eval json
uv run python upload_to_hub.py --model gpt-2-moe-8x-mixed --repo-id USER/NAME

uv run python train_gpt2_cosmopedia.py              # dense base
uv run python train_gpt2_lora_gsm8k.py              # LoRA on GSM8K (add --qlora for 4-bit base)
uv run python eval_gpt2_lora_gsm8k.py
```
