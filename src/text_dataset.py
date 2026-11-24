import itertools
import os
import random

import numpy as np
import tiktoken
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset

IGNORE_INDEX = -100

CHAT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

CONVSINQA_INSTRUCTION = (
    "In the context of this series of interconnected finance-related queries "
    "and the additional information provided by the pretext, table data, and "
    "posttext from a company's financial filings, please provide a response "
    "to the final question. This may require extracting information from the "
    "context and performing mathematical calculations. Please take into account "
    "the information provided in the preceding questions and their answers when "
    "formulating your response:"
)


def _encode(tokenizer, text):
    try:
        return tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    except TypeError:
        return tokenizer.encode(text)


def _table_to_text(table):
    """Serialize list-of-lists or dict-of-dicts tables into plain text."""
    if table is None:
        return ""
    if isinstance(table, str):
        return table
    if isinstance(table, list):
        return "\n".join(" | ".join(str(c) for c in row) for row in table)
    if isinstance(table, dict):
        # column -> {row_header: value}
        lines = []
        for col, rows in table.items():
            if isinstance(rows, dict):
                for row_h, val in rows.items():
                    lines.append(f"{row_h} | {col} | {val}")
            else:
                lines.append(f"{col} | {rows}")
        return "\n".join(lines)
    return str(table)


def format_convfinqa_example(example):
    """Build (instruction, input, response) from ConvFinQA / FLARE-style rows.

    Supported shapes:
      1) HF FLARE / ICE-PIXIU: {"query": "...Question:...\\nAnswer:", "answer": "..."}
      2) Structured: context/pre_text/post_text/table + conversations + question + answer
      3) Raw ConvFinQA conversation: pre_text, post_text, table, annotation
         (uses last turn as the supervised question/answer by default)
    """
    # --- (1) Already-rendered query prompt ---
    if "query" in example and ("answer" in example or "response" in example):
        query = example["query"].rstrip()
        if query.endswith("Answer:"):
            query = query[: -len("Answer:")].rstrip()
        answer = example.get("answer") or example.get("response")
        return CONVSINQA_INSTRUCTION, query, str(answer)

    # --- (3) Raw ConvFinQA conversation record ---
    if "annotation" in example and ("pre_text" in example or "table" in example):
        ann = example["annotation"]
        questions = ann.get("dialogue_break") or []
        answers = ann.get("exe_ans_list") or []
        turn = example.get("turn_ind", len(questions) - 1)
        turn = max(0, min(turn, len(questions) - 1)) if questions else 0

        pre = example.get("pre_text") or []
        post = example.get("post_text") or []
        if isinstance(pre, list):
            pre = " ".join(pre)
        if isinstance(post, list):
            post = " ".join(post)
        table_txt = _table_to_text(example.get("table"))

        prior = []
        for i in range(turn):
            q = questions[i]
            a = answers[i] if i < len(answers) else ""
            prior.append(f"q{i}: {q} {a}")

        parts = [
            f"Context: {pre} {table_txt} {post}".strip(),
        ]
        if prior:
            parts.append("Conversations:\n" + "\n".join(prior))
        parts.append(f"Question: {questions[turn] if questions else ''}")
        inp = "\n".join(parts)
        resp = str(answers[turn]) if turn < len(answers) else str(example.get("exe_ans", ""))
        return CONVSINQA_INSTRUCTION, inp, resp

    # --- (2) Structured fields ---
    context = example.get("context")
    if context is None and any(k in example for k in ("pre_text", "post_text", "table")):
        pre = example.get("pre_text") or ""
        post = example.get("post_text") or ""
        if isinstance(pre, list):
            pre = " ".join(pre)
        if isinstance(post, list):
            post = " ".join(post)
        context = f"{pre} {_table_to_text(example.get('table'))} {post}".strip()

    question = example.get("question")
    answer = example.get("answer") or example.get("response") or example.get("output")
    if context is not None and question is not None and answer is not None:
        conv = example.get("conversations") or example.get("dialogue") or []
        prior_lines = []
        if isinstance(conv, str):
            prior_lines.append(conv)
        elif isinstance(conv, list):
            for i, turn in enumerate(conv):
                if isinstance(turn, str):
                    prior_lines.append(turn)
                elif isinstance(turn, dict):
                    q = turn.get("question") or turn.get("q") or ""
                    a = turn.get("answer") or turn.get("a") or ""
                    prior_lines.append(f"q{i}: {q} {a}")

        parts = [f"Context: {context}"]
        if prior_lines:
            parts.append("Conversations:\n" + "\n".join(prior_lines))
        parts.append(f"Question: {question}")
        return CONVSINQA_INSTRUCTION, "\n".join(parts), str(answer)

    return None


def format_chat_example(example):
    """Normalize Alpaca / messages / ConvFinQA rows into (instruction, input, response)."""
    convfinqa = format_convfinqa_example(example)
    if convfinqa is not None:
        return convfinqa

    if "messages" in example:
        user_parts = []
        assistant = None
        for msg in example["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "human", "instruction"):
                user_parts.append(content)
            elif role in ("assistant", "gpt", "response") and assistant is None:
                assistant = content
        if not user_parts or assistant is None:
            raise ValueError(f"Could not parse messages example: {example!r}")
        return "\n".join(user_parts), "", assistant

    instruction = example.get("instruction") or example.get("prompt") or example.get("question") or ""
    inp = example.get("input") or ""
    response = (
        example.get("output")
        or example.get("response")
        or example.get("answer")
        or example.get("text")
        or ""
    )
    if not instruction or not response:
        raise ValueError(
            "Example needs instruction/prompt + output/response, messages, "
            f"or ConvFinQA fields. Got keys={list(example.keys())}"
        )
    return instruction, inp, response


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = _encode(tokenizer, text)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class MemmapDataset(Dataset):
    """Windowed LM dataset over a uint16 token file; nothing is held in RAM."""

    def __init__(self, path, max_length, stride):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_length = max_length
        self.stride = stride
        self.n = len(range(0, len(self.tokens) - max_length, stride))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.stride
        chunk = torch.from_numpy(self.tokens[i : i + self.max_length + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


class ChatDataset(Dataset):
    """Single-turn instruction SFT dataset.

    Each example is formatted, truncated/padded to max_length. Targets use
    IGNORE_INDEX on prompt + padding so CE loss only trains the response.

    Long prompts (e.g. ConvFinQA) are left-truncated so the question + answer
    still fit in the window.
    """

    def __init__(
        self,
        examples,
        tokenizer,
        max_length,
        template=CHAT_TEMPLATE,
        ignore_index=IGNORE_INDEX,
        pad_token_id=50256,  # GPT-2 <|endoftext|>
    ):
        self.input_ids = []
        self.target_ids = []
        response_marker = "### Response:\n"

        for example in examples:
            instruction, inp, response = format_chat_example(example)
            text = template.format(
                instruction=instruction.strip(),
                input=inp.strip() if inp else "None",
                response=response.strip(),
            )
            if not text.endswith("<|endoftext|>"):
                text = text + "<|endoftext|>"

            ids = _encode(tokenizer, text)
            prompt_text = text.split(response_marker, 1)[0] + response_marker
            prompt_len = len(_encode(tokenizer, prompt_text))

            # Left-truncate so the question + answer still fit in the window
            if len(ids) > max_length:
                dropped = len(ids) - max_length
                ids = ids[-max_length:]
                prompt_len = max(0, prompt_len - dropped)

            seq_len = len(ids)
            if seq_len < max_length:
                ids = ids + [pad_token_id] * (max_length - seq_len)

            targets = []
            for i in range(max_length):
                next_i = i + 1
                if next_i >= max_length or next_i >= seq_len or next_i < prompt_len:
                    targets.append(ignore_index)
                else:
                    targets.append(ids[next_i])

            self.input_ids.append(torch.tensor(ids, dtype=torch.long))
            self.target_ids.append(torch.tensor(targets, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_lm(text, config: dict, shuffle=True, drop_last=True):
    """Windowed next-token LM dataloader over a single text string."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(text, tokenizer, config["context_length"], config["stride"])
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
    )


def _load_cosmopedia(subsets, max_samples=None, seed=123, with_subset=False):
    """Load one or more Cosmopedia subsets and concatenate; max_samples applies per subset.

    with_subset adds a "subset" column; it doesn't change row order, so shuffles/splits are identical.
    """
    subsets = [subsets] if isinstance(subsets, str) else list(subsets)
    parts = []
    for name in subsets:
        ds = load_dataset("HuggingFaceTB/cosmopedia", name, split="train")
        if max_samples is not None:
            ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
        ds = ds.select_columns(["text"])
        if with_subset:
            ds = ds.add_column("subset", [name] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts)


def create_dataloader_cosmopedia(
    config: dict,
    subset="stanford",
    max_samples: int = 5000,
    val_ratio: float = 0.1,
    seed: int = 123,
):
    """LM dataloaders from Cosmopedia `text` column.

    Full stanford split is ~1M rows — subset first. Only `train` exists, so we
    carve out val. Docs are joined with <|endoftext|> then windowed.
    """
    raw = _load_cosmopedia(subset, max_samples, seed).shuffle(seed=seed)
    split = raw.train_test_split(test_size=val_ratio, seed=seed)

    train_text = "<|endoftext|>".join(split["train"]["text"])
    val_text = "<|endoftext|>".join(split["test"]["text"])

    train_loader = create_dataloader_lm(train_text, config, shuffle=True, drop_last=True)
    val_loader = create_dataloader_lm(val_text, config, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _tokenize_to_bin(texts, path, batch_size=10_000, max_tokens=None):
    """Tokenize an iterable of docs (<|endoftext|> after each) into a uint16 file.

    max_tokens stops (truncating the last doc) once that many tokens are written; `texts` may be
    a shared iterator so a later call continues where this one stopped.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token
    tmp = path + ".tmp"
    written = 0
    done = False
    if max_tokens is not None:
        batch_size = max(1, min(batch_size, max_tokens // 2000))  # don't read far past a small budget

    def flush(f, batch):
        nonlocal written, done
        for ids in tokenizer.encode_ordinary_batch(batch):
            ids = ids + [eot]
            if max_tokens is not None and written + len(ids) >= max_tokens:
                ids = ids[: max_tokens - written]
                done = True
            np.array(ids, dtype=np.uint16).tofile(f)
            written += len(ids)
            if done:
                return

    with open(tmp, "wb") as f:
        batch = []
        for i, text in enumerate(texts, 1):
            batch.append(text)
            if len(batch) == batch_size:
                flush(f, batch)
                batch = []
                print(f"Tokenized {i} docs ({written:,} tokens) -> {path}")
                if done:
                    break
        if batch and not done:
            flush(f, batch)
    os.replace(tmp, path)
    return written


def create_dataloader_cosmopedia_full(
    config: dict,
    subset="stanford",
    val_ratio: float = 0.001,
    seed: int = 123,
    cache_dir: str = "data-cache",
):
    """Loaders over the entire Cosmopedia subset. Tokenizes once into cache_dir, then memory-maps."""
    os.makedirs(cache_dir, exist_ok=True)
    tag = subset if isinstance(subset, str) else "+".join(subset)
    paths = {k: os.path.join(cache_dir, f"cosmopedia-{tag}-{k}.bin") for k in ("train", "val")}
    if not all(os.path.exists(p) for p in paths.values()):
        raw = _load_cosmopedia(subset)
        split = raw.train_test_split(test_size=val_ratio, seed=seed)
        for key, ds in (("val", split["test"]), ("train", split["train"])):
            _tokenize_to_bin((row["text"] for row in ds), paths[key])

    train_ds = MemmapDataset(paths["train"], config["context_length"], config["stride"])
    val_ds = MemmapDataset(paths["val"], config["context_length"], config["stride"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, drop_last=False)
    return train_loader, val_loader


def create_dataloader_chat(
    examples,
    config: dict,
    val_ratio: float = 0.1,
    seed: int = 123,
    shuffle_train: bool = True,
):
    """Train/val loaders for instruction SFT from a list of example dicts."""
    if not examples:
        raise ValueError("examples is empty")

    rng = random.Random(seed)
    examples = list(examples)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_ratio)) if len(examples) > 1 else 0
    val_examples = examples[:n_val]
    train_examples = examples[n_val:] or examples

    tokenizer = tiktoken.get_encoding("gpt2")
    max_length = config["context_length"]
    train_ds = ChatDataset(train_examples, tokenizer, max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
        num_workers=0,
        drop_last=len(train_ds) >= config["batch_size"],
    )
    if val_examples:
        val_ds = ChatDataset(val_examples, tokenizer, max_length)
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    else:
        val_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    return train_loader, val_loader


def create_dataloader_gsm8k(config: dict):
    """Instruction SFT loaders for openai/gsm8k: `train` for training, `test` for eval."""
    raw = load_dataset("openai/gsm8k", "main")
    tokenizer = tiktoken.get_encoding("gpt2")
    max_length = config["context_length"]
    train_ds = ChatDataset(raw["train"], tokenizer, max_length)
    val_ds = ChatDataset(raw["test"], tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, drop_last=False)
    return train_loader, val_loader


def _stream(path, column="text", buffer_size=1_000, **kw):
    def make(seed):
        ds = load_dataset(path, split="train", streaming=True, **kw)
        return (r[column] for r in ds.shuffle(seed=seed, buffer_size=buffer_size))

    return make


def _stream_cosmopedia(seed, subsets=("stanford", "khanacademy", "openstax", "wikihow", "auto_math_text", "stories")):
    return itertools.chain.from_iterable(_stream("HuggingFaceTB/cosmopedia", name=s)(seed) for s in subsets)


# name -> (share of token budget, stream factory(seed) -> text iterator, tokenize batch size).
# Shares sum to 1; books are whole volumes so they use small batches.
MIXED_SOURCES = {
    "fineweb": (0.40, _stream("HuggingFaceFW/fineweb", name="sample-10BT"), 10_000),
    "fineweb-edu": (0.25, _stream("HuggingFaceFW/fineweb-edu", name="sample-10BT"), 10_000),
    "cosmopedia": (0.20, _stream_cosmopedia, 10_000),
    "books": (0.10, _stream("emozilla/pg19", buffer_size=8), 16),  # small buffer: each doc is a whole book
    "wikipedia": (0.025, _stream("wikimedia/wikipedia", name="20231101.en"), 1_000),
    "code": (0.025, _stream("codeparrot/codeparrot-clean", column="content"), 1_000),
}


def create_dataloader_mixed(
    config: dict,
    total_tokens: int = 1_000_000_000,
    val_tokens: int = 2_000_000,
    seed: int = 123,
    cache_dir: str = "data-cache",
    sources: dict = None,
):
    """Loaders over a weighted corpus mix (see MIXED_SOURCES), sized by token budget.

    Each source is streamed, tokenized once into train/val .bin files under cache_dir (budget =
    share * total_tokens; val = share * val_tokens), then memory-mapped. Train windows are shuffled
    across sources by the DataLoader. Changing total_tokens/val_tokens re-tokenizes.
    """
    sources = sources or MIXED_SOURCES
    os.makedirs(cache_dir, exist_ok=True)
    train_parts, val_parts = [], []
    for name, (share, make, batch_size) in sources.items():
        paths = {
            k: os.path.join(cache_dir, f"mix-{name}-{total_tokens}-{val_tokens}-{k}.bin") for k in ("train", "val")
        }
        if not all(os.path.exists(p) for p in paths.values()):
            texts = make(seed)
            _tokenize_to_bin(texts, paths["val"], batch_size, max_tokens=int(share * val_tokens))
            _tokenize_to_bin(texts, paths["train"], batch_size, max_tokens=int(share * total_tokens))
        for key, parts in (("train", train_parts), ("val", val_parts)):
            parts.append(MemmapDataset(paths[key], config["context_length"], config["stride"]))

    train_loader = DataLoader(
        ConcatDataset(train_parts), batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        ConcatDataset(val_parts), batch_size=config["batch_size"], shuffle=False, num_workers=0, drop_last=False
    )
    return train_loader, val_loader
