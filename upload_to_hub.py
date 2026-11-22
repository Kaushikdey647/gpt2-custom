import argparse
import json
import os
import shutil
import tempfile

import torch
from huggingface_hub import EvalResult, HfApi, ModelCardData

from eval_common import eval_results_path
from utils import checkpoint_path, loss_plot_path


def model_card(model_name, checkpoint):
    config = checkpoint["config"]
    kind = "MoE" if config.get("use_moe") else "dense"
    path = eval_results_path(model_name)
    ev = json.load(open(path)) if os.path.exists(path) else None
    results = ev["results"] if ev else None
    train_subsets = ", ".join(ev["train_subsets"]) if ev else "Cosmopedia"

    eval_results = []
    if results:
        for name, r in results.items():
            if name == "overall_all":
                continue  # mixes held-out and unseen subsets; the table in the body shows it
            overall = name == "overall_in_dist"
            for metric_type, metric_name, value in (
                ("perplexity", "Perplexity", r["ppl"]),
                ("loss", "Cross-entropy loss (nats/token)", r["loss"]),
            ):
                eval_results.append(
                    EvalResult(
                        task_type="text-generation",
                        task_name="Text Generation",
                        dataset_type="HuggingFaceTB/cosmopedia",
                        dataset_name="Cosmopedia (held-out, all trained subsets)" if overall else f"Cosmopedia ({name})",
                        dataset_config=None if overall else name,
                        dataset_split="train",  # Cosmopedia only ships `train`; eval docs are carved out of it
                        dataset_args={
                            "eval_split": "held-out" if overall else r["split"],
                            "docs_per_subset": ev["rows_per_subset"],
                            "context_length": ev["context_length"],
                            "tokens": r["tokens"],
                        },
                        metric_type=metric_type,
                        metric_name=metric_name,
                        metric_value=round(value, 4),
                    )
                )
    tags = ["gpt2", "text-generation"] + (["mixture-of-experts"] if kind == "MoE" else [])
    meta = "---\n" + ModelCardData(
        model_name=model_name,
        library_name="pytorch",
        pipeline_tag="text-generation",
        language="en",
        datasets=["HuggingFaceTB/cosmopedia"],
        metrics=["perplexity", "loss"] if eval_results else None,
        tags=tags,
        eval_results=eval_results or None,
    ).to_yaml() + "\n---"

    lines = [meta, "", f"# {model_name}", "",
             f"Custom GPT-2 style {kind} model, trained from scratch on Cosmopedia ({train_subsets}).", "",
             "## Model", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {config[k]} |" for k in ("num_layers", "num_heads", "embedding_dim", "context_length", "vocab_size")
              if k in config]
    if kind == "MoE":
        lines += [f"| experts / top-k | {config['num_experts']} / {config['top_k']} |",
                  f"| expert hidden dim | {config['expert_hidden_dim']} |"]
    lines += ["", "## Training", "",
              f"- epochs: {checkpoint['epoch'] + 1} ({checkpoint['global_step'] + 1:,} steps, {checkpoint['tokens_seen']:,} tokens)",
              f"- batch size {config.get('batch_size')}, lr {config.get('learning_rate')}, AdamW",
              f"- last in-training val loss: {checkpoint.get('val_loss'):.4f}", "", "![losses](losses.png)", ""]
    if results:
        lines += ["## Evaluation", "",
                  f"{ev['rows_per_subset']} Cosmopedia docs per subset, context {ev['context_length']}. "
                  "`held-out` rows are docs of a training subset that were never trained on; `unseen subset` rows come from "
                  "subsets the model was not trained on. Loss is token-weighted cross-entropy (nats); overall rows are "
                  "token-weighted. The headline (model-index) number is overall (held-out).",
                  "", "| subset | split | tokens | loss | perplexity |", "|---|---|---:|---:|---:|"]
        for name, r in results.items():
            if name.startswith("overall"):
                lines.append(f"| **{r['split']}** | | {r['tokens']:,} | **{r['loss']:.3f}** | **{r['ppl']:.2f}** |")
            else:
                lines.append(f"| {name} | {r['split']} | {r['tokens']:,} | {r['loss']:.3f} | {r['ppl']:.2f} |")
        lines.append("")
    lines += ["## Usage", "",
              "Load `pytorch_model.bin` into the model built from `config.json` using the source repo's `build_model`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Upload a checkpoint from model-data/ to the HF Hub.")
    parser.add_argument("--model", required=True, help="Checkpoint name without .pt")
    parser.add_argument("--repo-id", required=True, help="e.g. username/gpt-2-moe-8x")
    parser.add_argument("--card-only", action="store_true", help="Only update README.md (skips re-uploading weights)")
    parser.add_argument("--public", action="store_true", help="Create the repo as public (default: private)")
    args = parser.parse_args()

    checkpoint = torch.load(checkpoint_path(args.model), map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    if args.card_only:
        HfApi().upload_file(
            path_or_fileobj=model_card(args.model, checkpoint).encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message="Update model card with eval results",
        )
        print(f"Updated card: https://huggingface.co/{args.repo_id}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        # Weights only — optimizer state is not needed for inference
        torch.save(checkpoint["model_state_dict"], os.path.join(tmp, "pytorch_model.bin"))
        if os.path.exists(loss_plot_path(args.model)):
            shutil.copy(loss_plot_path(args.model), os.path.join(tmp, "losses.png"))
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(model_card(args.model, checkpoint))

        api = HfApi()
        api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=tmp, commit_message=f"Upload {args.model}")
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
