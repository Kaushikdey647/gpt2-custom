import argparse
import json
import os
import shutil
import tempfile

import torch
from huggingface_hub import HfApi

from utils import checkpoint_path, loss_plot_path


def main():
    parser = argparse.ArgumentParser(description="Upload a checkpoint from model-data/ to the HF Hub.")
    parser.add_argument("--model", required=True, help="Checkpoint name without .pt")
    parser.add_argument("--repo-id", required=True, help="e.g. username/gpt-2-moe-8x")
    parser.add_argument("--public", action="store_true", help="Create the repo as public (default: private)")
    args = parser.parse_args()

    checkpoint = torch.load(checkpoint_path(args.model), map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    with tempfile.TemporaryDirectory() as tmp:
        # Weights only — optimizer state is not needed for inference
        torch.save(checkpoint["model_state_dict"], os.path.join(tmp, "pytorch_model.bin"))
        if os.path.exists(loss_plot_path(args.model)):
            shutil.copy(loss_plot_path(args.model), os.path.join(tmp, "losses.png"))
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(
                f"# {args.model}\n\nCustom GPT-2 style model "
                f"({'MoE' if config.get('use_moe') else 'dense'}), trained from scratch.\n\n"
                f"- epochs: {checkpoint['epoch'] + 1}\n- val loss: {checkpoint.get('val_loss')}\n\n"
                "Load `pytorch_model.bin` into the model built from `config.json` using the source repo's `build_model`.\n"
            )

        api = HfApi()
        api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=tmp, commit_message=f"Upload {args.model}")
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
