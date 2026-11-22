from eval_common import main_guard, run_eval
from train_gpt2_moe_cosmopedia import MAX_SAMPLES, MODEL_NAME, SUBSETS, WANDB_PROJECT


def main():
    run_eval("Evaluate a GPT-2 MoE checkpoint trained by train_gpt2_moe_cosmopedia.py.", MODEL_NAME, WANDB_PROJECT, SUBSETS, MAX_SAMPLES)


if __name__ == "__main__":
    main_guard(main)
