"""
Config-driven runner script.
Reads configs/train_config.yaml and launches training + optional self-eval.
Usage:
  python scripts_run_from_config.py --config configs_train_config.yaml --eval
"""
import argparse
import os
import subprocess
import yaml

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["base_model"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]
    lora = cfg.get("lora", {})
    training = cfg.get("training", {})

    cmd = [
        "python", "training_train_adapter.py",
        "--base_model", base_model,
        "--train_file", train_file,
        "--output_dir", output_dir,
        "--r", str(lora.get("r", 8)),
        "--alpha", str(lora.get("alpha", 16)),
        "--dropout", str(lora.get("dropout", 0.05)),
        "--lr", str(training.get("lr", 2e-4)),
        "--epochs", str(training.get("epochs", 3)),
        "--batch_size", str(training.get("batch_size", 8)),
        "--max_length", str(training.get("max_length", 512)),
    ]

    if training.get("load_in_8bit", False):
        cmd.append("--load_in_8bit")
    if training.get("no_fp16", False):
        cmd.append("--no_fp16")

    print("[+] Running training:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if args.eval:
        eval_cfg = cfg.get("evaluation", {})
        eval_file = eval_cfg.get("eval_file", "data/eval.jsonl")
        out_dir = os.path.join(output_dir, "eval")

        eval_cmd = [
            "python", "training_self_eval_enhanced.py",
            "--model_dir", os.path.join(output_dir, "adapter"),
            "--eval_file", eval_file,
            "--out_dir", out_dir,
        ]

        print("[+] Running self-eval:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--eval", action="store_true", help="Run self-eval after training")
    args = ap.parse_args()
    main(args)
