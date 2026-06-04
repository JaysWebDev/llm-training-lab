"""
Experiment Manager
==================
Proposes candidate hyperparameter sets for LoRA training.
Runs each candidate, logs metrics, and requires human approval to promote.

Usage:
  python training/experiment_manager.py --propose --n_candidates 3 --base_model meta-llama/Llama-3.1-8b --train_file data/train.jsonl
"""

import argparse
import json
import os
import random
from datetime import datetime


SEARCH_SPACE = {
    "r": [4, 8, 16],
    "alpha": [8, 16, 32],
    "lr": [5e-5, 2e-4, 5e-4],
    "dropout": [0.05, 0.1],
}


class ExperimentManager:
    def __init__(self, base_model, train_file, out_dir):
        self.base_model = base_model
        self.train_file = train_file
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def propose_candidates(self, n=3):
        candidates = []
        for _ in range(n):
            cand = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
            cand["batch_size"] = random.choice([1, 2, 4, 8])
            candidates.append(cand)
        return candidates

    def save_proposals(self, candidates):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir, f"proposals_{ts}.json")
        data = {
            "base_model": self.base_model,
            "train_file": self.train_file,
            "candidates": candidates,
            "timestamp": ts,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[✓] Proposals saved to {path}")
        return path

    def run_candidate(self, cand):
        """
        Placeholder: actually call train_adapter.py with these args.
        For safety, we only log here. Human can decide to launch.
        """
        print("[>] Candidate:", cand)
        # TODO: integrate subprocess call to train_adapter.py with args.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true", help="Propose new candidates")
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out_dir", default="experiments")
    args = ap.parse_args()

    mgr = ExperimentManager(args.base_model, args.train_file, args.out_dir)

    if args.propose:
        cands = mgr.propose_candidates(args.n_candidates)
        mgr.save_proposals(cands)
        for c in cands:
            mgr.run_candidate(c)
