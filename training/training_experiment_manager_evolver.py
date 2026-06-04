"""
Experiment Manager - Evolutionary extension
Proposes hyperparameter candidates using an evolutionary strategy that mutates top performers.
Optionally run candidates locally using your config-driven runner (scripts/run_from_config.py).
Stores proposals, trials, and simple evaluation scores under `experiments/` for history.
Usage:
  python training_experiment_manager_evolver.py --propose --n_candidates 4
"""
import argparse
import copy
import json
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

SEARCH_SPACE = {
    "r": [4, 8, 16, 32],
    "alpha": [8, 16, 32, 64],
    "lr": [5e-5, 1e-4, 2e-4, 5e-4],
    "dropout": [0.0, 0.05, 0.1],
    "batch_size": [1, 2, 4, 8],
}

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def load_history():
    runs = []
    for p in sorted(EXPERIMENTS_DIR.glob("run_*")):
        meta = p / "meta.json"
        if meta.exists():
            try:
                with open(meta, "r") as f:
                    m = json.load(f)
                    m["run_dir"] = str(p)
                    runs.append(m)
            except Exception:
                continue
    return runs

def score_from_eval(report_dir: Path):
    report_json = report_dir / "report.json"
    if not report_json.exists():
        return None
    try:
        with open(report_json, "r") as f:
            rep = json.load(f)
        examples = rep.get("examples", [])
        if not examples:
            return None
        consistent_count = sum(1 for e in examples if e.get("exact_match"))
        return consistent_count / len(examples)
    except Exception:
        return None

def propose_random(n):
    cands = []
    for _ in range(n):
        cand = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        cands.append(cand)
    return cands

def propose_from_history(n, history, exploit_top_k=3, mutate_prob=0.4):
    if not history:
        return propose_random(n)
    scored = [r for r in history if r.get("score") is not None]
    if not scored:
        return propose_random(n)
    scored_sorted = sorted(scored, key=lambda x: x.get("score", 0), reverse=True)
    top = scored_sorted[:exploit_top_k]
    proposals = []
    while len(proposals) < n:
        parent = random.choice(top)
        params = parent.get("params", {})
        child = copy.deepcopy(params)
        for k in SEARCH_SPACE.keys():
            if random.random() < mutate_prob:
                child[k] = random.choice(SEARCH_SPACE[k])
        if "batch_size" not in child:
            child["batch_size"] = random.choice(SEARCH_SPACE["batch_size"])
        proposals.append(child)
    return proposals

def write_proposal(proposal, base_config, out_dir: Path):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = out_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base_config)
    lora = cfg.setdefault("lora", {})
    training = cfg.setdefault("training", {})
    lora["r"] = int(proposal.get("r", lora.get("r", 8)))
    lora["alpha"] = int(proposal.get("alpha", lora.get("alpha", 16)))
    lora["dropout"] = float(proposal.get("dropout", lora.get("dropout", 0.05)))
    training["lr"] = float(proposal.get("lr", training.get("lr", 2e-4)))
    training["epochs"] = int(training.get("epochs", 3))
    training["batch_size"] = int(proposal.get("batch_size", training.get("batch_size", 8)))
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    meta = {
        "proposal": proposal,
        "cfg_path": str(cfg_path),
        "created_at": ts,
        "status": "proposed",
    }
    with open(run_dir / "proposal.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Proposal written to {run_dir}")
    return run_dir

def run_proposal(run_dir: Path, eval_after=False):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("missing config.yaml in proposal folder")
    cmd = ["python", "scripts/run_from_config.py", "--config", str(cfg_path)]
    if eval_after:
        cmd.append("--eval")
    run_log = run_dir / "run.log"
    print(f"[+] Running training: {' '.join(cmd)}")
    with open(run_log, "wb") as out:
        process = subprocess.Popen(cmd, stdout=out, stderr=out)
        ret = process.wait()
    meta_file = run_dir / "meta.json"
    meta = {
        "status": "completed" if ret == 0 else "failed",
        "return_code": ret,
        "finished_at": datetime.utcnow().isoformat(),
    }
    eval_score = None
    eval_dir = run_dir / "eval"
    if eval_dir.exists():
        eval_score = score_from_eval(eval_dir)
    adapter_manifest = run_dir / "adapter_manifest.json"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg.get("output_dir", "adapters/exp1"))
    manifest_src = out_dir / "manifest.json"
    if manifest_src.exists():
        try:
            shutil.copy2(manifest_src, adapter_manifest)
            with open(adapter_manifest, "r") as f:
                m = json.load(f)
            meta["params"] = m.get("params")
            meta["base_model"] = m.get("base_model")
        except Exception:
            pass
    meta["score"] = eval_score
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Run finished. status={meta['status']}, score={eval_score}")
    return meta

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true")
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--base_config", default="configs/train_config.yaml")
    ap.add_argument("--exploit_top_k", type=int, default=3)
    ap.add_argument("--mutate_prob", type=float, default=0.4)
    ap.add_argument("--auto_run", action="store_true", help="If set, immediately run proposed candidates")
    ap.add_argument("--eval", action="store_true", help="Run self-eval after training (only used with --auto_run)")
    args = ap.parse_args()
    with open(args.base_config, "r") as f:
        base_cfg = yaml.safe_load(f)
    history = load_history()
    proposals = []
    if not history:
        proposals = propose_random(args.n_candidates)
    else:
        proposals = propose_from_history(args.n_candidates, history, exploit_top_k=args.exploit_top_k, mutate_prob=args.mutate_prob)
    run_dirs = []
    for p in proposals:
        rd = write_proposal(p, base_cfg, EXPERIMENTS_DIR)
        run_dirs.append(rd)
    if args.auto_run:
        for rd in run_dirs:
            meta = run_proposal(rd, eval_after=args.eval)
    print("[✓] Proposal generation complete. Review experiments/ for details.")

if __name__ == "__main__":
    main()
