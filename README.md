# 🧠 LLM Training Lab

**Self-contained toolkit for fine-tuning LLMs with LoRA/QLoRA, evaluating adapters with BLEU/ROUGE/perplexity, and running evolutionary hyperparameter search — all from a single config file.**

jays.website/appdevelopment/ · [JaysWebDev](https://github.com/JaysWebDev)

---

## 1. Overview

A research-grade training pipeline for fine-tuning open LLMs on instruction datasets. Designed to run on a single GPU or in 8-bit mode on consumer hardware. The experiment manager proposes hyperparameter candidates, runs them, and gates promotion behind human review.

**Stack:** Python · HuggingFace Transformers · PEFT · `bitsandbytes` · FastAPI

---

## 2. Features

- **LoRA/QLoRA fine-tuning** — configurable rank, alpha, dropout, 8-bit loading
- **Instruction dataset format** — JSONL with `instruction`, `input`, `output` fields
- **Self-evaluation** — BLEU, ROUGE-1/2/L, exact match, perplexity proxy, Markdown reports
- **Evolutionary experiment manager** — proposes hyperparameter candidates, runs each, requires human approval to promote
- **Config-driven runner** — one YAML controls train + eval, no code changes needed
- **Proposal review dashboard** — minimal FastAPI UI to approve/reject experiment proposals

---

## 3. Setup

```bash
git clone https://github.com/JaysWebDev/llm-training-lab
cd llm-training-lab
pip install transformers datasets peft accelerate bitsandbytes orjson fastapi uvicorn evaluate
```

---

## 4. Quick Start

```bash
# 1. Prepare your dataset as JSONL
# data/train.jsonl: {"instruction": "...", "input": "...", "output": "..."}

# 2. Edit configs/train_config.yaml — set base_model and train_file

# 3. Run training + eval in one command
python scripts_run_from_config.py --config configs/train_config.yaml --eval
```

Output adapters land in `adapters/`, eval reports in `reports/`.

---

## 5. Config Reference

```yaml
# configs/train_config.yaml
base_model: "meta-llama/Llama-3.1-8b"
train_file: "data/train.jsonl"
output_dir: "adapters/exp1"

lora:
  r: 8              # LoRA rank — higher = more capacity, more VRAM
  alpha: 16         # scaling factor (alpha/r = effective learning rate multiplier)
  dropout: 0.05

training:
  lr: 2.0e-4
  epochs: 3
  batch_size: 8
  max_length: 512
  load_in_8bit: true   # enables QLoRA — fits ~8B models on 12GB VRAM
  no_fp16: false
```

---

## 6. Evolutionary Experiment Manager

Proposes hyperparameter candidates from a search space, runs each, and holds for approval before promoting to the main adapter.

```bash
# Propose 3 candidates and auto-run them
python training/training_experiment_manager_evolver.py \
  --propose --n_candidates 3 \
  --base_model meta-llama/Llama-3.1-8b \
  --train_file data/train.jsonl \
  --auto_run --eval

# Start the approval dashboard at http://localhost:8002
uvicorn dashboard_review:APP --host 0.0.0.0 --port 8002
```

Candidates not approved in the dashboard are automatically discarded.

---

## 7. Stand-Alone Eval

Evaluate any adapter against a held-out JSONL file:

```bash
python training/training_self_eval_enhanced.py \
  --model_dir adapters/exp1/adapter \
  --eval_file data/eval.jsonl \
  --out_dir reports/exp1
```

Outputs `report.json` (machine-readable) and `report.md` (human-readable summary).

---

## 8. Dataset Format

```jsonl
{"instruction": "Summarize the following text.", "input": "Long article...", "output": "Short summary."}
{"instruction": "Answer the question.", "input": "What is LoRA?", "output": "Low-Rank Adaptation..."}
```

Empty `input` is fine for instruction-only tasks.

---

## 9. Hardware Requirements

| Config | Minimum VRAM |
|--------|-------------|
| 7B model, 8-bit | ~8 GB |
| 8B model, 8-bit | ~10 GB |
| 13B model, 8-bit | ~14 GB |
| Any model, fp16 | 2× the 8-bit requirement |

Tested on RTX 3060 (12GB) and RTX 3070 (8GB).

---

## 10. Troubleshooting

**`CUDA out of memory`** — Reduce `batch_size` to 1, enable `gradient_accumulation_steps: 8`, set `load_in_8bit: true`.

**`orjson` not found** — `pip install orjson`.

**`evaluate` module missing** — `pip install evaluate rouge-score sacrebleu`.

**Model download slow** — Set `HF_HUB_OFFLINE=1` after first download to use local cache.

---

jays.website/appdevelopment/ · [JaysWebDev](https://github.com/JaysWebDev)
