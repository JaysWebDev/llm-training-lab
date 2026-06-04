"""
Enhanced self-evaluation script for trained adapters.
Computes BLEU, ROUGE, exact match, and a simple perplexity proxy.
Outputs JSON and Markdown reports.
Usage:
  python training_self_eval_enhanced.py --model_dir adapters/demo/adapter --eval_file data/eval.jsonl --out_dir reports/demo
"""
import argparse
import os
import json
import orjson
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    import evaluate
except Exception:
    evaluate = None

def load_jsonl(path: str):
    data = []
    with open(path, "rb") as f:
        for line in f:
            data.append(orjson.loads(line))
    return data

def exact_match(pred, gold):
    return int(pred.strip() == gold.strip())

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[+] Loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=128)

    eval_data = load_jsonl(args.eval_file)
    results = []

    rouge = evaluate.load("rouge") if evaluate else None
    bleu = evaluate.load("bleu") if evaluate else None

    total_em = 0
    rouge_scores = []
    bleu_scores = []
    perp_list = []

    import torch

    for ex in eval_data:
        q = ex.get("instruction", "")
        if ex.get("input"):
            q = q + "\n" + ex.get("input", "")
        expected = ex.get("output", "").strip()

        gen_out = pipe(q)[0]["generated_text"]
        generated = gen_out[len(q):].strip() if gen_out.startswith(q) else gen_out.strip()

        em = exact_match(generated, expected)
        total_em += em

        if rouge:
            r = rouge.compute(predictions=[generated], references=[expected])
            rouge_scores.append(r)

        if bleu:
            b = bleu.compute(predictions=[generated.split()], references=[[expected.split()]])
            bleu_scores.append(b.get("bleu", 0.0))

        try:
            enc = tokenizer(generated, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
                loss = out.loss.item()
                perp = math.exp(loss)
            perp_list.append(perp)
        except Exception:
            perp_list.append(None)

        entry = {
            "question": q,
            "expected": expected,
            "generated": generated,
            "exact_match": em,
            "perplexity": perp_list[-1],
        }
        results.append(entry)

    n = len(results)
    avg_em = total_em / n if n else 0.0
    avg_perp = sum([p for p in perp_list if p is not None]) / len([p for p in perp_list if p is not None]) if any(p is not None for p in perp_list) else None

    summary = {
        "n_samples": n,
        "exact_match_rate": avg_em,
        "avg_perplexity": avg_perp,
    }

    if rouge_scores:
        agg = {}
        for item in rouge_scores:
            for k, v in item.items():
                agg.setdefault(k, []).append(v)
        summary["rouge"] = {k: sum(v)/len(v) for k, v in agg.items()}

    if bleu_scores:
        summary["bleu_mean"] = sum(bleu_scores)/len(bleu_scores)

    report = {"summary": summary, "examples": results[:10]}

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(args.out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(f"# Self-Eval Report\\n\\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\\n\\n## Examples\\n")
        for r in results[:10]:
            f.write(f"Q: {r['question']}\\n\\nA: {r['generated']}\\n\\nExpected: {r['expected']}\\n\\nExact Match: {r['exact_match']}\\n\\n---\\n")

    print(f"[✓] Report written to {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to fine-tuned adapter")
    ap.add_argument("--eval_file", required=True, help="JSONL eval set")
    ap.add_argument("--out_dir", required=True, help="Directory to write report")
    args = ap.parse_args()
    main(args)
