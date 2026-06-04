"""
LoRA fine-tuning script for Hugging Face Transformers.
Input dataset: JSONL with {instruction, input, output}
Output: adapter directory + training logs + eval metrics
"""
import argparse
import os
import json
import orjson
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

def load_jsonl(path: str):
    data = []
    with open(path, "rb") as f:
        for line in f:
            data.append(orjson.loads(line))
    return data

def format_example(ex):
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    prompt = f"Instruction: {instruction}\\nInput: {inp}\\nResponse: "
    text = prompt + out + "\\n"
    return {"text": text}

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[+] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        device_map="auto"
    )

    print(f"[+] Loading dataset: {args.train_file}")
    raw = load_jsonl(args.train_file)
    ds = list(map(format_example, raw))
    dataset = Dataset.from_list(ds)

    def tok(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"]).with_format("torch")

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["c_attn"],  # DialoGPT uses c_attn, not q_proj/v_proj
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=not args.no_fp16,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("[+] Starting training...")
    trainer.train()

    print("[+] Saving adapter and tokenizer")
    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "output_dir": args.output_dir,
        "params": vars(args),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[✓] Training complete. Adapter saved to {args.output_dir}/adapter")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Base model name (HF hub)")
    ap.add_argument("--train_file", required=True, help="Path to JSONL training data")
    ap.add_argument("--output_dir", required=True, help="Directory to save adapter")
    ap.add_argument("--r", type=int, default=8, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit model loading")
    ap.add_argument("--no_fp16", action="store_true", help="Disable fp16")
    args = ap.parse_args()
    main(args)
