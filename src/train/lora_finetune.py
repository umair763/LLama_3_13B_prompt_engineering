from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


@dataclass
class Record:
    instruction: str
    input: str
    output: str


PROMPT_TEMPLATE = (
    "SYSTEM:\n"
    "You are an expert task planner. Break the task into actionable subtasks (30–90 minutes). Output JSON only.\n\n"
    "USER:\nTask: \"{task}\"\nDeadline: \"{deadline}\"\nDescription: \"{description}\"\n\n"
    "JSON:\n"
)


def build_prompt(rec: Dict[str, Any]) -> str:
    task = rec.get("task") or rec.get("title") or ""
    deadline = rec.get("deadline") or ""
    description = rec.get("description") or ""
    return PROMPT_TEMPLATE.format(task=task, deadline=deadline, description=description)


def format_example(rec: Dict[str, Any]) -> str:
    prompt = build_prompt(rec)
    target = rec.get("expectedJson") or rec.get("output") or rec.get("expectedSubtasks") or {}
    if isinstance(target, (dict, list)):
        target = json.dumps(target, ensure_ascii=False)
    return prompt + str(target)


def make_dataset(path: str):
    # Supports JSONL or JSON array
    if path.endswith(".jsonl"):
        data = [json.loads(line) for line in open(path, "r", encoding="utf-8").read().splitlines() if line.strip()]
    else:
        data = json.load(open(path, "r", encoding="utf-8"))
    texts = [format_example(rec) for rec in data]
    return texts


def tokenize_fn(tokenizer, texts, max_len: int):
    return tokenizer(texts, truncation=True, max_length=max_len, return_tensors=None)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for subtask generation")
    parser.add_argument("--base-model", required=True, help="HF model id or local path")
    parser.add_argument("--train-data", required=True, help="Path to JSON/JSONL training data")
    parser.add_argument("--output-dir", required=True, help="Where to save LoRA adapters")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=["q_proj", "v_proj"])  # common for many LLMs
    model = get_peft_model(model, lora_cfg)

    texts = make_dataset(args.train_data)
    tokenized = tokenize_fn(tokenizer, texts, args.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, tokenized):
            self.input_ids = tokenized["input_ids"]
            self.attn = tokenized["attention_mask"]
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return {"input_ids": torch.tensor(self.input_ids[idx]), "attention_mask": torch.tensor(self.attn[idx])}

    ds = ListDataset(tokenized)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapters + tokenizer (small, good for offline use)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Note: base weights are not saved; only adapters.

if __name__ == "__main__":
    main()
