"""
Fine-tune Mistral-7B-Instruct with QLoRA for task decomposition.

This script is designed for Kaggle (T4 GPU) and local runs. It:
- Loads dataset robustly (Kaggle or local data/)
- Applies prompt-masked tokenization for causal LM
- Configures 4-bit QLoRA + LoRA modules
- Splits train/eval and evaluates with ROUGE-L + BLEU
- Plots training curves and saves metrics
"""

import os
import json
import random
import math
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import matplotlib.pyplot as plt
# Optional evaluation library; we'll install or skip gracefully if missing
try:
    import evaluate as eval_lib
except Exception:
    eval_lib = None


# ===========================
# 1. Config & environment
# ===========================
set_seed(42)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0"
MAX_SEQ_LEN = 512

# Robust base dir handling for scripts and notebooks where __file__ may be undefined
def get_base_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

BASE_DIR = get_base_dir()

# Detect Kaggle working dir and set output paths
KAGGLE_WORKING = "/kaggle/working"
OUT_DIR = (
    os.path.join(KAGGLE_WORKING, "mistral-lora-task")
    if os.path.exists(KAGGLE_WORKING)
    else os.path.join(BASE_DIR, "..", "outputs", "mistral-lora-task")
)
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================
# 2. Load dataset (Kaggle or local)
# ===========================
def load_dataset() -> List[Dict]:
    """Load raw JSON dataset from Kaggle input or local data folder."""
    possible_paths = [
        "/kaggle/input/task-dataset/tasks.json",
        "/kaggle/input/task-dataset/tasks_dataset.json",
        os.path.join(BASE_DIR, "..", "data", "tasks_dataset.json"),
        os.path.join(BASE_DIR, "..", "data", "Generated dataset.json"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "Dataset not found. Place tasks_dataset.json in data/ or Kaggle input."
    )


raw_data = load_dataset()


# ===========================
# 3. Prepare dataset for instruction tuning
# ===========================
train_examples: List[Dict] = []

for item in raw_data:
    # Support multiple potential schema shapes
    task_name = (
        item.get("task name")
        or item.get("task")
        or item.get("title")
        or ""
    )
    task_name = (task_name or "").strip()

    subtasks_list = item.get("subtask") or item.get("subtasks") or []

    if not task_name or len(subtasks_list) == 0:
        continue

    # Pull subtask name fields robustly
    def _sub_name(s):
        return (s.get("subtask name") or s.get("name") or s.get("title") or "").strip()

    numbered = [f"{i+1}. {_sub_name(s)}" for i, s in enumerate(subtasks_list) if _sub_name(s)]
    if not numbered:
        continue

    subtasks_text = "\n".join(numbered)

    input_text = (
        "### Instruction:\n"
        "Break the following task into clear, ordered subtasks.\n\n"
        f"### Task:\n{task_name}\n"
    )

    train_examples.append({"input_text": input_text, "output_text": subtasks_text})

dataset_all = Dataset.from_list(train_examples)
split_dataset = dataset_all.train_test_split(test_size=0.1, seed=42)
train_ds = split_dataset["train"]
eval_ds = split_dataset["test"]


# ===========================
# 4. Load tokenizer with chat formatting
# ===========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Ensure padding token available for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def build_prompt(user_text: str) -> str:
    user_text = user_text.strip()
    # Prefer the model's chat template
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback to Mistral-style tokens
    return f"[INST] {user_text} [/INST]"


def tokenize_example(example: Dict) -> Dict:
    user = example["input_text"].strip()
    assistant = example["output_text"].strip()

    prompt = build_prompt(user)
    full_text = prompt + assistant + tokenizer.eos_token

    # Tokenize full text with padding
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )

    # Tokenize prompt without padding to get mask length
    prompt_ids = tokenizer(
        prompt, truncation=True, max_length=MAX_SEQ_LEN, padding=False
    )["input_ids"]
    labels = full["input_ids"].copy()

    # Mask prompt tokens
    prompt_len = len(prompt_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    # Mask padding tokens
    labels = [(-100 if tok == tokenizer.pad_token_id else tok) for tok in labels]

    full["labels"] = labels
    return full


train_tok = train_ds.map(tokenize_example, remove_columns=["input_text", "output_text"]) 
eval_tok = eval_ds.map(tokenize_example, remove_columns=["input_text", "output_text"]) 


# ===========================
# 5. Load base model in 4-bit (QLoRA)
# ===========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # T4 supports fp16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# Prepare for k-bit training (gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # required for gradient checkpointing


# ===========================
# 6. Apply LoRA
# ===========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


# ===========================
# 7. Data collator
# ===========================
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict]):
        input_ids = [ex["input_ids"] for ex in batch]
        attention_mask = [ex["attention_mask"] for ex in batch]
        labels = [ex["labels"] for ex in batch]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


data_collator = DataCollatorForCausalLM(tokenizer)


# ===========================
# 8. Training setup
# ===========================
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=25,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    eval_steps=200,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# ===========================
# 9. Train
# ===========================
train_result = trainer.train()
trainer.save_model(OUT_DIR)


# ===========================
# 10. Evaluation (ROUGE-L, BLEU) via generation
# ===========================
# Ensure evaluation library is available; attempt lightweight install if not
if eval_lib is None:
    try:
        import sys, subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "evaluate", "sacrebleu"], check=False)
        import evaluate as eval_lib
    except Exception:
        eval_lib = None

if eval_lib is not None:
    rouge = eval_lib.load("rouge")
    bleu = eval_lib.load("sacrebleu")
else:
    rouge = None
    bleu = None

def generate_output(prompt: str, max_new_tokens: int = 256) -> str:
    model.eval()
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt.strip()}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = f"[INST] {prompt.strip()} [/INST]"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # Try to extract assistant portion after prompt
    return full_text.replace(input_text, "").strip()


def evaluate_model(eval_dataset: Dataset, max_samples: int = 100) -> Dict:
    preds = []
    refs = []
    n = min(max_samples, len(eval_dataset))
    for i in range(n):
        ex = eval_dataset[i]
        prompt = ex["input_text"]
        ref = ex["output_text"]
        pred = generate_output(prompt)
        preds.append(pred)
        refs.append(ref)
    results = {}
    if rouge is not None:
        try:
            results.update(rouge.compute(predictions=preds, references=refs))
        except Exception:
            pass
    if bleu is not None:
        try:
            bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
            results["bleu"] = bleu_res.get("score", 0.0)
        except Exception:
            pass
    if not results:
        # Minimal fallback: average token overlap ratio
        def overlap(a: str, b: str) -> float:
            sa, sb = set(a.split()), set(b.split())
            return len(sa & sb) / max(1, len(sb))
        results["token_overlap"] = sum(overlap(p, r) for p, r in zip(preds, refs)) / max(1, len(refs))
    return results


metrics = evaluate_model(split_dataset["test"], max_samples=100)

# Save metrics JSON
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("✅ Evaluation metrics:", metrics)


# ===========================
# 11. Plot training curves
# ===========================
def plot_training_curves(trainer_obj: Trainer, out_dir: str):
    logs = trainer_obj.state.log_history
    steps, losses, eval_steps, eval_losses = [], [], [], []
    for entry in logs:
        if "loss" in entry and "epoch" in entry:
            steps.append(entry.get("step", len(steps)))
            losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(8, 5))
    if losses:
        plt.plot(steps, losses, label="train loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Training/Eval Loss")
    plt.legend()
    fig_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"📈 Saved training curves to {fig_path}")


plot_training_curves(trainer, OUT_DIR)


# ===========================
# 12. Example inference
# ===========================
example_task = "Build a personal finance tracker app"
example_prompt = (
    "### Instruction:\n"
    "Break the following task into clear, ordered subtasks.\n\n"
    f"### Task:\n{example_task}\n"
)

predicted = generate_output(example_prompt, max_new_tokens=256)
print("\nGenerated subtasks:\n", predicted)

