"""Train a LoRA adapter from JSONL data on a GPU machine."""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATA_PATH = os.getenv("DATA_PATH", "data/train.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "artifacts/adapter")
HF_TOKEN = os.getenv("HF_TOKEN")


def main() -> None:
    model_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **model_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", **model_kwargs
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=2048),
        remove_columns=ds.column_names,
    )

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            save_strategy="epoch",
            logging_steps=50,
            report_to="none",
        ),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    ).train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved → {OUTPUT_DIR}")
    print("Next: aws s3 sync artifacts/adapter/ s3://<bucket>/adapter/")


if __name__ == "__main__":
    main()
