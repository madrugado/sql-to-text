#!/usr/bin/env python3
"""
Train a model for SQL-to-text generation.

Supports both causal LMs (Qwen) and seq2seq models (RuT5/T5).
Fine-tunes on PAUQ dataset to generate natural language questions from SQL queries.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)

# Model configuration
DEFAULT_MODEL_NAME = "cointegrated/rut5-base"
OUTPUT_DIR = "./output/sql_to_text"
DATA_DIR = "./data"

# Training configuration
MAX_LENGTH = 512
BATCH_SIZE = 4  # Adjust based on your GPU memory
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    model_name: str = DEFAULT_MODEL_NAME
    output_dir: str = OUTPUT_DIR
    max_length: int = MAX_LENGTH
    batch_size: int = BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


def load_pauq_data(data_dir: str, split: str = "train") -> List[Dict]:
    """Load PAUQ dataset from JSON file."""
    filename = f"pauq_{split}.json"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from {filename}")
    return data


def prepare_training_data(data: List[Dict], is_seq2seq: bool = False) -> List[Dict]:
    """
    Prepare data for fine-tuning.

    For seq2seq (RuT5/T5): separate input and target
    For causal LM: single text format
    """
    prepared = []

    for item in data:
        sql_query = item.get("query", {}).get("en", "")
        question = item.get("question", {}).get("en", "")

        if not sql_query or not question:
            continue

        # Clean up the SQL query
        sql_query = sql_query.strip()
        question = question.strip()

        if is_seq2seq:
            # Seq2seq format: separate input and target
            prepared.append({
                "input": f"SQL: {sql_query}",
                "target": question
            })
        else:
            # Causal LM format: single text with instruction
            formatted_text = f"SQL: {sql_query}\nQuestion: {question}"
            prepared.append({"text": formatted_text})

    print(f"Prepared {len(prepared)} training examples")
    return prepared


def tokenize_function(examples, tokenizer, max_length, is_seq2seq: bool = False):
    """Tokenize the text data."""
    if is_seq2seq:
        # Seq2seq: tokenize input and target separately
        inputs = tokenizer(
            examples["input"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        targets = tokenizer(
            examples["target"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        inputs["labels"] = targets["input_ids"]
        # Set padding token labels to -100 (ignore loss)
        labels = inputs["labels"]
        labels = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
            for labels_seq in labels
        ]
        inputs["labels"] = labels
        return inputs
    else:
        # Causal LM: single text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized


def prepare_datasets(tokenizer, config: TrainingConfig, is_seq2seq: bool = False):
    """Prepare train and validation datasets."""
    print("Loading training data...")
    train_data = load_pauq_data(DATA_DIR, "train")
    train_prepared = prepare_training_data(train_data, is_seq2seq=is_seq2seq)

    print("Loading validation data...")
    val_data = load_pauq_data(DATA_DIR, "dev")
    val_prepared = prepare_training_data(val_data, is_seq2seq=is_seq2seq)

    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_prepared)
    val_dataset = Dataset.from_list(val_prepared)

    # Determine columns to remove based on format
    cols_to_remove = ["input", "target"] if is_seq2seq else ["text"]

    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_length, is_seq2seq=is_seq2seq),
        batched=True,
        remove_columns=cols_to_remove,
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_length, is_seq2seq=is_seq2seq),
        batched=True,
        remove_columns=cols_to_remove,
    )

    return train_dataset, val_dataset


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer. Returns model, tokenizer, and is_seq2seq flag."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Try seq2seq first, fallback to causal LM
    try:
        from transformers import GenerationConfig
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        is_seq2seq = True
        print("Detected: Seq2Seq model (encoder-decoder architecture)")
    except (OSError, ValueError, KeyError):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            is_seq2seq = False
            print("Detected: Causal LM (decoder-only architecture)")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, is_seq2seq


def train_model(config: TrainingConfig):
    """Main training function."""
    print("Starting SQL-to-text training...")

    # Load model and tokenizer (auto-detects model type)
    model, tokenizer, is_seq2seq = load_model_and_tokenizer(config.model_name)

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(tokenizer, config, is_seq2seq=is_seq2seq)

    # Data collator
    if is_seq2seq:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )
        TrainerClass = Seq2SeqTrainer
        TrainingArgsClass = Seq2SeqTrainingArguments
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        TrainerClass = Trainer
        TrainingArgsClass = TrainingArguments

    # Training arguments
    training_args = TrainingArgsClass(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        predict_with_generate=True if is_seq2seq else False,
    )

    # Create trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("Training completed!")
    return trainer


def evaluate_model(model_path: str, num_samples: int = 5):
    """Evaluate the trained model with sample predictions."""
    print(f"\nEvaluating model: {model_path}")

    # Load trained model (auto-detect model type)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Try seq2seq first, fallback to causal LM
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        is_seq2seq = True
    except (OSError, ValueError, KeyError):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            is_seq2seq = False
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    val_data = load_pauq_data(DATA_DIR, "dev")

    print("\nSample predictions:")
    print("=" * 80)

    for i, item in enumerate(val_data[:num_samples]):
        sql_query = item.get("query", {}).get("en", "")
        actual_question = item.get("question", {}).get("en", "")

        # Format input
        prompt = f"SQL: {sql_query}\nQuestion:"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if is_seq2seq:
            # Seq2Seq generates the target directly
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            predicted_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Causal LM generates continuation
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_question = full_output.split("Question:")[-1].strip()

        print(f"\n--- Sample {i+1} ---")
        print(f"SQL: {sql_query}")
        print(f"Expected: {actual_question}")
        print(f"Predicted: {predicted_question}")
        print("-" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train model for SQL-to-text")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model name (e.g., cointegrated/rut5-base, Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="both",
        help="Run mode: train, eval, or both",
    )
    parser.add_argument(
        "--model-path",
        default=OUTPUT_DIR,
        help="Path to trained model for evaluation",
    )

    args = parser.parse_args()

    config = TrainingConfig(model_name=args.model)

    if args.mode in ["train", "both"]:
        trainer = train_model(config)

    if args.mode in ["eval", "both"]:
        model_path = args.model_path if args.mode == "eval" else config.output_dir
        evaluate_model(model_path)


if __name__ == "__main__":
    main()
