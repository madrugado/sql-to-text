#!/usr/bin/env python3
"""
Evaluate a small Qwen model on the dev set from the data folder.

This script loads a Qwen model and evaluates it on the PAUQ dev set,
computing metrics like BLEU, ROUGE, CHRF, and LaBSE similarity.
Results are saved to a CSV file with model name and data count.
"""

import argparse
import csv
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm
from sacrebleu.metrics import CHRF

# Model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = "./data"

# Generation configuration
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DO_SAMPLE = True
NUM_BEAMS = 1


def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)


def load_dev_data(data_dir: str) -> List[Dict]:
    """Load PAUQ dev dataset from JSON file."""
    filepath = os.path.join(data_dir, "pauq_dev.json")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dev data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from dev set")
    return data


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Try to load as Seq2Seq model (for T5, BART, etc.)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model.model_type = "seq2seq"
    except ValueError:
        # Fall back to CausalLM model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model.model_type = "causal"

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def generate_question(
    model,
    tokenizer,
    sql_query: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
) -> str:
    """Generate a question from SQL query using few-shot prompting."""

    # Few-shot examples in Russian
    examples = """Вы генератор SQL-вопросов. Преобразуйте SQL-запросы в естественные вопросы на русском языке.

Примеры:

SQL: SELECT count(*) FROM singer;
Вопрос: Сколько у нас певцов?

SQL: SELECT name ,  country ,  age FROM singer ORDER BY age DESC;
Вопрос: Покажите имя, страну и возраст всех певцов, отсортированных по возрасту от старшего к младшему.

SQL: SELECT name FROM singer WHERE country = 'USA';
Вопрос: Каковы имена певцов из США?

SQL: SELECT count(*) FROM album WHERE singer_id = 1;
Вопрос: Сколько альбомов у певца 1?

SQL: {sql_query}
Вопрос:"""

    prompt = examples.format(sql_query=sql_query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Different generation params for seq2seq vs causal models
        if getattr(model, "model_type", None) == "seq2seq":
            # T5/BART style
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
            )
            # Decode the generated output (no need to split)
            predicted_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Causal model style
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_question = full_output.split("Вопрос:")[-1].strip()

    # Clean up: stop at newlines that suggest the model is continuing
    predicted_question = predicted_question.split("\n")[0].strip()

    return predicted_question


def compute_bleu_score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute BLEU score between references and hypotheses."""
    smoothing = SmoothingFunction()

    # Tokenize references and hypotheses
    ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
    hyp_tokens = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]

    # Calculate BLEU-1 to BLEU-4
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        try:
            score = corpus_bleu(
                [[ref] for ref in ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=smoothing.method1,
            )
            bleu_scores[f"BLEU-{n}"] = score * 100
        except Exception:
            bleu_scores[f"BLEU-{n}"] = 0.0

    return bleu_scores


def compute_chrf_score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute CHRF score between references and hypotheses."""
    chrf = CHRF()

    # CHRF expects the reference as a single string with ||| delimiters
    ref_str = " ||| ".join(refs for refs in references)
    hyp_str = "\n".join(hypotheses)

    # Compute CHRF score
    result = chrf.corpus_score(hyp_str, [ref_str])

    return {
        "CHRF": result.score,
        "CHRF+": result.score,  # CHRF+ is same as CHRF for single references
    }


def compute_labse_similarity(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute LaBSE similarity between references and hypotheses.

    LaBSE (Language-agnostic BERT Sentence Embedding) uses multilingual sentence
    embeddings to compute cosine similarity.
    """
    print("Loading LaBSE model...")
    labse_model_name = "sentence-transformers/LaBSE"
    labse_tokenizer = AutoTokenizer.from_pretrained(labse_model_name)
    labse_model = AutoModel.from_pretrained(labse_model_name)

    # Move to same device as the main model
    labse_device = torch.device("cpu")  # Use CPU to avoid MPS memory issues

    labse_model = labse_model.to(labse_device)
    labse_model.eval()

    similarities = []

    print("Computing LaBSE similarities...")
    with torch.no_grad():
        for ref, hyp in zip(references, hypotheses):
            # Tokenize
            ref_inputs = labse_tokenizer(ref, return_tensors="pt", padding=True, truncation=True, max_length=128)
            hyp_inputs = labse_tokenizer(hyp, return_tensors="pt", padding=True, truncation=True, max_length=128)

            # Move to device
            ref_inputs = {k: v.to(labse_device) for k, v in ref_inputs.items()}
            hyp_inputs = {k: v.to(labse_device) for k, v in hyp_inputs.items()}

            # Get embeddings
            ref_outputs = labse_model(**ref_inputs)
            hyp_outputs = labse_model(**hyp_inputs)

            # Use mean pooling
            ref_embedding = ref_outputs.last_hidden_state.mean(dim=1)
            hyp_embedding = hyp_outputs.last_hidden_state.mean(dim=1)

            # Compute cosine similarity
            similarity = F.cosine_similarity(ref_embedding, hyp_embedding)
            similarities.append(similarity.item())

    # Clean up
    del labse_model
    del labse_tokenizer

    return {
        "LaBSE-Similarity": sum(similarities) / len(similarities) * 100,
    }


def compute_rouge_score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores between references and hypotheses."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores) * 100,
        "ROUGE-2": sum(rouge2_scores) / len(rouge2_scores) * 100,
        "ROUGE-L": sum(rougeL_scores) / len(rougeL_scores) * 100,
    }


def evaluate_model(
    model_path: str,
    num_samples: int = None,
    output_file: str = None,
    batch_size: int = 1,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
):
    """
    Evaluate model on dev set.

    Args:
        model_path: Path to the model (can be local or HuggingFace hub)
        num_samples: Number of samples to evaluate (None for all)
        output_file: Path to save predictions (None for no saving)
        batch_size: Batch size for evaluation (for future batch generation support)
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling (vs greedy decoding)
    """
    print("=" * 80)
    print("Starting evaluation...")
    print("=" * 80)

    # Download NLTK data
    download_nltk_data()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load dev data
    dev_data = load_dev_data(DATA_DIR)

    # Limit samples if specified
    if num_samples is not None:
        dev_data = dev_data[:num_samples]
        print(f"Evaluating on {num_samples} samples")
    else:
        print(f"Evaluating on all {len(dev_data)} samples")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    references = []

    for i, item in enumerate(tqdm(dev_data, desc="Generating")):
        sql_query = item.get("query", {}).get("ru", item.get("query", {}).get("en", ""))
        actual_question = item.get("question", {}).get("ru", item.get("question", {}).get("en", ""))

        if not sql_query or not actual_question:
            continue

        # Generate prediction
        predicted_question = generate_question(model, tokenizer, sql_query, max_new_tokens, temperature, do_sample)

        predictions.append({
            "id": item.get("id", f"sample_{i}"),
            "sql": sql_query,
            "reference": actual_question,
            "prediction": predicted_question,
        })

        references.append(actual_question)

    # Compute metrics
    print("\nComputing evaluation metrics...")
    hypotheses = [p["prediction"] for p in predictions]

    # BLEU scores
    bleu_scores = compute_bleu_score(references, hypotheses)

    # ROUGE scores
    rouge_scores = compute_rouge_score(references, hypotheses)

    # CHRF scores
    chrf_scores = compute_chrf_score(references, hypotheses)

    # LaBSE similarity
    labse_scores = compute_labse_similarity(references, hypotheses)

    # Get model name (use path basename for local models)
    model_name = os.path.basename(model_path) if os.path.exists(model_path) else model_path
    data_count = len(predictions)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Total samples evaluated: {data_count}")

    print("\nBLEU Scores:")
    for metric, score in bleu_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nCHRF Scores:")
    for metric, score in chrf_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\nLaBSE Similarity:")
    for metric, score in labse_scores.items():
        print(f"  {metric}: {score:.2f}")
    print("=" * 80)

    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 80)
    for i, pred in enumerate(predictions[:10]):
        print(f"\n--- Sample {i+1} ---")
        print(f"SQL: {pred['sql']}")
        print(f"Reference: {pred['reference']}")
        print(f"Prediction: {pred['prediction']}")

    # Save predictions if output file specified
    if output_file:
        print(f"\nSaving predictions to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print("Predictions saved!")

    # Save evaluation results to CSV (append mode)
    csv_file = "evaluation_results.csv"
    csv_exists = os.path.exists(csv_file)

    # Flatten all metrics into a single dict
    all_metrics = {
        "model_name": model_name,
        "data_count": data_count,
        **bleu_scores,
        **rouge_scores,
        **chrf_scores,
        **labse_scores,
    }

    print(f"\nSaving evaluation results to {csv_file}...")
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(all_metrics)
    print(f"Evaluation results saved to {csv_file}!")

    # Return metrics
    return {
        "bleu": bleu_scores,
        "rouge": rouge_scores,
        "chrf": chrf_scores,
        "labse": labse_scores,
        "total_samples": len(predictions),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen model on PAUQ dev set"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Path to model (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save predictions JSON (default: no saving)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)",
    )

    args = parser.parse_args()

    # Run evaluation with command-line arguments
    evaluate_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.no_sample,
    )


if __name__ == "__main__":
    main()
