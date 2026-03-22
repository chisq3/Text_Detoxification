#!/usr/bin/env python3
"""
Phase 1 experiments: Toxicity classification benchmark.

Runs and compares 3 Hugging Face backbones on the same prepared splits:
- distilbert-base-uncased
- roberta-base
- microsoft/deberta-v3-base
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2Tokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from core.io import load_tsv_dataset
from core.reporting import export_leaderboard
from core.seed import set_seed
from core.wandb_utils import finish_run, init_run, log_metrics, update_summary


@dataclass
class ExperimentConfig:
    base_dir: str
    output_root: str
    best_model_dir: str
    project_name: str
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


CANDIDATE_MODELS = [
    "distilbert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
]


def load_tokenizer_with_fallback(model_name: str):
    """Try fast tokenizer first, then gracefully fallback to slow tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        print(f"Fast tokenizer failed for {model_name}: {exc}")
        if model_name == "microsoft/deberta-v3-base":
            return DebertaV2Tokenizer.from_pretrained(model_name)
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def load_split(tsv_path: str) -> Dataset:
    return load_tsv_dataset(
        tsv_path=tsv_path,
        required_columns=["text", "label"],
        column_types={"text": str, "label": int},
    )


def build_datasets(config: ExperimentConfig) -> Dict[str, Dataset]:
    processed_dir = os.path.join(config.base_dir, "data", "processed")
    return {
        "train": load_split(os.path.join(processed_dir, "cls_train.tsv")),
        "val": load_split(os.path.join(processed_dir, "cls_val.tsv")),
        "test": load_split(os.path.join(processed_dir, "cls_test.tsv")),
    }


def tokenize_datasets(datasets: Dict[str, Dataset], tokenizer, max_length: int) -> Dict[str, Dataset]:
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized = {
        split: ds.map(_tokenize, batched=True)
        for split, ds in datasets.items()
    }

    for split in tokenized:
        tokenized[split].set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def print_comparison_table(results: List[Dict[str, float]]) -> None:
    if not results:
        print("No results to display")
        return

    print("PHASE 1 COMPARISON (TEST SET)")
    header = f"{'Model':44s} | {'Acc':>8s} | {'F1':>8s} | {'Precision':>10s} | {'Recall':>8s}"
    print(header)
    print("-" * len(header))

    for row in sorted(results, key=lambda x: x["test_f1"], reverse=True):
        print(
            f"{row['model_name']:44s} | "
            f"{row['test_accuracy']:.4f} | "
            f"{row['test_f1']:.4f} | "
            f"{row['test_precision']:.4f} | "
            f"{row['test_recall']:.4f}"
        )

    best = max(results, key=lambda x: x["test_f1"])
    print(f"Champion #1: {best['model_name']} (Test F1 = {best['test_f1']:.4f})")


def render_phase1_report_text(results: List[Dict[str, float]]) -> str:
    ordered = sorted(results, key=lambda x: x["test_f1"], reverse=True)
    lines = []
    lines.append("PHASE 1 COMPARISON (TEST SET)")
    lines.append(f"{'Rank':4s} | {'Model':44s} | {'Acc':>8s} | {'F1':>8s} | {'Precision':>10s} | {'Recall':>8s}")
    lines.append("-" * 96)
    for idx, row in enumerate(ordered, start=1):
        lines.append(
            f"{idx:>4d} | {row['model_name']:44s} | "
            f"{row['test_accuracy']:.4f} | {row['test_f1']:.4f} | "
            f"{row['test_precision']:.4f} | {row['test_recall']:.4f}"
        )
    lines.append(f"Champion #1: {ordered[0]['model_name']} (Test F1 = {ordered[0]['test_f1']:.4f})")
    return "\n".join(lines) + "\n"


def export_phase1_leaderboard(results: List[Dict[str, float]], output_dir: str) -> None:
    ordered = sorted(results, key=lambda x: x["test_f1"], reverse=True)
    export_leaderboard(
        results=ordered,
        output_dir=output_dir,
        csv_name="phase1_all_models_report.csv",
        json_name="phase1_all_models_report.json",
        txt_name="phase1_all_models_report.txt",
        text_writer=render_phase1_report_text,
    )


def run_single_experiment(
    model_name: str,
    datasets: Dict[str, Dataset],
    config: ExperimentConfig,
) -> Tuple[Dict[str, float], Trainer, object]:
    print(f"Running model: {model_name}")

    run_name = f"phase1-{sanitize_model_name(model_name)}"
    run_output_dir = os.path.join(config.output_root, sanitize_model_name(model_name))

    tokenizer = load_tokenizer_with_fallback(model_name)
    tokenized = tokenize_datasets(datasets, tokenizer, config.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.label2id = {"neutral": 0, "toxic": 1}
    model.config.id2label = {0: "neutral", 1: "toxic"}

    init_run(
        project_name=config.project_name,
        run_name=run_name,
        config={
            "model_name": model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "seed": config.seed,
        },
    )

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=config.seed,
        report_to="wandb",
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    result = {
        "model_name": model_name,
        "test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
        "test_f1": float(test_metrics.get("test_f1", 0.0)),
        "test_precision": float(test_metrics.get("test_precision", 0.0)),
        "test_recall": float(test_metrics.get("test_recall", 0.0)),
    }

    log_metrics(
        {
            "final/test_accuracy": result["test_accuracy"],
            "final/test_f1": result["test_f1"],
            "final/test_precision": result["test_precision"],
            "final/test_recall": result["test_recall"],
        }
    )
    update_summary(
        {
            "champion_metric/test_f1": result["test_f1"],
            "champion_metric/test_accuracy": result["test_accuracy"],
        }
    )
    finish_run()

    return result, trainer, tokenizer


def save_champion(trainer: Trainer, tokenizer, best_model_dir: str, best_result: Dict[str, float]) -> None:
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    os.makedirs(best_model_dir, exist_ok=True)

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    metadata_path = os.path.join(best_model_dir, "champion_phase1.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Phase 1 Champion Model\n")
        f.write(f"model_name: {best_result['model_name']}\n")
        f.write(f"test_accuracy: {best_result['test_accuracy']:.6f}\n")
        f.write(f"test_f1: {best_result['test_f1']:.6f}\n")
        f.write(f"test_precision: {best_result['test_precision']:.6f}\n")
        f.write(f"test_recall: {best_result['test_recall']:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 benchmark experiments")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/storage/student6/toxicity-detection-detoxification",
        help="Project root containing data/processed",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="toxicity-phase1-benchmark",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/storage/student6/toxicity-detection-detoxification/results/phase1_experiments",
        help="Directory to store per-model checkpoints/logs",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="/storage/student6/toxicity-detection-detoxification/best_phase1_model",
        help="Directory to export champion model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=CANDIDATE_MODELS,
        help="Model list to run. Defaults to full benchmark trio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = ExperimentConfig(
        base_dir=args.base_dir,
        output_root=args.output_root,
        best_model_dir=args.best_model_dir,
        project_name=args.project_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    os.makedirs(config.output_root, exist_ok=True)

    datasets = build_datasets(config)
    print(
        f"Train={len(datasets['train'])} | Val={len(datasets['val'])} | Test={len(datasets['test'])}"
    )

    all_results: List[Dict[str, float]] = []
    champion_trainer = None
    champion_tokenizer = None
    champion_result = None

    for model_name in args.models:
        result, trainer, tokenizer = run_single_experiment(model_name, datasets, config)
        all_results.append(result)

        if champion_result is None or result["test_f1"] > champion_result["test_f1"]:
            champion_result = result
            champion_trainer = trainer
            champion_tokenizer = tokenizer

    print_comparison_table(all_results)
    export_phase1_leaderboard(all_results, config.output_root)
    export_phase1_leaderboard(all_results, config.best_model_dir)

    if champion_trainer is None or champion_tokenizer is None or champion_result is None:
        raise RuntimeError("No champion found. Experiments may have failed.")

    save_champion(champion_trainer, champion_tokenizer, config.best_model_dir, champion_result)
    print(f"\nChampion exported to: {config.best_model_dir}")


if __name__ == "__main__":
    main()
