#!/usr/bin/env python3
"""
Phase 2 experiments: Detoxification Seq2Seq benchmark.

Runs and compares 3 Hugging Face seq2seq backbones on prepared splits:
- t5-base
- facebook/bart-base
- t5-small
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
    max_source_len: int = 128
    max_target_len: int = 128
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_beams: int = 6
    seed: int = 42


CANDIDATE_MODELS = [
    "t5-base",
    "facebook/bart-base",
    "t5-small",
]


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def load_tokenizer_with_fallback(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        print(f"Fast tokenizer failed for {model_name}: {exc}")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def load_split(tsv_path: str) -> Dataset:
    return load_tsv_dataset(
        tsv_path=tsv_path,
        required_columns=["source", "target"],
        column_types={"source": str, "target": str},
    )


def build_datasets(config: ExperimentConfig) -> Dict[str, Dataset]:
    processed_dir = os.path.join(config.base_dir, "data", "processed")
    return {
        "train": load_split(os.path.join(processed_dir, "s2s_train.tsv")),
        "val": load_split(os.path.join(processed_dir, "s2s_val.tsv")),
        "test": load_split(os.path.join(processed_dir, "s2s_test.tsv")),
    }


def make_preprocess_fn(tokenizer, cfg: ExperimentConfig):
    def _preprocess(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=cfg.max_source_len,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=cfg.max_target_len,
            truncation=True,
            padding="max_length",
        )
        label_ids = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in row]
            for row in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return _preprocess


def compute_metrics_builder(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    chrf_metric = evaluate.load("chrf")

    def _compute(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[r] for r in decoded_labels],
        )
        rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        chrf = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)

        pred_lens = [len(p.split()) for p in decoded_preds]
        ref_lens = [len(r.split()) for r in decoded_labels]

        exact_match = float(np.mean([int(p == r) for p, r in zip(decoded_preds, decoded_labels)]))

        return {
            "bleu": float(bleu["score"]),
            "chrf": float(chrf["score"]),
            "rouge1": float(rouge["rouge1"]),
            "rouge2": float(rouge["rouge2"]),
            "rougeL": float(rouge["rougeL"]),
            "rougeLsum": float(rouge["rougeLsum"]),
            "exact_match": exact_match,
            "avg_pred_len": float(np.mean(pred_lens) if pred_lens else 0.0),
            "avg_ref_len": float(np.mean(ref_lens) if ref_lens else 0.0),
        }

    return _compute


def print_comparison_table(results: List[Dict[str, float]]) -> None:
    if not results:
        print("No results to display")
        return

    print("PHASE 2 COMPARISON (TEST SET)")
    header = (
        f"{'Model':34s} | {'BLEU':>7s} | {'chrF':>7s} | {'R1':>7s} | {'R2':>7s} | {'RL':>7s} | "
        f"{'EM':>7s} | {'Loss':>8s}"
    )
    print(header)
    print("-" * len(header))

    for row in sorted(results, key=lambda x: x["test_bleu"], reverse=True):
        print(
            f"{row['model_name']:34s} | {row['test_bleu']:.4f} | {row['test_chrf']:.4f} | "
            f"{row['test_rouge1']:.4f} | {row['test_rouge2']:.4f} | {row['test_rougeL']:.4f} | "
            f"{row['test_exact_match']:.4f} | {row['test_loss']:.4f}"
        )

    best = max(results, key=lambda x: x["test_bleu"])
    print(f"Champion #2: {best['model_name']} (Test BLEU = {best['test_bleu']:.4f})")


def render_phase2_report_text(results: List[Dict[str, float]]) -> str:
    ordered = sorted(results, key=lambda x: x["test_bleu"], reverse=True)
    lines = []
    lines.append("PHASE 2 COMPARISON (TEST SET)")
    lines.append(
        f"{'Rank':4s} | {'Model':34s} | {'BLEU':>7s} | {'chrF':>7s} | {'R1':>7s} | {'R2':>7s} | "
        f"{'RL':>7s} | {'EM':>7s} | {'Loss':>8s}"
    )
    lines.append("-" * 130)
    for idx, row in enumerate(ordered, start=1):
        lines.append(
            f"{idx:>4d} | {row['model_name']:34s} | {row['test_bleu']:.4f} | {row['test_chrf']:.4f} | "
            f"{row['test_rouge1']:.4f} | {row['test_rouge2']:.4f} | {row['test_rougeL']:.4f} | "
            f"{row['test_exact_match']:.4f} | {row['test_loss']:.4f}"
        )
    lines.append(f"Champion #2: {ordered[0]['model_name']} (Test BLEU = {ordered[0]['test_bleu']:.4f})")
    return "\n".join(lines) + "\n"


def export_phase2_leaderboard(results: List[Dict[str, float]], output_dir: str) -> None:
    ordered = sorted(results, key=lambda x: x["test_bleu"], reverse=True)
    export_leaderboard(
        results=ordered,
        output_dir=output_dir,
        csv_name="phase2_all_models_report.csv",
        json_name="phase2_all_models_report.json",
        txt_name="phase2_all_models_report.txt",
        text_writer=render_phase2_report_text,
    )


def run_single_experiment(
    model_name: str,
    datasets: Dict[str, Dataset],
    cfg: ExperimentConfig,
) -> Tuple[Dict[str, float], Seq2SeqTrainer, object]:
    print(f"Running model: {model_name}")

    run_name = f"phase2-{sanitize_model_name(model_name)}"
    run_output_dir = os.path.join(cfg.output_root, sanitize_model_name(model_name))

    tokenizer = load_tokenizer_with_fallback(model_name)
    preprocess_fn = make_preprocess_fn(tokenizer, cfg)

    tokenized = {
        split: ds.map(preprocess_fn, batched=True, remove_columns=["source", "target"])
        for split, ds in datasets.items()
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    init_run(
        project_name=cfg.project_name,
        run_name=run_name,
        config={
            "model_name": model_name,
            "max_source_len": cfg.max_source_len,
            "max_target_len": cfg.max_target_len,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
            "num_beams": cfg.num_beams,
            "seed": cfg.seed,
        },
    )

    args = Seq2SeqTrainingArguments(
        output_dir=run_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        predict_with_generate=True,
        generation_max_length=cfg.max_target_len,
        generation_num_beams=cfg.num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        seed=cfg.seed,
        report_to="wandb",
        run_name=run_name,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    result = {
        "model_name": model_name,
        "test_bleu": float(test_metrics.get("test_bleu", 0.0)),
        "test_chrf": float(test_metrics.get("test_chrf", 0.0)),
        "test_rouge1": float(test_metrics.get("test_rouge1", 0.0)),
        "test_rouge2": float(test_metrics.get("test_rouge2", 0.0)),
        "test_rougeL": float(test_metrics.get("test_rougeL", 0.0)),
        "test_rougeLsum": float(test_metrics.get("test_rougeLsum", 0.0)),
        "test_exact_match": float(test_metrics.get("test_exact_match", 0.0)),
        "test_avg_pred_len": float(test_metrics.get("test_avg_pred_len", 0.0)),
        "test_avg_ref_len": float(test_metrics.get("test_avg_ref_len", 0.0)),
        "test_loss": float(test_metrics.get("test_loss", 0.0)),
        "test_runtime": float(test_metrics.get("test_runtime", 0.0)),
        "test_samples_per_second": float(test_metrics.get("test_samples_per_second", 0.0)),
        "test_steps_per_second": float(test_metrics.get("test_steps_per_second", 0.0)),
    }

    log_metrics({f"final/{k}": v for k, v in result.items() if k != "model_name"})
    update_summary(
        {
            "champion_metric/test_bleu": result["test_bleu"],
            "champion_metric/test_rougeL": result["test_rougeL"],
        }
    )
    finish_run()

    return result, trainer, tokenizer


def save_champion(trainer, tokenizer, best_model_dir: str, best_result: Dict[str, float]) -> None:
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    os.makedirs(best_model_dir, exist_ok=True)

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    metadata_path = os.path.join(best_model_dir, "champion_phase2.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Phase 2 Champion Model\n")
        f.write(f"model_name: {best_result['model_name']}\n")
        f.write(f"test_bleu: {best_result['test_bleu']:.6f}\n")
        f.write(f"test_chrf: {best_result['test_chrf']:.6f}\n")
        f.write(f"test_rouge1: {best_result['test_rouge1']:.6f}\n")
        f.write(f"test_rouge2: {best_result['test_rouge2']:.6f}\n")
        f.write(f"test_rougeL: {best_result['test_rougeL']:.6f}\n")
        f.write(f"test_rougeLsum: {best_result['test_rougeLsum']:.6f}\n")
        f.write(f"test_exact_match: {best_result['test_exact_match']:.6f}\n")
        f.write(f"test_loss: {best_result['test_loss']:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 benchmark experiments")
    parser.add_argument("--base_dir", type=str, default="/storage/student6/toxicity-detection-detoxification")
    parser.add_argument("--project_name", type=str, default="toxicity-phase2-benchmark")
    parser.add_argument("--output_root", type=str, default="/storage/student6/toxicity-detection-detoxification/results/phase2_experiments")
    parser.add_argument("--best_model_dir", type=str, default="/storage/student6/toxicity-detection-detoxification/best_phase2_model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_source_len", type=int, default=128)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_beams", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", type=str, default=CANDIDATE_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = ExperimentConfig(
        base_dir=args.base_dir,
        output_root=args.output_root,
        best_model_dir=args.best_model_dir,
        project_name=args.project_name,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_beams=args.num_beams,
        seed=args.seed,
    )

    os.makedirs(cfg.output_root, exist_ok=True)

    print("Loading datasets...")
    datasets = build_datasets(cfg)
    print(f"Train={len(datasets['train'])} | Val={len(datasets['val'])} | Test={len(datasets['test'])}")

    all_results: List[Dict[str, float]] = []
    champion_result = None
    champion_trainer = None
    champion_tokenizer = None

    for model_name in args.models:
        result, trainer, tokenizer = run_single_experiment(model_name, datasets, cfg)
        all_results.append(result)

        if champion_result is None or result["test_bleu"] > champion_result["test_bleu"]:
            champion_result = result
            champion_trainer = trainer
            champion_tokenizer = tokenizer

    print_comparison_table(all_results)
    export_phase2_leaderboard(all_results, cfg.output_root)

    if champion_result is None or champion_trainer is None or champion_tokenizer is None:
        raise RuntimeError("No champion found. Experiments may have failed.")

    save_champion(champion_trainer, champion_tokenizer, cfg.best_model_dir, champion_result)
    export_phase2_leaderboard(all_results, cfg.best_model_dir)
    print(f"\nChampion exported to: {cfg.best_model_dir}")


if __name__ == "__main__":
    main()
