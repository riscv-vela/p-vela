#!/usr/bin/env python3
"""
DistilRoBERTa SST-2 FP32 baseline evaluator.

This script is the stage-0 reference for the new DistilRoBERTa pipeline.
Its job is simple:
1. Load the pretrained Hugging Face DistilRoBERTa SST-2 classifier in FP32.
2. Run evaluation on the SST-2 split we choose, usually validation.
3. Save a stable reference accuracy plus per-sample predictions.

Why this file matters:
- Every later stage is compared against this FP32 baseline first.
- We keep sample indices and logits so later int8 / C / Gemmini stages can
  align their outputs against the exact same evaluation set.

Outputs:
- outputs/fp32/metrics.json
- outputs/fp32/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_spec import DEFAULT_MODEL_NAME


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fp32"
DEFAULT_SPLIT = "validation"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128


# Parse only the minimal options needed to define the stage-0 baseline run.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_fp32_model(model_name: str, local_files_only: bool):
    # Stage 0 always uses the original Hugging Face FP32 classifier as-is.
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.eval()
    return tokenizer, model


def load_sst2(split: str):
    return load_dataset("glue", "sst2", split=split)


# Persist the stage-0 reference so later stages compare against fixed files.
def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def run_fp32_eval(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataset,
    batch_size: int,
    max_length: int,
    limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_size = len(dataset) if limit is None else min(len(dataset), limit)
    rows: list[dict[str, Any]] = []
    correct = 0

    # Main stage-0 pass:
    # run the original FP32 model over SST-2 and keep per-sample outputs.
    # Core evaluation loop:
    # tokenize in batches, run model once, then persist per-sample outputs
    # in original dataset order so later stages can compare by idx directly.
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch = dataset[start:end]
        texts = batch["sentence"]
        labels = torch.tensor(batch["label"], dtype=torch.long)

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)

        correct += int((preds == labels).sum().item())

        for offset, sentence in enumerate(texts):
            idx = start + offset
            rows.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "label": int(labels[offset].item()),
                    "pred": int(preds[offset].item()),
                    "logits": [float(x) for x in logits[offset].tolist()],
                }
            )

        if start % (batch_size * 10) == 0:
            print(f"[fp32] processed {end}/{total_size}")

    accuracy = (correct / total_size) if total_size > 0 else 0.0
    metrics = {
        "model_name": model.name_or_path,
        "split": str(dataset.split) if hasattr(dataset, "split") else "unknown",
        "num_samples": total_size,
        "batch_size": batch_size,
        "max_length": max_length,
        "accuracy": accuracy,
    }
    return metrics, rows


def main() -> None:
    args = parse_args()
    local_files_only = False if args.allow_download else args.local_files_only
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load the original pretrained model.
    print("[fp32] loading tokenizer and model")
    tokenizer, model = load_fp32_model(args.model_name, local_files_only=local_files_only)

    # 2) Load the exact evaluation split that later stages will reuse.
    print(f"[fp32] loading SST-2 split={args.split}")
    dataset = load_sst2(args.split)

    # 3) Measure baseline accuracy and collect per-sample predictions/logits.
    metrics, rows = run_fp32_eval(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
    )

    # 4) Save the canonical FP32 reference artifacts for 01/02/03 and beyond.
    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "predictions.jsonl"
    write_metrics(metrics_path, metrics)
    write_jsonl(predictions_path, rows)

    print(f"[fp32] accuracy        : {metrics['accuracy']:.6f}")
    print(f"[fp32] num_samples     : {metrics['num_samples']}")
    print(f"[fp32] metrics path    : {metrics_path}")
    print(f"[fp32] predictions path: {predictions_path}")


if __name__ == "__main__":
    main()
