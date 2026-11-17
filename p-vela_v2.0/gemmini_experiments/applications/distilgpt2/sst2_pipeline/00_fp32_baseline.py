#!/usr/bin/env python3
"""
DistilGPT2 SST-2 FP32 baseline evaluator.

This script is the stage-0 reference for the DistilGPT2 pipeline.
Its job is:
1. Load a GPT2 sequence-classification checkpoint in FP32.
2. Fix the tokenizer/model padding policy so batched evaluation is stable.
3. Run evaluation on SST-2 and save per-sample outputs.

Why this file matters:
- Every later stage is compared against this FP32 baseline first.
- DistilGPT2 classification depends on the last non-pad token position, so we
  save that index per sample as part of the canonical reference.

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
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer

from model_spec import ARCH_NAME, DEFAULT_BASE_MODEL_NAME, DEFAULT_MODEL_NAME, DEFAULT_NUM_LABELS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fp32"
DEFAULT_SPLIT = "validation"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 128


def load_gpt2_tokenizer(model_name: str, local_files_only: bool):
    try:
        return GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except OSError:
        return AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_fast=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SST-2 fine-tuned GPT2 sequence-classification checkpoint or local path.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--allow-base-model",
        action="store_true",
        help="Allow raw distilgpt2/gpt2 checkpoints without an SST-2 fine-tuned head.",
    )
    return parser.parse_args()


def load_sst2(split: str):
    return load_dataset("glue", "sst2", split=split)


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def maybe_reject_base_model(model_name: str, allow_base_model: bool) -> None:
    raw_model_names = {
        "distilgpt2",
        "gpt2",
        "distilbert/distilgpt2",
        DEFAULT_BASE_MODEL_NAME,
        "openai-community/gpt2",
    }
    if model_name in raw_model_names and not allow_base_model:
        raise ValueError(
            "Raw GPT2 LM checkpoints are not valid SST-2 baselines here. "
            "Pass an SST-2 fine-tuned GPT2 sequence-classification checkpoint, "
            "or use --allow-base-model only for smoke testing."
        )


def ensure_padding_policy(tokenizer, model) -> dict[str, Any]:
    uses_eos_as_pad = False

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
        uses_eos_as_pad = True

    if getattr(model.config, "pad_token_id", None) != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id

    return {
        "pad_token_id": int(tokenizer.pad_token_id),
        "pad_token": str(tokenizer.pad_token),
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        "eos_token": str(tokenizer.eos_token) if tokenizer.eos_token is not None else None,
        "uses_eos_as_pad": uses_eos_as_pad,
    }


def validate_model_structure(model) -> None:
    model_type = getattr(model.config, "model_type", None)
    if model_type != "gpt2":
        raise ValueError(f"Expected a GPT2-family classifier, got model_type={model_type!r}")
    if not hasattr(model, "score"):
        raise ValueError("Expected GPT2ForSequenceClassification-style model with a score head.")


def try_load_peft_sequence_classifier(model_name: str, local_files_only: bool):
    try:
        from peft import PeftConfig, PeftModel
    except ImportError:
        return None

    try:
        peft_config = PeftConfig.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception:
        return None

    base_model_name = peft_config.base_model_name_or_path or DEFAULT_BASE_MODEL_NAME
    base_config = AutoConfig.from_pretrained(base_model_name, local_files_only=local_files_only)
    num_labels = max(
        int(getattr(base_config, "num_labels", DEFAULT_NUM_LABELS) or DEFAULT_NUM_LABELS),
        DEFAULT_NUM_LABELS,
    )
    base_config.num_labels = num_labels
    base_config.id2label = {idx: f"LABEL_{idx}" for idx in range(num_labels)}
    base_config.label2id = {label: idx for idx, label in base_config.id2label.items()}

    tokenizer = load_gpt2_tokenizer(base_model_name, local_files_only=local_files_only)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        config=base_config,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_name,
        local_files_only=local_files_only,
    )

    load_info = {
        "load_mode": "peft_adapter",
        "adapter_model_name": model_name,
        "base_model_name": base_model_name,
    }
    return tokenizer, model, load_info


def load_fp32_model(model_name: str, local_files_only: bool, allow_base_model: bool):
    maybe_reject_base_model(model_name, allow_base_model)

    peft_loaded = try_load_peft_sequence_classifier(model_name, local_files_only=local_files_only)
    if peft_loaded is not None:
        tokenizer, model, load_info = peft_loaded
        validate_model_structure(model)
        pad_info = ensure_padding_policy(tokenizer, model)
        model.eval()
        return tokenizer, model, pad_info, load_info

    try:
        tokenizer = load_gpt2_tokenizer(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        mode_hint = (
            "The requested checkpoint is missing from local cache. "
            "Use --allow-download or pass a local path/model id that is already available."
            if local_files_only
            else "Failed to load the requested checkpoint from Hugging Face or local path."
        )
        raise OSError(f"{mode_hint} model_name={model_name!r}") from exc

    validate_model_structure(model)
    pad_info = ensure_padding_policy(tokenizer, model)
    model.eval()
    load_info = {
        "load_mode": "plain_sequence_classification",
        "adapter_model_name": None,
        "base_model_name": model_name,
    }
    return tokenizer, model, pad_info, load_info


def compute_last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.to(torch.int64).sum(dim=-1)
    return torch.clamp(lengths - 1, min=0)


def tokenize_gpt2_batch(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id must be set before batch tokenization.")

    rows: list[list[int]] = []
    for text in texts:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        if not token_ids:
            if tokenizer.eos_token_id is None:
                raise ValueError("Cannot repair empty GPT2 tokenization without eos_token_id.")
            token_ids = [int(tokenizer.eos_token_id)]
        rows.append(token_ids)

    padded_length = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), padded_length),
        fill_value=int(tokenizer.pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(rows), padded_length), dtype=torch.long)

    for row_idx, token_ids in enumerate(rows):
        seq_len = len(token_ids)
        input_ids[row_idx, :seq_len] = torch.tensor(token_ids, dtype=torch.long)
        attention_mask[row_idx, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def run_fp32_eval(
    model,
    tokenizer,
    dataset,
    batch_size: int,
    max_length: int,
    limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_size = len(dataset) if limit is None else min(len(dataset), limit)
    rows: list[dict[str, Any]] = []
    correct = 0

    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch = dataset[start:end]
        texts = batch["sentence"]
        labels = torch.tensor(batch["label"], dtype=torch.long)

        inputs = tokenize_gpt2_batch(
            tokenizer=tokenizer,
            texts=texts,
            max_length=max_length,
        )
        last_token_idx = compute_last_token_indices(inputs["attention_mask"])

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)

        correct += int((preds == labels).sum().item())

        for offset, sentence in enumerate(texts):
            idx = start + offset
            input_length = int(inputs["attention_mask"][offset].sum().item())
            rows.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "label": int(labels[offset].item()),
                    "pred": int(preds[offset].item()),
                    "logits": [float(x) for x in logits[offset].tolist()],
                    "input_length": input_length,
                    "last_token_idx": int(last_token_idx[offset].item()),
                }
            )

        if start % (batch_size * 10) == 0:
            print(f"[fp32] processed {end}/{total_size}")

    accuracy = (correct / total_size) if total_size > 0 else 0.0
    metrics = {
        "arch_name": ARCH_NAME,
        "model_name": model.name_or_path,
        "model_type": str(getattr(model.config, "model_type", "unknown")),
        "split": str(dataset.split) if hasattr(dataset, "split") else "unknown",
        "num_samples": total_size,
        "batch_size": batch_size,
        "max_length": max_length,
        "num_labels": int(getattr(model.config, "num_labels", 0)),
        "accuracy": accuracy,
    }
    return metrics, rows


def main() -> None:
    args = parse_args()
    local_files_only = False if args.allow_download else args.local_files_only
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[fp32] loading tokenizer and model")
    tokenizer, model, pad_info, load_info = load_fp32_model(
        args.model_name,
        local_files_only=local_files_only,
        allow_base_model=args.allow_base_model,
    )

    print(f"[fp32] loading SST-2 split={args.split}")
    dataset = load_sst2(args.split)

    metrics, rows = run_fp32_eval(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
    )
    metrics.update(pad_info)
    metrics.update(load_info)

    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "predictions.jsonl"
    write_metrics(metrics_path, metrics)
    write_jsonl(predictions_path, rows)

    print(f"[fp32] accuracy        : {metrics['accuracy']:.6f}")
    print(f"[fp32] num_samples     : {metrics['num_samples']}")
    print(f"[fp32] pad_token_id    : {metrics['pad_token_id']}")
    print(f"[fp32] uses_eos_as_pad : {metrics['uses_eos_as_pad']}")
    print(f"[fp32] metrics path    : {metrics_path}")
    print(f"[fp32] predictions path: {predictions_path}")


if __name__ == "__main__":
    main()
