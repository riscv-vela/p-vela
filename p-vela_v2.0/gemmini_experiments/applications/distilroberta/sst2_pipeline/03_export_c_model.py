#!/usr/bin/env python3
"""
Export DistilRoBERTa quantized artifacts into C-consumable binary packs.

This script is the stage-3 bridge between the Python-side quantized pipeline
and the upcoming guest C implementation.

Why this exporter uses raw binary + manifest instead of one giant C header:
- The quantized DistilRoBERTa model is large.
- Emitting a single static header would create very large source files and
  slow compilation a lot.
- Raw binary files plus a structured manifest are easier for the next C stage
  to load from hostshare while keeping exact tensor shapes and dtypes.

What this script exports:
1. Model parameter pack:
   - all arrays from outputs/int8_host/model_int8.npz except debug helper keys
2. Evaluation sample pack:
   - tokenized SST-2 samples
   - labels
   - host-int8 reference predictions / logits
3. Debug reference pack:
   - saved debug tensors from stage-2
   - debug summary rows copied as JSON
4. A top-level manifest JSON describing every exported binary file

Output root:
- guest/generated/export_manifest.json
- guest/generated/model_bin/
- guest/generated/eval_bin/
- guest/generated/debug_bin/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from model_spec import DEFAULT_MODEL_NAME


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_LENGTH = 128
DEFAULT_INT8_OUTPUT_DIR = PIPELINE_DIR / "outputs" / "int8_host"
DEFAULT_FP32_OUTPUT_DIR = PIPELINE_DIR / "outputs" / "fp32"
DEFAULT_GUEST_GENERATED_DIR = PIPELINE_DIR / "guest" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--int8-output-dir", type=Path, default=DEFAULT_INT8_OUTPUT_DIR)
    parser.add_argument("--fp32-output-dir", type=Path, default=DEFAULT_FP32_OUTPUT_DIR)
    parser.add_argument("--guest-generated-dir", type=Path, default=DEFAULT_GUEST_GENERATED_DIR)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        memo_marker = "\n  [memo]"
        if memo_marker not in text:
            raise
        json_part = text.split(memo_marker, 1)[0].rstrip()
        if json_part.endswith(","):
            json_part = json_part[:-1]
        json_part = json_part + "\n}"
        return json.loads(json_part)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_array_bin(output_dir: Path, name: str, array: np.ndarray) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.bin"
    contiguous = np.ascontiguousarray(array)
    contiguous.tofile(path)
    return {
        "name": name,
        "file": path.name,
        "dtype": str(contiguous.dtype),
        "shape": [int(x) for x in contiguous.shape],
        "numel": int(contiguous.size),
        "nbytes": int(contiguous.nbytes),
    }


def export_model_pack(npz_path: Path, model_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    npz = np.load(npz_path)
    entries: list[dict[str, Any]] = []
    config_summary: dict[str, Any] = {}

    for key in sorted(npz.files):
        if key.startswith("debug__"):
            continue
        arr = np.asarray(npz[key])
        entries.append(write_array_bin(model_dir, key, arr))
        if key.startswith("config__"):
            config_summary[key] = arr.item() if arr.ndim == 0 else arr.tolist()

    return entries, config_summary


def build_eval_pack(
    tokenizer,
    dataset,
    host_rows: list[dict[str, Any]],
    max_length: int,
    eval_limit: int | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    total = len(host_rows) if eval_limit is None else min(len(host_rows), eval_limit)
    rows = host_rows[:total]

    input_ids = np.zeros((total, max_length), dtype=np.int32)
    attention_mask = np.zeros((total, max_length), dtype=np.int32)
    labels = np.zeros((total,), dtype=np.int32)
    host_preds = np.zeros((total,), dtype=np.int32)
    host_logits = np.zeros((total, 2), dtype=np.float32)
    indices = np.zeros((total,), dtype=np.int32)

    manifest_rows: list[dict[str, Any]] = []

    # Tokenize the exact evaluation samples which the C stage will consume.
    for out_idx, row in enumerate(rows):
        sample_idx = int(row["idx"])
        sample = dataset[sample_idx]
        encoded = tokenizer(
            sample["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids[out_idx] = encoded["input_ids"][0].astype(np.int32)
        attention_mask[out_idx] = encoded["attention_mask"][0].astype(np.int32)
        labels[out_idx] = int(sample["label"])
        host_preds[out_idx] = int(row["pred"])
        host_logits[out_idx] = np.asarray(row["logits"], dtype=np.float32)
        indices[out_idx] = sample_idx
        manifest_rows.append(
            {
                "order": out_idx,
                "sample_idx": sample_idx,
                "label": int(sample["label"]),
                "host_int8_pred": int(row["pred"]),
            }
        )

    arrays = {
        "indices": indices,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "host_int8_preds": host_preds,
        "host_int8_logits": host_logits,
    }
    return arrays, manifest_rows


def export_eval_pack(
    tokenizer,
    dataset,
    host_rows: list[dict[str, Any]],
    max_length: int,
    eval_limit: int | None,
    eval_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arrays, manifest_rows = build_eval_pack(tokenizer, dataset, host_rows, max_length, eval_limit)
    entries = [write_array_bin(eval_dir, name, array) for name, array in arrays.items()]
    return entries, manifest_rows


def export_debug_pack(
    debug_tensors_path: Path,
    debug_summary_path: Path,
    debug_samples_path: Path,
    debug_dir: Path,
) -> dict[str, Any]:
    debug_npz = np.load(debug_tensors_path)
    tensor_entries = []
    for key in sorted(debug_npz.files):
        tensor_entries.append(write_array_bin(debug_dir, key, np.asarray(debug_npz[key])))

    debug_summary = load_json(debug_summary_path)
    debug_samples = load_json(debug_samples_path)
    (debug_dir / "debug_summary.json").write_text(json.dumps(debug_summary, indent=2), encoding="utf-8")
    (debug_dir / "debug_samples.json").write_text(json.dumps(debug_samples, indent=2), encoding="utf-8")

    return {
        "tensor_entries": tensor_entries,
        "debug_summary_file": "debug_summary.json",
        "debug_samples_file": "debug_samples.json",
        "num_debug_summary_rows": len(debug_summary),
        "num_debug_samples": len(debug_samples),
    }


def main() -> None:
    args = parse_args()
    local_files_only = False if args.allow_download else args.local_files_only

    model_npz_path = args.int8_output_dir / "model_int8.npz"
    host_metrics_path = args.int8_output_dir / "metrics.json"
    host_predictions_path = args.int8_output_dir / "predictions.jsonl"
    debug_tensors_path = args.int8_output_dir / "debug_tensors.npz"
    debug_summary_path = args.int8_output_dir / "debug_summary.json"
    debug_samples_path = args.int8_output_dir / "debug_samples.json"
    fp32_metrics_path = args.fp32_output_dir / "metrics.json"

    required_paths = [
        model_npz_path,
        host_metrics_path,
        host_predictions_path,
        debug_tensors_path,
        debug_summary_path,
        debug_samples_path,
        fp32_metrics_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required pipeline outputs. Run 00, 01, 02 first:\n" + "\n".join(missing)
        )

    root = args.guest_generated_dir
    model_dir = root / "model_bin"
    eval_dir = root / "eval_bin"
    debug_dir = root / "debug_bin"
    ensure_clean_dir(model_dir)
    ensure_clean_dir(eval_dir)
    ensure_clean_dir(debug_dir)

    print("[export_c] loading tokenizer and dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=local_files_only)
    dataset = load_dataset("glue", "sst2", split=args.split)

    print("[export_c] loading stage outputs")
    host_metrics = load_json(host_metrics_path)
    fp32_metrics = load_json(fp32_metrics_path)
    host_rows = load_jsonl(host_predictions_path)

    print("[export_c] exporting model parameter binaries")
    model_entries, config_summary = export_model_pack(model_npz_path, model_dir)

    print("[export_c] exporting evaluation sample binaries")
    eval_entries, eval_rows = export_eval_pack(
        tokenizer=tokenizer,
        dataset=dataset,
        host_rows=host_rows,
        max_length=args.max_length,
        eval_limit=args.eval_limit,
        eval_dir=eval_dir,
    )

    print("[export_c] exporting debug reference binaries")
    debug_manifest = export_debug_pack(
        debug_tensors_path=debug_tensors_path,
        debug_summary_path=debug_summary_path,
        debug_samples_path=debug_samples_path,
        debug_dir=debug_dir,
    )

    manifest = {
        "model_name": args.model_name,
        "split": args.split,
        "max_length": args.max_length,
        "eval_limit": args.eval_limit,
        "fp32_metrics": fp32_metrics,
        "host_int8_metrics": host_metrics,
        "config_summary": config_summary,
        "model_pack": {
            "dir": str(model_dir.relative_to(root)),
            "num_entries": len(model_entries),
            "entries": model_entries,
        },
        "eval_pack": {
            "dir": str(eval_dir.relative_to(root)),
            "num_entries": len(eval_entries),
            "num_samples": len(eval_rows),
            "sample_rows": eval_rows,
            "entries": eval_entries,
        },
        "debug_pack": {
            "dir": str(debug_dir.relative_to(root)),
            **debug_manifest,
        },
    }

    manifest_path = root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[export_c] model entries      : {len(model_entries)}")
    print(f"[export_c] eval samples       : {len(eval_rows)}")
    print(f"[export_c] debug tensors      : {len(debug_manifest['tensor_entries'])}")
    print(f"[export_c] manifest           : {manifest_path}")
    print(f"[export_c] model dir          : {model_dir}")
    print(f"[export_c] eval dir           : {eval_dir}")
    print(f"[export_c] debug dir          : {debug_dir}")


if __name__ == "__main__":
    main()
