#!/usr/bin/env python3
"""
DistilRoBERTa linear-only int8 quantization artifact builder.

This script is the stage-1 preparation step for the new DistilRoBERTa pipeline.
Its purpose is not to evaluate accuracy yet. Instead, it prepares the
quantized artifacts which later stages will consume.

What this script does:
1. Load the pretrained FP32 DistilRoBERTa SST-2 classifier.
2. Collect activation statistics from all nn.Linear modules using a
   calibration subset of SST-2.
3. Quantize every Linear weight tensor to int8 with per-channel scale.
4. Keep non-linear parts in FP32 form so later stages can run
   "Linear int8 + non-linear FP32" inference.
5. Save quantized model tensors, quantization metadata, and a fixed set of
   debug samples chosen from the stage-0 FP32 predictions.

What "linear-only" means here in the current experiment:
- Quantized: every nn.Linear in DistilRoBERTa
- Kept FP32: embeddings, layernorm, softmax, GELU, residual path

Outputs:
- outputs/int8_host/model_int8.npz
- outputs/int8_host/quant_meta.json
- outputs/int8_host/debug_samples.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_spec import (
    ARCH_NAME,
    DEFAULT_MODEL_NAME,
    build_model_config_payload,
    extract_fp32_non_linear_tensors,
    normalize_linear_name,
    sanitize_key,
    should_quantize_linear,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "int8_host"
DEFAULT_FP32_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fp32"
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_LENGTH = 128
DEFAULT_CALIBRATION_SAMPLES = 128
DEFAULT_CALIBRATION_BATCH_SIZE = 16
DEFAULT_DEBUG_SAMPLES = 16


# Per-Linear artifacts exported from stage 1 and later consumed by stage 2/C.
@dataclass
class LinearTensorPack:
    weight_q: np.ndarray
    weight_scale: np.ndarray
    input_scale: float
    output_scale: float
    bias_fp32: np.ndarray
    in_features: int
    out_features: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES)
    parser.add_argument("--calibration-batch-size", type=int, default=DEFAULT_CALIBRATION_BATCH_SIZE)
    parser.add_argument("--debug-samples", type=int, default=DEFAULT_DEBUG_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fp32-output-dir", type=Path, default=DEFAULT_FP32_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_model(model_name: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.eval()
    return tokenizer, model


def load_sst2(split: str):
    return load_dataset("glue", "sst2", split=split)

def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    weight_np = weight.detach().cpu().numpy().astype(np.float32)
    # Per-output-channel weight quantization:
    # for each row o, weight_scale[o] = max(abs(weight[o, :])) / 127.
    absmax = np.max(np.abs(weight_np), axis=1)
    scale = np.maximum(absmax / 127.0, 1.0e-8).astype(np.float32)
    # Quantize each weight row with its own scale, then clip into int8 range.
    weight_q = np.rint(weight_np / scale[:, None])
    weight_q = np.clip(weight_q, -128, 127).astype(np.int8)
    return weight_q, scale


def load_fp32_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def select_debug_samples(rows: list[dict[str, Any]], num_samples: int) -> list[dict[str, Any]]:
    if not rows:
        return []

    def margin(row: dict[str, Any]) -> float:
        logits = row["logits"]
        sorted_logits = sorted(logits, reverse=True)
        return float(sorted_logits[0] - sorted_logits[1])

    chosen: list[dict[str, Any]] = []
    chosen_idx: set[int] = set()

    # First keep low-margin samples because they are more sensitive to
    # quantization drift and are good debugging targets later.
    hard_rows = sorted(rows, key=margin)
    hard_quota = min(num_samples // 2, len(hard_rows))
    for row in hard_rows:
        if len(chosen) >= hard_quota:
            break
        idx = int(row["idx"])
        if idx not in chosen_idx:
            chosen.append(row)
            chosen_idx.add(idx)

    # Fill the rest with evenly spaced samples so we do not debug only
    # borderline examples.
    target_total = min(num_samples, len(rows))
    if target_total > len(chosen):
        positions = np.linspace(0, len(rows) - 1, num=target_total, dtype=int).tolist()
        for pos in positions:
            row = rows[pos]
            idx = int(row["idx"])
            if idx not in chosen_idx:
                chosen.append(row)
                chosen_idx.add(idx)
            if len(chosen) >= target_total:
                break

    chosen.sort(key=lambda row: int(row["idx"]))
    return [
        {
            "idx": int(row["idx"]),
            "label": int(row["label"]),
            "pred": int(row["pred"]),
            "sentence": row["sentence"],
            "logits": [float(x) for x in row["logits"]],
            "margin": margin(row),
        }
        for row in chosen
    ]


def register_linear_hooks(model: nn.Module):
    stats: dict[str, dict[str, float]] = {}
    handles = []

    def update_absmax(name: str, tensor: torch.Tensor, key: str) -> None:
        absmax = float(tensor.detach().abs().max().item())
        prev = stats.setdefault(name, {"input_absmax": 0.0, "output_absmax": 0.0})
        prev[key] = max(prev[key], absmax)

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        norm_name = normalize_linear_name(module_name)
        if not should_quantize_linear(norm_name):
            continue

        # Core calibration hook:
        # record both Linear input and output max magnitude so later stages
        # can quantize/dequantize each op using a consistent per-tensor scale.
        def make_hook(name: str):
            def hook(_module, module_inputs, module_output):
                # Calibration step:
                # record max-abs of the Linear input activation and output activation.
                update_absmax(name, module_inputs[0], "input_absmax")
                update_absmax(name, module_output, "output_absmax")

            return hook

        handles.append(module.register_forward_hook(make_hook(norm_name)))

    return stats, handles


def collect_activation_stats(
    model: nn.Module,
    tokenizer,
    dataset,
    max_length: int,
    calibration_samples: int,
    calibration_batch_size: int,
) -> tuple[dict[str, dict[str, float]], int]:
    stats, handles = register_linear_hooks(model)
    total = min(len(dataset), calibration_samples)

    # Stage-1 calibration pass:
    # run a small subset only to estimate activation ranges for each Linear.
    # Calibration uses a small SST-2 subset. We only need representative
    # activation ranges here, not evaluation accuracy yet.
    for start in range(0, total, calibration_batch_size):
        end = min(start + calibration_batch_size, total)
        batch = dataset[start:end]
        inputs = tokenizer(
            batch["sentence"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            _ = model(**inputs)

        if start % (calibration_batch_size * 4) == 0:
            print(f"[quant] calibration processed {end}/{total}")

    for handle in handles:
        handle.remove()

    for name, values in stats.items():
        # Activation quantization scales are derived from calibration absmax values.
        # input_scale = input_absmax / 127, output_scale = output_absmax / 127.
        values["input_scale"] = max(values["input_absmax"] / 127.0, 1.0e-8)
        values["output_scale"] = max(values["output_absmax"] / 127.0, 1.0e-8)

    return stats, total


def build_quantized_linear_packs(
    model: nn.Module,
    activation_stats: dict[str, dict[str, float]],
) -> dict[str, LinearTensorPack]:
    packs: dict[str, LinearTensorPack] = {}

    # Build one quantized pack per nn.Linear:
    # quantized weight, per-channel weight scale, activation scales, FP32 bias.
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        norm_name = normalize_linear_name(module_name)
        if not should_quantize_linear(norm_name):
            continue
        stats = activation_stats[norm_name]
        weight_q, weight_scale = quantize_weight_per_channel(module.weight)
        # Bias stays in FP32 in this experiment. It is added after the int32 matmul
        # is rescaled back into floating point in later stages.
        bias_fp32 = (
            module.bias.detach().cpu().numpy().astype(np.float32)
            if module.bias is not None
            else np.zeros((module.out_features,), dtype=np.float32)
        )

        packs[norm_name] = LinearTensorPack(
            weight_q=weight_q,
            weight_scale=weight_scale,
            input_scale=float(stats["input_scale"]),
            output_scale=float(stats["output_scale"]),
            bias_fp32=bias_fp32,
            in_features=int(module.in_features),
            out_features=int(module.out_features),
        )

    return packs


def build_npz_payload(
    model,
    linear_packs: dict[str, LinearTensorPack],
    activation_stats: dict[str, dict[str, float]],
    debug_samples: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}

    # Export model config so later Python/C stages can reconstruct dimensions.
    config = model.config
    payload.update(build_model_config_payload(config))

    for name, pack in sorted(linear_packs.items()):
        key = sanitize_key(name)
        # Export all tensors needed to reproduce the quantized Linear in later stages.
        payload[f"{key}__weight_q"] = pack.weight_q.astype(np.int8)
        payload[f"{key}__bias_fp32"] = pack.bias_fp32.astype(np.float32)
        payload[f"{key}__weight_scale"] = pack.weight_scale.astype(np.float32)
        payload[f"{key}__input_scale"] = np.asarray(pack.input_scale, dtype=np.float32)
        payload[f"{key}__output_scale"] = np.asarray(pack.output_scale, dtype=np.float32)
        payload[f"{key}__in_features"] = np.asarray(pack.in_features, dtype=np.int32)
        payload[f"{key}__out_features"] = np.asarray(pack.out_features, dtype=np.int32)
        payload[f"{key}__input_absmax"] = np.asarray(activation_stats[name]["input_absmax"], dtype=np.float32)
        payload[f"{key}__output_absmax"] = np.asarray(activation_stats[name]["output_absmax"], dtype=np.float32)

    # Also export FP32 tensors for the non-linear / surrounding parts of the model.
    payload.update(extract_fp32_non_linear_tensors(model))

    if debug_samples:
        # Fixed debug samples let stage 2 / guest C / Gemmini compare the same cases.
        payload["debug__indices"] = np.asarray([row["idx"] for row in debug_samples], dtype=np.int32)
        payload["debug__labels"] = np.asarray([row["label"] for row in debug_samples], dtype=np.int32)
        payload["debug__fp32_preds"] = np.asarray([row["pred"] for row in debug_samples], dtype=np.int32)
        payload["debug__fp32_logits"] = np.asarray([row["logits"] for row in debug_samples], dtype=np.float32)

    return payload


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_quant_meta(
    model_name: str,
    split: str,
    max_length: int,
    calibration_used: int,
    linear_packs: dict[str, LinearTensorPack],
) -> dict[str, Any]:
    linear_names = sorted(linear_packs.keys())
    return {
        "model_name": model_name,
        "split": split,
        "max_length": max_length,
        "quantization_scheme": "all_linear_int8_weight_per_channel",
        "calibration_samples_used": calibration_used,
        "num_linear_layers": len(linear_names),
        "linear_layers": [
            {
                "name": name,
                "key": sanitize_key(name),
                "in_features": linear_packs[name].in_features,
                "out_features": linear_packs[name].out_features,
                "weight_scale_min": float(np.min(linear_packs[name].weight_scale)),
                "weight_scale_max": float(np.max(linear_packs[name].weight_scale)),
                "input_scale": linear_packs[name].input_scale,
                "output_scale": linear_packs[name].output_scale,
            }
            for name in linear_names
        ],
    }


def main() -> None:
    args = parse_args()
    local_files_only = False if args.allow_download else args.local_files_only
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 depends on stage 0 outputs to select stable debug samples.
    predictions_path = args.fp32_output_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing FP32 baseline predictions: {predictions_path}. Run 00_fp32_baseline.py first."
        )

    # 1) Load the original FP32 model and calibration dataset.
    print(f"[quant] loading tokenizer and {ARCH_NAME} model")
    tokenizer, model = load_model(args.model_name, local_files_only=local_files_only)

    print(f"[quant] loading SST-2 split={args.split}")
    dataset = load_sst2(args.split)

    # 2) Collect per-Linear activation ranges from a calibration subset.
    print("[quant] collecting activation statistics")
    activation_stats, calibration_used = collect_activation_stats(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_length=args.max_length,
        calibration_samples=args.calibration_samples,
        calibration_batch_size=args.calibration_batch_size,
    )

    # 3) Quantize every nn.Linear and gather the exported parameter packs.
    print("[quant] quantizing all Linear weights")
    linear_packs = build_quantized_linear_packs(model, activation_stats)

    # 4) Reuse stage-0 predictions to choose fixed debug samples.
    fp32_rows = load_fp32_prediction_rows(predictions_path)
    debug_samples = select_debug_samples(fp32_rows, args.debug_samples)

    # 5) Save one NPZ payload plus metadata for later host/C/Gemmini stages.
    payload = build_npz_payload(model, linear_packs, activation_stats, debug_samples)
    model_npz_path = args.output_dir / "model_int8.npz"
    np.savez(model_npz_path, **payload)

    quant_meta = build_quant_meta(
        model_name=args.model_name,
        split=args.split,
        max_length=args.max_length,
        calibration_used=calibration_used,
        linear_packs=linear_packs,
    )
    quant_meta_path = args.output_dir / "quant_meta.json"
    debug_samples_path = args.output_dir / "debug_samples.json"
    write_json(quant_meta_path, quant_meta)
    write_json(debug_samples_path, debug_samples)

    print(f"[quant] linear layers       : {len(linear_packs)}")
    print(f"[quant] calibration used   : {calibration_used}")
    print(f"[quant] debug samples      : {len(debug_samples)}")
    print(f"[quant] model npz          : {model_npz_path}")
    print(f"[quant] quant meta         : {quant_meta_path}")
    print(f"[quant] debug sample list  : {debug_samples_path}")


if __name__ == "__main__":
    main()
