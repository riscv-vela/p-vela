#!/usr/bin/env python3
"""
DistilGPT2 projection-only int8 quantization artifact builder.

This script is the stage-1 preparation step for the DistilGPT2 pipeline.
Its purpose is not to evaluate accuracy yet. Instead, it prepares the
quantized artifacts which later stages will consume.

What this script does:
1. Load the FP32 DistilGPT2 SST-2 classifier.
2. Collect activation statistics from GPT2 projection modules using a
   calibration subset of SST-2.
3. Quantize GPT2 projection weights to int8 with per-channel scale.
4. Keep non-linear parts and the classifier head in FP32 form.
5. Save quantized model tensors, quantization metadata, and fixed debug samples
   chosen from the stage-0 FP32 predictions.

What "projection-only" means here in the current experiment:
- Quantized: GPT2 `c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`
- Kept FP32: embeddings, layernorm, softmax, GELU, residual path, classifier head

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
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D

from model_spec import (
    ARCH_NAME,
    DEFAULT_BASE_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
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
DEFAULT_ACTIVATION_PERCENTILE = 100.0
DEFAULT_CALIBRATION_VALUES_PER_HOOK = 32768


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
    parser.add_argument("--activation-percentile", type=float, default=DEFAULT_ACTIVATION_PERCENTILE)
    parser.add_argument("--calibration-values-per-hook", type=int, default=DEFAULT_CALIBRATION_VALUES_PER_HOOK)
    parser.add_argument("--debug-samples", type=int, default=DEFAULT_DEBUG_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fp32-output-dir", type=Path, default=DEFAULT_FP32_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def sample_abs_values(tensor: torch.Tensor, max_values: int) -> np.ndarray:
    flat = tensor.detach().abs().reshape(-1).cpu().numpy().astype(np.float32)
    if flat.size <= max_values:
        return flat
    indices = np.linspace(0, flat.size - 1, num=max_values, dtype=np.int64)
    return flat[indices]


def load_gpt2_tokenizer(model_name: str, local_files_only: bool):
    try:
        return GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except OSError:
        return AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_fast=False,
        )


def load_sst2(split: str):
    return load_dataset("glue", "sst2", split=split)


def ensure_padding_policy(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id


def load_model(model_name: str, local_files_only: bool):
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise ImportError("peft is required for the DistilGPT2 stage-1 pipeline.") from exc

    tokenizer = None
    model = None
    try:
        peft_config = PeftConfig.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception:
        peft_config = None

    if peft_config is not None:
        base_model_name = peft_config.base_model_name_or_path or DEFAULT_BASE_MODEL_NAME
        tokenizer = load_gpt2_tokenizer(base_model_name, local_files_only=local_files_only)
        config = AutoConfig.from_pretrained(base_model_name, local_files_only=local_files_only)
        config.num_labels = max(int(getattr(config, "num_labels", DEFAULT_NUM_LABELS) or DEFAULT_NUM_LABELS), DEFAULT_NUM_LABELS)
        config.id2label = {idx: f"LABEL_{idx}" for idx in range(config.num_labels)}
        config.label2id = {label: idx for idx, label in config.id2label.items()}
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            local_files_only=local_files_only,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            local_files_only=local_files_only,
        )
    else:
        tokenizer = load_gpt2_tokenizer(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

    ensure_padding_policy(tokenizer, model)
    model.eval()
    return tokenizer, model


def tokenize_gpt2_batch(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
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


def is_quantizable_projection_module(module_name: str, module: nn.Module) -> bool:
    if isinstance(module, Conv1D):
        return should_quantize_linear(normalize_linear_name(module_name))
    if isinstance(module, nn.Linear) and hasattr(module, "base_layer") and isinstance(module.base_layer, Conv1D):
        return should_quantize_linear(normalize_linear_name(module_name))
    return False


def get_canonical_weight_and_bias(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    if isinstance(module, Conv1D):
        weight = module.weight.detach().transpose(0, 1).contiguous()
        bias = module.bias.detach() if module.bias is not None else None
        in_features = int(module.weight.shape[0])
        out_features = int(module.weight.shape[1])
        return weight, bias, in_features, out_features

    if isinstance(module, nn.Linear) and hasattr(module, "base_layer") and isinstance(module.base_layer, Conv1D):
        # Stage-2 and the guest C path keep the PEFT wrapper alive and add the
        # LoRA delta in FP32 on top of the quantized base projection.
        # So the int8 pack must contain the raw base Conv1D weight here, not a
        # pre-merged base+delta matrix.
        weight = module.base_layer.weight.detach().transpose(0, 1).contiguous()
        bias = module.base_layer.bias.detach() if module.base_layer.bias is not None else None
        in_features = int(module.base_layer.weight.shape[0])
        out_features = int(module.base_layer.weight.shape[1])
        return weight, bias, in_features, out_features

    raise TypeError(f"Unsupported projection module type: {type(module)!r}")


def quantize_weight_per_channel(weight_out_in: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    weight_np = weight_out_in.detach().cpu().numpy().astype(np.float32)
    absmax = np.max(np.abs(weight_np), axis=1)
    scale = np.maximum(absmax / 127.0, 1.0e-8).astype(np.float32)
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

    hard_rows = sorted(rows, key=margin)
    hard_quota = min(num_samples // 2, len(hard_rows))
    for row in hard_rows:
        if len(chosen) >= hard_quota:
            break
        idx = int(row["idx"])
        if idx not in chosen_idx:
            chosen.append(row)
            chosen_idx.add(idx)

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
            "input_length": int(row["input_length"]),
            "last_token_idx": int(row["last_token_idx"]),
        }
        for row in chosen
    ]


def register_projection_hooks(model: nn.Module, calibration_values_per_hook: int):
    stats: dict[str, dict[str, Any]] = {}
    handles = []

    def update_absmax(name: str, tensor: torch.Tensor, key: str) -> None:
        absmax = float(tensor.detach().abs().max().item())
        prev = stats.setdefault(
            name,
            {
                "input_absmax": 0.0,
                "output_absmax": 0.0,
                "input_samples": [],
                "output_samples": [],
            },
        )
        prev[key] = max(prev[key], absmax)

    def update_samples(name: str, tensor: torch.Tensor, key: str) -> None:
        sampled = sample_abs_values(tensor, calibration_values_per_hook)
        prev = stats.setdefault(
            name,
            {
                "input_absmax": 0.0,
                "output_absmax": 0.0,
                "input_samples": [],
                "output_samples": [],
            },
        )
        prev[key].append(sampled)

    for module_name, module in model.named_modules():
        if not is_quantizable_projection_module(module_name, module):
            continue

        norm_name = normalize_linear_name(module_name)

        def make_hook(name: str):
            def hook(_module, module_inputs, module_output):
                update_absmax(name, module_inputs[0], "input_absmax")
                update_absmax(name, module_output, "output_absmax")
                update_samples(name, module_inputs[0], "input_samples")
                update_samples(name, module_output, "output_samples")

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
    activation_percentile: float,
    calibration_values_per_hook: int,
) -> tuple[dict[str, dict[str, float]], int]:
    stats, handles = register_projection_hooks(model, calibration_values_per_hook)
    total = min(len(dataset), calibration_samples)

    for start in range(0, total, calibration_batch_size):
        end = min(start + calibration_batch_size, total)
        batch = dataset[start:end]
        inputs = tokenize_gpt2_batch(
            tokenizer=tokenizer,
            texts=batch["sentence"],
            max_length=max_length,
        )
        with torch.no_grad():
            _ = model(**inputs)

        if start % (calibration_batch_size * 4) == 0:
            print(f"[quant] calibration processed {end}/{total}")

    for handle in handles:
        handle.remove()

    for name, values in stats.items():
        input_samples = np.concatenate(values["input_samples"]) if values["input_samples"] else np.zeros((1,), dtype=np.float32)
        output_samples = np.concatenate(values["output_samples"]) if values["output_samples"] else np.zeros((1,), dtype=np.float32)
        input_clip = float(np.percentile(input_samples, activation_percentile))
        output_clip = float(np.percentile(output_samples, activation_percentile))
        values["input_clip_absmax"] = min(values["input_absmax"], max(input_clip, 1.0e-8))
        values["output_clip_absmax"] = min(values["output_absmax"], max(output_clip, 1.0e-8))
        values["input_scale"] = max(values["input_clip_absmax"] / 127.0, 1.0e-8)
        values["output_scale"] = max(values["output_clip_absmax"] / 127.0, 1.0e-8)
        del values["input_samples"]
        del values["output_samples"]

    return stats, total


def build_quantized_linear_packs(
    model: nn.Module,
    activation_stats: dict[str, dict[str, float]],
) -> dict[str, LinearTensorPack]:
    packs: dict[str, LinearTensorPack] = {}

    for module_name, module in model.named_modules():
        if not is_quantizable_projection_module(module_name, module):
            continue

        norm_name = normalize_linear_name(module_name)
        stats = activation_stats[norm_name]
        weight, bias, in_features, out_features = get_canonical_weight_and_bias(module)
        weight_q, weight_scale = quantize_weight_per_channel(weight)
        bias_fp32 = (
            bias.cpu().numpy().astype(np.float32)
            if bias is not None
            else np.zeros((out_features,), dtype=np.float32)
        )

        packs[norm_name] = LinearTensorPack(
            weight_q=weight_q,
            weight_scale=weight_scale,
            input_scale=float(stats["input_scale"]),
            output_scale=float(stats["output_scale"]),
            bias_fp32=bias_fp32,
            in_features=in_features,
            out_features=out_features,
        )

    return packs


def extract_optional_lora_payload(model: nn.Module) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}

    for module_name, module in model.named_modules():
        if not hasattr(module, "base_layer") or not isinstance(module.base_layer, Conv1D):
            continue
        norm_name = normalize_linear_name(module_name)
        if not should_quantize_linear(norm_name):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        active_adapters = list(getattr(module, "active_adapters", []))
        if not active_adapters:
            continue

        adapter_name = active_adapters[0]
        lora_a = module.lora_A[adapter_name].weight.detach().cpu().numpy().astype(np.float32)
        lora_b = module.lora_B[adapter_name].weight.detach().cpu().numpy().astype(np.float32)
        scaling = float(module.scaling[adapter_name])
        key = sanitize_key(norm_name)

        payload[f"{key}__lora_a"] = lora_a
        payload[f"{key}__lora_b"] = lora_b
        payload[f"{key}__lora_scaling"] = np.asarray(scaling, dtype=np.float32)
        payload[f"{key}__lora_rank"] = np.asarray(lora_a.shape[0], dtype=np.int32)

    return payload


def build_npz_payload(
    model,
    linear_packs: dict[str, LinearTensorPack],
    activation_stats: dict[str, dict[str, float]],
    debug_samples: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    payload.update(build_model_config_payload(model.config))

    for name, pack in sorted(linear_packs.items()):
        key = sanitize_key(name)
        payload[f"{key}__weight_q"] = pack.weight_q.astype(np.int8)
        payload[f"{key}__bias_fp32"] = pack.bias_fp32.astype(np.float32)
        payload[f"{key}__weight_scale"] = pack.weight_scale.astype(np.float32)
        payload[f"{key}__input_scale"] = np.asarray(pack.input_scale, dtype=np.float32)
        payload[f"{key}__output_scale"] = np.asarray(pack.output_scale, dtype=np.float32)
        payload[f"{key}__in_features"] = np.asarray(pack.in_features, dtype=np.int32)
        payload[f"{key}__out_features"] = np.asarray(pack.out_features, dtype=np.int32)
        payload[f"{key}__input_absmax"] = np.asarray(activation_stats[name]["input_absmax"], dtype=np.float32)
        payload[f"{key}__output_absmax"] = np.asarray(activation_stats[name]["output_absmax"], dtype=np.float32)

    payload.update(extract_fp32_non_linear_tensors(model))
    payload.update(extract_optional_lora_payload(model))

    if debug_samples:
        payload["debug__indices"] = np.asarray([row["idx"] for row in debug_samples], dtype=np.int32)
        payload["debug__labels"] = np.asarray([row["label"] for row in debug_samples], dtype=np.int32)
        payload["debug__fp32_preds"] = np.asarray([row["pred"] for row in debug_samples], dtype=np.int32)
        payload["debug__fp32_logits"] = np.asarray([row["logits"] for row in debug_samples], dtype=np.float32)
        payload["debug__last_token_idx"] = np.asarray([row["last_token_idx"] for row in debug_samples], dtype=np.int32)
        payload["debug__input_length"] = np.asarray([row["input_length"] for row in debug_samples], dtype=np.int32)

    return payload


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_quant_meta(
    model_name: str,
    split: str,
    max_length: int,
    calibration_used: int,
    activation_percentile: float,
    linear_packs: dict[str, LinearTensorPack],
) -> dict[str, Any]:
    linear_names = sorted(linear_packs.keys())
    return {
        "model_name": model_name,
        "arch_name": ARCH_NAME,
        "split": split,
        "max_length": max_length,
        "quantization_scheme": "gpt2_projection_int8_weight_per_channel",
        "calibration_samples_used": calibration_used,
        "activation_percentile": activation_percentile,
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

    predictions_path = args.fp32_output_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing FP32 baseline predictions: {predictions_path}. Run 00_fp32_baseline.py first."
        )

    print(f"[quant] loading tokenizer and {ARCH_NAME} model")
    tokenizer, model = load_model(args.model_name, local_files_only=local_files_only)

    print(f"[quant] loading SST-2 split={args.split}")
    dataset = load_sst2(args.split)

    print("[quant] collecting activation statistics")
    activation_stats, calibration_used = collect_activation_stats(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_length=args.max_length,
        calibration_samples=args.calibration_samples,
        calibration_batch_size=args.calibration_batch_size,
        activation_percentile=args.activation_percentile,
        calibration_values_per_hook=args.calibration_values_per_hook,
    )

    print("[quant] quantizing GPT2 projection weights")
    linear_packs = build_quantized_linear_packs(model, activation_stats)

    fp32_rows = load_fp32_prediction_rows(predictions_path)
    debug_samples = select_debug_samples(fp32_rows, args.debug_samples)

    payload = build_npz_payload(model, linear_packs, activation_stats, debug_samples)
    model_npz_path = args.output_dir / "model_int8.npz"
    np.savez(model_npz_path, **payload)

    quant_meta = build_quant_meta(
        model_name=args.model_name,
        split=args.split,
        max_length=args.max_length,
        calibration_used=calibration_used,
        activation_percentile=args.activation_percentile,
        linear_packs=linear_packs,
    )
    quant_meta_path = args.output_dir / "quant_meta.json"
    debug_samples_path = args.output_dir / "debug_samples.json"
    write_json(quant_meta_path, quant_meta)
    write_json(debug_samples_path, debug_samples)

    print(f"[quant] projection layers   : {len(linear_packs)}")
    print(f"[quant] calibration used   : {calibration_used}")
    print(f"[quant] debug samples      : {len(debug_samples)}")
    print(f"[quant] model npz          : {model_npz_path}")
    print(f"[quant] quant meta         : {quant_meta_path}")
    print(f"[quant] debug sample list  : {debug_samples_path}")


if __name__ == "__main__":
    main()
