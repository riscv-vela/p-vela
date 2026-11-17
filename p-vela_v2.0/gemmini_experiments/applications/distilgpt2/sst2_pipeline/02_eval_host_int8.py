#!/usr/bin/env python3
"""
DistilGPT2 host-side int8 evaluation for projection-only quantization.

This script is the stage-2 evaluator in the DistilGPT2 pipeline.
Its purpose is to answer two questions:
1. After quantizing GPT2 projection modules to int8, how much accuracy drops
   on SST-2 validation?
2. At a few fixed debug points, how far do host-side int8 tensors drift from
   the original FP32 tensors?

Outputs:
- outputs/int8_host/metrics.json
- outputs/int8_host/predictions.jsonl
- outputs/int8_host/debug_tensors.npz
- outputs/int8_host/debug_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
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
    DEBUG_HOOK_POINTS,
    DEFAULT_BASE_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    LAST_TOKEN_HIDDEN_MODULE,
    normalize_linear_name,
    sanitize_key,
    should_quantize_linear,
    strip_peft_prefix,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "int8_host"
DEFAULT_FP32_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fp32"
DEFAULT_SPLIT = "validation"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 128


class QuantizedProjection(nn.Module):
    def __init__(
        self,
        weight_q: np.ndarray,
        bias_fp32: np.ndarray,
        input_scale: float,
        weight_scale: np.ndarray,
        output_scale: float,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_q", torch.from_numpy(weight_q.astype(np.int8)))
        self.register_buffer("bias_fp32", torch.from_numpy(bias_fp32.astype(np.float32)))
        self.register_buffer("weight_scale", torch.from_numpy(weight_scale.astype(np.float32)))
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_q = torch.clamp(torch.round(x / self.input_scale), -128, 127).to(torch.int32)
        acc = torch.matmul(x_q, self.weight_q.to(torch.int32).transpose(0, 1))
        y = acc.to(torch.float32) * (self.input_scale * self.weight_scale)
        y = y + self.bias_fp32
        # In this mixed-precision reference path, GEMM runs in int8/int32 but the
        # result is brought back to FP32 immediately for residual/LN/non-linear ops.
        # The next projection quantizes its own input again, so an extra
        # requantize-dequantize step here only adds avoidable noise.
        return y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fp32-output-dir", type=Path, default=DEFAULT_FP32_OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_gpt2_tokenizer(model_name: str, local_files_only: bool):
    try:
        return GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except OSError:
        return AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_fast=False,
        )


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
        raise ImportError("peft is required for the DistilGPT2 stage-2 pipeline.") from exc

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


def load_sst2(split: str):
    return load_dataset("glue", "sst2", split=split)


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


def get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def is_quantizable_projection_module(module_name: str, module: nn.Module) -> bool:
    if isinstance(module, Conv1D):
        return should_quantize_linear(normalize_linear_name(module_name))
    if isinstance(module, nn.Linear) and hasattr(module, "base_layer") and isinstance(module.base_layer, Conv1D):
        return should_quantize_linear(normalize_linear_name(module_name))
    return False


def replace_projection_modules_with_quantized(model: nn.Module, npz: np.lib.npyio.NpzFile) -> nn.Module:
    quant_model = copy.deepcopy(model)

    for module_name, module in list(quant_model.named_modules()):
        if not is_quantizable_projection_module(module_name, module):
            continue

        norm_name = normalize_linear_name(module_name)
        key = sanitize_key(norm_name)
        qmodule = QuantizedProjection(
            weight_q=npz[f"{key}__weight_q"],
            bias_fp32=npz[f"{key}__bias_fp32"],
            input_scale=float(npz[f"{key}__input_scale"].item()),
            weight_scale=npz[f"{key}__weight_scale"],
            output_scale=float(npz[f"{key}__output_scale"].item()),
        )
        parent, attr_name = get_parent_module(quant_model, module_name)
        setattr(parent, attr_name, qmodule)

    quant_model.eval()
    return quant_model


def run_eval(
    model: nn.Module,
    tokenizer,
    dataset,
    batch_size: int,
    max_length: int,
    limit: int | None,
) -> tuple[list[dict[str, Any]], float]:
    total_size = len(dataset) if limit is None else min(len(dataset), limit)
    rows: list[dict[str, Any]] = []
    correct = 0

    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch = dataset[start:end]
        labels = torch.tensor(batch["label"], dtype=torch.long)
        inputs = tokenize_gpt2_batch(
            tokenizer=tokenizer,
            texts=batch["sentence"],
            max_length=max_length,
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)

        correct += int((preds == labels).sum().item())

        for offset, sentence in enumerate(batch["sentence"]):
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
                    "last_token_idx": input_length - 1,
                }
            )

        if start % (batch_size * 10) == 0:
            print(f"[int8_host] processed {end}/{total_size}")

    accuracy = (correct / total_size) if total_size > 0 else 0.0
    return rows, accuracy


def tensor_error(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs_flat = lhs.astype(np.float32).reshape(-1)
    rhs_flat = rhs.astype(np.float32).reshape(-1)
    diff = np.abs(lhs_flat - rhs_flat)
    lhs_norm = float(np.linalg.norm(lhs_flat))
    rhs_norm = float(np.linalg.norm(rhs_flat))
    denom = max(lhs_norm * rhs_norm, 1.0e-12)
    cosine = float(np.dot(lhs_flat, rhs_flat) / denom)
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "cosine": cosine,
    }


def squeeze_tensor(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(value)}")
    arr = value.detach().cpu().numpy().astype(np.float32)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def resolve_module_name(model: nn.Module, target_name: str) -> str:
    modules = dict(model.named_modules())
    if target_name in modules:
        return target_name
    for module_name in modules.keys():
        if strip_peft_prefix(module_name) == target_name:
            return module_name
    raise KeyError(f"Could not resolve module name {target_name!r}")


def attach_debug_hooks(model: nn.Module, sink: dict[str, np.ndarray]):
    handles = []

    for module_name, point_name in DEBUG_HOOK_POINTS.items():
        resolved_name = resolve_module_name(model, module_name)
        module = dict(model.named_modules())[resolved_name]

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                sink[name] = squeeze_tensor(output)

            return hook

        handles.append(module.register_forward_hook(make_hook(point_name)))

    last_hidden_name = resolve_module_name(model, LAST_TOKEN_HIDDEN_MODULE)
    last_hidden_module = dict(model.named_modules())[last_hidden_name]

    def last_hidden_hook(_module, _inputs, output):
        sink["_last_hidden_seq"] = squeeze_tensor(output)

    handles.append(last_hidden_module.register_forward_hook(last_hidden_hook))
    return handles


def run_debug_trace(
    model: nn.Module,
    tokenizer,
    sentence: str,
    max_length: int,
) -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {}
    handles = attach_debug_hooks(model, trace)

    inputs = tokenize_gpt2_batch(
        tokenizer=tokenizer,
        texts=[sentence],
        max_length=max_length,
    )
    last_token_idx = int(inputs["attention_mask"][0].sum().item()) - 1

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    trace["last_token_hidden"] = trace["_last_hidden_seq"][last_token_idx]
    del trace["_last_hidden_seq"]
    trace["logits"] = squeeze_tensor(outputs.logits)
    return trace


def build_debug_outputs(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    tokenizer,
    dataset,
    debug_samples: list[dict[str, Any]],
    max_length: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    tensors_payload: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    for sample in debug_samples:
        idx = int(sample["idx"])
        sentence = dataset[idx]["sentence"]
        fp32_trace = run_debug_trace(fp32_model, tokenizer, sentence, max_length)
        int8_trace = run_debug_trace(int8_model, tokenizer, sentence, max_length)

        for point in ["layer0_qkv_lin", "layer0_out_lin", "layer0_ffn_lin1", "last_token_hidden", "logits"]:
            fp32_tensor = fp32_trace[point]
            int8_tensor = int8_trace[point]
            stats = tensor_error(fp32_tensor, int8_tensor)
            summary_rows.append(
                {
                    "sample_idx": idx,
                    "point": point,
                    "shape": list(fp32_tensor.shape),
                    "max_abs": stats["max_abs"],
                    "mean_abs": stats["mean_abs"],
                    "cosine": stats["cosine"],
                }
            )

            base = f"sample_{idx}__{point}"
            tensors_payload[f"fp32__{base}"] = fp32_tensor.astype(np.float32)
            tensors_payload[f"int8__{base}"] = int8_tensor.astype(np.float32)

    return tensors_payload, summary_rows


def compare_with_fp32(fp32_rows: list[dict[str, Any]], int8_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(fp32_rows) != len(int8_rows):
        raise ValueError("FP32 and int8 prediction lengths do not match")

    pred_mismatches = 0
    mismatched_indices: list[int] = []
    fp32_correct = 0

    for fp32_row, int8_row in zip(fp32_rows, int8_rows):
        if int(fp32_row["idx"]) != int(int8_row["idx"]):
            raise ValueError("FP32 and int8 prediction rows are misaligned")
        if int(fp32_row["pred"]) == int(fp32_row["label"]):
            fp32_correct += 1
        if int(fp32_row["pred"]) != int(int8_row["pred"]):
            pred_mismatches += 1
            mismatched_indices.append(int(fp32_row["idx"]))

    return {
        "fp32_subset_accuracy": (fp32_correct / len(fp32_rows)) if fp32_rows else 0.0,
        "prediction_mismatches": pred_mismatches,
        "mismatched_indices": mismatched_indices,
    }


def main() -> None:
    args = parse_args()
    local_files_only = False if args.allow_download else args.local_files_only
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_npz_path = args.output_dir / "model_int8.npz"
    debug_samples_path = args.output_dir / "debug_samples.json"
    fp32_metrics_path = args.fp32_output_dir / "metrics.json"
    fp32_predictions_path = args.fp32_output_dir / "predictions.jsonl"

    if not model_npz_path.exists():
        raise FileNotFoundError(f"Missing quantized model artifact: {model_npz_path}. Run 01 first.")
    if not debug_samples_path.exists():
        raise FileNotFoundError(f"Missing debug sample list: {debug_samples_path}. Run 01 first.")
    if not fp32_metrics_path.exists() or not fp32_predictions_path.exists():
        raise FileNotFoundError("Missing FP32 baseline outputs. Run 00 first.")

    print("[int8_host] loading tokenizer and models")
    tokenizer, fp32_model = load_model(args.model_name, local_files_only=local_files_only)
    npz = np.load(model_npz_path)
    int8_model = replace_projection_modules_with_quantized(fp32_model, npz)

    print(f"[int8_host] loading SST-2 split={args.split}")
    dataset = load_sst2(args.split)

    print("[int8_host] evaluating host-side int8 model")
    int8_rows, int8_accuracy = run_eval(
        model=int8_model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
    )

    fp32_metrics = load_json(fp32_metrics_path)
    fp32_rows = load_jsonl(fp32_predictions_path)
    if args.limit is not None:
        fp32_rows = fp32_rows[: args.limit]

    compare_summary = compare_with_fp32(fp32_rows, int8_rows)
    debug_samples = load_json(debug_samples_path)

    print("[int8_host] collecting debug traces")
    debug_tensors, debug_summary = build_debug_outputs(
        fp32_model=fp32_model,
        int8_model=int8_model,
        tokenizer=tokenizer,
        dataset=dataset,
        debug_samples=debug_samples,
        max_length=args.max_length,
    )

    metrics = {
        "arch_name": ARCH_NAME,
        "model_name": args.model_name,
        "split": args.split,
        "num_samples": len(int8_rows),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "accuracy": int8_accuracy,
        "fp32_accuracy": float(compare_summary["fp32_subset_accuracy"]),
        "fp32_full_accuracy": float(fp32_metrics["accuracy"]),
        "accuracy_drop": float(compare_summary["fp32_subset_accuracy"]) - float(int8_accuracy),
        "prediction_mismatches_vs_fp32": compare_summary["prediction_mismatches"],
    }

    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "predictions.jsonl"
    debug_tensors_path = args.output_dir / "debug_tensors.npz"
    debug_summary_path = args.output_dir / "debug_summary.json"

    write_json(metrics_path, metrics)
    write_jsonl(predictions_path, int8_rows)
    np.savez(debug_tensors_path, **debug_tensors)
    write_json(debug_summary_path, debug_summary)

    print(f"[int8_host] accuracy              : {int8_accuracy:.6f}")
    print(f"[int8_host] fp32 accuracy         : {float(compare_summary['fp32_subset_accuracy']):.6f}")
    print(f"[int8_host] accuracy drop         : {metrics['accuracy_drop']:.6f}")
    print(f"[int8_host] pred mismatches       : {compare_summary['prediction_mismatches']}")
    print(f"[int8_host] metrics path          : {metrics_path}")
    print(f"[int8_host] predictions path      : {predictions_path}")
    print(f"[int8_host] debug tensors path    : {debug_tensors_path}")
    print(f"[int8_host] debug summary path    : {debug_summary_path}")


if __name__ == "__main__":
    main()
