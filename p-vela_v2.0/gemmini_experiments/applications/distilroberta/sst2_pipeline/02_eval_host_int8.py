#!/usr/bin/env python3
"""
DistilRoBERTa host-side int8 evaluation for linear-only quantization.

This script is the stage-2 evaluator in the new DistilRoBERTa pipeline.
Its purpose is to answer two questions:
1. After quantizing every nn.Linear to int8, how much accuracy drops on
   SST-2 validation?
2. At a few fixed debug points, how far do the host-side int8 tensors drift
   from the original FP32 tensors?

How this script works:
- Load the original pretrained DistilRoBERTa model as the FP32 reference.
- Load the stage-1 quantization artifacts from model_int8.npz.
- Clone the model structure and replace every nn.Linear with a QuantizedLinear
  module. QuantizedLinear performs:
  input quantize -> int8/int32 matmul -> per-channel rescale -> bias add
  -> output requantize/dequantize
- Evaluate the quantized model on SST-2.
- Compare the final predictions against stage-0 FP32 results.
- Run both FP32 and host-int8 models on a fixed set of debug samples and save
  tensor comparison summaries.

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_spec import (
    ARCH_NAME,
    DEBUG_HOOK_POINTS,
    DEFAULT_MODEL_NAME,
    LAST_CLS_HIDDEN_MODULE,
    normalize_linear_name,
    sanitize_key,
    should_quantize_linear,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "int8_host"
DEFAULT_FP32_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fp32"
DEFAULT_SPLIT = "validation"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 128


# Drop-in replacement for nn.Linear used only in the host-side reference path.
class QuantizedLinear(nn.Module):
    """
    Host-side quantized Linear module.

    The goal here is not kernel speed. Instead, we want a clean and explicit
    reference implementation of:
    - int8 input quantization
    - int8 weight storage
    - int32 accumulation
    - float bias add
    - output requantize / dequantize
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Host-side int8 Linear reference:
        # this function is the actual int8 GEMM path that replaces nn.Linear.
        # 1) Quantize FP32 activations into the int8 domain.
        x_q = torch.clamp(torch.round(x / self.input_scale), -128, 127).to(torch.int32)
        w_q = self.weight_q.to(torch.int32)
        # 2) Integer GEMM: int8 values accumulated in int32.
        acc = torch.matmul(x_q, w_q.transpose(0, 1))
        # Per-output-channel weight scales reduce distortion in wide layers,
        # especially FFN and classifier projections.
        # 3) Rescale back to FP32, add FP32 bias, then requantize/dequantize.
        y = acc.to(torch.float32) * (self.input_scale * self.weight_scale)
        y = y + self.bias_fp32
        y_q = torch.clamp(torch.round(y / self.output_scale), -128, 127)
        return y_q.to(torch.float32) * self.output_scale


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


def get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_modules_with_quantized(model: nn.Module, npz: np.lib.npyio.NpzFile) -> nn.Module:
    quant_model = copy.deepcopy(model)

    # Keep the original Hugging Face model structure and replace only nn.Linear.
    # Everything else, such as embeddings/layernorm/softmax/GELU/residual, stays FP32.
    # After this replacement, model(**inputs) runs full inference with only the
    # Linear/GEMM path redirected into QuantizedLinear.forward().
    # Replace every nn.Linear with QuantizedLinear using the stage-1 artifact.
    for module_name, module in list(quant_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        norm_name = normalize_linear_name(module_name)
        if not should_quantize_linear(norm_name):
            continue
        key = sanitize_key(norm_name)
        qmodule = QuantizedLinear(
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

    # Main stage-2 evaluation pass:
    # run the host-side linear-only int8 model over SST-2 and save per-sample outputs.
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        batch = dataset[start:end]
        labels = torch.tensor(batch["label"], dtype=torch.long)
        inputs = tokenizer(
            batch["sentence"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            # This is the top-level host inference call.
            # The model structure is still Hugging Face DistilRoBERTa, but every
            # nn.Linear inside it has already been replaced with QuantizedLinear.
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1)

        correct += int((preds == labels).sum().item())

        for offset, sentence in enumerate(batch["sentence"]):
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


def attach_debug_hooks(model: nn.Module, sink: dict[str, np.ndarray]):
    handles = []

    for module_name, point_name in DEBUG_HOOK_POINTS.items():
        module = dict(model.named_modules())[module_name]

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                sink[name] = squeeze_tensor(output)

            return hook

        handles.append(module.register_forward_hook(make_hook(point_name)))

    pre_classifier = dict(model.named_modules())[LAST_CLS_HIDDEN_MODULE]

    def cls_hidden_hook(_module, inputs):
        sink["last_cls_hidden"] = squeeze_tensor(inputs[0])

    handles.append(pre_classifier.register_forward_pre_hook(cls_hidden_hook))
    return handles


def squeeze_tensor(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(value)}")
    arr = value.detach().cpu().numpy().astype(np.float32)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def run_debug_trace(
    model: nn.Module,
    tokenizer,
    sentence: str,
    max_length: int,
) -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {}
    handles = attach_debug_hooks(model, trace)

    inputs = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

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

    # Trace fixed checkpoints in both FP32 and host-int8 models so later stages
    # can compare against the exact same tensors.
    # Debug tracing is run sample-by-sample on fixed indices so later C and
    # Gemmini stages can compare against the exact same checkpoints.
    for sample in debug_samples:
        idx = int(sample["idx"])
        sentence = dataset[idx]["sentence"]
        fp32_trace = run_debug_trace(fp32_model, tokenizer, sentence, max_length)
        int8_trace = run_debug_trace(int8_model, tokenizer, sentence, max_length)

        for point in ["layer0_q_lin", "layer0_attn_out", "layer0_ffn_lin1", "last_cls_hidden", "logits"]:
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

    # Stage 2 consumes stage-0 baseline outputs and stage-1 quantized artifacts.
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

    # 1) Load the FP32 model structure, then load stage-1 quantized parameters.
    print(f"[int8_host] loading tokenizer and {ARCH_NAME} models")
    tokenizer, fp32_model = load_model(args.model_name, local_files_only=local_files_only)
    npz = np.load(model_npz_path)

    # 2) Build the host-side linear-only int8 reference model.
    int8_model = replace_linear_modules_with_quantized(fp32_model, npz)

    # 3) Evaluate the host int8 model on the same SST-2 split as stage 0.
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

    # 4) Compare final predictions against the saved FP32 baseline.
    compare_summary = compare_with_fp32(fp32_rows, int8_rows)
    debug_samples = load_json(debug_samples_path)

    # 5) Save intermediate tensor comparisons for later C / Gemmini validation.
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
