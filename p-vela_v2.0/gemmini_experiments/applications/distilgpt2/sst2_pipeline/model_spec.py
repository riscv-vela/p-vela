from __future__ import annotations

from typing import Any

import numpy as np


ARCH_NAME = "DistilGPT2"
DEFAULT_MODEL_NAME = "etechoptimist/polarityclassifier-distilgpt2-lora"
DEFAULT_BASE_MODEL_NAME = "distilbert/distilgpt2"
DEFAULT_NUM_LABELS = 2

DEBUG_HOOK_POINTS = {
    "transformer.h.0.attn.c_attn": "layer0_qkv_lin",
    "transformer.h.0.attn.c_proj": "layer0_out_lin",
    "transformer.h.0.mlp.c_fc": "layer0_ffn_lin1",
}
LAST_TOKEN_HIDDEN_MODULE = "transformer.ln_f"


def sanitize_key(name: str) -> str:
    return name.replace(".", "__")


def strip_peft_prefix(module_name: str) -> str:
    prefixes = (
        "base_model.model.",
        "base_model.",
    )
    for prefix in prefixes:
        if module_name.startswith(prefix):
            return module_name[len(prefix):]
    return module_name


def normalize_linear_name(module_name: str) -> str:
    clean_name = strip_peft_prefix(module_name)
    prefix = "transformer.h."
    if clean_name.startswith(prefix):
        tail = clean_name[len(prefix):]
        layer_str, rest = tail.split(".", 1)
        rest = rest.replace("attn.c_attn.base_layer", "qkv_lin")
        rest = rest.replace("attn.c_attn", "qkv_lin")
        rest = rest.replace("attn.c_proj", "out_lin")
        rest = rest.replace("mlp.c_fc", "ffn_lin1")
        rest = rest.replace("mlp.c_proj", "ffn_lin2")
        return f"layer{layer_str}.{rest}"
    if clean_name == "score.modules_to_save.default":
        return "classifier"
    if clean_name == "score.original_module":
        return "classifier"
    if clean_name == "score":
        return "classifier"
    return clean_name


def should_quantize_linear(norm_name: str) -> bool:
    if norm_name == "classifier":
        return False
    quantized_suffixes = ("qkv_lin", "out_lin", "ffn_lin1", "ffn_lin2")
    return any(norm_name.endswith(suffix) for suffix in quantized_suffixes)


def extract_fp32_non_linear_tensors(model: Any) -> dict[str, np.ndarray]:
    state = model.state_dict()
    keep_keys = [
        "base_model.model.transformer.wte.weight",
        "base_model.model.transformer.wpe.weight",
        "base_model.model.transformer.ln_f.weight",
        "base_model.model.transformer.ln_f.bias",
        "base_model.model.score.modules_to_save.default.weight",
        "base_model.model.score.modules_to_save.default.bias",
        "base_model.model.score.original_module.weight",
        "base_model.model.score.original_module.bias",
    ]

    extra_layernorm_keys = []
    for name in state.keys():
        clean_name = strip_peft_prefix(name)
        if clean_name.endswith("ln_1.weight") or clean_name.endswith("ln_1.bias"):
            extra_layernorm_keys.append(name)
        if clean_name.endswith("ln_2.weight") or clean_name.endswith("ln_2.bias"):
            extra_layernorm_keys.append(name)

    unique_keys = sorted(set(keep_keys + extra_layernorm_keys))
    tensors: dict[str, np.ndarray] = {}
    for key in unique_keys:
        if key in state:
            tensors[sanitize_key(strip_peft_prefix(key))] = (
                state[key].detach().cpu().numpy().astype(np.float32)
            )

    return tensors


def build_model_config_payload(config: Any) -> dict[str, np.ndarray]:
    return {
        "config__n_layers": np.asarray(config.n_layer, dtype=np.int32),
        "config__dim": np.asarray(config.n_embd, dtype=np.int32),
        "config__hidden_dim": np.asarray(config.n_inner or (config.n_embd * 4), dtype=np.int32),
        "config__n_heads": np.asarray(config.n_head, dtype=np.int32),
        "config__max_position_embeddings": np.asarray(config.n_positions, dtype=np.int32),
        "config__num_labels": np.asarray(config.num_labels, dtype=np.int32),
        "config__vocab_size": np.asarray(config.vocab_size, dtype=np.int32),
        "config__pad_token_id": np.asarray(config.pad_token_id, dtype=np.int32),
        "config__eos_token_id": np.asarray(config.eos_token_id, dtype=np.int32),
    }
