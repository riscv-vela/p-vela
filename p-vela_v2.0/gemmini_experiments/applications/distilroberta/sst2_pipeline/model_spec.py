from __future__ import annotations

from typing import Any

import numpy as np


ARCH_NAME = "DistilRoBERTa"
DEFAULT_MODEL_NAME = "gokuls/distilroberta-sst2"

DEBUG_HOOK_POINTS = {
    "roberta.encoder.layer.0.attention.self.query": "layer0_q_lin",
    "roberta.encoder.layer.0.attention.output.dense": "layer0_attn_out",
    "roberta.encoder.layer.0.intermediate.dense": "layer0_ffn_lin1",
}
LAST_CLS_HIDDEN_MODULE = "classifier.dense"


def sanitize_key(name: str) -> str:
    return name.replace(".", "__")


def normalize_linear_name(module_name: str) -> str:
    prefix = "roberta.encoder.layer."
    if module_name.startswith(prefix):
        tail = module_name[len(prefix):]
        layer_str, rest = tail.split(".", 1)
        rest = rest.replace("attention.self.query", "q_lin")
        rest = rest.replace("attention.self.key", "k_lin")
        rest = rest.replace("attention.self.value", "v_lin")
        rest = rest.replace("attention.output.dense", "out_lin")
        rest = rest.replace("intermediate.dense", "ffn_lin1")
        rest = rest.replace("output.dense", "ffn_lin2")
        return f"layer{layer_str}.{rest}"
    if module_name == "classifier.dense":
        return "pre_classifier"
    if module_name == "classifier.out_proj":
        return "classifier"
    return module_name


def should_quantize_linear(norm_name: str) -> bool:
    return True


def extract_fp32_non_linear_tensors(model: Any) -> dict[str, np.ndarray]:
    state = model.state_dict()
    keep_keys = [
        "roberta.embeddings.word_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight",
        "roberta.embeddings.token_type_embeddings.weight",
        "roberta.embeddings.LayerNorm.weight",
        "roberta.embeddings.LayerNorm.bias",
        "roberta.encoder.layer.0.attention.output.LayerNorm.weight",
        "roberta.encoder.layer.0.attention.output.LayerNorm.bias",
        "roberta.encoder.layer.0.output.LayerNorm.weight",
        "roberta.encoder.layer.0.output.LayerNorm.bias",
    ]

    extra_layernorm_keys = []
    for name in state.keys():
        if name.endswith("LayerNorm.weight") or name.endswith("LayerNorm.bias"):
            extra_layernorm_keys.append(name)

    unique_keys = sorted(set(keep_keys + extra_layernorm_keys))
    tensors: dict[str, np.ndarray] = {}
    for key in unique_keys:
        if key in state:
            tensors[sanitize_key(key)] = state[key].detach().cpu().numpy().astype(np.float32)

    return tensors


def build_model_config_payload(config: Any) -> dict[str, np.ndarray]:
    return {
        "config__n_layers": np.asarray(config.num_hidden_layers, dtype=np.int32),
        "config__dim": np.asarray(config.hidden_size, dtype=np.int32),
        "config__hidden_dim": np.asarray(config.intermediate_size, dtype=np.int32),
        "config__n_heads": np.asarray(config.num_attention_heads, dtype=np.int32),
        "config__max_position_embeddings": np.asarray(config.max_position_embeddings, dtype=np.int32),
        "config__num_labels": np.asarray(config.num_labels, dtype=np.int32),
        "config__vocab_size": np.asarray(config.vocab_size, dtype=np.int32),
        "config__type_vocab_size": np.asarray(config.type_vocab_size, dtype=np.int32),
        "config__pad_token_id": np.asarray(config.pad_token_id, dtype=np.int32),
    }
