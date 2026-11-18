"""LoRA: low-rank adapters on a frozen base model (Hu et al., 2021).

Adapters live in the same parameter dict as the base weights, under
``<weight-name>.lora_A`` (in, r) and ``<weight-name>.lora_B`` (r, out). The
model's forward/backward include the ``(x @ A) @ B`` delta automatically
(``model._site_forward`` / ``model._site_backward``), and ``B`` starts at
zero, so a freshly attached adapter is a mathematical no-op -- the test suite
asserts logits are bit-identical before/after attachment.

Because the backward accepts a ``train_only`` set, fine-tuning a LoRA on CPU
also *skips the frozen weights' gradient matmuls*, not just their updates.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .config import ModelConfig
from .ops import DTYPE

DEFAULT_TARGETS = ("q", "k", "v", "o")
_SITE_SUFFIXES = {"q": "attn.q.weight", "k": "attn.k.weight", "v": "attn.v.weight",
                  "o": "attn.o.weight", "fc": "mlp.fc.weight", "proj": "mlp.proj.weight",
                  "gate": "mlp.gate.weight", "up": "mlp.up.weight", "down": "mlp.down.weight"}


def add_lora(
    params: dict[str, np.ndarray],
    config: ModelConfig,
    r: int = 8,
    alpha: float = 16.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    seed: int = 0,
) -> dict:
    """Attach LoRA adapters in place. Returns the adapter metadata dict.

    The ``alpha / r`` scaling is absorbed into ``A`` at initialisation
    (``delta = (x @ (A*s)) @ B`` == ``s * (x @ A) @ B``), so the model code
    needs no knowledge of alpha and the adapters stay a no-op at B = 0.
    """
    if r <= 0:
        raise ValueError("LoRA rank r must be positive")
    rng = np.random.default_rng(seed)
    d = config.d_model
    hd = config.resolved_head_dim
    H = config.n_heads
    KV = config.resolved_n_kv_heads
    mlp = config.d_mlp
    scale = alpha / r

    site_shapes = {
        "q": (d, H * hd), "o": (H * hd, d),
        "k": (d, KV * hd), "v": (d, KV * hd),
        "fc": (d, mlp), "proj": (mlp, d),
        "gate": (d, mlp), "up": (d, mlp), "down": (mlp, d),
    }

    for t in targets:
        if t not in site_shapes:
            raise ValueError(f"unknown LoRA target {t!r}; known: {sorted(site_shapes)}")
        if config.activation != "swiglu" and t in ("gate", "up", "down"):
            raise ValueError(f"target {t!r} needs activation='swiglu'")
        if config.activation == "swiglu" and t in ("fc", "proj"):
            raise ValueError(f"target {t!r} needs activation='gelu'")

    for i in range(config.n_layers):
        for t in targets:
            name = f"h.{i}.{_SITE_SUFFIXES[t]}"
            if name not in params:
                raise KeyError(f"base model has no parameter {name}")
            fan_in, fan_out = site_shapes[t]
            # standard LoRA init: A random small, B exactly zero -> no-op start
            params[name + ".lora_A"] = (rng.standard_normal((fan_in, r)) * 0.02 * scale).astype(DTYPE)
            params[name + ".lora_B"] = np.zeros((r, fan_out), dtype=DTYPE)

    meta = {"r": r, "alpha": alpha, "targets": list(targets), "seed": seed}
    return meta


def trainable_keys(params: dict[str, np.ndarray]) -> set[str]:
    return {k for k in params if k.endswith(".lora_A") or k.endswith(".lora_B")}


def n_trainable(params: dict[str, np.ndarray]) -> int:
    return sum(int(params[k].size) for k in trainable_keys(params))


def save_lora(path: str, params: dict[str, np.ndarray], meta: dict) -> None:
    """Save adapters + metadata (base weights are untouched by LoRA training)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: v for k, v in params.items() if ".lora_" in k})
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_lora(path: str, params: dict[str, np.ndarray]) -> dict:
    """Load adapters onto an existing (frozen-base) parameter dict."""
    meta_path = path + ".json"
    with np.load(path) as data:
        for k in data.files:
            params[k] = np.asarray(data[k], dtype=DTYPE)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    return meta
