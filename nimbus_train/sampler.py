"""Sampling for quick evaluation.

Each step re-runs the full sequence through :func:`nimbus_train.model.forward`
-- deliberately simple, because efficient incremental decoding (KV cache,
prefill/decode split) is nimbus-infer's job. This sampler exists so you can
inspect a checkpoint *during* training without leaving the repo.
"""
from __future__ import annotations

import numpy as np

from . import model as M


def generate(
    config,
    params,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: int = 0,
    stop_token: int | None = None,
) -> str:
    """Sample a continuation. temperature 0 -> greedy; top_p > 0 enables
    nucleus sampling; top_k > 0 truncates to the k most likely tokens."""
    rng = np.random.default_rng(seed)
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt encodes to no tokens")

    for _ in range(max_new_tokens):
        window = ids[-config.max_seq_len:]
        logits, _ = M.forward(config, params, np.asarray([window]), need_grad=False)
        probs = _next_distribution(logits[0, -1], temperature, top_k, top_p)
        if temperature == 0.0:
            nxt = int(np.argmax(probs))
        else:
            nxt = int(rng.choice(len(probs), p=probs))
        if stop_token is not None and nxt == stop_token:
            break
        ids.append(nxt)
    return tokenizer.decode(ids)


def _next_distribution(last_logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> np.ndarray:
    logits = np.asarray(last_logits, dtype=np.float64)
    if temperature > 0.0:
        logits = logits / temperature
    if top_k > 0:
        kth = np.sort(logits)[-min(top_k, logits.size)]
        logits = np.where(logits < kth, -np.inf, logits)
    shifted = logits - np.max(logits[np.isfinite(logits)] if np.isfinite(logits).any() else logits)
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    if top_p > 0.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        keep = order[cum - probs[order] < top_p]
        if keep.size == 0:
            keep = order[:1]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()
    return probs
