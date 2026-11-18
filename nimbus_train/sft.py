"""Supervised fine-tuning with completion-only loss masking.

Works in two modes:

* ``mode="full"``   -- every parameter trains (small models / CPU budgets).
* ``mode="lora"``   -- frozen base + low-rank adapters; the backward pass
  skips the frozen weights' gradient matmuls entirely.

The dataset is JSONL ``{"prompt": ..., "completion": ...}``; the loss counts
only positions whose *target* token is inside the completion.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .config import ModelConfig
from .data import SFTDataset, masked_loss_and_grad
from .lora import add_lora, load_lora, n_trainable, trainable_keys
from .optimizer import AdamW, clip_gradients, cosine_lr
from .ops import DTYPE
from .trainer import _append_log


def sft(
    config: ModelConfig,
    params: dict[str, np.ndarray],
    data_path: str,
    tokenizer_dir: str,
    out_dir: str,
    seq_len: int = 256,
    steps: int = 300,
    batch_size: int = 8,
    lr: float = 1e-3,
    warmup: int = 20,
    weight_decay: float = 0.0,
    clip: float = 1.0,
    mode: str = "lora",
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_targets: tuple[str, ...] = ("q", "k", "v", "o"),
    lora_path: str | None = None,
    log_every: int = 10,
    ckpt_every: int = 100,
    seed: int = 0,
    log_fn=print,
) -> dict:
    """Fine-tune ``params`` in place on prompt/completion pairs."""
    from .tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load(tokenizer_dir)

    dataset = SFTDataset(data_path, tokenizer, seq_len)
    if len(dataset) == 0:
        raise ValueError(f"no usable records in {data_path}")

    meta = None
    if mode == "lora":
        if lora_path:
            meta = load_lora(lora_path, params)
            log_fn(f"[lora] loaded adapters from {lora_path}")
        else:
            meta = add_lora(params, config, r=lora_r, alpha=lora_alpha, targets=lora_targets, seed=seed)
            log_fn(f"[lora] attached r={lora_r} alpha={lora_alpha} targets={lora_targets}")
        train_on = trainable_keys(params)
        log_fn(f"[lora] trainable {n_trainable(params):,} / {sum(int(v.size) for v in params.values()):,} params")
    elif mode == "full":
        train_on = None
    else:
        raise ValueError(f"mode must be 'full' or 'lora', got {mode!r}")

    opt = AdamW(params, lr=lr, weight_decay=weight_decay, exclude_from_wd=("norm", "bias", "lora"))
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    history: list[float] = []

    for step in range(steps):
        cur_lr = cosine_lr(step, lr, warmup, steps, min_ratio=0.1)
        x, y, mask = dataset.batch(batch_size, rng)
        loss, grads = masked_loss_and_grad(config, params, x, y, mask)
        grads = {k: v for k, v in grads.items() if train_on is None or k in train_on}
        clip_gradients(grads, clip)
        opt.step(params, grads, lr=cur_lr)
        history.append(loss)
        if (step + 1) % log_every == 0 or step == steps - 1:
            recent = sum(history[-log_every:]) / len(history[-log_every:])
            log_fn(f"sft step {step + 1:>5}/{steps}  loss {loss:.4f}  avg {recent:.4f}  lr {cur_lr:.2e}")
            _append_log(out_dir, [step + 1, f"{loss:.4f}", "", f"{cur_lr:.2e}", "", f"{time.time()-t0:.0f}"])
        if (step + 1) % ckpt_every == 0 or step == steps - 1:
            np.savez(os.path.join(out_dir, "params.npz"), **params)
            if meta is not None:
                from .lora import save_lora

                save_lora(os.path.join(out_dir, "lora.npz"), params, meta)

    return {"steps": steps, "final_loss": history[-1] if history else None}
