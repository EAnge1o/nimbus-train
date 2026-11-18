"""Direct Preference Optimization (Rafailov et al., 2023) -- hand-written.

The DPO objective replaces the reward model with the frozen reference policy:

    loss = -log sigma( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                              - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

The gradient with respect to the policy logits is derived by hand and verified
against finite differences in the test suite:

    d loss / d logit_t  =  c * (softmax(logit_t) - onehot(y_t))

with ``c = -/+ beta * (1 - sigma(z)) / n_pairs`` for the chosen/rejected
sequences. The reference model never receives gradients, and with LoRA the
backward skips the frozen base entirely.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .config import ModelConfig
from .lora import add_lora, load_lora, n_trainable, trainable_keys
from .model import backward, forward
from .ops import DTYPE
from .optimizer import AdamW, clip_gradients, cosine_lr
from .trainer import _append_log
from .tokenizer import BPETokenizer


def _logsigmoid(z):
    return -np.logaddexp(0.0, -z)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logprobs_from_logits(logits, y, mask):
    """Sum of target-token log-softmax over masked positions + fp32 probs."""
    logits64 = logits.astype(np.float64)
    shifted = logits64 - logits64.max(axis=-1, keepdims=True)
    log_softmax = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    n = y.shape[0]
    lp = np.zeros(n)
    for b in range(n):
        lp[b] = log_softmax[b][mask[b], y[b][mask[b]]].sum()
    return lp, np.exp(log_softmax).astype(DTYPE)


def _dpo_dlogits(probs, y, mask, coef):
    """Hand-derived gradient of sum-logp wrt logits, scaled per row.

    ``coef`` is (n,) -- one signed beta*(1-sigma)/n factor per pair row.
    For each masked position ``t``:  d log p(y_t) / d logit_t = onehot - p,
    so the scaled gradient is ``coef * (onehot - p)``; unmasked logits get 0.
    """
    dlogits = np.zeros_like(probs)
    rows, pos = np.nonzero(mask)
    dlogits[rows, pos, :] = -coef[rows][:, None] * probs[rows, pos, :]
    dlogits[rows, pos, y[rows, pos]] += coef[rows]
    return dlogits


def dpo(
    config: ModelConfig,
    params: dict[str, np.ndarray],
    ref_params: dict[str, np.ndarray],
    data_path: str,
    tokenizer_dir: str,
    out_dir: str,
    seq_len: int = 256,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 5e-4,
    beta: float = 0.1,
    warmup: int = 10,
    clip: float = 1.0,
    mode: str = "lora",
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_path: str | None = None,
    log_every: int = 10,
    ckpt_every: int = 100,
    seed: int = 0,
    log_fn=print,
) -> dict:
    """Train ``params`` on preference pairs against the frozen ``ref_params``."""
    tokenizer = BPETokenizer.load(tokenizer_dir)

    pairs = []
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if not pairs:
        raise ValueError(f"no preference pairs in {data_path}")

    # tokenize once: (x, y, mask) per row, chosen and rejected stacked
    def encode_side(prompt, completion):
        p_ids = tokenizer.encode(prompt)
        c_ids = tokenizer.encode(completion)
        total = p_ids + c_ids
        total = total[:seq_len]
        n = len(total)
        x = np.zeros(seq_len, dtype=np.int64)
        y = np.zeros(seq_len, dtype=np.int64)
        mask = np.zeros(seq_len, dtype=bool)
        x[:n - 1] = total[:-1]
        y[:n - 1] = total[1:]
        # loss on positions whose target is a completion token
        mask[len(p_ids) - 1: n - 1] = True
        mask &= np.arange(seq_len) < n - 1
        return x, y, mask

    encoded = []
    for rec in pairs:
        prompt = rec["prompt"]
        xc, yc, mc = encode_side(prompt, rec["chosen"])
        xr, yr, mr = encode_side(prompt, rec["rejected"])
        encoded.append(((xc, yc, mc), (xr, yr, mr)))

    meta = None
    if mode == "lora":
        if lora_path:
            meta = load_lora(lora_path, params)
        else:
            meta = add_lora(params, config, r=lora_r, alpha=lora_alpha, seed=seed)
        train_on = trainable_keys(params)
        log_fn(f"[dpo/lora] trainable {n_trainable(params):,} params, beta={beta}")
    elif mode == "full":
        train_on = None
    else:
        raise ValueError(f"mode must be 'full' or 'lora', got {mode!r}")

    opt = AdamW(params, lr=lr, weight_decay=0.0, exclude_from_wd=("norm", "bias", "lora"))
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    history: list[tuple[float, float]] = []

    for step in range(steps):
        cur_lr = cosine_lr(step, lr, warmup, steps, min_ratio=0.1)
        idx = rng.integers(0, len(encoded), size=batch_size)

        xs, ys, ms = [], [], []
        for i in idx:
            (xc, yc, mc), (xr, yr, mr) = encoded[int(i)]
            xs += [xc, xr]
            ys += [yc, yr]
            ms += [mc, mr]
        x = np.stack(xs)
        y = np.stack(ys)
        mask = np.stack(ms)

        # reference logprobs (frozen, no gradient)
        ref_logits, _ = forward(config, ref_params, x, need_grad=False)
        lp_ref, _ = _logprobs_from_logits(ref_logits, y, mask)
        # policy forward once: logits for logprobs, ctx for the backward pass
        pol_logits, ctx = forward(config, params, x, need_grad=True)
        lp_pol, probs = _logprobs_from_logits(pol_logits, y, mask)

        pol_c, pol_r = lp_pol[0::2], lp_pol[1::2]
        ref_c, ref_r = lp_ref[0::2], lp_ref[1::2]
        z = beta * ((pol_c - ref_c) - (pol_r - ref_r))
        loss = -_logsigmoid(z).mean()
        acc = float((z > 0).mean())

        # dloss/dz = -(1 - sigma(z)) / n_pairs;  z depends on pol_c (+) and pol_r (-)
        dz_dpol_c = -beta * (1.0 - _sigmoid(z)) / len(z)
        dz_dpol_r = +beta * (1.0 - _sigmoid(z)) / len(z)
        coef_pair = np.empty(batch_size * 2, dtype=np.float64)
        coef_pair[0::2] = dz_dpol_c
        coef_pair[1::2] = dz_dpol_r

        dlogits = _dpo_dlogits(probs, y, mask, coef_pair)
        grads = backward(config, params, ctx, dlogits, train_on)
        grads = {k: v for k, v in grads.items() if train_on is None or k in train_on}
        clip_gradients(grads, clip)
        opt.step(params, grads, lr=cur_lr)

        history.append((float(loss), acc))
        if (step + 1) % log_every == 0 or step == steps - 1:
            recent = history[-log_every:]
            avg_loss = sum(l for l, _ in recent) / len(recent)
            avg_acc = sum(a for _, a in recent) / len(recent)
            log_fn(f"dpo step {step + 1:>5}/{steps}  loss {loss:.4f}  avg {avg_loss:.4f}  acc {avg_acc:.2f}  lr {cur_lr:.2e}")
            _append_log(out_dir, [step + 1, f"{loss:.4f}", f"acc={avg_acc:.2f}", f"{cur_lr:.2e}", "", f"{time.time()-t0:.0f}"])
        if (step + 1) % ckpt_every == 0 or step == steps - 1:
            np.savez(os.path.join(out_dir, "params.npz"), **params)
            if meta is not None:
                from .lora import save_lora

                save_lora(os.path.join(out_dir, "lora.npz"), params, meta)

    return {"steps": steps, "final_loss": history[-1][0] if history else None,
            "final_acc": history[-1][1] if history else None}
