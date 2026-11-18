"""AdamW with decoupled weight decay, plus the LR schedule and grad clipping.

Hand-written to match the zero-dependency constraint: no torch, no optax.
The optimizer state (first/second moments, step count) is a plain dict keyed by
parameter name, so checkpoints stay transparent and resumable.
"""
from __future__ import annotations

import numpy as np

from .ops import DTYPE


def cosine_lr(step: int, base_lr: float, warmup: int, total_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup -> cosine decay to ``min_ratio * base_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(progress, 1.0)
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress)))


def global_norm(grads: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values())))


def clip_gradients(grads: dict[str, np.ndarray], max_norm: float) -> float:
    """Scale grads in place so their global L2 norm is ``max_norm``.

    Returns the pre-clip norm (for logging). No-op when the norm is smaller.
    """
    norm = global_norm(grads)
    if norm > max_norm and norm > 0:
        factor = max_norm / norm
        for k in grads:
            grads[k] *= factor
    return norm


class AdamW:
    """Decoupled weight decay Adam (Loshchilov & Hutter), bias-corrected.

    ``exclude_from_wd`` names parameters (by key substring) that should not get
    weight decay -- conventionally 1-D tensors: norms and biases.
    """

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        exclude_from_wd: tuple[str, ...] = ("norm", "bias"),
    ) -> None:
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.exclude_from_wd = exclude_from_wd
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.step_count = 0

    def _needs_wd(self, key: str) -> bool:
        return not any(s in key for s in self.exclude_from_wd)

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], lr: float | None = None) -> None:
        """One update in place. ``lr`` overrides the stored rate (schedules)."""
        lr = self.lr if lr is None else lr
        b1, b2 = self.betas
        self.step_count += 1
        t = self.step_count
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for key, g in grads.items():
            p = params[key]
            g = g.astype(DTYPE, copy=False)
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)
            m, v = self.m[key], self.v[key]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0 and self._needs_wd(key):
                update = update + self.weight_decay * p
            p -= lr * update

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> dict:
        return {"step": self.step_count, "m": self.m, "v": self.v}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step"]
        self.m = state["m"]
        self.v = state["v"]
