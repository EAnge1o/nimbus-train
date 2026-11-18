"""The pretraining loop: batches -> loss_and_grad -> clip -> AdamW.

Checkpoints are plain NumPy archives (``params.npz`` + ``optimizer.npz``) plus
a JSON config, so a run directory is transparent and resumable. A CSV log
(step, train_loss, val_loss, lr, tok/s, elapsed) is appended continuously.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from .config import ModelConfig
from .data import BatchSampler, estimate_loss
from .model import loss_and_grad
from .ops import DTYPE
from .optimizer import AdamW, clip_gradients, cosine_lr


class TrainConfig:
    """Hyperparameters of a pretraining run."""

    def __init__(
        self,
        steps: int = 1000,
        batch_size: int = 32,
        seq_len: int = 256,
        lr: float = 3e-3,
        warmup: int = 200,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        clip: float = 1.0,
        eval_every: int = 200,
        eval_batches: int = 16,
        log_every: int = 10,
        ckpt_every: int = 500,
        seed: int = 0,
    ) -> None:
        self.steps = steps
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.warmup = warmup
        self.min_lr_ratio = min_lr_ratio
        self.weight_decay = weight_decay
        self.betas = betas
        self.clip = clip
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.seed = seed

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        d["betas"] = tuple(d["betas"])
        return cls(**d)


def save_run(run_dir: str, config: ModelConfig, params: dict, train_cfg: TrainConfig,
             step: int, opt: AdamW | None = None) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        fh.write(config.to_json())
    with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump({**train_cfg.to_dict(), "step": step}, fh, indent=2)
    np.savez(os.path.join(run_dir, "params.npz"), **params)
    if opt is not None:
        state = {"step": np.asarray(opt.step_count)}
        for k, v in opt.m.items():
            state[f"m.{k}"] = v
        for k, v in opt.v.items():
            state[f"v.{k}"] = v
        np.savez(os.path.join(run_dir, "optimizer.npz"), **state)


def load_run(run_dir: str):
    """Load (config, train_config, params, optimizer-state-or-None, step)."""
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        config = ModelConfig.from_json(fh.read())
    with open(os.path.join(run_dir, "train_config.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    step = int(raw.get("step", 0))
    train_cfg = TrainConfig.from_dict({k: v for k, v in raw.items() if k != "step"})
    with np.load(os.path.join(run_dir, "params.npz")) as data:
        params = {k: np.asarray(data[k], dtype=DTYPE) for k in data.files}
    opt_state = None
    opt_path = os.path.join(run_dir, "optimizer.npz")
    if os.path.exists(opt_path):
        opt_state = {}
        with np.load(opt_path) as data:
            opt_state["step"] = int(data["step"])
            opt_state["m"] = {k[2:]: data[k] for k in data.files if k.startswith("m.")}
            opt_state["v"] = {k[2:]: data[k] for k in data.files if k.startswith("v.")}
    return config, train_cfg, params, opt_state, step


def _append_log(run_dir: str, row: list) -> None:
    path = os.path.join(run_dir, "log.csv")
    os.makedirs(run_dir, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["step", "train_loss", "val_loss", "lr", "tok_per_s", "elapsed_s"])
        w.writerow(row)


def pretrain(
    config: ModelConfig,
    train_cfg: TrainConfig,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    run_dir: str,
    resume: bool = True,
    log_fn=print,
) -> dict:
    """Run pretraining; returns the final stats dict."""
    step = 0
    params = None
    if resume and os.path.exists(os.path.join(run_dir, "params.npz")):
        loaded_cfg, loaded_train, params, opt_state, step = load_run(run_dir)
        if loaded_cfg.to_dict() != config.to_dict():
            raise ValueError("run dir holds a different model config; pass --fresh")
        # the caller's train config wins -- resuming with a larger ``steps``
        # target simply extends the schedule from the checkpointed step
        log_fn(f"[resume] continuing from step {step} (stored run used {loaded_train.to_dict()})")

    if params is None:
        from .model import init_params

        params = init_params(config, seed=train_cfg.seed)

    train_sampler = BatchSampler(train_tokens, train_cfg.seq_len)
    val_sampler = BatchSampler(val_tokens, train_cfg.seq_len) if val_tokens.size > train_cfg.seq_len else None

    opt = AdamW(
        params,
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )
    if step and opt_state is not None:
        opt.load_state_dict({"step": opt_state["step"], "m": opt_state["m"], "v": opt_state["v"]})

    rng = np.random.default_rng(train_cfg.seed + 12345 + step)
    t0 = time.time()
    tokens_seen = 0
    history: dict[str, list] = {"step": [], "train": [], "val": []}

    log_fn(
        f"params={sum(int(v.size) for v in params.values()):,} "
        f"layers={config.n_layers} d={config.d_model} vocab={config.vocab_size} "
        f"batch={train_cfg.batch_size}x{train_cfg.seq_len} steps={train_cfg.steps}"
    )

    while step < train_cfg.steps:
        lr = cosine_lr(step, train_cfg.lr, train_cfg.warmup, train_cfg.steps, train_cfg.min_lr_ratio)
        x, y = train_sampler.batch(train_cfg.batch_size, rng)
        loss, grads = loss_and_grad(config, params, x, y)
        clip_gradients(grads, train_cfg.clip)
        opt.step(params, grads, lr=lr)
        step += 1
        tokens_seen += x.size

        if step % train_cfg.log_every == 0 or step == train_cfg.steps:
            elapsed = time.time() - t0
            tps = tokens_seen / max(elapsed, 1e-9)
            val_loss = ""
            if val_sampler is not None and (step % train_cfg.eval_every == 0 or step == train_cfg.steps):
                val_loss = estimate_loss(config, params, val_sampler, train_cfg.batch_size, train_cfg.eval_batches)
                history["val"].append((step, val_loss))
            history["step"].append(step)
            history["train"].append(loss)
            _append_log(
                run_dir,
                [step, f"{loss:.4f}", f"{val_loss:.4f}" if val_loss != "" else "", f"{lr:.2e}", f"{tps:.0f}", f"{elapsed:.0f}"],
            )
            log_fn(
                f"step {step:>6}/{train_cfg.steps}  loss {loss:.4f}"
                + (f"  val {val_loss:.4f}" if val_loss != "" else "")
                + f"  lr {lr:.2e}  {tps:,.0f} tok/s"
            )

        if step % train_cfg.ckpt_every == 0 or step == train_cfg.steps:
            save_run(run_dir, config, params, train_cfg, step, opt)

    stats = {
        "steps": step,
        "final_loss": history["train"][-1] if history["train"] else None,
        "history": history,
        "wallclock_s": time.time() - t0,
    }
    return stats
