"""Corpus preparation and batch sampling.

The token stream is stored as a raw ``uint16`` binary file (one id per 2
bytes), memory-mapped at train time so corpora larger than RAM cost nothing.
Documents are joined with the ``<|endoftext|>`` id when the tokenizer has one.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .tokenizer import BPETokenizer, ENDOFTEXT


def read_text_files(paths: list[str]) -> str:
    """Concatenate files or directories (glob patterns allowed)."""
    parts: list[str] = []
    for p in paths:
        matches = sorted(glob.glob(p)) or [p]
        for m in matches:
            if os.path.isdir(m):
                for f in sorted(glob.glob(os.path.join(m, "**", "*"), recursive=True)):
                    if os.path.isfile(f):
                        with open(f, encoding="utf-8") as fh:
                            parts.append(fh.read())
            elif os.path.exists(m):
                with open(m, encoding="utf-8") as fh:
                    parts.append(fh.read())
            else:
                raise FileNotFoundError(m)
    return "\n".join(parts)


def encode_corpus(tokenizer: BPETokenizer, text: str, val_frac: float = 0.01):
    """Encode documents into train/val token arrays.

    Documents are split on the end-of-text token if the tokenizer knows one,
    otherwise on blank-line boundaries; a contiguity-preserving tail slice
    becomes the validation split.
    """
    sep = tokenizer.special_tokens.get(ENDOFTEXT)
    if sep is not None:
        raw_docs = text.split(ENDOFTEXT)
    else:
        raw_docs = text.split("\n\n")
    docs = [d for d in (doc.strip() for doc in raw_docs) if d]

    ids: list[int] = []
    boundaries: list[int] = [0]
    for doc in docs:
        toks = tokenizer.encode(doc)
        ids.extend(toks)
        if sep is not None:
            ids.append(sep)
        boundaries.append(len(ids))
    all_ids = np.asarray(ids, dtype=np.uint16)

    split = int(len(all_ids) * (1.0 - val_frac))
    # snap to a document boundary so val starts on a fresh document
    snap = min(boundaries, key=lambda b: abs(b - split))
    if snap == 0:
        snap = boundaries[1] if len(boundaries) > 1 else len(all_ids)
    return all_ids[:snap], all_ids[snap:]


def save_tokens(ids: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ids.astype(np.uint16).tofile(path)


def load_tokens(path: str) -> np.memmap:
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    if arr.size == 0:
        raise ValueError(f"empty token file: {path}")
    return arr


class BatchSampler:
    """Random-window batches over a token stream (nanoGPT style)."""

    def __init__(self, tokens: np.ndarray, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        if tokens.size <= seq_len:
            raise ValueError(f"token stream ({tokens.size}) shorter than seq_len ({seq_len})")

    def batch(self, batch_size: int, rng: np.random.Generator):
        ix = rng.integers(0, self.tokens.size - self.seq_len - 1, size=batch_size)
        x = np.stack([self.tokens[i : i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.tokens[i + 1 : i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        return x, y

    def eval_batches(self, batch_size: int, n_batches: int, rng: np.random.Generator):
        """Deterministic-ish validation windows spread across the stream."""
        n = max(1, min(n_batches, self.tokens.size // (batch_size * self.seq_len)))
        span = self.tokens.size - self.seq_len - 1
        ix = np.linspace(0, span, num=n * batch_size, dtype=np.int64)[: n * batch_size]
        rng.shuffle(ix)
        out = []
        for b in range(n):
            sel = ix[b * batch_size : (b + 1) * batch_size]
            x = np.stack([self.tokens[i : i + self.seq_len] for i in sel]).astype(np.int64)
            y = np.stack([self.tokens[i + 1 : i + 1 + self.seq_len] for i in sel]).astype(np.int64)
            out.append((x, y))
        return out


def estimate_loss(cfg, params, sampler: BatchSampler, batch_size: int, n_batches: int, seed: int = 0):
    """Mean cross-entropy over validation windows (no gradients)."""
    from . import model as M
    from . import ops as O

    rng = np.random.default_rng(seed)
    total, count = 0.0, 0
    for x, y in sampler.eval_batches(batch_size, n_batches, rng):
        logits, _ = M.forward(cfg, params, x, need_grad=False)
        loss, _ = O.cross_entropy_fwd(logits, y)
        total += loss
        count += 1
    return total / count


# ---------------------------------------------------------------------------
# SFT: prompt + completion pairs with loss masking
# ---------------------------------------------------------------------------
class SFTDataset:
    """JSONL of {"prompt": ..., "completion": ...} -- loss only on completion."""

    def __init__(self, path: str, tokenizer: BPETokenizer, seq_len: int) -> None:
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prompt = np.asarray(tokenizer.encode(rec["prompt"]), dtype=np.int64)
                completion = np.asarray(tokenizer.encode(rec["completion"]), dtype=np.int64)
                if prompt.size + completion.size > seq_len:
                    completion = completion[: seq_len - prompt.size]
                if completion.size == 0:
                    continue  # nothing to learn from
                self.items.append((prompt, completion))
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.items)

    def batch(self, batch_size: int, rng: np.random.Generator):
        """A random batch: (x, y, mask) where mask marks completion positions.

        With ``seq = prompt + completion`` we set ``x = seq[:-1]``,
        ``y = seq[1:]`` and mask y-positions whose *target* is a completion
        token (i.e. ``j >= len(prompt) - 1``); padding is always masked out.
        """
        idx = rng.integers(0, len(self.items), size=batch_size)
        seq = self.seq_len
        x = np.zeros((batch_size, seq), dtype=np.int64)
        y = np.zeros((batch_size, seq), dtype=np.int64)
        mask = np.zeros((batch_size, seq), dtype=bool)
        for b, i in enumerate(idx):
            prompt, completion = self.items[int(i)]
            s = np.concatenate([prompt, completion])
            n = s.size
            x[b, : n - 1] = s[:-1]
            y[b, : n - 1] = s[1:]
            mask[b, max(prompt.size - 1, 0) : n - 1] = True
        return x, y, mask


def masked_loss_and_grad(cfg, params, x, y, mask):
    """Cross-entropy averaged over unmasked positions only, with gradients."""
    from . import model as M
    from . import ops as O

    logits, ctx = M.forward(cfg, params, x, need_grad=True)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_y = y.reshape(-1)
    flat_mask = mask.reshape(-1)
    sel = np.nonzero(flat_mask)[0]
    sel_logits = flat_logits[sel]
    sel_y = flat_y[sel]
    if sel.size == 0:
        raise ValueError("batch has no unmasked positions")
    probs = O.softmax_fwd(sel_logits)
    n = sel.size
    picked = probs[np.arange(n), sel_y]
    loss = -np.log(np.maximum(picked, 1e-30)).mean()
    dsel = probs.copy()
    dsel[np.arange(n), sel_y] -= 1.0
    dsel /= n
    dlogits = np.zeros_like(flat_logits)
    dlogits[sel] = dsel
    grads = M.backward(cfg, params, ctx, dlogits.reshape(logits.shape))
    return float(loss), grads
