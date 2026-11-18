"""Hand-derived forward and backward passes for every primitive.

There is no autograd anywhere in this project. Each op exposes a *forward*
function returning its output plus a small ``aux`` bundle of intermediates,
and a *backward* function turning the upstream gradient into gradients of the
inputs. Every backward is verified against finite differences (tests/
test_gradcheck.py) and, when PyTorch happens to be installed, against real
autograd (tests/test_torch_parity.py).

CPU engineering notes
---------------------
NumPy has two kinds of speed: BLAS-backed matmuls (which use every core -- a
GPU-free laptop still reaches 300+ GFLOPS in float32) and *single-threaded*
elementwise loops, which dominate a NumPy training step (``np.exp`` costs
tens of cycles per element on one core). NumPy ufuncs release the GIL on
large arrays, so the hot elementwise ops below fan out over axis-0 chunks
through a small thread pool, with an automatic serial fallback for small
inputs. The arithmetic per chunk is bit-identical to the serial path, which
is exactly what the gradient tests assert.

A few functions are *destructive* (they overwrite an input). Each one
documents it, and the model code only calls them on arrays it no longer
needs -- e.g. attention scores are consumed by softmax and never kept.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# The trainer works in float32 end to end, matching nimbus-infer's DTYPE.
DTYPE = np.float32

# ---------------------------------------------------------------------------
# small thread pool for elementwise fan-out
# ---------------------------------------------------------------------------
_POOL: ThreadPoolExecutor | None = None
_WORKERS = max(1, min(16, int(os.environ.get("NIMBUS_NUM_WORKERS", 0)) or (os.cpu_count() or 4)))
_WORKERS = min(_WORKERS, 16)
_MIN_PARALLEL = 1_500_000  # elements; smaller inputs run serially


def _get_pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=_WORKERS)
    return _POOL


def set_num_workers(n: int) -> None:
    """Resize the elementwise thread pool (call before training starts)."""
    global _WORKERS, _POOL
    if _POOL is not None:
        _POOL.shutdown()
        _POOL = None
    _WORKERS = max(1, int(n))


def _map_chunks(fn, *arrays):
    """Execute ``fn(*chunks)`` over aligned axis-0 chunks.

    Arrays sharing the first array's leading axis are sliced in lockstep;
    smaller arrays (shared weights/biases) are passed through unchanged.
    Returns the list of results in chunk order (a single-element list on the
    serial path). Array results are concatenated by the caller; partial
    reductions are summed by the caller.
    """
    first = arrays[0]
    n = first.shape[0]
    workers = _WORKERS if first.size >= _MIN_PARALLEL else 1
    k = min(workers, n)
    if k <= 1:
        return [fn(*arrays)]
    bounds = np.linspace(0, n, k + 1).astype(int)
    pool = _get_pool()

    def cut(a, lo, hi):
        return a[lo:hi] if a.ndim and a.shape[0] == n else a

    futures = [
        pool.submit(fn, *[cut(a, int(lo), int(hi)) for a in arrays])
        for lo, hi in zip(bounds[:-1], bounds[1:])
        if lo < hi
    ]
    return [f.result() for f in futures]


def as_float(x: np.ndarray, dtype=DTYPE) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------------------
# linear:  y = x @ W (+ b), weights stored (in_features, out_features)
# ---------------------------------------------------------------------------
def linear_fwd(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None):
    if bias is None:
        return x @ weight, None
    return x @ weight + bias, None


def linear_bwd(x, weight, bias, dy):
    """Gradients of (x, weight, bias). ``dy`` may have any batch dims."""
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    dx = dy @ weight.T
    dweight = x2.T @ dy2
    dbias = dy2.sum(axis=0) if bias is not None else None
    return dx, dweight, dbias


# ---------------------------------------------------------------------------
# layer norm:  y = (x - mean) / sqrt(var + eps) * w + b
# ---------------------------------------------------------------------------
def layer_norm_fwd(x, weight, bias, eps: float = 1e-5):
    xf = x.reshape(-1, x.shape[-1])

    def chunk(xc, wc, bc):
        mean = xc.mean(axis=-1, keepdims=True)
        var = xc.var(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (xc - mean) * inv_std
        return x_hat * wc + bc, x_hat, inv_std

    parts = _map_chunks(chunk, xf, weight, bias)
    y = np.concatenate([p[0] for p in parts], axis=0)
    x_hat = np.concatenate([p[1] for p in parts], axis=0)
    inv_std = np.concatenate([p[2] for p in parts], axis=0)
    return y.reshape(x.shape), (x_hat.reshape(x.shape), inv_std.reshape(x.shape[:-1] + (1,)))


def layer_norm_bwd(x_hat, inv_std, weight, dy):
    xf = x_hat.reshape(-1, x_hat.shape[-1])
    dyf = dy.reshape(-1, dy.shape[-1])
    isf = inv_std.reshape(-1, 1) if inv_std.ndim > 1 else inv_std[:, None]

    def chunk(xhc, isc, dyc):
        dw = (dyc * xhc).sum(axis=0)
        db = dyc.sum(axis=0)
        dyw = dyc * weight
        dx = isc * (
            dyw - dyw.mean(axis=-1, keepdims=True)
            - xhc * (dyw * xhc).mean(axis=-1, keepdims=True)
        )
        return dx, dw, db

    parts = _map_chunks(chunk, xf, isf, dyf)
    dx = np.concatenate([p[0] for p in parts], axis=0)
    dw = sum(p[1] for p in parts)
    db = sum(p[2] for p in parts)
    return dx.reshape(dy.shape), dw, db


# ---------------------------------------------------------------------------
# rms norm (weight only):  y = x / sqrt(mean(x^2) + eps) * w
# ---------------------------------------------------------------------------
def rms_norm_fwd(x, weight, eps: float = 1e-6):
    xf = x.reshape(-1, x.shape[-1])

    def chunk(xc, wc):
        ms = (xc * xc).mean(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(ms + eps)
        x_hat = xc * inv_std
        return x_hat * wc, x_hat, inv_std

    parts = _map_chunks(chunk, xf, weight)
    y = np.concatenate([p[0] for p in parts], axis=0)
    x_hat = np.concatenate([p[1] for p in parts], axis=0)
    inv_std = np.concatenate([p[2] for p in parts], axis=0)
    return y.reshape(x.shape), (x_hat.reshape(x.shape), inv_std.reshape(x.shape[:-1] + (1,)))


def rms_norm_bwd(x_hat, inv_std, weight, dy):
    xf = x_hat.reshape(-1, x_hat.shape[-1])
    dyf = dy.reshape(-1, dy.shape[-1])
    isf = inv_std.reshape(-1, 1) if inv_std.ndim > 1 else inv_std[:, None]

    def chunk(xhc, isc, dyc):
        dw = (dyc * xhc).sum(axis=0)
        dyw = dyc * weight
        dx = isc * (dyw - xhc * (dyw * xhc).mean(axis=-1, keepdims=True))
        return dx, dw

    parts = _map_chunks(chunk, xf, isf, dyf)
    dx = np.concatenate([p[0] for p in parts], axis=0)
    dw = sum(p[1] for p in parts)
    return dx.reshape(dy.shape), dw


# ---------------------------------------------------------------------------
# gelu, tanh approximation (GPT-2 flavour)
# ---------------------------------------------------------------------------
_GELU_C = 0.7978845608028654  # sqrt(2 / pi)


def gelu_fwd(x):
    def chunk(xc):
        inner = _GELU_C * (xc + 0.044715 * xc ** 3)
        t = np.tanh(inner)
        return 0.5 * xc * (1.0 + t), t

    parts = _map_chunks(chunk, x)
    y = np.concatenate([p[0] for p in parts], axis=0)
    t = np.concatenate([p[1] for p in parts], axis=0)
    return y, t


def gelu_bwd(x, t, dy):
    def chunk(xc, tc, dyc):
        dt_dx = _GELU_C * (1.0 + 3 * 0.044715 * xc ** 2)
        return dyc * (0.5 * (1.0 + tc) + 0.5 * xc * (1.0 - tc * tc) * dt_dx)

    parts = _map_chunks(chunk, x, t, dy)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------
def silu_fwd(x):
    def chunk(xc):
        s = 1.0 / (1.0 + np.exp(-xc))
        return xc * s, s

    parts = _map_chunks(chunk, x)
    y = np.concatenate([p[0] for p in parts], axis=0)
    s = np.concatenate([p[1] for p in parts], axis=0)
    return y, s


def silu_bwd(x, s, dy):
    def chunk(xc, sc, dyc):
        return dyc * sc * (1.0 + xc * (1.0 - sc))

    parts = _map_chunks(chunk, x, s, dy)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# softmax over the last axis. DESTRUCTIVE: the input is overwritten in place.
# (attention scores and logits are never needed after softmax -- only the
# probability matrix is kept for the backward pass.)
# ---------------------------------------------------------------------------
def softmax_fwd(x):
    def chunk(xc):
        xc -= xc.max(axis=-1, keepdims=True)
        np.exp(xc, out=xc)
        xc /= xc.sum(axis=-1, keepdims=True)

    _map_chunks(chunk, x)
    return x


def softmax_bwd(probs, dy, out=None):
    """Gradient of the softmax input: ``p * (dy - sum(dy * p))``.

    DESTRUCTIVE by default: the result is written into ``probs``. Pass
    ``out=<array with the same shape>`` to write elsewhere. Callers that still
    need ``probs`` afterwards must compute those uses first (the attention
    backward reads ``probs`` for the value gradient before calling this).
    """
    target = probs if out is None else out

    def chunk(tc, pc, dyc):
        s = (dyc * pc).sum(axis=-1, keepdims=True)
        tc *= dyc - s

    _map_chunks(chunk, target, probs, dy)
    return target


# ---------------------------------------------------------------------------
# cross-entropy fused with softmax, mean over all positions.
# DESTRUCTIVE on ``logits``: the array is overwritten with the probability
# matrix, which is returned for the backward pass.
# ---------------------------------------------------------------------------
def cross_entropy_fwd(logits, targets):
    probs = softmax_fwd(logits)
    flat_p = probs.reshape(-1, probs.shape[-1])
    flat_t = targets.reshape(-1)
    n = flat_p.shape[0]
    picked = flat_p[np.arange(n), flat_t]
    loss = -np.log(np.maximum(picked, 1e-30)).mean()
    return loss, probs


def cross_entropy_bwd(probs, targets):
    """DESTRUCTIVE: writes ``(p - onehot) / n`` into ``probs`` and returns it."""
    flat = probs.reshape(-1, probs.shape[-1])
    flat_t = targets.reshape(-1)
    n = flat.shape[0]
    flat[np.arange(n), flat_t] -= 1.0
    flat /= n
    return probs


# ---------------------------------------------------------------------------
# rotary position embeddings -- a rotation is linear, so the backward pass is
# simply the inverse rotation.
# ---------------------------------------------------------------------------
def build_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    t = np.arange(max_seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)


def rope_fwd(x, cos, sin):
    """Rotate ``x`` (..., seq, head_dim) by precomputed (seq, head_dim/2) tables."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return (
        np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1),
        (x1, x2),
    )


def rope_bwd(dy, cos, sin):
    """Inverse rotation -- the rotation matrix is orthogonal, so its transpose
    is its inverse."""
    half = dy.shape[-1] // 2
    d1, d2 = dy[..., :half], dy[..., half:]
    return np.concatenate([d1 * cos + d2 * sin, -d1 * sin + d2 * cos], axis=-1)


# ---------------------------------------------------------------------------
# embedding lookup:  y[i] = W[ids[i]]
# ---------------------------------------------------------------------------
def embedding_fwd(weight, ids):
    return weight[ids]


def embedding_bwd(weight_shape, ids, dy):
    dw = np.zeros(weight_shape, dtype=dy.dtype)
    flat_ids = ids.reshape(-1)
    flat_dy = dy.reshape(-1, dy.shape[-1])
    np.add.at(dw, flat_ids, flat_dy)
    return dw
