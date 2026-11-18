"""Finite-difference verification of every hand-written backward.

Runs in float64: float32 forward rounding noise would swamp the finite
differences. The arithmetic is dtype-agnostic, so this checks the same code
path training uses.
"""
import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train import model as M
from nimbus_train import ops as O


def _gpt2(**kw) -> ModelConfig:
    base = dict(vocab_size=64, n_layers=2, n_heads=4, d_model=32, d_mlp=96, max_seq_len=16)
    base.update(kw)
    return ModelConfig(name="gc-gpt2", **base)


def _llama(**kw) -> ModelConfig:
    base = dict(vocab_size=64, n_layers=2, n_heads=4, n_kv_heads=2, d_model=32,
                d_mlp=88, max_seq_len=16, norm="rmsnorm", activation="swiglu",
                pos_encoding="rope", tie_embeddings=False, norm_eps=1e-6)
    base.update(kw)
    return ModelConfig(name="gc-llama", **base)


def check_gradients(cfg, n_per_param=20, eps=1e-6, seed=0, params=None):
    rng = np.random.default_rng(seed)
    if params is None:
        params = M.init_params(cfg, seed=seed, dtype=np.float64)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, 10))
    targets = rng.integers(0, cfg.vocab_size, size=(2, 10))
    loss, grads = M.loss_and_grad(cfg, params, tokens, targets)

    checked, fails = 0, []
    for k in sorted(params.keys()):
        arr = params[k]
        for _ in range(min(n_per_param, arr.size)):
            idx = np.unravel_index(int(rng.integers(0, arr.size)), arr.shape)
            orig = arr[idx]
            arr[idx] = orig + eps
            lp, _ = M.loss_and_grad(cfg, params, tokens, targets)
            arr[idx] = orig - eps
            lm, _ = M.loss_and_grad(cfg, params, tokens, targets)
            arr[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[k][idx]
            diff = abs(num - ana)
            # near-zero gradients judged by absolute difference
            if diff > 1e-8 and diff / max(abs(num), abs(ana)) > 1e-3:
                fails.append((k, idx, float(num), float(ana)))
            checked += 1
    return checked, fails, loss


@pytest.mark.parametrize(
    "cfg",
    [
        _gpt2(),                                   # tied, layernorm, gelu, learned pos
        _gpt2(tie_embeddings=False),               # untied head
        _llama(),                                  # rope, rmsnorm, swiglu, gqa
        _llama(n_kv_heads=4),                      # plain mha rope variant
    ],
    ids=["gpt2", "gpt2-untied", "llama-gqa", "llama-mha"],
)
def test_model_gradients_finite_difference(cfg):
    checked, fails, loss = check_gradients(cfg)
    assert fails == [], f"gradient mismatches: {fails[:4]}"
    assert checked >= 400


def test_cross_entropy_matches_reference():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 7))
    targets = rng.integers(0, 7, size=5)
    loss, probs = O.cross_entropy_fwd(logits.copy(), targets)
    shifted = logits - logits.max(-1, keepdims=True)
    ref = -(shifted - np.log(np.exp(shifted).sum(-1, keepdims=True)))[np.arange(5), targets].mean()
    assert abs(loss - ref) < 1e-6


@pytest.mark.parametrize("name", ["linear", "layer_norm", "rms_norm", "gelu", "silu", "softmax", "rope", "embedding"])
def test_op_gradients_finite_difference(name):
    """Per-op checks with random upstream gradients."""
    rng = np.random.default_rng(hash(name) % 2**32)
    eps = 1e-6

    def fd(f, inputs, idx, upstream):
        pert = []
        orig = inputs[idx].copy()
        for sign in (+1, -1):
            inputs[idx] = orig + sign * eps
            pert.append(float(np.sum(f() * upstream)))
        inputs[idx] = orig
        return (pert[0] - pert[1]) / (2 * eps)

    if name == "linear":
        x = rng.standard_normal((4, 6))
        W = rng.standard_normal((6, 5))
        b = rng.standard_normal(5)
        dy = rng.standard_normal((4, 5))
        y, _ = O.linear_fwd(x, W, b)
        dx, dW, db = O.linear_bwd(x, W, b, dy)
        def f_x(): return O.linear_fwd(x, W, b)[0]
        assert abs(fd(f_x, x, (2, 3), dy) - dx[2, 3]) < 1e-6
        def f_W(): return O.linear_fwd(x, W, b)[0]
        assert abs(fd(f_W, W, (1, 2), dy) - dW[1, 2]) < 1e-6
        def f_b(): return O.linear_fwd(x, W, b)[0]
        assert abs(fd(f_b, b, (3,), dy) - db[3]) < 1e-6
    elif name == "layer_norm":
        x = rng.standard_normal((3, 8))
        w = rng.standard_normal(8)
        b = rng.standard_normal(8)
        dy = rng.standard_normal((3, 8))
        y, aux = O.layer_norm_fwd(x, w, b)
        dx, dW, db = O.layer_norm_bwd(aux[0], aux[1], w, dy)
        for arr, grad, idx in ((x, dx, (1, 2)), (w, dW, (4,)), (b, db, (5,))):
            got = fd(lambda: O.layer_norm_fwd(x, w, b)[0], arr, idx, dy)
            assert abs(got - grad[idx]) < 1e-6, (name, idx, got, grad[idx])
    elif name == "rms_norm":
        x = rng.standard_normal((3, 8))
        w = rng.standard_normal(8)
        dy = rng.standard_normal((3, 8))
        y, aux = O.rms_norm_fwd(x, w)
        dx, dW = O.rms_norm_bwd(aux[0], aux[1], w, dy)
        for arr, grad, idx in ((x, dx, (2, 6)), (w, dW, (3,))):
            got = fd(lambda: O.rms_norm_fwd(x, w)[0], arr, idx, dy)
            assert abs(got - grad[idx]) < 1e-6
    elif name == "gelu":
        x = rng.standard_normal((6, 5))
        dy = rng.standard_normal((6, 5))
        y, t = O.gelu_fwd(x)
        dx = O.gelu_bwd(x, t, dy)
        got = fd(lambda: O.gelu_fwd(x)[0], x, (2, 3), dy)
        assert abs(got - dx[2, 3]) < 1e-6
    elif name == "silu":
        x = rng.standard_normal((6, 5))
        dy = rng.standard_normal((6, 5))
        y, s = O.silu_fwd(x)
        dx = O.silu_bwd(x, s, dy)
        got = fd(lambda: O.silu_fwd(x)[0], x, (4, 1), dy)
        assert abs(got - dx[4, 1]) < 1e-6
    elif name == "softmax":
        x = rng.standard_normal((4, 6))
        dy = rng.standard_normal((4, 6))
        probs = O.softmax_fwd(x.copy())
        dx = O.softmax_bwd(probs.copy(), dy)
        got = fd(lambda: O.softmax_fwd(x.copy()), x, (1, 4), dy)
        assert abs(got - dx[1, 4]) < 1e-6
    elif name == "rope":
        x = rng.standard_normal((2, 3, 8))  # (heads, seq, hd)
        cos, sin = O.build_rope(8, 4)
        cos, sin = cos[:3][None, :, :], sin[:3][None, :, :]
        dy = rng.standard_normal((2, 3, 8))
        y, _ = O.rope_fwd(x, cos, sin)
        dx = O.rope_bwd(dy, cos, sin)
        got = fd(lambda: O.rope_fwd(x, cos, sin)[0], x, (1, 2, 5), dy)
        assert abs(got - dx[1, 2, 5]) < 1e-6
    elif name == "embedding":
        W = rng.standard_normal((10, 4))
        ids = np.array([3, 7, 7, 1])
        dy = rng.standard_normal((4, 4))
        dW = O.embedding_bwd(W.shape, ids, dy)
        got = fd(lambda: O.embedding_fwd(W, ids), W, (7, 2), dy)
        assert abs(got - dW[7, 2]) < 1e-6
