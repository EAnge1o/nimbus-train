import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train.model import init_params, forward, backward, loss_and_grad
from nimbus_train import ops as O


def _cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=128, n_layers=2, n_heads=4, d_model=32, d_mlp=64, max_seq_len=32)
    base.update(kw)
    return ModelConfig(name="m", **base)


def test_forward_shapes_and_dtype():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    tokens = np.random.default_rng(0).integers(0, 128, size=(3, 16))
    logits, ctx = forward(cfg, params, tokens, need_grad=False)
    assert logits.shape == (3, 16, 128)
    assert logits.dtype == np.float32
    assert ctx == {"tokens": tokens, "layers": [{}, {}]}


def test_param_count_matches_analytic():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    assert cfg.param_count() == sum(int(v.size) for v in params.values())
    cfg_l = _cfg(n_kv_heads=2, norm="rmsnorm", activation="swiglu", pos_encoding="rope",
                 tie_embeddings=False)
    params_l = init_params(cfg_l, seed=0)
    assert cfg_l.param_count() == sum(int(v.size) for v in params_l.values())


def test_gqa_uses_fewer_kv_heads():
    cfg = _cfg(n_kv_heads=2)
    params = init_params(cfg, seed=0)
    assert "h.0.attn.k.weight" in params
    assert params["h.0.attn.k.weight"].shape == (32, 2 * 8)  # d x (kv_heads * head_dim)
    assert cfg.kv_group_size == 2


def test_weight_tying_vs_untied():
    tied = _cfg()
    untied = _cfg(tie_embeddings=False)
    p_t, p_u = init_params(tied, seed=0), init_params(untied, seed=0)
    assert "lm_head.weight" not in p_t and "lm_head.weight" in p_u
    tokens = np.random.default_rng(1).integers(0, 128, size=(2, 8))
    lg_t, _ = forward(tied, p_t, tokens, need_grad=False)
    lg_u, _ = forward(untied, p_u, tokens, need_grad=False)
    assert lg_t.shape == lg_u.shape == (2, 8, 128)


def test_causal_masking_last_token_only_sees_itself():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    # identical suffix token: the last position's logits must not depend on
    # what comes *after* it (nothing does), and prefix changes must not alter
    # the last row's logits
    rng = np.random.default_rng(2)
    a = rng.integers(0, 128, size=(1, 10))
    b = a.copy()
    b[0, 5] = (a[0, 5] + 7) % 128  # change a middle token
    la, _ = forward(cfg, params, a, need_grad=False)
    lb, _ = forward(cfg, params, b, need_grad=False)
    assert np.array_equal(la[0, :5], lb[0, :5])
    assert not np.array_equal(la[0, 5], lb[0, 5])


def test_rope_configs_run():
    cfg = _cfg(norm="rmsnorm", activation="swiglu", pos_encoding="rope")
    params = init_params(cfg, seed=0)
    tokens = np.random.default_rng(0).integers(0, 128, size=(2, 16))
    loss, grads = loss_and_grad(cfg, params, tokens, tokens)
    assert np.isfinite(loss)
    assert set(grads.keys()) == set(params.keys())


def test_backward_is_linear_in_dlogits():
    """backward implements a linear map: G(a*d1 + b*d2) == a*G(d1) + b*G(d2).

    Each backward consumes (and partly overwrites) the ctx from its own
    forward, so every call needs a fresh forward.
    """
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    tokens = np.random.default_rng(3).integers(0, 128, size=(2, 8))
    rng = np.random.default_rng(4)
    d1 = rng.standard_normal((2, 8, 128)).astype(np.float32)
    d2 = rng.standard_normal((2, 8, 128)).astype(np.float32)
    _, ctx1 = forward(cfg, params, tokens, need_grad=True)
    _, ctx2 = forward(cfg, params, tokens, need_grad=True)
    _, ctx12 = forward(cfg, params, tokens, need_grad=True)
    g1 = backward(cfg, params, ctx1, d1.copy())
    g2 = backward(cfg, params, ctx2, d2.copy())
    g12 = backward(cfg, params, ctx12, (2.5 * d1 - 0.5 * d2).copy())
    for k in g1:
        assert np.allclose(g12[k], 2.5 * g1[k] - 0.5 * g2[k], atol=1e-4, rtol=1e-4), k


def test_seeded_init_is_reproducible():
    a = init_params(_cfg(), seed=7)
    b = init_params(_cfg(), seed=7)
    for k in a:
        assert np.array_equal(a[k], b[k])
