import numpy as np
import pytest

from nimbus_train.optimizer import AdamW, clip_gradients, cosine_lr, global_norm


def test_cosine_schedule_warmup_and_decay():
    lr = cosine_lr(0, 1.0, warmup=100, total_steps=1000)
    assert lr == pytest.approx(1.0 / 100)
    assert cosine_lr(50, 1.0, warmup=100, total_steps=1000) == pytest.approx(0.51)
    assert cosine_lr(100, 1.0, warmup=100, total_steps=1000) == pytest.approx(1.0)
    mid = cosine_lr(550, 1.0, warmup=100, total_steps=1000)
    assert 0.5 < mid < 1.0
    assert cosine_lr(1000, 1.0, warmup=100, total_steps=1000, min_ratio=0.1) == pytest.approx(0.1)


def test_adamw_matches_manual_first_step():
    p = np.array([1.0, -2.0], dtype=np.float32)
    params = {"w": p}
    grads = {"w": np.array([0.5, -1.0], dtype=np.float32)}
    opt = AdamW(params, lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt.step(params, grads)
    # m=0.5g, v=0.05g^2; bc1=0.1, bc2=0.05 -> mhat=g, vhat=g^2 -> step = g/(|g|+eps)
    expected = 0.1 * np.array([0.5, -1.0]) / (np.array([0.5, 1.0]) + 1e-8)
    assert np.allclose(params["w"], np.array([1.0, -2.0]) - expected, atol=1e-7)


def test_adamw_weight_decay_decoupled_and_excluded():
    params = {"w": np.ones(4, dtype=np.float32), "norm.weight": np.ones(4, dtype=np.float32)}
    grads = {"w": np.zeros(4, dtype=np.float32), "norm.weight": np.zeros(4, dtype=np.float32)}
    opt = AdamW(params, lr=0.1, weight_decay=0.2)
    opt.step(params, grads)
    assert params["w"][0] == pytest.approx(1.0 - 0.1 * 0.2 * 1.0)   # decayed
    assert params["norm.weight"][0] == pytest.approx(1.0)           # excluded (1-D norm)


def test_grad_clip():
    grads = {"a": np.full(10, 3.0, dtype=np.float32)}  # norm = sqrt(10*9) ~ 9.49
    pre = clip_gradients(grads, 1.0)
    assert pre == pytest.approx(np.sqrt(90.0), rel=1e-5)
    assert global_norm(grads) == pytest.approx(1.0, rel=1e-5)
    # no-op when under the cap
    grads2 = {"a": np.full(10, 0.01, dtype=np.float32)}
    pre2 = clip_gradients(grads2, 1.0)
    assert global_norm(grads2) == pytest.approx(np.sqrt(10) * 0.01, rel=1e-5)


def test_adamw_converges_on_quadratic():
    """End-to-end sanity: AdamW with a decaying rate minimizes ||w||^2.

    (Plain Adam oscillates around the minimum with amplitude ~lr -- the decay
    is what makes it settle.)
    """
    params = {"w": np.array([10.0, -3.0], dtype=np.float32)}
    opt = AdamW(params, lr=0.5, weight_decay=0.0)
    for t in range(500):
        grads = {"w": 2.0 * params["w"]}
        opt.step(params, grads, lr=0.5 * (0.98 ** t))
    assert np.allclose(params["w"], 0.0, atol=0.05)


def test_optimizer_state_roundtrip():
    params = {"w": np.ones(3, dtype=np.float32)}
    grads = {"w": np.full(3, 0.5, dtype=np.float32)}
    opt = AdamW(params)
    opt.step(params, grads)
    state = opt.state_dict()
    opt2 = AdamW(params)
    opt2.load_state_dict(state)
    assert opt2.step_count == opt.step_count
    assert np.allclose(opt2.m["w"], opt.m["w"])
