import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train.model import init_params, forward, loss_and_grad
from nimbus_train.export import merge_lora
from nimbus_train.lora import add_lora, load_lora, n_trainable, save_lora, trainable_keys


def _cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=128, n_layers=2, n_heads=4, d_model=32, d_mlp=64, max_seq_len=32)
    base.update(kw)
    return ModelConfig(name="l", **base)


def _tokens(cfg):
    return np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 8))


def test_attach_is_noop():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    tokens = _tokens(cfg)
    lg0, _ = forward(cfg, params, tokens, need_grad=False)
    add_lora(params, cfg, r=4, alpha=8)
    lg1, _ = forward(cfg, params, tokens, need_grad=False)
    assert np.array_equal(lg0, lg1)


def test_trainable_param_count_is_tiny():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    base = sum(int(v.size) for v in params.values())
    add_lora(params, cfg, r=4)
    trainable = n_trainable(params)
    total = sum(int(v.size) for v in params.values())
    assert trainable == 4 * 2 * (32 * 4 + 4 * 32)  # 4 sites x 2 layers x (A + B)
    assert trainable / total < 0.15
    assert trainable_keys(params) == {k for k in params if ".lora_" in k}


def test_lora_gradients_match_full_gradients():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    add_lora(params, cfg, r=4, alpha=8)
    tokens = _tokens(cfg)
    targets = np.random.default_rng(1).integers(0, cfg.vocab_size, size=(2, 8))
    _, grads_full = loss_and_grad(cfg, params, tokens, targets)
    train_on = trainable_keys(params)
    _, grads_lora = loss_and_grad(cfg, params, tokens, targets, train_only=train_on)
    assert set(grads_lora.keys()) == train_on
    for k in train_on:
        assert np.allclose(grads_full[k], grads_lora[k], rtol=1e-5, atol=1e-8), k


def test_lora_training_reduces_loss():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    add_lora(params, cfg, r=8, alpha=16)
    train_on = trainable_keys(params)
    from nimbus_train.optimizer import AdamW, clip_gradients

    rng = np.random.default_rng(2)
    tokens = rng.integers(0, cfg.vocab_size, size=(4, 16))
    targets = rng.integers(0, cfg.vocab_size, size=(4, 16))
    loss0, _ = loss_and_grad(cfg, params, tokens, targets, train_on)
    opt = AdamW(params, lr=0.05, weight_decay=0.0)
    for _ in range(60):
        loss, grads = loss_and_grad(cfg, params, tokens, targets, train_on)
        grads = {k: v for k, v in grads.items() if k in train_on}
        clip_gradients(grads, 1.0)
        opt.step(params, grads)
    loss1, _ = loss_and_grad(cfg, params, tokens, targets, train_on)
    assert loss1 < loss0 - 0.3  # adapters are actually learning


def test_merge_lora_preserves_logits():
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    add_lora(params, cfg, r=4, alpha=8)
    rng = np.random.default_rng(5)
    for k in trainable_keys(params):  # simulate trained adapters
        params[k] += (0.05 * rng.standard_normal(params[k].shape)).astype(np.float32)
    tokens = _tokens(cfg)
    lg_adapter, _ = forward(cfg, params, tokens, need_grad=False)
    merged = merge_lora(params)
    lg_merged, _ = forward(cfg, merged, tokens, need_grad=False)
    assert np.abs(lg_adapter - lg_merged).max() < 1e-5
    assert not any(".lora_" in k for k in merged)


def test_save_load_adapter(tmp_path):
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    meta = add_lora(params, cfg, r=4, alpha=8, seed=3)
    path = str(tmp_path / "lora.npz")
    save_lora(path, params, meta)
    fresh = init_params(cfg, seed=0)
    loaded = load_lora(path, fresh)
    assert loaded["r"] == 4 and loaded["alpha"] == 8
    for k in trainable_keys(params):
        assert np.array_equal(params[k], fresh[k])


def test_swiglu_targets():
    cfg = _cfg(norm="rmsnorm", activation="swiglu", pos_encoding="rope", tie_embeddings=False)
    params = init_params(cfg, seed=0)
    add_lora(params, cfg, r=4, targets=("q", "k", "v", "o", "gate", "up", "down"))
    tokens = _tokens(cfg)
    lg0, _ = forward(cfg, params, tokens, need_grad=False)
    lg1, _ = forward(cfg, params, tokens, need_grad=False)
    assert np.array_equal(lg0, lg1)


def test_gelu_target_rejected_for_swiglu():
    cfg = _cfg(norm="rmsnorm", activation="swiglu", pos_encoding="rope", tie_embeddings=False)
    params = init_params(cfg, seed=0)
    with pytest.raises(ValueError):
        add_lora(params, cfg, r=4, targets=("fc",))
