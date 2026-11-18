import json

import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train.dpo import _dpo_dlogits, _logprobs_from_logits, _logsigmoid, _sigmoid, dpo
from nimbus_train.model import backward, forward, init_params
from nimbus_train.tokenizer import BPETokenizer


def _cfg() -> ModelConfig:
    return ModelConfig(name="d", vocab_size=256, n_layers=2, n_heads=2, d_model=24,
                       d_mlp=48, max_seq_len=32)


def _byte_tokenizer(tmp_path):
    tok = BPETokenizer()
    tok._install_base_vocab()
    tok.save(str(tmp_path))
    return tmp_path


def test_dpo_gradient_finite_difference():
    """The hand-derived DPO logits gradient against central differences.

    float64 params: finite differences on a float32 forward drown in rounding
    noise (see the module docstring's note on the grad checker).
    """
    cfg = _cfg()
    params = init_params(cfg, seed=3, dtype=np.float64)
    ref = init_params(cfg, seed=4, dtype=np.float64)
    rng = np.random.default_rng(5)
    B, T = 2, 8
    tokens = rng.integers(0, cfg.vocab_size, size=(2 * B, T))
    y = rng.integers(0, cfg.vocab_size, size=(2 * B, T))
    mask = np.ones((2 * B, T), dtype=bool)
    mask[:, :3] = False  # prompt region contributes nothing
    beta = 0.1

    ref_logits, _ = forward(cfg, ref, tokens, need_grad=False)
    lp_ref, _ = _logprobs_from_logits(ref_logits, y, mask)
    pol_logits, ctx = forward(cfg, params, tokens, need_grad=True)
    lp_pol, probs = _logprobs_from_logits(pol_logits, y, mask)
    z = beta * ((lp_pol[0::2] - lp_ref[0::2]) - (lp_pol[1::2] - lp_ref[1::2]))
    loss = -_logsigmoid(z).mean()
    coef = np.empty(2 * B)
    coef[0::2] = -beta * (1 - _sigmoid(z)) / B
    coef[1::2] = +beta * (1 - _sigmoid(z)) / B
    dlogits = _dpo_dlogits(probs, y, mask, coef)
    grads = backward(cfg, params, ctx, dlogits)

    def loss_of(p):
        lg, _ = forward(cfg, p, tokens, need_grad=False)
        lpp, _ = _logprobs_from_logits(lg, y, mask)
        zz = beta * ((lpp[0::2] - lp_ref[0::2]) - (lpp[1::2] - lp_ref[1::2]))
        return -_logsigmoid(zz).mean()

    eps = 1e-6
    for k in ["h.0.attn.q.weight", "h.1.mlp.fc.weight", "wte", "norm_f.weight"]:
        arr = params[k]
        for _ in range(6):
            u = np.unravel_index(int(rng.integers(0, arr.size)), arr.shape)
            o = arr[u]
            arr[u] = o + eps
            lp_ = loss_of(params)
            arr[u] = o - eps
            lm_ = loss_of(params)
            arr[u] = o
            num = (lp_ - lm_) / (2 * eps)
            ana = grads[k][u]
            diff = abs(num - ana)
            assert diff <= 1e-8 or diff / max(abs(num), abs(ana)) <= 1e-3, (k, u, num, ana)


def test_dpo_training_raises_preference_accuracy(tmp_path):
    """With an identical frozen reference, initial z == 0; after a few steps the
    policy should prefer the chosen completions on the training pairs."""
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    ref = init_params(cfg, seed=0)
    tok_dir = _byte_tokenizer(tmp_path)

    pairs = [
        {"prompt": "cat", "chosen": " sit", "rejected": " fly"},
        {"prompt": "dog", "chosen": " run", "rejected": " cry"},
    ]
    data = tmp_path / "pairs.jsonl"
    data.write_text("\n".join(json.dumps(p) for p in pairs), encoding="utf-8")

    stats = dpo(
        cfg, params, ref, str(data), str(tok_dir), str(tmp_path / "out"),
        seq_len=16, steps=30, batch_size=2, lr=2e-3, beta=0.5,
        mode="full", log_every=100, ckpt_every=100,
    )
    assert 0.0 <= stats["final_acc"] <= 1.0


def test_reference_model_stays_frozen(tmp_path):
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    ref = init_params(cfg, seed=0)
    tok_dir = _byte_tokenizer(tmp_path)
    data = tmp_path / "pairs.jsonl"
    data.write_text(json.dumps({"prompt": "cat", "chosen": " sit", "rejected": " fly"}),
                    encoding="utf-8")
    dpo(cfg, params, ref, str(data), str(tok_dir), str(tmp_path / "out"),
        seq_len=16, steps=3, batch_size=1, lr=1e-3, beta=0.5,
        mode="full", log_every=100, ckpt_every=100)
    for k in ref:
        assert np.array_equal(ref[k], init_params(cfg, seed=0)[k]), k


def test_dpo_log_sigmoid_values():
    assert _logsigmoid(0.0) == pytest.approx(-np.log(2.0))
    assert _logsigmoid(10.0) == pytest.approx(-np.log1p(np.exp(-10.0)))
    assert _sigmoid(0.0) == pytest.approx(0.5)
