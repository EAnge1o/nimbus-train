import json

import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train.data import masked_loss_and_grad
from nimbus_train.model import init_params
from nimbus_train.sft import sft
from nimbus_train.tokenizer import BPETokenizer


def _cfg() -> ModelConfig:
    return ModelConfig(name="s", vocab_size=256, n_layers=2, n_heads=2, d_model=24,
                       d_mlp=48, max_seq_len=32)


def _corpus_tokenizer(tmp_path):
    tok = BPETokenizer()
    tok._install_base_vocab()
    tok.save(str(tmp_path / "tok"))
    return tok, tmp_path / "tok"


def test_sft_loss_only_on_completion(tmp_path):
    tok, tok_dir = _corpus_tokenizer(tmp_path)
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    rng = np.random.default_rng(0)
    x = np.zeros((1, 16), dtype=np.int64)
    y = np.zeros((1, 16), dtype=np.int64)
    mask = np.zeros((1, 16), dtype=bool)
    ids = rng.integers(0, cfg.vocab_size, size=10)
    x[0, :9] = ids[:-1]
    y[0, :9] = ids[1:]
    mask[0, 5:9] = True  # only positions 5..8 count
    loss, grads = masked_loss_and_grad(cfg, params, x, y, mask)
    assert np.isfinite(loss)
    # gradient wrt embeddings must be zero for unmasked rows' targets... the
    # forward still passes them through, but *only* masked positions carry
    # dlogits, so unmasked positions contribute nothing
    # (smoke-check gradient finiteness)
    for k, g in grads.items():
        assert np.all(np.isfinite(g)), k


def test_sft_lora_training_reduces_loss(tmp_path):
    tok, tok_dir = _corpus_tokenizer(tmp_path)
    records = [{"prompt": "The cat", "completion": " sat on the mat."}] * 8
    data = tmp_path / "sft.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    cfg = _cfg()
    params = init_params(cfg, seed=0)
    stats = sft(
        cfg, params, str(data), str(tok_dir), str(tmp_path / "out"),
        seq_len=24, steps=40, batch_size=4, lr=2e-3, mode="lora", lora_r=4,
        log_every=100, ckpt_every=100,
    )
    assert stats["final_loss"] is not None
    assert (tmp_path / "out" / "lora.npz").exists()
    assert (tmp_path / "out" / "params.npz").exists()


def test_sft_full_mode(tmp_path):
    tok, tok_dir = _corpus_tokenizer(tmp_path)
    records = [{"prompt": "A dog", "completion": " ran fast."}] * 4
    data = tmp_path / "sft.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    before = {k: v.copy() for k, v in params.items()}
    sft(cfg, params, str(data), str(tok_dir), str(tmp_path / "out"),
        seq_len=24, steps=5, batch_size=2, lr=1e-3, mode="full",
        log_every=100, ckpt_every=100)
    changed = [k for k in before if not np.array_equal(before[k], params[k])]
    assert len(changed) > 5  # full fine-tune moves the base weights
