import json
import os

import numpy as np
import pytest

from nimbus_train.data import (
    BatchSampler,
    SFTDataset,
    encode_corpus,
    estimate_loss,
    masked_loss_and_grad,
    save_tokens,
)
from nimbus_train.config import ModelConfig
from nimbus_train.model import init_params
from nimbus_train.tokenizer import BPETokenizer, ENDOFTEXT


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    corpus = ("Once upon a time there was a cat. The cat sat on the mat.\n\n"
              "A dog ran in the park. The dog was happy.\n\n") * 20
    t = BPETokenizer.train(corpus, vocab_size=280, special_tokens=(ENDOFTEXT,))
    d = tmp_path_factory.mktemp("tok")
    t.save(str(d))
    return t, str(d)


def test_encode_corpus_splits_and_holds_out(tok):
    t, _ = tok
    corpus = ("Once upon a time there was a cat. The cat sat on the mat." + ENDOFTEXT) * 50
    train, val = encode_corpus(t, corpus, val_frac=0.1)
    assert train.dtype == np.uint16
    assert 0 < val.size < train.size
    assert train.size + val.size > 400


def test_batch_sampler_shapes_and_targets(tok):
    t, _ = tok
    ids = t.encode("Once upon a time there was a cat. " * 50)
    ids = np.asarray(ids, dtype=np.uint16)
    sampler = BatchSampler(ids, seq_len=16)
    rng = np.random.default_rng(0)
    x, y = sampler.batch(4, rng)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    # y is x shifted by one within each row
    for b in range(4):
        assert np.array_equal(y[b, :-1], x[b, 1:])


def test_save_and_load_tokens_roundtrip(tmp_path):
    ids = np.arange(1000, dtype=np.uint16) % 300
    path = str(tmp_path / "tokens.bin")
    save_tokens(ids, path)
    loaded = np.memmap(path, dtype=np.uint16, mode="r")
    assert np.array_equal(np.asarray(loaded), ids)


def test_sft_dataset_masking(tmp_path, tok):
    t, tok_dir = tok
    records = [
        {"prompt": "The cat", "completion": " sat on the mat."},
        {"prompt": "A dog", "completion": " ran in the park."},
    ]
    path = tmp_path / "sft.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    ds = SFTDataset(str(path), t, seq_len=32)
    assert len(ds) == 2
    rng = np.random.default_rng(0)
    x, y, mask = ds.batch(2, rng)
    # every masked (loss-counted) target must be a completion token
    p0 = t.encode("The cat")
    c0 = t.encode(" sat on the mat.")
    for b in range(2):
        masked_pos = np.nonzero(mask[b])[0]
        assert masked_pos.size > 0
        # first masked position predicts the first completion token
        assert y[b, masked_pos[0]] == c0[0]


def test_masked_loss_and_grad_matches_manual():
    cfg = ModelConfig(name="m", vocab_size=64, n_layers=2, n_heads=2, d_model=32, d_mlp=64, max_seq_len=16)
    params = init_params(cfg, seed=0, dtype=np.float64)  # fp64: clean finite differences
    rng = np.random.default_rng(1)
    x = rng.integers(0, 64, size=(2, 8))
    y = rng.integers(0, 64, size=(2, 8))
    mask = np.ones((2, 8), dtype=bool)
    mask[:, :4] = False
    loss, grads = masked_loss_and_grad(cfg, params, x, y, mask)

    # finite-difference spot check (fp32 params are fine for a coarse check)
    eps = 1e-6
    idx = (3, 5)  # a 2-D parameter: q weights are (d_model, n_heads*head_dim)
    key = "h.0.attn.q.weight"
    orig = params[key][idx]
    params[key][idx] = orig + eps
    lp, _ = masked_loss_and_grad(cfg, params, x, y, mask)
    params[key][idx] = orig - eps
    lm, _ = masked_loss_and_grad(cfg, params, x, y, mask)
    params[key][idx] = orig
    num = (lp - lm) / (2 * eps)
    diff = abs(num - grads[key][idx])
    assert diff <= 1e-8 or diff / max(abs(num), abs(grads[key][idx])) <= 1e-3


def test_estimate_loss_deterministic():
    cfg = ModelConfig(name="m", vocab_size=64, n_layers=2, n_heads=2, d_model=32, d_mlp=64, max_seq_len=16)
    params = init_params(cfg, seed=0)
    ids = np.random.default_rng(0).integers(0, 64, size=5000).astype(np.uint16)
    sampler = BatchSampler(ids, seq_len=16)
    l1 = estimate_loss(cfg, params, sampler, batch_size=4, n_batches=3, seed=0)
    l2 = estimate_loss(cfg, params, sampler, batch_size=4, n_batches=3, seed=0)
    assert l1 == l2
