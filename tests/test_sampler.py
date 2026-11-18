import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train.model import init_params
from nimbus_train.sampler import generate
from nimbus_train.tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def byte_tok():
    tok = BPETokenizer()
    tok._install_base_vocab()
    return tok


@pytest.fixture(scope="module")
def tiny_model():
    cfg = ModelConfig(name="s", vocab_size=256, n_layers=2, n_heads=2, d_model=32,
                      d_mlp=64, max_seq_len=64)
    return cfg, init_params(cfg, seed=0)


def test_greedy_is_deterministic(byte_tok, tiny_model):
    cfg, params = tiny_model
    a = generate(cfg, params, byte_tok, "abc", max_new_tokens=12, temperature=0.0, seed=0)
    b = generate(cfg, params, byte_tok, "abc", max_new_tokens=12, temperature=0.0, seed=99)
    assert a == b


def test_seeded_sampling_is_reproducible(byte_tok, tiny_model):
    cfg, params = tiny_model
    a = generate(cfg, params, byte_tok, "abc", max_new_tokens=12, temperature=0.9, seed=7)
    b = generate(cfg, params, byte_tok, "abc", max_new_tokens=12, temperature=0.9, seed=7)
    assert a == b


def test_top_k_top_p_run_and_differ_by_seed(byte_tok, tiny_model):
    cfg, params = tiny_model
    a = generate(cfg, params, byte_tok, "abc", max_new_tokens=16, temperature=0.8, top_k=5, seed=1)
    b = generate(cfg, params, byte_tok, "abc", max_new_tokens=16, temperature=0.8, top_k=5, seed=2)
    c = generate(cfg, params, byte_tok, "abc", max_new_tokens=16, temperature=0.8, top_p=0.9, seed=3)
    assert all(len(s) > 0 for s in (a, b, c))


def test_prompt_is_in_output(byte_tok, tiny_model):
    cfg, params = tiny_model
    out = generate(cfg, params, byte_tok, "hello", max_new_tokens=8, temperature=0.0)
    assert out.startswith("hello")


def test_empty_prompt_rejected(byte_tok, tiny_model):
    cfg, params = tiny_model
    with pytest.raises(ValueError):
        generate(cfg, params, byte_tok, "", max_new_tokens=4)
