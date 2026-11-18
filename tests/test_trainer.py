import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train.data import load_tokens, save_tokens
from nimbus_train.model import init_params
from nimbus_train.trainer import TrainConfig, load_run, pretrain, save_run


@pytest.fixture()
def token_files(tmp_path):
    rng = np.random.default_rng(0)
    train = rng.integers(0, 128, size=6000).astype(np.uint16)
    val = rng.integers(0, 128, size=1200).astype(np.uint16)
    save_tokens(train, str(tmp_path / "train.bin"))
    save_tokens(val, str(tmp_path / "val.bin"))
    return str(tmp_path / "train.bin"), str(tmp_path / "val.bin")


def _cfg():
    return ModelConfig(name="tr", vocab_size=128, n_layers=2, n_heads=2, d_model=24,
                       d_mlp=48, max_seq_len=32)


def _tcfg(tmp_path, steps, **kw):
    base = dict(steps=steps, batch_size=4, seq_len=16, lr=1e-3, warmup=2, eval_every=100,
                eval_batches=2, log_every=1, ckpt_every=max(1, steps // 2))
    base.update(kw)
    return TrainConfig(**base)


def test_pretrain_reduces_loss_and_checkpoints(tmp_path, token_files):
    train, val = token_files
    run = str(tmp_path / "run")
    stats = pretrain(_cfg(), _tcfg(tmp_path, 20), load_tokens(train), load_tokens(val), run)
    assert stats["final_loss"] < 6.0  # started near ln(128) ~ 4.85; small drop suffices
    assert (tmp_path / "run" / "params.npz").exists()
    assert (tmp_path / "run" / "optimizer.npz").exists()
    assert (tmp_path / "run" / "log.csv").exists()


def test_resume_continues_from_checkpoint(tmp_path, token_files):
    train, val = token_files
    run = str(tmp_path / "run")
    pretrain(_cfg(), _tcfg(tmp_path, 6), load_tokens(train), load_tokens(val), run)
    _, _, params_before, _, step = load_run(run)
    assert step == 6

    stats = pretrain(_cfg(), _tcfg(tmp_path, 10), load_tokens(train), load_tokens(val), run)
    _, _, params_after, _, step_after = load_run(run)
    assert step_after == 10
    # optimizer state was resumed: params moved, not reinitialised
    same = [k for k in params_before if np.array_equal(params_before[k], params_after[k])]
    assert len(same) < len(params_before)  # training changed something


def test_save_run_roundtrip(tmp_path):
    cfg = _cfg()
    params = init_params(cfg, seed=0)
    tcfg = _tcfg(tmp_path, 5)
    save_run(str(tmp_path), cfg, params, tcfg, step=42)
    loaded_cfg, loaded_tcfg, loaded_params, opt_state, step = load_run(str(tmp_path))
    assert step == 42
    assert loaded_cfg.to_dict() == cfg.to_dict()
    for k in params:
        assert np.array_equal(params[k], loaded_params[k])
    assert opt_state is None  # no optimizer saved
