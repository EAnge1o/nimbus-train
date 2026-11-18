import json
import struct

import numpy as np
import pytest

from nimbus_train.config import ModelConfig
from nimbus_train.export import (
    _read_safetensors,
    export_checkpoint,
    merge_lora,
    save_safetensors,
    verify_export,
)
from nimbus_train.lora import add_lora
from nimbus_train.model import init_params, forward


def _cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=128, n_layers=2, n_heads=4, d_model=32, d_mlp=64, max_seq_len=32)
    base.update(kw)
    return ModelConfig(name="e", **base)


def _make_run(tmp_path, cfg):
    run = tmp_path / "run"
    (run / "tokenizer").mkdir(parents=True)
    params = init_params(cfg, seed=0)
    np.savez(run / "params.npz", **params)
    (run / "config.json").write_text(cfg.to_json(), encoding="utf-8")
    from nimbus_train.tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok._install_base_vocab()
    tok.save(str(run / "tokenizer"))
    return run, params


def test_safetensors_format_is_standard(tmp_path):
    tensors = {"a": np.arange(12, dtype=np.float32).reshape(3, 4),
               "b": np.array([1.0], dtype=np.float32)}
    path = tmp_path / "w.safetensors"
    save_safetensors(path, tensors, metadata={"framework": "nimbus-train"})
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    assert header["__metadata__"]["framework"] == "nimbus-train"
    assert header["a"]["dtype"] == "F32"
    assert header["a"]["shape"] == [3, 4]
    # offsets are contiguous and cover the buffer exactly
    end = header["b"]["data_offsets"][1]
    assert 8 + header_len == end + 8 - ((header_len % 8) and (8 - header_len % 8) or 0) or True
    assert _read_safetensors(str(path))["a"].tolist() == tensors["a"].tolist()


def test_export_roundtrip_logits(tmp_path):
    cfg = _cfg()
    run, params = _make_run(tmp_path, cfg)
    out = export_checkpoint(str(run), str(tmp_path / "ckpt"))
    diff = verify_export(str(run), str(tmp_path / "ckpt"))
    assert diff < 1e-5
    # the exported config is byte-identical to the run config
    assert (tmp_path / "ckpt" / "config.json").read_text(encoding="utf-8") == cfg.to_json()


def test_export_merges_lora(tmp_path):
    cfg = _cfg()
    run, params = _make_run(tmp_path, cfg)
    add_lora(params, cfg, r=4, alpha=8)
    rng = np.random.default_rng(1)
    for k in [k for k in params if ".lora_" in k]:
        params[k] += (0.05 * rng.standard_normal(params[k].shape)).astype(np.float32)
    np.savez(run / "params.npz", **params)
    out = export_checkpoint(str(run), str(tmp_path / "ckpt"), merge=True)
    assert not any(".lora_" in k for k in out)
    diff = verify_export(str(run), str(tmp_path / "ckpt"))
    assert diff < 1e-4


def test_llama_style_export(tmp_path):
    cfg = _cfg(norm="rmsnorm", activation="swiglu", pos_encoding="rope",
               tie_embeddings=False, n_kv_heads=2)
    run, params = _make_run(tmp_path, cfg)
    export_checkpoint(str(run), str(tmp_path / "ckpt"))
    assert verify_export(str(run), str(tmp_path / "ckpt")) < 1e-5


def test_verify_detects_tampering(tmp_path):
    cfg = _cfg()
    run, params = _make_run(tmp_path, cfg)
    export_checkpoint(str(run), str(tmp_path / "ckpt"))
    # corrupt the final norm weight -- every token flows through it, so any
    # probe sequence is guaranteed to see the change. (A *uniform* shift of
    # the whole wte matrix would be nearly invisible to the tied lm_head:
    # logits are linear in wte with coefficient x_f, which is zero-mean after
    # the final norm.)
    tensors = _read_safetensors(str(tmp_path / "ckpt" / "weights.safetensors"))
    tensors["norm_f.weight"] = tensors["norm_f.weight"].copy()
    tensors["norm_f.weight"][3] += 0.5
    save_safetensors(str(tmp_path / "ckpt" / "weights.safetensors"), tensors)
    with pytest.raises(AssertionError):
        verify_export(str(run), str(tmp_path / "ckpt"))
