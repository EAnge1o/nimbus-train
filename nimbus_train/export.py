"""Export a trained model as a nimbus-infer checkpoint.

The output directory is exactly what ``nimbus_infer.Engine.from_pretrained``
expects:

    <out>/config.json          -- a nimbus-infer-compatible ModelConfig
    <out>/weights.safetensors  -- hand-written safetensors container (fp32)
    <out>/tokenizer/           -- vocab.json + merges.txt (GPT-2 format)

Training-side extras (``params.npz``, ``optimizer.npz``, ``log.csv``) are not
copied: the checkpoint contains only what inference needs. LoRA adapters can
be merged into the base weights first with ``merge_lora``.
"""
from __future__ import annotations

import json
import os
import shutil
import struct

import numpy as np

from .config import ModelConfig

_SAFETENSORS_DTYPES = {
    np.dtype(np.float64): "F64",
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.int64): "I64",
    np.dtype(np.int32): "I32",
    np.dtype(np.int16): "I16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint64): "U64",
    np.dtype(np.uint32): "U32",
    np.dtype(np.uint16): "U16",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}


def save_safetensors(path: str, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write a safetensors file without any third-party dependency.

    Format: uint64 little-endian header length, a JSON header mapping each
    tensor to {dtype, shape, data_offsets}, then the raw buffers back to back.
    """
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    blobs = []
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        dtype = np.dtype(array.dtype)
        if dtype not in _SAFETENSORS_DTYPES:
            raise ValueError(f"unsupported dtype for safetensors: {dtype}")
        data = array.tobytes()
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob)


def merge_lora(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fold every attached LoRA adapter into its base weight.

    W' = W + B @ A  (layouts: W (in, out), A (in, r), B (r, out)).
    Returns a new dict without the ``.lora_*`` keys.
    """
    out = {}
    for k, v in params.items():
        if ".lora_" in k:
            continue
        out[k] = v
    for k in [k for k in params if k.endswith(".lora_A")]:
        base = k[: -len(".lora_A")]
        a = params[k]  # (in, r)
        b = params[base + ".lora_B"]  # (r, out)
        out[base] = np.asarray(params[base], dtype=np.float32) + (a @ b).astype(np.float32)
    return out


def export_checkpoint(
    run_dir: str,
    out_dir: str,
    tokenizer_dir: str | None = None,
    merge: bool = True,
    metadata: dict[str, str] | None = None,
) -> dict:
    """Write a nimbus-infer checkpoint from a nimbus-train run directory.

    ``merge=True`` folds attached LoRA adapters into the base weights.
    Returns the exported tensor dict for verification.
    """
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        config = ModelConfig.from_json(fh.read())

    with np.load(os.path.join(run_dir, "params.npz")) as data:
        params = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    if merge:
        params = merge_lora(params)

    if tokenizer_dir is None:
        tokenizer_dir = os.path.join(run_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(f"tokenizer directory not found: {tokenizer_dir}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        fh.write(config.to_json())
    save_safetensors(
        os.path.join(out_dir, "weights.safetensors"),
        params,
        metadata=metadata or {"framework": "nimbus-train", "format": "fp32"},
    )
    tok_out = os.path.join(out_dir, "tokenizer")
    if os.path.exists(tok_out):
        shutil.rmtree(tok_out)
    shutil.copytree(tokenizer_dir, tok_out)
    return params


def verify_export(run_dir: str, out_dir: str, atol: float = 1e-4) -> float:
    """Re-read the exported safetensors and compare logits against the run dir.

    Returns the max absolute logit difference over a few probe sequences.
    """
    from . import model as M
    from .ops import DTYPE

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        config = ModelConfig.from_json(fh.read())
    with np.load(os.path.join(run_dir, "params.npz")) as data:
        train_params = {k: np.asarray(data[k], dtype=DTYPE) for k in data.files}
    train_params = merge_lora(train_params)

    # hand-parse the exported file back
    tensors = _read_safetensors(os.path.join(out_dir, "weights.safetensors"))

    rng = np.random.default_rng(0)
    worst = 0.0
    for seq_len in (1, 16):
        tokens = rng.integers(0, config.vocab_size, size=(2, seq_len))
        lg_train, _ = M.forward(config, train_params, tokens, need_grad=False)
        lg_export, _ = M.forward(config, tensors, tokens, need_grad=False)
        worst = max(worst, float(np.abs(lg_train - lg_export).max()))
    if worst > atol:
        raise AssertionError(f"export mismatch: max logit diff {worst:.2e} > {atol:.0e}")
    return worst


def _read_safetensors(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
        buffer = fh.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = {"F32": np.float32, "F64": np.float64, "I64": np.int64,
                 "U8": np.uint8, "I8": np.int8}[info["dtype"]]
        start, end = info["data_offsets"]
        tensors[name] = np.frombuffer(buffer[start:end], dtype=dtype).reshape(info["shape"])
    return tensors
