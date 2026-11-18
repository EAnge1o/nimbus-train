"""End-to-end pipeline through the CLI: tokenizer -> data -> pretrain ->
sft -> export -> generate, all on tiny configs so it finishes in seconds."""
import json

import numpy as np
import pytest

from nimbus_train.cli import main


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("pipeline")
    corpus = ws / "corpus.txt"
    corpus.write_text(
        ("Once upon a time there was a cat. The cat sat on the mat." + "<|endoftext|>") * 60,
        encoding="utf-8",
    )
    return ws


def _run(*args):
    main(list(args))


def test_full_pipeline(workspace, capsys):
    ws = str(workspace)
    _run("train-tokenizer", "--text", f"{ws}/corpus.txt", "--vocab-size", "300",
         "--out", f"{ws}/tokenizer")
    assert (workspace / "tokenizer" / "vocab.json").exists()

    _run("prepare-data", "--text", f"{ws}/corpus.txt", "--tokenizer", f"{ws}/tokenizer",
         "--val-frac", "0.1", "--out", f"{ws}/data")
    meta = json.loads((workspace / "data" / "meta.json").read_text())
    assert meta["train_tokens"] > 500

    _run("pretrain", "--data", f"{ws}/data", "--out", f"{ws}/run",
         "--layers", "2", "--heads", "2", "--d-model", "32", "--d-mlp", "64",
         "--seq", "32", "--steps", "6", "--batch", "4", "--lr", "1e-3",
         "--warmup", "2", "--eval-every", "100", "--log-every", "2",
         "--ckpt-every", "5")
    assert (workspace / "run" / "params.npz").exists()
    assert (workspace / "run" / "config.json").exists()
    assert (workspace / "run" / "log.csv").exists()

    # resume to a larger step count must not restart from zero
    log_before = (workspace / "run" / "log.csv").read_text().strip().splitlines()
    _run("pretrain", "--data", f"{ws}/data", "--out", f"{ws}/run",
         "--layers", "2", "--heads", "2", "--d-model", "32", "--d-mlp", "64",
         "--seq", "32", "--steps", "8", "--batch", "4", "--lr", "1e-3",
         "--warmup", "2", "--eval-every", "100", "--log-every", "2",
         "--ckpt-every", "100")
    log_after = (workspace / "run" / "log.csv").read_text().strip().splitlines()
    assert len(log_after) > len(log_before)
    first_new_step = int(log_after[len(log_before)].split(",")[0])
    assert first_new_step == 8  # resumed at 6, log_every=2 -> next log is 8

    # sft with prompt/completion pairs
    sft_data = workspace / "sft.jsonl"
    sft_data.write_text(json.dumps({"prompt": "The cat", "completion": " sat on the mat."}),
                        encoding="utf-8")
    _run("sft", "--run", f"{ws}/run", "--data", str(sft_data), "--out", f"{ws}/sft",
         "--seq", "32", "--steps", "3", "--batch", "2", "--mode", "lora",
         "--lora-r", "4")
    assert (workspace / "sft" / "lora.npz").exists()

    # export + generate
    _run("export", "--run", f"{ws}/run", "--out", f"{ws}/ckpt")
    assert (workspace / "ckpt" / "weights.safetensors").exists()
    _run("generate", "--checkpoint", f"{ws}/ckpt", "--prompt", "Once upon a time",
         "--max-new-tokens", "12", "--temperature", "0.0")
    out = capsys.readouterr().out
    assert out.strip()

    _run("eval", "--run", f"{ws}/run", "--data", f"{ws}/data", "--batches", "2")
    assert "val loss" in capsys.readouterr().out


def test_dpo_via_cli(workspace):
    ws = str(workspace)
    # self-contained: train a tiny base first (the module-scoped pipeline test
    # may not have run)
    if not (workspace / "data" / "meta.json").exists():
        _run("train-tokenizer", "--text", f"{ws}/corpus.txt", "--vocab-size", "300",
             "--out", f"{ws}/tokenizer")
        _run("prepare-data", "--text", f"{ws}/corpus.txt", "--tokenizer", f"{ws}/tokenizer",
             "--val-frac", "0.1", "--out", f"{ws}/data")
    if not (workspace / "run" / "params.npz").exists():
        _run("pretrain", "--data", f"{ws}/data", "--out", f"{ws}/run",
             "--layers", "2", "--heads", "2", "--d-model", "32", "--d-mlp", "64",
             "--seq", "32", "--steps", "4", "--batch", "4", "--lr", "1e-3",
             "--warmup", "2", "--eval-every", "100", "--log-every", "2",
             "--ckpt-every", "100")
    pairs = workspace / "pairs.jsonl"
    pairs.write_text(json.dumps({"prompt": "The cat", "chosen": " sat", "rejected": " flew"}),
                     encoding="utf-8")
    _run("dpo", "--run", f"{ws}/run", "--data", str(pairs), "--out", f"{ws}/dpo",
         "--seq", "32", "--steps", "3", "--batch", "1", "--beta", "0.5",
         "--mode", "lora", "--lora-r", "4")
    assert (workspace / "dpo" / "params.npz").exists()
