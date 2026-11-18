"""Command-line interface: every stage of the pipeline as a subcommand.

    python -m nimbus_train train-tokenizer --text corpus.txt --vocab-size 8192 --out data/tokenizer
    python -m nimbus_train prepare-data    --text corpus.txt --tokenizer data/tokenizer --out data
    python -m nimbus_train pretrain        --data data --out runs/base --layers 6 --steps 8000 ...
    python -m nimbus_train sft             --run runs/base --data sft.jsonl --mode lora ...
    python -m nimbus_train dpo             --run runs/base --data pairs.jsonl --mode lora ...
    python -m nimbus_train export          --run runs/base --out checkpoints/base
    python -m nimbus_train generate        --checkpoint checkpoints/base --prompt "Once upon a time"
    python -m nimbus_train eval            --run runs/base
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=None, help="grouped-query attention")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-mlp", type=int, default=None, help="default: 4x d-model")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=None, help="default: --seq (learned pos) / 2048 (rope)")
    ap.add_argument("--family", choices=("gpt2", "llama"), default="gpt2",
                    help="gpt2: layernorm+gelu+learned-pos; llama: rmsnorm+swiglu+rope(+gqa)")


def _build_config(args, vocab_size: int):
    from .config import ModelConfig

    d_mlp = args.d_mlp or 4 * args.d_model
    max_seq = args.max_seq or (args.seq if args.family == "gpt2" else 2048)
    return ModelConfig(
        name=args.name or "nimbus-model",
        vocab_size=vocab_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        d_model=args.d_model,
        d_mlp=d_mlp,
        max_seq_len=max_seq,
        norm="rmsnorm" if args.family == "llama" else "layernorm",
        activation="swiglu" if args.family == "llama" else "gelu",
        pos_encoding="rope" if args.family == "llama" else "learned",
        tie_embeddings=not args.family == "llama",
        norm_eps=1e-6 if args.family == "llama" else 1e-5,
    )


# ---------------------------------------------------------------------------
def cmd_train_tokenizer(ap: argparse.Namespace) -> None:
    from .data import read_text_files
    from .tokenizer import BPETokenizer

    text = read_text_files(ap.text)
    tok = BPETokenizer.train(text, vocab_size=ap.vocab_size,
                             special_tokens=("<|endoftext|>",) if not ap.no_eot else ())
    tok.save(ap.out)
    print(f"tokenizer: vocab={len(tok):,} -> {ap.out}")


def cmd_prepare_data(ap: argparse.Namespace) -> None:
    from .data import encode_corpus, read_text_files, save_tokens
    from .tokenizer import BPETokenizer

    tok = BPETokenizer.load(ap.tokenizer)
    text = read_text_files(ap.text)
    train_ids, val_ids = encode_corpus(tok, text, val_frac=ap.val_frac)
    save_tokens(train_ids, os.path.join(ap.out, "train.bin"))
    save_tokens(val_ids, os.path.join(ap.out, "val.bin"))
    meta = {"train_tokens": int(train_ids.size), "val_tokens": int(val_ids.size),
            "vocab_size": len(tok), "val_frac": ap.val_frac}
    with open(os.path.join(ap.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"tokens: train {train_ids.size:,}  val {val_ids.size:,} -> {ap.out}")


def cmd_pretrain(ap: argparse.Namespace) -> None:
    import shutil

    from .config import ModelConfig
    from .data import load_tokens
    from .trainer import TrainConfig, pretrain

    with open(os.path.join(ap.data, "meta.json"), encoding="utf-8") as fh:
        vocab_size = json.load(fh)["vocab_size"]
    config = _build_config(ap, vocab_size)
    tcfg = TrainConfig(
        steps=ap.steps, batch_size=ap.batch, seq_len=ap.seq, lr=ap.lr, warmup=ap.warmup,
        min_lr_ratio=ap.min_lr_ratio, weight_decay=ap.wd, clip=ap.clip,
        eval_every=ap.eval_every, eval_batches=ap.eval_batches, log_every=ap.log_every,
        ckpt_every=ap.ckpt_every, seed=ap.seed,
    )
    train = load_tokens(os.path.join(ap.data, "train.bin"))
    val = load_tokens(os.path.join(ap.data, "val.bin"))
    # make the run dir self-contained: sft/dpo/export expect <run>/tokenizer.
    # the tokenizer lives either inside the data dir or next to it.
    tok_src = None
    for cand in (os.path.join(ap.data, "tokenizer"),
                 os.path.join(os.path.dirname(os.path.abspath(ap.data)), "tokenizer")):
        if os.path.isdir(cand):
            tok_src = cand
            break
    if tok_src is not None:
        tok_dst = os.path.join(ap.out, "tokenizer")
        if not os.path.isdir(tok_dst):
            shutil.copytree(tok_src, tok_dst)
    pretrain(config, tcfg, train, val, ap.out, resume=not ap.fresh)


def cmd_sft(ap: argparse.Namespace) -> None:
    from .sft import sft

    from .trainer import load_run

    config, _, params, _, _ = load_run(ap.run)
    sft(
        config, params, ap.data, os.path.join(ap.run, "tokenizer"), ap.out,
        seq_len=ap.seq, steps=ap.steps, batch_size=ap.batch, lr=ap.lr,
        mode=ap.mode, lora_r=ap.lora_r, lora_alpha=ap.lora_alpha, lora_path=ap.lora,
        seed=ap.seed, log_every=ap.log_every, ckpt_every=ap.ckpt_every,
    )


def cmd_dpo(ap: argparse.Namespace) -> None:
    from .dpo import dpo
    from .trainer import load_run

    config, _, params, _, _ = load_run(ap.run)
    with np.load(os.path.join(ap.run, "params.npz")) as data:
        ref_params = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    dpo(
        config, params, ref_params, ap.data, os.path.join(ap.run, "tokenizer"), ap.out,
        seq_len=ap.seq, steps=ap.steps, batch_size=ap.batch, lr=ap.lr, beta=ap.beta,
        mode=ap.mode, lora_r=ap.lora_r, lora_alpha=ap.lora_alpha, lora_path=ap.lora,
        seed=ap.seed, log_every=ap.log_every, ckpt_every=ap.ckpt_every,
    )


def cmd_export(ap: argparse.Namespace) -> None:
    from .export import export_checkpoint, verify_export

    params = export_checkpoint(ap.run, ap.out, tokenizer_dir=ap.tokenizer, merge=not ap.no_merge)
    diff = verify_export(ap.run, ap.out)
    n = sum(int(v.size) for v in params.values())
    print(f"exported {n:,} params -> {ap.out}")
    print(f"verified: max logit diff {diff:.2e}")


def cmd_generate(ap: argparse.Namespace) -> None:
    from .config import ModelConfig
    from .export import _read_safetensors
    from .sampler import generate
    from .tokenizer import BPETokenizer

    with open(os.path.join(ap.checkpoint, "config.json"), encoding="utf-8") as fh:
        config = ModelConfig.from_json(fh.read())
    tok = BPETokenizer.load(os.path.join(ap.checkpoint, "tokenizer"))
    npz = os.path.join(ap.checkpoint, "weights.npz")
    if os.path.exists(npz):
        with np.load(npz) as data:
            params = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    else:
        params = _read_safetensors(os.path.join(ap.checkpoint, "weights.safetensors"))
    text = generate(
        config, params, tok, ap.prompt,
        max_new_tokens=ap.max_new_tokens, temperature=ap.temperature,
        top_k=ap.top_k, top_p=ap.top_p, seed=ap.seed,
        stop_token=tok.special_tokens.get("<|endoftext|>"),
    )
    print(text)


def cmd_eval(ap: argparse.Namespace) -> None:
    from .data import load_tokens, estimate_loss, BatchSampler
    from .trainer import load_run

    config, train_cfg, params, _, step = load_run(ap.run)
    data_dir = ap.data or os.path.dirname(os.path.abspath(ap.run))
    val = load_tokens(os.path.join(data_dir, "val.bin"))
    sampler = BatchSampler(val, train_cfg.seq_len)
    loss = estimate_loss(config, params, sampler, train_cfg.batch_size, ap.batches)
    print(f"step {step}: val loss {loss:.4f} (perplexity {np.exp(loss):.2f})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="nimbus-train",
        description="Train transformers from scratch in pure NumPy (hand-written backprop).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train-tokenizer", help="train a byte-level BPE tokenizer")
    p.add_argument("--text", nargs="+", required=True)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--no-eot", action="store_true", help="omit the <|endoftext|> token")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_train_tokenizer)

    p = sub.add_parser("prepare-data", help="encode a corpus into token bins")
    p.add_argument("--text", nargs="+", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prepare_data)

    p = sub.add_parser("pretrain", help="pretrain a model on token bins")
    _add_model_args(p)
    p.add_argument("--data", required=True, help="directory with train.bin/val.bin")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    p.set_defaults(fn=cmd_pretrain)

    p = sub.add_parser("sft", help="supervised fine-tune (full or LoRA)")
    p.add_argument("--run", required=True, help="base run directory")
    p.add_argument("--data", required=True, help="JSONL: prompt/completion")
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mode", choices=("lora", "full"), default="lora")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora", default=None, help="resume from an adapter file")
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_sft)

    p = sub.add_parser("dpo", help="direct preference optimization (full or LoRA)")
    p.add_argument("--run", required=True, help="base run directory (also the ref policy)")
    p.add_argument("--data", required=True, help="JSONL: prompt/chosen/rejected")
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--mode", choices=("lora", "full"), default="lora")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora", default=None)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_dpo)

    p = sub.add_parser("export", help="write a nimbus-infer checkpoint")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--no-merge", action="store_true", help="keep LoRA adapters separate")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("generate", help="sample from an exported checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("eval", help="validation loss / perplexity of a run")
    p.add_argument("--run", required=True)
    p.add_argument("--data", default=None, help="directory with val.bin (default: next to run)")
    p.add_argument("--batches", type=int, default=32)
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
