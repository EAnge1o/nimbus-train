import json
import os

import numpy as np
import pytest

from nimbus_train.tokenizer import BPETokenizer, ENDOFTEXT, bytes_to_unicode


CORPUS = (
    "Once upon a time there was a little girl named Lily. She loved the park! "
    "Tom and Sam played with a red ball. ANNA found 123 shells. "
) * 30


def test_train_and_roundtrip_ascii():
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    text = "Once upon a time there was a little girl named Lily."
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert ids[0] != ids[1]  # produced real merges


def test_roundtrip_unicode_and_bytes():
    tok = BPETokenizer.train(CORPUS, vocab_size=400)
    for text in ["中文测试 123", "emoji 🙂 ok", "tabs\tand\nnewlines", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_byte_fallback_needs_no_unk():
    tok = BPETokenizer.train(CORPUS, vocab_size=280)
    text = "Ω≈ç√∫ rare bytes"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_vocab_file_format_is_gpt2_compatible(tmp_path):
    tok = BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=(ENDOFTEXT,))
    tok.save(str(tmp_path))
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    merges = (tmp_path / "merges.txt").read_text(encoding="utf-8").splitlines()
    assert merges[0] == "#version: 0.2"
    assert len(vocab) == 300
    # base bytes occupy ids 0..255 with the GPT-2 unicode mapping
    bm = bytes_to_unicode()
    assert vocab[bm[65]] == 65  # 'A'
    assert vocab[ENDOFTEXT] == 299
    # every merge line joins two existing tokens
    produced = {bm[b] for b in range(256)}
    for line in merges[1:]:
        a, b = line.split(" ")
        assert a in produced and b in produced
        produced.add(a + b)


def test_load_roundtrip(tmp_path):
    tok = BPETokenizer.train(CORPUS, vocab_size=320, special_tokens=(ENDOFTEXT,))
    tok.save(str(tmp_path))
    tok2 = BPETokenizer.load(str(tmp_path))
    text = "Sam played with a red ball.\nSecond line."
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.decode(tok2.encode(text)) == text
    assert tok2.special_tokens == {ENDOFTEXT: 319}


def test_special_token_split_on_encode():
    tok = BPETokenizer.train(CORPUS, vocab_size=300, special_tokens=(ENDOFTEXT,))
    text = f"a{ENDOFTEXT}b"
    ids = tok.encode(text)
    assert tok.special_tokens[ENDOFTEXT] in ids


def test_deterministic_training():
    a = BPETokenizer.train(CORPUS, vocab_size=280)
    b = BPETokenizer.train(CORPUS, vocab_size=280)
    assert a.encoder == b.encoder
    assert a.merges == b.merges


def test_vocab_size_bounds():
    with pytest.raises(ValueError):
        BPETokenizer.train(CORPUS, vocab_size=255)
    # pure byte tokenizer: no merges, no specials
    tiny = BPETokenizer.train(CORPUS, vocab_size=256, special_tokens=())
    assert len(tiny.merges) == 0
    assert len(tiny) == 256
