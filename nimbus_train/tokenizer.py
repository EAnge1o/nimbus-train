"""Byte-level BPE tokenizer, trained and saved in the GPT-2 file format.

Produces ``vocab.json`` + ``merges.txt`` pairs that load in nimbus-infer
(and in HuggingFace ``tokenizers``), with an optional special token such as
``<|endoftext|>`` appended after the learned merges.

The trainer is the incremental algorithm (pair counts + an inverted index +
a lazy max-heap), so a multi-megabyte corpus trains a multi-thousand-merge
vocabulary in minutes -- a naive recount-per-merge implementation would take
hours. Encoding memoizes per pre-token, which makes bulk corpus encoding fast.
"""
from __future__ import annotations

import heapq
import json
import os
from collections import Counter

try:  # pragma: no cover - depends on optional package
    import regex as _re

    _PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
except ImportError:  # pragma: no cover - default path
    import re as _re

    _PATTERN = (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+"
        r"| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
    )

_COMPILED = _re.compile(_PATTERN)

ENDOFTEXT = "<|endoftext|>"


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 reversible byte <-> printable-unicode map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _merge_word(word: list, pair: tuple) -> list:
    """Replace every non-overlapping occurrence of ``pair`` in ``word``."""
    first, second = pair
    out = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == first and word[i + 1] == second:
            out.append(first + second)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer compatible with the GPT-2 file format."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: dict[tuple[str, str], int] | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.encoder: dict[str, int] = dict(vocab or {})
        self.decoder: dict[int, str] = {i: t for t, i in self.encoder.items()}
        self.merges: dict[tuple[str, str], int] = dict(merges or {})
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self._cache: dict[str, list[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def __len__(self) -> int:
        return self.vocab_size

    # -- core BPE --------------------------------------------------------
    def _bpe_word(self, byte_str: str) -> list[str]:
        """Apply ranked merges to one pre-token's byte-level string."""
        if byte_str in self._cache:
            return self._cache[byte_str]
        word = list(byte_str)
        if len(word) > 1:
            while True:
                best, best_rank = None, None
                for pair in zip(word[:-1], word[1:]):
                    rank = self.merges.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best, best_rank = pair, rank
                if best is None:
                    break
                word = _merge_word(word, best)
        self._cache[byte_str] = word
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        # special tokens first, longest match first, then BPE between them
        if self.special_tokens:
            pattern = "(" + "|".join(_re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            segments = _re.split(pattern, text)
        else:
            segments = [text]
        for seg in segments:
            if not seg:
                continue
            if seg in self.special_tokens:
                ids.append(self.special_tokens[seg])
                continue
            for chunk in _COMPILED.findall(seg):
                piece = "".join(_BYTE_ENCODER[b] for b in chunk.encode("utf-8"))
                for tok in self._bpe_word(piece):
                    ids.append(self.encoder[tok])
        return ids

    def decode(self, ids) -> str:
        pieces = []
        for i in ids:
            tok = self.decoder.get(int(i))
            if tok is None:
                continue
            pieces.append(tok)
        data = bytes(_BYTE_DECODER.get(ch, ord("?")) for ch in "".join(pieces))
        return data.decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        tok = self.decoder.get(int(token_id))
        if tok is None:
            return b""
        return bytes(_BYTE_DECODER.get(ch, ord("?")) for ch in tok)

    # -- training --------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 512, special_tokens: tuple[str, ...] = (ENDOFTEXT,), verbose: bool = False) -> "BPETokenizer":
        """Train a byte-level BPE vocabulary (incremental pair counting)."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (one id per byte)")
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size too small for base vocab + special tokens")

        word_counts: Counter = Counter()
        for chunk in _COMPILED.findall(text):
            seq = tuple(_BYTE_ENCODER[b] for b in chunk.encode("utf-8"))
            if seq:
                word_counts[seq] += 1

        # pair counts + inverted index pair -> set of word ids containing it
        pair_counts: Counter = Counter()
        pair_to_words: dict[tuple, set] = {}
        words = list(word_counts.items())  # [(tuple_token, count)]
        for wid, (seq, count) in enumerate(words):
            for pair in zip(seq[:-1], seq[1:]):
                pair_counts[pair] += count
                pair_to_words.setdefault(pair, set()).add(wid)

        # lazy max-heap keyed on pair counts (negated for max)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[str, str]] = []
        for step in range(n_merges):
            while heap:
                neg, pair = heapq.heappop(heap)
                if pair_counts.get(pair) == -neg and -neg > 0:
                    break
            else:
                break  # heap exhausted: no pairs left
            merges.append(pair)
            affected = pair_to_words.pop(pair, set())
            for wid in affected:
                seq, count = words[wid]
                # subtract the old pair counts of this word
                for p in zip(seq[:-1], seq[1:]):
                    if pair_counts.get(p):
                        pair_counts[p] -= count
                        if pair_counts[p] <= 0:
                            del pair_counts[p]
                        else:
                            # keep the heap in sync on decreases too: the exact-
                            # match check on pop needs an entry for every count
                            heapq.heappush(heap, (-pair_counts[p], p))
                new_seq = tuple(_merge_word(list(seq), pair))
                words[wid] = (new_seq, count)
                # add the new pair counts
                for p in zip(new_seq[:-1], new_seq[1:]):
                    pair_counts[p] += count
                    pair_to_words.setdefault(p, set()).add(wid)
                    heapq.heappush(heap, (-pair_counts[p], p))
            pair_counts.pop(pair, None)
            if verbose and (step + 1) % 500 == 0:
                # ascii() keeps this console-safe: byte-level tokens are
                # unicode chars that some Windows codepages cannot print
                print(f"  merge {step + 1}/{n_merges}: {ascii(pair)}")

        tok = cls()
        tok._install_base_vocab()
        for rank, (a, b) in enumerate(merges):
            tok.merges[(a, b)] = rank
            token = a + b
            tok.encoder[token] = 256 + rank
            tok.decoder[256 + rank] = token
        for st in special_tokens:
            idx = tok.vocab_size
            tok.special_tokens[st] = idx
            tok.encoder[st] = idx
            tok.decoder[idx] = st
        return tok

    def _install_base_vocab(self) -> None:
        for b in range(256):
            ch = _BYTE_ENCODER[b]
            self.encoder[ch] = b
            self.decoder[b] = ch

    # -- persistence (GPT-2 format: vocab.json + merges.txt) --------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as fh:
            json.dump(self.encoder, fh, ensure_ascii=False)
        with open(os.path.join(directory, "merges.txt"), "w", encoding="utf-8") as fh:
            fh.write("#version: 0.2\n")
            ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
            for (a, b), _ in ordered:
                fh.write(f"{a} {b}\n")

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        with open(os.path.join(directory, "vocab.json"), encoding="utf-8") as fh:
            encoder = json.load(fh)
        merges: dict[tuple[str, str], int] = {}
        with open(os.path.join(directory, "merges.txt"), encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges[(parts[0], parts[1])] = len(merges)
        # special tokens = vocab entries that no merge can produce
        produced = {a + b for (a, b) in merges} | {_BYTE_ENCODER[b] for b in range(256)}
        special = {t: i for t, i in encoder.items() if t not in produced}
        return cls(vocab=encoder, merges=merges, special_tokens=special)
