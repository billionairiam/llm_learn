# tokenizer.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Iterator

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = list(merges)

        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        self.special_tokens = special_tokens or []

        next_id = max(self.vocab.keys(), default=-1) + 1
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.token_to_id:
                self.vocab[next_id] = tok_bytes
                self.token_to_id[tok_bytes] = next_id
                next_id += 1

        self.special_token_to_id = {
            tok: self.token_to_id[tok.encode("utf-8")]
            for tok in self.special_tokens
        }

        self._cache: dict[bytes, list[int]] = {}

        if self.special_tokens:
            escaped = [re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for piece in self._split_special(text):
            if piece in self.special_token_to_id:
                ids.append(self.special_token_to_id[piece])
                continue

            for match in re.finditer(PAT, piece):
                pretoken = match.group(0)
                pretoken_bytes = pretoken.encode("utf-8")
                ids.extend(self._encode_pretoken(pretoken_bytes))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def _split_special(self, text: str) -> list[str]:
        if self.special_pattern is None:
            return [text]

        parts = self.special_pattern.split(text)
        return [p for p in parts if p != ""]

    def _encode_pretoken(self, pretoken_bytes: bytes) -> list[int]:
        if pretoken_bytes in self._cache:
            return self._cache[pretoken_bytes]

        parts: tuple[bytes, ...] = tuple(bytes([b]) for b in pretoken_bytes)
        parts = self._apply_bpe(parts)

        ids = [self.token_to_id[p] for p in parts]
        self._cache[pretoken_bytes] = ids
        return ids

    def _apply_bpe(self, parts: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(parts) <= 1:
            return parts

        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank = float("inf")

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_parts: list[bytes] = []
            i = 0
            while i < len(parts):
                if (
                    i < len(parts) - 1
                    and parts[i] == best_pair[0]
                    and parts[i + 1] == best_pair[1]
                ):
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1

            parts = tuple(new_parts)

        return parts

    @staticmethod
    def _load_vocab(path: str) -> dict[int, bytes]:
        path_obj = Path(path)

        try:
            with open(path_obj, "rb") as f:
                obj = pickle.load(f)
            return Tokenizer._normalize_vocab(obj)
        except Exception:
            pass

        with open(path_obj, "r", encoding="utf-8") as f:
            obj = json.load(f)

        return Tokenizer._normalize_vocab(obj)

    @staticmethod
    def _load_merges(path: str) -> list[tuple[bytes, bytes]]:
        path_obj = Path(path)

        try:
            with open(path_obj, "rb") as f:
                obj = pickle.load(f)
            return Tokenizer._normalize_merges(obj)
        except Exception:
            pass

        if path_obj.suffix == ".json":
            with open(path_obj, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return Tokenizer._normalize_merges(obj)

        # GPT-2 style merges.txt
        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        merges: list[tuple[bytes, bytes]] = []

        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue

                left, right = line.split()

                left_bytes = bytes(byte_decoder[c] for c in left)
                right_bytes = bytes(byte_decoder[c] for c in right)

                merges.append((left_bytes, right_bytes))

        return merges

    @staticmethod
    def _normalize_vocab(obj) -> dict[int, bytes]:
        # Case 1: {id: bytes}
        if all(isinstance(v, (bytes, bytearray, list, str)) for v in obj.values()):
            result: dict[int, bytes] = {}
            for k, v in obj.items():
                idx = int(k)

                if isinstance(v, bytes):
                    result[idx] = v
                elif isinstance(v, bytearray):
                    result[idx] = bytes(v)
                elif isinstance(v, list):
                    result[idx] = bytes(v)
                elif isinstance(v, str):
                    result[idx] = v.encode("latin-1")
                else:
                    raise TypeError(f"Unsupported vocab value: {type(v)}")

            return result

        # Case 2: GPT-2 style {token_string: id}
        if all(isinstance(v, int) for v in obj.values()):
            byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
            result = {}

            for token_str, idx in obj.items():
                token_bytes = bytes(byte_decoder[c] for c in token_str)
                result[idx] = token_bytes

            return result

        raise TypeError("Unsupported vocab format")

    @staticmethod
    def _normalize_merges(obj) -> list[tuple[bytes, bytes]]:
        def to_bytes(x) -> bytes:
            if isinstance(x, bytes):
                return x
            if isinstance(x, bytearray):
                return bytes(x)
            if isinstance(x, list):
                return bytes(x)
            if isinstance(x, str):
                return x.encode("latin-1")
            raise TypeError(f"Unsupported merge element: {type(x)}")

        return [(to_bytes(a), to_bytes(b)) for a, b in obj]


def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 byte encoder helper.
    Needed only when loading GPT-2 style vocab.json / merges.txt.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))
