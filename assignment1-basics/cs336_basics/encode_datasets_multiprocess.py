#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import numpy as np

from tokenizer import Tokenizer


_BYTES_RE = re.compile(r"""b(['\"])(?:\\.|(?!\1).)*\1""")

_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_DTYPE: np.dtype | None = None


def parse_bytes_literal(s: str) -> bytes:
    value = ast.literal_eval(s.strip())
    if not isinstance(value, bytes):
        raise TypeError(f"Expected bytes literal, got: {s!r}")
    return value


def load_vocab(path: str) -> dict[int, bytes]:
    """
    Supports lines like:

        0    b'\\x00'
        1    b'\\x01'
        256  b' t'
    """
    vocab: dict[int, bytes] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            match = _BYTES_RE.search(line)
            if match is None:
                continue

            idx_part = line[: match.start()].strip()
            if not idx_part:
                raise ValueError(f"Missing vocab id at line {line_no}: {line!r}")

            idx = int(idx_part)
            token = parse_bytes_literal(match.group(0))
            vocab[idx] = token

    if not vocab:
        raise ValueError(f"No vocab entries loaded from {path}")

    return vocab


def load_merges(path: str) -> list[tuple[bytes, bytes]]:
    """
    Supports lines like:

        0    b' '    b't'
        1    b' '    b'a'
        2    b'h'    b'e'
        5    b' t'   b'he'
    """
    merges: list[tuple[bytes, bytes]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            matches = list(_BYTES_RE.finditer(line))
            if len(matches) < 2:
                continue

            left = parse_bytes_literal(matches[0].group(0))
            right = parse_bytes_literal(matches[1].group(0))
            merges.append((left, right))

    if not merges:
        raise ValueError(f"No merges loaded from {path}")

    return merges


def detect_format(
    path: str,
    eot_token: str = "<|endoftext|>",
    sample_size: int = 4 * 1024 * 1024,
) -> str:
    """
    Returns one of: "eot", "jsonl", "lines".
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(sample_size)

    if eot_token in sample:
        return "eot"

    for line in sample.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and ("text" in obj or "content" in obj):
                return "jsonl"
        except json.JSONDecodeError:
            pass

        break

    return "lines"


def iter_documents(
    path: str,
    fmt: str = "auto",
    eot_token: str = "<|endoftext|>",
    chunk_size: int = 4 * 1024 * 1024,
    add_eot_after_jsonl_doc: bool = True,
) -> Iterator[str]:
    """
    Stream documents without loading the full train file into memory.

    fmt:
      - "eot":   big text file separated by <|endoftext|>
      - "jsonl": one JSON object per line, usually {"text": "..."}
      - "lines": one plain-text document per line
      - "auto":  detect from a small sample
    """
    if fmt == "auto":
        fmt = detect_format(path, eot_token=eot_token)
        print(f"[info] detected format for {path}: {fmt}")

    if fmt == "eot":
        buffer = ""

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                buffer += chunk
                parts = buffer.split(eot_token)

                # All except the final part are complete docs.
                for doc in parts[:-1]:
                    if doc:
                        yield doc
                    # Keep the separator as a special token for LM training.
                    yield eot_token

                # The final part may be an incomplete doc.
                buffer = parts[-1]

        if buffer:
            yield buffer

        return

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if "text" in obj:
                            yield obj["text"]
                        elif "content" in obj:
                            yield obj["content"]
                        else:
                            yield line
                    else:
                        yield str(obj)
                except json.JSONDecodeError:
                    yield line

                if add_eot_after_jsonl_doc:
                    yield eot_token

        return

    if fmt == "lines":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                yield line
                if add_eot_after_jsonl_doc:
                    yield eot_token

        return

    raise ValueError(f"Unknown fmt: {fmt!r}")


def batched(iterable: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []

    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def maybe_limit(iterable: Iterator[str], limit: int | None) -> Iterator[str]:
    if limit is None:
        yield from iterable
    else:
        yield from islice(iterable, limit)


def _init_worker(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    dtype_name: str,
) -> None:
    global _WORKER_TOKENIZER
    global _WORKER_DTYPE

    _WORKER_TOKENIZER = Tokenizer(vocab, merges, special_tokens=special_tokens)
    _WORKER_DTYPE = np.dtype(dtype_name)


def _encode_batch(docs: list[str]) -> np.ndarray:
    if _WORKER_TOKENIZER is None or _WORKER_DTYPE is None:
        raise RuntimeError("Worker tokenizer was not initialized")

    ids: list[int] = []
    for doc in docs:
        ids.extend(_WORKER_TOKENIZER.encode(doc))

    return np.asarray(ids, dtype=_WORKER_DTYPE)


def raw_bin_to_npy(
    tmp_path: Path,
    output_path: Path,
    n_tokens: int,
    dtype: np.dtype,
    copy_chunk_tokens: int = 10_000_000,
) -> None:
    npy = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(n_tokens,),
    )

    raw = np.memmap(
        tmp_path,
        mode="r",
        dtype=dtype,
        shape=(n_tokens,),
    )

    for start in range(0, n_tokens, copy_chunk_tokens):
        end = min(start + copy_chunk_tokens, n_tokens)
        npy[start:end] = raw[start:end]

    npy.flush()

    del raw
    del npy


def save_token_ids_multiprocess(
    *,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    input_path: str,
    output_path: str,
    fmt: str,
    num_workers: int,
    batch_size: int,
    dtype: np.dtype,
    eot_token: str,
    limit_docs: int | None = None,
) -> None:
    max_token_id = max(vocab.keys())
    if max_token_id > np.iinfo(dtype).max:
        raise ValueError(
            f"max token id {max_token_id} is too large for {dtype}; "
            f"max allowed is {np.iinfo(dtype).max}"
        )

    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(str(output_path_obj) + ".tmp.bin")

    print(f"[info] input:        {input_path_obj}")
    print(f"[info] output:       {output_path_obj}")
    print(f"[info] tmp:          {tmp_path}")
    print(f"[info] workers:      {num_workers}")
    print(f"[info] batch size:   {batch_size}")
    print(f"[info] dtype:        {dtype}")
    print(f"[info] format:       {fmt}")
    if limit_docs is not None:
        print(f"[info] limit docs:   {limit_docs:,}")

    t0 = time.time()
    total_tokens = 0
    total_batches = 0
    total_docs_seen = 0

    docs_iter = iter_documents(
        str(input_path_obj),
        fmt=fmt,
        eot_token=eot_token,
    )
    docs_iter = maybe_limit(docs_iter, limit_docs)
    doc_batches = batched(docs_iter, batch_size=batch_size)

    # Main process writes the file. Workers only encode.
    with open(tmp_path, "wb") as out:
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(vocab, merges, special_tokens, dtype.name),
        ) as pool:
            # imap preserves input order. This matters for LM training data.
            for arr in pool.imap(_encode_batch, doc_batches, chunksize=1):
                arr.tofile(out)

                total_tokens += int(arr.size)
                total_batches += 1
                total_docs_seen += batch_size

                if total_batches % 100 == 0:
                    elapsed = time.time() - t0
                    tok_per_sec = total_tokens / max(elapsed, 1e-9)
                    print(
                        f"[progress] batches={total_batches:,}, "
                        f"approx_docs={total_docs_seen:,}, "
                        f"tokens={total_tokens:,}, "
                        f"tokens/sec={tok_per_sec:,.0f}"
                    )

    print("[info] converting raw .bin to standard .npy ...")

    raw_bin_to_npy(
        tmp_path=tmp_path,
        output_path=output_path_obj,
        n_tokens=total_tokens,
        dtype=dtype,
    )

    tmp_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(
        f"[done] saved {total_tokens:,} tokens to {output_path_obj} "
        f"in {elapsed:.2f}s, tokens/sec={total_tokens / max(elapsed, 1e-9):,.0f}"
    )


def encode_one_dataset(
    *,
    name: str,
    data_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    fmt: str,
    special_tokens: list[str],
    num_workers: int,
    batch_size: int,
    dtype: np.dtype,
    eot_token: str,
    limit_docs: int | None,
) -> None:
    print(f"\n========== Encoding {name} ==========")

    print(f"[info] loading vocab:  {vocab_path}")
    vocab = load_vocab(vocab_path)

    print(f"[info] loading merges: {merges_path}")
    merges = load_merges(merges_path)

    print(f"[info] vocab size:     {len(vocab):,}")
    print(f"[info] merges count:   {len(merges):,}")

    save_token_ids_multiprocess(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        input_path=data_path,
        output_path=output_path,
        fmt=fmt,
        num_workers=num_workers,
        batch_size=batch_size,
        dtype=dtype,
        eot_token=eot_token,
        limit_docs=limit_docs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream + multiprocess encode TinyStories and OpenWebText into uint16 .npy token arrays."
    )

    parser.add_argument("--tinystories-data", required=True)
    parser.add_argument("--tinystories-vocab", required=True)
    parser.add_argument("--tinystories-merges", required=True)
    parser.add_argument(
        "--tinystories-output",
        default="outputs_tinystories/tinystories_train.npy",
        help="Output .npy path for TinyStories token ids.",
    )
    parser.add_argument(
        "--tinystories-format",
        choices=["auto", "eot", "jsonl", "lines"],
        default="auto",
    )

    parser.add_argument("--owt-data", required=True)
    parser.add_argument("--owt-vocab", required=True)
    parser.add_argument("--owt-merges", required=True)
    parser.add_argument(
        "--owt-output",
        default="outputs_owt/owt_train.npy",
        help="Output .npy path for OpenWebText token ids.",
    )
    parser.add_argument(
        "--owt-format",
        choices=["auto", "eot", "jsonl", "lines"],
        default="auto",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes used for tokenizer.encode. Default: 4.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of streamed documents per worker task. Default: 128.",
    )
    parser.add_argument(
        "--dtype",
        choices=["uint16", "uint32"],
        default="uint16",
        help="Datatype for serialized token ids. Default: uint16.",
    )
    parser.add_argument(
        "--eot-token",
        default="<|endoftext|>",
        help="Special end-of-text token. Default: <|endoftext|>.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Debug only: encode at most this many streamed document pieces per dataset.",
    )
    parser.add_argument(
        "--skip-tinystories",
        action="store_true",
        help="Skip TinyStories.",
    )
    parser.add_argument(
        "--skip-owt",
        action="store_true",
        help="Skip OpenWebText.",
    )

    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    dtype = np.dtype(args.dtype)
    special_tokens = [args.eot_token]

    if not args.skip_tinystories:
        encode_one_dataset(
            name="TinyStories",
            data_path=args.tinystories_data,
            vocab_path=args.tinystories_vocab,
            merges_path=args.tinystories_merges,
            output_path=args.tinystories_output,
            fmt=args.tinystories_format,
            special_tokens=special_tokens,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            dtype=dtype,
            eot_token=args.eot_token,
            limit_docs=args.limit_docs,
        )

    if not args.skip_owt:
        encode_one_dataset(
            name="OpenWebText",
            data_path=args.owt_data,
            vocab_path=args.owt_vocab,
            merges_path=args.owt_merges,
            output_path=args.owt_output,
            fmt=args.owt_format,
            special_tokens=special_tokens,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            dtype=dtype,
            eot_token=args.eot_token,
            limit_docs=args.limit_docs,
        )


if __name__ == "__main__":
    main()
