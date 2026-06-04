import argparse
import json
import random
import re
import ast
import numpy as np
from pathlib import Path

from tokenizer import Tokenizer

_BYTES_RE = re.compile(r"""b(['"])(?:\\.|(?!\1).)*\1""")

def parse_bytes_literal(s: str) -> bytes:
    value = ast.literal_eval(s.strip())
    if not isinstance(value, bytes):
        raise TypeError(f"Expected bytes literal, got: {s}")
    return value


def load_vocab(path: str) -> dict[int, bytes]:
    """
    支持这种格式：

    0    b'\\x00'
    1    b'\\x01'
    256  b' t'
    """
    vocab: dict[int, bytes] = {}

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        match = _BYTES_RE.search(line)
        if match is None:
            continue

        idx_part = line[:match.start()].strip()
        idx = int(idx_part)

        token = parse_bytes_literal(match.group(0))
        vocab[idx] = token

    return vocab

def load_merges(path: str) -> list[tuple[bytes, bytes]]:
    """
    支持这种格式：

    0    b' '    b't'
    1    b' '    b'a'
    2    b'h'    b'e'
    5    b' t'   b'he'
    """
    merges: list[tuple[bytes, bytes]] = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        matches = list(_BYTES_RE.finditer(line))
        if len(matches) < 2:
            continue

        left = parse_bytes_literal(matches[0].group(0))
        right = parse_bytes_literal(matches[1].group(0))

        merges.append((left, right))

    return merges

def load_documents(path: str) -> list[str]:
    """
    支持两种数据格式：

    1. 用 <|endoftext|> 分隔的大文本
    2. 每行一个 document 的 jsonl / txt
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    if "<|endoftext|>" in text:
        docs = text.split("<|endoftext|>")
        docs = [doc.strip() for doc in docs if doc.strip()]
        return docs

    docs = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # 尝试 jsonl
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                if "text" in obj:
                    docs.append(obj["text"])
                elif "content" in obj:
                    docs.append(obj["content"])
                else:
                    docs.append(line)
            else:
                docs.append(str(obj))
        except json.JSONDecodeError:
            docs.append(line)

    return docs

import time

def measure_throughput(tokenizer, docs):
    total_bytes = 0
    total_tokens = 0

    start = time.perf_counter()

    for doc in docs:
        ids = tokenizer.encode(doc)
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(ids)

    elapsed = time.perf_counter() - start
    throughput = total_bytes / elapsed

    return total_bytes, total_tokens, elapsed, throughput

def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> tuple[int, int, float]:
    total_bytes = 0
    total_tokens = 0

    for doc in docs:
        ids = tokenizer.encode(doc)

        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(ids)

    ratio = total_bytes / total_tokens

    return total_bytes, total_tokens, ratio

from pathlib import Path
import numpy as np


def save_token_ids(
    tokenizer,
    docs: list[str],
    output_path: str,
    dtype=np.uint16,
) -> None:
    """
    Encode docs into token IDs and save as a NumPy .npy array.

    Matches this usage style:
        ids = tokenizer.encode(doc)
    """

    output_path = Path(output_path)

    max_token_id = max(tokenizer.vocab.keys())
    if dtype == np.uint16 and max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"max token id {max_token_id} is too large for uint16"
        )

    # First pass: count tokens
    total_tokens = 0
    for doc in docs:
        total_tokens += len(tokenizer.encode(doc))

    # Create output .npy array without holding everything in memory
    arr = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(total_tokens,),
    )

    # Second pass: write token IDs
    offset = 0
    for doc in docs:
        ids = tokenizer.encode(doc)
        n = len(ids)
        arr[offset : offset + n] = ids
        offset += n

    arr.flush()

    print(f"Saved {total_tokens:,} tokens to {output_path}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tinystories-data", required=True)
    parser.add_argument("--tinystories-vocab", required=True)
    parser.add_argument("--tinystories-merges", required=True)

    parser.add_argument("--owt-data", required=True)
    parser.add_argument("--owt-vocab", required=True)
    parser.add_argument("--owt-merges", required=True)

    parser.add_argument("--num-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    special_tokens = ["<|endoftext|>"]

    tiny_vocab = load_vocab(args.tinystories_vocab)
    tiny_merges = load_merges(args.tinystories_merges)
    tiny_tokenizer = Tokenizer(tiny_vocab, tiny_merges, special_tokens=special_tokens)

    owt_vocab = load_vocab(args.owt_vocab)
    owt_merges = load_merges(args.owt_merges)
    owt_tokenizer = Tokenizer(owt_vocab, owt_merges, special_tokens=special_tokens)

    tiny_docs = load_documents(args.tinystories_data)
    owt_docs = load_documents(args.owt_data)

    # tiny_sample = random.sample(tiny_docs, args.num_docs)
    # owt_sample = random.sample(owt_docs, args.num_docs)

    # tiny_bytes, tiny_tokens, tiny_ratio = compression_ratio(tiny_tokenizer, tiny_sample)
    # owt_bytes, owt_tokens, owt_ratio = compression_ratio(owt_tokenizer, owt_sample)

    # _, _, _, throughput = measure_throughput(owt_tokenizer, owt_sample)

    save_token_ids(tiny_tokenizer, tiny_docs, "/home/maliang/llm_learn/assignment1-basics/outputs/train.npy")
    save_token_ids(owt_tokenizer, owt_docs, "/home/maliang/llm_learn/assignment1-basics/outputs_owt/train.npy")

    # print("TinyStories tokenizer:")
    # print(f"  documents:    {args.num_docs}")
    # print(f"  total bytes:  {tiny_bytes}")
    # print(f"  total tokens: {tiny_tokens}")
    # print(f"  bytes/token:  {tiny_ratio:.4f}")
    # print()

    # print("OpenWebText tokenizer:")
    # print(f"  documents:    {args.num_docs}")
    # print(f"  total bytes:  {owt_bytes}")
    # print(f"  total tokens: {owt_tokens}")
    # print(f"  bytes/token:  {owt_ratio:.4f}")
    # print()

    # print("Deliverable:")
    # print(
    #     f"For 10 sampled TinyStories documents, my TinyStories 10K tokenizer achieved "
    #     f"{tiny_ratio:.4f} bytes/token. "
    #     f"For 10 sampled OpenWebText documents, my OpenWebText 32K tokenizer achieved "
    #     f"{owt_ratio:.4f} bytes/token."
    # )

    # pile_bytes = 825 * 1024**3
    # pile_seconds = pile_bytes / throughput
    # pile_hours = pile_seconds / 3600

    # print(f"throughput: {throughput:.2f} bytes/s")
    # print(f"Pile tokenization time: {pile_hours:.2f} hours")

if __name__ == "__main__":
    main()
