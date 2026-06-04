from __future__ import annotations

import argparse
import heapq
import os
import pickle
import resource
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKEN = b"<|endoftext|>"


class _ReversePairKey:
    """
    heapq is a min-heap.

    We want:
      1. larger count first
      2. if count ties, lexicographically greater pair first

    So we store -count, and reverse the pair comparison here.
    """

    __slots__ = ("key",)

    def __init__(self, key: tuple[bytes, bytes]):
        self.key = key

    def __lt__(self, other: "_ReversePairKey") -> bool:
        return self.key > other.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReversePairKey) and self.key == other.key


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    Boundaries are moved forward to the next split_special_token.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as bytes"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _count_pretokens_in_chunk(
    args: tuple[str, int, int, bytes],
) -> Counter[tuple[int, ...]]:
    input_path, start, end, split_special_token = args

    counts: Counter[tuple[int, ...]] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    # Special token is a hard boundary and should not be included in merge statistics.
    for doc_bytes in chunk.split(split_special_token):
        if not doc_bytes:
            continue

        text = doc_bytes.decode("utf-8", errors="ignore")

        for match in re.finditer(PAT, text):
            token_bytes = match.group().encode("utf-8")
            seq = tuple(token_bytes)
            if seq:
                counts[seq] += 1

    return counts


def build_seq_counts_multiprocess(
    input_path: str,
    num_processes: int,
    split_special_token: bytes = SPECIAL_TOKEN,
) -> tuple[Counter[tuple[int, ...]], float]:
    start_time = time.perf_counter()

    num_processes = max(1, num_processes)

    # More chunks than processes usually balances work better.
    desired_num_chunks = num_processes * 4

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_special_token,
        )

    tasks = [
        (input_path, start, end, split_special_token)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    total: Counter[tuple[int, ...]] = Counter()

    if num_processes == 1:
        for task in tasks:
            total.update(_count_pretokens_in_chunk(task))
    else:
        with Pool(processes=num_processes) as pool:
            for sub_counts in pool.imap_unordered(_count_pretokens_in_chunk, tasks):
                total.update(sub_counts)

    elapsed = time.perf_counter() - start_time
    return total, elapsed


def _merge_sequence(
    seq: tuple[int, ...],
    pair: tuple[int, int],
    new_token_id: int,
) -> tuple[int, ...]:
    a, b = pair
    out: list[int] = []
    i = 0

    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(new_token_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1

    return tuple(out)


def _heap_entry(
    pair: tuple[int, int],
    count: int,
    vocab: dict[int, bytes],
):
    a, b = pair
    return (-count, _ReversePairKey((vocab[a], vocab[b])), pair)


def train_bpe_tinystories(
    input_path: str,
    vocab_size: int = 10_000,
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], dict[str, float]]:
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    total_start = time.perf_counter()

    # 1. Initial byte vocabulary.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 2. Add TinyStories special token.
    special_token_text = "<|endoftext|>"
    vocab[len(vocab)] = special_token_text.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than initial vocab size {len(vocab)}"
        )

    # 3. Multiprocessing pre-tokenization.
    seq_counts, pretokenize_time = build_seq_counts_multiprocess(
        input_path=input_path,
        num_processes=num_workers,
        split_special_token=SPECIAL_TOKEN,
    )

    # 4. seq_id -> current token sequence / frequency.
    seqid2_seqs: dict[int, tuple[int, ...]] = {}
    seqid2_freq: dict[int, int] = {}

    for seq_id, (seq, freq) in enumerate(seq_counts.items()):
        seqid2_seqs[seq_id] = seq
        seqid2_freq[seq_id] = freq

    del seq_counts

    # 5. Build initial pair_counts and pair_to_seqids.
    build_pairs_start = time.perf_counter()

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_seqids: dict[tuple[int, int], set[int]] = defaultdict(set)

    for seq_id, seq in seqid2_seqs.items():
        freq = seqid2_freq[seq_id]
        for pair in zip(seq, seq[1:]):
            pair_counts[pair] += freq
            pair_to_seqids[pair].add(seq_id)

    heap = [
        _heap_entry(pair, count, vocab)
        for pair, count in pair_counts.items()
        if count > 0
    ]
    heapq.heapify(heap)

    build_pairs_time = time.perf_counter() - build_pairs_start

    # 6. BPE merge loop.
    merge_start = time.perf_counter()

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        best_pair: tuple[int, int] | None = None
        best_count = 0

        # Lazy deletion: ignore stale heap entries.
        while heap:
            neg_count, _, pair = heapq.heappop(heap)
            count = -neg_count

            if pair_counts.get(pair, 0) == count and count > 0:
                best_pair = pair
                best_count = count
                break

        if best_pair is None or best_count <= 0:
            break

        affected_seqs = set(pair_to_seqids.get(best_pair, set()))

        if not affected_seqs:
            pair_counts.pop(best_pair, None)
            continue

        a, b = best_pair
        new_token_id = len(vocab)
        new_token_bytes = vocab[a] + vocab[b]

        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[a], vocab[b]))

        touched_pairs: set[tuple[int, int]] = set()

        for seq_id in affected_seqs:
            old_seq = seqid2_seqs[seq_id]
            freq = seqid2_freq[seq_id]

            # Remove old pair contributions from this seq.
            for old_pair in zip(old_seq, old_seq[1:]):
                new_count = pair_counts.get(old_pair, 0) - freq
                if new_count > 0:
                    pair_counts[old_pair] = new_count
                else:
                    pair_counts.pop(old_pair, None)

                ids = pair_to_seqids.get(old_pair)
                if ids is not None:
                    ids.discard(seq_id)
                    if not ids:
                        pair_to_seqids.pop(old_pair, None)

                touched_pairs.add(old_pair)

            # Apply current merge.
            new_seq = _merge_sequence(old_seq, best_pair, new_token_id)
            seqid2_seqs[seq_id] = new_seq

            # Add new pair contributions from this seq.
            for new_pair in zip(new_seq, new_seq[1:]):
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
                pair_to_seqids[new_pair].add(seq_id)
                touched_pairs.add(new_pair)

        # Push updated pair counts into heap.
        for pair in touched_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, _heap_entry(pair, count, vocab))

    merge_time = time.perf_counter() - merge_start
    total_time = time.perf_counter() - total_start

    timings = {
        "pretokenize_time": pretokenize_time,
        "build_pairs_time": build_pairs_time,
        "merge_time": merge_time,
        "total_time": total_time,
    }

    return vocab, merges, timings


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Linux: KB; macOS: bytes.
    if sys.platform == "darwin":
        return usage / (1024 * 1024)

    return usage / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to TinyStories training txt file")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    start_mem = peak_memory_mb()

    vocab, merges, timings = train_bpe_tinystories(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        num_workers=args.workers,
    )

    end_mem = peak_memory_mb()

    vocab_path = os.path.join(args.out, "tinystories_vocab.pkl")
    merges_path = os.path.join(args.out, "tinystories_merges.pkl")
    vocab_txt_path = os.path.join(args.out, "tinystories_vocab.txt")
    merges_txt_path = os.path.join(args.out, "tinystories_merges.txt")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # Human-readable inspection files.
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for token_id, token_bytes in vocab.items():
            f.write(f"{token_id}\t{token_bytes!r}\n")

    with open(merges_txt_path, "w", encoding="utf-8") as f:
        for i, (a, b) in enumerate(merges):
            f.write(f"{i}\t{a!r}\t{b!r}\n")

    longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

    print("Done.")
    print(f"vocab size: {len(vocab)}")
    print(f"merges: {len(merges)}")
    print(f"workers: {args.workers}")
    print()
    print(f"pre-tokenization time: {timings['pretokenize_time']:.2f}s")
    print(f"initial pair build time: {timings['build_pairs_time']:.2f}s")
    print(f"BPE merge loop time: {timings['merge_time']:.2f}s")
    print(f"total time: {timings['total_time']:.2f}s")
    print()
    print(f"peak memory: {end_mem:.2f} MB")
    print(f"peak memory increase: {end_mem - start_mem:.2f} MB")
    print()
    print(f"longest token id: {longest_id}")
    print(f"longest token length: {len(longest_token)} bytes")
    print(f"longest token repr: {longest_token!r}")
    print()
    print(f"saved vocab pickle: {vocab_path}")
    print(f"saved merges pickle: {merges_path}")
    print(f"saved vocab txt: {vocab_txt_path}")
    print(f"saved merges txt: {merges_txt_path}")


if __name__ == "__main__":
    main()