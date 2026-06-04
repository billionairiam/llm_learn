from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


def _split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    special_tokens = [tok for tok in special_tokens if tok]
    if not special_tokens:
        return [text]

    pattern = "(" + "|".join(
        re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)
    ) + ")"
    return re.split(pattern, text)


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


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Initial byte vocabulary.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 2. Add special tokens.
    seen_special: set[bytes] = set()
    for tok in special_tokens:
        tok_bytes = tok.encode("utf-8")
        if tok_bytes not in seen_special:
            vocab[len(vocab)] = tok_bytes
            seen_special.add(tok_bytes)

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than initial vocab size {len(vocab)}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    special_set = set(special_tokens)

    # 3. Pre-tokenize and count pre-token byte sequences.
    #
    # Important optimization:
    # Internally use int token IDs instead of bytes.
    #
    # Example:
    #   b"low" -> (108, 111, 119)
    seq_counts: Counter[tuple[int, ...]] = Counter()

    for part in _split_by_special_tokens(text, special_tokens):
        if not part:
            continue

        # Special tokens are hard boundaries and excluded from merge statistics.
        if part in special_set:
            continue

        for match in re.finditer(PAT, part):
            token_bytes = match.group().encode("utf-8")
            seq = tuple(token_bytes)
            if seq:
                seq_counts[seq] += 1

    # seq_id -> current token sequence
    seqid2_seqs: dict[int, tuple[int, ...]] = {}

    # seq_id -> frequency
    seqid2_freq: dict[int, int] = {}

    for seq_id, (seq, freq) in enumerate(seq_counts.items()):
        seqid2_seqs[seq_id] = seq
        seqid2_freq[seq_id] = freq

    # 4. Build initial pair_counts and pair_to_seqids.
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_seqids: dict[tuple[int, int], set[int]] = defaultdict(set)

    for seq_id, seq in seqid2_seqs.items():
        freq = seqid2_freq[seq_id]
        for pair in zip(seq, seq[1:]):
            pair_counts[pair] += freq
            pair_to_seqids[pair].add(seq_id)

    # 5. Build heap for fast best-pair lookup.
    heap = [
        _heap_entry(pair, count, vocab)
        for pair, count in pair_counts.items()
        if count > 0
    ]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []

    # 6. BPE merge loop.
    while len(vocab) < vocab_size:
        # Lazy deletion from heap.
        # Pop until we find an entry whose count still matches pair_counts.
        best_pair: tuple[int, int] | None = None
        best_count = 0

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

        # Only update pre-tokens that contain best_pair.
        for seq_id in affected_seqs:
            old_seq = seqid2_seqs[seq_id]
            freq = seqid2_freq[seq_id]

            # Remove old pair contributions for this word.
            for old_pair in zip(old_seq, old_seq[1:]):
                new_count = pair_counts.get(old_pair, 0) - freq
                if new_count > 0:
                    pair_counts[old_pair] = new_count
                else:
                    pair_counts.pop(old_pair, None)

                seq_ids = pair_to_seqids.get(old_pair)
                if seq_ids is not None:
                    seq_ids.discard(seq_id)
                    if not seq_ids:
                        pair_to_seqids.pop(old_pair, None)

                touched_pairs.add(old_pair)

            # Apply merge in this word.
            new_seq = _merge_sequence(old_seq, best_pair, new_token_id)
            seqid2_seqs[seq_id] = new_seq

            # Add new pair contributions for this word.
            for new_pair in zip(new_seq, new_seq[1:]):
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
                pair_to_seqids[new_pair].add(seq_id)
                touched_pairs.add(new_pair)

        # Push updated pair counts into heap.
        # Old heap entries are allowed to stay; they will be ignored lazily.
        for pair in touched_pairs:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, _heap_entry(pair, count, vocab))

    return vocab, merges
