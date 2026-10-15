"""Dataset-level statistics for the ``stats`` command (Phase 1 EDA).

All functions here are pure: they take records or values and return plain
JSON-serializable objects, with no file I/O and no global state. Thresholds
(e.g. bucket edges) come from the config, never from this module.
"""

from __future__ import annotations

import hashlib
import itertools
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cluster_rag.data import Record, summarize_records


def label_frequency_buckets(
    label_counts: Mapping[str, int], head_min: int, torso_min: int
) -> dict[str, list[str]]:
    """Partition labels into head / torso / tail buckets by example count.

    A label is head when ``count >= head_min``, torso when
    ``torso_min <= count < head_min``, and tail otherwise. Within each bucket
    labels are ordered by descending count, then alphabetically.
    """
    head: list[str] = []
    torso: list[str] = []
    tail: list[str] = []
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= head_min:
            head.append(label)
        elif count >= torso_min:
            torso.append(label)
        else:
            tail.append(label)
    return {"head": head, "torso": torso, "tail": tail}


def label_cooccurrence(records: Sequence[Record], top_n: int) -> list[tuple[tuple[str, str], int]]:
    """Count unordered label pairs co-occurring within a record; return the top_n.

    Labels are deduplicated within each record, so a pair is counted at most
    once per record regardless of duplicates (which :class:`Record` forbids
    anyway, this is defensive).
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        labels = sorted(set(record.labels))
        pair_counts.update(itertools.combinations(labels, 2))
    return pair_counts.most_common(top_n)


def leading_prefixes(
    texts: Sequence[str],
    *,
    max_words: int = 3,
    lowercase: bool = True,
    top_n: int = 10,
) -> dict[int, list[tuple[str, int]]]:
    """Count the most common leading 1..max_words word prefixes.

    Purpose: detect formulaic openers ("Continue to", "Ensure that", ...) that
    may dominate embedding space over topic content (PLAN.md sections 2/6).
    Words are whitespace-split, surrounding punctuation is stripped, and
    optionally lowercased before counting.
    """
    counters: dict[int, Counter[str]] = {n: Counter() for n in range(1, max_words + 1)}
    for text in texts:
        words = [word.strip(string.punctuation) for word in text.split()]
        if lowercase:
            words = [word.lower() for word in words]
        for n in counters:
            if len(words) >= n:
                prefix = " ".join(words[:n])
                if prefix:
                    counters[n][prefix] += 1
    return {n: counts.most_common(top_n) for n, counts in counters.items()}


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file, streamed; used for data provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_stats(
    records: Sequence[Record],
    *,
    head_min: int,
    torso_min: int,
    prefix_max_words: int = 3,
    prefix_lowercase: bool = True,
    top_n_labels: int = 15,
    top_n_pairs: int = 10,
    top_n_prefixes: int = 10,
) -> dict[str, Any]:
    """Build the full JSON-serializable dataset statistics report.

    Combines the record overview (:func:`cluster_rag.data.summarize_records`)
    with frequency buckets, top label co-occurrences and leading prefixes.
    All sizes and thresholds are parameters; compute nothing hard-coded here.
    """
    label_counts: Counter[str] = Counter()
    for record in records:
        label_counts.update(record.labels)
    buckets = label_frequency_buckets(label_counts, head_min, torso_min)

    bucket_sections: dict[str, Any] = {"thresholds": {"head_min": head_min, "torso_min": torso_min}}
    for bucket, labels in buckets.items():
        bucket_sections[bucket] = {
            "n_labels": len(labels),
            "label_counts": [[label, label_counts[label]] for label in labels],
        }

    prefixes = leading_prefixes(
        (record.text for record in records),
        max_words=prefix_max_words,
        lowercase=prefix_lowercase,
        top_n=top_n_prefixes,
    )
    return {
        "overview": summarize_records(records, top_labels=top_n_labels),
        "frequency_buckets": bucket_sections,
        "cooccurrence_top": [
            [first, second, count]
            for (first, second), count in label_cooccurrence(records, top_n_pairs)
        ],
        "prefixes": {str(n_words): pairs for n_words, pairs in prefixes.items()},
    }
