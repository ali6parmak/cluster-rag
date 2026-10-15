"""Tests for the dataset statistics module (Phase 1 EDA helpers)."""

from __future__ import annotations

from pathlib import Path

from cluster_rag.data import Record
from cluster_rag.dataset_stats import (
    build_dataset_stats,
    file_sha256,
    label_cooccurrence,
    label_frequency_buckets,
    leading_prefixes,
)


def _rec(rec_id: int, labels: list[str], text: str = "some text") -> Record:
    return Record.model_validate(
        {"id": str(rec_id), "text": text, "language": "en", "labels": labels}
    )


def test_label_frequency_buckets() -> None:
    """Bucket edges follow head_min / torso_min, ordered by count then name."""
    counts = {"a": 500, "b": 120, "c": 100, "d": 99, "e": 40, "f": 39, "g": 1}
    buckets = label_frequency_buckets(counts, head_min=100, torso_min=40)
    assert buckets["head"] == ["a", "b", "c"]
    assert buckets["torso"] == ["d", "e"]
    assert buckets["tail"] == ["f", "g"]


def test_label_frequency_buckets_tie_breaking() -> None:
    """Equal counts are ordered alphabetically."""
    buckets = label_frequency_buckets({"y": 10, "x": 10}, head_min=11, torso_min=10)
    assert buckets["torso"] == ["x", "y"]


def test_label_cooccurrence() -> None:
    """Pairs are counted once per record; deduplication inside a record is safe."""
    records = [
        _rec(1, ["A", "B"]),
        _rec(2, ["A", "B", "C"]),
        _rec(3, ["A"]),
    ]
    pairs = label_cooccurrence(records, top_n=5)
    assert pairs[0] == (("A", "B"), 2)
    assert (("A", "C"), 1) in pairs
    assert (("B", "C"), 1) in pairs


def test_leading_prefixes() -> None:
    """Prefixes are counted per length, lowercased, punctuation-stripped."""
    texts = [
        "Continue to implement the plan",
        "Continue to deliver results",
        "Ensure that everyone is safe",
        "Ratify, as soon as possible, the treaty",
    ]
    prefixes = leading_prefixes(texts, max_words=3, lowercase=True, top_n=5)
    assert prefixes[1][0] == ("continue", 2)
    assert ("ratify", 1) in prefixes[1]  # trailing comma stripped
    assert prefixes[2][0] == ("continue to", 2)
    assert ("ensure that", 1) in prefixes[2]
    assert ("continue to implement", 1) in prefixes[3]
    assert ("ratify as soon", 1) in prefixes[3]


def test_leading_prefixes_case_sensitive() -> None:
    """With lowercase=False, case differences create separate counts."""
    prefixes = leading_prefixes(
        ["Continue now", "continue now"], lowercase=False, max_words=2, top_n=5
    )
    assert all(count == 1 for _, count in prefixes[1])


def test_file_sha256(tmp_path) -> None:
    """Known SHA-256 vector for the bytes b'hello'."""
    source = tmp_path / "data.bin"
    source.write_bytes(b"hello")
    assert file_sha256(source) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_file_sha256_matches_stdlib(sample_jsonl: Path) -> None:
    """Cross-check against a direct hashlib computation on the fixture file."""
    import hashlib

    expected = hashlib.sha256(sample_jsonl.read_bytes()).hexdigest()
    assert file_sha256(sample_jsonl) == expected


def test_build_dataset_stats_structure() -> None:
    """The report combines overview, buckets, co-occurrences and prefixes."""
    records = [
        _rec(1, ["A", "B"], text="alpha beta"),
        _rec(2, ["A", "B"], text="alpha gamma"),
        _rec(3, ["C"], text="beta"),
    ]
    stats = build_dataset_stats(records, head_min=2, torso_min=1, top_n_pairs=2, top_n_prefixes=2)

    assert stats["overview"]["n_records"] == 3
    assert stats["overview"]["n_unique_labels"] == 3

    buckets = stats["frequency_buckets"]
    assert buckets["thresholds"] == {"head_min": 2, "torso_min": 1}
    assert buckets["head"]["label_counts"] == [["A", 2], ["B", 2]]
    assert buckets["torso"]["label_counts"] == [["C", 1]]
    assert buckets["tail"]["label_counts"] == []

    assert stats["cooccurrence_top"] == [["A", "B", 2]]
    assert list(stats["prefixes"]) == ["1", "2", "3"]
    assert stats["prefixes"]["1"][0] == ("alpha", 2)
