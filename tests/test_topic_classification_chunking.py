"""Tests for the EURLEX-4K chunk-mean document view (TC-7)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.chunking import mean_pool, word_chunks


def test_short_text_is_one_chunk() -> None:
    assert word_chunks("a b c", chunk_words=200) == ["a b c"]


def test_chunks_are_non_overlapping_and_cover_every_word() -> None:
    text = " ".join(f"w{index}" for index in range(450))
    chunks = word_chunks(text, chunk_words=200)
    assert chunks == [
        " ".join(f"w{index}" for index in range(0, 200)),
        " ".join(f"w{index}" for index in range(200, 400)),
        " ".join(f"w{index}" for index in range(400, 450)),
    ]
    assert " ".join(chunks).split() == text.split()


def test_exact_multiple_has_no_empty_tail() -> None:
    text = " ".join(f"w{index}" for index in range(400))
    assert len(word_chunks(text, chunk_words=200)) == 2


def test_blank_text_has_no_chunks() -> None:
    assert word_chunks("   \n\t ", chunk_words=200) == []


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        word_chunks("a b", chunk_words=0)


def test_mean_pool_averages_and_normalizes() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pooled = mean_pool(vectors)
    assert pooled.dtype == np.float32
    assert pooled.shape == (2,)
    assert np.linalg.norm(pooled) == pytest.approx(1.0)
    assert pooled[0] == pytest.approx(pooled[1])


def test_mean_pool_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError):
        mean_pool(np.empty((0, 4), dtype=np.float32))
