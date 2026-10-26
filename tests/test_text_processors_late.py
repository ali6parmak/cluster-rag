"""Tests for the late chunking (Stage 2.7): the token capability, the spans, the pooling.

The contracts under test:

- the **optional token capability** of the ``Embedder`` port: it is probed,
  never assumed (a backend without it is reported as having none, through
  the disk cache too);
- the **character spans** of a chunker: exact for the word windows, the
  generic fallback for the others, and a clear error for a chunker whose
  chunks are not substrings of the text;
- the **pooling**: the model's own formula (``Normalize(mean(tokens))``) and
  the normalized-token variant, with unit rows and a clear error for a chunk
  without a token;
- the **driver's late side**: the arms it builds, the names they get and the
  "not available for this backend" path.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.embed.base import (
    EmbedderInfo,
    TokenEmbedding,
    prefix_shift,
    token_capability,
)
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.omct_analysis import LateSide, encode_late_side
from cluster_rag.text_processors.base import BaseChunker, SpanChunker
from cluster_rag.text_processors.chunkers.header import HeaderChunker
from cluster_rag.text_processors.chunkers.word_windows import (
    WordWindowsChunker,
    WordWindowsConfig,
)
from cluster_rag.text_processors.late_chunking import (
    LATE_POOLING_MEAN,
    LATE_POOLING_NORM,
    pool_chunk_vectors,
    pooling_report,
    token_chunk_indices,
)


class TokenEmbedder:
    """A fake embedder with the **token-level** capability.

    One token per word, each token's vector points at its own axis, so a
    pooled chunk vector says exactly which words it covers.
    """

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.encode_calls = 0

    @property
    def info(self) -> EmbedderInfo:
        """The interface facts of this fake embedder."""
        return EmbedderInfo(
            model_id="token-embedder",
            backend="fake",
            dimension=self.dimension,
            revision_or_digest="fake",
        )

    def encode(self, texts, kind) -> np.ndarray:
        """The model's own embedding of a text (the mean of its token vectors)."""
        self.encode_calls += 1
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            words = text.split()
            for index in range(len(words)):
                vectors[row, index % self.dimension] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors

    def encode_tokens(self, texts, kind) -> list[TokenEmbedding]:
        """One token per word, each with its own character span."""
        embeddings = []
        for text in texts:
            vectors = np.zeros((len(text.split()), self.dimension), dtype=np.float32)
            spans = []
            for index, word in enumerate(text.split()):
                start = text.index(word)
                vectors[index, index % self.dimension] = 1.0
                spans.append((start, start + len(word)))
            embeddings.append(TokenEmbedding(vectors=vectors, spans=tuple(spans)))
        return embeddings


class PlainEmbedder:
    """An embedder **without** the token-level capability (like Ollama's)."""

    @property
    def info(self) -> EmbedderInfo:
        """The interface facts of this fake embedder."""
        return EmbedderInfo(
            model_id="plain-embedder",
            backend="fake",
            dimension=4,
            revision_or_digest="fake",
        )

    def encode(self, texts, kind) -> np.ndarray:
        """A deterministic vector per text."""
        return np.ones((len(texts), 4), dtype=np.float32)


def test_token_capability_is_probed_not_assumed() -> None:
    """A backend without the token capability is reported as having none."""
    assert token_capability(TokenEmbedder()) is not None
    assert token_capability(PlainEmbedder()) is None
    # a backend that only claims the capability through the cache wrapper
    assert token_capability(object()) is None


def test_the_cache_forwards_the_capability_of_its_backend(tmp_path) -> None:
    """The cache decorator reports the *backend's* capability, not its own."""
    cached_tokens = CachedEmbedder(
        TokenEmbedder(),  # type: ignore[arg-type]
        tmp_path,
        preprocessing_version="test-v1",
    )
    cached_plain = CachedEmbedder(
        PlainEmbedder(),  # type: ignore[arg-type]
        tmp_path,
        preprocessing_version="test-v1",
    )
    assert token_capability(cached_tokens) is not None
    assert token_capability(cached_plain) is None
    with pytest.raises(ValueError, match="token-level"):
        cached_plain.encode_tokens(["a text"], "document")


def test_prefix_shift_counts_the_characters_in_front_of_the_text() -> None:
    """A plain prefix and a template both shift the text by a known amount."""
    assert prefix_shift("query: ") == len("query: ")
    assert prefix_shift("title: none | text: {content}") == len("title: none | text: ")
    assert prefix_shift("") == 0


def test_the_word_windows_spans_are_exact() -> None:
    """The windows' spans are the words' own character offsets."""
    chunker = WordWindowsChunker(WordWindowsConfig(n_words=2, stride=1))
    text = "alpha beta gamma delta"
    spans = chunker.segment_spans([text])
    assert spans == [[(0, 10), (6, 16), (11, 22)]]
    # every span is the text region of the matching chunk
    for chunk, (start, end) in zip(chunker.segment([text])[0], spans[0], strict=True):
        assert text[start:end].split() == chunk.split()


def test_the_generic_span_fallback_locates_the_chunks() -> None:
    """A chunker whose chunks are substrings gets spans from the fallback."""

    class RepeatingChunker(BaseChunker):
        """A chunker whose chunks repeat inside the text."""

        name = "repeating"

        def segment(self, texts):
            """Split each text into two halves."""
            return [[text[: len(text) // 2], text[len(text) // 2 :]] for text in texts]

    chunker = RepeatingChunker()
    spans = chunker.segment_spans(["abcdefghij"])
    assert spans == [[(0, 5), (5, 10)]]


def test_a_chunker_without_substring_chunks_has_no_spans() -> None:
    """The header chunker's chunks are not substrings: the error says so."""
    chunker = HeaderChunker()
    text = " ".join(f"word{index}." for index in range(30))
    with pytest.raises(ValueError, match="does not provide character spans"):
        chunker.segment_spans([text])


def test_the_chunkers_are_span_capable() -> None:
    """The word windows implement the span capability."""
    assert isinstance(WordWindowsChunker(), SpanChunker)


def test_token_chunk_indices_uses_the_overlap_rule() -> None:
    """A token joins the chunks its span overlaps; an empty span joins none."""
    token_spans = [(0, 3), (3, 6), (6, 9), (0, 0)]
    assert token_chunk_indices(token_spans, [(0, 6)]) == [[0, 1]]
    assert token_chunk_indices(token_spans, [(3, 9)]) == [[1, 2]]
    assert token_chunk_indices(token_spans, [(0, 3), (6, 9)]) == [[0], [2]]


def test_pool_chunk_vectors_uses_the_models_own_formula() -> None:
    """The pooled vector is ``Normalize(mean(tokens))`` of the chunk's tokens."""
    tokens = TokenEmbedding(
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        spans=((0, 2), (2, 4), (4, 6)),
    )
    pooled = pool_chunk_vectors(tokens, [(0, 4)], pooling=LATE_POOLING_MEAN)
    expected = np.asarray([0.5, 0.5], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(pooled[0], expected, atol=1e-6)
    assert np.isclose(np.linalg.norm(pooled[0]), 1.0, atol=1e-6)


def test_pool_chunk_vectors_normalizes_the_tokens_first_for_the_norm_rule() -> None:
    """The ``norm`` rule weighs every token the same before averaging."""
    tokens = TokenEmbedding(
        vectors=np.asarray([[3.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        spans=((0, 2), (2, 4)),
    )
    mean_rule = pool_chunk_vectors(tokens, [(0, 4)], pooling=LATE_POOLING_MEAN)
    norm_rule = pool_chunk_vectors(tokens, [(0, 4)], pooling=LATE_POOLING_NORM)
    assert not np.allclose(mean_rule[0], norm_rule[0])
    # the norm rule's mean is the mean of the two unit vectors
    expected = np.asarray([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
    assert np.allclose(norm_rule[0], expected, atol=1e-6)


def test_pool_chunk_vectors_rejects_an_unknown_rule_and_an_empty_chunk() -> None:
    """The errors are explicit: an unknown rule, a chunk without a token."""
    tokens = TokenEmbedding(
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        spans=((0, 2),),
    )
    with pytest.raises(ValueError, match="Unknown late pooling rule"):
        pool_chunk_vectors(tokens, [(0, 2)], pooling="median")
    with pytest.raises(ValueError, match="has no token"):
        pool_chunk_vectors(tokens, [(5, 9)], pooling=LATE_POOLING_MEAN)


def test_the_pooling_report_counts_the_coverage() -> None:
    """The report says how many tokens landed in a chunk."""
    tokens = TokenEmbedding(
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        spans=((0, 2), (2, 4), (4, 6)),
    )
    report = pooling_report(tokens, [(0, 4)])
    assert report["n_content_tokens"] == 3
    assert report["n_covered_tokens"] == 2
    assert report["token_coverage"] == pytest.approx(2 / 3)
    assert report["min_tokens_per_chunk"] == 2


def test_the_late_side_builds_one_encoding_per_pooling_rule() -> None:
    """One token pass, one encoding per rule, and the provenance in the meta."""
    chunker = WordWindowsChunker(WordWindowsConfig(n_words=2, stride=1))
    encodings = encode_late_side(
        TokenEmbedder(),  # type: ignore[arg-type]
        chunker,
        ["alpha beta gamma"],
        "document",
        poolings=(LATE_POOLING_MEAN, LATE_POOLING_NORM),
    )
    assert set(encodings) == {LATE_POOLING_MEAN, LATE_POOLING_NORM}
    for rule, encoding in encodings.items():
        assert encoding.n_texts == 1
        assert encoding.n_chunks == 2
        assert encoding.meta["late_chunking"] is True
        assert encoding.meta["pooling"] == rule
        assert encoding.meta["token_coverage_mean"] == pytest.approx(1.0)


def test_the_late_side_refuses_a_backend_without_the_token_capability() -> None:
    """The "not available for this backend" path says what is missing."""
    chunker = WordWindowsChunker(WordWindowsConfig(n_words=2, stride=1))
    with pytest.raises(ValueError, match="token-level embeddings"):
        encode_late_side(
            PlainEmbedder(),  # type: ignore[arg-type]
            chunker,
            ["alpha beta gamma"],
            "document",
        )


def test_the_late_side_names_its_arms() -> None:
    """The arm name carries the pooling rule and the chunker's slug."""
    side = LateSide(
        chunker=WordWindowsChunker(WordWindowsConfig(n_words=25, stride=12)),
        paragraph_embedder=TokenEmbedder(),  # type: ignore[arg-type]
        poolings=(LATE_POOLING_MEAN, LATE_POOLING_NORM),
    )
    assert side.slug(LATE_POOLING_MEAN) == "late_mean_words25"
    assert side.slug(LATE_POOLING_NORM) == "late_norm_words25"
