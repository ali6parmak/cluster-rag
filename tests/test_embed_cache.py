"""Tests for the embedding disk cache (Phase 2).

Key insight to keep verified: a cache key of ``(model_id, prefix,
preprocessing_version, sha1(text))`` separates every component, and hits make
ZERO calls into the wrapped embedder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cluster_rag.embed.cache import CachedEmbedder, cache_key_path
from fake_embedder import CountingEmbedder, FakeEmbedder


@pytest.fixture()
def wrapped(tmp_path: Path):
    """A counting-wrapped fake embedder plus its cache, freshly created."""
    inner = CountingEmbedder(FakeEmbedder(model_id="fake-embedder", dimension=6))
    cache = CachedEmbedder(inner, tmp_path / "cache", preprocessing_version="raw-v1")
    return inner, tmp_path / "cache", cache


def test_cache_key_path_layout(tmp_path: Path) -> None:
    """The key is (model, prefix, preprocessing, sha1) as path components; slugs are safe."""
    path = cache_key_path(
        tmp_path,
        model_id="intfloat/e5-base-v2",
        prefix="query: ",
        preprocessing_version="raw-v1",
        text="continue to act",
    )
    text_sha1 = __import__("hashlib").sha1(b"continue to act").hexdigest()
    assert path.relative_to(tmp_path).parts == (
        "v1",
        "intfloat%2Fe5-base-v2",
        "raw-v1",
        "query%3A%20",
        f"{text_sha1}.npy",
    )
    # every key component participates
    different = [
        cache_key_path(
            tmp_path,
            model_id="other/model",
            prefix="query: ",
            preprocessing_version="raw-v1",
            text="continue to act",
        ),
        cache_key_path(
            tmp_path,
            model_id="intfloat/e5-base-v2",
            prefix="passage: ",
            preprocessing_version="raw-v1",
            text="continue to act",
        ),
        cache_key_path(
            tmp_path,
            model_id="intfloat/e5-base-v2",
            prefix="query: ",
            preprocessing_version="lower-v1",
            text="continue to act",
        ),
        cache_key_path(
            tmp_path,
            model_id="intfloat/e5-base-v2",
            prefix="query: ",
            preprocessing_version="raw-v1",
            text="other text",
        ),
    ]
    assert len({path, *different}) == 5
    # empty prefix has a stable placeholder component
    noprefix = cache_key_path(
        tmp_path, model_id="m", prefix="", preprocessing_version="raw-v1", text="t"
    )
    assert noprefix.relative_to(tmp_path).parts[3] == "-"


def test_first_call_encodes_and_stores(wrapped) -> None:
    """Cold cache: every text is delegated and stored under its key."""
    inner, cache_root, cache = wrapped
    out = cache.encode(["alpha", "beta"], "query")
    assert inner.total_texts_encoded == 2
    assert cache.cache_hits == 0 and cache.cache_misses == 2 and cache.n_stored == 2
    files = list(cache_root.rglob("*.npy"))
    assert len(files) == 2
    assert out.shape == (2, 6) and out.dtype == np.float32


def test_cache_hit_skips_re_encoding(wrapped) -> None:
    """Second identical call: zero encode calls, identical vectors back."""
    inner, _, cache = wrapped
    first = cache.encode(["alpha", "beta"], "query")
    assert inner._inner.call_count == 1  # cold: one encode call
    inner.total_texts_encoded = 0  # reset the proof counter

    second = cache.encode(["alpha", "beta"], "query")
    assert inner.total_texts_encoded == 0  # <-- the skip
    assert inner._inner.call_count == 1  # still the cold call: no re-encoding happened
    assert cache.cache_hits == 2 and cache.cache_misses == 2
    assert np.allclose(first, second, atol=0.0)  # byte-identical vectors


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    """A new CachedEmbedder over the same directory serves the old vectors."""
    first_inner = FakeEmbedder(model_id="fake-embedder", dimension=6)
    first = CachedEmbedder(first_inner, tmp_path, preprocessing_version="raw-v1")
    first_out = first.encode(["alpha"], "query")

    second_inner = FakeEmbedder(model_id="fake-embedder", dimension=6)
    second = CachedEmbedder(second_inner, tmp_path, preprocessing_version="raw-v1")
    second_out = second.encode(["alpha"], "query")
    assert second_inner.call_count == 0
    assert np.allclose(first_out, second_out, atol=0.0)


def test_cache_key_components_invalidate_hits(tmp_path: Path) -> None:
    """Different model / prefix / preprocessing version force a fresh encode."""
    inner_a = FakeEmbedder(model_id="fake-a", dimension=6)
    cache_a = CachedEmbedder(inner_a, tmp_path, preprocessing_version="raw-v1")
    cache_a.encode(["x"], "query")
    assert inner_a.call_count == 1

    inner_b = FakeEmbedder(model_id="fake-b", dimension=6)
    cache_b = CachedEmbedder(inner_b, tmp_path, preprocessing_version="raw-v1")
    cache_b.encode(["x"], "query")
    assert inner_b.call_count == 1  # different model: miss

    inner_c = FakeEmbedder(model_id="fake-a", dimension=6, prefixes={"query": "query: "})
    cache_c = CachedEmbedder(inner_c, tmp_path, preprocessing_version="raw-v1")
    cache_c.encode(["x"], "query")
    assert inner_c.call_count == 1  # different prefix: miss

    inner_d = FakeEmbedder(model_id="fake-a", dimension=6)
    cache_d = CachedEmbedder(inner_d, tmp_path, preprocessing_version="lower-v1")
    cache_d.encode(["x"], "query")
    assert inner_d.call_count == 1  # different preprocessing version: miss


def test_cache_passes_kind_and_reuses_across_calls(wrapped) -> None:
    """The inner embedder receives the original texts and the right kind; second
    identical call never reaches it."""
    inner, _, cache = wrapped
    cache.encode(["alpha"], "query")
    cache.encode(["alpha"], "query")
    assert inner._inner.encode_calls == [("query", ("alpha",))]  # one cold call only
    assert (cache.cache_hits, cache.cache_misses) == (1, 1)


def test_cache_dedups_repeated_texts_within_batch(wrapped) -> None:
    """Identical texts in one batch are encoded and stored once."""
    inner, cache_root, cache = wrapped
    out = cache.encode(["alpha", "alpha", "beta"], "query")
    assert inner.total_texts_encoded == 2  # alpha once + beta
    assert cache.n_stored == 2
    assert np.allclose(out[0], out[1], atol=0.0)
    assert len(list(cache_root.rglob("*.npy"))) == 2


def test_cache_rejects_corrupt_file(wrapped) -> None:
    """A cache file with the wrong shape errors instead of misaligning rows."""
    _, cache_root, cache = wrapped
    cache.encode(["alpha"], "query")
    path = next(cache_root.rglob("*.npy"))
    np.save(path, np.zeros(3, dtype=np.float32))
    with pytest.raises(RuntimeError, match="shape .*(3,).* \\(6,"):
        cache.encode(["alpha"], "query")


def test_cache_empty_batch_and_info_passthrough(wrapped) -> None:
    """Empty batches short-circuit; info carries the cache provenance."""
    inner, cache_root, cache = wrapped
    out = cache.encode([], "query")
    assert out.shape == (0, 6)
    info = cache.info
    assert info.model_id == "fake-embedder"
    assert info.extra["cache_root"] == str(cache_root)
    assert info.extra["preprocessing_version"] == "raw-v1"


def test_cache_info_reports_hits(wrapped) -> None:
    """Hit/miss counters are queryable for run metadata."""
    _, _, cache = wrapped
    cache.encode(["a"], "query")
    cache.encode(["a"], "query")
    assert (cache.cache_hits, cache.cache_misses) == (1, 1)
