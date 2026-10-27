"""Tests for the reranker's port, its cache and its config (the track's CS-15).

Contracts under test:

- the **request shapes**: one request per (query, document) pair for the
  pair-based families, one request per query for the listwise family;
- the **pair template**: the ``{query}`` / ``{document}`` placeholders, and
  the refusal of an unknown placeholder;
- the **cache**: the key covers the model's identity, its family, the score's
  shape and the request (the document **order** included), a hit never
  reaches the model, and the miss is stored;
- the **config**: the pinned revision, the run-id-safe name, the template's
  family and the enabled-needs-models rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster_rag.config import RerankerModelConfig, UprRerankConfig
from cluster_rag.rerank.base import (
    KIND_CROSS_ENCODER,
    KIND_LISTWISE,
    KIND_LLM_YESNO,
    RerankInfo,
    RerankRequest,
    listwise_requests,
    pair_requests,
    render_template,
)
from cluster_rag.rerank.cache import (
    CachedReranker,
    rerank_cache_key,
    rerank_cache_path,
)

REVISION = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"


class _FakeReranker:
    """A reranker that returns a fixed score per document (no model)."""

    def __init__(self) -> None:
        self.calls = 0
        self._info = RerankInfo(model_id="fake/model", revision=REVISION, kind=KIND_CROSS_ENCODER)

    @property
    def info(self) -> RerankInfo:
        """The fake model's provenance."""
        return self._info

    def score(self, requests):  # type: ignore[no-untyped-def]
        """One score per document: its length."""
        self.calls += len(requests)
        return [[float(len(document)) for document in request.documents] for request in requests]


def _model(**overrides) -> RerankerModelConfig:
    """A pinned cross-encoder config."""
    payload = {
        "name": "fake",
        "model_id": "fake/model",
        "revision": REVISION,
        "kind": KIND_CROSS_ENCODER,
    }
    payload.update(overrides)
    return RerankerModelConfig(**payload)


def test_the_pair_requests_are_one_per_pair() -> None:
    """The pair-based families score one (query, document) pair per request."""
    requests = pair_requests(["q1", "q2"], [["a", "b"], ["c"]])
    assert len(requests) == 3
    assert requests[0] == RerankRequest(query="q1", documents=("a",))
    assert requests[2] == RerankRequest(query="q2", documents=("c",))


def test_the_pair_requests_reject_mismatched_lengths() -> None:
    """The queries and the document lists must agree."""
    with pytest.raises(ValueError, match="same length"):
        pair_requests(["q1"], [["a"], ["b"]])


def test_the_listwise_requests_carry_the_whole_candidate_list() -> None:
    """The listwise family reads the candidate list in one context."""
    requests = listwise_requests(["q1"], [["a", "b", "c"]])
    assert len(requests) == 1
    assert requests[0].documents == ("a", "b", "c")


def test_the_template_renders_both_sides() -> None:
    """The pair template renders the query and the document."""
    rendered = render_template("question:{query}\n\npassage:{document}", "Q", "D")
    assert rendered == "question:Q\n\npassage:D"


def test_the_template_refuses_an_unknown_placeholder() -> None:
    """A template with another placeholder is a typo, not a silent no-op."""
    with pytest.raises(ValueError, match="placeholders"):
        render_template("question:{query} {unknown}", "Q", "D")


def test_the_cache_key_covers_the_model_the_shape_and_the_request() -> None:
    """The key changes with the model, the revision, the shape and the request."""
    request = RerankRequest(query="q", documents=("a", "b"))
    base = rerank_cache_key(
        model_id="fake/model",
        revision=REVISION,
        kind=KIND_CROSS_ENCODER,
        request=request,
        shape={"max_length": 512},
    )
    assert base == rerank_cache_key(
        model_id="fake/model",
        revision=REVISION,
        kind=KIND_CROSS_ENCODER,
        request=request,
        shape={"max_length": 512},
    )
    assert base != rerank_cache_key(
        model_id="other/model",
        revision=REVISION,
        kind=KIND_CROSS_ENCODER,
        request=request,
        shape={"max_length": 512},
    )
    assert base != rerank_cache_key(
        model_id="fake/model",
        revision=REVISION,
        kind=KIND_LISTWISE,
        request=request,
        shape={"max_length": 512},
    )
    assert base != rerank_cache_key(
        model_id="fake/model",
        revision=REVISION,
        kind=KIND_CROSS_ENCODER,
        request=request,
        shape={"max_length": 1024},
    )
    # the document ORDER is part of the request (a listwise score depends on it)
    assert base != rerank_cache_key(
        model_id="fake/model",
        revision=REVISION,
        kind=KIND_CROSS_ENCODER,
        request=RerankRequest(query="q", documents=("b", "a")),
        shape={"max_length": 512},
    )


def test_the_cache_path_is_sharded_and_readable() -> None:
    """The cache file lives under the model's own directory, sharded by the key."""
    path = rerank_cache_path(
        "/tmp/cache",
        model_id="fake/model",
        revision=REVISION,
        kind=KIND_CROSS_ENCODER,
        key="abcdef0123456789",
    )
    assert "rerank" in path.parts
    assert path.name == "abcdef0123456789.json"
    assert path.parent.name == "ab"


def test_the_cache_serves_a_hit_without_the_model(tmp_path: Path) -> None:
    """A hit never reaches the model; a miss is stored and replayed."""
    fake = _FakeReranker()
    cached = CachedReranker(fake, tmp_path, shape={"max_length": 512})
    requests = [RerankRequest(query="q", documents=("a", "bb"))]
    first = cached.score(requests)
    assert first == [[1.0, 2.0]]
    assert fake.calls == 1
    assert cached.cache_misses == 1
    assert cached.cache_hits == 0

    second = cached.score(requests)
    assert second == first
    assert fake.calls == 1  # the second call came from the cache
    assert cached.cache_hits == 1


def test_the_cache_stores_a_readable_payload(tmp_path: Path) -> None:
    """The stored file carries the model's identity and the scores."""
    fake = _FakeReranker()
    cached = CachedReranker(fake, tmp_path, shape={"max_length": 512})
    cached.score([RerankRequest(query="q", documents=("a",))])
    files = list((tmp_path / "rerank").rglob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text("utf-8"))
    assert payload["scores"] == [1.0]
    assert payload["key"]["model_id"] == "fake/model"
    assert payload["key"]["revision"] == REVISION


def test_the_config_requires_a_pinned_revision() -> None:
    """The revision is a commit hash; anything shorter is a typo."""
    with pytest.raises(ValueError, match="commit hash"):
        _model(revision="main")


def test_the_config_requires_a_run_id_safe_name() -> None:
    """The arm's handle ends up in a run id."""
    with pytest.raises(ValueError, match="filesystem-safe"):
        _model(name="not safe")


def test_the_config_binds_the_template_to_the_yes_no_family() -> None:
    """The pair template belongs to the 'llm_yesno' family alone."""
    with pytest.raises(ValueError, match="needs the pair template"):
        _model(kind=KIND_LLM_YESNO)
    with pytest.raises(ValueError, match="takes no pair template"):
        _model(kind=KIND_CROSS_ENCODER, template="q:{query} d:{document}")


def test_the_rerank_stage_needs_a_model_when_enabled() -> None:
    """An enabled stage without a model is a config error, not an empty run."""
    with pytest.raises(ValueError, match="carries no model"):
        UprRerankConfig(enabled=True)
    config = UprRerankConfig(enabled=True, models=(_model(),))
    assert config.candidate_set == "decision"
    assert config.candidate_k == 10
