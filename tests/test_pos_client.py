"""Unit tests for the **part-of-speech tagger at the edge** (the track's CS-34).

The client is a model client, so most of its tests use a **fake** tagger (no
download, no network): the disk cache's own contract (the key, the hit/miss
counters, the atomic write) and the model's **digest**. The nltk backend
itself needs the model, so its test is an **integration** test (marked and
skipped by default, like the embedder's and the decision model's).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster_rag.pos.base import TaggerInfo
from cluster_rag.pos.cache import (
    CachedTagger,
    tagger_cache_key,
    tagger_cache_path,
)
from cluster_rag.pos.nltk_tagger import model_digest


class FakeTagger:
    """A tagger that tags every token with the same tag (no model, no network)."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def info(self) -> TaggerInfo:
        """The fake model's provenance."""
        return TaggerInfo(model_id="fake-tagger", digest="f" * 64, backend="fake")

    def tag(self, text: str) -> list[tuple[str, str]]:
        """Every token tagged ``NN``."""
        from cluster_rag.text_processors.features.pair_features import tokenize

        self.calls += 1
        return [(token, "NN") for token in tokenize(text)]


def test_the_cache_key_covers_the_model_and_the_text() -> None:
    """The key: the model's identity and the text's own hash."""
    first = tagger_cache_key(model_id="m", digest="d", text="alpha")
    same = tagger_cache_key(model_id="m", digest="d", text="alpha")
    other_text = tagger_cache_key(model_id="m", digest="d", text="beta")
    other_digest = tagger_cache_key(model_id="m", digest="e", text="alpha")
    assert first == same
    assert first != other_text
    assert first != other_digest


def test_the_cache_path_keeps_the_cache_its_own_root(tmp_path: Path) -> None:
    """The tagger's cache never mixes with the embedding cache."""
    path = tagger_cache_path(tmp_path, model_id="m", key="abc")
    assert path.parent.parent.parent == tmp_path / "pos"
    assert path.suffix == ".json"


def test_the_cache_serves_the_hits_and_stores_the_misses(tmp_path: Path) -> None:
    """The cache's own contract: a hit, a miss and the stored answer."""
    fake = FakeTagger()
    tagger = CachedTagger(fake, tmp_path)
    first = tagger.tag("alpha beta")
    assert tagger.cache_misses == 1 and tagger.cache_hits == 0 and tagger.n_stored == 1
    second = tagger.tag("alpha beta")
    assert tagger.cache_hits == 1 and tagger.cache_misses == 1
    assert first == second
    assert fake.calls == 1  # the second call never reached the tagger
    # the stored file is readable and carries the tags
    stored = list(tmp_path.rglob("*.json"))
    assert len(stored) == 1
    payload = json.loads(stored[0].read_text(encoding="utf-8"))
    assert payload["tags"] == [["alpha", "NN"], ["beta", "NN"]]
    assert payload["key"]["digest"] == "f" * 64


def test_the_cache_passes_the_provenance_through(tmp_path: Path) -> None:
    """The artifact records the real model's id and digest, cache or not."""
    tagger = CachedTagger(FakeTagger(), tmp_path)
    info = tagger.info
    assert info.model_id == "fake-tagger"
    assert info.digest == "f" * 64
    assert info.extra["cache_root"] == str(tmp_path)


def test_the_digest_covers_the_models_own_files(tmp_path: Path) -> None:
    """The digest changes when the model's files change."""
    model = tmp_path / "tagger"
    model.mkdir()
    (model / "a.json").write_text('{"a": 1}', encoding="utf-8")
    first = model_digest(model)
    assert len(first) == 64
    assert model_digest(model) == first
    (model / "a.json").write_text('{"a": 2}', encoding="utf-8")
    assert model_digest(model) != first
    # an empty or missing directory has no digest
    assert model_digest(tmp_path / "nothing") == ""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert model_digest(empty) == ""


@pytest.mark.integration
def test_the_nltk_backend_tags_the_track_tokens(tmp_path: Path) -> None:
    """The nltk backend: the model's download, its digest and the tags.

    An **integration** test: it downloads the model, so it is marked and
    skipped by default (like the embedder's and the decision model's).
    """
    from cluster_rag.pos.nltk_tagger import NltkTagger

    client = NltkTagger(data_root=tmp_path / "nltk_data")
    info = client.info
    assert info.model_id == "averaged_perceptron_tagger_eng"
    assert len(info.digest) == 64
    tagged = client.tag("implement the strategy")
    assert [token for token, _tag in tagged] == ["implement", "the", "strategy"]
    tags = dict(tagged)
    assert tags["implement"] == "VB"
    assert tags["strategy"] == "NN"
