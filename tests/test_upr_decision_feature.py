"""Tests for the decision probability as a UPR pack feature (CS-13).

Contracts under test:

- the **decision group**: the pack's cube carries the decision probability as
  one more column, and the group is **off** by default (the earlier runs'
  artifacts keep their exact content);
- the **cell-local** contract: the feature is the matrix the caller passes
  in, unchanged (the pack never calls the model);
- the **shape contract**: a missing or a mis-shaped matrix is an error, not a
  silent zero column.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.config import UprDecisionFeatureConfig, UprLearnedConfig
from cluster_rag.text_processors.features.upr_features import (
    UPR_FEATURE_GROUPS,
    UprFeatureEmbedders,
    UprFeatureExtractor,
    UprLabel,
)


class _FakeEmbedder:
    """A deterministic fake embedder (the tests need no model)."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls = 0

    def encode(self, texts, kind):  # noqa: ANN001, ANN201
        """A deterministic vector per text."""
        self.calls += 1
        out = np.zeros((len(list(texts)), self.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            out[index, index % self.dimension] = 1.0
            out[index, 0] = float(len(text) % 7) / 10.0
        return out

    @property
    def info(self):  # noqa: ANN201
        """The embedder's info (the dimension)."""
        return type("Info", (), {"dimension": self.dimension})()


class _FakeChunker:
    """A chunker that returns the text itself (one chunk per text)."""

    def segment(self, texts):  # noqa: ANN001, ANN201
        """One chunk per text."""
        return [[text] for text in texts]

    @property
    def slug(self) -> str:
        """The chunker's slug."""
        return "fake"

    def cache_version(self, version: str) -> str:
        """The chunker's cache version."""
        return version


def _extractor(groups: tuple[str, ...]) -> UprFeatureExtractor:
    """An extractor with the fake embedders."""
    embedder = _FakeEmbedder()
    return UprFeatureExtractor(
        UprFeatureEmbedders(
            text=embedder,
            chunks=embedder,
            labels=embedder,
        ),
        labels=(UprLabel(name="Detention"), UprLabel(name="Poverty")),
        text_chunker=_FakeChunker(),
        groups=groups,
    )


def test_the_decision_group_is_off_by_default() -> None:
    """The earlier runs' artifacts keep their exact content (the CS-8d precedent)."""
    assert "decision" in UPR_FEATURE_GROUPS
    assert UprLearnedConfig().groups == ("embedding", "lexical")
    assert UprLearnedConfig().decision_feature.enabled is False
    assert UprDecisionFeatureConfig().enabled is False


def test_the_decision_feature_is_the_matrix_the_caller_passes_in() -> None:
    """The pack never calls the model: the column is the matrix, unchanged."""
    extractor = _extractor(("decision",))
    scores = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    cube = extractor.extract(["one", "two"], scores)
    assert cube.shape == (2, 2, 1)
    assert cube[0, 0, 0] == pytest.approx(0.9)
    assert cube[0, 1, 0] == pytest.approx(0.1)
    assert cube[1, 1, 0] == pytest.approx(0.8)


def test_the_decision_group_needs_the_matrix() -> None:
    """A missing matrix is an error, not a silent zero column."""
    extractor = _extractor(("decision",))
    with pytest.raises(ValueError, match="decision"):
        extractor.extract(["one", "two"])


def test_the_decision_matrix_must_have_the_grid_shape() -> None:
    """A mis-shaped matrix is an error too."""
    extractor = _extractor(("decision",))
    with pytest.raises(ValueError, match="shape"):
        extractor.extract(["one", "two"], np.zeros((2, 3)))


def test_the_decision_feature_config_needs_a_pinned_model() -> None:
    """An enabled feature without a model is a config error."""
    with pytest.raises(ValueError, match="model"):
        UprDecisionFeatureConfig(enabled=True)


def test_the_decision_feature_config_takes_a_pinned_model() -> None:
    """The model is pinned by id and digest (the track's hygiene)."""
    from cluster_rag.config import DecisionModelConfig

    model = DecisionModelConfig(
        name="tev1",
        model_id="tev1:latest",
        digest="7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740",
        instructions="Is the text about the topic in the question?",
    )
    feature = UprDecisionFeatureConfig(enabled=True, model=model)
    assert feature.model is not None
    assert feature.model.digest
