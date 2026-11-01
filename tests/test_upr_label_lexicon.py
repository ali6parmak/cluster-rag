"""Tests for the **label lexicon** as a channel (the track's CS-35).

Contracts under test (TEXT_PROCESSING_PLAN.md section 7.4, CS-35):

- the pack's **mined-words group** (``label_mined``): it is gated like the
  catalog's groups (no lexicon, no group) and its column is the mined words'
  soft coverage by the text — the same statistic as the description's
  coverage;
- the **label catalog's** schema: the optional ``mined_words`` field;
- the **label-embedding path**: the ``mined_words`` variant renders the
  label's corpus-mined words and records the lexicon in the artifact;
- the **fit side's cross-fit**: the fold-by-fold extraction with per-fold
  label sides (the guard that keeps the channel honest).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster_rag.cli import main
from cluster_rag.commands import embed as embed_commands
from cluster_rag.config import LabelTextsConfig, UprLearnedConfig
from cluster_rag.label_catalog import load_label_catalog
from cluster_rag.text_processors.chunkers import build_chunker
from cluster_rag.text_processors.features.upr_features import (
    LEXICON_GROUPS,
    UPR_FEATURE_GROUPS,
    UprFeatureEmbedders,
    UprFeatureExtractor,
    UprLabel,
    upr_feature_group,
)
from fake_embedder import FakeEmbedder

LABELS = (
    UprLabel(name="Detention", text="This recommendation is about Detention."),
    UprLabel(name="Poverty", text="This recommendation is about Poverty."),
)

MINED_LABELS = (
    UprLabel(
        name="Detention",
        text="This recommendation is about Detention.",
        mined_words=("detention", "arrest"),
    ),
    UprLabel(
        name="Poverty",
        text="This recommendation is about Poverty.",
        mined_words=("poverty",),
    ),
)


def _extractor(
    *,
    groups: tuple[str, ...],
    labels: tuple[UprLabel, ...] = LABELS,
    lexicon_available: bool = False,
) -> UprFeatureExtractor:
    """A pack over the fake embedder (no network, no model download)."""
    backend = FakeEmbedder(dimension=8)
    return UprFeatureExtractor(
        UprFeatureEmbedders(text=backend, chunks=backend, labels=backend),
        labels=labels,
        text_chunker=build_chunker("atoms", {"split_clauses": True}),
        top_k=3,
        groups=groups,
        lexicon_available=lexicon_available,
    )


# --- the pack's group ---


def test_the_mined_group_is_gated_like_the_catalog_ones() -> None:
    """Without a lexicon the mined-words group cannot be selected."""
    with pytest.raises(ValueError, match="label lexicon"):
        _extractor(groups=("embedding", "label_mined"))
    assert LEXICON_GROUPS == ("label_mined",)
    assert "label_mined" in UPR_FEATURE_GROUPS


def test_the_mined_column_is_the_words_coverage_by_the_text() -> None:
    """The column: the mined words' soft coverage by the text."""
    extractor = _extractor(groups=("label_mined",), labels=MINED_LABELS, lexicon_available=True)
    assert extractor.feature_names == ("labelmined_coverage",)
    assert upr_feature_group("labelmined_coverage") == "label_mined"
    cube = extractor.extract(["detention and arrest", "poverty relief"])
    # the first text carries both mined words of the first label
    assert cube[0, 0, 0] == pytest.approx(1.0)
    # and none of the second label's word
    assert cube[0, 1, 0] == pytest.approx(0.0)
    # the second text carries the second label's word
    assert cube[1, 1, 0] == pytest.approx(1.0)
    assert cube[1, 0, 0] == pytest.approx(0.0)


def test_the_pack_describes_the_lexicons_availability() -> None:
    """The pack's description records the lexicon and the group's gap."""
    extractor = _extractor(groups=("embedding",), lexicon_available=True)
    described = extractor.describe()
    assert described["label_lexicon"]["available"] is True
    assert described["label_lexicon"]["gated_groups"] == ["label_mined"]
    assert described["label_lexicon"]["groups_not_selected"] == ["label_mined"]


def test_the_pack_omits_the_lexicon_block_when_there_is_none() -> None:
    """No lexicon, no block: every earlier run's artifact keeps its content."""
    extractor = _extractor(groups=("embedding",))
    assert "label_lexicon" not in extractor.describe()


# --- the catalog's schema ---


def test_the_catalog_carries_optional_mined_words(tmp_path: Path) -> None:
    """The loader parses the optional ``mined_words`` field."""
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"name": "Detention", "mined_words": ["detention", "custody"]})
        + "\n"
        + json.dumps({"name": "Poverty", "description": "poverty reduction"})
        + "\n",
        encoding="utf-8",
    )
    catalog = load_label_catalog(path)
    assert catalog["Detention"].mined_words == ("detention", "custody")
    assert catalog["Poverty"].mined_words == ()


def test_the_catalog_rejects_a_broken_mined_words_field(tmp_path: Path) -> None:
    """A malformed ``mined_words`` field fails loudly."""
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"name": "Detention", "mined_words": "detention"}) + "\n", "utf-8")
    with pytest.raises(ValueError, match="mined_words"):
        load_label_catalog(path)


# --- the label-embedding path ---


MINED_LABEL_EXTRA = """
embed:
  backend: fake
  model_id: fake-embedder
label_texts:
  variant: mined_words
  template: "This recommendation is about {label}."
"""


def test_the_mined_words_variant_renders_the_lexicon(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """The label artifact: the mined words' text and the lexicon's block."""
    monkeypatch.setattr(
        embed_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(sample_jsonl, run_id="upr-labelembed-mined", extra=MINED_LABEL_EXTRA)
    assert main(["split", "--config", str(config_file)]) == 0
    assert main(["labelembed", "--config", str(config_file)]) == 0
    meta = json.loads(
        (tmp_path / "results" / "upr-labelembed-mined" / "meta.json").read_text("utf-8")
    )
    label_texts = meta["label_texts"]
    assert label_texts["variant"] == "mined_words"
    assert label_texts["top_k"] == 10
    assert label_texts["expansion_min_prefix"] == 4
    # the lexicon's block: the routes, the words and the provenance
    lexicon = label_texts["lexicon"]
    assert lexicon["corpus"]["split"] == "train"
    assert set(lexicon["routes"]) == {"mined", "expanded", "fallback"}
    assert lexicon["provenance"]["split"] == "train"
    # every label's rendered text carries its words (or its name, for the
    # fallback route)
    from cluster_rag.data import load_jsonl

    records = load_jsonl(sample_jsonl).records
    labels = sorted({label for record in records for label in record.labels})
    rows = {row["label"]: row for row in lexicon["per_label"]}
    assert len(label_texts["rendered_texts"]) == len(labels)
    for label, text in zip(labels, label_texts["rendered_texts"], strict=True):
        row = rows[label]
        expected = " ".join(word["word"] for word in row["words"]) or label
        assert text == f"This recommendation is about {expected}."


# --- the fit side's cross-fit ---


def test_the_cross_fit_extracts_the_fit_side_fold_by_fold() -> None:
    """Each fold's cells are extracted with that fold's label side."""
    from cluster_rag.commands.upr import _upr_cross_fit_cubes

    class _Setup:
        train_records = ()
        labels = ("Detention",)

    setup = _Setup()
    seen: list[int] = []

    def _label_sides(keep):
        seen.append(len(keep))
        return [UprLabel(name="Detention", text=f"fold with {len(keep)} records")]

    def _factory(sides):
        backend = FakeEmbedder(dimension=8)
        return UprFeatureExtractor(
            UprFeatureEmbedders(text=backend, chunks=backend, labels=backend),
            labels=sides,
            text_chunker=build_chunker("atoms", {"split_clauses": True}),
            top_k=3,
            groups=("embedding",),
        )

    cubes, meta = _upr_cross_fit_cubes(
        setup,  # type: ignore[arg-type]
        folds=2,
        seed=42,
        label_sides=_label_sides,
        extractor_factory=_factory,
    )
    assert meta["folds"] == 2 and meta["seed"] == 42
    assert meta["fold_sizes"] == [0, 0]
    assert seen == [0, 0]
    assert "train" in cubes


def test_the_cross_fit_needs_two_folds() -> None:
    """A cross-fit with fewer than two folds is refused."""
    from cluster_rag.commands.upr import _upr_cross_fit_cubes

    with pytest.raises(ValueError, match="at least 2 folds"):
        _upr_cross_fit_cubes(
            None,  # type: ignore[arg-type]
            folds=1,
            seed=42,
            label_sides=lambda keep: [],
            extractor_factory=lambda sides: None,  # type: ignore[arg-type,return-value]
        )


def test_the_learned_config_carries_the_lexicon_knobs() -> None:
    """The lexicon's knobs live in the learned arm's config (off by default)."""
    learned = UprLearnedConfig(enabled=True)
    assert learned.label_lexicon.enabled is False
    assert learned.label_lexicon.top_k == 10
    assert learned.label_lexicon.expansion_min_prefix == 4
    assert learned.label_lexicon.folds == 5


def test_the_label_texts_config_renders_the_mined_words() -> None:
    """The ``mined_words`` variant needs its words and renders the template."""
    config = LabelTextsConfig(
        variant="mined_words", template="This recommendation is about {label}."
    )
    assert config.render_text("Detention", None, ["detention", "arrest"]) == (
        "This recommendation is about detention arrest."
    )
    with pytest.raises(ValueError, match="needs the mined words"):
        config.render_text("Detention")
