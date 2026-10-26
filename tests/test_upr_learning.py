"""Tests for the UPR feature pack, the label catalog and the learned arm (CS-8d).

Contracts under test (TEXT_PROCESSING_PLAN.md section 7.1, CS-8d):

- the UPR pack: the feature list, the group selection, the **catalog-gated**
  groups (the label-description coverage and the label-type one-hots need a
  label catalog), the K-dependent names, and the pack's **cell-local,
  label-free** character (no corpus statistic, no document identity, no length
  feature);
- the label catalog's loader (the edge: a JSONL file with a name /
  description / parent per label);
- the learned arm's driver: the fit on **train**, the evaluation on **val**,
  the three floors (the chance, the control-role and the random-vector one)
  and their construction (the class counts and the weights are preserved).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cluster_rag.config import (
    UprAnalysisConfig,
    UprLearnedConfig,
    UprLearnedFloorsConfig,
)
from cluster_rag.embed.random import RandomEmbedder
from cluster_rag.label_catalog import LabelEntry, load_label_catalog
from cluster_rag.text_processors.chunkers import build_chunker
from cluster_rag.text_processors.features.pair_features import ProfileUnits
from cluster_rag.text_processors.features.upr_features import (
    CATALOG_GROUPS,
    DEFAULT_UPR_FEATURE_GROUPS,
    UPR_FEATURE_GROUPS,
    UprFeatureEmbedders,
    UprFeatureExtractor,
    UprLabel,
    upr_feature_group,
)
from cluster_rag.text_processors.upr_diagnostics import feature_diagnostics
from cluster_rag.upr_learning import score_upr_learned_arms
from fake_embedder import FakeEmbedder

LABELS = (
    UprLabel(name="Detention", text="This recommendation is about Detention."),
    UprLabel(name="Poverty", text="This recommendation is about Poverty."),
)

DECISION = {"rule": "top_k", "k": 1}


def _extractor(
    *,
    dimension: int = 8,
    top_k: int = 3,
    groups: tuple[str, ...] | None = None,
    labels: tuple[UprLabel, ...] = LABELS,
    catalog_available: bool = False,
    embedder: object | None = None,
) -> UprFeatureExtractor:
    """A pack over the fake embedder (no network, no model download)."""
    backend = embedder or FakeEmbedder(dimension=dimension)
    return UprFeatureExtractor(
        UprFeatureEmbedders(text=backend, chunks=backend, labels=backend),
        labels=labels,
        text_chunker=build_chunker("atoms", {"split_clauses": True}),
        top_k=top_k,
        groups=groups,
        catalog_available=catalog_available,
    )


# --- the pack ---


def test_the_pack_has_the_planned_features() -> None:
    """The global cosine, the chunk cosines, the spread, the margin, the lexical block."""
    extractor = _extractor()
    assert extractor.feature_names == (
        "cosine_full",
        "chunk_cosine_best",
        "chunk_cosine_top2_mean",
        "chunk_cosine_top3_mean",
        "chunk_cosine_spread",
        "chunk_peak_margin",
        "lex_label_coverage",
        "lex_text_coverage",
        "lex_char_ngram_cosine",
    )
    assert extractor.groups == DEFAULT_UPR_FEATURE_GROUPS


def test_the_groups_partition_the_pack() -> None:
    """Every feature belongs to exactly one group (the ablation's unit)."""
    extractor = _extractor()
    for name in extractor.all_feature_names:
        assert upr_feature_group(name) in UPR_FEATURE_GROUPS


def test_the_catalog_gated_groups_need_a_catalog() -> None:
    """Without a catalog the description and the type features cannot be selected."""
    with pytest.raises(ValueError, match="label catalog"):
        _extractor(groups=("embedding", "label_part"))
    with pytest.raises(ValueError, match="label catalog"):
        _extractor(groups=("label_type",))


def test_the_catalog_gated_groups_work_with_a_catalog() -> None:
    """With a catalog the description coverage and the type one-hot appear."""
    labels = (
        UprLabel(name="Detention", description="arrest and detention", parent="Liberties"),
        UprLabel(name="Poverty", description="income and poverty", parent="Development"),
    )
    extractor = _extractor(
        groups=("embedding", "label_part", "label_type"),
        labels=labels,
        catalog_available=True,
    )
    assert "labelpart_description_coverage" in extractor.feature_names
    assert "labeltype_Liberties" in extractor.feature_names
    assert "labeltype_Development" in extractor.feature_names


def test_the_pack_extracts_a_text_by_label_cube() -> None:
    """The output is the grid's cells: (n_texts, n_labels, n_features)."""
    extractor = _extractor()
    cube = extractor.extract(["Ratify the convention", "Ensure housing for all"])
    assert cube.shape == (2, 2, len(extractor.feature_names))
    assert cube.dtype == np.float32


def test_the_pack_is_label_free_and_cell_local() -> None:
    """The same text scores the same against the same label, whatever else is around."""
    extractor = _extractor()
    text = "Ratify the convention on enforced disappearances"
    alone = extractor.extract([text])
    together = extractor.extract([text, "An unrelated text about housing"])
    assert np.allclose(alone[0], together[0])


def test_the_chunk_cosine_is_the_max_over_the_chunks() -> None:
    """The best chunk cosine is the maximum of the chunk similarities."""
    extractor = _extractor()
    text = "Ratify the convention. Ensure housing for all families."
    cube = extractor.extract([text])
    # the best chunk cosine dominates the K-best mean, and the peak margin
    # is exactly the difference between the two
    assert cube[0, 0, 1] >= cube[0, 0, 3] - 1e-6
    assert cube[0, 0, 5] == pytest.approx(cube[0, 0, 1] - cube[0, 0, 3], abs=1e-6)


def test_the_top_k_of_two_is_refused() -> None:
    """``top_k=2`` would duplicate the top-2 mean column."""
    with pytest.raises(ValueError, match="top-2"):
        _extractor(top_k=2)


def test_the_pack_describes_the_catalog_gap() -> None:
    """The artifact records the gap the two catalog-gated groups leave."""
    described = _extractor().describe()
    assert described["label_catalog"]["available"] is False
    assert set(described["label_catalog"]["groups_not_selected"]) == set(CATALOG_GROUPS)


# --- the chunk profile (the CS-8e group) ---


def _profile_extractor() -> UprFeatureExtractor:
    """A pack with the chunk profile's group selected."""
    backend = FakeEmbedder(dimension=8)
    units = ProfileUnits(
        sentences=build_chunker("units", {"kind": "sentence"}),
        ngrams=build_chunker("units", {"kind": "ngram", "n_words": 5}),
    )
    return UprFeatureExtractor(
        UprFeatureEmbedders(
            text=backend,
            chunks=backend,
            labels=backend,
            text_sentences=backend,
            text_ngrams=backend,
        ),
        labels=LABELS,
        text_chunker=build_chunker("atoms", {"split_clauses": True}),
        top_k=3,
        groups=("chunk_profile",),
        profile_units=units,
        profile_top_k=3,
    )


def test_the_chunk_profile_needs_the_unit_splitters() -> None:
    """The profile's group without the unit splitters is a config-time error."""
    with pytest.raises(ValueError, match="unit splitters"):
        _extractor(groups=("chunk_profile",))


def test_the_chunk_profile_computes_its_ten_columns() -> None:
    """The profile's ranks (two granularities) and the four alignment kinds."""
    extractor = _profile_extractor()
    assert extractor.feature_names == (
        "cprof_sent_top1",
        "cprof_sent_top2",
        "cprof_sent_top3",
        "cprof_ngram_top1",
        "cprof_ngram_top2",
        "cprof_ngram_top3",
        "cprof_align_sent",
        "cprof_align_ngram",
        "cprof_align_lexical",
        "cprof_align_semantic_only",
    )
    cube = extractor.extract(["Ratify the convention. Ensure housing for all families."])
    assert cube.shape == (1, 2, 10)
    # the alignment kinds are one-hot: exactly one granularity and one
    # lexicalness per cell
    for column in range(2):
        assert cube[0, column, 6] + cube[0, column, 7] == pytest.approx(1.0)
        assert cube[0, column, 8] + cube[0, column, 9] == pytest.approx(1.0)


def test_the_chunk_profile_ranks_are_the_sorted_unit_similarities() -> None:
    """The top-1 rank is the block's maximum and the ranks are decreasing."""
    extractor = _profile_extractor()
    cube = extractor.extract(["Ratify the convention. Ensure housing for all families."])
    assert cube[0, 0, 0] >= cube[0, 0, 1] - 1e-6
    assert cube[0, 0, 3] >= cube[0, 0, 4] - 1e-6


# --- the per-feature diagnostics ---


def test_the_feature_diagnostics_rank_and_separate() -> None:
    """A hand-computed case: the AUC, the separation and the length correlation."""
    features = np.asarray(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=np.float32,
    )
    gold = np.asarray([[True, False, True]])
    block = feature_diagnostics(features, gold, [10], ("good", "bad"))
    good = block["features"]["good"]
    bad = block["features"]["bad"]
    # the "good" feature is high on the gold cells and low on the background
    # one, so it separates the two sides; the "bad" one is high on the
    # background cell and low on one gold cell, so it is anti-correlated
    # (the hand-computed AUC: the ranks are 1, 2.5, 2.5)
    assert good["auc"] == pytest.approx(1.0)
    assert bad["auc"] == pytest.approx(0.25)
    assert good["gold_mean"] == pytest.approx(1.0)
    assert good["background_mean"] == pytest.approx(0.0)
    assert good["n_gold_cells"] == 2
    assert good["n_background_cells"] == 1


def test_the_feature_diagnostics_report_the_length_correlation() -> None:
    """A length proxy is visible: the feature's correlation with the length."""
    features = np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    gold = np.asarray([[True], [False], [False]])
    block = feature_diagnostics(features, gold, [1, 2, 3], ("lengthish",))
    assert block["features"]["lengthish"]["length_correlation"] == pytest.approx(1.0)


# --- the label catalog (the edge) ---


def test_the_label_catalog_is_read_from_a_jsonl_file(tmp_path: Path) -> None:
    """One object per line, with a name and optionally a description and a parent."""
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"name": "Detention", "description": "arrest", "parent": "Liberties"})
        + "\n"
        + json.dumps({"name": "Poverty"})
        + "\n",
        encoding="utf-8",
    )
    catalog = load_label_catalog(path)
    assert catalog["Detention"] == LabelEntry("Detention", "arrest", "Liberties")
    assert catalog["Poverty"] == LabelEntry("Poverty", None, None)


def test_the_label_catalog_rejects_a_repeated_label(tmp_path: Path) -> None:
    """A repeated label is an error, not a silent overwrite."""
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"name": "Detention"}) + "\n" + json.dumps({"name": "Detention"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repeats"):
        load_label_catalog(path)


# --- the driver: the fit, the evaluation and the floors ---


class _FixedCombiner:
    """A combiner that scores every row with a fixed value (the floor's shape)."""

    def __init__(self) -> None:
        self.rows: int | None = None

    def fit(self, features, labels, sample_weight=None) -> _FixedCombiner:
        self.rows = int(np.asarray(features).shape[0])
        return self

    def score(self, features) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        return matrix[:, 0] * 0.0 + 0.5


def test_the_learned_arm_fits_on_train_and_scores_on_val() -> None:
    """The combiner never sees the evaluated split."""
    extractor = _extractor()
    train_gold = np.asarray([[True, False], [False, True]])
    eval_gold = np.asarray([[True, False]])
    result = score_upr_learned_arms(
        train_texts=["train one", "train two"],
        train_gold=train_gold,
        eval_texts=["eval one"],
        eval_gold=eval_gold,
        extractor=extractor,
        combiner_factory=_FixedCombiner,
        decision=DECISION,
        ks=(1,),
        shortlist_ms=(1,),
        permutation_resamples=0,
    )
    assert result.scores["learned"].shape == (1, 2)
    assert result.meta["training"]["n_train_rows"] == 4
    assert result.meta["training"]["n_train_gold"] == 2
    assert result.meta["training"]["n_train_non_gold"] == 2


def test_the_control_role_floor_keeps_the_class_counts() -> None:
    """The positives move into the non-gold space; the counts stay the same."""
    from cluster_rag.upr_learning import _control_role_labels

    labels = np.asarray([1, 0, 0, 1, 0, 0])
    permuted = _control_role_labels(labels, np.random.default_rng(0))
    assert int((permuted == 1).sum()) == int((labels == 1).sum())
    # every positive is now on a row that was not gold
    assert not np.any(permuted[labels == 1] == 1)


def test_the_chance_floor_permutes_the_whole_label_vector() -> None:
    """The class counts stay exactly as they are; only the assignment changes."""
    from cluster_rag.upr_learning import _permuted_labels

    labels = np.asarray([1, 0, 0, 1, 0, 0])
    permuted = _permuted_labels(labels, np.random.default_rng(0))
    assert int((permuted == 1).sum()) == int((labels == 1).sum())


def test_the_random_vector_floor_replaces_the_embedding_sides() -> None:
    """The floor's pack has random vectors on every side; the lexical features stay."""
    extractor = _extractor()
    random_extractor = _extractor(embedder=RandomEmbedder(dimension=8, seed=1))
    train_gold = np.asarray([[True, False], [False, True]])
    eval_gold = np.asarray([[True, False]])
    result = score_upr_learned_arms(
        train_texts=["train one", "train two"],
        train_gold=train_gold,
        eval_texts=["eval one"],
        eval_gold=eval_gold,
        extractor=extractor,
        combiner_factory=_FixedCombiner,
        decision=DECISION,
        ks=(1,),
        shortlist_ms=(1,),
        permutation_resamples=0,
        random_extractor=random_extractor,
    )
    assert "learned_random" in result.scores
    assert "random_vectors" in result.meta["floors"]


def test_the_floors_are_measured_on_the_headline_statistic() -> None:
    """The floors report the decision view and the ranking view, like the arm."""
    extractor = _extractor()
    train_gold = np.asarray([[True, False], [False, True]])
    eval_gold = np.asarray([[True, False]])
    result = score_upr_learned_arms(
        train_texts=["train one", "train two"],
        train_gold=train_gold,
        eval_texts=["eval one"],
        eval_gold=eval_gold,
        extractor=extractor,
        combiner_factory=_FixedCombiner,
        decision=DECISION,
        ks=(1,),
        shortlist_ms=(1,),
        permutation_resamples=2,
    )
    summary = result.meta["floors"]["chance"]["summary"]
    assert summary["n_draws"] == 2
    assert "micro_f1" in summary
    assert "ndcg@1" in summary


# --- the config ---


def test_the_learned_section_is_off_by_default() -> None:
    """The earlier runs' artifacts keep their exact content (the knob's precedent)."""
    assert UprLearnedConfig().enabled is False
    assert UprAnalysisConfig().learned.enabled is False


def test_the_learned_section_rejects_an_unknown_group() -> None:
    """A typo in a group name fails before the run starts."""
    with pytest.raises(ValueError, match="Unknown feature group"):
        UprLearnedConfig(enabled=True, groups=("embedding", "nope"))


def test_the_learned_section_rejects_a_repeated_group() -> None:
    """A repeated group is an error."""
    with pytest.raises(ValueError, match="repeat"):
        UprLearnedConfig(enabled=True, groups=("embedding", "embedding"))


def test_the_learned_section_rejects_a_top_k_of_two() -> None:
    """``top_k=2`` would duplicate the pack's top-2 mean column."""
    with pytest.raises(ValueError, match="top-2"):
        UprLearnedConfig(enabled=True, top_k=2)


def test_the_floors_section_carries_the_three_floors() -> None:
    """All three floors are on by default (the plan's discipline)."""
    floors = UprLearnedFloorsConfig()
    assert floors.chance and floors.control_role and floors.random_vectors
    assert floors.permutation_resamples == 5


# --- the reporting blocks (CS-8f) ---


def test_the_reporting_block_carries_the_calibration_and_the_rules() -> None:
    """The reporting block has the calibration and the three decision rules."""
    extractor = _extractor()
    train_gold = np.asarray([[True, False], [False, True]])
    eval_gold = np.asarray([[True, False]])
    result = score_upr_learned_arms(
        train_texts=["train one", "train two"],
        train_gold=train_gold,
        eval_texts=["eval one"],
        eval_gold=eval_gold,
        extractor=extractor,
        combiner_factory=_FixedCombiner,
        decision=DECISION,
        ks=(1,),
        shortlist_ms=(1,),
        permutation_resamples=0,
        reporting={"enabled": True, "bins": 5, "threshold_quantiles": 9},
        bucket_masks={"all": np.asarray([True, True])},
        length_masks={"all": np.asarray([True])},
    )
    reporting = result.meta["reporting"]
    assert reporting["calibration"]["pooled"]["n"] == 2
    assert reporting["calibration"]["per_label_bucket"]["all"]["n"] == 2
    assert reporting["calibration"]["per_length_bucket"]["all"]["n"] == 2
    rules = reporting["decision_rules"]["rules"]
    assert set(rules) == {"top_k", "score_threshold", "top_k_min_score"}
    # the threshold comes from the fit side, never from the evaluated split
    assert reporting["decision_rules"]["threshold"]["value"] is not None
    assert reporting["decision_rules"]["threshold"]["n_candidates"] > 0


def test_the_reporting_block_is_absent_by_default() -> None:
    """The earlier runs' artifacts keep their exact content."""
    extractor = _extractor()
    result = score_upr_learned_arms(
        train_texts=["train one", "train two"],
        train_gold=np.asarray([[True, False], [False, True]]),
        eval_texts=["eval one"],
        eval_gold=np.asarray([[True, False]]),
        extractor=extractor,
        combiner_factory=_FixedCombiner,
        decision=DECISION,
        ks=(1,),
        shortlist_ms=(1,),
        permutation_resamples=0,
    )
    assert "reporting" not in result.meta
