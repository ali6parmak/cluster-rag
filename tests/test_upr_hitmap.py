"""Tests for the track's **hit map** (CS-18).

Covers the two versions' pure sides and the CLI command:

- **Version A — the lexical map**: the per-word decomposition is **exact** (the
  contributions sum to the feature's own value), the words are classified as
  **distinctive** or **generic**, and the aggregates (the concentration, the
  held-out labels' split) are computed over the gold cells;
- **Version B — the ablation map**: the two ablation units, the removal's
  semantics, the variants (a one-unit text is **not ablatable**), the
  label-linked units and the three paired reads with their falsification flags;
- the CLI command: the artifact, the table row and the config-error contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cluster_rag.cli import main
from cluster_rag.commands import upr as upr_commands
from cluster_rag.config import DecisionModelConfig, UprAnalysisConfig, UprHitMapConfig
from cluster_rag.decision.base import DecisionAnswer, DecisionInfo
from cluster_rag.text_processors.decision.ablation_map import (
    UNIT_CLAUSE,
    UNIT_SENTENCE,
    ablatable_summary,
    build_ablation_cells,
    build_variants,
    is_formulaic,
    label_linked_units,
    measure_ablation,
    remove_unit,
    text_units,
)
from cluster_rag.text_processors.features.corpus_map import (
    build_corpus_map,
    corpus_map_block,
    log_odds,
)
from cluster_rag.text_processors.features.lexical_map import (
    WORD_DISTINCTIVE,
    WORD_GENERIC,
    aggregate_lexical_map,
    build_lexical_map,
    build_text_side_map,
    cell_coverage,
    exactness_check,
    label_word_classes,
    label_word_contributions,
    text_side_coverage,
    text_word_contributions,
)
from cluster_rag.text_processors.surrogate import (
    agreement_with_carrier,
    build_design_matrix,
    build_vocabulary,
    fit_ridge_surrogate,
    full_coefficient_vector,
    surrogate_predictions,
    surrogate_quality,
    topic_attribution,
)

OPENERS: tuple[str, ...] = ("continue to", "ensure that the", "take all necessary measures to")


# --- Version A: the lexical feature's own decomposition ---


def test_the_decomposition_is_exact() -> None:
    """The contributions of one cell sum to the feature's own value."""
    text = "Ensure freedom of expression on the Internet"
    contributions = label_word_contributions(text, "Freedom of opinion and expression")
    assert sum(item.contribution for item in contributions) == pytest.approx(
        cell_coverage(text, "Freedom of opinion and expression")
    )
    # the per-word share is the word's own coverage over the name's word count
    assert len(contributions) == 3  # "of" and "and" are stopwords
    for item in contributions:
        assert item.contribution == pytest.approx(item.coverage / 3)


def test_a_missing_word_contributes_nothing() -> None:
    """A word the text does not have contributes zero."""
    contributions = label_word_contributions("Ratify the convention", "Enforced disappearances")
    assert contributions[0].word == "enforced"
    assert contributions[0].coverage == pytest.approx(0.0)
    assert contributions[0].contribution == pytest.approx(0.0)


def test_the_words_are_classified_distinctive_or_generic() -> None:
    """A word another label also has is generic; a label's own word is distinctive."""
    classes = label_word_classes(["Right to health", "Health", "Right to education"])
    assert classes["Right to health"]["health"] == WORD_GENERIC
    assert classes["Right to health"]["right"] == WORD_GENERIC
    assert classes["Right to education"]["education"] == WORD_DISTINCTIVE
    assert classes["Health"]["health"] == WORD_GENERIC


def test_the_map_sums_to_the_feature_over_the_whole_grid() -> None:
    """The map's exactness check: the largest deviation is zero."""
    texts = ["Ensure freedom of expression on the Internet", "Ratify the convention"]
    labels = ["Freedom of opinion and expression", "International instruments"]
    cells = build_lexical_map(texts, labels)
    check = exactness_check(cells, texts, labels)
    assert check["n_cells"] == 4
    assert check["max_abs_error"] == pytest.approx(0.0)


def test_the_held_out_split_is_read_over_the_gold_cells() -> None:
    """The held-out labels' coverage is split between generic and distinctive words."""
    texts = ["Ensure access to health for all", "Ratify the health convention"]
    labels = ["Right to health", "Health"]
    gold = np.asarray([[True, False], [True, False]])
    classes = label_word_classes(labels)
    cells = build_lexical_map(texts, labels)
    block = aggregate_lexical_map(
        cells,
        labels=labels,
        gold=gold,
        buckets={"Right to health": "unseen", "Health": "head"},
        word_classes=classes,
        held_out_labels=["Right to health"],
        catch_all_labels=("Health",),
    )
    held_out = block["held_out_labels"]
    assert held_out["labels"] == ["Right to health"]
    # "right" is generic (another label has it) and "health" is generic too
    assert held_out["generic_share"] == pytest.approx(1.0)
    assert block["catch_all_labels"]["labels"] == ["Health"]
    # the concentration is reported: one word may carry the whole feature
    assert block["concentration"]["top1_word_share"] is not None


# --- Version B: the ablation plan and the measurement ---


def test_the_two_units_split_the_text_differently() -> None:
    """The clause unit is the finer one: a comma splits, a full stop does not."""
    text = "Ensure freedom of expression, by revoking decree 170"
    sentences = text_units(text, unit=UNIT_SENTENCE, openers=OPENERS)
    clauses = text_units(text, unit=UNIT_CLAUSE, openers=OPENERS)
    assert len(sentences) == 1
    assert len(clauses) == 2
    assert clauses[0].text == "Ensure freedom of expression"
    assert clauses[1].text == "by revoking decree 170"


def test_the_opener_marks_the_boilerplate() -> None:
    """A unit that opens with a config opener is formulaic."""
    assert is_formulaic("Continue to expand social protection", OPENERS)
    assert not is_formulaic("Expand social protection programmes", OPENERS)
    units = text_units(
        "Continue to expand social protection, for the most vulnerable",
        unit=UNIT_CLAUSE,
        openers=OPENERS,
    )
    assert [unit.formulaic for unit in units] == [True, False]


def test_the_removal_takes_the_separator_with_the_unit() -> None:
    """The ablated text is the text without the unit (and its separator)."""
    text = "Ensure freedom of expression, by revoking decree 170"
    units = text_units(text, unit=UNIT_CLAUSE, openers=OPENERS)
    assert remove_unit(text, units[0]) == "by revoking decree 170"
    assert remove_unit(text, units[1]) == "Ensure freedom of expression"


def test_a_one_unit_text_is_not_ablatable() -> None:
    """A text with one unit yields no variant (it is reported, never dropped)."""
    texts = ["Ensure freedom of expression", "Ensure freedom of expression, on the Internet"]
    units = tuple(text_units(text, unit=UNIT_CLAUSE, openers=OPENERS) for text in texts)
    variants = build_variants(texts, units)
    summary = ablatable_summary(units)
    # the first text has one unit: it contributes no variant at all
    assert [variant.text_row for variant in variants] == [1, 1]
    assert summary["n_texts"] == 2
    assert summary["n_ablatable"] == 1
    assert summary["n_not_ablatable"] == 1
    assert summary["share_not_ablatable"] == pytest.approx(0.5)
    # the variant count is the one the variants themselves carry
    assert summary["n_variants"] == len(variants)


def test_the_label_linked_unit_is_the_one_that_names_the_label() -> None:
    """The linked unit is the one whose content words overlap the label's name."""
    units = text_units("Ensure freedom of expression, by revoking decree 170", unit=UNIT_CLAUSE)
    assert label_linked_units(units, "Freedom of opinion and expression") == (0,)
    assert label_linked_units(units, "Poverty") == ()


def test_the_ablation_reads_the_drop_and_the_falsification() -> None:
    """The label-linked unit's drop is larger, and the blind counterpart's is not."""
    texts = ["Ensure freedom of expression, by revoking decree 170"]
    labels = ["Freedom of opinion and expression", "Poverty"]
    gold = np.asarray([[True, False]])
    units = (text_units(texts[0], unit=UNIT_CLAUSE, openers=OPENERS),)
    variants = build_variants(texts, units)
    assert len(variants) == 2
    # the intact text: the gold label at 0.9, the background at 0.1
    full = np.asarray([[0.9, 0.1]])
    # the label-linked unit removed: the gold label falls to 0.2
    # the other unit removed: the gold label barely moves
    ablated = np.asarray([[0.2, 0.05], [0.85, 0.09]])
    blind_full = np.asarray([[0.5, 0.5]])
    blind_ablated = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    block = measure_ablation(
        texts=texts,
        labels=labels,
        gold=gold,
        units=units,
        variants=variants,
        full_scores=full,
        ablated_scores=ablated,
        blind_full_scores=blind_full,
        blind_ablated_scores=blind_ablated,
        resamples=50,
        seed=1,
    )
    linked = block["reads"]["label_linked_vs_other"]
    assert linked["left_mean"] == pytest.approx(0.7)
    assert linked["right_mean"] == pytest.approx(0.05)
    assert linked["difference"] == pytest.approx(0.65)
    gold_read = block["reads"]["gold_vs_background"]
    assert gold_read["left_mean"] > gold_read["right_mean"]
    falsification = block["falsification"]
    assert falsification["label_linked_drop_greater"]["holds"] is True
    assert falsification["label_driven"]["holds"] is True
    # the blind counterpart's pattern is flat: it says nothing about the labels
    assert block["label_blind_counterpart"]["label_linked_vs_other"]["difference"] == pytest.approx(
        0.0
    )


def test_the_ablation_cells_carry_the_ablated_text() -> None:
    """The variant's cell asks about the ablated text (and the blind one shifts)."""
    texts = ["Ensure freedom of expression, on the Internet"]
    units = (text_units(texts[0], unit=UNIT_CLAUSE, openers=OPENERS),)
    variants = build_variants(texts, units)
    cells = build_ablation_cells(variants, ["Freedom of expression", "Poverty"])
    # two variants, two labels each
    assert len(cells) == 4
    assert cells[0].paragraph_text == "on the Internet"
    assert cells[2].paragraph_text == "Ensure freedom of expression"
    blind = build_ablation_cells(variants, ["Freedom of expression", "Poverty"], mismatched=True)
    assert blind[0].question_text == "Poverty"
    assert blind[1].question_text == "Freedom of expression"


# --- the CLI command ---


class FakeDecider:
    """A decider that answers from the state's text (no server, no model)."""

    def __init__(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_stored = 0
        self._info = DecisionInfo(model_id="fake:latest", digest="f" * 64, backend="fake")

    @property
    def info(self) -> DecisionInfo:
        """The fake model's provenance."""
        return self._info

    @property
    def usage(self) -> dict[str, int]:
        """The fake model's call counters."""
        return {"n_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def decide(self, state, questions):
        """Answer each question with 0.9 when the label's words are in the text."""
        text = str(state.get("paragraph", "")).lower()
        answers = {}
        for name, spec in questions.items():
            label = spec.instructions.split("Question: ", 1)[-1].split("\n", 1)[0].lower()
            value = 0.9 if label and label in text else 0.1
            answers[name] = DecisionAnswer(name=name, type="noul", value=value)
        return answers


HITMAP_EXTRA = """
eval:
  predictor:
    name: most_frequent
  decision:
    rule: top_k
    k: 2
upr_analysis:
  length_buckets: [40, 80]
  decision:
    models:
      - name: fake-paragraph-noul
        model_id: fake:latest
        digest: {digest}
        state: paragraph
        question_type: noul
        instructions: Is the text about the topic in the question?
        max_input_tokens: 2500
    scope: grid
    prompt_version: v1
    mismatched_control: true
  hitmap:
    unit: {unit}
    include_lexical_map: true
    include_label_blind: true
    catch_all_labels: [General, Other]
"""


@pytest.fixture()
def fake_decider(monkeypatch) -> FakeDecider:
    """Wire the CLI to a fake decider (no server, no model download)."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    return decider


def _hitmap_run(
    sample_jsonl,
    write_config,
    tmp_path,
    monkeypatch,
    unit="clause",
    run_id="upr-hitmap-test",
    table=None,
):
    """Run the hit-map command on the tiny fixture dataset."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = HITMAP_EXTRA.format(digest="f" * 64, unit=unit)
    config_file = write_config(
        sample_jsonl,
        run_id=run_id,
        extra=extra,
        experiment_table_text_processing_upr=table,
    )
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    code = main(["hitmap-upr", "--config", str(config_file)])
    results_dir = tmp_path / "results" / run_id
    path = results_dir / "hitmap.json"
    artifact = json.loads(path.read_text("utf-8")) if path.is_file() else None
    return code, artifact, results_dir


def test_hitmap_upr_writes_the_artifact(sample_jsonl, write_config, tmp_path, monkeypatch) -> None:
    """The run writes the map: the text set, the lexical map and the ablation."""
    code, artifact, results_dir = _hitmap_run(sample_jsonl, write_config, tmp_path, monkeypatch)
    assert code == 0
    assert artifact["command"] == "hitmap-upr"
    assert artifact["split"] == "val"
    assert (results_dir / "config.yaml").is_file()
    # the text set is the whole split, and its shape is reported
    assert artifact["text_set"]["n_texts"] > 0
    assert "share_not_ablatable" in artifact["text_set"]
    assert artifact["text_set"]["unit"] == "clause"
    # the lexical map (Version A) carries the exactness check
    lexical = artifact["lexical_map"]
    assert lexical["feature"] == "lex_label_coverage"
    assert lexical["exactness"]["max_abs_error"] == pytest.approx(0.0)
    # the ablation map (Version B) carries the three reads and the flags
    arm = next(iter(artifact["ablation_map"]["arms"].values()))
    assert set(arm["measurement"]["reads"]) == {
        "label_linked_vs_other",
        "formulaic_vs_topical",
        "gold_vs_background",
    }
    assert set(arm["measurement"]["falsification"]) >= {
        "label_linked_drop_greater",
        "formulaic_drop_smaller",
        "label_driven",
    }


def test_hitmap_upr_appends_the_table_row(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """The run appends one row to the track's UPR table."""
    table = tmp_path / "table.md"
    code, _, _ = _hitmap_run(sample_jsonl, write_config, tmp_path, monkeypatch, table=table)
    assert code == 0
    text = table.read_text("utf-8")
    assert "upr-hitmap-test" in text
    assert "clause" in text


def test_hitmap_upr_needs_a_model(sample_jsonl, write_config, tmp_path, monkeypatch) -> None:
    """Without a pinned decision model the command refuses to run."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = """
eval:
  predictor:
    name: most_frequent
upr_analysis:
  hitmap:
    unit: clause
"""
    config_file = write_config(sample_jsonl, run_id="upr-hitmap-nomodel", extra=extra)
    assert main(["split", "--config", str(config_file)]) == 0
    assert main(["hitmap-upr", "--config", str(config_file)]) == 1


def test_the_hit_map_config_contract() -> None:
    """The config carries the unit and the map's own knobs."""
    config = UprHitMapConfig()
    assert config.unit == "clause"
    assert config.include_lexical_map is True
    assert config.include_label_blind is True
    assert config.catch_all_labels == ("General", "Other")
    with pytest.raises(ValueError):
        UprHitMapConfig(unit="paragraph")


def test_the_analysis_config_carries_the_hit_map() -> None:
    """The hit map is a section of the UPR analysis config (off by default)."""
    analysis = UprAnalysisConfig()
    assert isinstance(analysis.hitmap, UprHitMapConfig)


def test_the_sweep_cell_resolves_the_hitmap_command() -> None:
    """The sweep cell's command and overrides are the map's."""
    from cluster_rag.sweep import HitMapUprCell

    cell = HitMapUprCell(
        label="upr-hitmap-clause",
        k=2,
        split="val",
        embedding_run="upr-embed-embeddinggemma",
        analysis=UprAnalysisConfig(
            decision=UprAnalysisConfig().decision.model_copy(
                update={
                    "models": (
                        DecisionModelConfig(
                            name="fake",
                            model_id="fake:latest",
                            digest="f" * 64,
                            state="paragraph",
                            question_type="noul",
                            instructions="Is the text about the topic in the question?",
                        ),
                    )
                }
            ),
            hitmap=UprHitMapConfig(unit="clause"),
        ),
    )
    assert cell.command == "hitmap-upr"
    overrides = cell.overrides
    assert overrides["eval"]["split"] == "val"
    assert overrides["upr_analysis"]["hitmap"]["unit"] == "clause"


def test_the_pure_core_stays_pure() -> None:
    """The map's modules import no model client (the track's boundary rule)."""
    import cluster_rag.text_processors.decision.ablation_map as ablation_map
    import cluster_rag.text_processors.features.corpus_map as corpus_map
    import cluster_rag.text_processors.features.lexical_map as lexical_map

    for module in (ablation_map, lexical_map, corpus_map):
        source = Path(module.__file__).read_text("utf-8")
        for forbidden in ("httpx", "sqlite3", "torch", "sentence_transformers"):
            assert forbidden not in source


# --- mode B: the text-side word map ---


def test_the_text_side_decomposition_is_exact() -> None:
    """The text's words sum to the cell's ``lex_text_coverage``."""
    text = "Ensure safe working conditions and protect the labour rights of migrant workers"
    contributions = text_word_contributions(text, "Labour rights")
    assert sum(item.contribution for item in contributions) == pytest.approx(
        text_side_coverage(text, "Labour rights")
    )
    # "labour" is fully covered by the label's name, "migrant" is not
    words = {item.word: item for item in contributions}
    assert words["labour"].coverage == pytest.approx(1.0)
    assert words["migrant"].coverage == pytest.approx(0.0)
    assert words["labour"].contribution > words["migrant"].contribution


def test_the_text_side_map_covers_the_grid_and_its_words() -> None:
    """The map's own aggregates: the exactness, the words and the per-cell rows."""
    texts = ["Ensure the labour rights of migrant workers", "Ratify the convention"]
    labels = ["Labour rights", "International instruments"]
    gold = np.asarray([[True, False], [False, True]])
    block = build_text_side_map(texts, labels, gold)
    assert block["exactness"]["max_abs_error"] == pytest.approx(0.0)
    assert block["exactness"]["n_cells"] == 4
    assert block["grid"]["n_cells"] == 4
    # the gold cell's decomposition carries the matching word
    cell = block["per_gold_cell"][0]
    assert cell["label"] == "Labour rights"
    assert cell["words"][0]["word"] == "labour"
    # the corpus-level words are the ones that matched ("rights" matches more
    # labels partially, so it can outrank "labour" in the corpus totals)
    matched = {word["word"] for word in block["corpus_top_words"]}
    assert "labour" in matched
    assert block["concentration"]["n_words"] >= 1


# --- mode C: the surrogate attribution ---


def test_the_surrogate_reads_the_topic_it_is_asked_about() -> None:
    """The per-topic coefficients separate the two topics' own words."""
    texts = [
        "ensure freedom of expression",
        "ratify the migrant workers convention",
        "freedom of expression online",
        "migrant workers rights",
    ]
    vocabulary = build_vocabulary(texts, min_document_frequency=1)
    design = build_design_matrix(texts, vocabulary)
    targets = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.75]])
    fit = fit_ridge_surrogate(design, targets, ridge_lambda=1e-6)
    assert fit.coefficients.shape == (2, len(vocabulary))
    rows = topic_attribution(fit, vocabulary, ["Freedom of expression", "Migrant workers"])
    first = {word["word"] for word in rows[0]["top_positive_words"]}
    second = {word["word"] for word in rows[1]["top_positive_words"]}
    assert "expression" in first and "freedom" in first
    assert "migrant" in second or "workers" in second
    assert not (first & second)


def test_the_surrogate_quality_is_reported_per_topic() -> None:
    """The quality block carries the per-topic correlation and R²."""
    texts = ["freedom of expression", "migrant workers", "expression online", "workers rights"]
    vocabulary = build_vocabulary(texts, min_document_frequency=1)
    design = build_design_matrix(texts, vocabulary)
    targets = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    fit = fit_ridge_surrogate(design, targets, ridge_lambda=1e-6)
    quality = surrogate_quality(surrogate_predictions(fit, design), targets)
    assert len(quality["per_topic_correlation"]) == 2
    assert quality["mean_correlation"] == pytest.approx(1.0)


def test_the_carrier_agreement_counts_the_words_inside_the_clause() -> None:
    """The cross-check: the top words that fall inside the carrier clause."""
    attribution = [
        {
            "topic": "Freedom of movement",
            "top_positive_words": [
                {"word": "movement", "coefficient": 1.0},
                {"word": "migrant", "coefficient": 0.5},
            ],
        }
    ]
    carriers = {(0, 0): "as well as freedom of movement"}
    block = agreement_with_carrier(
        attribution, carriers, text_words={0: "as well as freedom of movement for all"}
    )
    assert block["n_pairs"] == 1
    # "movement" is inside the clause, "migrant" is not in the text at all
    assert block["share_inside_the_carrier"] == pytest.approx(0.5)
    assert block["share_elsewhere_in_the_text"] == pytest.approx(0.0)
    assert block["share_absent_from_the_text"] == pytest.approx(0.5)
    # the chance level is the carrier clause's own share of the text's words
    assert block["chance_level"] is not None
    assert block["beats_the_chance_level"] in (True, False, None)


def test_the_surrogate_is_fitted_on_the_train_side_only() -> None:
    """The vocabulary is the fit side's: a val-only word is not a feature."""
    vocabulary = build_vocabulary(
        ["freedom of expression", "labour rights"], min_document_frequency=1
    )
    # the vocabulary is sorted, so the design matrix's columns are the same
    # from one run to the next
    assert vocabulary == ("expression", "freedom", "labour", "rights")


# --- mode C: the surrogate's full coefficient vector (CS-24) ---


def test_the_full_vector_matches_the_fit() -> None:
    """The stored vector is the fitted one, number for number."""
    texts = [
        "freedom of expression",
        "migrant workers",
        "expression online",
        "workers rights",
    ]
    vocabulary = build_vocabulary(texts, min_document_frequency=1)
    design = build_design_matrix(texts, vocabulary)
    targets = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    fit = fit_ridge_surrogate(design, targets, ridge_lambda=1e-6)
    vector = full_coefficient_vector(fit, vocabulary, ["Freedom", "Migrants"])
    assert vector["vocabulary"] == list(vocabulary)
    assert vector["topics"] == ["Freedom", "Migrants"]
    assert len(vector["coefficients"]) == 2
    for index in range(2):
        assert len(vector["coefficients"][index]) == len(vocabulary)
        for position in range(len(vocabulary)):
            assert vector["coefficients"][index][position] == pytest.approx(
                fit.coefficients[index][position], abs=1e-6
            )
    # the top-15 lists and the full vector agree (the lists are a reading of it)
    attribution = topic_attribution(fit, vocabulary, ["Freedom", "Migrants"])
    for entry in attribution[0]["top_positive_words"]:
        position = list(vocabulary).index(entry["word"])
        assert vector["coefficients"][0][position] == pytest.approx(entry["coefficient"], abs=1e-6)


def test_the_full_vector_refuses_a_mismatched_shape() -> None:
    """The vector's shape must match the fit (a silent misalignment is refused)."""
    texts = ["freedom of expression", "labour rights"]
    vocabulary = build_vocabulary(texts, min_document_frequency=1)
    design = build_design_matrix(texts, vocabulary)
    fit = fit_ridge_surrogate(design, np.asarray([[0.9, 0.1], [0.1, 0.9]]), ridge_lambda=1.0)
    with pytest.raises(ValueError):
        full_coefficient_vector(fit, vocabulary, ["only one topic"])
    with pytest.raises(ValueError):
        full_coefficient_vector(fit, vocabulary[:-1], ["A", "B"])


# --- the CLI: the two new modes ---


def test_hitmap_upr_runs_the_surrogate_mode(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """The surrogate mode writes its block: the fit, the quality and the attribution."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = (
        HITMAP_EXTRA.format(digest="f" * 64, unit="clause")
        + """
    include_surrogate: true
    include_ablation: true
    surrogate_min_df: 1
"""
    )
    config_file = write_config(sample_jsonl, run_id="upr-hitmap-surrogate-test", extra=extra)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["hitmap-upr", "--config", str(config_file)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "upr-hitmap-surrogate-test" / "hitmap.json").read_text("utf-8")
    )
    arm = next(iter(payload["ablation_map"]["arms"].values()))
    surrogate = arm["surrogate"]
    assert surrogate["fit"]["side"].startswith("the train split")
    assert surrogate["fit"]["n_texts"] > 0
    assert "mean_correlation" in surrogate["quality"]
    assert len(surrogate["per_topic"]) == payload["text_set"]["n_labels"]
    # the topic-faithfulness yardstick is reported beside the attribution
    assert "the real attribution's own-name share" in surrogate["topic_faithfulness"]
    # the full coefficient vector (CS-24): every topic's whole row
    vector = surrogate["full_vector"]
    assert len(vector["topics"]) == payload["text_set"]["n_labels"]
    assert len(vector["coefficients"]) == len(vector["topics"])
    n_features = surrogate["fit"]["n_features"]
    assert len(vector["vocabulary"]) == n_features
    assert all(len(row) == n_features for row in vector["coefficients"])
    # the top-15 lists are a reading of the same vector
    words = list(vector["vocabulary"])
    for row in vector["coefficients"][:5]:
        assert all(isinstance(value, float) for value in row)
    first = vector["coefficients"][0]
    stored = surrogate["per_topic"][0]
    for entry in stored["top_positive_words"]:
        position = words.index(entry["word"])
        assert first[position] == pytest.approx(entry["coefficient"], abs=1e-6)


def test_hitmap_upr_runs_the_text_side_mode(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """The text-side mode writes its block (and skips the ablation)."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = (
        HITMAP_EXTRA.format(digest="f" * 64, unit="clause")
        + """
    include_text_side_map: true
    include_surrogate: false
    include_ablation: false
"""
    )
    config_file = write_config(sample_jsonl, run_id="upr-hitmap-text-test", extra=extra)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["hitmap-upr", "--config", str(config_file)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "upr-hitmap-text-test" / "hitmap.json").read_text("utf-8")
    )
    assert payload["text_side_map"]["exactness"]["max_abs_error"] == pytest.approx(0.0)
    arm = next(iter(payload["ablation_map"]["arms"].values()))
    assert arm["measurement"]["applicable"] is False


# --- the corpus map (CS-23): the train split's own word<->label association ---


def test_the_corpus_map_counts_and_measures_are_hand_computed() -> None:
    """The four measures over a hand-computed 2x2 table."""
    texts = ["Alpha beta gamma", "Alpha delta", "Beta epsilon", "Zeta eta theta"]
    label_sets = [["L1"], ["L1"], ["L2"], ["L2"]]
    corpus = build_corpus_map(texts, label_sets, ["L1", "L2"])
    rows = {pair.word: pair for pair in corpus.topic_rows("L1")}
    # "alpha" is in both of L1's texts and in no other text: a = 2, b = 0,
    # c = 0, d = 2. The lift is 2.0 (the word doubles the label's rate) and
    # the smoothed log-odds is log(5) + log(5).
    alpha = rows["alpha"]
    assert (alpha.n_with, alpha.df) == (2, 2)
    assert alpha.lift == pytest.approx(2.0)
    assert alpha.log_odds == pytest.approx(math.log(25.0))
    # "beta" is in one of L1's two texts and in one of L2's: a = 1, b = 1,
    # c = 1, d = 1. The lift is 1.0 and the log-odds is 0.0 — the word says
    # nothing about the label.
    beta = rows["beta"]
    assert (beta.n_with, beta.df) == (1, 2)
    assert beta.lift == pytest.approx(1.0)
    assert beta.log_odds == pytest.approx(0.0)
    # a word that only ever appears with another label is not in this table
    assert "epsilon" not in rows
    # the reverse table is the same data read the other way
    reverse = {pair.label: pair for pair in corpus.word_rows("beta")}
    assert set(reverse) == {"L1", "L2"}
    assert reverse["L2"].n_with == 1


def test_the_log_odds_is_the_smoothed_two_by_two_table() -> None:
    """The score is the 2x2 table's log-ratio, with the 0.5 smoothing."""
    # a = 30, b = 5, c = 55, d = 1586
    assert log_odds(n_with=30, df=35, n_label=85, n_texts=1676) == pytest.approx(
        math.log(30.5 / 5.5) + math.log(1586.5 / 55.5)
    )
    # a word that is in every text and never with the label is negative:
    # a = 2, b = 98, c = 0, d = 0
    assert log_odds(n_with=2, df=100, n_label=2, n_texts=100) == pytest.approx(math.log(2.5 / 98.5))
    # counts that do not form a 2x2 table are refused, never silently smoothed
    with pytest.raises(ValueError):
        log_odds(n_with=1, df=100, n_label=2, n_texts=100)


def test_a_label_without_a_train_example_has_an_empty_table() -> None:
    """The held-out labels' tables are empty by construction, never fabricated."""
    texts = ["Alpha beta", "Beta gamma"]
    corpus = build_corpus_map(texts, [["L1"], ["L1"]], ["L1", "L2"])
    assert corpus.topic_rows("L2") == ()
    assert corpus.labels_without_examples() == ("L2",)
    assert corpus.word_rows("alpha")
    block = corpus_map_block(corpus, split="train")
    empty = next(row for row in block["per_label"] if row["label"] == "L2")
    assert empty["n_corpus_texts"] == 0
    assert empty["words"] == []
    assert "no train example" in empty["note"]
    assert block["corpus"]["labels_without_examples"] == ["L2"]


def test_the_corpus_map_never_sees_a_val_text() -> None:
    """The map is built from the train split alone (the leakage rule)."""
    train_texts = ["Alpha beta"]
    val_texts = ["Zeta omega"]
    corpus = build_corpus_map(train_texts, [["L1"]], ["L1"])
    assert "zeta" not in corpus.per_word
    assert corpus.n_texts == 1
    # the val text would have added a word and a label: it must not appear
    with_val = build_corpus_map(train_texts + val_texts, [["L1"], ["L2"]], ["L1", "L2"])
    assert "zeta" in with_val.per_word
    assert with_val.n_texts == 2


def test_the_corpus_map_flag_is_on_by_default() -> None:
    """The new mode is on by default (the reviewer's decision of 2026-11-01)."""
    assert UprHitMapConfig().include_corpus_map is True
    assert UprHitMapConfig(include_corpus_map=False).include_corpus_map is False


def test_hitmap_upr_writes_the_corpus_map_block(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """The corpus-map mode writes its block: the corpus, the measures, the tables."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = (
        HITMAP_EXTRA.format(digest="f" * 64, unit="clause")
        + """
    include_lexical_map: false
    include_text_side_map: false
    include_surrogate: false
    include_ablation: false
    include_corpus_map: true
"""
    )
    config_file = write_config(sample_jsonl, run_id="upr-hitmap-corpus-test", extra=extra)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["hitmap-upr", "--config", str(config_file)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "upr-hitmap-corpus-test" / "hitmap.json").read_text("utf-8")
    )
    corpus = payload["corpus_map"]
    assert corpus["corpus"]["split"] == "train"
    assert corpus["corpus"]["n_texts"] > 0
    assert corpus["corpus"]["n_words"] > 0
    assert corpus["corpus"]["n_pairs"] > 0
    # every pair is stored, with the counts behind it
    per_label = corpus["per_label"]
    assert len(per_label) == payload["text_set"]["n_labels"]
    # the CS-23 keys are untouched; CS-31's measures are added BESIDE them
    cs23_keys = {"word", "n_with", "df", "lift", "log_odds"}
    cs31_keys = {
        "log_odds_informative",
        "standard_error",
        "z_score",
        "g2",
        "p_value",
        "log_odds_conditional",
    }
    for row in per_label:
        for word in row["words"]:
            assert cs23_keys <= set(word)
            assert cs31_keys <= set(word)
            assert word["n_with"] <= word["df"]
            assert 0.0 <= word["p_value"] <= 1.0
    # the estimator's summary: the head word per measure and the readings the
    # change set's falsification is stated against
    estimator = corpus["estimator"]
    assert estimator["head_agreement"]["n_labels_with_a_table"] > 0
    assert "z_score_head_differs_from_log_odds" in estimator["head_agreement"]
    assert estimator["conditional_smoothing"]["n_pairs"] == corpus["corpus"]["n_pairs"]
    assert 0.0 <= estimator["conditional_smoothing"]["sign_flip_share"] <= 1.0
    # the reverse table holds the same pairs
    assert sum(len(row["labels"]) for row in corpus["per_word"]) == corpus["corpus"]["n_pairs"]
    # the corpus is the train split: a word that only a val text has is absent
    words = {row["word"] for row in corpus["per_word"]}
    assert "malnutrition" not in words
    assert "programmes" not in words


def test_the_word_classes_are_in_the_map_and_the_surrogate_blocks(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """CS-33: the class per pair, the readout, and the surrogate's own readout."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = (
        HITMAP_EXTRA.format(digest="f" * 64, unit="clause")
        + """
    include_lexical_map: false
    include_text_side_map: false
    include_surrogate: true
    include_ablation: false
    include_corpus_map: true
"""
    )
    config_file = write_config(sample_jsonl, run_id="upr-hitmap-word-classes-test", extra=extra)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["hitmap-upr", "--config", str(config_file)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "upr-hitmap-word-classes-test" / "hitmap.json").read_text("utf-8")
    )
    # the map's own block: the class per pair, the readout and the registered test
    classes = payload["corpus_map"]["word_classes"]
    assert classes["n_labelled_tables"] > 0
    assert classes["generic_threshold"] >= 3
    assert set(classes["readout"]) == {"distinctive", "shared", "generic"}
    assert "holds" in classes["falsification"]
    for row in payload["corpus_map"]["per_label"]:
        for word in row["words"]:
            assert word["word_class"] in {"distinctive", "shared", "generic"}
            assert isinstance(word["is_name_word"], bool)
    # the surrogate's own block: the model's habits, split the same way
    surrogate = payload["ablation_map"]["arms"]["fake-paragraph-noul"]["surrogate"]
    assert surrogate["word_classes"]["available"] is True
    assert set(surrogate["word_classes"]["readout"]) == {"distinctive", "shared", "generic"}
    assert surrogate["word_classes"]["n_coefficients_in_a_class"] > 0


def test_the_corpus_map_can_be_turned_off(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """A run that asks for no corpus map carries none (the flag stays a switch)."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = (
        HITMAP_EXTRA.format(digest="f" * 64, unit="clause")
        + """
    include_lexical_map: false
    include_text_side_map: true
    include_surrogate: false
    include_ablation: false
    include_corpus_map: false
"""
    )
    config_file = write_config(sample_jsonl, run_id="upr-hitmap-no-corpus-test", extra=extra)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["hitmap-upr", "--config", str(config_file)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "upr-hitmap-no-corpus-test" / "hitmap.json").read_text("utf-8")
    )
    assert payload["corpus_map"] is None


def test_the_corpus_map_is_computed_without_being_asked_for(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """The default is on: a run that says nothing about the map still carries it."""
    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    extra = (
        HITMAP_EXTRA.format(digest="f" * 64, unit="clause")
        + """
    include_lexical_map: false
    include_text_side_map: true
    include_surrogate: false
    include_ablation: false
"""
    )
    config_file = write_config(sample_jsonl, run_id="upr-hitmap-default-corpus-test", extra=extra)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["hitmap-upr", "--config", str(config_file)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "upr-hitmap-default-corpus-test" / "hitmap.json").read_text("utf-8")
    )
    assert payload["corpus_map"] is not None
    assert payload["corpus_map"]["corpus"]["n_pairs"] > 0
