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
from cluster_rag.text_processors.features.lexical_map import (
    WORD_DISTINCTIVE,
    WORD_GENERIC,
    aggregate_lexical_map,
    build_lexical_map,
    cell_coverage,
    exactness_check,
    label_word_classes,
    label_word_contributions,
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
    import cluster_rag.text_processors.features.lexical_map as lexical_map

    for module in (ablation_map, lexical_map):
        source = Path(module.__file__).read_text("utf-8")
        for forbidden in ("httpx", "sqlite3", "torch", "sentence_transformers"):
            assert forbidden not in source
