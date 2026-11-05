"""Tests for the error viewer's data layer (V1).

The fixture is a **tiny synthetic** run: seven labels, two train texts, four
val texts, three label tables and one empty table per held-out label. Every
number on the screen is checked by hand here, so the test cannot agree with the
layer by construction. No network, no model, no real artifact is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"

LABELS = ("A", "B", "C", "D", "E", "F", "G")
HOLDOUT = ("C", "E", "F", "G")
TRAIN = (
    {"id": "train-1", "text": "alpha beta", "language": "en", "labels": ["A", "E"]},
    {"id": "train-2", "text": "gamma delta", "language": "en", "labels": ["B", "F"]},
)
VAL = (
    {"id": "val-1", "text": "alpha beta", "language": "en", "labels": ["A"]},
    {"id": "val-2", "text": "omega", "language": "en", "labels": ["C"]},
    {"id": "val-3", "text": "delta", "language": "en", "labels": ["D"]},
    {"id": "val-4", "text": "omega", "language": "en", "labels": ["G"]},
)
# The map's hand-made tables: A sums to 3, B to 3, D to 5; C/E/F/G are empty.
TABLES: dict[str, list[dict[str, Any]]] = {
    "A": [
        {"word": "alpha", "z_score": 2.0, "log_odds": 3.0, "n_with": 1, "df": 2},
        {"word": "beta", "z_score": 1.0, "log_odds": 2.0, "n_with": 1, "df": 2},
        {"word": "gamma", "z_score": -1.0, "log_odds": -1.0, "n_with": 1, "df": 2},
    ],
    "B": [
        {"word": "beta", "z_score": 3.0, "log_odds": 4.0, "n_with": 1, "df": 2},
        {"word": "gamma", "z_score": 2.0, "log_odds": 1.5, "n_with": 1, "df": 2},
    ],
    "D": [
        {"word": "delta", "z_score": 5.0, "log_odds": 6.0, "n_with": 1, "df": 1},
    ],
    "C": [],
    "E": [],
    "F": [],
    "G": [],
}
# The classifier's val counts: v1 tp+fp, v2 two fakes + a near-miss,
# v3 tp+fp, v4 two fakes + a miss.  tp=2, fp=6, fn=2 -> micro-F1 = 4/12.
EXPECTED_F1 = 4.0 / 12.0


@dataclass(frozen=True)
class Fixture:
    """The tiny run's paths and the dataset's sha256."""

    config: Path
    results: Path
    data_raw: Path
    sha: str


def _load_module() -> Any:
    """Load the data layer from the repository's tools folder."""
    spec = importlib.util.spec_from_file_location("error_view", TOOLS / "error_view.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["error_view"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ev() -> Any:
    """The data-layer module."""
    return _load_module()


def _write_json(path: Path, payload: Any) -> None:
    """Write one JSON file, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _word_row(
    word: str,
    z_score: float,
    log_odds: float,
    *,
    n_with: int = 1,
    df: int = 2,
) -> dict[str, Any]:
    """One map row (the artifact's shape)."""
    return {
        "word": word,
        "n_with": n_with,
        "df": df,
        "lift": 2.0,
        "log_odds": log_odds,
        "z_score": z_score,
        "word_class": "distinctive",
        "is_name_word": False,
    }


def _map_rows() -> dict[str, list[dict[str, Any]]]:
    """The hand-made tables, with every field the layer reads."""
    rows = {label: [] for label in LABELS}
    rows["A"] = [
        _word_row("alpha", 2.0, 3.0),
        _word_row("beta", 1.0, 2.0),
        _word_row("gamma", -1.0, -1.0),
    ]
    rows["B"] = [_word_row("beta", 3.0, 4.0), _word_row("gamma", 2.0, 1.5)]
    rows["D"] = [_word_row("delta", 5.0, 6.0, df=1)]
    return rows


def _build_fixture(tmp_path: Path) -> Fixture:
    """Write the tiny run and return its paths."""
    data_raw = tmp_path / "data" / "upr.jsonl"
    data_raw.parent.mkdir(parents=True, exist_ok=True)
    data_raw.write_text(
        "\n".join(json.dumps(record) for record in (*TRAIN, *VAL)) + "\n", encoding="utf-8"
    )
    sha = hashlib.sha256(data_raw.read_bytes()).hexdigest()
    splits = tmp_path / "splits"
    for name, records in (("train", TRAIN), ("val", VAL)):
        _write_json(
            splits / f"{name}_ids.json",
            {
                "split": name,
                "run_id": "base",
                "source": str(data_raw),
                "source_sha256": sha,
                "ids": [record["id"] for record in records],
            },
        )
    _write_json(
        splits / "label_holdout.json",
        {
            "labels": list(HOLDOUT),
            "run_id": "base",
            "source": str(data_raw),
            "source_sha256": sha,
        },
    )
    results = tmp_path / "results"
    tables = _map_rows()
    _write_json(
        results / "upr-hitmap-corpus" / "hitmap.json",
        {
            "run_id": "upr-hitmap-corpus",
            "command": "hitmap-upr",
            "split": "val",
            "source": str(data_raw),
            "source_sha256": sha,
            "corpus_map": {
                "corpus": {
                    "split": "train",
                    "n_texts": len(TRAIN),
                    "n_labels": len(LABELS),
                    "n_words": 4,
                    "n_pairs": 6,
                    "labels_with_examples": 3,
                    "labels_without_examples": list(HOLDOUT),
                    "note": "the tiny fixture",
                },
                "per_label": [
                    {
                        "label": label,
                        "n_corpus_texts": len(TRAIN),
                        "n_words": len(tables[label]),
                        "words": tables[label],
                    }
                    for label in LABELS
                ],
                "per_word": [],
            },
        },
    )
    _write_json(
        results / "upr-eval-corpus-map-sum-zscore-positive-val" / "metrics.json",
        {
            "run_id": "upr-eval-corpus-map-sum-zscore-positive-val",
            "command": "evaluate",
            "split": "val",
            "source": str(data_raw),
            "source_sha256": sha,
            "n_records": len(VAL),
            "n_records_train_context": len(TRAIN),
            "n_labels": len(LABELS),
            "seed": 42,
            "decision": {"rule": "top_k", "k": 2, "threshold": None, "min_score": None},
            "buckets": {
                "labels_by_bucket": {
                    "head": ["A", "B"],
                    "tail": ["D"],
                    "unseen": list(HOLDOUT) + ["E", "F"],
                }
            },
            "report": {
                "overall": {
                    "n_gold_cells": len(VAL),
                    "f1": {"micro_f1": EXPECTED_F1, "n_predicted_cells": 8},
                }
            },
        },
    )
    vocabulary = ["alpha", "beta", "gamma"]
    coefficients = {
        "A": [0.5, -0.2, 0.0],
        "B": [0.0, 0.3, 0.1],
        "C": [0.0, 0.0, 0.0],
        "D": [0.0, 0.0, 0.0],
        "E": [0.0, 0.0, 0.0],
        "F": [0.0, 0.0, 0.0],
        "G": [0.0, 0.0, 0.0],
    }
    _write_json(
        results / "upr-hitmap-surrogate" / "hitmap.json",
        {
            "run_id": "upr-hitmap-surrogate",
            "split": "val",
            "source": str(data_raw),
            "source_sha256": sha,
            "ablation_map": {
                "arms": {
                    "fake-paragraph-noul": {
                        "surrogate": {
                            "full_vector": {
                                "vocabulary": vocabulary,
                                "topics": list(LABELS),
                                "coefficients": [coefficients[label] for label in LABELS],
                            }
                        }
                    }
                }
            },
        },
    )
    counts_a = [[1, 1, 0], [0, 2, 1], [1, 1, 0], [0, 2, 1]]
    counts_b = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
    _write_analysis(results, "upr-analyze-a", counts_a, 4.0 / 12.0)
    _write_analysis(results, "upr-analyze-b", counts_b, 1.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"run_id: test\nseed: 42\npaths:\n"
        f"  data_raw: {data_raw.as_posix()}\n"
        f"  data_processed: {(tmp_path / 'processed').as_posix()}\n"
        f"  data_splits: {splits.as_posix()}\n"
        f"  cache: {(tmp_path / 'cache').as_posix()}\n"
        f"  results: {results.as_posix()}\n",
        encoding="utf-8",
    )
    return Fixture(config=config, results=results, data_raw=data_raw, sha=sha)


def _write_analysis(
    results: Path,
    run_id: str,
    counts: list[list[int]],
    micro_f1: float,
    *,
    split: str = "val",
    per_text: bool = True,
    arm: str = "full_text",
) -> None:
    """Write one synthetic analysis artifact."""
    block: dict[str, Any] = {
        "arm": arm,
        "metrics": {"overall": {"f1": {"micro_f1": micro_f1}}},
    }
    if per_text:
        block["per_text"] = {"f1_counts": counts}
    _write_json(
        results / run_id / "analysis.json",
        {
            "run_id": run_id,
            "command": "analyze-upr",
            "split": split,
            "source": "data/upr.jsonl",
            "decision": {"rule": "top_k", "k": 2},
            "dataset": {"n_texts": len(counts), "n_labels": len(LABELS)},
            "arms": {arm: block},
        },
    )


@pytest.fixture()
def fixture(ev: Any, tmp_path: Path) -> Fixture:
    """A fresh tiny run (each test may tamper with its own copy)."""
    return _build_fixture(tmp_path)


@pytest.fixture()
def tables(ev: Any, fixture: Fixture) -> Any:
    """The three tables of the untampered fixture."""
    context = ev.load_context(ev.resolve_sources(fixture.config))
    return ev.build_tables(context, k=2, near_miss_k=5)


# --- the classifier's arithmetic --------------------------------------------


def test_the_scores_and_the_floor_are_hand_computable(ev: Any, fixture: Fixture) -> None:
    """Every score, rank and prediction is the hand-computed one."""
    context = ev.load_context(ev.resolve_sources(fixture.config))
    scores = ev.map_scores(context.tables, context.records, context.labels)
    assert scores[0].tolist() == [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert scores[2].tolist() == [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    decisions = ev.decision_scores(scores, context.tables, context.labels)
    # the four empty tables carry the row's smallest seen score minus one
    assert decisions[0, 2] == -1.0
    assert decisions[0, 6] == -1.0
    rank = ev.ranks(decisions)
    assert rank[0].tolist() == [1, 2, 4, 3, 5, 6, 7]
    predicted = ev.top_k(decisions, 2)
    assert predicted[0].tolist() == [True, True, False, False, False, False, False]
    assert predicted[2].tolist() == [True, False, False, True, False, False, False]


def test_the_error_taxonomy_boundaries(ev: Any) -> None:
    """The five types at their exact boundaries (k=2, K=5)."""
    assert ev.error_of(True, True, 1, 2, 5) == "tp"
    assert ev.error_of(False, True, 1, 2, 5) == "fp"
    assert ev.error_of(True, False, 3, 2, 5) == "near_miss"
    assert ev.error_of(True, False, 5, 2, 5) == "near_miss"
    assert ev.error_of(True, False, 6, 2, 5) == "fn"
    assert ev.error_of(False, False, 3, 2, 5) == "ok"


def test_text_f1_and_length_buckets(ev: Any) -> None:
    """The F1 edge case and the analyzer's length edges."""
    assert ev.text_f1(0, 0, 0) == 0.0
    assert ev.text_f1(1, 0, 0) == 1.0
    assert ev.text_f1(0, 1, 1) == 0.0
    assert [ev.length_bucket(value) for value in (0, 99, 100, 199, 200, 299, 300)] == [
        "0-99",
        "0-99",
        "100-199",
        "100-199",
        "200-299",
        "200-299",
        "300+",
    ]


# --- the three tables -------------------------------------------------------


def test_the_tables_have_the_hand_computed_cells(tables: Any) -> None:
    """The cells, the errors, the scores and the ranks are the hand-computed ones."""
    assert len(tables.texts) == 4
    assert len(tables.cells) == 28
    assert len(tables.words) == 4
    cells = {(row.text_id, row.label): row for row in tables.cells}
    assert cells[("val-1", "A")].error == "tp"
    assert cells[("val-1", "B")].error == "fp"
    assert cells[("val-2", "C")].error == "near_miss"
    assert cells[("val-2", "C")].rank == 4
    assert cells[("val-4", "G")].error == "fn"
    assert cells[("val-4", "G")].rank == 7
    # an empty table's score is 0.0, never the floor; it has no evidence
    assert cells[("val-2", "C")].score == 0.0
    assert not cells[("val-2", "C")].has_evidence
    assert not cells[("val-2", "C")].has_table
    # the zero-score tie-break is a prediction with no evidence
    assert cells[("val-2", "A")].predicted
    assert not cells[("val-2", "A")].has_evidence
    # the per-text counts (near-miss counts as fn, the artifact-compatible view)
    texts = {row.text_id: row for row in tables.texts}
    assert (texts["val-1"].tp, texts["val-1"].fp, texts["val-1"].fn) == (1, 1, 0)
    assert (texts["val-2"].tp, texts["val-2"].fp, texts["val-2"].fn) == (0, 2, 1)
    assert texts["val-1"].has_empty_slot is False
    assert texts["val-2"].has_empty_slot is True
    assert texts["val-4"].has_empty_slot is True
    assert texts["val-1"].margin == 3.0  # rank 2 (3.0) minus rank 3 (0.0)
    assert texts["val-3"].margin == 0.0  # rank 2 (0.0) minus rank 3 (0.0)
    assert sum(1 for cell in tables.cells if cell.predicted and not cell.has_evidence) == 5


def test_every_prediction_is_the_harness_top_k(tables: Any) -> None:
    """Every text has exactly two predictions (the run's 1,320/1,320 analogue)."""
    assert all(row.n_pred == 2 for row in tables.texts)
    assert sum(row.n_pred for row in tables.texts) == 8
    assert all(row.tp + row.fp == 2 for row in tables.texts)


def test_the_words_sum_to_the_scores(tables: Any) -> None:
    """The words table is the score's exact decomposition."""
    totals: dict[tuple[str, str], float] = {}
    for row in tables.words:
        totals[(row.text_id, row.label)] = (
            totals.get((row.text_id, row.label), 0.0) + row.contribution
        )
    for cell in tables.cells:
        if cell.score > 0.0:
            assert totals[(cell.text_id, cell.label)] == pytest.approx(cell.score)
            assert totals[(cell.text_id, cell.label)] > 0.0
        else:
            assert (cell.text_id, cell.label) not in totals


def test_the_surrogate_join_is_never_zeroed(tables: Any) -> None:
    """A word in the model's vocabulary gets its coefficient; delta gets None."""
    words = {(row.text_id, row.label, row.word): row for row in tables.words}
    assert words[("val-1", "A", "alpha")].surrogate_coefficient == pytest.approx(0.5)
    assert words[("val-1", "A", "beta")].surrogate_coefficient == pytest.approx(-0.2)
    # delta is not in the surrogate's vocabulary: None, never 0.0
    assert words[("val-3", "D", "delta")].surrogate_coefficient is None
    # only the positive z-scores are rows
    assert all(row.z_score > 0.0 for row in tables.words)
    assert ("val-1", "A", "gamma") not in words


# --- the guard --------------------------------------------------------------


def test_the_guard_passes_and_reproduces_the_artifact(ev: Any, fixture: Fixture) -> None:
    """The fixture reproduces its own metric, and all five checks hold."""
    report = ev.check_all(ev.resolve_sources(fixture.config))
    assert report.ok, [check.detail for check in report.checks if not check.ok]
    assert report.artifact_micro_f1 == pytest.approx(EXPECTED_F1, abs=1e-12)
    assert report.recomputed_micro_f1 == pytest.approx(EXPECTED_F1, abs=1e-12)
    assert report.provenance.k == 2
    assert report.provenance.near_miss_k == 5
    assert report.provenance.data_sha256 == fixture.sha
    # the five checks are all present
    assert [check.name for check in report.checks] == [
        "reproduction",
        "map provenance",
        "evaluation artifact",
        "surrogate",
        "compare series",
    ]


def test_the_guard_detects_a_tampered_metric(ev: Any, fixture: Fixture) -> None:
    """A metric that the recomputation cannot reproduce fails check 1."""
    path = fixture.results / "upr-eval-corpus-map-sum-zscore-positive-val" / "metrics.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["report"]["overall"]["f1"]["micro_f1"] = 0.9
    _write_json(path, payload)
    report = ev.check_all(ev.resolve_sources(fixture.config))
    assert not report.ok
    assert not report.checks[0].ok
    assert "0.9" in report.checks[0].detail


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["corpus_map"]["corpus"].update({"split": "val"}),
        lambda payload: payload.update({"source_sha256": "0" * 64}),
        lambda payload: payload["corpus_map"]["per_label"][0].update({"label": "Nope"}),
        lambda payload: payload["corpus_map"]["per_label"][2].update(
            {"words": [{"word": "alpha", "z_score": 1.0, "log_odds": 1.0, "n_with": 1, "df": 1}]}
        ),
    ],
)
def test_the_guard_detects_tampered_map_provenance(ev: Any, fixture: Fixture, mutate: Any) -> None:
    """The map's split, sha, label space and empty-table set are all guarded."""
    path = fixture.results / "upr-hitmap-corpus" / "hitmap.json"
    payload = json.loads(path.read_text("utf-8"))
    mutate(payload)
    _write_json(path, payload)
    report = ev.check_all(ev.resolve_sources(fixture.config))
    assert not report.ok
    assert not report.checks[1].ok


def test_the_guard_detects_a_tampered_eval_artifact(ev: Any, fixture: Fixture) -> None:
    """A split or a cell count the tool did not reproduce fails check 3."""
    path = fixture.results / "upr-eval-corpus-map-sum-zscore-positive-val" / "metrics.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["split"] = "train"
    payload["report"]["overall"]["f1"]["n_predicted_cells"] = 9
    _write_json(path, payload)
    report = ev.check_all(ev.resolve_sources(fixture.config))
    assert not report.ok
    assert not report.checks[2].ok


def test_the_guard_detects_a_tampered_surrogate(ev: Any, fixture: Fixture) -> None:
    """A surrogate whose topics are not the label space fails check 4."""
    path = fixture.results / "upr-hitmap-surrogate" / "hitmap.json"
    payload = json.loads(path.read_text("utf-8"))
    vector = payload["ablation_map"]["arms"]["fake-paragraph-noul"]["surrogate"]["full_vector"]
    vector["topics"] = vector["topics"][:-1]
    _write_json(path, payload)
    report = ev.check_all(ev.resolve_sources(fixture.config))
    assert not report.ok
    assert not report.checks[3].ok


def test_the_guard_refuses_a_series_that_does_not_add_up(ev: Any, fixture: Fixture) -> None:
    """A series whose counts contradict its own metric fails check 5."""
    _write_analysis(fixture.results, "upr-analyze-bad", [[1, 1, 0]] * 4, 0.5)
    report = ev.check_all(ev.resolve_sources(fixture.config))
    assert not report.ok
    assert not report.checks[4].ok
    catalog = ev.load_run_series(fixture.results, expected_n=4)
    assert any("upr-analyze-bad" in reason for reason in catalog.rejected)


def test_the_cli_check_exits_zero_then_one(ev: Any, fixture: Fixture, capsys: Any) -> None:
    """``--check`` is the headless guard: 0 when it reproduces, 1 when it does not."""
    assert ev.main(["--check", "--config", str(fixture.config)]) == 0
    output = capsys.readouterr().out
    assert "RESULT: OK" in output
    assert "micro-F1: artifact" in output
    path = fixture.results / "upr-eval-corpus-map-sum-zscore-positive-val" / "metrics.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["report"]["overall"]["f1"]["micro_f1"] = 0.9
    _write_json(path, payload)
    assert ev.main(["--check", "--config", str(fixture.config)]) == 1
    assert "RESULT: MISMATCH" in capsys.readouterr().out


# --- the aggregates ---------------------------------------------------------


def test_the_label_rates_are_hand_computed(ev: Any, tables: Any) -> None:
    """A's FP rate, C's miss rate and D's precision are the hand-computed ones."""
    rates = {row.label: row for row in ev.label_rates(tables.cells, LABELS)}
    assert (rates["A"].tp, rates["A"].fp, rates["A"].n_pred) == (1, 3, 4)
    assert rates["A"].precision == pytest.approx(0.25)
    assert rates["A"].recall == pytest.approx(1.0)
    assert rates["A"].fp_rate == pytest.approx(1.0)  # 3 of the 3 non-gold cells
    assert rates["C"].near_miss == 1
    assert rates["C"].miss_rate == pytest.approx(1.0)
    assert (rates["D"].precision, rates["D"].recall) == (1.0, 1.0)
    assert rates["G"].fn == 1


def test_the_confusion_pairs_are_hand_computed(ev: Any, tables: Any) -> None:
    """The gold -> predicted swaps, with the support and the share."""
    pairs = {(row.gold_label, row.predicted_label): row for row in ev.label_confusion(tables.cells)}
    assert set(pairs) == {("A", "B"), ("C", "A"), ("C", "B"), ("D", "A"), ("G", "A"), ("G", "B")}
    assert all(row.n == 1 for row in pairs.values())
    assert pairs[("A", "B")].support == 1
    assert pairs[("A", "B")].share == pytest.approx(1.0)


def test_the_false_positive_words_are_hand_computed(ev: Any, tables: Any) -> None:
    """alpha/beta carry one hit and one FP; delta carries only a hit."""
    words = {row.word: row for row in ev.false_positive_words(tables.cells, tables.words)}
    assert (words["alpha"].n_tp, words["alpha"].n_fp, words["alpha"].n_cells) == (1, 0, 1)
    assert (words["beta"].n_tp, words["beta"].n_fp, words["beta"].n_cells) == (1, 1, 2)
    assert (words["delta"].n_tp, words["delta"].n_fp) == (1, 0)


def test_the_coverage_holes(ev: Any, tables: Any) -> None:
    """The four labels never reached are the four empty tables."""
    holes = ev.coverage_holes(tables.cells, LABELS)
    assert holes.never_predicted == ("C", "E", "F", "G")
    assert holes.never_predicted_with_evidence == ("C", "E", "F", "G")
    assert holes.no_table == ("C", "E", "F", "G")


def test_the_bucket_rates(ev: Any, fixture: Fixture, tables: Any) -> None:
    """The length and frequency buckets' counts and micro-F1."""
    context = ev.load_context(ev.resolve_sources(fixture.config))
    rows = {
        (row.kind, row.bucket): row
        for row in ev.bucket_rates(tables.texts, tables.cells, LABELS, context.bucket_of_label)
    }
    assert rows[("length", "0-99")].n_cells == 28
    assert (rows[("length", "0-99")].tp, rows[("length", "0-99")].fp) == (2, 6)
    assert (rows[("length", "0-99")].fn, rows[("length", "0-99")].near_miss) == (1, 1)
    assert rows[("length", "0-99")].micro_f1 == pytest.approx(EXPECTED_F1)
    assert rows[("frequency", "head")].micro_f1 == pytest.approx(0.25)
    assert rows[("frequency", "tail")].micro_f1 == pytest.approx(1.0)
    assert rows[("frequency", "unseen")].near_miss == 1
    assert rows[("frequency", "unseen")].fn == 1


# --- compare runs -----------------------------------------------------------


def test_the_series_catalog_and_the_compare(ev: Any, fixture: Fixture, tables: Any) -> None:
    """The two synthetic runs pair up; the deltas are the hand-computed ones."""
    catalog = ev.load_run_series(fixture.results, expected_n=4)
    assert [series.label for series in catalog.series] == [
        "upr-analyze-a :: full_text",
        "upr-analyze-b :: full_text",
    ]
    assert catalog.rejected == ()
    context = ev.load_context(ev.resolve_sources(fixture.config))
    a, b = catalog.series
    rows = ev.compare_series(a, b, context.val_ids)
    assert len(rows) == 4
    assert all(row.improved for row in rows)
    assert rows[0].f1_a == pytest.approx(2.0 / 3.0)
    assert rows[0].f1_b == pytest.approx(1.0)
    assert (rows[0].delta_tp, rows[0].delta_fp, rows[0].delta_fn) == (0, -1, 0)
    assert (rows[1].delta_tp, rows[1].delta_fp, rows[1].delta_fn) == (1, -2, -1)
    # the dictionary series is the recomputed one, at the artifact's metric
    dictionary = ev.dictionary_series(context, tables, k=2)
    assert dictionary.micro_f1 == pytest.approx(EXPECTED_F1)
    assert [tuple(row) for row in dictionary.counts] == [(1, 1, 0), (0, 2, 1), (1, 1, 0), (0, 2, 1)]
    with pytest.raises(ValueError):
        ev.compare_series(a, ev.RunSeries("x", "y", 0.5, 3, a.counts), context.val_ids)


def test_the_series_catalog_skips_the_ineligible(ev: Any, fixture: Fixture) -> None:
    """A run without per_text and a non-val run are not offered; an omct run is ignored."""
    _write_analysis(fixture.results, "upr-analyze-nopred", [[1, 0, 0]] * 4, 1.0, per_text=False)
    _write_analysis(fixture.results, "upr-analyze-train", [[1, 0, 0]] * 4, 1.0, split="train")
    _write_analysis(fixture.results, "omct-analyze-x", [[1, 0, 0]] * 4, 1.0)
    catalog = ev.load_run_series(fixture.results, expected_n=4)
    assert [series.run_id for series in catalog.series] == ["upr-analyze-a", "upr-analyze-b"]
    assert catalog.rejected == ()
    assert any("upr-analyze-nopred" in reason for reason in catalog.ineligible)
    assert any("upr-analyze-train" in reason for reason in catalog.ineligible)
    assert all("omct-analyze-x" not in series.label for series in catalog.series)


# --- the one-text helpers and export ---------------------------------------


def test_the_evidence_ranking_and_the_contrast(ev: Any, fixture: Fixture) -> None:
    """CS-47's evidence ranking omits the zero-score labels; the contrast B follows."""
    context = ev.load_context(ev.resolve_sources(fixture.config))
    record = context.records[0]
    ranking = ev.evidence_ranking(context.tables, record.text, context.labels)
    assert [(row.rank, row.label, row.score) for row in ranking] == [(1, "A", 3.0), (2, "B", 3.0)]
    assert ev.contrast_label(context.tables, record.text, record.labels, context.labels) == "B"
    # a text with no evidence has no contrast
    assert ev.contrast_label(context.tables, "omega", ["C"], context.labels) == ""


def test_the_csv_export_has_the_provenance_block(ev: Any, tables: Any) -> None:
    """The export is a CSV with the provenance as # comments."""
    text = ev.to_csv(
        tables.texts[:2],
        ("text_id", "gold_labels", "predicted_labels", "tp", "text_f1"),
        provenance=["data: the tiny run", "cut: k=2"],
    )
    lines = text.splitlines()
    assert lines[0] == "# data: the tiny run"
    assert lines[1] == "# cut: k=2"
    assert lines[2] == "text_id,gold_labels,predicted_labels,tp,text_f1"
    assert len(lines) == 5
    assert "val-1,A,A | B,1," in text


def test_the_cli_dump_prints_the_table(ev: Any, fixture: Fixture, capsys: Any) -> None:
    """``--dump cells`` prints the guarded table; a failed guard refuses."""
    assert ev.main(["--dump", "cells", "--limit", "3", "--config", str(fixture.config)]) == 0
    output = capsys.readouterr().out
    assert "# config:" in output
    assert output.splitlines()[0].startswith("#")
    assert "text_id,label,gold,predicted,score,rank,error" in output
    assert len([line for line in output.splitlines() if line and not line.startswith("#")]) == 4


# --- the pure-core boundary --------------------------------------------------


def test_the_data_layer_has_no_model_or_ui_imports(ev: Any) -> None:
    """The data layer imports neither Streamlit nor a model client."""
    import ast  # noqa: PLC0415

    tree = ast.parse((TOOLS / "error_view.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("streamlit", "torch", "sentence_transformers", "httpx", "sqlite3"):
        assert forbidden not in imported
    assert "streamlit" in (TOOLS / "error_app.py").read_text(encoding="utf-8")


def test_the_hitmap_rows_are_the_stored_z_scores(ev: Any, fixture: Fixture) -> None:
    """The matrix helper: text order, occurrence counts, and the silence as None."""
    context = ev.load_context(ev.resolve_sources(fixture.config))
    rows = ev.hitmap_rows(context.tables, "alpha beta beta", ("A", "B"))
    assert [(row.word, row.n_occurrences, row.z_scores) for row in rows] == [
        ("alpha", 1, (2.0, None)),
        ("beta", 2, (1.0, 3.0)),
    ]
    # a word absent from every table is still a row; the cells are silence, not 0.0
    rows = ev.hitmap_rows(context.tables, "omega", ("A", "B"))
    assert [(row.word, row.z_scores) for row in rows] == [("omega", (None, None))]
