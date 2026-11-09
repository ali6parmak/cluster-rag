"""Tests for the OMCT dictionary error viewer's data layer (CS-58).

The fixture is a **tiny synthetic run**: ten reviewed pairs over three
paragraphs, two documents and four questions (one with no positive pair, so its
table is empty). The artifact's judged scores and rank percentiles are computed
from the same ten pairs with the round's own pure functions, and the tests then
check the layer's rebuild **by hand** (the fold map's positive z-scores summed
over the paragraph's units), the fill arithmetic, the class boundaries, the
error taxonomy and the guard's tamper detection.

No network, no model, no real artifact is read by the fixture. One optional
real-artifact AppTest at the end is skipped when `results/` is absent.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

TOOLS = Path(__file__).resolve().parent.parent / "tools"

RUN_ID = "omct-toy"
STABILITY_RUN = "omct-stability-toy"
ABSTAIN_RUN = "omct-abstain-toy"

# The toy source: 10 pairs over 2 documents, 3 paragraphs and 4 questions.
# Q4 has no positive pair, so its table is empty (the floor's toy case).
PAIRS = (
    ("T01", "Doc A", "1", "Paragraph one alpha beta common", 1, "Question one", 1),
    ("T02", "Doc A", "1", "Paragraph one alpha beta common", 2, "Question two", 0),
    ("T03", "Doc A", "2", "Paragraph two beta gamma common", 1, "Question one", 1),
    ("T04", "Doc A", "2", "Paragraph two beta gamma common", 2, "Question two", 1),
    ("T05", "Doc A", "2", "Paragraph two beta gamma common", 3, "Question three", 0),
    ("T06", "Doc B", "1", "Paragraph three alpha delta common", 1, "Question one", 0),
    ("T07", "Doc B", "1", "Paragraph three alpha delta common", 2, "Question two", 1),
    ("T08", "Doc B", "1", "Paragraph three alpha delta common", 3, "Question three", 1),
    ("T09", "Doc A", "1", "Paragraph one alpha beta common", 3, "Question three", 0),
    ("T10", "Doc A", "2", "Paragraph two beta gamma common", 4, "Question four", 0),
)


def _load_module() -> Any:
    """Load the data layer from the repository's tools folder."""
    spec = importlib.util.spec_from_file_location("omct_error_view", TOOLS / "omct_error_view.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["omct_error_view"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ev() -> Any:
    """The data-layer module."""
    return _load_module()


@dataclass(frozen=True)
class Toy:
    """The tiny run's paths and the guard's pinned counts."""

    config: Path
    results: Path
    source: Path
    artifact: Path
    stability: Path
    expected: dict[str, int]


def _write_json(path: Path, payload: Any) -> None:
    """Write one JSON file, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_row(row: tuple[str, str, str, str, int, str, int]) -> dict[str, Any]:
    """One source pair in the collaborator's own shape."""
    pair_id, document, paragraph, paragraph_text, question_number, question_text, label = row
    return {
        "pair_id": pair_id,
        "source_csv": "toy.csv",
        "document": document,
        "paragraph_number": paragraph,
        "paragraph_text": paragraph_text,
        "question_number": question_number,
        "question_text": question_text,
        "label": label,
        "label_origin": "toy",
    }


def _build_toy(tmp_path: Path, ev: Any) -> Toy:
    """Write the tiny source and artifact, computing its scores with the pure cores."""
    from cluster_rag.text_processors.features.question_map import build_question_map
    from cluster_rag.text_processors.lexical.question_map_score import score_grid

    source_rows = [_source_row(row) for row in PAIRS]
    source = tmp_path / "data" / "omct_pairs.json"
    _write_json(
        source,
        {
            "name": "toy",
            "description": "the CS-58 test fixture",
            "n_pairs": len(source_rows),
            "n_positive": sum(row["label"] for row in source_rows),
            "n_negative": sum(1 for row in source_rows if row["label"] == 0),
            "pairs": source_rows,
        },
    )
    paragraphs = list(dict.fromkeys(row["paragraph_text"] for row in source_rows))
    questions = list(dict.fromkeys(row["question_text"] for row in source_rows))
    paragraph_row = {text: index for index, text in enumerate(paragraphs)}
    question_col = {text: index for index, text in enumerate(questions)}
    texts = [row["paragraph_text"] for row in source_rows]
    labels = [row["label"] for row in source_rows]
    question_texts = [row["question_text"] for row in source_rows]
    rows = np.asarray([paragraph_row[text] for text in texts], dtype=np.int64)
    columns = np.asarray([question_col[text] for text in question_texts], dtype=np.int64)
    assignment = ev.fold_ids(len(source_rows), folds=ev.FOLDS, seed=ev.SEED)
    full_map = build_question_map(texts, question_texts, labels)
    fold_maps = {}
    for fold in range(ev.FOLDS):
        keep = [index for index in range(len(source_rows)) if int(assignment[index]) != fold]
        fold_maps[fold] = build_question_map(
            [texts[index] for index in keep],
            [question_texts[index] for index in keep],
            [labels[index] for index in keep],
        )
    full_grid = score_grid(paragraphs, questions, full_map, scoring="sum_zscore_positive")
    fold_grids = {
        fold: score_grid(paragraphs, questions, question_map, scoring="sum_zscore_positive")
        for fold, question_map in fold_maps.items()
    }
    assembled = full_grid.copy()
    for index in range(len(source_rows)):
        assembled[rows[index], columns[index]] = fold_grids[int(assignment[index])][
            rows[index], columns[index]
        ]
    from cluster_rag.text_processors.control import (
        percentiles_from_positions,
        rank_positions,
    )

    pairs = tuple(
        ev.SourcePair(
            pair_id=row["pair_id"],
            document=row["document"],
            paragraph_number=row["paragraph_number"],
            paragraph_text=row["paragraph_text"],
            question_number=row["question_number"],
            question_text=row["question_text"],
            label=row["label"],
        )
        for row in source_rows
    )
    candidates = ev._candidate_columns(pairs, paragraph_row, question_col, len(paragraphs))
    positions = rank_positions(assembled, rows, columns, candidates)
    percentiles = percentiles_from_positions(positions, rows, candidates)
    tables = {
        question: [asdict(row) for row in full_map.per_question.get(question, ())]
        for question in questions
    }
    words = {
        word: {"tables": len(word_rows), "df": int(full_map.word_df.get(word, 0))}
        for word, word_rows in full_map.per_word.items()
    }
    availability = []
    for index, pair in enumerate(pairs):
        fold = int(assignment[index])
        fold_map = fold_maps[fold]
        fold_rows = fold_map.per_question.get(pair.question_text, ())
        wanted = set(ev.content_tokens(pair.paragraph_text, ev.DEFAULT_STOPWORDS))
        availability.append(
            {
                "pair_id": pair.pair_id,
                "document": pair.document,
                "fold": fold,
                "label": pair.label,
                "has_table": bool(fold_rows),
                "n_evidence": sum(1 for row in fold_rows if row.word in wanted and row.z_score > 0),
            }
        )
    judged = [
        {
            "pair_id": pair.pair_id,
            "cell_id": pair.pair_id,
            "document": pair.document,
            "paragraph_number": pair.paragraph_number,
            "question_number": pair.question_number,
            "label": pair.label,
            "score": float(assembled[int(rows[index]), int(columns[index])]),
            "rank_percentile": float(percentiles[index]),
        }
        for index, pair in enumerate(pairs)
    ]
    n_floored = sum(
        1
        for index in range(len(pairs))
        if not fold_maps[int(assignment[index])].per_question.get(question_texts[index])
    )
    readout = {
        "primary": "sum_zscore_positive",
        "scorings": {
            arm: {
                "within_auc": 0.5,
                "pooled_auc": 0.5,
                "n_floor_pairs": n_floored,
                "n_pairs": len(pairs),
            }
            for arm in ("sum_positive", "sum_zscore_positive", "support_count")
        },
    }
    artifact_payload = {
        "run_id": RUN_ID,
        "command": "analyze-omct",
        "seed": ev.SEED,
        "n_valid_pairs": len(pairs),
        "dictionary": {
            "corpus": {
                "n_reviewed_pairs": len(pairs),
                "n_positive": sum(labels),
                "n_negative": len(pairs) - sum(labels),
                "n_questions": len(questions),
                "n_paragraphs": len(paragraphs),
                "n_documents": 2,
            },
            "cross_fit": {"folds": ev.FOLDS, "seed": ev.SEED},
            "map": {
                "n_questions": len(questions),
                "n_questions_with_table": sum(1 for rows in tables.values() if rows),
                "n_questions_empty": sum(1 for rows in tables.values() if not rows),
                "empty_questions": [question for question, rows in tables.items() if not rows],
                "n_words": len(words),
                "n_stored_pairs": sum(len(rows) for rows in tables.values()),
            },
            "tables": tables,
            "words": words,
            "availability": {"per_pair": availability},
        },
        "models": [
            {
                "name": "toy",
                "kind": "dictionary",
                "model": "toy",
                "arms": {
                    "dictionary": {
                        "arm": "dictionary",
                        "scoring": "sum_zscore_positive",
                        "judged_pairs": judged,
                        "control_cells": [
                            {
                                "cell_id": "C1",
                                "document": "Doc A",
                                "paragraph_number": "1",
                                "question_number": 4,
                                "score": -1.0,
                                "rank_percentile": 0.9,
                            },
                            {
                                "cell_id": "C2",
                                "document": "Doc B",
                                "paragraph_number": "1",
                                "question_number": 3,
                                "score": 0.5,
                                "rank_percentile": 0.2,
                            },
                            {
                                "cell_id": "C3",
                                "document": "Doc A",
                                "paragraph_number": "2",
                                "question_number": 4,
                                "score": -1.0,
                                "rank_percentile": 0.8,
                            },
                        ],
                    }
                },
                "readout": readout,
            }
        ],
    }
    results = tmp_path / "results"
    artifact = results / RUN_ID / "analysis.json"
    _write_json(artifact, artifact_payload)
    stability = results / STABILITY_RUN / "analysis.json"
    _write_json(
        stability,
        {
            "run_id": STABILITY_RUN,
            "seed": ev.SEED,
            "dictionary": {
                "stability": {
                    "resamples": 5,
                    "seed": ev.SEED,
                    "links": [
                        {
                            "question": "Question one",
                            "word": "alpha",
                            "present_share": 0.9,
                            "sign_stability": 0.8,
                            "head_stability": 0.6,
                        },
                        {
                            "question": "Question one",
                            "word": "beta",
                            "present_share": 0.7,
                            "sign_stability": 0.5,
                            "head_stability": 0.1,
                        },
                    ],
                }
            },
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run_id": "toy",
                "seed": ev.SEED,
                "paths": {
                    "data_raw": str(tmp_path / "upr.jsonl"),
                    "data_processed": str(tmp_path / "processed"),
                    "data_splits": str(tmp_path / "splits"),
                    "cache": str(tmp_path / "cache"),
                    "results": str(results),
                    "data_external": str(source),
                },
            }
        ),
        encoding="utf-8",
    )
    return Toy(
        config=config,
        results=results,
        source=source,
        artifact=artifact,
        stability=stability,
        expected={
            "EXPECTED_PAIRS": len(pairs),
            "EXPECTED_POSITIVE": sum(labels),
            "EXPECTED_NEGATIVE": len(pairs) - sum(labels),
            "EXPECTED_QUESTIONS": len(questions),
            "EXPECTED_TABLES": sum(1 for rows in tables.values() if rows),
            "EXPECTED_EMPTY": sum(1 for rows in tables.values() if not rows),
            "EXPECTED_STORED_PAIRS": sum(len(rows) for rows in tables.values()),
            "EXPECTED_WORDS": len(words),
            "EXPECTED_CONTROL": 3,
            "EXPECTED_DOCUMENTS": 2,
        },
    )


@pytest.fixture()
def toy(ev: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Toy:
    """The tiny run, with the guard's production pins replaced by the toy's counts."""
    built = _build_toy(tmp_path, ev)
    for name, value in built.expected.items():
        monkeypatch.setattr(ev, name, value)
    return built


def _sources(ev: Any, toy: Toy, **overrides: Any) -> Any:
    """The toy's resolved sources."""
    values = {
        "run_id": RUN_ID,
        "abstain_run": ABSTAIN_RUN,
        "stability_run": STABILITY_RUN,
    }
    values.update(overrides)
    return ev.resolve_sources(toy.config, **values)


def _context(ev: Any, toy: Toy) -> Any:
    """The toy's loaded context."""
    return ev.load_context(_sources(ev, toy))


# --- the arithmetic and the fold rebuild ------------------------------------


def test_the_fold_scheme_is_the_tracks(ev: Any) -> None:
    """Five folds over ten pairs, contiguous chunks of a seeded permutation."""
    assignment = ev.fold_ids(10, folds=ev.FOLDS, seed=ev.SEED)
    assert len(assignment) == 10
    assert sorted(np.bincount(assignment).tolist()) == [2, 2, 2, 2, 2]
    assert np.array_equal(assignment, ev.fold_ids(10, folds=ev.FOLDS, seed=ev.SEED))
    assert not np.array_equal(assignment, ev.fold_ids(10, folds=ev.FOLDS, seed=ev.SEED + 1))


def test_the_fold_rebuild_reproduces_a_known_pair(ev: Any, toy: Toy) -> None:
    """A non-floored pair's rebuilt score is the fold map's positive z sum, by hand."""
    context = _context(ev, toy)
    chosen = next(
        pair for pair in context.pairs if not ev.floored(context, context.pair_index[pair.pair_id])
    )
    index = context.pair_index[chosen.pair_id]
    fold_map = context.fold_maps[int(context.fold_ids[index])]
    wanted = set(ev.content_tokens(chosen.paragraph_text, ev.DEFAULT_STOPWORDS))
    expected = sum(
        row.z_score
        for row in fold_map.per_question.get(chosen.question_text, ())
        if row.word in wanted and row.z_score > 0
    )
    view = ev.pair_view(context, chosen.pair_id, with_moves=False)
    assert view.rebuilt_score == pytest.approx(expected, abs=1e-12)
    assert view.rebuilt_score == pytest.approx(view.artifact_score, abs=1e-12)
    assert view.reproduction_ok


def test_the_contributions_sum_to_the_score(ev: Any, toy: Toy) -> None:
    """The displayed contributions are the score's own decomposition."""
    context = _context(ev, toy)
    for pair in context.pairs:
        view = ev.pair_view(context, pair.pair_id, with_moves=False)
        if view.floored:
            assert view.contributions == ()
            continue
        assert sum(item.contribution for item in view.contributions) == pytest.approx(
            view.rebuilt_score, abs=1e-12
        )


def test_the_signed_words_keep_the_negative_side(ev: Any, toy: Toy) -> None:
    """The signed table is a superset of the positive-only decomposition."""
    context = _context(ev, toy)
    for pair in context.pairs:
        view = ev.pair_view(context, pair.pair_id, with_moves=False)
        if view.floored:
            assert view.signed_words == ()
            continue
        assert {word.word for word in view.contributions} <= {
            word.word for word in view.signed_words
        }
        assert all(word.contribution >= 0.0 for word in view.signed_words)
        assert sum(word.contribution for word in view.signed_words if word.summed) == pytest.approx(
            view.rebuilt_score, abs=1e-12
        )


@pytest.mark.skipif(
    not (Path("results") / "omct-analyze-voyage-dict-z" / "analysis.json").is_file(),
    reason="the real artifacts are not present",
)
def test_the_signed_table_shows_the_sign_reversal(ev: Any) -> None:
    """On AI-30-Q276, "within" is against the z arm and for the log-odds arm."""
    context = ev.load_context(ev.resolve_sources())
    z_view = ev.pair_view(context, "AI-30-Q276", arm="sum_zscore_positive", with_moves=False)
    within = next(word for word in z_view.signed_words if word.word == "within")
    assert within.z_score < 0.0
    assert within.log_odds > 0.0
    assert within.direction == "negative"
    assert within.contribution == pytest.approx(0.0)
    assert not within.summed
    odds_view = ev.pair_view(context, "AI-30-Q276", arm="sum_positive", with_moves=False)
    within_odds = next(word for word in odds_view.signed_words if word.word == "within")
    assert within_odds.direction == "positive"
    assert within_odds.summed
    assert within_odds.contribution == pytest.approx(within_odds.log_odds)


def test_the_fill_values_are_hand_computed(ev: Any) -> None:
    """CS-56's three fills on a known row (and on the empty row)."""
    seen = np.asarray([1.0, 3.0, 8.0])
    assert ev.empty_fill(seen, "floor") == pytest.approx(0.0)
    assert ev.empty_fill(seen, "neutral") == pytest.approx(3.0)
    assert ev.empty_fill(seen, "max") == pytest.approx(8.0)
    empty = np.asarray([])
    assert ev.empty_fill(empty, "floor") == pytest.approx(-1.0)
    assert ev.empty_fill(empty, "neutral") == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ev.empty_fill(seen, "nonsense")


def test_the_rank_moves_cover_the_three_fills(ev: Any, toy: Toy) -> None:
    """The pair view carries one move per fill; the floor move is the baseline."""
    context = _context(ev, toy)
    view = ev.pair_view(context, context.pairs[0].pair_id)
    assert [move.fill for move in view.moves] == list(ev.FILLS)
    assert view.moves[0].rank == view.rank
    assert view.moves[0].delta_rank == 0
    assert view.moves[0].delta_percentile == pytest.approx(0.0)


# --- the classes and the error taxonomy -------------------------------------


def test_the_class_boundaries_are_derived(ev: Any) -> None:
    """≤ 2 distinctive, 3–52 shared, ≥ 53 generic (209 labelled tables)."""
    assert ev.classify_word(1, 209) == "distinctive"
    assert ev.classify_word(2, 209) == "distinctive"
    assert ev.classify_word(3, 209) == "shared"
    assert ev.classify_word(52, 209) == "shared"
    assert ev.classify_word(53, 209) == "generic"
    assert ev.classify_word(209, 209) == "generic"


def test_the_error_taxonomy(ev: Any) -> None:
    """The evidence cut and the floored class, hand-computed."""
    assert ev.error_of(1, False, 0.5) == "true_positive"
    assert ev.error_of(1, False, 0.0) == "false_negative"
    assert ev.error_of(0, False, 0.5) == "false_positive"
    assert ev.error_of(0, False, 0.0) == "true_negative"
    assert ev.error_of(1, True, -1.0) == "floored"
    assert ev.error_of(0, True, -1.0) == "floored"


def test_the_question_view_counts_classes_and_folds(ev: Any, toy: Toy) -> None:
    """Q1's pair/positive counts, its classes and the five fold rows."""
    context = _context(ev, toy)
    view = ev.question_view(context, "Question one")
    assert (view.question_number, view.pairs, view.positives, view.negatives) == (1, 3, 2, 1)
    assert view.n_labelled_tables == 3
    assert view.distinctive_max_tables == 2
    assert view.generic_min_tables == 3
    assert sum(view.class_counts.values()) == len(view.words)
    # "common", "paragraph" and "alpha" sit in all three tables -> generic
    assert view.class_counts["generic"] == 3
    assert len(view.folds) == 5
    assert sum(fold.pairs for fold in view.folds) == 3
    assert view.table_size == len(view.words)


def test_the_stability_join_is_lazy_and_by_question(ev: Any, toy: Toy) -> None:
    """The 60 MB artifact is read only on demand; the join is by (question, word)."""
    context = _context(ev, toy)
    index = ev.load_stability_index(_sources(ev, toy))
    assert index.run_id == STABILITY_RUN
    assert index.resamples == 5
    view = ev.question_view(context, "Question one", stability_index=index.links)
    assert view.stability_loaded
    by_word = {row.word: row for row in view.stability}
    assert by_word["alpha"].present_share == pytest.approx(0.9)
    assert by_word["alpha"].survives_075
    assert not by_word["beta"].survives_075


def test_the_errors_and_the_control_are_never_pooled(ev: Any, toy: Toy) -> None:
    """The judged set is classified; the control cells are a separate, label-free strip."""
    context = _context(ev, toy)
    rows = ev.error_rows(context)
    assert len(rows) == len(PAIRS)
    assert sum(1 for row in rows if row.floored) >= 1  # Q4's pair is always floored
    control = ev.control_strip(context)
    assert len(control) == 3
    assert all(not hasattr(cell, "label") for cell in control)
    judged = {row.pair_id for row in rows}
    assert not ({cell.cell_id for cell in control} & judged)


# --- the compare view -------------------------------------------------------


def test_the_compare_views_are_right_minus_left(ev: Any, toy: Toy) -> None:
    """Two arms over the same ten pairs; the movement is the right minus the left."""
    context = _context(ev, toy)
    left = ev.run_view(context, "sum_positive")
    right = ev.run_view(context, "sum_zscore_positive")
    rows = ev.compare_views(left, right, context, context)
    assert len(rows) == len(PAIRS)
    assert {row.pair_id for row in rows} == {pair.pair_id for pair in context.pairs}
    for row in rows:
        assert row.delta_rank == row.right_rank - row.left_rank
        assert row.delta_percentile == pytest.approx(
            row.right_percentile - row.left_percentile, abs=1e-12
        )


# --- the guard --------------------------------------------------------------


def test_the_guard_passes_and_reproduces_the_artifact(ev: Any, toy: Toy) -> None:
    """The toy's guard is green and the rebuild is exact."""
    context = _context(ev, toy)
    report = ev.check_all(_sources(ev, toy), context=context)
    assert report.ok, report.lines()
    assert report.reproduction.n == len(PAIRS)
    assert report.reproduction.missing == 0
    assert report.reproduction.max_score_diff <= ev.SCORE_TOLERANCE
    assert report.reproduction.max_rank_diff <= ev.RANK_TOLERANCE


def test_the_guard_rejects_a_doctored_count(ev: Any, toy: Toy) -> None:
    """A map count that no longer matches the body of the artifact fails the guard."""
    payload = json.loads(toy.artifact.read_text(encoding="utf-8"))
    payload["dictionary"]["map"]["n_stored_pairs"] += 1
    toy.artifact.write_text(json.dumps(payload), encoding="utf-8")
    context = _context(ev, toy)
    report = ev.check_all(_sources(ev, toy), context=context)
    assert not report.ok
    assert any(check.name == "map size" and not check.ok for check in report.checks)


def test_the_guard_rejects_a_doctored_score(ev: Any, toy: Toy) -> None:
    """A judged score that the fold rebuild cannot reproduce blocks the contributions."""
    payload = json.loads(toy.artifact.read_text(encoding="utf-8"))
    arm = payload["models"][0]["arms"]["dictionary"]
    arm["judged_pairs"][0]["score"] += 0.1
    toy.artifact.write_text(json.dumps(payload), encoding="utf-8")
    context = _context(ev, toy)
    report = ev.check_all(_sources(ev, toy), context=context)
    assert not report.ok
    assert any(check.name == "fold rebuild" and not check.ok for check in report.checks)
    view = ev.pair_view(context, arm["judged_pairs"][0]["pair_id"], with_moves=False)
    assert not view.reproduction_ok
    assert view.contributions == ()


def test_the_cli_check_exits_zero_then_one(
    ev: Any, toy: Toy, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--check` exits 0 on the toy, then 1 after the count is doctored."""
    argv = ["--config", str(toy.config), "--run", RUN_ID, "--abstain-run", ABSTAIN_RUN]
    assert ev.main([*argv, "--check"]) == 0
    assert "RESULT: OK" in capsys.readouterr().out
    payload = json.loads(toy.artifact.read_text(encoding="utf-8"))
    payload["dictionary"]["map"]["n_questions"] += 1
    toy.artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert ev.main([*argv, "--check"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


# --- purity -----------------------------------------------------------------


def test_the_data_layer_has_no_ui_or_model_imports() -> None:
    """The data layer imports no Streamlit, no model client and no network client."""
    tree = ast.parse((TOOLS / "omct_error_view.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("streamlit", "torch", "sentence_transformers", "httpx", "sqlite3"):
        assert forbidden not in imported
    assert "numpy" in imported


# --- the word-pair runs (CS-59 / CS-60) -------------------------------------


WPAIRS = Path("results") / "omct-analyze-voyage-wpairs" / "analysis.json"
WPAIRS_PARAGRAPH = Path("results") / "omct-analyze-voyage-wpairs-paragraph" / "analysis.json"


@pytest.mark.skipif(not WPAIRS.is_file(), reason="the CS-59 artifact is not present")
def test_the_word_pair_guard_reproduces_the_stored_aucs(ev: Any) -> None:
    """Both word-pair runs pass the guard; the stored AUCs are reproduced exactly."""
    for run in (ev.WPAIRS_RUN, ev.WPAIRS_PARAGRAPH_RUN):
        sources = ev.resolve_sources(ev.DEFAULT_CONFIG, run_id=run)
        context = ev.load_any_context(sources)
        assert isinstance(context, ev.WordPairContext)
        report = ev.check_all(sources, context=context)
        assert report.ok, report.lines()
        assert report.word_pair is not None
        assert report.word_pair.table_ok
        assert report.word_pair.floor_ok
        assert report.word_pair.aucs
        for auc in report.word_pair.aucs:
            assert auc.abs_diff <= ev.SCORE_TOLERANCE


@pytest.mark.skipif(not WPAIRS.is_file(), reason="the CS-59 artifact is not present")
def test_the_word_pair_pair_view_and_reference(ev: Any) -> None:
    """A held-out-question pair shows its shared words and the reference arm."""
    sources = ev.resolve_sources(ev.DEFAULT_CONFIG, run_id=ev.WPAIRS_RUN)
    context = ev.load_any_context(sources)
    assert isinstance(context, ev.WordPairContext)
    view = ev.word_pair_pair_view(context, "AI-11-Q70")
    assert view.held_out == "question"
    assert view.contributions
    assert any(row.in_fold for row in view.contributions)
    assert sum(row.contribution for row in view.contributions) == pytest.approx(
        view.scores[view.primary_fill], abs=1e-12
    )
    assert view.reference_score == pytest.approx(4.406361730316086, abs=1e-12)
    question = ev.word_pair_question_view(context, context.questions[0])
    assert question.pairs > 0
    assert 0.0 <= question.coverage <= 1.0
    assert ev.word_pair_comparisons(context)
    errors = ev.word_pair_error_rows(context)
    assert len(errors) == len(context.pairs)
    assert sum(1 for row in errors if row.floored) == 76


@pytest.mark.skipif(not WPAIRS.is_file(), reason="the CS-59 artifact is not present")
def test_the_word_pair_guard_rejects_a_doctored_auc(ev: Any, tmp_path: Path) -> None:
    """A stored AUC that the rebuild does not match fails the guard."""
    payload = json.loads(WPAIRS.read_text(encoding="utf-8"))
    model = next(entry for entry in payload["models"] if entry.get("kind") == "dictionary")
    block = model["readout"]["word_pairs"]["held_out_questions"]
    block["within_auc"] = float(block["within_auc"]) + 0.01
    artifact = tmp_path / ev.WPAIRS_RUN / "analysis.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    sources = ev.resolve_sources(ev.DEFAULT_CONFIG, results=tmp_path, run_id=ev.WPAIRS_RUN)
    context = ev.load_word_pair_context(sources)
    report = ev.check_all(sources, context=context)
    assert not report.ok
    assert any(check.name == "word-pair AUC" and not check.ok for check in report.checks)


# --- the optional real-artifact app smoke test ------------------------------


@pytest.mark.skipif(
    not (Path("results") / "omct-analyze-voyage-dict-z" / "analysis.json").is_file(),
    reason="the real artifacts are not present",
)
def test_the_app_renders_the_four_views() -> None:
    """A real-artifact AppTest smoke test (skipped when results/ is absent)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    os.environ["OMCT_VIEW_CONFIG"] = str(Path("configs/base.yaml").resolve())
    os.environ["OMCT_VIEW_RESULTS"] = str(Path("results").resolve())
    os.environ["OMCT_VIEW_RUN"] = "omct-analyze-voyage-dict-z"
    app = AppTest.from_file(str(TOOLS / "omct_error_app.py"), default_timeout=180)
    app.run()
    assert not list(app.exception)
    for view in ("Errors", "One question", "Compare arms", "One pair"):
        app.sidebar.radio[0].set_value(view).run()
        assert not list(app.exception), view


@pytest.mark.skipif(not WPAIRS.is_file(), reason="the CS-59 artifact is not present")
def test_the_app_renders_the_word_pair_views() -> None:
    """A real-artifact AppTest smoke test for the CS-59 run (skipped when absent)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    os.environ["OMCT_VIEW_CONFIG"] = str(Path("configs/base.yaml").resolve())
    os.environ["OMCT_VIEW_RESULTS"] = str(Path("results").resolve())
    os.environ["OMCT_VIEW_RUN"] = "omct-analyze-voyage-wpairs"
    app = AppTest.from_file(str(TOOLS / "omct_error_app.py"), default_timeout=240)
    app.run()
    assert not list(app.exception)
    for view in ("Errors", "One question", "Compare arms", "One pair"):
        app.sidebar.radio[0].set_value(view).run()
        assert not list(app.exception), view
