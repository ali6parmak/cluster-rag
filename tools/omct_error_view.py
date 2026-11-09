"""The OMCT dictionary error viewer's **data layer** (CS-58).

The round CS-48 … CS-57 left ten ``results/omct-analyze-*/analysis.json``
artifacts that say *how well* the dictionary ranks the 1,520 reviewed pairs.
This layer answers the next question: *which pair does it get wrong, with which
words, on which question, and how does the empty-table fill move its rank?*

It reads, and only reads:

- ``data/external/omct_pairs.json`` (the reviewed pairs; subset A);
- one ``omct-analyze-*`` run artifact (its full map, its judged pairs and its
  control cells);
- the abstain run's ``readout.abstention_arms`` for the fill guard;
- the stability run's ``dictionary.stability.links`` (lazily, by the app).

**The trap.** ``dictionary.tables`` is the **full** map, while a judged pair's
stored score is **out-of-fold**: five folds over the pairs, each fold's map
built from the other folds. The layer rebuilds those five maps with the round's
own pure functions (``build_question_map`` and ``score_grid``) and the fold
scheme of ``commands/omct.py``'s ``_dictionary_fold_ids``. It reproduces every
judged score and every rank percentile at 1e-12; a mismatch is a guard failure,
never smoothed over. The recomputation is for **attribution and the guard
only**: it is labelled as such and never becomes a new artifact or metric.

The 4,000 unjudged control cells (subset B) are exposed by
:func:`control_strip` only. They carry no label, are never classified as
errors, and are never pooled with the judged set.

Pure: numpy + the standard library + the track's pure cores. No model client,
no Streamlit, no network, no write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cluster_rag.config import AppConfig, load_config
from cluster_rag.pairs import PairRecord, load_pairs
from cluster_rag.text_processors.control import percentiles_from_positions, rank_positions
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.features.phrase_map import phrase_units
from cluster_rag.text_processors.features.question_map import QuestionMap, build_question_map
from cluster_rag.text_processors.lexical.question_map_score import score_grid
from cluster_rag.text_processors.lexical.word_class import (
    DISTINCTIVE_MAX_TABLES,
    WORD_CLASSES,
    classify_word,
    generic_threshold,
)

# The tool's own configuration surface (every one has a CLI flag or an app
# control; nothing is hard-coded in the logic).
DEFAULT_CONFIG = Path("configs/base.yaml")
DEFAULT_RUN = "omct-analyze-voyage-dict-z"
RUN_CHOICES: tuple[str, ...] = (
    "omct-analyze-voyage-dict-z",
    "omct-analyze-voyage-dict",
    "omct-analyze-dictmap",
    "omct-analyze-voyage-dict-phrases",
    "omct-analyze-voyage-dict-stability",
    "omct-analyze-voyage-dict-lexicon",
    "omct-analyze-voyage-dict-abstain",
    "omct-analyze-voyage-dict-qpairs",
)
ABSTAIN_RUN = "omct-analyze-voyage-dict-abstain"
STABILITY_RUN = "omct-analyze-voyage-dict-stability"

# The round's pinned unit and guard numbers (CS-48 … CS-57).
SEED = 42
FOLDS = 5
EXPECTED_PAIRS = 1520
EXPECTED_POSITIVE = 524
EXPECTED_NEGATIVE = 996
EXPECTED_QUESTIONS = 294
EXPECTED_TABLES = 209
EXPECTED_EMPTY = 85
EXPECTED_STORED_PAIRS = 27979
EXPECTED_WORDS = 2082
EXPECTED_CONTROL = 4000
EXPECTED_DOCUMENTS = 3
EXPECTED_FLOOR_PAIRS = 518
EXPECTED_FLOOR_NEGATIVE_SHARE = 0.7780
EXPECTED_NON_FLOOR = 1002
FLOOR_NEGATIVE_SHARE_TOLERANCE = 5e-5

# The rebuild's tolerance: exact, because the same pure functions and the same
# sorted token order are used.
SCORE_TOLERANCE = 1e-12
RANK_TOLERANCE = 1e-12

# CS-56's empty-table fills and the additive arms this viewer offers.
FILLS: tuple[str, ...] = ("floor", "neutral", "max")
ARM_CHOICES: tuple[str, ...] = (
    "sum_positive",
    "sum_zscore_positive",
    "support_count",
)

# The evidence cut the Errors view pre-registers: score > 0 means the map gave
# the pair points. It is a display rule, not the run's metric (the run reports
# AUCs, not an F1 cut).
ERROR_CLASSES: tuple[str, ...] = (
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "floored",
)


class MapOnlyRunError(ValueError):
    """The selected run carries no dictionary scorer (CS-48's map-only run)."""


# --- the sources ------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """The files the viewer reads, resolved.

    Attributes:
        config_path: The config file the paths came from.
        config: The resolved app config.
        data_external: The reviewed-pairs JSON (read-only).
        results: The run-artifact folder.
        run_id: The selected run.
        abstain_run: The fill comparison's run.
        stability_run: The per-link stability run.
    """

    config_path: Path
    config: AppConfig
    data_external: Path
    results: Path
    run_id: str
    abstain_run: str
    stability_run: str


@dataclass(frozen=True)
class SourcePair:
    """One reviewed pair from the source file."""

    pair_id: str
    document: str
    paragraph_number: str
    paragraph_text: str
    question_number: int
    question_text: str
    label: int


@dataclass(frozen=True)
class JudgedPair:
    """One row of the artifact's ``arms.dictionary.judged_pairs``."""

    pair_id: str
    cell_id: str
    document: str
    paragraph_number: str
    question_number: int
    label: int
    score: float
    rank_percentile: float


@dataclass(frozen=True)
class ControlCell:
    """One unjudged cell of the artifact's control set (subset B)."""

    cell_id: str
    document: str
    paragraph_number: str
    question_number: int
    score: float
    rank_percentile: float


@dataclass(frozen=True)
class RunArtifact:
    """The selected run's dictionary blocks.

    Attributes:
        run_id: The artifact's own run id.
        seed: The run's seed.
        n_valid_pairs: The source's valid-pair count, as the artifact records it.
        dictionary: The artifact's ``dictionary`` block.
        model: The ``kind == "dictionary"`` model entry (``None`` = map-only).
        judged: The judged-pair rows, in the artifact's order.
        control: The control cells, in the artifact's order.
        scoring: The arm the judged rows were scored with.
        unit_name: ``words`` or ``phrases`` (the artifact's map unit).
        readout: The model entry's ``readout`` block.
    """

    run_id: str
    seed: int
    n_valid_pairs: int
    dictionary: dict[str, Any]
    model: dict[str, Any] | None
    judged: tuple[JudgedPair, ...]
    control: tuple[ControlCell, ...]
    scoring: str
    unit_name: str
    readout: dict[str, Any]
    control_has_label: bool = False

    @property
    def arms(self) -> tuple[str, ...]:
        """The arms the artifact's readout measured, in the viewer's order."""
        measured = {str(name) for name in (self.readout.get("scorings") or {})}
        ordered = tuple(arm for arm in ARM_CHOICES if arm in measured)
        return ordered or ((self.scoring,) if self.scoring else ())


@dataclass(frozen=True)
class Provenance:
    """Every source, named, with its sha256."""

    config_path: str
    data_path: str
    data_sha256: str
    run_id: str
    run_path: str
    run_sha256: str
    seed: int
    folds: int
    unit: str
    scoring: str
    n_pairs: int
    n_judged: int
    n_control: int
    n_documents: int

    def lines(self) -> list[str]:
        """The provenance block, one labelled line per source."""
        return [
            f"config: {self.config_path}",
            f"source: {self.data_path} sha256 {self.data_sha256}",
            f"run: {self.run_path} sha256 {self.run_sha256}",
            f"run id: {self.run_id}; seed {self.seed}; folds {self.folds}; "
            f"unit {self.unit}; primary arm {self.scoring}",
            f"pairs: {self.n_pairs} reviewed; {self.n_judged} judged; "
            f"{self.n_control} unjudged control cells (subset B, never ground truth); "
            f"{self.n_documents} documents",
            "note: OMCT auxiliary data only; never a UPR metric; results/ read-only.",
        ]


@dataclass(frozen=True)
class GridSet:
    """The full, per-fold and assembled grids of one (arm, fill)."""

    scoring: str
    fill: str
    full: np.ndarray
    folds: Mapping[int, np.ndarray]
    assembled: np.ndarray


@dataclass
class Context:
    """Everything read or rebuilt for the selected run.

    The ``cache`` holds the lazily built ``(arm, fill)`` grids; it is a cache of
    deterministic recomputations, never a hidden input.
    """

    sources: Sources
    metadata: dict[str, Any]
    pairs: tuple[SourcePair, ...]
    run: RunArtifact
    fold_ids: np.ndarray
    paragraphs: tuple[str, ...]
    questions: tuple[str, ...]
    paragraph_row: dict[str, int]
    question_col: dict[str, int]
    pair_index: dict[str, int]
    rows: np.ndarray
    columns: np.ndarray
    labels: np.ndarray
    candidate_columns: tuple[tuple[int, ...], ...]
    unit_name: str
    units: Any
    full_map: QuestionMap
    fold_maps: dict[int, QuestionMap]
    primary: GridSet
    cache: dict[tuple[str, str], GridSet] = field(default_factory=dict)

    @property
    def scoring(self) -> str:
        """The run's primary arm."""
        return self.run.scoring


def sha256_file(path: Path) -> str:
    """One file's sha256.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no file at {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """One JSON artifact.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no artifact at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_sources(
    config_path: Path = DEFAULT_CONFIG,
    *,
    results: Path | None = None,
    run_id: str = DEFAULT_RUN,
    abstain_run: str = ABSTAIN_RUN,
    stability_run: str = STABILITY_RUN,
) -> Sources:
    """Resolve the paths from the config (and the overrides)."""
    config = load_config(config_path)
    if results is not None:
        results_path = Path(results)
    elif config.paths.results:
        results_path = Path(config.paths.results)
    else:
        results_path = Path("results")
    return Sources(
        config_path=Path(config_path),
        config=config,
        data_external=Path(config.paths.data_external),
        results=results_path,
        run_id=run_id,
        abstain_run=abstain_run,
        stability_run=stability_run,
    )


def _source_pairs(records: Sequence[PairRecord]) -> tuple[SourcePair, ...]:
    """The loader's records as plain frozen rows."""
    return tuple(
        SourcePair(
            pair_id=record.pair_id,
            document=record.document,
            paragraph_number=record.paragraph_number,
            paragraph_text=record.paragraph_text,
            question_number=record.question_number,
            question_text=record.question_text,
            label=record.label,
        )
        for record in records
    )


def load_run(sources: Sources, run_id: str | None = None) -> RunArtifact:
    """Read one run artifact's dictionary blocks.

    Raises:
        FileNotFoundError: If the artifact is missing.
        MapOnlyRunError: If the run carries no dictionary scorer.
    """
    selected = run_id or sources.run_id
    path = sources.results / selected / "analysis.json"
    payload = read_json(path)
    dictionary = payload.get("dictionary")
    if not isinstance(dictionary, Mapping):
        raise ValueError(f"{path} carries no top-level 'dictionary' block")
    model = next(
        (entry for entry in payload.get("models") or [] if entry.get("kind") == "dictionary"),
        None,
    )
    readout = dict(model.get("readout") or {}) if isinstance(model, Mapping) else {}
    arm = dict(((model or {}).get("arms") or {}).get("dictionary") or {})
    judged = tuple(
        JudgedPair(
            pair_id=str(row.get("pair_id")),
            cell_id=str(row.get("cell_id")),
            document=str(row.get("document")),
            paragraph_number=str(row.get("paragraph_number")),
            question_number=int(row.get("question_number") or 0),
            label=int(row.get("label") or 0),
            score=float(row.get("score") or 0.0),
            rank_percentile=float(row.get("rank_percentile") or 0.0),
        )
        for row in arm.get("judged_pairs") or []
    )
    control = tuple(
        ControlCell(
            cell_id=str(row.get("cell_id")),
            document=str(row.get("document")),
            paragraph_number=str(row.get("paragraph_number")),
            question_number=int(row.get("question_number") or 0),
            score=float(row.get("score") or 0.0),
            rank_percentile=float(row.get("rank_percentile") or 0.0),
        )
        for row in arm.get("control_cells") or []
    )
    if not judged:
        raise MapOnlyRunError(
            f"{selected!r} is the map-only run (CS-48): it carries no "
            "models[kind='dictionary'] judged pairs. Select a scorer run."
        )
    unit_name = str((dictionary.get("map") or {}).get("units") or "words")
    scoring = str(arm.get("scoring") or readout.get("primary") or "")
    control_has_label = any("label" in row for row in arm.get("control_cells") or [])
    return RunArtifact(
        run_id=str(payload.get("run_id") or selected),
        seed=int(payload.get("seed") or SEED),
        n_valid_pairs=int(payload.get("n_valid_pairs") or 0),
        dictionary=dict(dictionary),
        model=dict(model) if isinstance(model, Mapping) else None,
        judged=judged,
        control=control,
        scoring=scoring,
        unit_name=unit_name,
        readout=readout,
        control_has_label=control_has_label,
    )


def fold_ids(n_pairs: int, *, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """The cross-fit's fold assignment (the track's scheme, copied exactly).

    ``np.random.default_rng(seed).permutation(n_pairs)`` split into ``folds``
    contiguous chunks. The viewer must use the same assignment the artifacts
    used, so the implementation is the command's own, line for line.
    """
    order = np.random.default_rng(seed).permutation(n_pairs)
    chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(order, folds)]
    assignment = np.empty(n_pairs, dtype=np.int64)
    for fold, chunk in enumerate(chunks):
        assignment[chunk] = fold
    return assignment


def _candidate_columns(
    pairs: Sequence[SourcePair],
    paragraph_row: Mapping[str, int],
    question_col: Mapping[str, int],
    n_paragraphs: int,
) -> tuple[tuple[int, ...], ...]:
    """Each paragraph's same-document candidate questions (the run's scope)."""
    by_document: dict[str, set[int]] = {}
    paragraph_documents: dict[str, set[str]] = {}
    for pair in pairs:
        by_document.setdefault(pair.document, set()).add(question_col[pair.question_text])
        paragraph_documents.setdefault(pair.paragraph_text, set()).add(pair.document)
    text_by_row = {row: text for text, row in paragraph_row.items()}
    candidates: list[tuple[int, ...]] = []
    for row in range(n_paragraphs):
        columns: set[int] = set()
        for document in paragraph_documents.get(text_by_row.get(row, ""), set()):
            columns.update(by_document.get(document, set()))
        candidates.append(tuple(sorted(columns)))
    return tuple(candidates)


def load_context(sources: Sources) -> Context:
    """Read the source and the run, rebuild the five fold maps and the grids.

    Raises:
        FileNotFoundError: If a source is missing.
        MapOnlyRunError: If the selected run has no dictionary scorer.
        ValueError: If the source and the artifact cannot be aligned.
    """
    report = load_pairs(sources.data_external)
    pairs = _source_pairs(report.records)
    run = load_run(sources)
    n_pairs = len(pairs)
    paragraph_texts = [pair.paragraph_text for pair in pairs]
    question_texts = [pair.question_text for pair in pairs]
    labels = [pair.label for pair in pairs]
    paragraphs = tuple(dict.fromkeys(paragraph_texts))
    questions = tuple(dict.fromkeys(question_texts))
    paragraph_row = {text: index for index, text in enumerate(paragraphs)}
    question_col = {text: index for index, text in enumerate(questions)}
    pair_index = {pair.pair_id: index for index, pair in enumerate(pairs)}
    rows = np.asarray([paragraph_row[text] for text in paragraph_texts], dtype=np.int64)
    columns = np.asarray([question_col[text] for text in question_texts], dtype=np.int64)
    assignment = fold_ids(n_pairs, folds=FOLDS, seed=run.seed)
    units = phrase_units if run.unit_name == "phrases" else None
    full_map = build_question_map(paragraph_texts, question_texts, labels, units=units)
    fold_maps: dict[int, QuestionMap] = {}
    for fold in range(FOLDS):
        keep = [index for index in range(n_pairs) if int(assignment[index]) != fold]
        fold_maps[fold] = build_question_map(
            [paragraph_texts[index] for index in keep],
            [question_texts[index] for index in keep],
            [labels[index] for index in keep],
            units=units,
        )
    primary = _build_grid_set(
        scoring=run.scoring,
        fill="floor",
        paragraphs=paragraphs,
        questions=questions,
        full_map=full_map,
        fold_maps=fold_maps,
        rows=rows,
        columns=columns,
        fold_ids=assignment,
        units=units,
    )
    return Context(
        sources=sources,
        metadata=dict(report.metadata),
        pairs=pairs,
        run=run,
        fold_ids=assignment,
        paragraphs=paragraphs,
        questions=questions,
        paragraph_row=paragraph_row,
        question_col=question_col,
        pair_index=pair_index,
        rows=rows,
        columns=columns,
        labels=np.asarray(labels, dtype=np.int64),
        candidate_columns=_candidate_columns(pairs, paragraph_row, question_col, len(paragraphs)),
        unit_name=run.unit_name,
        units=units,
        full_map=full_map,
        fold_maps=fold_maps,
        primary=primary,
    )


def _build_grid_set(
    *,
    scoring: str,
    fill: str,
    paragraphs: Sequence[str],
    questions: Sequence[str],
    full_map: QuestionMap,
    fold_maps: Mapping[int, QuestionMap],
    rows: np.ndarray,
    columns: np.ndarray,
    fold_ids: np.ndarray,
    units: Any,
) -> GridSet:
    """Score the full and per-fold grids, then assemble the judged cells."""
    full = score_grid(paragraphs, questions, full_map, scoring=scoring, units=units, fill=fill)
    folds = {
        fold: score_grid(
            paragraphs, questions, question_map, scoring=scoring, units=units, fill=fill
        )
        for fold, question_map in fold_maps.items()
    }
    assembled = full.copy()
    for index in range(len(fold_ids)):
        assembled[rows[index], columns[index]] = folds[int(fold_ids[index])][
            rows[index], columns[index]
        ]
    return GridSet(scoring=scoring, fill=fill, full=full, folds=folds, assembled=assembled)


def grids_for(context: Context, arm: str, fill: str = "floor") -> GridSet:
    """The ``(arm, fill)`` grids, built once and cached on the context.

    Raises:
        ValueError: On an unknown arm or fill.
    """
    if arm not in ARM_CHOICES:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_CHOICES}")
    if fill not in FILLS:
        raise ValueError(f"unknown fill {fill!r}; expected one of {FILLS}")
    key = (arm, fill)
    if key not in context.cache:
        if key == (context.run.scoring, "floor"):
            context.cache[key] = context.primary
        else:
            context.cache[key] = _build_grid_set(
                scoring=arm,
                fill=fill,
                paragraphs=context.paragraphs,
                questions=context.questions,
                full_map=context.full_map,
                fold_maps=context.fold_maps,
                rows=context.rows,
                columns=context.columns,
                fold_ids=context.fold_ids,
                units=context.units,
            )
    return context.cache[key]


def build_provenance(context: Context) -> Provenance:
    """The provenance block from the loaded context."""
    sources = context.sources
    run_path = sources.results / sources.run_id / "analysis.json"
    corpus = context.run.dictionary.get("corpus") or {}
    map_block = context.run.dictionary.get("map") or {}
    return Provenance(
        config_path=str(sources.config_path),
        data_path=str(sources.data_external),
        data_sha256=sha256_file(sources.data_external),
        run_id=context.run.run_id,
        run_path=str(run_path),
        run_sha256=sha256_file(run_path),
        seed=context.run.seed,
        folds=int((context.run.dictionary.get("cross_fit") or {}).get("folds") or 0),
        unit=context.unit_name,
        scoring=context.run.scoring,
        n_pairs=len(context.pairs),
        n_judged=len(context.run.judged),
        n_control=len(context.run.control),
        n_documents=int(corpus.get("n_documents") or map_block.get("n_documents") or 0),
    )


# --- the pair arithmetic ----------------------------------------------------


def floored(context: Context, index: int) -> bool:
    """Whether the pair's question has no table in its own fold's map."""
    question = context.pairs[index].question_text
    return not context.fold_maps[int(context.fold_ids[index])].per_question.get(question)


def fold_counts(context: Context, index: int) -> tuple[int, int]:
    """The question's positives and pair count in the pair's own fold map."""
    question = context.pairs[index].question_text
    fold_map = context.fold_maps[int(context.fold_ids[index])]
    return (
        int(fold_map.question_positive_counts.get(question, 0)),
        int(fold_map.question_counts.get(question, 0)),
    )


def rank_in_candidates(
    context: Context,
    scores: np.ndarray,
    index: int,
) -> tuple[int, int, float]:
    """One pair's 1-based rank, candidate count and percentile in its paragraph.

    The candidate list is the paragraph's **same-document** questions (the
    run's own scope); ties keep the candidates' own order (a stable sort).
    """
    row = int(context.rows[index])
    column = int(context.columns[index])
    candidates = context.candidate_columns[row]
    order = sorted(candidates, key=lambda col: -float(scores[row, col]))
    position = order.index(column)
    return position + 1, len(candidates), position / len(candidates)


def empty_fill(seen: np.ndarray, fill: str) -> float:
    """The CS-56 fill for a row whose seen cells are ``seen``.

    ``floor`` is the smallest seen score minus one (the "impossible" rule);
    ``neutral`` is the median of the seen scores ("no evidence"); ``max`` is the
    largest ("possible"). An empty row fills with 0.0 for ``neutral`` and -1.0
    otherwise, the scorer's own convention.

    Raises:
        ValueError: On an unknown fill.
    """
    values = np.asarray(seen, dtype=np.float64)
    if fill == "neutral":
        return float(np.median(values)) if values.size else 0.0
    if fill == "max":
        return float(values.max()) if values.size else -1.0
    if fill == "floor":
        return float(values.min() - 1.0) if values.size else -1.0
    raise ValueError(f"unknown fill {fill!r}; expected one of {FILLS}")


@dataclass(frozen=True)
class Contribution:
    """One word's contribution to a judged pair's out-of-fold score."""

    word: str
    n_with: int
    df: int
    log_odds: float
    z_score: float
    contribution: float


@dataclass(frozen=True)
class RankMove:
    """One pair's score and rank under one fill."""

    fill: str
    score: float
    rank: int
    n_candidates: int
    delta_rank: int
    delta_percentile: float


@dataclass(frozen=True)
class PairView:
    """Everything the One pair view shows."""

    pair: SourcePair
    artifact_score: float
    rebuilt_score: float
    reproduction_ok: bool
    arm: str
    fill: str
    score: float
    rank: int
    n_candidates: int
    rank_percentile: float
    floored: bool
    fold: int
    fold_positive: int
    fold_pairs: int
    contributions: tuple[Contribution, ...]
    moves: tuple[RankMove, ...]


def _units_of(context: Context, text: str) -> set[str]:
    """The unit set of a paragraph under the run's unit."""
    units = context.units or (lambda value: content_tokens(value, DEFAULT_STOPWORDS))
    return set(units(text))


def contributions_for(context: Context, index: int, arm: str) -> tuple[Contribution, ...]:
    """The fold-map rows that gave the pair points under ``arm``.

    Only the additive arms are offered. ``sum_positive`` keeps the positive
    log-odds, ``sum_zscore_positive`` the positive z-scores, and
    ``support_count`` one point per present table unit. A floored pair has no
    table and therefore no contributions.
    """
    if floored(context, index):
        return ()
    question = context.pairs[index].question_text
    fold_map = context.fold_maps[int(context.fold_ids[index])]
    wanted = _units_of(context, context.pairs[index].paragraph_text)
    rows: list[Contribution] = []
    for row in fold_map.per_question.get(question, ()):
        if row.word not in wanted:
            continue
        if arm == "sum_positive":
            value = row.log_odds if row.log_odds > 0.0 else 0.0
        elif arm == "sum_zscore_positive":
            value = row.z_score if row.z_score > 0.0 else 0.0
        elif arm == "support_count":
            value = 1.0
        else:
            raise ValueError(f"arm {arm!r} has no additive contribution")
        if value <= 0.0:
            continue
        rows.append(
            Contribution(
                word=row.word,
                n_with=int(row.n_with),
                df=int(row.df),
                log_odds=float(row.log_odds),
                z_score=float(row.z_score),
                contribution=float(value),
            )
        )
    rows.sort(key=lambda contribution: (-contribution.contribution, contribution.word))
    return tuple(rows)


def pair_view(
    context: Context,
    pair_id: str,
    *,
    arm: str | None = None,
    fill: str = "floor",
    with_moves: bool = True,
) -> PairView:
    """The One pair view's row, rebuilt from the pair's own fold map."""
    if pair_id not in context.pair_index:
        raise KeyError(f"no reviewed pair {pair_id!r}")
    index = context.pair_index[pair_id]
    selected_arm = arm or context.run.scoring
    grids = grids_for(context, selected_arm, fill)
    row = int(context.rows[index])
    column = int(context.columns[index])
    fold = int(context.fold_ids[index])
    rebuilt = float(grids.folds[fold][row, column])
    artifact_score = next(
        (row.score for row in context.run.judged if row.pair_id == pair_id),
        float("nan"),
    )
    reproduction_ok = bool(abs(rebuilt - artifact_score) <= SCORE_TOLERANCE)
    rank, n_candidates, percentile = rank_in_candidates(context, grids.assembled, index)
    is_floored = floored(context, index)
    fold_positive, fold_pairs = fold_counts(context, index)
    contributions = contributions_for(context, index, selected_arm) if reproduction_ok else ()
    moves: list[RankMove] = []
    if with_moves:
        base_rank = rank
        base_percentile = percentile
        for candidate_fill in FILLS:
            other = grids_for(context, selected_arm, candidate_fill)
            other_rank, other_n, other_percentile = rank_in_candidates(
                context, other.assembled, index
            )
            moves.append(
                RankMove(
                    fill=candidate_fill,
                    score=float(other.assembled[row, column]),
                    rank=other_rank,
                    n_candidates=other_n,
                    delta_rank=other_rank - base_rank,
                    delta_percentile=other_percentile - base_percentile,
                )
            )
    return PairView(
        pair=context.pairs[index],
        artifact_score=artifact_score,
        rebuilt_score=rebuilt,
        reproduction_ok=reproduction_ok,
        arm=selected_arm,
        fill=fill,
        score=float(grids.assembled[row, column]),
        rank=rank,
        n_candidates=n_candidates,
        rank_percentile=percentile,
        floored=is_floored,
        fold=fold,
        fold_positive=fold_positive,
        fold_pairs=fold_pairs,
        contributions=contributions,
        moves=tuple(moves),
    )


# --- the three views' tables ------------------------------------------------


@dataclass(frozen=True)
class ErrorRow:
    """One judged pair's error class under the evidence cut."""

    pair_id: str
    document: str
    paragraph_number: str
    question_number: int
    label: int
    score: float
    rank: int
    n_candidates: int
    rank_percentile: float
    floored: bool
    error: str


def error_of(label: int, is_floored: bool, score: float) -> str:
    """One pair's error class.

    A floored pair is its **own** class (the map is silent), never folded into
    the false negatives. Otherwise the pre-registered evidence cut decides:
    ``score > 0`` is the map giving the pair points.
    """
    if is_floored:
        return "floored"
    if label == 1:
        return "true_positive" if score > 0.0 else "false_negative"
    return "false_positive" if score > 0.0 else "true_negative"


def error_rows(
    context: Context,
    *,
    arm: str | None = None,
    fill: str = "floor",
) -> tuple[ErrorRow, ...]:
    """Every judged pair classified, in the artifact's order (control excluded)."""
    selected_arm = arm or context.run.scoring
    grids = grids_for(context, selected_arm, fill)
    positions = rank_positions(
        grids.assembled, context.rows, context.columns, context.candidate_columns
    )
    percentiles = percentiles_from_positions(positions, context.rows, context.candidate_columns)
    by_pair = {row.pair_id: row for row in context.run.judged}
    rows: list[ErrorRow] = []
    for index, pair in enumerate(context.pairs):
        judged = by_pair.get(pair.pair_id)
        if judged is None:
            continue
        is_floored = floored(context, index)
        rows.append(
            ErrorRow(
                pair_id=pair.pair_id,
                document=pair.document,
                paragraph_number=pair.paragraph_number,
                question_number=pair.question_number,
                label=pair.label,
                score=float(grids.assembled[int(context.rows[index]), int(context.columns[index])]),
                rank=int(positions[index]) + 1,
                n_candidates=len(context.candidate_columns[int(context.rows[index])]),
                rank_percentile=float(percentiles[index]),
                floored=is_floored,
                error=error_of(
                    pair.label,
                    is_floored,
                    float(grids.assembled[int(context.rows[index]), int(context.columns[index])]),
                ),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class QuestionWord:
    """One word of a question's full-map table, with its derived class."""

    word: str
    n_with: int
    df: int
    lift: float
    log_odds: float
    z_score: float
    standard_error: float
    tables: int
    word_class: str


@dataclass(frozen=True)
class QuestionFold:
    """One question's availability in one fold."""

    fold: int
    pairs: int
    positives: int
    table_size: int
    with_table: int
    with_evidence: int
    floored: int


@dataclass(frozen=True)
class StabilityLink:
    """One link's bootstrap stability (CS-54)."""

    word: str
    present_share: float
    sign_stability: float
    head_stability: float
    survives_075: bool
    survives_090: bool


@dataclass(frozen=True)
class QuestionView:
    """Everything the One question view shows."""

    question: str
    question_number: int
    pairs: int
    positives: int
    negatives: int
    table_size: int
    words: tuple[QuestionWord, ...]
    folds: tuple[QuestionFold, ...]
    class_counts: Mapping[str, int]
    n_labelled_tables: int
    distinctive_max_tables: int
    generic_min_tables: int
    stability: tuple[StabilityLink, ...]
    stability_loaded: bool


def question_view(
    context: Context,
    question: str,
    *,
    limit: int | None = None,
    stability_index: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> QuestionView:
    """The One question view's table, counts, availability and classes.

    Args:
        context: The loaded run.
        question: The question text.
        limit: How many of the table's words to return (``None`` = all).
        stability_index: The CS-54 artifact's links by question, when the
            stability run is selected. ``None`` means "not loaded".
    """
    dictionary = context.run.dictionary
    tables = dictionary.get("tables") or {}
    rows = list(tables.get(question) or [])
    words = dictionary.get("words") or {}
    n_labelled_tables = int((dictionary.get("map") or {}).get("n_questions_with_table") or 0)
    built: list[QuestionWord] = []
    for row in rows:
        word = str(row.get("word"))
        counts = words.get(word) or {}
        built.append(
            QuestionWord(
                word=word,
                n_with=int(row.get("n_with") or 0),
                df=int(row.get("df") or 0),
                lift=float(row.get("lift") or 0.0),
                log_odds=float(row.get("log_odds") or 0.0),
                z_score=float(row.get("z_score") or 0.0),
                standard_error=float(row.get("standard_error") or 0.0),
                tables=int(counts.get("tables") or 0),
                word_class=classify_word(int(counts.get("tables") or 0), n_labelled_tables)
                if int(counts.get("tables") or 0) > 0
                else "",
            )
        )
    built.sort(key=lambda word: (-word.z_score, word.word))
    if limit is not None:
        built = built[: max(1, limit)]
    class_counts: Counter[str] = Counter()
    for word in built:
        if word.word_class:
            class_counts[word.word_class] += 1
    question_pairs = [pair for pair in context.pairs if pair.question_text == question]
    pairs = len(question_pairs)
    positives = sum(1 for pair in question_pairs if pair.label == 1)
    availability = {
        str(row.get("pair_id")): row
        for row in ((dictionary.get("availability") or {}).get("per_pair") or [])
    }
    fold_rows: list[QuestionFold] = []
    for fold in range(FOLDS):
        in_fold = [
            index
            for index, pair in enumerate(context.pairs)
            if pair.question_text == question and int(context.fold_ids[index]) == fold
        ]
        table_size = len(context.fold_maps[fold].per_question.get(question, ()))
        with_table = sum(
            1
            for index in in_fold
            if bool((availability.get(context.pairs[index].pair_id) or {}).get("has_table"))
        )
        with_evidence = sum(
            int((availability.get(context.pairs[index].pair_id) or {}).get("n_evidence") or 0) > 0
            for index in in_fold
        )
        fold_rows.append(
            QuestionFold(
                fold=fold,
                pairs=len(in_fold),
                positives=sum(1 for index in in_fold if context.pairs[index].label == 1),
                table_size=table_size,
                with_table=with_table,
                with_evidence=with_evidence,
                floored=len(in_fold) - with_table,
            )
        )
    stability: list[StabilityLink] = []
    if stability_index is not None:
        for word in built:
            link = (stability_index.get(question) or {}).get(word.word)
            if link is None:
                continue
            stability.append(
                StabilityLink(
                    word=word.word,
                    present_share=float(link.get("present_share") or 0.0),
                    sign_stability=float(link.get("sign_stability") or 0.0),
                    head_stability=float(link.get("head_stability") or 0.0),
                    survives_075=float(link.get("sign_stability") or 0.0) >= 0.75,
                    survives_090=float(link.get("sign_stability") or 0.0) >= 0.90,
                )
            )
    number = next(
        (pair.question_number for pair in context.pairs if pair.question_text == question), 0
    )
    return QuestionView(
        question=question,
        question_number=number,
        pairs=pairs,
        positives=positives,
        negatives=pairs - positives,
        table_size=len(rows),
        words=tuple(built),
        folds=tuple(fold_rows),
        class_counts={name: class_counts.get(name, 0) for name in WORD_CLASSES},
        n_labelled_tables=n_labelled_tables,
        distinctive_max_tables=DISTINCTIVE_MAX_TABLES,
        generic_min_tables=generic_threshold(n_labelled_tables),
        stability=tuple(stability),
        stability_loaded=stability_index is not None,
    )


@dataclass(frozen=True)
class StabilityIndex:
    """The CS-54 links, indexed by question and word (lazy source)."""

    run_id: str
    resamples: int
    seed: int
    links: Mapping[str, Mapping[str, Mapping[str, Any]]]


def load_stability_index(sources: Sources) -> StabilityIndex:
    """Read the stability run's 60 MB artifact and index its links.

    This is the only lazy source: the app calls it when the One question view
    opens with the stability run selected.

    Raises:
        FileNotFoundError: If the artifact is missing.
        ValueError: If it carries no ``dictionary.stability.links``.
    """
    payload = read_json(sources.results / sources.stability_run / "analysis.json")
    stability = (payload.get("dictionary") or {}).get("stability")
    if not isinstance(stability, Mapping):
        raise ValueError(f"{sources.stability_run!r} carries no dictionary.stability block")
    links = stability.get("links") or []
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in links:
        question = str(row.get("question"))
        word = str(row.get("word"))
        indexed.setdefault(question, {})[word] = row
    return StabilityIndex(
        run_id=str(payload.get("run_id") or sources.stability_run),
        resamples=int(stability.get("resamples") or 0),
        seed=int(stability.get("seed") or SEED),
        links=indexed,
    )


def control_strip(context: Context) -> tuple[ControlCell, ...]:
    """The 4,000 unjudged control cells, as **data**, with no label and no class.

    The caller must keep them separate from every error list: they are subset
    B, they were never reviewed, and they are never ground truth.
    """
    return context.run.control


# --- the compare view -------------------------------------------------------


@dataclass(frozen=True)
class RunView:
    """One (run, arm) series over the same 1,520 pairs."""

    run_id: str
    arm: str
    unit: str
    scores: np.ndarray
    ranks: np.ndarray
    percentiles: np.ndarray
    floored: np.ndarray
    metrics: Mapping[str, Any] | None


def run_view(context: Context, arm: str, fill: str = "floor") -> RunView:
    """One run's per-pair scores, ranks and floored flags for one arm."""
    grids = grids_for(context, arm, fill)
    positions = rank_positions(
        grids.assembled, context.rows, context.columns, context.candidate_columns
    )
    percentiles = percentiles_from_positions(positions, context.rows, context.candidate_columns)
    metrics = (context.run.readout.get("scorings") or {}).get(arm)
    return RunView(
        run_id=context.run.run_id,
        arm=arm,
        unit=context.unit_name,
        scores=np.asarray(
            [
                grids.assembled[int(context.rows[i]), int(context.columns[i])]
                for i in range(len(context.pairs))
            ]
        ),
        ranks=np.asarray(positions, dtype=np.int64) + 1,
        percentiles=np.asarray(percentiles, dtype=np.float64),
        floored=np.asarray([floored(context, i) for i in range(len(context.pairs))], dtype=bool),
        metrics=dict(metrics) if isinstance(metrics, Mapping) else None,
    )


@dataclass(frozen=True)
class CompareRow:
    """One pair under two (run, arm) series."""

    pair_id: str
    document: str
    label: int
    left_score: float
    right_score: float
    left_rank: int
    right_rank: int
    left_percentile: float
    right_percentile: float
    delta_rank: int
    delta_percentile: float


def compare_views(
    left: RunView,
    right: RunView,
    context_left: Context,
    context_right: Context,
) -> tuple[CompareRow, ...]:
    """Join two run views on ``pair_id`` and report the rank movement.

    ``delta`` is **right minus left**: a positive ``delta_rank`` means the pair
    sits lower (a worse rank) under the right series.
    """
    right_by_pair = {pair.pair_id: index for index, pair in enumerate(context_right.pairs)}
    rows: list[CompareRow] = []
    for index, pair in enumerate(context_left.pairs):
        other = right_by_pair.get(pair.pair_id)
        if other is None:
            continue
        rows.append(
            CompareRow(
                pair_id=pair.pair_id,
                document=pair.document,
                label=pair.label,
                left_score=float(left.scores[index]),
                right_score=float(right.scores[other]),
                left_rank=int(left.ranks[index]),
                right_rank=int(right.ranks[other]),
                left_percentile=float(left.percentiles[index]),
                right_percentile=float(right.percentiles[other]),
                delta_rank=int(right.ranks[other]) - int(left.ranks[index]),
                delta_percentile=float(right.percentiles[other]) - float(left.percentiles[index]),
            )
        )
    return tuple(rows)


# --- the guard --------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One guard check: what it verified and whether it held."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Reproduction:
    """The fold rebuild's agreement with the artifact's judged pairs."""

    n: int
    missing: int
    max_score_diff: float
    max_rank_diff: float


@dataclass(frozen=True)
class GuardReport:
    """The checks, the two reproduction maxima and the provenance."""

    checks: tuple[Check, ...]
    provenance: Provenance
    reproduction: Reproduction

    @property
    def ok(self) -> bool:
        """Whether every check held."""
        return all(check.ok for check in self.checks)

    def lines(self) -> list[str]:
        """The report's own lines (every number names its source)."""
        output = ["PROVENANCE", *[f"  {line}" for line in self.provenance.lines()], "", "GUARD"]
        for index, check in enumerate(self.checks, start=1):
            status = "OK" if check.ok else "MISMATCH"
            output.append(f"  {index}. {check.name}: {status}")
            output.append(f"     {check.detail}")
        output.append("")
        output.append(
            "fold rebuild: "
            f"{self.reproduction.n - self.reproduction.missing}/{self.reproduction.n} pairs; "
            f"max |score| {self.reproduction.max_score_diff:.3e} "
            f"(atol {SCORE_TOLERANCE:g}); "
            f"max |rank percentile| {self.reproduction.max_rank_diff:.3e} "
            f"(atol {RANK_TOLERANCE:g})"
        )
        output.append(f"RESULT: {'OK' if self.ok else 'MISMATCH'}")
        return output


def reproduce(context: Context) -> Reproduction:
    """Rebuild every judged score and rank percentile from the fold maps."""
    grids = context.primary
    positions = rank_positions(
        grids.assembled, context.rows, context.columns, context.candidate_columns
    )
    percentiles = percentiles_from_positions(positions, context.rows, context.candidate_columns)
    score_diffs: list[float] = []
    rank_diffs: list[float] = []
    missing = 0
    for judged in context.run.judged:
        index = context.pair_index.get(judged.pair_id)
        if index is None:
            missing += 1
            continue
        fold = int(context.fold_ids[index])
        rebuilt = float(grids.folds[fold][int(context.rows[index]), int(context.columns[index])])
        score_diffs.append(abs(rebuilt - judged.score))
        rank_diffs.append(abs(float(percentiles[index]) - judged.rank_percentile))
    return Reproduction(
        n=len(context.run.judged),
        missing=missing,
        max_score_diff=max(score_diffs) if score_diffs else float("inf"),
        max_rank_diff=max(rank_diffs) if rank_diffs else float("inf"),
    )


def _run_shape_check(context: Context) -> Check:
    """The run id, the seed/folds, the judged/control counts and the documents."""
    corpus = context.run.dictionary.get("corpus") or {}
    cross_fit = context.run.dictionary.get("cross_fit") or {}
    folder = context.sources.run_id
    problems: list[str] = []
    if context.run.run_id != folder:
        problems.append(f"run_id {context.run.run_id!r} is not the folder's {folder!r}")
    if context.run.seed != SEED:
        problems.append(f"seed is {context.run.seed}, expected {SEED}")
    if int(cross_fit.get("folds") or 0) != FOLDS:
        problems.append(f"cross_fit.folds is {cross_fit.get('folds')}, expected {FOLDS}")
    if int(context.run.n_valid_pairs) != EXPECTED_PAIRS:
        problems.append(f"n_valid_pairs is {context.run.n_valid_pairs}, expected {EXPECTED_PAIRS}")
    if len(context.run.judged) != EXPECTED_PAIRS:
        problems.append(f"{len(context.run.judged)} judged rows, expected {EXPECTED_PAIRS}")
    if len(context.run.control) != EXPECTED_CONTROL:
        problems.append(f"{len(context.run.control)} control cells, expected {EXPECTED_CONTROL}")
    if int(corpus.get("n_documents") or 0) != EXPECTED_DOCUMENTS:
        problems.append(f"{corpus.get('n_documents')} documents, expected {EXPECTED_DOCUMENTS}")
    if context.run.control_has_label:
        problems.append("a control cell carries a label (it must never be ground truth)")
    detail = (
        "; ".join(problems)
        if problems
        else f"run {context.run.run_id}, seed {SEED}, folds {FOLDS}, "
        f"{len(context.run.judged)} judged pairs, {len(context.run.control)} control cells, "
        f"{corpus.get('n_documents')} documents"
    )
    return Check("run shape", not problems, detail)


def _map_check(context: Context) -> Check:
    """The full map's structural counts (and the word unit's pinned size)."""
    dictionary = context.run.dictionary
    map_block = dictionary.get("map") or {}
    tables = dictionary.get("tables") or {}
    words = dictionary.get("words") or {}
    problems: list[str] = []
    if int(map_block.get("n_questions") or 0) != EXPECTED_QUESTIONS:
        problems.append(
            f"n_questions is {map_block.get('n_questions')}, expected {EXPECTED_QUESTIONS}"
        )
    if int(map_block.get("n_questions_with_table") or 0) != EXPECTED_TABLES:
        problems.append(
            "n_questions_with_table is "
            f"{map_block.get('n_questions_with_table')}, expected {EXPECTED_TABLES}"
        )
    if int(map_block.get("n_questions_empty") or 0) != EXPECTED_EMPTY:
        problems.append(
            f"n_questions_empty is {map_block.get('n_questions_empty')}, expected {EXPECTED_EMPTY}"
        )
    if len(tables) != EXPECTED_QUESTIONS:
        problems.append(f"{len(tables)} tables, expected {EXPECTED_QUESTIONS}")
    if len(words) != int(map_block.get("n_words") or 0):
        problems.append(f"{len(words)} word entries, the map reports {map_block.get('n_words')}")
    stored = sum(len(rows) for rows in tables.values())
    if stored != int(map_block.get("n_stored_pairs") or 0):
        problems.append(
            f"the tables hold {stored} rows, the map reports {map_block.get('n_stored_pairs')}"
        )
    unit_note = ""
    if context.unit_name == "words":
        if stored != EXPECTED_STORED_PAIRS:
            problems.append(f"{stored} stored pairs, expected {EXPECTED_STORED_PAIRS}")
        if len(words) != EXPECTED_WORDS:
            problems.append(f"{len(words)} words, expected {EXPECTED_WORDS}")
        unit_note = f"word unit: {EXPECTED_STORED_PAIRS} stored pairs, {EXPECTED_WORDS} words"
    else:
        unit_note = (
            f"{context.unit_name} unit: {stored} stored pairs, {len(words)} words "
            "(unit-specific; the word map's pins do not apply)"
        )
    detail = "; ".join(problems) if problems else unit_note
    return Check("map size", not problems, detail)


def _source_check(context: Context) -> Check:
    """The source's own counts, and every judged row's join and label."""
    metadata = context.metadata
    problems: list[str] = []
    if int(metadata.get("n_pairs") or 0) != EXPECTED_PAIRS:
        problems.append(f"n_pairs is {metadata.get('n_pairs')}, expected {EXPECTED_PAIRS}")
    if int(metadata.get("n_positive") or 0) != EXPECTED_POSITIVE:
        problems.append(f"n_positive is {metadata.get('n_positive')}, expected {EXPECTED_POSITIVE}")
    if int(metadata.get("n_negative") or 0) != EXPECTED_NEGATIVE:
        problems.append(f"n_negative is {metadata.get('n_negative')}, expected {EXPECTED_NEGATIVE}")
    if len(context.pairs) != EXPECTED_PAIRS:
        problems.append(f"{len(context.pairs)} source pairs, expected {EXPECTED_PAIRS}")
    labels = sum(1 for pair in context.pairs if pair.label == 1)
    if labels != EXPECTED_POSITIVE:
        problems.append(f"{labels} positive source pairs, expected {EXPECTED_POSITIVE}")
    missing = [row.pair_id for row in context.run.judged if row.pair_id not in context.pair_index]
    if missing:
        problems.append(f"{len(missing)} judged pair ids do not join the source: {missing[:3]}")
    mismatched = [
        row.pair_id
        for row in context.run.judged
        if row.pair_id in context.pair_index
        and row.label != context.pairs[context.pair_index[row.pair_id]].label
    ]
    if mismatched:
        problems.append(f"{len(mismatched)} judged labels disagree: {mismatched[:3]}")
    detail = (
        "; ".join(problems)
        if problems
        else f"{len(context.pairs)} source pairs ({EXPECTED_POSITIVE} positive, "
        f"{EXPECTED_NEGATIVE} negative); every judged pair joins and agrees"
    )
    return Check("source join", not problems, detail)


def _fold_check(context: Context, reproduction: Reproduction) -> Check:
    """Every judged score and rank percentile reproduced from the fold maps."""
    problems: list[str] = []
    if reproduction.missing:
        problems.append(f"{reproduction.missing} judged pairs have no source row")
    if not (reproduction.max_score_diff <= SCORE_TOLERANCE):
        problems.append(
            f"max score difference {reproduction.max_score_diff:.3e} > {SCORE_TOLERANCE:g}"
        )
    if not (reproduction.max_rank_diff <= RANK_TOLERANCE):
        problems.append(
            f"max rank-percentile difference {reproduction.max_rank_diff:.3e} > {RANK_TOLERANCE:g}"
        )
    detail = (
        "; ".join(problems)
        if problems
        else f"{reproduction.n} pairs rebuilt from the five fold maps; max score difference "
        f"{reproduction.max_score_diff:.3e}, max rank-percentile difference "
        f"{reproduction.max_rank_diff:.3e} (atol {SCORE_TOLERANCE:g})"
    )
    return Check("fold rebuild", not problems, detail)


def _fill_check(sources: Sources) -> Check:
    """CS-56's fill block, checked when the abstain run is selected."""
    if sources.run_id != sources.abstain_run:
        return Check(
            "fill block",
            True,
            f"the abstain run {sources.abstain_run!r} is not selected; "
            "the fill block is checked when it is",
        )
    path = sources.results / sources.abstain_run / "analysis.json"
    if not path.is_file():
        return Check("fill block", False, f"no abstain artifact at {path}")
    payload = read_json(path)
    model = next(
        (entry for entry in payload.get("models") or [] if entry.get("kind") == "dictionary"),
        None,
    )
    arms = ((model or {}).get("readout") or {}).get("abstention_arms") or {}
    floor_pairs = int(arms.get("floor_pairs") or 0)
    share = float(arms.get("floor_negative_share") or 0.0)
    non_floor = int((arms.get("floor") or {}).get("non_floor", {}).get("n_pairs") or 0)
    problems: list[str] = []
    if floor_pairs != EXPECTED_FLOOR_PAIRS:
        problems.append(f"floor_pairs is {floor_pairs}, expected {EXPECTED_FLOOR_PAIRS}")
    if abs(share - EXPECTED_FLOOR_NEGATIVE_SHARE) > FLOOR_NEGATIVE_SHARE_TOLERANCE:
        problems.append(
            f"floor_negative_share is {share:.6f}, expected {EXPECTED_FLOOR_NEGATIVE_SHARE:.4f}"
        )
    if non_floor != EXPECTED_NON_FLOOR:
        problems.append(f"non_floor is {non_floor}, expected {EXPECTED_NON_FLOOR}")
    detail = (
        "; ".join(problems)
        if problems
        else f"floor_pairs {floor_pairs}, floor_negative_share {share:.4f}, non_floor {non_floor}"
    )
    return Check("fill block", not problems, detail)


def check_all(sources: Sources, *, context: Context | None = None) -> GuardReport:
    """Run the guard checks (never raises on a mismatch)."""
    context = context or load_context(sources)
    reproduction = reproduce(context)
    checks = (
        _run_shape_check(context),
        _map_check(context),
        _source_check(context),
        _fold_check(context, reproduction),
        _fill_check(sources),
    )
    return GuardReport(
        checks=checks,
        provenance=build_provenance(context),
        reproduction=reproduction,
    )


# --- the CSV helpers --------------------------------------------------------


def _value(row: Any, field_name: str) -> Any:
    """One CSV cell from a dataclass or a mapping."""
    value = row.get(field_name) if isinstance(row, Mapping) else getattr(row, field_name)
    if isinstance(value, (list, tuple)):
        return " | ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def to_csv(
    rows: Sequence[Any],
    fields: Sequence[str],
    *,
    provenance: Sequence[str] = (),
) -> str:
    """The rows as CSV, with an optional ``#``-prefixed provenance block."""
    buffer = io.StringIO()
    for line in provenance:
        buffer.write(f"# {line}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([_value(row, field_name) for field_name in fields])
    return buffer.getvalue()


# --- the CLI ----------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The tool's own arguments (every default is a named constant)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN, help=f"default: {DEFAULT_RUN}")
    parser.add_argument("--abstain-run", default=ABSTAIN_RUN)
    parser.add_argument("--stability-run", default=STABILITY_RUN)
    parser.add_argument("--arm", default=None, help="an arm from the run's readout")
    parser.add_argument("--fill", choices=FILLS, default="floor")
    parser.add_argument("--question", default=None, help="the One question view's question")
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the guard and print its report (exit 0/1)",
    )
    parser.add_argument(
        "--dump",
        choices=("pairs", "errors", "control", "question"),
        default=None,
        help="print one table as CSV (the guard must pass first)",
    )
    parser.add_argument("--limit", type=int, default=None, help="how many rows --dump prints")
    parser.add_argument(
        "--provenance",
        action="store_true",
        help="print the provenance block only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """The tool's entry point: ``--check`` (the guard) and ``--dump`` (a table)."""
    args = _parse_args(argv)
    try:
        sources = resolve_sources(
            args.config,
            results=args.results,
            run_id=args.run,
            abstain_run=args.abstain_run,
            stability_run=args.stability_run,
        )
        if not (args.check or args.dump or args.provenance):
            _parse_args(["--help"])
            return 0
        context = load_context(sources)
        report = check_all(sources, context=context)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"omct_error_view: {exc}", file=sys.stderr)
        return 1
    if args.provenance and not (args.check or args.dump):
        print("\n".join(report.provenance.lines()))
        return 0 if report.ok else 1
    if not report.ok:
        print("\n".join(report.lines()))
        return 1
    if args.check:
        print("\n".join(report.lines()))
        return 0
    if args.dump == "control":
        table: Sequence[Any] = control_strip(context)
        fields = (
            "cell_id",
            "document",
            "paragraph_number",
            "question_number",
            "score",
            "rank_percentile",
        )
    elif args.dump == "question":
        question = args.question or context.questions[0]
        view = question_view(context, question)
        table = view.words
        fields = (
            "word",
            "n_with",
            "df",
            "lift",
            "log_odds",
            "z_score",
            "standard_error",
            "tables",
            "word_class",
        )
    else:
        arm = args.arm or context.run.scoring
        rows = error_rows(context, arm=arm, fill=args.fill)
        if args.dump == "errors":
            table = rows
            fields = (
                "pair_id",
                "document",
                "paragraph_number",
                "question_number",
                "label",
                "score",
                "rank",
                "n_candidates",
                "rank_percentile",
                "floored",
                "error",
            )
        else:
            by_pair = {row.pair_id: row for row in rows}
            judged_score = {row.pair_id: row.score for row in context.run.judged}
            table = [
                {
                    "pair_id": pair.pair_id,
                    "document": pair.document,
                    "label": pair.label,
                    "artifact_score": judged_score.get(pair.pair_id, 0.0),
                    "rebuilt_score": pair_view(
                        context, pair.pair_id, arm=arm, fill=args.fill, with_moves=False
                    ).rebuilt_score,
                    "rank": by_pair[pair.pair_id].rank,
                    "floored": by_pair[pair.pair_id].floored,
                    "error": by_pair[pair.pair_id].error,
                }
                for pair in context.pairs
                if pair.pair_id in by_pair
            ]
            fields = (
                "pair_id",
                "document",
                "label",
                "artifact_score",
                "rebuilt_score",
                "rank",
                "floored",
                "error",
            )
    if args.limit is not None:
        table = table[: args.limit]
    print(to_csv(table, fields, provenance=report.provenance.lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
