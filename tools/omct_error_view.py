"""The OMCT dictionary error viewer's **data layer** (CS-58).

The round CS-48 … CS-57 left ten ``results/omct-analyze-*/analysis.json``
artifacts that say *how well* the dictionary ranks the 1,520 reviewed pairs;
CS-59 / CS-60 then added the two **word-pair** artifacts (one global table over
the words a paragraph and a question share, held out by question or by
paragraph). This layer answers the next question: *which pair does it get
wrong, with which words, on which question, and how does the empty-table fill
move its rank?*

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
from cluster_rag.eval.metrics import grouped_auc_decomposition
from cluster_rag.omct_analysis.view import DatasetView, build_dataset_view
from cluster_rag.pairs import PairRecord, load_pairs
from cluster_rag.text_processors.control import percentiles_from_positions, rank_positions
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.features.phrase_map import phrase_units
from cluster_rag.text_processors.features.question_map import QuestionMap, build_question_map
from cluster_rag.text_processors.features.word_pair_map import (
    WordPairMap,
    build_word_pair_map,
    cross_fit_word_pair_questions,
    cross_fit_word_pair_seen,
    paragraph_fold_ids,
    question_fold_ids,
)
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
    "omct-analyze-voyage-wpairs",
    "omct-analyze-voyage-wpairs-paragraph",
)
WPAIRS_RUN = "omct-analyze-voyage-wpairs"
WPAIRS_PARAGRAPH_RUN = "omct-analyze-voyage-wpairs-paragraph"
ABSTAIN_RUN = "omct-analyze-voyage-dict-abstain"
STABILITY_RUN = "omct-analyze-voyage-dict-stability"
WORD_PAIR_ARMS: tuple[str, ...] = ("sum_zscore_positive", "support_count")

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
class WordPairTableStats:
    """The global table's stored shape (the artifact's ``table.full``)."""

    n_pairs: int
    n_positive: int
    n_negative: int
    n_shared_words: int
    n_rows: int
    n_single_rows: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> WordPairTableStats:
        """Read the artifact's table-shape mapping."""
        return cls(
            n_pairs=int(data.get("n_pairs") or 0),
            n_positive=int(data.get("n_positive") or 0),
            n_negative=int(data.get("n_negative") or 0),
            n_shared_words=int(data.get("n_shared_words") or 0),
            n_rows=int(data.get("n_rows") or 0),
            n_single_rows=int(data.get("n_single_rows") or 0),
        )

    def lines(self) -> list[str]:
        """The shape as one line."""
        return [
            f"{self.n_pairs} pairs ({self.n_positive} positive, {self.n_negative} negative); "
            f"{self.n_shared_words} shared words; {self.n_rows} rows "
            f"({self.n_single_rows} with df == 1)"
        ]


@dataclass(frozen=True)
class WordPairRun:
    """A word-pair run's artifact blocks (CS-59 / CS-60).

    The model entry's ``arms.dictionary`` is the **per-question reference arm**,
    never the word-pair arm; the word-pair table lives in
    ``readout.word_pairs`` (CS-59) or ``readout.word_pairs_paragraph`` (CS-60).
    """

    run_id: str
    seed: int
    n_valid_pairs: int
    model: dict[str, Any]
    block_name: str
    held_out: str
    primary: str
    primary_fill: str
    reference_scoring: str
    judged: tuple[JudgedPair, ...]
    control: tuple[ControlCell, ...]
    readout: dict[str, Any]
    block: dict[str, Any]
    control_has_label: bool = False

    @property
    def arms(self) -> tuple[str, ...]:
        """The word-pair arms the viewer offers (the primary and the count)."""
        return WORD_PAIR_ARMS


@dataclass(frozen=True)
class WordPairAuc:
    """One stored word-pair AUC and the rebuilt one."""

    name: str
    rebuilt: float
    stored: float

    @property
    def abs_diff(self) -> float:
        """The absolute difference."""
        return abs(self.rebuilt - self.stored)


@dataclass(frozen=True)
class WordPairReproduction:
    """The word-pair guard's own readout."""

    aucs: tuple[WordPairAuc, ...]
    table_ok: bool
    floor_ok: bool
    table_detail: str
    floor_detail: str


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


@dataclass
class WordPairContext:
    """Everything read or rebuilt for a word-pair run (CS-59 / CS-60)."""

    sources: Sources
    metadata: dict[str, Any]
    pairs: tuple[SourcePair, ...]
    run: WordPairRun
    view: DatasetView
    paragraphs: tuple[str, ...]
    questions: tuple[str, ...]
    paragraph_row: dict[str, int]
    question_col: dict[str, int]
    pair_index: dict[str, int]
    rows: np.ndarray
    columns: np.ndarray
    labels: np.ndarray
    candidate_columns: tuple[tuple[int, ...], ...]
    fold_ids: np.ndarray
    fold_maps: dict[int, WordPairMap]
    grids: dict[str, dict[str, np.ndarray]]
    floor: np.ndarray
    n_shared: np.ndarray
    n_known: np.ndarray
    n_single: np.ndarray
    full_table: WordPairMap
    reference: Context
    aucs: tuple[WordPairAuc, ...]
    table_ok: bool
    floor_ok: bool
    table_detail: str
    floor_detail: str

    @property
    def scoring(self) -> str:
        """The run's word-pair primary arm."""
        return self.run.primary

    @property
    def arms(self) -> tuple[str, ...]:
        """The word-pair arms on offer."""
        return self.run.arms


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


def _judged_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[JudgedPair, ...]:
    """Parse the artifact's judged-pair rows."""
    return tuple(
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
        for row in rows
    )


def _control_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[ControlCell, ...]:
    """Parse the artifact's control-cell rows."""
    return tuple(
        ControlCell(
            cell_id=str(row.get("cell_id")),
            document=str(row.get("document")),
            paragraph_number=str(row.get("paragraph_number")),
            question_number=int(row.get("question_number") or 0),
            score=float(row.get("score") or 0.0),
            rank_percentile=float(row.get("rank_percentile") or 0.0),
        )
        for row in rows
    )


def load_word_pair_run(sources: Sources, run_id: str | None = None) -> WordPairRun:
    """Read a CS-59 / CS-60 word-pair artifact.

    The block is ``readout.word_pairs`` (questions held out) or
    ``readout.word_pairs_paragraph`` (paragraphs held out); the
    ``arms.dictionary`` rows are the **per-question reference arm**, kept as the
    reference and never presented as the word-pair score.

    Raises:
        FileNotFoundError: If the artifact is missing.
        ValueError: If the run is not a word-pair run.
    """
    selected = run_id or sources.run_id
    path = sources.results / selected / "analysis.json"
    payload = read_json(path)
    model = next(
        (entry for entry in payload.get("models") or [] if entry.get("kind") == "dictionary"),
        None,
    )
    if not isinstance(model, Mapping):
        raise ValueError(f"{path} carries no kind='dictionary' model entry")
    readout = dict(model.get("readout") or {})
    if "word_pairs" in readout:
        block_name, held_out = "word_pairs", "question"
    elif "word_pairs_paragraph" in readout:
        block_name, held_out = "word_pairs_paragraph", "paragraph"
    else:
        raise ValueError(f"{selected!r} is not a word-pair run (no word-pairs block)")
    block = dict(readout[block_name])
    arm = dict(((model.get("arms") or {}).get("dictionary")) or {})
    judged = _judged_rows(arm.get("judged_pairs") or [])
    control = _control_rows(arm.get("control_cells") or [])
    if not judged:
        raise ValueError(f"{selected!r} carries no per-question reference judged rows")
    primary = str(block.get("primary") or "sum_zscore_positive")
    primary_fill = str(
        block.get("primary_fill") or ("neutral" if held_out == "paragraph" else "floor")
    )
    reference_scoring = str(readout.get("primary") or arm.get("scoring") or "sum_zscore_positive")
    control_has_label = any("label" in row for row in arm.get("control_cells") or [])
    return WordPairRun(
        run_id=str(payload.get("run_id") or selected),
        seed=int(payload.get("seed") or SEED),
        n_valid_pairs=int(payload.get("n_valid_pairs") or 0),
        model=dict(model),
        block_name=block_name,
        held_out=held_out,
        primary=primary,
        primary_fill=primary_fill,
        reference_scoring=reference_scoring,
        judged=judged,
        control=control,
        readout=readout,
        block=block,
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


def _build_reference_context(
    sources: Sources,
    metadata: Mapping[str, Any],
    pairs: Sequence[SourcePair],
    view: DatasetView,
    run: WordPairRun,
    pair_folds: np.ndarray,
) -> Context:
    """The per-question reference arm's context (the same pair folds).

    ``arms.dictionary`` in a word-pair artifact is the per-question z arm under
    the round's pair folds. This rebuilds that arm's judged scores and ranks so
    the viewer can keep the field-for-field reference guard.
    """
    paragraph_texts = [pair.paragraph_text for pair in pairs]
    question_texts = [pair.question_text for pair in pairs]
    labels = [pair.label for pair in pairs]
    paragraphs = tuple(view.paragraphs)
    questions = tuple(view.questions)
    paragraph_row = {text: index for index, text in enumerate(paragraphs)}
    question_col = {text: index for index, text in enumerate(questions)}
    rows = np.asarray([paragraph_row[text] for text in paragraph_texts], dtype=np.int64)
    columns = np.asarray([question_col[text] for text in question_texts], dtype=np.int64)
    full_map = build_question_map(paragraph_texts, question_texts, labels)
    fold_maps: dict[int, QuestionMap] = {}
    for fold in range(FOLDS):
        keep = [index for index in range(len(pairs)) if int(pair_folds[index]) != fold]
        fold_maps[fold] = build_question_map(
            [paragraph_texts[index] for index in keep],
            [question_texts[index] for index in keep],
            [labels[index] for index in keep],
        )
    primary = _build_grid_set(
        scoring=run.reference_scoring,
        fill="floor",
        paragraphs=paragraphs,
        questions=questions,
        full_map=full_map,
        fold_maps=fold_maps,
        rows=rows,
        columns=columns,
        fold_ids=pair_folds,
        units=None,
    )
    fake_run = RunArtifact(
        run_id=run.run_id,
        seed=run.seed,
        n_valid_pairs=run.n_valid_pairs,
        dictionary={},
        model=None,
        judged=run.judged,
        control=run.control,
        scoring=run.reference_scoring,
        unit_name="words",
        readout={},
        control_has_label=run.control_has_label,
    )
    return Context(
        sources=sources,
        metadata=dict(metadata),
        pairs=tuple(pairs),
        run=fake_run,
        fold_ids=pair_folds,
        paragraphs=paragraphs,
        questions=questions,
        paragraph_row=paragraph_row,
        question_col=question_col,
        pair_index={pair.pair_id: index for index, pair in enumerate(pairs)},
        rows=rows,
        columns=columns,
        labels=np.asarray(labels, dtype=np.int64),
        candidate_columns=tuple(view.candidate_columns()),
        unit_name="words",
        units=None,
        full_map=full_map,
        fold_maps=fold_maps,
        primary=primary,
    )


def _score_vector(
    matrix: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """The judged pairs' scores from an assembled grid."""
    return np.asarray(
        [float(matrix[int(rows[i]), int(columns[i])]) for i in range(len(rows))],
        dtype=np.float64,
    )


def _within_auc(labels: np.ndarray, scores: np.ndarray, documents: Sequence[str]) -> float:
    """The within-document AUC (the run's headline)."""
    decomposition = grouped_auc_decomposition(
        labels.tolist(), scores.tolist(), [str(document) for document in documents]
    )
    value = decomposition.get("within_group_auc")
    if value is None:
        raise ValueError("the within-document AUC is undefined")
    return float(value)


def _table_stats_match(full_table: WordPairMap, stored: Mapping[str, Any]) -> bool:
    """Whether the rebuilt full table agrees with the artifact's shape."""
    rebuilt = full_table.stats()
    expected = WordPairTableStats.from_mapping(stored)
    return all(
        int(rebuilt[key]) == getattr(expected, key)
        for key in (
            "n_pairs",
            "n_positive",
            "n_negative",
            "n_shared_words",
            "n_rows",
            "n_single_rows",
        )
    )


def _table_stats_detail(full_table: WordPairMap, stored: Mapping[str, Any]) -> str:
    """The rebuilt/stored table shape, for the guard's detail."""
    rebuilt = full_table.stats()
    expected = WordPairTableStats.from_mapping(stored)
    return (
        f"rebuilt {rebuilt['n_pairs']} pairs / {rebuilt['n_shared_words']} shared words / "
        f"{rebuilt['n_rows']} rows ({rebuilt['n_single_rows']} df == 1); "
        f"stored {expected.n_pairs} / {expected.n_shared_words} / {expected.n_rows} "
        f"({expected.n_single_rows})"
    )


def load_word_pair_context(sources: Sources) -> WordPairContext:
    """Read a word-pair run and rebuild its fold tables, grids and guard readout.

    The selected evaluation is the run's held-out side: the **questions** for
    CS-59, the **paragraphs** for CS-60. CS-59's pair-level "seen" arm is rebuilt
    beside it for the guard (0.6808), but the viewer's one-pair/errors view uses
    the held-out-question arm.

    Raises:
        FileNotFoundError: If a source is missing.
        ValueError: If the artifact and the source cannot be aligned.
    """
    report = load_pairs(sources.data_external)
    pairs = _source_pairs(report.records)
    run = load_word_pair_run(sources)
    n_pairs = len(pairs)
    view = build_dataset_view(
        report.records,
        control_size=max(1, len(run.control)),
        seed=run.seed,
    )
    paragraph_texts = [pair.paragraph_text for pair in pairs]
    question_texts = [pair.question_text for pair in pairs]
    labels = [pair.label for pair in pairs]
    paragraphs = tuple(view.paragraphs)
    questions = tuple(view.questions)
    paragraph_row = {text: index for index, text in enumerate(paragraphs)}
    question_col = {text: index for index, text in enumerate(questions)}
    rows = np.asarray([paragraph_row[text] for text in paragraph_texts], dtype=np.int64)
    columns = np.asarray([question_col[text] for text in question_texts], dtype=np.int64)
    documents = [pair.document for pair in pairs]
    pair_folds_ref = fold_ids(n_pairs, folds=FOLDS, seed=run.seed)
    if run.held_out == "question":
        question_folds = question_fold_ids(questions, FOLDS, run.seed)
        assignment = np.asarray(
            [int(question_folds[pair.question_text]) for pair in pairs], dtype=np.int64
        )
        cross = cross_fit_word_pair_questions(
            paragraph_texts,
            question_texts,
            labels,
            question_folds,
            paragraphs=paragraphs,
            questions=questions,
            scorings=WORD_PAIR_ARMS,
            fills=("floor", "neutral"),
        )
        seen_cross = cross_fit_word_pair_seen(
            paragraph_texts,
            question_texts,
            labels,
            pair_folds_ref,
            paragraphs=paragraphs,
            questions=questions,
            scorings=(run.primary,),
            fills=("floor",),
        )
        aucs = (
            WordPairAuc(
                "seen (pair folds, floor)",
                _within_auc(
                    np.asarray(labels, dtype=np.int64),
                    _score_vector(seen_cross["scorings"][run.primary]["floor"], rows, columns),
                    documents,
                ),
                float((run.block.get("seen") or {}).get("within_auc") or 0.0),
            ),
            WordPairAuc(
                "held-out questions (floor)",
                _within_auc(
                    np.asarray(labels, dtype=np.int64),
                    _score_vector(cross["scorings"][run.primary]["floor"], rows, columns),
                    documents,
                ),
                float((run.block.get("held_out_questions") or {}).get("within_auc") or 0.0),
            ),
        )
        floor_ok = int(seen_cross["floor"].sum()) == int(
            ((run.block.get("seen") or {}).get("floor") or {}).get("n_floor_pairs") or 0
        ) and int(cross["floor"].sum()) == int(
            ((run.block.get("held_out_questions") or {}).get("floor") or {}).get("n_floor_pairs")
            or 0
        )
        seen_floor = ((run.block.get("seen") or {}).get("floor") or {}).get("n_floor_pairs")
        held_floor = ((run.block.get("held_out_questions") or {}).get("floor") or {}).get(
            "n_floor_pairs"
        )
        floor_detail = (
            f"seen floor {int(seen_cross['floor'].sum())} vs {seen_floor}; "
            f"held-out floor {int(cross['floor'].sum())} vs {held_floor}"
        )
    else:
        paragraph_folds = paragraph_fold_ids(paragraphs, FOLDS, run.seed)
        assignment = np.asarray(
            [int(paragraph_folds[pair.paragraph_text]) for pair in pairs], dtype=np.int64
        )
        cross = cross_fit_word_pair_seen(
            paragraph_texts,
            question_texts,
            labels,
            assignment,
            paragraphs=paragraphs,
            questions=questions,
            scorings=WORD_PAIR_ARMS,
            fills=("floor", "neutral"),
        )
        aucs = (
            WordPairAuc(
                "held-out paragraphs (neutral)",
                _within_auc(
                    np.asarray(labels, dtype=np.int64),
                    _score_vector(cross["scorings"][run.primary]["neutral"], rows, columns),
                    documents,
                ),
                float((run.block.get("neutral") or {}).get("within_auc") or 0.0),
            ),
            WordPairAuc(
                "held-out paragraphs (floor)",
                _within_auc(
                    np.asarray(labels, dtype=np.int64),
                    _score_vector(cross["scorings"][run.primary]["floor"], rows, columns),
                    documents,
                ),
                float((run.block.get("floor_arm") or {}).get("within_auc") or 0.0),
            ),
        )
        floor_ok = int(cross["floor"].sum()) == int(
            ((run.block.get("neutral") or {}).get("floor") or {}).get("n_floor_pairs") or 0
        )
        floor_detail = (
            f"held-out floor {int(cross['floor'].sum())} vs "
            f"{((run.block.get('neutral') or {}).get('floor') or {}).get('n_floor_pairs')}"
        )
    fold_maps: dict[int, WordPairMap] = {}
    for fold in range(FOLDS):
        keep = [index for index in range(n_pairs) if int(assignment[index]) != fold]
        fold_maps[fold] = build_word_pair_map(
            [paragraph_texts[index] for index in keep],
            [question_texts[index] for index in keep],
            [labels[index] for index in keep],
        )
    full_table = build_word_pair_map(paragraph_texts, question_texts, labels)
    table_block = (run.block.get("table") or {}).get("full") or {}
    reference = _build_reference_context(
        sources,
        report.metadata,
        pairs,
        view,
        run,
        pair_folds_ref,
    )
    return WordPairContext(
        sources=sources,
        metadata=dict(report.metadata),
        pairs=pairs,
        run=run,
        view=view,
        paragraphs=paragraphs,
        questions=questions,
        paragraph_row=paragraph_row,
        question_col=question_col,
        pair_index={pair.pair_id: index for index, pair in enumerate(pairs)},
        rows=rows,
        columns=columns,
        labels=np.asarray(labels, dtype=np.int64),
        candidate_columns=tuple(view.candidate_columns()),
        fold_ids=assignment,
        fold_maps=fold_maps,
        grids=dict(cross["scorings"]),
        floor=np.asarray(cross["floor"], dtype=bool),
        n_shared=np.asarray(cross["n_shared"], dtype=np.int64),
        n_known=np.asarray(cross["n_known"], dtype=np.int64),
        n_single=np.asarray(cross["n_single"], dtype=np.int64),
        full_table=full_table,
        reference=reference,
        aucs=aucs,
        table_ok=_table_stats_match(full_table, table_block),
        floor_ok=floor_ok,
        table_detail=_table_stats_detail(full_table, table_block),
        floor_detail=floor_detail,
    )


def load_any_context(sources: Sources) -> Context | WordPairContext:
    """Load the selected run, branching on its artifact shape."""
    payload = read_json(sources.results / sources.run_id / "analysis.json")
    if isinstance(payload.get("dictionary"), Mapping):
        return load_context(sources)
    model = next(
        (entry for entry in payload.get("models") or [] if entry.get("kind") == "dictionary"),
        None,
    )
    readout = dict((model or {}).get("readout") or {})
    if "word_pairs" in readout or "word_pairs_paragraph" in readout:
        return load_word_pair_context(sources)
    raise ValueError(
        f"{sources.run_id!r} is neither a per-question dictionary run nor a word-pair run"
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


def build_provenance(context: Context | WordPairContext) -> Provenance:
    """The provenance block from the loaded context."""
    sources = context.sources
    run_path = sources.results / sources.run_id / "analysis.json"
    if isinstance(context, WordPairContext):
        return Provenance(
            config_path=str(sources.config_path),
            data_path=str(sources.data_external),
            data_sha256=sha256_file(sources.data_external),
            run_id=context.run.run_id,
            run_path=str(run_path),
            run_sha256=sha256_file(run_path),
            seed=context.run.seed,
            folds=FOLDS,
            unit="word pairs",
            scoring=context.run.primary,
            n_pairs=len(context.pairs),
            n_judged=len(context.run.judged),
            n_control=len(context.run.control),
            n_documents=len({pair.document for pair in context.pairs}),
        )
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
class SignedWord:
    """One paragraph word in the question's fold table, with its **signed** evidence.

    ``direction`` follows the selected arm's own measure: the log-odds for
    ``sum_positive``, the z-score for the z arms (and for ``support_count``,
    which has no sign of its own). ``summed`` says whether the positive-only
    arm actually adds this word: a negative word is shown, but contributes
    **0** under ``sum_positive`` / ``sum_zscore_positive``. The raw log-odds
    and z are both kept, because they can disagree in sign (CS-50).
    """

    word: str
    n_with: int
    df: int
    log_odds: float
    z_score: float
    contribution: float
    direction: str
    summed: bool


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
    is_primary: bool
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
    signed_words: tuple[SignedWord, ...]
    moves: tuple[RankMove, ...]


def _units_of(context: Context, text: str) -> set[str]:
    """The unit set of a paragraph under the run's unit."""
    units = context.units or (lambda value: content_tokens(value, DEFAULT_STOPWORDS))
    return set(units(text))


def signed_words_for(context: Context, index: int, arm: str) -> tuple[SignedWord, ...]:
    """Every paragraph word in the question's fold table, signed by the arm's measure.

    The fold map is the pair's own out-of-fold map, so this is the evidence the
    artifact's score actually used. Negative words are **kept**: they are the
    map's evidence against the question, and the viewer shows them even though
    the positive-only arms do not subtract them. A floored pair has no table,
    so it has no words.
    """
    if floored(context, index):
        return ()
    question = context.pairs[index].question_text
    fold_map = context.fold_maps[int(context.fold_ids[index])]
    wanted = _units_of(context, context.pairs[index].paragraph_text)
    rows: list[tuple[float, SignedWord]] = []
    for row in fold_map.per_question.get(question, ()):
        if row.word not in wanted:
            continue
        if arm == "sum_positive":
            measure = float(row.log_odds)
            contribution = measure if measure > 0.0 else 0.0
            summed = measure > 0.0
        elif arm == "sum_zscore_positive":
            measure = float(row.z_score)
            contribution = measure if measure > 0.0 else 0.0
            summed = measure > 0.0
        elif arm == "support_count":
            measure = float(row.z_score)
            contribution = 1.0
            summed = True
        else:
            raise ValueError(f"arm {arm!r} has no additive contribution")
        direction = "positive" if measure > 0.0 else "negative" if measure < 0.0 else "zero"
        rows.append(
            (
                measure,
                SignedWord(
                    word=row.word,
                    n_with=int(row.n_with),
                    df=int(row.df),
                    log_odds=float(row.log_odds),
                    z_score=float(row.z_score),
                    contribution=float(contribution),
                    direction=direction,
                    summed=summed,
                ),
            )
        )
    rows.sort(key=lambda item: (-item[0], item[1].word))
    return tuple(word for _, word in rows)


def contributions_for(context: Context, index: int, arm: str) -> tuple[Contribution, ...]:
    """The fold-map rows that gave the pair points under ``arm``.

    This is the positive-only decomposition of the score: the words the arm
    actually sums. The signed table is :func:`signed_words_for`; this helper is
    kept for the CLI and the tests, and for the score's own audit.
    """
    return _contributions_from_signed(signed_words_for(context, index, arm))


def _contributions_from_signed(words: Sequence[SignedWord]) -> tuple[Contribution, ...]:
    """The summed, positive part of a signed word list (the score's decomposition)."""
    rows = [
        Contribution(
            word=word.word,
            n_with=word.n_with,
            df=word.df,
            log_odds=word.log_odds,
            z_score=word.z_score,
            contribution=word.contribution,
        )
        for word in words
        if word.summed and word.contribution > 0.0
    ]
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
    is_primary = selected_arm == context.run.scoring
    reproduction_ok = bool(abs(rebuilt - artifact_score) <= SCORE_TOLERANCE) if is_primary else True
    rank, n_candidates, percentile = rank_in_candidates(context, grids.assembled, index)
    is_floored = floored(context, index)
    fold_positive, fold_pairs = fold_counts(context, index)
    signed_words = signed_words_for(context, index, selected_arm) if reproduction_ok else ()
    contributions = _contributions_from_signed(signed_words)
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
        is_primary=is_primary,
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
        signed_words=signed_words,
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
class WordPairContribution:
    """One shared word and its fold-table evidence for a word-pair run."""

    word: str
    in_fold: bool
    df: int | None
    n_with: int | None
    z_score: float | None
    contribution: float
    direction: str
    single: bool


@dataclass(frozen=True)
class WordPairPairView:
    """Everything the One pair view shows for a word-pair run."""

    pair: SourcePair
    fold: int
    held_out: str
    floored: bool
    primary_fill: str
    arm: str
    scores: Mapping[str, float]
    ranks: Mapping[str, tuple[int, int, float]]
    contributions: tuple[WordPairContribution, ...]
    reference_score: float
    reference_rank: int
    reference_percentile: float
    reference_scoring: str


def _rank_in_row(
    scores: np.ndarray,
    row: int,
    column: int,
    candidates: Sequence[int],
) -> tuple[int, int, float]:
    """One cell's 1-based rank, candidate count and percentile in its row."""
    order = sorted(candidates, key=lambda col: -float(scores[row, col]))
    position = order.index(column)
    return position + 1, len(candidates), position / len(candidates)


def _shared_words(paragraph: str, question: str) -> set[str]:
    """The pair's shared content words (the word-pair table's unit)."""
    return set(content_tokens(paragraph, DEFAULT_STOPWORDS)) & set(
        content_tokens(question, DEFAULT_STOPWORDS)
    )


def word_pair_pair_view(
    context: WordPairContext,
    pair_id: str,
    *,
    arm: str | None = None,
) -> WordPairPairView:
    """One judged pair under the word-pair run's held-out fold."""
    if pair_id not in context.pair_index:
        raise KeyError(f"no reviewed pair {pair_id!r}")
    index = context.pair_index[pair_id]
    selected_arm = arm or context.run.primary
    pair = context.pairs[index]
    fold = int(context.fold_ids[index])
    fold_map = context.fold_maps[fold]
    wanted = _shared_words(pair.paragraph_text, pair.question_text)
    rows: list[WordPairContribution] = []
    for word in wanted:
        row = fold_map.rows.get(word)
        if row is None:
            rows.append(
                WordPairContribution(
                    word=word,
                    in_fold=False,
                    df=None,
                    n_with=None,
                    z_score=None,
                    contribution=0.0,
                    direction="not in fold",
                    single=False,
                )
            )
            continue
        if selected_arm == "support_count":
            contribution = 1.0
        else:
            contribution = float(row.z_score) if row.z_score > 0.0 else 0.0
        direction = "positive" if row.z_score > 0.0 else "negative" if row.z_score < 0.0 else "zero"
        rows.append(
            WordPairContribution(
                word=word,
                in_fold=True,
                df=int(row.df),
                n_with=int(row.n_with),
                z_score=float(row.z_score),
                contribution=float(contribution),
                direction=direction,
                single=bool(row.df == 1),
            )
        )
    rows.sort(
        key=lambda row: (
            not row.in_fold,
            -row.contribution,
            -(row.z_score or 0.0),
            row.word,
        )
    )
    scores = {
        candidate_fill: float(
            context.grids[selected_arm][candidate_fill][
                int(context.rows[index]), int(context.columns[index])
            ]
        )
        for candidate_fill in ("floor", "neutral")
    }
    ranks = {
        candidate_fill: _rank_in_row(
            context.grids[selected_arm][candidate_fill],
            int(context.rows[index]),
            int(context.columns[index]),
            context.candidate_columns[int(context.rows[index])],
        )
        for candidate_fill in ("floor", "neutral")
    }
    judged = next(row for row in context.run.judged if row.pair_id == pair_id)
    reference_rank, _, reference_percentile = _rank_in_row(
        context.reference.primary.assembled,
        int(context.rows[index]),
        int(context.columns[index]),
        context.candidate_columns[int(context.rows[index])],
    )
    return WordPairPairView(
        pair=pair,
        fold=fold,
        held_out=context.run.held_out,
        floored=bool(context.floor[index]),
        primary_fill=context.run.primary_fill,
        arm=selected_arm,
        scores=scores,
        ranks=ranks,
        contributions=tuple(rows),
        reference_score=float(judged.score),
        reference_rank=reference_rank,
        reference_percentile=reference_percentile,
        reference_scoring=context.run.reference_scoring,
    )


@dataclass(frozen=True)
class WordPairQuestionWord:
    """One question word's relation to the full global table."""

    word: str
    in_table: bool
    df: int | None
    n_with: int | None
    paragraph_df: int | None
    z_score: float | None
    log_odds: float | None
    single: bool


@dataclass(frozen=True)
class WordPairQuestionView:
    """Everything the One question view shows for a word-pair run."""

    question: str
    question_number: int
    words: tuple[WordPairQuestionWord, ...]
    pairs: int
    known_pairs: int
    coverage: float
    table_stats: WordPairTableStats


def word_pair_question_view(context: WordPairContext, question: str) -> WordPairQuestionView:
    """The question's words in the full global table and its pair coverage."""
    words = tuple(dict.fromkeys(content_tokens(question, DEFAULT_STOPWORDS)))
    built = []
    for word in words:
        row = context.full_table.rows.get(word)
        built.append(
            WordPairQuestionWord(
                word=word,
                in_table=row is not None,
                df=int(row.df) if row else None,
                n_with=int(row.n_with) if row else None,
                paragraph_df=int(row.paragraph_df) if row else None,
                z_score=float(row.z_score) if row else None,
                log_odds=float(row.log_odds) if row else None,
                single=bool(row.df == 1) if row else False,
            )
        )
    indices = [index for index, pair in enumerate(context.pairs) if pair.question_text == question]
    known = sum(
        1
        for index in indices
        if any(
            word in context.full_table.rows
            for word in _shared_words(
                context.pairs[index].paragraph_text,
                context.pairs[index].question_text,
            )
        )
    )
    number = next(
        (pair.question_number for pair in context.pairs if pair.question_text == question), 0
    )
    return WordPairQuestionView(
        question=question,
        question_number=number,
        words=tuple(built),
        pairs=len(indices),
        known_pairs=known,
        coverage=(known / len(indices)) if indices else 0.0,
        table_stats=WordPairTableStats.from_mapping(
            (context.run.block.get("table") or {}).get("full") or {}
        ),
    )


def word_pair_error_rows(
    context: WordPairContext,
    *,
    arm: str | None = None,
    fill: str | None = None,
) -> tuple[ErrorRow, ...]:
    """Every judged pair classified under the word-pair run's held-out fold."""
    selected_arm = arm or context.run.primary
    selected_fill = fill or context.run.primary_fill
    matrix = context.grids[selected_arm][selected_fill]
    positions = rank_positions(matrix, context.rows, context.columns, context.candidate_columns)
    percentiles = percentiles_from_positions(positions, context.rows, context.candidate_columns)
    rows: list[ErrorRow] = []
    for index, pair in enumerate(context.pairs):
        score = float(matrix[int(context.rows[index]), int(context.columns[index])])
        rows.append(
            ErrorRow(
                pair_id=pair.pair_id,
                document=pair.document,
                paragraph_number=pair.paragraph_number,
                question_number=pair.question_number,
                label=pair.label,
                score=score,
                rank=int(positions[index]) + 1,
                n_candidates=len(context.candidate_columns[int(context.rows[index])]),
                rank_percentile=float(percentiles[index]),
                floored=bool(context.floor[index]),
                error=error_of(pair.label, bool(context.floor[index]), score),
            )
        )
    return tuple(rows)


def word_pair_control_strip(context: WordPairContext) -> tuple[ControlCell, ...]:
    """The unjudged control cells as the artifact's **reference-arm** rows.

    The word-pair block stores aggregate control deltas, not per-cell word-pair
    scores; the reference arm's control rows are shown, labelled as the
    reference, never as word-pair scores and never as ground truth.
    """
    return context.run.control


@dataclass(frozen=True)
class WordPairComparison:
    """One stored paired comparison from a word-pair artifact."""

    label: str
    arm: float
    reference: float
    delta: float
    ci_low: float
    ci_high: float
    n_pairs: int | None
    reference_name: str


def _stored_comparison(
    label: str,
    block: Mapping[str, Any] | None,
    *,
    reference_name: str,
    n_pairs: int | None = None,
) -> WordPairComparison | None:
    """One stored paired block as a comparison row (``None`` if absent)."""
    if not isinstance(block, Mapping):
        return None
    paired = block.get("paired") if isinstance(block.get("paired"), Mapping) else block
    arm_auc = paired.get("auc_arm") if "auc_arm" in paired else paired.get("within_auc_arm")
    reference_auc = (
        paired.get("auc_reference")
        if "auc_reference" in paired
        else paired.get("within_auc_reference")
    )
    if arm_auc is None or reference_auc is None:
        return None
    return WordPairComparison(
        label=label,
        arm=float(arm_auc),
        reference=float(reference_auc),
        delta=float(paired.get("delta") or 0.0),
        ci_low=float(paired.get("ci_low") or 0.0),
        ci_high=float(paired.get("ci_high") or 0.0),
        n_pairs=n_pairs if n_pairs is not None else paired.get("n_pairs"),
        reference_name=reference_name,
    )


def word_pair_comparisons(context: WordPairContext) -> tuple[WordPairComparison, ...]:
    """The artifact's stored word-pair comparisons (no recomputation)."""
    block = context.run.block
    rows: list[WordPairComparison] = []
    if context.run.held_out == "question":
        seen = block.get("seen") or {}
        held = block.get("held_out_questions") or {}
        candidates = [
            _stored_comparison(
                "seen · floor vs per-question z",
                seen.get("reference"),
                reference_name="per-question z arm",
                n_pairs=int(seen.get("n_pairs") or 0) or None,
            ),
            _stored_comparison(
                "seen · neutral vs per-question neutral",
                seen.get("neutral_fill"),
                reference_name="per-question neutral",
                n_pairs=int(seen.get("n_pairs") or 0) or None,
            ),
            _stored_comparison(
                "held-out questions · floor vs cosine",
                held.get("reference"),
                reference_name="cosine",
                n_pairs=int(held.get("n_pairs") or 0) or None,
            ),
            _stored_comparison(
                "held-out questions · neutral vs cosine",
                held.get("neutral_fill"),
                reference_name="cosine",
                n_pairs=int(held.get("n_pairs") or 0) or None,
            ),
        ]
    else:
        neutral = block.get("neutral") or {}
        floor_arm = block.get("floor_arm") or {}
        candidates = [
            _stored_comparison(
                "held-out paragraphs · neutral vs cosine",
                neutral.get("paired_vs_cosine"),
                reference_name="cosine",
            ),
            _stored_comparison(
                "held-out paragraphs · floor vs cosine",
                floor_arm.get("paired_vs_cosine"),
                reference_name="cosine",
            ),
            _stored_comparison(
                "fallback · word-pair vs cosine",
                block.get("fallback"),
                reference_name="cosine",
            ),
            _stored_comparison(
                "fallback · word-pair vs per-question neutral",
                block.get("fallback_increment"),
                reference_name="per-question neutral",
            ),
            _stored_comparison(
                "head-to-head · per-question reference vs word-pair",
                block.get("head_to_head"),
                reference_name="per-question arm",
            ),
            _stored_comparison(
                "fill · neutral vs floor",
                (block.get("fill_delta") or {}).get("paired"),
                reference_name="floor",
            ),
        ]
    for candidate in candidates:
        if candidate is not None:
            rows.append(candidate)
    return tuple(rows)


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
    reproduction: Reproduction | None
    word_pair: WordPairReproduction | None = None

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
        if self.reproduction is not None:
            label = (
                "per-question reference rebuild" if self.word_pair is not None else "fold rebuild"
            )
            output.append(
                f"{label}: "
                f"{self.reproduction.n - self.reproduction.missing}/{self.reproduction.n} pairs; "
                f"max |score| {self.reproduction.max_score_diff:.3e} "
                f"(atol {SCORE_TOLERANCE:g}); "
                f"max |rank percentile| {self.reproduction.max_rank_diff:.3e} "
                f"(atol {RANK_TOLERANCE:g})"
            )
        if self.word_pair is not None:
            output.append("word-pair rebuild:")
            for auc in self.word_pair.aucs:
                output.append(
                    f"  {auc.name}: stored {auc.stored:.16f} vs rebuilt "
                    f"{auc.rebuilt:.16f} (|diff| {auc.abs_diff:.3e})"
                )
            output.append(
                f"  table: {self.word_pair.table_detail} "
                f"[{'OK' if self.word_pair.table_ok else 'MISMATCH'}]"
            )
            output.append(
                f"  floor: {self.word_pair.floor_detail} "
                f"[{'OK' if self.word_pair.floor_ok else 'MISMATCH'}]"
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


def _word_pair_run_shape_check(context: WordPairContext) -> Check:
    """The word-pair run's id, seed, counts, documents and held-out side."""
    run = context.run
    folder = context.sources.run_id
    problems: list[str] = []
    if run.run_id != folder:
        problems.append(f"run_id {run.run_id!r} is not the folder's {folder!r}")
    if run.seed != SEED:
        problems.append(f"seed is {run.seed}, expected {SEED}")
    if run.n_valid_pairs != EXPECTED_PAIRS:
        problems.append(f"n_valid_pairs is {run.n_valid_pairs}, expected {EXPECTED_PAIRS}")
    if len(run.judged) != EXPECTED_PAIRS:
        problems.append(f"{len(run.judged)} judged rows, expected {EXPECTED_PAIRS}")
    if len(run.control) != EXPECTED_CONTROL:
        problems.append(f"{len(run.control)} control cells, expected {EXPECTED_CONTROL}")
    documents = len({pair.document for pair in context.pairs})
    if documents != EXPECTED_DOCUMENTS:
        problems.append(f"{documents} documents, expected {EXPECTED_DOCUMENTS}")
    if run.control_has_label:
        problems.append("a control cell carries a label (it must never be ground truth)")
    if run.block_name not in ("word_pairs", "word_pairs_paragraph"):
        problems.append(f"unknown word-pair block {run.block_name!r}")
    if run.held_out not in ("question", "paragraph"):
        problems.append(f"unknown held-out side {run.held_out!r}")
    if run.primary_fill not in ("floor", "neutral"):
        problems.append(f"unknown primary fill {run.primary_fill!r}")
    detail = (
        "; ".join(problems)
        if problems
        else f"word-pair run {run.run_id}, seed {SEED}, folds {FOLDS}, "
        f"{len(run.judged)} judged pairs, {len(run.control)} control cells, {documents} documents; "
        f"held out: the {run.held_out}; primary fill {run.primary_fill}"
    )
    return Check("run shape", not problems, detail)


def _word_pair_table_check(context: WordPairContext) -> Check:
    """The artifact's `table.full` against the rebuilt global table."""
    return Check("word-pair table", context.table_ok, context.table_detail)


def _word_pair_auc_check(context: WordPairContext) -> Check:
    """The rebuilt held-out/seen AUCs against the artifact's stored ones."""
    problems = [
        f"{auc.name}: stored {auc.stored:.16f} vs rebuilt {auc.rebuilt:.16f} "
        f"(|diff| {auc.abs_diff:.3e})"
        for auc in context.aucs
        if auc.abs_diff > SCORE_TOLERANCE
    ]
    detail = (
        "; ".join(problems)
        if problems
        else "; ".join(
            f"{auc.name}: stored {auc.stored:.16f} vs rebuilt {auc.rebuilt:.16f}"
            for auc in context.aucs
        )
    )
    return Check("word-pair AUC", not problems and bool(context.aucs), detail)


def _word_pair_floor_check(context: WordPairContext) -> Check:
    """The rebuilt floor counts against the artifact's stored counts."""
    return Check("word-pair floor", context.floor_ok, context.floor_detail)


def _word_pair_reference_check(context: WordPairContext) -> Check:
    """The per-question reference arm, reproduced field for field."""
    reference = reproduce(context.reference)
    check = _fold_check(context.reference, reference)
    return Check("per-question reference", check.ok, check.detail)


def _word_pair_check_all(sources: Sources, context: WordPairContext) -> GuardReport:
    """The word-pair guard: the source, the table, the folds and the reference."""
    reference = reproduce(context.reference)
    word_pair = WordPairReproduction(
        aucs=context.aucs,
        table_ok=context.table_ok,
        floor_ok=context.floor_ok,
        table_detail=context.table_detail,
        floor_detail=context.floor_detail,
    )
    checks = (
        _word_pair_run_shape_check(context),
        _source_check(context),
        _word_pair_table_check(context),
        _word_pair_auc_check(context),
        _word_pair_floor_check(context),
        _word_pair_reference_check(context),
    )
    return GuardReport(
        checks=checks,
        provenance=build_provenance(context),
        reproduction=reference,
        word_pair=word_pair,
    )


def check_all(
    sources: Sources,
    *,
    context: Context | WordPairContext | None = None,
) -> GuardReport:
    """Run the guard checks (never raises on a mismatch)."""
    context = context or load_any_context(sources)
    if isinstance(context, WordPairContext):
        return _word_pair_check_all(sources, context)
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
    parser.add_argument("--fill", choices=FILLS, default=None)
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


def _dump_word_pair(
    context: WordPairContext,
    report: GuardReport,
    args: argparse.Namespace,
) -> int:
    """The word-pair CLI's ``--dump`` tables (the guard must have passed)."""
    if args.dump == "control":
        table: Sequence[Any] = word_pair_control_strip(context)
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
        view = word_pair_question_view(context, question)
        table = view.words
        fields = (
            "word",
            "in_table",
            "df",
            "n_with",
            "paragraph_df",
            "z_score",
            "log_odds",
            "single",
        )
    else:
        arm = args.arm or context.run.primary
        fill = args.fill or context.run.primary_fill
        table = word_pair_error_rows(context, arm=arm, fill=fill)
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
    if args.limit is not None:
        table = table[: args.limit]
    print(to_csv(table, fields, provenance=report.provenance.lines()))
    return 0


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
        context = load_any_context(sources)
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
    if isinstance(context, WordPairContext):
        return _dump_word_pair(context, report, args)
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
        fill = args.fill or "floor"
        rows = error_rows(context, arm=arm, fill=fill)
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
                        context, pair.pair_id, arm=arm, fill=fill, with_moves=False
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
