"""The error viewer's **data layer** (V1): the dictionary classifier's mistakes.

The viewer asks three questions of the UPR **val** split: *what did the
classifier get wrong*, *with which words*, and *how close was it to the cut*.
No analysis artifact stores the labels a text was predicted as, so V1
**recomputes** the dictionary classifier from the corpus map's own artifact
(`results/upr-hitmap-corpus/hitmap.json` -> `corpus_map.per_label`) with the
track's `map_score` (CS-31/CS-47) and the harness's `top_k` decision rule.

The layer produces three tidy tables -- :class:`TextRow`, :class:`CellRow` and
:class:`WordRow` -- and every view is a filter or a join over them. It also
carries the **guard**: the recomputed val micro-F1 is compared with the run's
own metrics artifact at 1e-9, and the map's provenance (train split, source
sha, label space, the eight held-out labels) is checked against the split. A
mismatch is printed and exits 1; it is never smoothed over.

Pure: numpy + the standard library, no model client, no Streamlit, no network.
The artifact rows are read as they are stored; the map is not rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cluster_rag.config import AppConfig, load_config
from cluster_rag.data import (
    Record,
    build_label_matrix,
    load_jsonl,
    load_split_ids,
    select_records_by_ids,
)
from cluster_rag.eval.multilabel import apply_decision_rule
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.lexical.contrastive import (
    DictionaryRow,
    default_contrast_label,
    dictionary_ranking,
    map_score,
)
from cluster_rag.text_processors.lexical.corpus_map_score import FLOOR_GAP

# The tool's own configuration surface (every one has a CLI flag or an app
# control; nothing is hard-coded in the logic).
DEFAULT_CONFIG = Path("configs/base.yaml")
DEFAULT_SPLIT = "val"
EXPECTED_K = 2
DEFAULT_NEAR_MISS_K = 5
LENGTH_EDGES: tuple[int, ...] = (100, 200, 300)
LENGTH_BUCKETS: tuple[str, ...] = ("0-99", "100-199", "200-299", "300+")
CORPUS_MAP_RUN = "upr-hitmap-corpus"
EVAL_RUN = "upr-eval-corpus-map-sum-zscore-positive-val"
SURROGATE_RUN = "upr-hitmap-surrogate"

# The guard's tolerances. The artifact's metrics are full precision, so the
# reproduction is exact; the contribution sum is a permutation of the map's
# own addition order, so it gets a float-level tolerance.
F1_TOLERANCE = 1e-9
SUM_TOLERANCE = 1e-9


# --- the sources ------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """The files the viewer reads, resolved.

    Attributes:
        config_path: The config file the paths came from.
        config: The resolved app config.
        data_raw: The raw JSONL (read-only).
        data_splits: The split id folder.
        results: The run-artifact folder.
        split: The split on screen (``val``).
        corpus_run: The corpus map's run id.
        eval_run: The reference evaluation run id.
        surrogate_run: The surrogate attribution's run id.
    """

    config_path: Path
    config: AppConfig
    data_raw: Path
    data_splits: Path
    results: Path
    split: str
    corpus_run: str
    eval_run: str
    surrogate_run: str


@dataclass(frozen=True)
class Context:
    """Everything read from the sources: the split, the map, the artifacts.

    Attributes:
        sources: The resolved sources (the provenance's paths).
        records: The split's records, in the split's own order.
        train_records: The train split's records (the map's corpus).
        labels: The label space, sorted (the classifier's columns).
        gold: The split's gold matrix (rows x labels).
        corpus_map: The corpus map's artifact.
        eval_metrics: The reference evaluation artifact.
        surrogate: The surrogate attribution's artifact.
        tables: Every label's word table (the artifact's own rows).
        bucket_of_label: Each label's frequency bucket (from the eval artifact).
        val_ids: The split's record ids, in the analyzer's own order.
        train_ids: The train split's record ids.
        val_provenance: The val split file's provenance.
        train_provenance: The train split file's provenance.
    """

    sources: Sources
    records: tuple[Record, ...]
    train_records: tuple[Record, ...]
    labels: tuple[str, ...]
    gold: np.ndarray
    corpus_map: dict[str, Any]
    eval_metrics: dict[str, Any]
    surrogate: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    bucket_of_label: dict[str, str]
    val_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    val_provenance: dict[str, Any]
    train_provenance: dict[str, Any]


@dataclass(frozen=True)
class Provenance:
    """The tool's own provenance: every source, named, with its sha256.

    Attributes:
        config_path: The config file.
        data_path: The raw dataset.
        data_sha256: The raw dataset's sha256.
        split: The split on screen.
        split_path: The split's id file.
        split_sha256: The split file's source sha256.
        n_records: The split's text count.
        corpus_run: The corpus map's run id.
        corpus_path: The map's artifact.
        corpus_sha256: The map artifact file's sha256.
        corpus_source_sha256: The dataset sha the map was built from.
        corpus_split: The map's corpus split (must be ``train``).
        n_corpus_texts: The map's corpus size.
        eval_run: The reference evaluation run id.
        eval_path: The evaluation artifact.
        eval_sha256: The evaluation artifact file's sha256.
        surrogate_run: The surrogate's run id.
        surrogate_path: The surrogate artifact.
        surrogate_sha256: The surrogate artifact file's sha256.
        k: The decision cut (the run's own ``top_k``).
        near_miss_k: The near-miss window's end.
        seed: The config's seed.
    """

    config_path: str
    data_path: str
    data_sha256: str
    split: str
    split_path: str
    split_sha256: str
    n_records: int
    corpus_run: str
    corpus_path: str
    corpus_sha256: str
    corpus_source_sha256: str
    corpus_split: str
    n_corpus_texts: int
    eval_run: str
    eval_path: str
    eval_sha256: str
    surrogate_run: str
    surrogate_path: str
    surrogate_sha256: str
    k: int
    near_miss_k: int
    seed: int

    def lines(self) -> list[str]:
        """The provenance block, one labelled line per source."""
        return [
            f"config: {self.config_path}",
            f"data: {self.data_path} sha256 {self.data_sha256}",
            f"split: {self.split} ({self.split_path} sha256 {self.split_sha256}; "
            f"{self.n_records} texts)",
            f"corpus map: {self.corpus_path} sha256 {self.corpus_sha256}; "
            f"corpus split {self.corpus_split}, {self.n_corpus_texts} texts, "
            f"source sha256 {self.corpus_source_sha256}",
            f"evaluation: {self.eval_path} sha256 {self.eval_sha256}",
            f"surrogate: {self.surrogate_path} sha256 {self.surrogate_sha256}",
            f"cut: top_k k={self.k}; near-miss window 3..{self.near_miss_k}; seed {self.seed}",
            "note: UPR val only; UPR test untouched; results/ read-only.",
        ]


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
    split: str = DEFAULT_SPLIT,
    corpus_run: str = CORPUS_MAP_RUN,
    eval_run: str = EVAL_RUN,
    surrogate_run: str = SURROGATE_RUN,
) -> Sources:
    """Resolve the paths from the config (and the overrides).

    Args:
        config_path: The config file.
        results: The results folder (default: the config's ``paths.results``).
        split: The split on screen.
        corpus_run: The corpus map's run id.
        eval_run: The reference evaluation run id.
        surrogate_run: The surrogate's run id.

    Returns:
        The resolved sources.
    """
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
        data_raw=Path(config.paths.data_raw),
        data_splits=Path(config.paths.data_splits),
        results=results_path,
        split=split,
        corpus_run=corpus_run,
        eval_run=eval_run,
        surrogate_run=surrogate_run,
    )


def load_context(sources: Sources) -> Context:
    """Read the split, the gold matrix and the three artifacts.

    Raises:
        FileNotFoundError: If a source is missing.
        ValueError: If the split's records do not line up with the split file.
    """
    report = load_jsonl(sources.data_raw)
    val_ids, val_provenance = load_split_ids(sources.data_splits, sources.split)
    train_ids, train_provenance = load_split_ids(sources.data_splits, "train")
    records = select_records_by_ids(report.records, val_ids)
    train_records = select_records_by_ids(report.records, train_ids)
    if len(records) != len(val_ids):
        raise ValueError(f"the split has {len(val_ids)} ids but {len(records)} records")
    labels = tuple(sorted({label for record in report.records for label in record.labels}))
    gold = build_label_matrix(records, labels)
    corpus = read_json(sources.results / sources.corpus_run / "hitmap.json")
    eval_metrics = read_json(sources.results / sources.eval_run / "metrics.json")
    surrogate = read_json(sources.results / sources.surrogate_run / "hitmap.json")
    block = corpus.get("corpus_map") or {}
    tables = {
        str(row.get("label")): [dict(word) for word in row.get("words", [])]
        for row in block.get("per_label", [])
    }
    buckets = (eval_metrics.get("buckets") or {}).get("labels_by_bucket") or {}
    bucket_of_label = {
        str(label): str(bucket) for bucket, names in buckets.items() for label in names
    }
    return Context(
        sources=sources,
        records=tuple(records),
        train_records=tuple(train_records),
        labels=labels,
        gold=gold,
        corpus_map=corpus,
        eval_metrics=eval_metrics,
        surrogate=surrogate,
        tables=tables,
        bucket_of_label=bucket_of_label,
        val_ids=tuple(str(record.id) for record in records),
        train_ids=tuple(str(record.id) for record in train_records),
        val_provenance=dict(val_provenance),
        train_provenance=dict(train_provenance),
    )


def build_provenance(context: Context, *, near_miss_k: int) -> Provenance:
    """The tool's own provenance block from the loaded context."""
    sources = context.sources
    corpus_info = (context.corpus_map.get("corpus_map") or {}).get("corpus") or {}
    decision = context.eval_metrics.get("decision") or {}
    eval_path = sources.results / sources.eval_run / "metrics.json"
    corpus_path = sources.results / sources.corpus_run / "hitmap.json"
    surrogate_path = sources.results / sources.surrogate_run / "hitmap.json"
    split_path = sources.data_splits / f"{sources.split}_ids.json"
    return Provenance(
        config_path=str(sources.config_path),
        data_path=str(sources.data_raw),
        data_sha256=sha256_file(sources.data_raw),
        split=sources.split,
        split_path=str(split_path),
        split_sha256=str(context.val_provenance.get("source_sha256", "")),
        n_records=len(context.records),
        corpus_run=sources.corpus_run,
        corpus_path=str(corpus_path),
        corpus_sha256=sha256_file(corpus_path),
        corpus_source_sha256=str(context.corpus_map.get("source_sha256", "")),
        corpus_split=str(corpus_info.get("split", "")),
        n_corpus_texts=int(corpus_info.get("n_texts", 0)),
        eval_run=sources.eval_run,
        eval_path=str(eval_path),
        eval_sha256=sha256_file(eval_path),
        surrogate_run=sources.surrogate_run,
        surrogate_path=str(surrogate_path),
        surrogate_sha256=sha256_file(surrogate_path),
        k=int(decision.get("k") or 0),
        near_miss_k=near_miss_k,
        seed=int(sources.config.seed),
    )


# --- the classifier's arithmetic --------------------------------------------


def text_words(text: str) -> tuple[str, ...]:
    """The text's content words, in the text's order, deduplicated (the map's unit)."""
    return tuple(dict.fromkeys(content_tokens(text, DEFAULT_STOPWORDS)))


def map_scores(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    records: Sequence[Record],
    labels: Sequence[str],
) -> np.ndarray:
    """Every text's score against every label, from the artifact's own rows.

    The score is the map's ``sum_zscore_positive`` arm: the sum of the positive
    z-scores of the text's words in the label's table (``map_score``). A label
    with an empty table scores 0 here; the floor is added by
    :func:`decision_scores`.

    Returns:
        A ``(n_texts, n_labels)`` float64 matrix.
    """
    scores = np.zeros((len(records), len(labels)), dtype=np.float64)
    for row, record in enumerate(records):
        wanted = set(text_words(record.text))
        for column, label in enumerate(labels):
            rows = tables.get(label) or ()
            if rows:
                scores[row, column] = map_score(rows, wanted)
    return scores


def decision_scores(
    scores: np.ndarray,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Sequence[str],
) -> np.ndarray:
    """The scores the classifier ranks, with the empty tables' floor.

    The floor is the track's own silence rule (CS-31): a label with no train
    example has no evidence, so it ranks last in its row -- the row's smallest
    seen score minus ``FLOOR_GAP``. The floor is a **ranking device**, never a
    score (the cells' ``score`` stays 0.0).
    """
    out = np.array(scores, dtype=np.float64, copy=True)
    empty = [column for column, label in enumerate(labels) if not tables.get(label)]
    if empty:
        seen = np.delete(out, empty, axis=1)
        floor = (
            (seen.min(axis=1) - FLOOR_GAP) if seen.shape[1] else np.full(out.shape[0], -FLOOR_GAP)
        )
        out[:, empty] = floor[:, None]
    return out


def ranks(decisions: np.ndarray) -> np.ndarray:
    """Every cell's 1-based rank (descending, stable ties, earlier column wins)."""
    order = np.argsort(-np.asarray(decisions, dtype=np.float64), axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(1, order.shape[1] + 1)
    return rank


def top_k(decisions: np.ndarray, k: int) -> np.ndarray:
    """The harness's own ``top_k`` decision matrix (the exact rule the run used)."""
    return apply_decision_rule(decisions, rule="top_k", k=k)


def error_of(gold: bool, predicted: bool, rank: int, k: int, near_miss_k: int) -> str:
    """One cell's error type.

    Args:
        gold: The cell is true.
        predicted: The cell is in the top-k.
        rank: The classifier's rank of the label in the text.
        k: The decision cut.
        near_miss_k: The near-miss window's end (gold at ``k < rank <= K``).

    Returns:
        ``tp`` / ``fp`` / ``fn`` / ``near_miss`` / ``ok``.
    """
    if gold and predicted:
        return "tp"
    if not gold and predicted:
        return "fp"
    if gold and not predicted:
        return "near_miss" if k < rank <= near_miss_k else "fn"
    return "ok"


def text_f1(tp: int, fp: int, fn: int) -> float:
    """The per-text F1 from its ``(tp, fp, fn)`` counts (0 when the denominator is 0)."""
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def micro_f1_from_counts(counts: Iterable[Sequence[int]]) -> float:
    """The micro-F1 over per-text ``(tp, fp, fn)`` rows."""
    tp = sum(int(row[0]) for row in counts)
    fp = sum(int(row[1]) for row in counts)
    fn = sum(int(row[2]) for row in counts)
    return text_f1(tp, fp, fn)


def length_bucket(n_chars: int) -> str:
    """The analyzer's length bucket for a character count."""
    for edge, name in zip(LENGTH_EDGES, LENGTH_BUCKETS, strict=False):
        if n_chars < edge:
            return name
    return LENGTH_BUCKETS[-1]


# --- the three tables -------------------------------------------------------


@dataclass(frozen=True)
class TextRow:
    """One text: its labels, the classifier's pick, its counts and the cut."""

    text_id: str
    text: str
    n_chars: int
    length_bucket: str
    gold_labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    tp: int
    fp: int
    fn: int
    n_gold: int
    n_pred: int
    text_f1: float
    rank1_score: float
    rank2_score: float
    rank3_score: float
    margin: float
    n_evidence: int
    has_empty_slot: bool
    run_id: str
    split: str


@dataclass(frozen=True)
class CellRow:
    """One (text, label) cell: gold, predicted, score, rank, error."""

    text_id: str
    label: str
    gold: bool
    predicted: bool
    score: float
    rank: int
    error: str
    has_table: bool
    has_evidence: bool
    gap_to_cut: float
    run_id: str
    split: str


@dataclass(frozen=True)
class WordRow:
    """One (text, label, word): the positive z-score that gave the label points."""

    text_id: str
    label: str
    word: str
    z_score: float
    contribution: float
    log_odds: float
    n_with: int
    df: int
    word_class: str
    is_name_word: bool
    surrogate_coefficient: float | None
    run_id: str
    split: str


@dataclass(frozen=True)
class Tables:
    """The three tidy tables every view filters or joins."""

    texts: tuple[TextRow, ...]
    cells: tuple[CellRow, ...]
    words: tuple[WordRow, ...]


def surrogate_coefficients(
    surrogate: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], set[str]]:
    """The surrogate's full vector, read per topic (CS-24).

    Returns:
        The topic -> (word -> coefficient) mapping, and the model's vocabulary.
        Both empty when the artifact carries no full vector.
    """
    arms = (surrogate.get("ablation_map") or {}).get("arms") or {}
    if not arms:
        return {}, set()
    arm = next(iter(arms.values()))
    vector = (arm.get("surrogate") or {}).get("full_vector") or {}
    vocabulary = [str(word) for word in (vector.get("vocabulary") or [])]
    topics = [str(topic) for topic in (vector.get("topics") or [])]
    rows = list(vector.get("coefficients") or [])
    coefficients: dict[str, dict[str, float]] = {}
    for topic, row in zip(topics, rows, strict=False):
        if not isinstance(row, list) or len(row) != len(vocabulary):
            continue
        coefficients[topic] = {
            word: float(value) for word, value in zip(vocabulary, row, strict=True)
        }
    return coefficients, set(vocabulary)


def build_tables(
    context: Context,
    *,
    k: int = EXPECTED_K,
    near_miss_k: int = DEFAULT_NEAR_MISS_K,
) -> Tables:
    """Build the three tables from the loaded context.

    The words table is the exact decomposition of the cells' scores: every
    (text, label, word) whose stored z-score is positive, so the contributions
    sum to the cell's score (the test pins it).

    Raises:
        ValueError: On a ``near_miss_k`` below ``k``, or if the prediction
            matrix and the ranks disagree.
    """
    if near_miss_k < k:
        raise ValueError(f"near_miss_k must be >= k, got {near_miss_k} < {k}")
    labels = context.labels
    scores = map_scores(context.tables, context.records, labels)
    decisions = decision_scores(scores, context.tables, labels)
    rank = ranks(decisions)
    predicted = top_k(decisions, k)
    if not np.array_equal(predicted, rank <= k):
        raise ValueError("the harness's top_k rule and the ranks disagree")
    order = np.argsort(-decisions, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(decisions, order, axis=1)
    cut_scores = sorted_scores[:, k - 1]
    coefficients, vocabulary = surrogate_coefficients(context.surrogate)
    by_word = {
        label: {str(row.get("word")): row for row in rows} for label, rows in context.tables.items()
    }

    texts: list[TextRow] = []
    cells: list[CellRow] = []
    words: list[WordRow] = []
    run_id = context.sources.eval_run
    split = context.sources.split
    for row, record in enumerate(context.records):
        gold_columns = np.flatnonzero(context.gold[row])
        pred_columns = np.flatnonzero(predicted[row])
        gold_labels = tuple(labels[int(column)] for column in record_labels(context, record))
        predicted_labels = tuple(labels[int(column)] for column in pred_columns)
        gold_set = {labels[int(column)] for column in gold_columns}
        pred_set = set(predicted_labels)
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        n_evidence = int((scores[row] > 0).sum())
        empty_slot = any(scores[row, int(column)] == 0.0 for column in pred_columns)
        own_words = text_words(record.text)
        for column, label in enumerate(labels):
            gold = bool(context.gold[row, column])
            is_predicted = bool(predicted[row, column])
            score = float(scores[row, column])
            cell_rank = int(rank[row, column])
            cells.append(
                CellRow(
                    text_id=str(record.id),
                    label=label,
                    gold=gold,
                    predicted=is_predicted,
                    score=score,
                    rank=cell_rank,
                    error=error_of(gold, is_predicted, cell_rank, k, near_miss_k),
                    has_table=bool(context.tables.get(label)),
                    has_evidence=score > 0.0,
                    gap_to_cut=score - float(cut_scores[row]),
                    run_id=run_id,
                    split=split,
                )
            )
            table = by_word.get(label, {})
            if not table:
                continue
            topic_coefficients = coefficients.get(label, {})
            for word in own_words:
                entry = table.get(word)
                if entry is None:
                    continue
                z_score = float(entry.get("z_score", 0.0))
                if z_score <= 0.0:
                    continue
                coefficient = topic_coefficients.get(word) if word in vocabulary else None
                words.append(
                    WordRow(
                        text_id=str(record.id),
                        label=label,
                        word=word,
                        z_score=z_score,
                        contribution=z_score,
                        log_odds=float(entry.get("log_odds", 0.0)),
                        n_with=int(entry.get("n_with", 0)),
                        df=int(entry.get("df", 0)),
                        word_class=str(entry.get("word_class", "")),
                        is_name_word=bool(entry.get("is_name_word", False)),
                        surrogate_coefficient=coefficient,
                        run_id=run_id,
                        split=split,
                    )
                )
        texts.append(
            TextRow(
                text_id=str(record.id),
                text=record.text,
                n_chars=len(record.text),
                length_bucket=length_bucket(len(record.text)),
                gold_labels=gold_labels,
                predicted_labels=predicted_labels,
                tp=tp,
                fp=fp,
                fn=fn,
                n_gold=len(gold_set),
                n_pred=len(pred_set),
                text_f1=text_f1(tp, fp, fn),
                rank1_score=float(sorted_scores[row, 0]),
                rank2_score=float(sorted_scores[row, 1]),
                rank3_score=float(sorted_scores[row, 2]),
                margin=float(sorted_scores[row, 1] - sorted_scores[row, 2]),
                n_evidence=n_evidence,
                has_empty_slot=empty_slot,
                run_id=run_id,
                split=split,
            )
        )
    _check_contributions(cells, words)
    return Tables(texts=tuple(texts), cells=tuple(cells), words=tuple(words))


def record_labels(context: Context, record: Record) -> list[int]:
    """The record's gold labels' columns, in the record's own order."""
    index = {label: column for column, label in enumerate(context.labels)}
    return [index[label] for label in record.labels if label in index]


def _check_contributions(cells: Sequence[CellRow], words: Sequence[WordRow]) -> None:
    """The words' contributions sum to the cells' scores (a build-time invariant)."""
    totals: Counter[tuple[str, str]] = Counter()
    for row in words:
        totals[(row.text_id, row.label)] += row.contribution
    for cell in cells:
        if abs(totals.get((cell.text_id, cell.label), 0.0) - cell.score) > SUM_TOLERANCE:
            raise ValueError(
                f"the words do not sum to the score for {cell.text_id!r}/{cell.label!r}: "
                f"{totals.get((cell.text_id, cell.label), 0.0)} vs {cell.score}"
            )


# --- the views' aggregates --------------------------------------------------


@dataclass(frozen=True)
class LabelRateRow:
    """One label's TP/FP/FN/near-miss counts and rates."""

    label: str
    bucket: str
    n_gold: int
    n_non_gold: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    near_miss: int
    precision: float
    recall: float
    miss_rate: float
    fp_rate: float


@dataclass(frozen=True)
class ConfusionRow:
    """One ``gold A -> predicted B`` confusion pair."""

    gold_label: str
    predicted_label: str
    n: int
    support: int
    share: float


@dataclass(frozen=True)
class WordImpactRow:
    """One word's share of the predicted cells: how many were false positives."""

    word: str
    n_cells: int
    n_tp: int
    n_fp: int
    mean_z: float
    precision: float


@dataclass(frozen=True)
class CoverageHoles:
    """The labels the classifier never reaches."""

    never_predicted: tuple[str, ...]
    never_predicted_with_evidence: tuple[str, ...]
    no_table: tuple[str, ...]


@dataclass(frozen=True)
class BucketRateRow:
    """One bucket's error counts and micro-F1 (near-miss counts as a miss)."""

    kind: str
    bucket: str
    n_cells: int
    tp: int
    fp: int
    fn: int
    near_miss: int
    micro_f1: float


def label_rates(
    cells: Sequence[CellRow],
    labels: Sequence[str],
    bucket_of_label: Mapping[str, str] | None = None,
) -> tuple[LabelRateRow, ...]:
    """Every label's TP/FP/FN/near-miss counts, with the rates the viewer shows.

    The miss rate is ``(fn + near_miss) / n_gold``; the FP rate is
    ``fp / n_non_gold``; precision counts the zero-score tie-break picks as the
    predictions they are.
    """
    n_texts = len({row.text_id for row in cells})
    labels = tuple(labels)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    near: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    gold: Counter[str] = Counter()
    for row in cells:
        if row.gold:
            gold[row.label] += 1
        if row.predicted:
            predicted[row.label] += 1
        if row.error == "tp":
            tp[row.label] += 1
        elif row.error == "fp":
            fp[row.label] += 1
        elif row.error == "fn":
            fn[row.label] += 1
        elif row.error == "near_miss":
            near[row.label] += 1
    rates: list[LabelRateRow] = []
    for label in labels:
        n_gold = gold[label]
        n_non_gold = n_texts - n_gold
        rates.append(
            LabelRateRow(
                label=label,
                bucket=str((bucket_of_label or {}).get(label, "")),
                n_gold=n_gold,
                n_non_gold=n_non_gold,
                n_pred=predicted[label],
                tp=tp[label],
                fp=fp[label],
                fn=fn[label],
                near_miss=near[label],
                precision=(tp[label] / predicted[label]) if predicted[label] else 0.0,
                recall=(tp[label] / n_gold) if n_gold else 0.0,
                miss_rate=((fn[label] + near[label]) / n_gold) if n_gold else 0.0,
                fp_rate=(fp[label] / n_non_gold) if n_non_gold else 0.0,
            )
        )
    return tuple(rates)


def label_confusion(cells: Sequence[CellRow]) -> tuple[ConfusionRow, ...]:
    """The ``gold A -> predicted B`` pairs, strongest first (A != B)."""
    gold_by_text: dict[str, set[str]] = {}
    pred_by_text: dict[str, set[str]] = {}
    support: Counter[str] = Counter()
    for row in cells:
        if row.gold:
            gold_by_text.setdefault(row.text_id, set()).add(row.label)
            support[row.label] += 1
        if row.predicted:
            pred_by_text.setdefault(row.text_id, set()).add(row.label)
    counts: Counter[tuple[str, str]] = Counter()
    for text_id, gold in gold_by_text.items():
        for predicted in pred_by_text.get(text_id, set()):
            if predicted not in gold:
                for gold_label in gold:
                    counts[(gold_label, predicted)] += 1
    rows = [
        ConfusionRow(
            gold_label=gold_label,
            predicted_label=predicted,
            n=n,
            support=support[gold_label],
            share=(n / support[gold_label]) if support[gold_label] else 0.0,
        )
        for (gold_label, predicted), n in counts.items()
    ]
    rows.sort(key=lambda row: (-row.n, row.gold_label, row.predicted_label))
    return tuple(rows)


def false_positive_words(
    cells: Sequence[CellRow],
    words: Sequence[WordRow],
) -> tuple[WordImpactRow, ...]:
    """The words that carry the predicted cells, split by TP and FP.

    Only predicted cells count: a word contributing to a label the classifier
    did not pick did not drive that prediction. The rows are sorted by the
    number of FP cells, so the words that most often filled a wrong pick lead.
    """
    by_key = {(row.text_id, row.label): row for row in cells}
    n_tp: Counter[str] = Counter()
    n_fp: Counter[str] = Counter()
    n_cells: Counter[str] = Counter()
    z_sum: Counter[str] = Counter()
    for row in words:
        cell = by_key.get((row.text_id, row.label))
        if cell is None or not cell.predicted:
            continue
        n_cells[row.word] += 1
        z_sum[row.word] += row.contribution
        if cell.gold:
            n_tp[row.word] += 1
        else:
            n_fp[row.word] += 1
    rows = [
        WordImpactRow(
            word=word,
            n_cells=count,
            n_tp=n_tp[word],
            n_fp=n_fp[word],
            mean_z=(z_sum[word] / count) if count else 0.0,
            precision=(n_tp[word] / count) if count else 0.0,
        )
        for word, count in n_cells.items()
    ]
    rows.sort(key=lambda row: (-row.n_fp, -row.n_cells, row.word))
    return tuple(rows)


def coverage_holes(cells: Sequence[CellRow], labels: Sequence[str]) -> CoverageHoles:
    """The labels never predicted, never predicted with evidence, and with no table."""
    predicted: set[str] = set()
    predicted_with_evidence: set[str] = set()
    no_table: set[str] = set()
    for row in cells:
        if row.predicted:
            predicted.add(row.label)
            if row.has_evidence:
                predicted_with_evidence.add(row.label)
        if not row.has_table:
            no_table.add(row.label)
    return CoverageHoles(
        never_predicted=tuple(label for label in labels if label not in predicted),
        never_predicted_with_evidence=tuple(
            label for label in labels if label not in predicted_with_evidence
        ),
        no_table=tuple(label for label in labels if label in no_table),
    )


def bucket_rates(
    texts: Sequence[TextRow],
    cells: Sequence[CellRow],
    labels: Sequence[str],
    bucket_of_label: Mapping[str, str] | None = None,
) -> tuple[BucketRateRow, ...]:
    """The error counts by length bucket and by frequency bucket."""
    length_by_text = {row.text_id: row.length_bucket for row in texts}
    rows: list[BucketRateRow] = []
    for kind, groups in (
        ("length", length_by_text),
        ("frequency", {label: str((bucket_of_label or {}).get(label, "")) for label in labels}),
    ):
        counts: dict[str, list[int]] = {}
        for row in cells:
            key = length_by_text[row.text_id] if kind == "length" else groups[row.label]
            if not key:
                continue
            acc = counts.setdefault(key, [0, 0, 0, 0, 0])  # cells, tp, fp, fn, near
            acc[0] += 1
            if row.error == "tp":
                acc[1] += 1
            elif row.error == "fp":
                acc[2] += 1
            elif row.error == "fn":
                acc[3] += 1
            elif row.error == "near_miss":
                acc[4] += 1
        order = LENGTH_BUCKETS if kind == "length" else ("head", "torso", "tail", "unseen")
        for bucket in order:
            if bucket not in counts:
                continue
            n_cells, tp, fp, fn, near = counts[bucket]
            rows.append(
                BucketRateRow(
                    kind=kind,
                    bucket=bucket,
                    n_cells=n_cells,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    near_miss=near,
                    micro_f1=text_f1(tp, fp, fn + near),
                )
            )
    return tuple(rows)


# --- compare runs -----------------------------------------------------------


@dataclass(frozen=True)
class RunSeries:
    """One run's one arm: its per-text counts and its own overall metric."""

    run_id: str
    arm: str
    micro_f1: float
    k: int
    counts: tuple[tuple[int, int, int], ...]

    @property
    def label(self) -> str:
        """The series' display name."""
        return f"{self.run_id} :: {self.arm}"


@dataclass(frozen=True)
class SeriesCatalog:
    """The comparable series, the refused ones, and the not-comparable ones."""

    series: tuple[RunSeries, ...]
    rejected: tuple[str, ...]
    ineligible: tuple[str, ...]


@dataclass(frozen=True)
class CompareRow:
    """One text's change between two series."""

    text_id: str
    f1_a: float
    f1_b: float
    delta_f1: float
    delta_tp: int
    delta_fp: int
    delta_fn: int
    improved: bool
    worsened: bool


def load_run_series(
    results: Path,
    *,
    expected_n: int,
    expected_split: str = DEFAULT_SPLIT,
) -> SeriesCatalog:
    """Every comparable ``upr-analyze-*`` series (and the refused ones).

    A series is comparable when its artifact is the val split, its arm carries
    ``per_text.f1_counts`` of the expected length, and the counts sum to the
    arm's own ``micro_f1`` at 1e-9. A failed series is **rejected** with its
    reason, never silently dropped; an arm without predictions (older runs) is
    merely not comparable.

    Args:
        results: The results folder.
        expected_n: The split's text count.
        expected_split: The split every series must be on.
    """
    series: list[RunSeries] = []
    rejected: list[str] = []
    ineligible: list[str] = []
    for path in sorted(results.glob("upr-analyze-*/analysis.json")):
        run_id = path.parent.name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rejected.append(f"{run_id}: unreadable analysis ({exc})")
            continue
        if payload.get("split") != expected_split:
            ineligible.append(f"{run_id}: split {payload.get('split')!r}, not {expected_split!r}")
            continue
        arms = payload.get("arms") or {}
        k = int(((payload.get("decision") or {}).get("k")) or 0)
        for arm_name, arm in arms.items():
            block = (arm or {}).get("per_text") or {}
            counts = block.get("f1_counts")
            label = f"{run_id} :: {arm_name}"
            if not counts:
                ineligible.append(f"{label}: no per_text.f1_counts")
                continue
            if len(counts) != expected_n:
                rejected.append(f"{label}: {len(counts)} per-text rows, expected {expected_n}")
                continue
            rows: list[tuple[int, int, int]] = []
            malformed = False
            for item in counts:
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    malformed = True
                    break
                row = (int(item[0]), int(item[1]), int(item[2]))
                if min(row) < 0:
                    malformed = True
                    break
                rows.append(row)
            if malformed:
                rejected.append(f"{label}: a per-text count is not [tp, fp, fn]")
                continue
            metric = (
                (((arm or {}).get("metrics") or {}).get("overall") or {}).get("f1") or {}
            ).get("micro_f1")
            if metric is None:
                rejected.append(f"{label}: no metrics.overall.f1.micro_f1")
                continue
            recomputed = micro_f1_from_counts(rows)
            if abs(recomputed - float(metric)) > F1_TOLERANCE:
                rejected.append(
                    f"{label}: counts sum to {recomputed:.12f}, the artifact says {metric}"
                )
                continue
            series.append(
                RunSeries(
                    run_id=run_id,
                    arm=str(arm_name),
                    micro_f1=float(metric),
                    k=k,
                    counts=tuple(rows),
                )
            )
    return SeriesCatalog(
        series=tuple(series), rejected=tuple(rejected), ineligible=tuple(ineligible)
    )


def dictionary_series(context: Context, tables: Tables, *, k: int = EXPECTED_K) -> RunSeries:
    """The dictionary run as a comparable series: its counts are recomputed."""
    counts = tuple((row.tp, row.fp, row.fn) for row in tables.texts)
    overall = (context.eval_metrics.get("report") or {}).get("overall") or {}
    micro = float((overall.get("f1") or {}).get("micro_f1") or 0.0)
    return RunSeries(
        run_id=context.sources.eval_run,
        arm="dictionary (recomputed)",
        micro_f1=micro,
        k=k,
        counts=counts,
    )


def compare_series(
    a: RunSeries,
    b: RunSeries,
    text_ids: Sequence[str],
) -> tuple[CompareRow, ...]:
    """Every text's change from series ``a`` to series ``b``.

    Raises:
        ValueError: If the two series use different cuts or hold different
            numbers of texts.
    """
    if a.k != b.k:
        raise ValueError(f"the two series use different cuts: {a.k} and {b.k}")
    if len(a.counts) != len(b.counts) or len(a.counts) != len(text_ids):
        raise ValueError(
            f"the series hold {len(a.counts)} and {len(b.counts)} texts, expected {len(text_ids)}"
        )
    rows: list[CompareRow] = []
    for text_id, left, right in zip(text_ids, a.counts, b.counts, strict=True):
        f1_a = text_f1(*left)
        f1_b = text_f1(*right)
        delta = f1_b - f1_a
        rows.append(
            CompareRow(
                text_id=text_id,
                f1_a=f1_a,
                f1_b=f1_b,
                delta_f1=delta,
                delta_tp=right[0] - left[0],
                delta_fp=right[1] - left[1],
                delta_fn=right[2] - left[2],
                improved=delta > 0.0,
                worsened=delta < 0.0,
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
class GuardReport:
    """The five checks, the two micro-F1 numbers, and the provenance."""

    checks: tuple[Check, ...]
    artifact_micro_f1: float
    recomputed_micro_f1: float
    provenance: Provenance

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
            "micro-F1: artifact "
            f"{self.artifact_micro_f1:.16f} vs recomputed {self.recomputed_micro_f1:.16f}"
            f" (atol {F1_TOLERANCE:g})"
        )
        output.append(f"RESULT: {'OK' if self.ok else 'MISMATCH'}")
        return output


def _surrogate_check(context: Context, split: str) -> Check:
    """The surrogate artifact's shape against the label space."""
    arms = (context.surrogate.get("ablation_map") or {}).get("arms") or {}
    if not arms:
        return Check("surrogate", False, "the artifact carries no ablation_map.arms")
    arm_name = next(iter(arms))
    vector = (arms[arm_name].get("surrogate") or {}).get("full_vector")
    if not isinstance(vector, Mapping):
        return Check(
            "surrogate",
            True,
            f"arm {arm_name!r} carries no full_vector: the model column is omitted",
        )
    vocabulary = list(vector.get("vocabulary") or [])
    topics = [str(topic) for topic in (vector.get("topics") or [])]
    rows = list(vector.get("coefficients") or [])
    problems: list[str] = []
    if context.surrogate.get("split") != split:
        problems.append(f"measured on split {context.surrogate.get('split')!r}, not {split!r}")
    if set(topics) != set(context.labels):
        problems.append("the topics are not the label space")
    if len(rows) != len(topics):
        problems.append(f"{len(rows)} coefficient rows for {len(topics)} topics")
    if any(not isinstance(row, list) or len(row) != len(vocabulary) for row in rows):
        problems.append("a coefficient row's length is not the vocabulary's length")
    detail = (
        f"arm {arm_name!r}: {len(topics)} topics, {len(vocabulary)} vocabulary words, "
        f"split {context.surrogate.get('split')!r}"
        if not problems
        else "; ".join(problems)
    )
    return Check("surrogate", not problems, detail)


def _series_check(context: Context) -> Check:
    """Every comparable analysis series reproduces its own artifact's metric."""
    catalog = load_run_series(
        context.sources.results,
        expected_n=len(context.records),
        expected_split=context.sources.split,
    )
    if catalog.rejected:
        shown = "; ".join(catalog.rejected[:3])
        more = f"; +{len(catalog.rejected) - 3} more" if len(catalog.rejected) > 3 else ""
        return Check(
            "compare series",
            False,
            f"{len(catalog.rejected)} refused: {shown}{more}",
        )
    return Check(
        "compare series",
        True,
        f"{len(catalog.series)} series validated (counts sum to their own micro_f1); "
        f"{len(catalog.ineligible)} arms are not comparable",
    )


def check_all(
    sources: Sources,
    *,
    near_miss_k: int = DEFAULT_NEAR_MISS_K,
    context: Context | None = None,
) -> GuardReport:
    """Run the five guard checks and return the report (never raises on a mismatch)."""
    context = context or load_context(sources)
    labels = context.labels
    scores = map_scores(context.tables, context.records, labels)
    decisions = decision_scores(scores, context.tables, labels)
    decision = context.eval_metrics.get("decision") or {}
    k = int(decision.get("k") or 0)
    predicted = top_k(decisions, k) if k >= 1 else np.zeros_like(decisions, dtype=bool)
    tp = int((predicted & context.gold).sum())
    fp = int((predicted & ~context.gold).sum())
    fn = int((~predicted & context.gold).sum())
    recomputed_f1 = text_f1(tp, fp, fn)
    artifact_f1 = float(
        (((context.eval_metrics.get("report") or {}).get("overall") or {}).get("f1") or {}).get(
            "micro_f1"
        )
        or 0.0
    )
    data_sha = sha256_file(sources.data_raw)
    corpus_info = (context.corpus_map.get("corpus_map") or {}).get("corpus") or {}
    per_label = (context.corpus_map.get("corpus_map") or {}).get("per_label") or []
    map_labels = {str(row.get("label")) for row in per_label}
    empty_tables = {str(row.get("label")) for row in per_label if not row.get("words")}
    holdout_path = sources.data_splits / "label_holdout.json"
    holdout: set[str] = set()
    holdout_problem = ""
    if holdout_path.is_file():
        holdout = {str(label) for label in (read_json(holdout_path).get("labels") or [])}
    else:
        holdout_problem = f"no label_holdout file at {holdout_path}"

    checks: list[Check] = []
    checks.append(
        Check(
            "reproduction",
            k == EXPECTED_K and abs(recomputed_f1 - artifact_f1) <= F1_TOLERANCE,
            f"artifact {artifact_f1:.16f} ({sources.eval_run}/metrics.json) vs recomputed "
            f"{recomputed_f1:.16f} (map_score + top_k, k={k}; {len(context.records)} texts, "
            f"{int(context.gold.sum())} gold cells, {int(predicted.sum())} predicted cells; "
            f"atol {F1_TOLERANCE:g})",
        )
    )
    map_problems: list[str] = []
    if corpus_info.get("split") != "train":
        map_problems.append(f"corpus split is {corpus_info.get('split')!r}, not 'train'")
    if context.corpus_map.get("source_sha256") != data_sha:
        map_problems.append("the map's source_sha256 is not the raw dataset's sha256")
    if int(corpus_info.get("n_texts", 0)) != len(context.train_ids):
        map_problems.append(
            f"the map's corpus has {corpus_info.get('n_texts')} texts, "
            f"the train split {len(context.train_ids)}"
        )
    if map_labels != set(labels):
        map_problems.append("the map's labels are not the dataset's label space")
    if holdout_problem:
        map_problems.append(holdout_problem)
    elif empty_tables != holdout:
        map_problems.append("the empty tables are not the held-out labels")
    train_sha = str(context.train_provenance.get("source_sha256", ""))
    if train_sha and train_sha != data_sha:
        map_problems.append("the train split's source_sha256 is not the raw dataset's sha256")
    checks.append(
        Check(
            "map provenance",
            not map_problems,
            "\n     ".join(map_problems)
            if map_problems
            else f"train corpus, {len(context.train_ids)} texts, source sha256 matches, "
            f"{len(empty_tables)} empty tables = the held-out labels",
        )
    )
    eval_problems: list[str] = []
    if context.eval_metrics.get("split") != sources.split:
        eval_problems.append(
            f"measured on split {context.eval_metrics.get('split')!r}, not {sources.split!r}"
        )
    if context.eval_metrics.get("run_id") != sources.eval_run:
        eval_problems.append(
            f"run_id {context.eval_metrics.get('run_id')!r} is not the folder's "
            f"{sources.eval_run!r}"
        )
    if k != EXPECTED_K:
        eval_problems.append(f"decision.k is {k}, expected {EXPECTED_K}")
    if int(context.eval_metrics.get("n_records", 0)) != len(context.records):
        eval_problems.append(
            f"n_records is {context.eval_metrics.get('n_records')}, "
            f"the split has {len(context.records)}"
        )
    if int(context.eval_metrics.get("n_labels", 0)) != len(labels):
        eval_problems.append(
            f"n_labels is {context.eval_metrics.get('n_labels')}, the data has {len(labels)}"
        )
    overall = (context.eval_metrics.get("report") or {}).get("overall") or {}
    if int(overall.get("n_gold_cells", 0)) != int(context.gold.sum()):
        eval_problems.append(
            f"n_gold_cells is {overall.get('n_gold_cells')}, "
            f"the split has {int(context.gold.sum())}"
        )
    if int((overall.get("f1") or {}).get("n_predicted_cells", 0)) != int(predicted.sum()):
        eval_problems.append(
            f"n_predicted_cells is {(overall.get('f1') or {}).get('n_predicted_cells')}, "
            f"the tool predicts {int(predicted.sum())}"
        )
    checks.append(
        Check(
            "evaluation artifact",
            not eval_problems,
            "\n     ".join(eval_problems)
            if eval_problems
            else f"val split, k={k}, {len(context.records)} texts, {len(labels)} labels, "
            f"{int(context.gold.sum())} gold cells, {int(predicted.sum())} predicted cells",
        )
    )
    checks.append(_surrogate_check(context, sources.split))
    checks.append(_series_check(context))
    return GuardReport(
        checks=tuple(checks),
        artifact_micro_f1=artifact_f1,
        recomputed_micro_f1=recomputed_f1,
        provenance=build_provenance(context, near_miss_k=near_miss_k),
    )


# --- the one-text helpers ---------------------------------------------------


def evidence_ranking(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    text: str,
    labels: Sequence[str],
) -> tuple[DictionaryRow, ...]:
    """The dictionary's evidence-only ranking (CS-47), for the one-text panel.

    It omits empty tables and zero-score labels; the classifier's prediction
    ranking is the ``cells`` table's ``rank``.
    """
    wanted = list(text_words(text))
    return dictionary_ranking(tables, wanted, labels=labels)


def contrast_label(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    text: str,
    gold_labels: Sequence[str],
    labels: Sequence[str],
) -> str:
    """The map's runner-up among the labels the text does not carry (CS-40)."""
    return default_contrast_label(
        tables,
        list(text_words(text)),
        exclude=gold_labels,
        labels=labels,
    )


@dataclass(frozen=True)
class HitmapRow:
    """One content word of a text and its stored z-score in each label's table.

    A ``None`` cell means the label's table does **not** hold the word: the map
    is silent about that pair, and the scorer counts it as 0.0. It is kept
    distinct from a stored 0.0 so the viewer never claims a measured zero.
    """

    word: str
    n_occurrences: int
    z_scores: tuple[float | None, ...]


def hitmap_rows(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    text: str,
    labels: Sequence[str],
) -> tuple[HitmapRow, ...]:
    """The text's content words (deduplicated, text order), one row each.

    Every word gets a cell per label: the stored ``z_score`` when the label's
    table holds the pair, and ``None`` when it does not (the map's silence
    convention — the scorer contributes 0.0, but the viewer says so instead of
    printing a zero). Words absent from every table are rows too, so "every
    word of the text" is on the screen.

    Args:
        tables: Every label's table (the artifact's own rows).
        text: The text on screen.
        labels: The labels to read, in the display's order.

    Returns:
        One :class:`HitmapRow` per distinct content word, in the text's order.
    """
    tokens = list(content_tokens(text, DEFAULT_STOPWORDS))
    counts = Counter(tokens)
    by_label = {
        label: {str(row.get("word")): row for row in (tables.get(label) or ())} for label in labels
    }
    rows: list[HitmapRow] = []
    for word in dict.fromkeys(tokens):
        values = []
        for label in labels:
            entry = by_label[label].get(word)
            values.append(None if entry is None else float(entry.get("z_score", 0.0)))
        rows.append(
            HitmapRow(
                word=word,
                n_occurrences=counts[word],
                z_scores=tuple(values),
            )
        )
    return tuple(rows)


# --- export -----------------------------------------------------------------


def _value(row: Any, field: str) -> Any:
    """One row's field value (a dataclass, a mapping or an attribute)."""
    if isinstance(row, Mapping):
        value = row.get(field, "")
    elif is_dataclass(row) and not isinstance(row, type):
        value = asdict(row).get(field, "")
    else:
        value = getattr(row, field, "")
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
    """The rows as CSV, with an optional ``#``-prefixed provenance block.

    Args:
        rows: The rows (dataclasses or mappings).
        fields: The columns, in order.
        provenance: The provenance lines to put at the top (as comments).
    """
    buffer = io.StringIO()
    for line in provenance:
        buffer.write(f"# {line}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([_value(row, field) for field in fields])
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
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--corpus-run", default=CORPUS_MAP_RUN)
    parser.add_argument("--eval-run", default=EVAL_RUN)
    parser.add_argument("--surrogate-run", default=SURROGATE_RUN)
    parser.add_argument(
        "--near-miss-k",
        type=int,
        default=DEFAULT_NEAR_MISS_K,
        help=f"the near-miss window's end (default: {DEFAULT_NEAR_MISS_K})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the guard and print its report (exit 0/1)",
    )
    parser.add_argument(
        "--dump",
        choices=("texts", "cells", "words"),
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
            split=args.split,
            corpus_run=args.corpus_run,
            eval_run=args.eval_run,
            surrogate_run=args.surrogate_run,
        )
        if not (args.check or args.dump or args.provenance):
            _parse_args(["--help"])
            return 0
        context = load_context(sources)
        report = check_all(sources, near_miss_k=args.near_miss_k, context=context)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error_view: {exc}", file=sys.stderr)
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
    tables = build_tables(context, k=report.provenance.k, near_miss_k=args.near_miss_k)
    table = {"texts": tables.texts, "cells": tables.cells, "words": tables.words}[args.dump]
    if args.limit is not None:
        table = table[: args.limit]
    fields = {
        "texts": (
            "text_id",
            "n_chars",
            "length_bucket",
            "gold_labels",
            "predicted_labels",
            "tp",
            "fp",
            "fn",
            "text_f1",
            "margin",
        ),
        "cells": (
            "text_id",
            "label",
            "gold",
            "predicted",
            "score",
            "rank",
            "error",
            "has_evidence",
            "gap_to_cut",
        ),
        "words": (
            "text_id",
            "label",
            "word",
            "z_score",
            "log_odds",
            "n_with",
            "df",
            "word_class",
            "is_name_word",
            "surrogate_coefficient",
        ),
    }[args.dump]
    print(to_csv(table, fields, provenance=report.provenance.lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
