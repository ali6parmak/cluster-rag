"""The text-processing track's UPR analysis driver (``analyze-upr``).

The UPR counterpart of :mod:`cluster_rag.omct_analysis`: the text-processing
track's Stage-5 (CS-8a) artifact, one JSON file per run, that characterizes
the UPR corpus **as a multi-label classification benchmark** and measures the
track's methods on it.

The UPR side differs from the OMCT side in one structural way, and the whole
design follows from it: the OMCT track had to **sample** a control (the
unjudged same-document cells, with a seed and a quota per document), because
its grid is 44,467 cells per document set and only 3.4% of it is judged. UPR's
grid is small and **fully enumerable** — 3,000 texts x 68 labels — and the gold
is the record's own label set, so the **non-gold cells are the background**:
complete, exact, and needing no seed. The artifact says so explicitly, so
nobody reads a UPR number as a sampled one.

What the artifact carries, per arm:

- the **UPR harness's own metrics** (the decision view: micro / macro /
  sample F1 and the coverage; the ranking view: P@k, R@k, nDCG@k and the
  shortlist recall) — overall, per **label bucket** (head / torso / tail /
  unseen) and per **text-length bucket** (the axis the chunking question is
  really about on UPR: a pooled number hides the long-text tail);
- the **gold side**: every gold cell's rank percentile inside its own text's
  label list (0 = best), the top-fifth share, and the hit@1 / hit@10 ("is the
  gold label among the text's top-k labels?");
- the **background side**: the same numbers for the non-gold cells, plus the
  **level and rank shift** against the whole-text reference arm — the
  control-side counterpart every headline number carries (the track's
  reporting rule);
- the **top-1 provenance** analogue: per text, how many of its top labels are
  gold and how many are not;
- the score distributions per bucket and per label.

The **whole-text reference arm** is the main pipeline's own scoring of the
same split with the same predictor: it must reproduce the ``evaluate`` run's
numbers field by field (the guard, the UPR analogue of
``tools/verify_analysis_baseline.py``).

Like :mod:`cluster_rag.omct_analysis`, this module orchestrates the pure
pieces (the track's pure core and the UPR harness's metrics) and never
imports a model client: the CLI wires the backends.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.data import Record, build_label_matrix
from cluster_rag.eval.multilabel import apply_decision_rule
from cluster_rag.eval.report import build_label_report
from cluster_rag.text_processors.upr_diagnostics import (
    cell_rank_percentiles,
    distribution,
    gold_hit_at_k,
    length_bucket_names,
    length_buckets,
    provenance,
    top_fifth_share,
)

logger = logging.getLogger(__name__)

# The name of the whole-text reference arm (the pipeline's own scoring).
BASELINE_ARM = "full_text"

# The hit@k grid of the gold-side report.
HIT_KS: tuple[int, ...] = (1, 10)

# The provenance's k grid (how many of a text's top labels are gold).
PROVENANCE_KS: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True)
class UprView:
    """The model-independent view of the UPR corpus.

    Holds the evaluated records, the label space (the sorted unique labels),
    the gold matrix, the label buckets, the texts' lengths and their length
    buckets. The model-independent part of the artifact is built from this
    object; the scores are per arm.
    """

    records: tuple[Record, ...]
    labels: tuple[str, ...]
    gold: np.ndarray
    buckets: Mapping[str, str]
    labels_by_bucket: Mapping[str, list[str]]
    lengths: tuple[int, ...]
    length_bucket_index: tuple[int, ...]
    length_bucket_names: tuple[str, ...]

    @property
    def n_texts(self) -> int:
        """The number of evaluated texts."""
        return len(self.records)

    @property
    def n_labels(self) -> int:
        """The size of the label space."""
        return len(self.labels)

    @property
    def n_gold_cells(self) -> int:
        """The number of gold (text, label) cells."""
        return int(self.gold.sum())

    @property
    def n_background_cells(self) -> int:
        """The number of non-gold cells (the background — fully enumerable)."""
        return int((~self.gold).sum())


def build_upr_view(
    records: Sequence[Record],
    labels: Sequence[str],
    buckets: Mapping[str, str],
    length_edges: Sequence[int],
) -> UprView:
    """Build the model-independent view of the evaluated split.

    Args:
        records: The evaluated records (the split's records, in the split's
            order).
        labels: The label space (the sorted unique labels of the dataset).
        buckets: The ``label -> bucket`` mapping (head / torso / tail /
            unseen).
        length_edges: The text-length bucket edges (in characters).

    Returns:
        The :class:`UprView` of the run.

    Raises:
        ValueError: If the records are empty, a label is missing from the
            bucket mapping or the gold matrix does not match the label space.
    """
    ordered = tuple(records)
    if not ordered:
        raise ValueError("the evaluated split is empty: nothing to analyze")
    label_space = tuple(labels)
    missing = [label for label in label_space if label not in buckets]
    if missing:
        raise ValueError(
            f"{len(missing)} label(s) missing from the bucket mapping, e.g. {missing[:3]!r}; "
            "every label needs a bucket"
        )
    gold = build_label_matrix(ordered, label_space)
    lengths = tuple(len(record.text) for record in ordered)
    grouped: dict[str, list[str]] = {}
    for label in label_space:
        grouped.setdefault(buckets[label], []).append(label)
    return UprView(
        records=ordered,
        labels=label_space,
        gold=gold,
        buckets=dict(buckets),
        labels_by_bucket={name: sorted(members) for name, members in grouped.items()},
        lengths=lengths,
        length_bucket_index=tuple(length_buckets(lengths, length_edges)),
        length_bucket_names=length_bucket_names(length_edges),
    )


def _decision_matrix(
    scores: np.ndarray,
    decision: Mapping[str, Any],
) -> np.ndarray:
    """The decision matrix of one operating point (``eval.decision``)."""
    return apply_decision_rule(
        scores,
        rule=decision["rule"],
        k=decision.get("k"),
        threshold=decision.get("threshold"),
        min_score=decision.get("min_score"),
    )


def _length_sections(
    view: UprView,
    scores: np.ndarray,
    predicted: np.ndarray,
    *,
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
) -> dict[str, Any]:
    """The report of every text-length bucket (the row-restricted view).

    The length bucket is a **row** restriction (a text-length band), so it is
    computed by restricting the rows and running the same report machinery.
    """
    sections: dict[str, Any] = {}
    for index, name in enumerate(view.length_bucket_names):
        rows = np.asarray(
            [
                position
                for position, bucket in enumerate(view.length_bucket_index)
                if bucket == index
            ],
            dtype=np.int64,
        )
        if not rows.size:
            continue
        section = build_label_report(
            scores[rows],
            view.gold[rows],
            view.labels,
            view.buckets,
            predicted[rows],
            ks=ks,
            shortlist_ms=shortlist_ms,
        )
        sections[name] = section["overall"]
    return sections


def _distributions(
    view: UprView,
    scores: np.ndarray,
) -> dict[str, Any]:
    """The score distributions per label bucket and per label.

    Every group is reported twice: the **gold** cells' scores and the
    **background** (non-gold) cells' scores, so the two sides are never mixed.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    truth = view.gold
    per_bucket: dict[str, Any] = {}
    for bucket, members in view.labels_by_bucket.items():
        columns = np.asarray([view.labels.index(label) for label in members], dtype=np.int64)
        if not columns.size:
            continue
        bucket_gold = truth[:, columns]
        per_bucket[bucket] = {
            "n_labels": int(columns.size),
            "gold": distribution(matrix[:, columns][bucket_gold]),
            "background": distribution(matrix[:, columns][~bucket_gold]),
        }
    per_label: dict[str, Any] = {}
    for position, label in enumerate(view.labels):
        column = matrix[:, position]
        is_gold = truth[:, position]
        per_label[label] = {
            "bucket": view.buckets[label],
            "gold": distribution(column[is_gold]),
            "background": distribution(column[~is_gold]),
        }
    return {"per_bucket": per_bucket, "per_label": per_label}


def arm_block(
    view: UprView,
    scores: np.ndarray,
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
    reference_scores: np.ndarray | None = None,
    chunks_per_text: Sequence[int] | None = None,
    arm: str = BASELINE_ARM,
    predictor: Mapping[str, Any] | None = None,
    chunker: str | None = None,
    aggregation: str | None = None,
) -> dict[str, Any]:
    """The report block of one arm (one scoring method).

    Args:
        view: The model-independent view of the split.
        scores: The arm's ``(n_texts, n_labels)`` score matrix.
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.
        reference_scores: The whole-text reference arm's scores (the shift's
            baseline). ``None`` for the reference arm itself.
        chunks_per_text: The number of chunks every text contributed (the
            chunked arms); ``None`` for the whole-text arm.
        arm: The arm's name.
        predictor: The arm's predictor description (the artifact's lineage).
        chunker: The arm's text-side chunker (the artifact's lineage).
        aggregation: The arm's folding rule (the artifact's lineage).

    Returns:
        The JSON-serializable block of the arm.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    truth = view.gold
    predicted = _decision_matrix(matrix, decision)
    report = build_label_report(
        matrix,
        truth,
        view.labels,
        view.buckets,
        predicted,
        ks=ks,
        shortlist_ms=shortlist_ms,
    )
    percentiles = cell_rank_percentiles(matrix)
    gold_percentiles = percentiles[truth]
    background_percentiles = percentiles[~truth]
    hits = gold_hit_at_k(matrix, truth, HIT_KS)
    shift = _shift_block(matrix, truth, reference_scores)
    block: dict[str, Any] = {
        "arm": arm,
        "predictor": dict(predictor) if predictor else None,
        "chunker": chunker,
        "aggregation": aggregation,
        "chunks_per_text": _chunks_block(chunks_per_text),
        "metrics": {
            "overall": report["overall"],
            "per_bucket": report["per_bucket"],
            "per_length_bucket": _length_sections(
                view, matrix, predicted, ks=ks, shortlist_ms=shortlist_ms
            ),
        },
        "gold": {
            "n_cells": view.n_gold_cells,
            "rank_p50": float(np.median(gold_percentiles)) if gold_percentiles.size else None,
            "top_fifth_share": top_fifth_share(gold_percentiles),
            **{f"hit@{k}": hits[f"hit@{k}"] for k in HIT_KS},
        },
        "background": {
            "n_cells": view.n_background_cells,
            "rank_p50": (
                float(np.median(background_percentiles)) if background_percentiles.size else None
            ),
            "top_fifth_share": top_fifth_share(background_percentiles),
            "mean": float(matrix[~truth].mean()) if view.n_background_cells else None,
            "p50": float(np.median(matrix[~truth])) if view.n_background_cells else None,
        },
        "shift": shift,
        "provenance": provenance(matrix, truth, PROVENANCE_KS),
        "distributions": _distributions(view, matrix),
    }
    return block


def _shift_block(
    scores: np.ndarray,
    gold: np.ndarray,
    reference_scores: np.ndarray | None,
) -> dict[str, Any]:
    """The background's level and rank shift against the reference arm."""
    from cluster_rag.text_processors.upr_diagnostics import level_and_rank_shift

    return level_and_rank_shift(scores, gold, reference_scores)


def _chunks_block(chunks_per_text: Sequence[int] | None) -> dict[str, Any] | None:
    """The chunk-count distribution of an arm (``None`` for the whole text)."""
    if chunks_per_text is None:
        return None
    values = np.asarray(list(chunks_per_text), dtype=np.float64)
    if not values.size:
        return {"n_texts": 0, "mean": None, "p50": None, "min": None, "max": None}
    return {
        "n_texts": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def build_upr_report(
    view: UprView,
    arms: Mapping[str, dict[str, Any]],
    *,
    run_id: str,
    split: str,
    source: str,
    source_sha256: str,
    predictor: Mapping[str, Any],
    embedding_run: Mapping[str, Any] | None,
    label_embedding_run: Mapping[str, Any] | None,
    decision: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
    length_edges: Sequence[int],
    guard: Mapping[str, Any] | None = None,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    """Assemble the run's artifact.

    The artifact's model-independent part (the gold matrix's shape, the
    background's completeness, the buckets) sits at the top; the arms sit
    under ``arms``, keyed by their name.

    Args:
        view: The model-independent view of the split.
        arms: The arms' blocks, keyed by the arm's name.
        run_id: The run's id.
        split: The evaluated split (``val`` or ``test``).
        source: The dataset path.
        source_sha256: The dataset's sha256 (the split's provenance).
        predictor: The reference predictor's description.
        embedding_run: The embedding artifact's lineage (``None``).
        label_embedding_run: The label-embedding artifact's lineage (``None``).
        decision: The decision rule in force.
        metrics_config: The metrics' configuration (the k grids).
        length_edges: The text-length bucket edges.
        guard: The reference-arm check against the ``evaluate`` run (the
            guard's outcome; ``None`` when it was not checked).
        total_seconds: The run's wall time.

    Returns:
        The JSON-serializable artifact.
    """
    return {
        "run_id": run_id,
        "command": "analyze-upr",
        "split": split,
        "source": source,
        "source_sha256": source_sha256,
        "dataset": {
            "n_texts": view.n_texts,
            "n_labels": view.n_labels,
            "n_gold_cells": view.n_gold_cells,
            "n_background_cells": view.n_background_cells,
            "background": {
                "kind": "the non-gold (text, label) cells",
                "sampling": None,
                "seed": None,
                "note": (
                    "UPR's background is fully enumerable: every (text, label) cell "
                    "that is not gold, with no sampling, no seed and no quota. The "
                    "OMCT track's control sample has no UPR counterpart because "
                    "there is nothing to sample."
                ),
            },
            "labels_by_bucket": view.labels_by_bucket,
            "text_lengths": {
                "edges": list(length_edges),
                "bucket_names": list(view.length_bucket_names),
                "n_texts_per_bucket": {
                    name: sum(1 for index in view.length_bucket_index if index == position)
                    for position, name in enumerate(view.length_bucket_names)
                },
                "mean_chars": float(np.mean(view.lengths)) if view.lengths else None,
                "min_chars": int(min(view.lengths)) if view.lengths else None,
                "max_chars": int(max(view.lengths)) if view.lengths else None,
            },
        },
        "predictor": dict(predictor),
        "embedding_run": dict(embedding_run) if embedding_run else None,
        "label_embedding_run": dict(label_embedding_run) if label_embedding_run else None,
        "decision": dict(decision),
        "metrics_config": dict(metrics_config),
        "guard": dict(guard) if guard else None,
        "arms": dict(arms),
        "total_seconds": total_seconds,
    }
