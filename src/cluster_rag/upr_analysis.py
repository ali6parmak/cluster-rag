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
from cluster_rag.eval.multilabel import (
    apply_decision_rule,
    per_row_f1_counts,
    per_row_ndcg,
)
from cluster_rag.eval.report import build_label_report
from cluster_rag.text_processors.aggregation import (
    Aggregation,
    aggregate_chunk_similarities,
    parse_aggregation,
)
from cluster_rag.text_processors.base import Chunker, SpanChunker
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
    *,
    per_label: bool = True,
) -> dict[str, Any]:
    """The score distributions per label bucket and per label.

    Every group is reported twice: the **gold** cells' scores and the
    **background** (non-gold) cells' scores, so the two sides are never mixed.
    The per-label breakdown is optional: a run with many arms would carry
    68 labels x 2 distributions per arm, and only the reference arm's
    per-label view is read.
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
    per_label_block: dict[str, Any] = {}
    if per_label:
        for position, label in enumerate(view.labels):
            column = matrix[:, position]
            is_gold = truth[:, position]
            per_label_block[label] = {
                "bucket": view.buckets[label],
                "gold": distribution(column[is_gold]),
                "background": distribution(column[~is_gold]),
            }
    return {"per_bucket": per_bucket, "per_label": per_label_block}


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
    per_label_distributions: bool = True,
    paired: Mapping[str, Any] | None = None,
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
        "chunks_per_text": chunks_block(chunks_per_text),
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
        "paired": dict(paired) if paired else None,
        "per_text": per_text_block(
            matrix,
            truth,
            predicted,
            view.labels,
            view.labels_by_bucket,
            ks=ks,
        ),
        "provenance": provenance(matrix, truth, PROVENANCE_KS),
        "distributions": _distributions(view, matrix, per_label=per_label_distributions),
    }
    return block


def per_text_block(
    matrix: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
    labels_by_bucket: Mapping[str, Sequence[str]],
    *,
    ks: Sequence[int],
) -> dict[str, Any]:
    """The arm's **per-text** summary: the compact form of its two views.

    The block is what makes a **cross-run** comparison possible: the two
    scorers of such a comparison live in two different runs (their score
    matrices are not in memory together), but the per-text counts (the
    decision view) and the per-text nDCG (the ranking view) are enough to
    bootstrap a **paired** interval over the texts.

    The counts are the per-text (true positives, false positives, false
    negatives) — overall and per **label bucket** (the unseen bucket is where
    the catalog's channels are supposed to help).
    """
    counts = per_row_f1_counts(predicted, truth)
    ndcg_k = 5 if 5 in ks else (ks[-1] if ks else 5)
    values = per_row_ndcg(matrix, truth, ndcg_k)
    block: dict[str, Any] = {
        "f1_counts": [[int(tp), int(fp), int(fn)] for tp, fp, fn in zip(*counts, strict=True)],
        "ndcg": {
            "k": int(ndcg_k),
            "values": [None if np.isnan(value) else float(value) for value in values],
        },
        "by_bucket": {},
        "note": (
            "the per-text summary of the two views: the (tp, fp, fn) counts of the "
            "decision view and the per-text nDCG. It is what a cross-run paired "
            "comparison needs (the two score matrices are not in memory together)."
        ),
    }
    index = {label: position for position, label in enumerate(labels)}
    for bucket, members in labels_by_bucket.items():
        columns = np.asarray([index[label] for label in members], dtype=np.int64)
        if not columns.size:
            continue
        bucket_counts = per_row_f1_counts(predicted[:, columns], truth[:, columns])
        bucket_values = per_row_ndcg(matrix[:, columns], truth[:, columns], ndcg_k)
        block["by_bucket"][bucket] = {
            "f1_counts": [
                [int(tp), int(fp), int(fn)] for tp, fp, fn in zip(*bucket_counts, strict=True)
            ],
            "ndcg": {
                "k": int(ndcg_k),
                "values": [None if np.isnan(value) else float(value) for value in bucket_values],
            },
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


def chunks_block(chunks_per_text: Sequence[int] | None) -> dict[str, Any] | None:
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


# --- the chunked arms (CS-8b) ---


@dataclass(frozen=True)
class ChunkedSide:
    """One side's chunked encoding: the chunk lists and the chunk vectors.

    ``vectors`` holds one row per chunk (a text's chunks are consecutive rows,
    in text order), so :meth:`group_rows` gives every text's rows.
    """

    chunk_lists: tuple[tuple[str, ...], ...]
    vectors: np.ndarray

    @property
    def n_texts(self) -> int:
        """The number of texts on this side."""
        return len(self.chunk_lists)

    @property
    def n_chunks(self) -> int:
        """The number of chunks (the vectors' rows)."""
        return int(self.vectors.shape[0])

    def group_rows(self) -> tuple[tuple[int, ...], ...]:
        """The row indices of every text's chunks (in text order)."""
        groups: list[tuple[int, ...]] = []
        cursor = 0
        for chunk_list in self.chunk_lists:
            groups.append(tuple(range(cursor, cursor + len(chunk_list))))
            cursor += len(chunk_list)
        return tuple(groups)

    def chunks_per_text(self) -> tuple[int, ...]:
        """The number of chunks every text contributed."""
        return tuple(len(chunk_list) for chunk_list in self.chunk_lists)

    def flat_chunks(self) -> list[str]:
        """Every chunk's text, in row order."""
        return [chunk for chunk_list in self.chunk_lists for chunk in chunk_list]


def encode_chunks(embedder: Any, chunker: Chunker, texts: Sequence[str], kind: str) -> ChunkedSide:
    """Chunk the texts and embed every chunk (one vector row per chunk).

    Args:
        embedder: The embedder (a port: the CLI wires the cached backend in).
        chunker: The chunker.
        texts: The texts to chunk.
        kind: The encode kind (``"query"`` for the text side, ``"label"`` for
            the label side).

    Returns:
        The encoded :class:`ChunkedSide`.

    Raises:
        ValueError: If the chunker produced no chunk for a text.
    """
    chunk_lists = chunker.segment(list(texts))
    chunks: list[str] = []
    for text, chunk_list in zip(texts, chunk_lists, strict=True):
        if not chunk_list:
            raise ValueError(f"the chunker produced no chunk for a text: {text[:60]!r}")
        chunks.extend(chunk_list)
    vectors = np.asarray(embedder.encode(chunks, kind))
    return ChunkedSide(chunk_lists=tuple(tuple(item) for item in chunk_lists), vectors=vectors)


def fold_label_chunks(matrix: np.ndarray, label_groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Fold a chunked **label** side: every label keeps its best chunk.

    The symmetric family (the OMCT variants exp05/exp10 chunk both sides): the
    label side is folded with the maximum — a label's evidence is its best
    matching chunk — and the text side is then folded with the arm's own rule.

    Args:
        matrix: The ``(n_text_chunks, n_label_chunks)`` similarity matrix.
        label_groups: The chunk rows of every label (in label order).

    Returns:
        The ``(n_text_chunks, n_labels)`` matrix of folded scores.

    Raises:
        ValueError: If a label has no chunk.
    """
    starts: list[int] = []
    for position, group in enumerate(label_groups):
        indices = list(group)
        if not indices:
            raise ValueError(f"label {position} has no chunk")
        starts.append(indices[0])
    return np.maximum.reduceat(matrix, np.asarray(starts, dtype=np.int64), axis=1)


def chunked_arm_scores(
    predictor: Any,
    side: ChunkedSide,
    rule: Aggregation,
    *,
    global_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Score a chunked side with the pipeline's predictor and fold it per text.

    The chunks are scored with the **same predictor** the whole-text arm uses
    (so the geometry is the pipeline's own), and the chunk-level scores are
    then folded into one score per (text, label) pair with the arm's rule.

    Args:
        predictor: The predictor (the port).
        side: The chunked text side.
        rule: The folding rule.
        global_scores: The whole-text score matrix (the second input of the
            local+global blend, ``exp06``).

    Returns:
        The ``(n_texts, n_labels)`` score matrix of the arm.
    """
    chunk_scores = predictor.score(side.flat_chunks(), query_vectors=side.vectors)
    return aggregate_chunk_similarities(
        np.asarray(chunk_scores, dtype=np.float64),
        side.group_rows(),
        rule,
        global_similarities=global_scores,
    )


def symmetric_arm_scores(
    text_side: ChunkedSide,
    label_side: ChunkedSide,
    rule: Aggregation,
    *,
    global_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Score the text chunks against the **label chunks** (the symmetric family).

    The label side is folded with the maximum first (a label keeps its best
    chunk), then the text side is folded with the arm's own rule — the same
    order as the OMCT symmetric variants.

    Args:
        text_side: The chunked text side.
        label_side: The chunked label side.
        rule: The text side's folding rule.
        global_scores: The whole-text score matrix (the local+global blend).

    Returns:
        The ``(n_texts, n_labels)`` score matrix of the arm.
    """
    matrix = (
        np.asarray(text_side.vectors, dtype=np.float64)
        @ np.asarray(label_side.vectors, dtype=np.float64).T
    )
    folded = fold_label_chunks(matrix, label_side.group_rows())
    return aggregate_chunk_similarities(
        folded,
        text_side.group_rows(),
        rule,
        global_similarities=global_scores,
    )


def arm_name(family: str, aggregation: Aggregation) -> str:
    """The arm's name: the family (which names its chunker) and the rule."""
    return f"{family}_{aggregation.slug}"


def encode_late_chunks(
    embedder: Any,
    chunker: Chunker,
    texts: Sequence[str],
    kind: str,
    *,
    pooling: str = "mean",
    reference_vectors: np.ndarray | None = None,
) -> tuple[ChunkedSide, list[dict[str, Any]]]:
    """Encode a side with the **late chunking**: the text once, the tokens pooled.

    The text is embedded **once** at the token level and every chunk's vector
    is pooled out of the tokens that fall in its character span, so every
    chunk carries the text's context.

    The two capabilities this path needs are **probed, never assumed**: the
    embedder must be able to return token-level embeddings (the HF backend
    can; Ollama's ``/api/embed`` cannot) and the chunker must be able to say
    where its chunks are.

    Args:
        embedder: The embedder (a port: the CLI wires the cached backend in).
        chunker: The chunker.
        texts: The texts to chunk.
        kind: The encode kind (``"query"`` for the text side).
        pooling: The pooling rule (``"mean"`` = the model's own formula,
            ``"norm"`` = the per-token-normalized variant).
        reference_vectors: The model's **own** text embeddings (the pipeline's
            query vectors), when they are known. The consistency check then
            reports the cosine between the text's pooled token vector and the
            model's own embedding of the same text — the check that the
            token-level path lives in the model's own space.

    Returns:
        The encoded side and the per-text pooling reports.

    Raises:
        ValueError: If the embedder has no token-level capability, the
            chunker has no character spans, or a chunk has no token.
    """
    from cluster_rag.embed.base import token_capability
    from cluster_rag.text_processors.late_chunking import (
        pool_chunk_vectors,
        pool_document_vector,
        pooling_report,
    )

    capability = token_capability(embedder)
    if capability is None:
        raise ValueError(
            "the late-chunked arm needs token-level embeddings; this backend "
            "cannot return them (Ollama's /api/embed returns one vector per "
            "text and nothing else), so the arm is HF-only"
        )
    if not isinstance(chunker, SpanChunker):
        raise ValueError(
            f"the late-chunked arm needs the chunker's character spans; the chunker "
            f"{getattr(chunker, 'name', chunker)!r} does not provide them"
        )
    text_list = list(texts)
    token_embeddings = capability.encode_tokens(text_list, kind)  # type: ignore[arg-type]
    span_lists = chunker.segment_spans(text_list)
    chunk_lists: list[tuple[str, ...]] = []
    pooled_rows: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    consistencies: list[float] = []
    for position, (text, tokens, spans) in enumerate(
        zip(text_list, token_embeddings, span_lists, strict=True)
    ):
        chunk_lists.append(tuple(text[start:end] for start, end in spans))
        pooled_rows.append(pool_chunk_vectors(tokens, spans, pooling=pooling))
        report = pooling_report(tokens, spans)
        if reference_vectors is not None and position < len(reference_vectors):
            own = np.asarray(reference_vectors[position], dtype=np.float64)
            norm = float(np.linalg.norm(own))
            if norm > 0:
                pooled = pool_document_vector(tokens)
                report["cosine_with_the_model_embedding"] = float(
                    np.dot(pooled.astype(np.float64), own / norm)
                )
                consistencies.append(report["cosine_with_the_model_embedding"])
        reports.append(report)
    vectors = (
        np.concatenate(pooled_rows, axis=0) if pooled_rows else np.zeros((0, 0), dtype=np.float32)
    )
    side = ChunkedSide(chunk_lists=tuple(chunk_lists), vectors=vectors)
    if consistencies:
        mean_cosine = float(np.mean(consistencies))
        for report in reports:
            report["pooling_consistency_mean"] = mean_cosine
    return side, reports


def fusion_components(
    context: Any,
    chunk_vectors: np.ndarray,
    *,
    k: int,
    power: float = 1.0,
) -> dict[str, np.ndarray]:
    """The fusion's three components, per chunk (the weight re-fit's inputs).

    The components do not depend on the fusion's weights, so they are
    computed **once** per chunker and the weight grid is explored by
    re-blending them (the expensive part — the kNN votes against the train
    index — is paid once).

    The components are the pipeline's own: :func:`knn_vote_scores` (the
    similarity-weighted votes of the top-k train neighbours),
    :func:`prototype_scores` (the label prototypes' cosines) and the
    zero-shot term (the query-label cosines).

    Args:
        context: The predictor context (the train vectors and the label
            vectors).
        chunk_vectors: The chunk vectors (the query side).
        k: The kNN component's neighbour count.
        power: The vote-weight exponent.

    Returns:
        The three component matrices (each ``(n_chunks, n_labels)``).
    """
    from cluster_rag.data import build_label_matrix
    from cluster_rag.retrieve.knn import knn_vote_scores
    from cluster_rag.retrieve.prototypes import label_prototypes, prototype_scores

    train_vectors = context.train_document_vectors
    label_vectors = context.label_vectors
    if train_vectors is None:
        raise ValueError(
            "the fusion components need the train document vectors (the retrieval geometry's index)"
        )
    train_matrix = build_label_matrix(context.train_records, context.labels)
    prototypes = label_prototypes(train_vectors, train_matrix)
    knn = knn_vote_scores(train_vectors, train_matrix, chunk_vectors, k=k, power=power)
    proto = prototype_scores(prototypes, chunk_vectors)
    zeroshot = (
        np.asarray(chunk_vectors, dtype=np.float64) @ np.asarray(label_vectors, dtype=np.float64).T
        if label_vectors is not None
        else np.zeros((chunk_vectors.shape[0], len(context.labels)), dtype=np.float64)
    )
    return {
        "knn": np.asarray(knn, dtype=np.float64),
        "proto": np.asarray(proto, dtype=np.float64),
        "zeroshot": np.asarray(zeroshot, dtype=np.float64),
    }


def aggregation_of(spec: str) -> Aggregation:
    """Parse a folding rule (the config-facing spelling)."""
    return parse_aggregation(spec)


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
    families: Mapping[str, Any] | None = None,
    learned: Mapping[str, Any] | None = None,
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
        learned: The learned arm's block (CS-8d: the pack, the combiner, the
            training's shape and the three floors); ``None`` when the run has
            no learned arm.
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
        "families": dict(families) if families else None,
        "learned": dict(learned) if learned else None,
        "arms": dict(arms),
        "total_seconds": total_seconds,
    }
