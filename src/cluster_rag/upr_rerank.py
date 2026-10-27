"""The UPR rerank probe's driver (``rerank-upr``).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-15) measures the **zero-shot
rerankers** on the UPR data as **scorers**: they re-order an existing
candidate set, and the run compares their ordering with **the ordering that
produced the set** (the decision model's own probabilities, the union's own
z-max, the champion's own scores).

**The like-for-like rule.** Every comparison holds the candidate set fixed
and changes only the ordering. The candidate set is the experiment's
independent variable (a reranker can re-order a set, never enlarge it), and
the run is configured with the set it re-orders (``upr_analysis.rerank``'s
``candidate_set``).

The driver orchestrates:

- the model-independent part (the UPR view: the texts, the label space and
  the gold matrix) — built by :mod:`cluster_rag.upr_analysis`;
- the **candidate sets** (the champion's, the decision model's and the
  union's own top-k) and their **ceilings**;
- the pure core (the label side, the label-blind counterpart, the rank
  fusion and the measurement) — in
  :mod:`cluster_rag.text_processors.rerank.upr_rerank`;
- the model client (the Hugging Face reranker behind the
  :class:`~cluster_rag.rerank.base.Reranker` port) — wired in by the CLI.

**The cost is part of the deliverable** (PLAN's Phase 6): every arm carries
its pairs per text, its pairs per second, its wall time and its peak GPU
memory, and the run measures the throughput directly (the cache is bypassed
for the probe), so the cost claim does not depend on the cache's state.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.config import RerankerModelConfig
from cluster_rag.rerank.base import (
    KIND_LISTWISE,
    Reranker,
    RerankRequest,
    listwise_requests,
)
from cluster_rag.text_processors.rerank.upr_rerank import (
    ORDER_LABEL_FIRST,
    ORDER_TEXT_FIRST,
    fuse_candidates,
    measure_reranked_arm,
)
from cluster_rag.upr_analysis import per_text_block

logger = logging.getLogger(__name__)

# The candidate sets the run may re-order, and the scorer each one comes from.
CANDIDATE_SETS: tuple[str, ...] = ("decision", "union", "champion")


@dataclass(frozen=True)
class RerankSources:
    """The three candidate sets' sources: the scorers they come from.

    The candidate set is the experiment's independent variable: the
    **decision model's** top-k (the primary), the **union's** top-k (the
    per-row z-max of the champion's and the decision model's scores) and the
    **champion's** top-k (the control, whose ceiling is the lowest).
    """

    champion: np.ndarray
    decision: np.ndarray
    union: np.ndarray

    def source(self, name: str) -> np.ndarray:
        """The score matrix of one candidate set's source.

        Raises:
            ValueError: If the set is unknown.
        """
        if name not in CANDIDATE_SETS:
            raise ValueError(
                f"unknown candidate set {name!r}; expected one of {list(CANDIDATE_SETS)}"
            )
        return getattr(self, name)


def candidate_set_matrix(sources: RerankSources, name: str) -> np.ndarray:
    """The score matrix a candidate set is read from (the set's own source)."""
    return sources.source(name)


def build_candidate_requests(
    texts: Sequence[str],
    documents: Sequence[Sequence[str]],
    *,
    kind: str,
    order: str,
) -> list[RerankRequest]:
    """The requests of one arm (the shape the model's family needs).

    - the **pair-based** families score one (query, document) pair per
      forward pass, so the order matters: ``text_first`` puts the text in the
      query slot and the label's document in the document slot (the
      pipeline's retrieval framing), ``label_first`` swaps them (the
      cross-encoders' training orientation);
    - the **listwise** family reads the whole candidate list in one context
      with a single query, so only ``text_first`` exists (the query is the
      text and the documents are the labels').

    Args:
        texts: The texts (one per row).
        documents: Every text's candidate documents, in the incumbent's order.
        kind: The model's family.
        order: The pair order variant.

    Returns:
        The requests.

    Raises:
        ValueError: If the order is unknown, or if the listwise family is
            asked for the reverse order (its API has no room for it).
    """
    if kind == KIND_LISTWISE:
        if order != ORDER_TEXT_FIRST:
            raise ValueError(
                "the listwise family takes one query and many documents: the "
                f"{order!r} order does not exist for it"
            )
        return listwise_requests(texts, documents)
    if order not in (ORDER_TEXT_FIRST, ORDER_LABEL_FIRST):
        raise ValueError(f"unknown pair order {order!r}")
    requests: list[RerankRequest] = []
    for text, row_documents in zip(texts, documents, strict=True):
        for document in row_documents:
            if order == ORDER_TEXT_FIRST:
                requests.append(RerankRequest(query=text, documents=(document,)))
            else:
                # The reverse order: the label's document is the query and the
                # text is the document.
                requests.append(RerankRequest(query=document, documents=(text,)))
    return requests


def run_upr_rerank_arm(
    *,
    texts: Sequence[str],
    gold: np.ndarray,
    columns: Sequence[Sequence[int]],
    incumbent: np.ndarray,
    reranker: Reranker,
    model_config: RerankerModelConfig,
    documents: Sequence[str],
    label_document: str,
    order: str,
    label_blind: bool,
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
    bucket_masks: Mapping[str, np.ndarray] | None = None,
    benchmark_requests: int = 0,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one arm: re-order the candidate set and measure the reordering.

    Args:
        texts: The texts (one per row).
        gold: The bool gold matrix (the same shape as the incumbent's).
        columns: Every text's candidate columns (the incumbent's order).
        incumbent: The incumbent's score matrix (the ordering that produced
            the candidate set).
        reranker: The reranker (the port; the CLI wires the client in).
        model_config: The model's config (its family, its name).
        documents: The label side's document per label (the label space's
            order).
        label_document: The label side's variant (``"description"`` or
            ``"name"``) — part of the arm's name.
        order: The pair order variant.
        label_blind: Whether this is the **label-blind counterpart**.
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.
        bucket_masks: The label buckets' column masks (the per-bucket view).
        benchmark_requests: How many requests the throughput probe may use
            (``0`` = no probe).

    Returns:
        The arm's block (the measurement, the cost and the paired comparison)
        and the fused score matrix.
    """
    n_texts = len(texts)
    before_usage = dict(getattr(reranker, "usage", {}))
    before_cache = {
        "hits": int(getattr(reranker, "cache_hits", 0)),
        "misses": int(getattr(reranker, "cache_misses", 0)),
    }
    per_text_documents: list[list[str]] = []
    for row_columns in columns:
        per_text_documents.append([documents[column] for column in row_columns])

    requests = build_candidate_requests(
        texts, per_text_documents, kind=model_config.kind, order=order
    )
    scores = reranker.score(requests)
    per_candidate = _scores_per_candidate(scores, columns, n_texts, model_config.kind)

    fused = fuse_candidates(incumbent, columns, per_candidate)
    measurement = measure_reranked_arm(
        fused,
        gold,
        decision=decision,
        ks=ks,
        shortlist_ms=shortlist_ms,
        bucket_masks=bucket_masks,
    )
    cost = _cost_block(
        reranker,
        per_candidate,
        n_texts,
        benchmark_requests,
        requests,
        before_usage,
        before_cache,
    )
    block: dict[str, Any] = {
        "arm": _arm_name(model_config, label_document, order, label_blind),
        "model": model_config.name,
        "kind": model_config.kind,
        "order": order,
        "label_blind": label_blind,
        "n_candidates_per_text": len(columns[0]) if columns else 0,
        "measurement": measurement,
        "cost": cost,
    }
    return block, fused


def _scores_per_candidate(
    scores: Sequence[Sequence[float]],
    columns: Sequence[Sequence[int]],
    n_texts: int,
    kind: str,
) -> list[list[float]]:
    """The reranker's scores, aligned with every text's candidate list.

    The pair-based families answer one request per (text, candidate), so their
    scores come back flattened; the listwise family answers one request per
    text, so its scores are already aligned.

    Raises:
        ValueError: If the scores do not match the candidate sets.
    """
    if kind == KIND_LISTWISE:
        if len(scores) != n_texts:
            raise ValueError(
                f"the listwise family answered {len(scores)} requests for {n_texts} texts"
            )
        return [list(row) for row in scores]
    flat = [value for row in scores for value in row]
    expected = sum(len(row) for row in columns)
    if len(flat) != expected:
        raise ValueError(f"the reranker answered {len(flat)} pair scores for {expected} candidates")
    out: list[list[float]] = []
    cursor = 0
    for row_columns in columns:
        size = len(row_columns)
        out.append(flat[cursor : cursor + size])
        cursor += size
    return out


def _cost_block(
    reranker: Reranker,
    per_candidate: Sequence[Sequence[float]],
    n_texts: int,
    benchmark_requests: int,
    requests: Sequence[RerankRequest],
    before: Mapping[str, float],
    before_cache: Mapping[str, int],
) -> dict[str, Any]:
    """The arm's cost block: the pairs, the throughput, the wall time, the memory.

    The client's counters are **cumulative** over its lifetime, so the arm's
    own cost is the *difference* between the counters before and after it.
    The **throughput probe** bypasses the cache, so the cost claim does not
    depend on the cache's state (a warm run's wall time is not a cost
    measurement).
    """
    usage = dict(getattr(reranker, "usage", {}))
    n_pairs = sum(len(row) for row in per_candidate)
    seconds = max(0.0, float(usage.get("seconds", 0.0)) - float(before.get("seconds", 0.0)))
    n_calls = max(0.0, float(usage.get("n_calls", 0.0)) - float(before.get("n_calls", 0.0)))
    n_truncated = max(
        0,
        int(usage.get("n_truncated_inputs", 0)) - int(before.get("n_truncated_inputs", 0)),
    )
    block: dict[str, Any] = {
        "n_pairs": n_pairs,
        "n_pairs_per_text": (n_pairs / n_texts) if n_texts else 0.0,
        "n_calls": int(n_calls),
        "seconds": seconds,
        "peak_gpu_mb": float(usage.get("peak_gpu_mb", 0.0)),
        "n_truncated_inputs": n_truncated,
        "cache": {
            "hits": int(getattr(reranker, "cache_hits", 0)) - int(before_cache.get("hits", 0)),
            "misses": int(getattr(reranker, "cache_misses", 0))
            - int(before_cache.get("misses", 0)),
        },
        "note": (
            "the cost of the arm: the pairs it asked for, the forward passes "
            "that actually ran (the cache's misses), their wall time and the "
            "peak GPU memory. The throughput probe below bypasses the cache, so "
            "the cost claim does not depend on the cache's state."
        ),
    }
    benchmark = getattr(reranker, "benchmark", None)
    if benchmark is not None and benchmark_requests > 0 and requests:
        # A strided sample, so the probe spans the texts rather than one
        # text's candidates (the padding, and so the cost, is per text).
        stride = max(1, len(requests) // benchmark_requests)
        sample = list(requests)[::stride][:benchmark_requests]
        block["throughput_probe"] = benchmark(sample)
        rate = float(block["throughput_probe"].get("pairs_per_second") or 0.0)
        block["projected_seconds_for_660_texts"] = (
            (block["n_pairs_per_text"] * 660.0 / rate) if rate > 0 else None
        )
    return block


def _arm_name(
    model_config: RerankerModelConfig,
    label_document: str,
    order: str,
    label_blind: bool,
) -> str:
    """The arm's name: the model, the label side, the order and the blind flag.

    The name is the artifact's key, so it must carry every variant that
    changes the measurement (the label side's variant included).
    """
    parts = [model_config.name, label_document, order]
    if label_blind:
        parts.append("blind")
    return "-".join(parts)


def paired_arm_comparison(
    fused: np.ndarray,
    incumbent: np.ndarray,
    gold: np.ndarray,
    labels: Sequence[str],
    labels_by_bucket: Mapping[str, Sequence[str]],
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int],
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """The paired comparison of a re-ordered arm against the incumbent.

    The resampling unit is the **text** (the track's discipline): both
    orderings are judged on the same resampled texts, so the interval is
    paired and no verdict rests on two independent numbers.

    Args:
        fused: The re-ordered arm's score matrix.
        incumbent: The incumbent's score matrix (the same shape).
        gold: The bool gold matrix (the same shape).
        labels: The label space (the matrix's column order).
        labels_by_bucket: The label buckets' members.
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        resamples: The number of bootstrap resamples.
        seed: The seed of the generator.

    Returns:
        The block: the micro-F1 and the nDCG comparisons, pooled and per
        bucket.
    """
    from cluster_rag.eval.multilabel import (
        apply_decision_rule,
        paired_counts_bootstrap,
        paired_values_bootstrap,
    )

    truth = np.asarray(gold, dtype=bool)
    predicted = apply_decision_rule(
        fused,
        rule=decision["rule"],
        k=decision.get("k"),
        threshold=decision.get("threshold"),
        min_score=decision.get("min_score"),
    )
    incumbent_predicted = apply_decision_rule(
        incumbent,
        rule=decision["rule"],
        k=decision.get("k"),
        threshold=decision.get("threshold"),
        min_score=decision.get("min_score"),
    )
    arm_block = per_text_block(fused, truth, predicted, labels, labels_by_bucket, ks=ks)
    incumbent_block = per_text_block(
        incumbent, truth, incumbent_predicted, labels, labels_by_bucket, ks=ks
    )

    def _counts(block: Mapping[str, Any], bucket: str | None) -> np.ndarray:
        rows = block["f1_counts"] if bucket is None else block["by_bucket"][bucket]["f1_counts"]
        return np.asarray(rows, dtype=np.float64)

    def _values(block: Mapping[str, Any], bucket: str | None) -> np.ndarray:
        payload = block["ndcg"] if bucket is None else block["by_bucket"][bucket]["ndcg"]
        return np.asarray(
            [np.nan if value is None else float(value) for value in payload["values"]],
            dtype=np.float64,
        )

    result: dict[str, Any] = {"pooled": {}, "per_bucket": {}}
    for bucket in [None, *labels_by_bucket]:
        f1 = paired_counts_bootstrap(
            _counts(arm_block, bucket),
            _counts(incumbent_block, bucket),
            resamples=resamples,
            seed=seed,
        )
        ndcg = paired_values_bootstrap(
            _values(arm_block, bucket),
            _values(incumbent_block, bucket),
            resamples=resamples,
            seed=seed,
        )
        payload = {"micro_f1": f1, "ndcg": ndcg}
        if bucket is None:
            result["pooled"] = payload
        else:
            result["per_bucket"][bucket] = payload
    return result


def build_upr_rerank_report(
    *,
    run_id: str,
    split: str,
    source: str,
    source_sha256: str,
    n_texts: int,
    n_labels: int,
    candidate_set: str,
    candidate_k: int,
    arms: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    ceiling: Mapping[str, Any],
    config_block: Mapping[str, Any],
    guard: Mapping[str, Any] | None = None,
    reference: Mapping[str, Any] | None = None,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    """Assemble the run's artifact.

    Returns:
        The JSON-serializable artifact.
    """
    return {
        "run_id": run_id,
        "command": "rerank-upr",
        "split": split,
        "source": source,
        "source_sha256": source_sha256,
        "sample": {
            "n_texts": n_texts,
            "n_labels": n_labels,
            "n_cells": n_texts * n_labels,
        },
        "candidate_set": {
            "name": candidate_set,
            "k": candidate_k,
            **dict(ceiling),
        },
        "incumbent": dict(incumbent),
        "config": dict(config_block),
        "guard": dict(guard) if guard else None,
        "reference_arm": dict(reference) if reference else None,
        "arms": dict(arms),
        "total_seconds": total_seconds,
    }
