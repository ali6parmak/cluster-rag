"""The UPR side's diagnostics (the text-processing track's Stage 5 / CS-8a).

The OMCT track measured its benchmark against a **sampled control**: the
unjudged same-document cells, drawn with a seed. UPR needs no such sample —
the label space is small (68 labels) and every (text, label) cell is
enumerable, so the **non-gold cells are the background**, complete and
unambiguous. That is the whole point of this module: the background is not a
sample, it is the complement of the gold, and every number it produces is
exact (no seed, no quota, no sampling error).

The blocks it provides, for one arm's score matrix (texts x labels):

- :func:`cell_rank_percentiles` — every cell's rank percentile inside **its own
  text's** label list (0 = the best-scoring label, the track's convention).
- :func:`gold_hit_at_k` — the gold cells' hit@k ("is the gold label among the
  text's top-k labels?"), the UPR analogue of the OMCT probe's same-doc hit@k.
- :func:`top_fifth_share` — the share of a group in the top fifth of its
  text's list (the track's "top-fifth share").
- :func:`provenance` — the top-1 provenance analogue: per text, how many of
  its top labels are gold and how many are not.
- :func:`level_and_rank_shift` — the background's **level** (score) and
  **rank** shift against a reference arm, the control-side counterpart every
  headline number carries.
- :func:`length_buckets` — the texts' length buckets (the axis the chunking
  hypothesis is about on UPR: a pooled number hides the long-text tail).

Pure core: numpy and the standard library only — no model clients, no I/O, no
hidden state (AGENTS.md dependency-direction rule).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# The share of a text's label list that counts as "the top fifth" (the track's
# convention, the same constant the OMCT artifact uses).
TOP_FIFTH = 0.2

# The quantiles of the reported score distributions.
DISTRIBUTION_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


def as_float_matrix(scores: np.ndarray) -> np.ndarray:
    """Coerce a score matrix to 2-d float64 and reject non-finite values.

    Raises:
        ValueError: If the matrix is not 2-d or holds NaN/inf.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be a 2-d (n_texts, n_labels) matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must be finite (NaN/inf would corrupt every ranking)")
    return matrix


def as_bool_matrix(gold: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Coerce a gold matrix to 2-d bool matching ``shape``.

    Raises:
        ValueError: If the matrix is not 2-d or does not match ``shape``.
    """
    matrix = np.asarray(gold, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError(f"gold must be a 2-d matrix, got {matrix.shape}")
    if matrix.shape != shape:
        raise ValueError(f"gold must have shape {shape}, got {matrix.shape}")
    return matrix


def cell_rank_percentiles(scores: np.ndarray) -> np.ndarray:
    """Every cell's rank percentile inside its own text's label list.

    **Convention** (the track's, unchanged from OMCT): the percentile is the
    cell's *position from the top* of its text's label list, divided by the
    list's size. ``0.0`` is the best-scoring label, the bottom label of a list
    of ``n`` sits at ``(n - 1) / n``, and a random label sits at about ``0.5``.
    Ties keep the labels' own order (a stable sort), so the number is
    reproducible.

    Args:
        scores: The ``(n_texts, n_labels)`` score matrix.

    Returns:
        A float matrix of the same shape with the cells' percentiles.
    """
    matrix = as_float_matrix(scores)
    n_labels = matrix.shape[1]
    order = np.argsort(-matrix, axis=1, kind="stable")
    ranks = np.empty_like(order)
    positions = np.arange(n_labels)
    rows = np.arange(matrix.shape[0])[:, None]
    ranks[rows, order] = positions
    return ranks.astype(np.float64) / float(n_labels)


def top_fifth_share(percentiles: np.ndarray) -> float | None:
    """The share of a group inside the top fifth of its text's label list."""
    values = np.asarray(percentiles, dtype=np.float64).ravel()
    if not values.size:
        return None
    return float((values < TOP_FIFTH).mean())


def gold_hit_at_k(scores: np.ndarray, gold: np.ndarray, ks: Sequence[int]) -> dict[str, Any]:
    """The gold cells' hit@k: is the gold label among the text's top-k labels?

    The UPR analogue of the OMCT probe's same-document hit@k, and the
    per-cell view of the ranking quality (the decision view's ``P@k`` is the
    per-text view of the same question).

    Args:
        scores: The ``(n_texts, n_labels)`` score matrix.
        gold: The bool gold matrix of the same shape.
        ks: The k values (each clipped to the label count).

    Returns:
        A dict with the hit rate per k, the number of gold cells and the
        number of texts with at least one gold label.
    """
    matrix = as_float_matrix(scores)
    truth = as_bool_matrix(gold, matrix.shape)
    if any(k < 1 for k in ks):
        raise ValueError(f"k values must be positive, got {ks}")
    n_labels = matrix.shape[1]
    order = np.argsort(-matrix, axis=1, kind="stable")
    gold_rows, gold_cols = np.nonzero(truth)
    n_cells = int(gold_rows.size)
    result: dict[str, Any] = {
        "n_gold_cells": n_cells,
        "n_texts": int(matrix.shape[0]),
        "n_texts_with_gold": int(truth.any(axis=1).sum()),
    }
    if not n_cells:
        result.update({f"hit@{k}": 0.0 for k in ks})
        return result
    # The rank of every gold cell inside its own text's list.
    positions = np.empty_like(order)
    positions[np.arange(matrix.shape[0])[:, None], order] = np.arange(n_labels)
    gold_positions = positions[gold_rows, gold_cols]
    for k in ks:
        k_eff = min(k, n_labels)
        result[f"hit@{k}"] = float((gold_positions < k_eff).mean())
    return result


def provenance(
    scores: np.ndarray,
    gold: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """The top-1 provenance analogue: what a text's best labels are.

    For every text, the number of **gold** labels among its top-k labels and
    the number of **non-gold** ones. The top-1 entry is the UPR analogue of
    the OMCT track's "top-1 same-document candidate per paragraph" row: it says
    what *this arm's* first choice for a text is — a **model-dependent**
    measurement, never a property of the dataset.

    Args:
        scores: The ``(n_texts, n_labels)`` score matrix.
        gold: The bool gold matrix of the same shape.
        ks: The k values (each clipped to the label count).

    Returns:
        A dict with, per k, the mean number of gold and non-gold labels in a
        text's top-k, the share of texts whose top-1 label is gold, and the
        counts.
    """
    matrix = as_float_matrix(scores)
    truth = as_bool_matrix(gold, matrix.shape)
    n_labels = matrix.shape[1]
    order = np.argsort(-matrix, axis=1, kind="stable")
    result: dict[str, Any] = {"n_texts": int(matrix.shape[0]), "n_labels": int(n_labels)}
    for k in ks:
        k_eff = min(k, n_labels)
        top = order[:, :k_eff]
        top_gold = np.take_along_axis(truth, top, axis=1)
        result[f"mean_gold_in_top{k}"] = float(top_gold.sum(axis=1).mean())
        result[f"mean_nongold_in_top{k}"] = float((~top_gold).sum(axis=1).mean())
    # The top-1 provenance: how many texts' best label is a gold label.
    top1_gold = int(np.take_along_axis(truth, order[:, :1], axis=1).sum())
    result["top1_gold"] = top1_gold
    result["top1_not_gold"] = int(matrix.shape[0]) - top1_gold
    result["top1_gold_share"] = top1_gold / matrix.shape[0] if matrix.shape[0] else 0.0
    return result


def background_mask(gold: np.ndarray) -> np.ndarray:
    """The background cells: every non-gold cell (the complete complement).

    On UPR the background needs no sampling: the label space is small and
    every cell is enumerable, so "the non-gold cells" is a complete statement
    rather than a sampled one.
    """
    matrix = np.asarray(gold, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError(f"gold must be a 2-d matrix, got {matrix.shape}")
    return ~matrix


def level_and_rank_shift(
    arm_scores: np.ndarray,
    gold: np.ndarray,
    reference_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """The background's **level** and **rank** shift against a reference arm.

    The track's reporting rule carries over: no discrimination number is
    quoted without its background-side counterpart. A chunked arm changes the
    score *scale* (a max over more candidates is systematically higher), so
    the level shift says whether a change in the headline number is a ranking
    change or a re-scaling.

    Args:
        arm_scores: The arm's ``(n_texts, n_labels)`` score matrix.
        gold: The bool gold matrix of the same shape.
        reference_scores: The reference arm's score matrix (the whole-text
            arm). ``None`` means "no reference": the shifts are ``None`` and
            only the levels are reported.

    Returns:
        A dict with the background's and the gold's score level (mean, p50),
        the gold cells' rank percentile (p50) and the background cells' rank
        percentile (p50), plus the deltas against the reference arm.
    """
    arm = as_float_matrix(arm_scores)
    truth = as_bool_matrix(gold, arm.shape)
    background = ~truth
    percentiles = cell_rank_percentiles(arm)
    gold_percentiles = percentiles[truth]
    background_percentiles = percentiles[background]
    result: dict[str, Any] = {
        "n_background_cells": int(background.sum()),
        "n_gold_cells": int(truth.sum()),
        "background_mean": float(arm[background].mean()) if background.any() else None,
        "background_p50": float(np.median(arm[background])) if background.any() else None,
        "gold_mean": float(arm[truth].mean()) if truth.any() else None,
        "gold_p50": float(np.median(arm[truth])) if truth.any() else None,
        "gold_rank_p50": float(np.median(gold_percentiles)) if gold_percentiles.size else None,
        "background_rank_p50": (
            float(np.median(background_percentiles)) if background_percentiles.size else None
        ),
        "gold_top_fifth_share": top_fifth_share(gold_percentiles),
        "background_top_fifth_share": top_fifth_share(background_percentiles),
    }
    if reference_scores is None:
        result["background_mean_delta"] = None
        result["background_rank_p50_delta"] = None
        result["gold_rank_p50_delta"] = None
        return result
    reference = as_float_matrix(reference_scores)
    if reference.shape != arm.shape:
        raise ValueError(
            f"the reference score matrix must have the arm's shape {arm.shape}, "
            f"got {reference.shape}"
        )
    reference_percentiles = cell_rank_percentiles(reference)
    reference_background = reference[background]
    reference_gold_percentiles = reference_percentiles[truth]
    result["background_mean_delta"] = (
        float(arm[background].mean() - reference_background.mean()) if background.any() else None
    )
    result["background_rank_p50_delta"] = (
        float(np.median(background_percentiles) - np.median(reference_percentiles[background]))
        if background_percentiles.size
        else None
    )
    result["gold_rank_p50_delta"] = (
        float(np.median(gold_percentiles) - np.median(reference_gold_percentiles))
        if gold_percentiles.size
        else None
    )
    return result


def distribution(values: np.ndarray) -> dict[str, Any]:
    """The summary of a score group: the count, the mean and the quantiles."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if not array.size:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    quantiles = np.quantile(array, DISTRIBUTION_QUANTILES)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def length_bucket_names(edges: Sequence[int]) -> tuple[str, ...]:
    """The names of the text-length buckets the edges define.

    The names carry the edges (``"0-119"``, ``"120-299"``, ..., ``"600+"``), so
    a reader never has to guess what a bucket covers.
    """
    if not edges:
        return ("all",)
    ordered = sorted(int(edge) for edge in edges)
    names = [f"0-{ordered[0] - 1}"]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        names.append(f"{previous}-{current - 1}")
    names.append(f"{ordered[-1]}+")
    return tuple(names)


def length_buckets(lengths: Sequence[int], edges: Sequence[int]) -> list[int]:
    """The bucket index of every text (``-1`` when there are no edges).

    The buckets are the upper edges of the length axis in characters: a text
    of 0-119 characters is in bucket 0, 120-299 in bucket 1, and so on; a
    text above the last edge is in the last bucket.
    """
    if not edges:
        return [-1] * len(lengths)
    ordered = sorted(int(edge) for edge in edges)
    return [sum(1 for edge in ordered if int(length) >= edge) for length in lengths]


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """The Pearson correlation of two vectors (``None`` when it is undefined)."""
    if left.size < 2 or right.size < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)),
    )
    if denominator <= 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def _rank_auc(scores: np.ndarray, truth: np.ndarray) -> float | None:
    """The AUC of a score against a binary truth (the rank-based definition).

    Ties get their average rank, so a constant feature's AUC is exactly 0.5.
    """
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # the average ranks of the tied groups
    sorted_scores = scores[order]
    start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                average = (start + 1 + index) / 2.0
                ranks[order[start:index]] = average
            start = index
    positive_rank_sum = float(ranks[truth].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def feature_diagnostics(
    features: np.ndarray,
    gold: np.ndarray,
    lengths: Sequence[int],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Every feature's own ranking quality, its background side and its length.

    The CS-6c precedent, adapted to the UPR grid: for every feature the block
    says

    - the **ranking quality**: the feature's own AUC against the gold cells
      (the feature used as a score, on its own) on the evaluated split and on
      the fit side;
    - the **background-side counterpart**: the gold cells' mean against the
      background's mean, and the standardized separation between them (the
      level counterpart of the AUC, the "does the feature separate the two
      sides" question);
    - the **length correlation**: the correlation between the feature's value
      and the text's length (a length proxy is visible here — the plan's
      leakage rule (ii) says such a feature must not enter the model, so it
      is measured, not fed).

    Args:
        features: The feature cube of the evaluated split
            (``(n_texts, n_labels, n_features)``).
        gold: The bool gold matrix of the same shape.
        lengths: The texts' lengths (one per row).
        feature_names: The feature names, in column order.

    Returns:
        A dict with one block per feature and the count of the cells.
    """
    cube = np.asarray(features, dtype=np.float64)
    truth = np.asarray(gold, dtype=bool)
    if cube.ndim != 3:
        raise ValueError(f"features must be a 3-d cube, got shape {cube.shape}")
    if truth.shape != cube.shape[:2]:
        raise ValueError(f"gold must have shape {cube.shape[:2]}, got {truth.shape}")
    n_texts, n_labels, n_features = cube.shape
    if len(feature_names) != n_features:
        raise ValueError(f"the feature names must be {n_features} long, got {len(feature_names)}")
    text_lengths = np.asarray(list(lengths), dtype=np.float64)
    if text_lengths.size != n_texts:
        raise ValueError(
            f"the lengths must have one entry per text ({n_texts}), got {text_lengths.size}"
        )
    # the length axis broadcast over the labels (one length per row)
    length_axis = np.repeat(text_lengths[:, None], n_labels, axis=1).reshape(-1)
    flat = cube.reshape(-1, n_features)
    flat_truth = truth.reshape(-1)
    background = ~flat_truth
    blocks: dict[str, Any] = {}
    for index, name in enumerate(feature_names):
        column = flat[:, index]
        gold_values = column[flat_truth]
        background_values = column[background]
        gold_mean = float(gold_values.mean()) if gold_values.size else None
        background_mean = float(background_values.mean()) if background_values.size else None
        background_std = float(background_values.std()) if background_values.size else None
        separation = None
        if gold_mean is not None and background_mean is not None and background_std:
            separation = (gold_mean - background_mean) / background_std
        blocks[name] = {
            "auc": _rank_auc(column, flat_truth),
            "gold_mean": gold_mean,
            "background_mean": background_mean,
            "background_std": background_std,
            "separation": separation,
            "length_correlation": _pearson(column, length_axis),
            "n_gold_cells": int(flat_truth.sum()),
            "n_background_cells": int(background.sum()),
        }
    return {
        "n_cells": int(flat_truth.size),
        "n_texts": n_texts,
        "n_labels": n_labels,
        "features": blocks,
        "note": (
            "every feature's own ranking quality (the AUC of the feature used "
            "as a score), its background-side counterpart (the gold cells' mean "
            "against the background's, and the standardized separation) and its "
            "length correlation (a length proxy is visible here: the plan's "
            "leakage rule (ii) keeps such a feature out of the model)."
        ),
    }
