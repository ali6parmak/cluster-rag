"""The leaf layer: the module's vocabulary and the small pure helpers.

The vocabulary is the grids and the names the arms are compared by (the
quantile grid, the hit@k grid, the baseline and decision arm names, the
safe-negative rule's variants, the semantic criteria). The helpers are
the leaf functions the other modules build on: the statistics (the AUC
decompositions, the rank-percentile shares), the naming (the chunker's
handles, the grid cell's id).

Nothing here knows about the dataset view or about an arm.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.eval.metrics import grouped_auc_decomposition, roc_auc
from cluster_rag.text_processors.base import Chunker

logger = logging.getLogger(__name__)


def _slug(text: str) -> str:
    """A readable, filesystem-safe slug of a name (empty collapses to "-")."""
    slug = "".join(character if character in _SLUG_SAFE else "_" for character in text)
    return slug[:60] or "-"


def document_code(document: str, pair_ids: Sequence[str]) -> str:
    """The short code of a document, used in the artifact's cell ids.

    The pairwise file's pair ids follow the convention
    ``<code>-<paragraph number>-Q<question number>``, so a document's code is
    recovered from its ids when they agree on one. Otherwise a readable slug
    of the document name is used. The mapping is recorded in the artifact, so
    every cell id can be resolved back to its document, paragraph and
    question.
    """
    codes: set[str] = set()
    for pair_id in pair_ids:
        match = _CELL_ID_PATTERN.match(pair_id)
        if match is None:
            return _slug(document)
        codes.add(match.group("code"))
    if len(codes) == 1:
        return codes.pop()
    return _slug(document)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    """Count + mean/min/max + quantiles of one score group (empty groups are safe)."""
    block: dict[str, Any] = {"n": int(values.size)}
    if values.size:
        block.update(
            {
                "mean": float(np.mean(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "quantiles": {
                    f"{p:g}": float(np.quantile(values, p)) for p in DISTRIBUTION_QUANTILES
                },
            }
        )
    return block


def _quantile(values: np.ndarray, probability: float) -> float | None:
    """One quantile of a group (``None`` for an empty group)."""
    if not values.size:
        return None
    return float(np.quantile(values, probability))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """The average ranks of the values (1-based; ties share the mean rank)."""
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or ordered[index] != ordered[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0 + 1.0
            start = index
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """The Pearson correlation of two vectors (``None`` when undefined)."""
    if left.size < 2 or left.size != right.size:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.sqrt(float((left_centered**2).sum()) * float((right_centered**2).sum())))
    if denominator <= 0.0:
        return None
    return float(float((left_centered * right_centered).sum()) / denominator)


def length_correlation(
    lengths: np.ndarray, scores: np.ndarray, groups: np.ndarray | None = None
) -> dict[str, Any]:
    """The correlation between a paragraph's length and a cell's score.

    The Stage-0 caveat asks for it: a chunked method takes a maximum over
    more candidates, so its scale shift is partly a **length** effect and
    partly a **document** effect — and a single global shift is only an
    approximation. The correlation says how much of a cell's score travels
    with its paragraph's length (the Pearson on the raw values and the
    Spearman on the ranks), overall and per group (the document).

    Args:
        lengths: The paragraph lengths (in characters) of the cells.
        scores: The cells' scores.
        groups: The cells' groups (the document); ``None`` skips the
            per-group breakdown.

    Returns:
        A JSON-serializable dict with the correlations and the counts.
    """
    report: dict[str, Any] = {
        "overall": {
            "n": int(lengths.size),
            "pearson": _pearson(lengths, scores),
            "spearman": _pearson(_average_ranks(lengths), _average_ranks(scores)),
        },
        "note": (
            "the correlation between a cell's paragraph length and the cell's "
            "score: it is how much of a method's scale shift travels with the "
            "paragraph's length (the Stage-0 caveat's length effect)"
        ),
    }
    if groups is not None:
        by_group: dict[str, Any] = {}
        for group in sorted(set(groups.tolist())):
            mask = groups == group
            by_group[group] = {
                "n": int(mask.sum()),
                "pearson": _pearson(lengths[mask], scores[mask]),
                "spearman": _pearson(_average_ranks(lengths[mask]), _average_ranks(scores[mask])),
            }
        report["by_group"] = by_group
    return report


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """The AUC of a group (None when a class is missing)."""
    try:
        return roc_auc(labels.tolist(), scores.tolist())
    except ValueError:
        return None


def _mean_of(values: Sequence[float | None]) -> float | None:
    """The mean of the usable values (None when there is none)."""
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _difference(left: float | None, right: float | None) -> float | None:
    """The difference of two optional numbers (None when either is missing)."""
    if left is None or right is None:
        return None
    return float(left - right)


def _weighted_mean(
    values: Sequence[tuple[str, float | None]], weights: Mapping[str, int]
) -> float | None:
    """The weighted mean of per-group values (None when no group has a value).

    The weight of a group is its number of positive-negative pairs, so the
    result is the same statistic as the within-group AUC (which averages the
    groups' AUCs with that weight).
    """
    weighted = 0.0
    total = 0
    for group, value in values:
        if value is None:
            continue
        weight = weights.get(group, 0)
        weighted += value * weight
        total += weight
    return float(weighted / total) if total else None


def _auc_decomposition(
    labels: np.ndarray, scores: np.ndarray, documents: np.ndarray
) -> dict[str, Any]:
    """The within-document / cross-document split of a pooled AUC.

    The pooled AUC of the reviewed set compares every positive with every
    negative, and 58.7% of those pairs are cross-document. A method that
    changes a document's score *level* therefore moves the pooled number
    without changing the ranking inside a document — so the two parts are
    reported separately (the track's per-document discipline).
    """
    try:
        return grouped_auc_decomposition(labels.tolist(), scores.tolist(), documents.tolist())
    except ValueError as exc:
        return {"note": str(exc)}


def _macro(values: Mapping[str, float | None]) -> float | None:
    """The macro average of the per-document values (None when none exists)."""
    present = [value for value in values.values() if value is not None]
    return float(np.mean(present)) if present else None


def _quantile_delta(arm: np.ndarray, reference: np.ndarray) -> float | None:
    """The difference of the medians of two groups (None for an empty group)."""
    if not arm.size or not reference.size:
        return None
    return float(np.quantile(arm, 0.5) - np.quantile(reference, 0.5))


def _floor_gap(positive_scores: np.ndarray, control_scores: np.ndarray) -> float | None:
    """The gap between the verified positives' mean and the control's mean."""
    if not positive_scores.size or not control_scores.size:
        return None
    return float(positive_scores.mean() - control_scores.mean())


def _top_fifth_share_delta(
    arm_percentiles: np.ndarray, reference_percentiles: np.ndarray
) -> float | None:
    """The difference of the top-fifth shares of two rank-percentile groups."""
    if not arm_percentiles.size or not reference_percentiles.size:
        return None
    return _top_quintile_share(arm_percentiles) - _top_quintile_share(reference_percentiles)


def _chunker_slug(chunker: Chunker) -> str:
    """The chunker's short handle (its arm-name prefix)."""
    slug = getattr(chunker, "slug", None)
    return slug if isinstance(slug, str) and slug else _chunker_name(chunker)


def _chunker_name(chunker: Chunker) -> str:
    """The chunker's config-facing name (its registry key)."""
    return getattr(chunker, "name", chunker.version)


def _cell_id(code: str, paragraph_number: str, question_number: int) -> str:
    """The id of one grid cell (the file's own pair-id convention)."""
    return f"{code}-{paragraph_number}-Q{question_number}"


def _top_quintile_share(percentiles: np.ndarray) -> float | None:
    """Share of a group in the top fifth of its paragraph's candidate list."""
    if not percentiles.size:
        return None
    return float((percentiles < 0.2).mean())


logger = logging.getLogger(__name__)


# The quantile grid of the reported distributions (mirrors the probe's).
DISTRIBUTION_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)


# The hit@k grid of the per-pair rank report (the "is the pair's question
# among the paragraph's top-k candidates" measure). The same grid the probe
# reports.
HIT_KS: tuple[int, ...] = (1, 10)


# The name of the baseline arm: the full-text scoring every other arm is
# compared with.
BASELINE_ARM = "full_text"


# The chunker of the baseline arm (the track's identity chunker).
BASELINE_CHUNKER = "full_text"


# The description of the baseline arm's scoring, recorded in the artifact.
FULL_TEXT_AGGREGATION = "one chunk per text (the plain full-text cosine)"


# The name of the Stage-3 feature the cosine's calibration map is fitted on
# (the pack's full-text cosine, the Stage-0 baseline arm's score).
COSINE_FEATURE_NAME = "cosine_full"


# The names of the safe-negative rule's variants (CS-6e). The rule is a
# **margin** rule: a cell is a safe negative when it sits at least the
# chosen margin below its own paragraph's best candidate.
RULE_VARIANT_COSINE = "cosine_margin"


RULE_VARIANT_ARM = "arm_margin"


RULE_VARIANT_BOTH = "both_margins"


RULE_VARIANTS: tuple[str, ...] = (RULE_VARIANT_COSINE, RULE_VARIANT_ARM, RULE_VARIANT_BOTH)


# The fit-side criteria of a swept parameter (the semantic threshold of
# Stage 2.3): the reviewed set's AUC or its floor gap.
SEMANTIC_CRITERION_AUC = "auc"


SEMANTIC_CRITERION_FLOOR_GAP = "floor_gap"


SEMANTIC_CRITERIA: tuple[str, ...] = (SEMANTIC_CRITERION_AUC, SEMANTIC_CRITERION_FLOOR_GAP)


# The pairwise file's own id convention: "<code>-<paragraph number>-Q<question
# number>" (e.g. "NU-10-Q2"). The code is the document's short handle.
_CELL_ID_PATTERN = re.compile(r"^(?P<code>[^-]+)-\d+-Q\d+$")


_SLUG_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


# The name of the Stage-3 arm (the learned scorer beside the full-text
# baseline arm).
STAGE3_ARM = "stage3"


# The name of the decision model's arm (Stage 3.5.2): the decision model is
# an *arm* of the analysis, not another embedding model.
DECISION_ARM = "decision"


# The decision rule's variants (CS-7b). The first two are the probability
# cuts (the fixed one and the fit-side fitted one), the third is the margin
# rule with the decision model's probability as the ranking signal.
DECISION_RULE_FIXED_CUT = "decision_fixed_cut"


DECISION_RULE_FIT_THRESHOLD = "decision_fit_threshold"


DECISION_RULE_MARGIN = "decision_margin"


DECISION_RULE_VARIANTS: tuple[str, ...] = (
    DECISION_RULE_FIXED_CUT,
    DECISION_RULE_FIT_THRESHOLD,
    DECISION_RULE_MARGIN,
)


# The decision model's fixed operating point: the probability below which the
# model calls a pair a non-match. A calibrated model makes an absolute cut
# meaningful (the track's rule against absolute cuts is about the *embedding*
# geometry, whose scale no absolute cut separates).
DECISION_FIXED_CUT = 0.5
