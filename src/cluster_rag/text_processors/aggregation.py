"""Aggregation rules: how chunk-level scores fold into one pair score.

The track measures *structural decomposition* of the paragraph side
(TEXT_PROCESSING_PLAN.md section 3): a paragraph is split into chunks, every
chunk is embedded, and the chunk x question similarity matrix has to be folded
into **one** score per (paragraph, question) pair. This module holds the
folding rules. It is a pure core (numpy only — no model clients, no I/O), so
a rule is testable by hand.

The Stage-1 rules (the paragraph's sentence/clause atoms):

- ``max`` — the best chunk match. It takes a maximum over *more* candidates
  than the full-text score does, so its scale is not comparable to the
  full-text scale: the control side of the artifact is what makes that
  inflation visible (the plan's reporting rule).
- ``top_k_mean`` — the mean of the *k* best chunk matches of a pair (the
  exp07 lineage without the decay): less peak-sensitive than ``max``.
- ``mean`` — the mean over all the chunk matches of a pair (a "coverage"
  score).

The Stage-2 rules (the plan's Stage-2 menu):

- ``decayed_top_k:k:decay:spread`` (exp07) — a **decaying average** of the
  *k* best chunk matches (the best match counts fully, the second half as
  much, the third a quarter, ...) minus a **spread penalty** (the standard
  deviation of the same *k* matches times the penalty weight). The rule asks
  "is the evidence concentrated in one chunk or spread over several?".
- ``local_global:alpha:<local rule>`` (exp06) — the **local+global blend**:
  ``alpha * (the local rule's score) + (1 - alpha) * (the full-text score)``.
  It needs the full-text score as a second input, so the arm's score is a
  blend of two matrices.
- ``peak:k:margin`` (exp08) — the **peak shortcut**: the score is the mean of
  the *k* best chunk matches (the "broad" evidence), unless a single chunk
  match stands out from that mean by at least the margin, in which case the
  score is the peak itself. The reference lineage used two *absolute* score
  bars; absolute cuts are forbidden on this data (the track's "label-0 above
  label-1 p95" anomaly — rank/margin rules only), so the rule is written as a
  margin over the pair's own evidence.

The rules are pure functions of a similarity matrix and a grouping (which
chunk rows belong to which text), so nothing here depends on the corpus, the
labels or the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The rule names (the config-facing spelling of a rule).
AGGREGATION_MAX = "max"
AGGREGATION_MEAN = "mean"
AGGREGATION_TOP_K_MEAN = "top_k_mean"
AGGREGATION_DECAYED_TOP_K = "decayed_top_k"
AGGREGATION_LOCAL_GLOBAL = "local_global"
AGGREGATION_PEAK = "peak"

AGGREGATION_NAMES: tuple[str, ...] = (
    AGGREGATION_MAX,
    AGGREGATION_MEAN,
    AGGREGATION_TOP_K_MEAN,
    AGGREGATION_DECAYED_TOP_K,
    AGGREGATION_LOCAL_GLOBAL,
    AGGREGATION_PEAK,
)


@dataclass(frozen=True)
class Aggregation:
    """One aggregation rule: its name and its parameters.

    Attributes:
        name: One of :data:`AGGREGATION_NAMES`.
        k: The number of best chunk matches the rule averages (``top_k_mean``,
            ``decayed_top_k``, ``peak``).
        decay: The decay of the weights of the *k* best matches
            (``decayed_top_k``; 1 = a plain mean, 0.5 = halving per rank).
        spread: The weight of the spread penalty (``decayed_top_k``).
        alpha: The weight of the local score in the blend
            (``local_global``).
        margin: The margin the peak must clear over the broad evidence
            (``peak``).
        local: The local rule of the blend (``local_global``).
    """

    name: str
    k: int | None = None
    decay: float | None = None
    spread: float | None = None
    alpha: float | None = None
    margin: float | None = None
    local: Aggregation | None = field(default=None, repr=False, compare=True)

    @property
    def spec(self) -> str:
        """The config-facing spelling of the rule (``"top_k_mean:2"``)."""
        if self.name == AGGREGATION_TOP_K_MEAN:
            return f"{self.name}:{self.k}"
        if self.name == AGGREGATION_DECAYED_TOP_K:
            return f"{self.name}:{self.k}:{self.decay}:{self.spread}"
        if self.name == AGGREGATION_LOCAL_GLOBAL:
            local = self.local.spec if self.local is not None else AGGREGATION_MAX
            return f"{self.name}:{self.alpha}:{local}"
        if self.name == AGGREGATION_PEAK:
            return f"{self.name}:{self.k}:{self.margin}"
        return self.name

    @property
    def slug(self) -> str:
        """A filesystem/label-safe spelling of the rule (``"top_k_mean_2"``)."""
        return self.spec.replace(":", "_")

    def needs_global(self) -> bool:
        """Whether the rule needs the full-text score matrix as an input."""
        return self.name == AGGREGATION_LOCAL_GLOBAL


def _parse_int(parameter: str, spec: str) -> int:
    """Parse a positive integer parameter.

    Raises:
        ValueError: If the parameter is not a positive integer.
    """
    try:
        value = int(parameter)
    except ValueError as exc:
        raise ValueError(f"The aggregation {spec!r} needs an integer k: {parameter!r}") from exc
    if value < 1:
        raise ValueError(f"The aggregation {spec!r} needs an integer k >= 1: {parameter!r}")
    return value


def _parse_float(parameter: str, spec: str) -> float:
    """Parse a finite float parameter.

    Raises:
        ValueError: If the parameter is not a finite number.
    """
    try:
        value = float(parameter)
    except ValueError as exc:
        raise ValueError(f"The aggregation {spec!r} needs a number: {parameter!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"The aggregation {spec!r} needs a finite number: {parameter!r}")
    return value


def parse_aggregation(spec: str) -> Aggregation:
    """Parse a config-facing aggregation spec into a rule.

    The spelling is positional: ``"max"``, ``"mean"``, ``"top_k_mean:k"``,
    ``"decayed_top_k:k:decay:spread"``, ``"local_global:alpha:<local rule>"``
    and ``"peak:k:margin"``. The ``k`` is explicit, so no rule carries a
    hidden default.

    Args:
        spec: The rule's spelling.

    Returns:
        The parsed :class:`Aggregation`.

    Raises:
        ValueError: If the rule name is unknown or a parameter is missing,
            malformed or out of range.
    """
    parts = spec.split(":")
    name = parts[0].strip()
    if name not in AGGREGATION_NAMES:
        raise ValueError(
            f"Unknown aggregation {spec!r}; expected one of "
            f"{list(AGGREGATION_NAMES)} (top_k_mean needs an explicit k, "
            "e.g. 'top_k_mean:2')"
        )
    parameters = parts[1:]
    if name in (AGGREGATION_MAX, AGGREGATION_MEAN):
        if parameters:
            raise ValueError(f"The aggregation {name!r} takes no parameter: {spec!r}")
        return Aggregation(name=name)
    if name == AGGREGATION_TOP_K_MEAN:
        if len(parameters) != 1:
            raise ValueError(
                f"The aggregation 'top_k_mean' needs an explicit k: {spec!r} (e.g. 'top_k_mean:2')"
            )
        return Aggregation(name=name, k=_parse_int(parameters[0], spec))
    if name == AGGREGATION_DECAYED_TOP_K:
        if len(parameters) != 3:
            raise ValueError(
                f"The aggregation 'decayed_top_k' needs k, decay and spread: {spec!r} "
                "(e.g. 'decayed_top_k:3:0.5:0.25')"
            )
        k = _parse_int(parameters[0], spec)
        decay = _parse_float(parameters[1], spec)
        spread = _parse_float(parameters[2], spec)
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"The decay of {spec!r} must be in (0, 1], got {decay}")
        if spread < 0.0:
            raise ValueError(f"The spread penalty of {spec!r} must be >= 0, got {spread}")
        return Aggregation(name=name, k=k, decay=decay, spread=spread)
    if name == AGGREGATION_LOCAL_GLOBAL:
        if len(parameters) < 2:
            raise ValueError(
                f"The aggregation 'local_global' needs alpha and a local rule: {spec!r} "
                "(e.g. 'local_global:0.5:max')"
            )
        alpha = _parse_float(parameters[0], spec)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"The alpha of {spec!r} must be in [0, 1], got {alpha}")
        local = parse_aggregation(":".join(parameters[1:]))
        if local.needs_global():
            raise ValueError(f"The local rule of {spec!r} must not need the full-text score")
        return Aggregation(name=name, alpha=alpha, local=local)
    if name == AGGREGATION_PEAK:
        if len(parameters) != 2:
            raise ValueError(
                f"The aggregation 'peak' needs k and a margin: {spec!r} (e.g. 'peak:3:0.05')"
            )
        k = _parse_int(parameters[0], spec)
        margin = _parse_float(parameters[1], spec)
        if margin < 0.0:
            raise ValueError(f"The margin of {spec!r} must be >= 0, got {margin}")
        return Aggregation(name=name, k=k, margin=margin)
    raise ValueError(f"Unknown aggregation rule {name!r}")  # pragma: no cover


def parse_aggregations(specs: Sequence[str]) -> tuple[Aggregation, ...]:
    """Parse a list of aggregation specs (order preserved, duplicates kept)."""
    return tuple(parse_aggregation(spec) for spec in specs)


def _best_k(rows: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` best values of every column, in descending order."""
    ordered = np.sort(rows, axis=0)[::-1]
    return ordered[:k]


def _apply_rule(
    rows: np.ndarray,
    rule: Aggregation,
    global_row: np.ndarray | None,
) -> np.ndarray:
    """Apply one rule to the chunk rows of a single text.

    Args:
        rows: The ``(n_chunks, n_columns)`` similarity rows of one text.
        rule: The rule to apply.
        global_row: The text's full-text score row (needed by ``local_global``).

    Returns:
        The aggregated score of every column.

    Raises:
        ValueError: If the rule needs the full-text score and none was given.
    """
    if rule.name == AGGREGATION_MAX:
        return rows.max(axis=0)
    if rule.name == AGGREGATION_MEAN:
        return rows.mean(axis=0)
    if rule.name == AGGREGATION_TOP_K_MEAN:
        k = min(int(rule.k or 1), rows.shape[0])
        return _best_k(rows, k).mean(axis=0)
    if rule.name == AGGREGATION_DECAYED_TOP_K:
        k = min(int(rule.k or 1), rows.shape[0])
        best = _best_k(rows, k)
        weights = float(rule.decay or 1.0) ** np.arange(k, dtype=np.float64)
        decayed = (best * weights[:, None]).sum(axis=0) / weights.sum()
        spread = best.std(axis=0) if k > 1 else np.zeros(best.shape[1], dtype=np.float64)
        return decayed - float(rule.spread or 0.0) * spread
    if rule.name == AGGREGATION_PEAK:
        k = min(int(rule.k or 1), rows.shape[0])
        best = _best_k(rows, k)
        broad = best.mean(axis=0)
        peak = best[0]
        return np.where((peak - broad) >= float(rule.margin or 0.0), peak, broad)
    if rule.name == AGGREGATION_LOCAL_GLOBAL:
        if global_row is None:
            raise ValueError(
                f"The aggregation {rule.spec!r} needs the full-text score matrix "
                "(the local+global blend)"
            )
        local = rule.local or Aggregation(name=AGGREGATION_MAX)
        return (
            float(rule.alpha or 0.0) * _apply_rule(rows, local, None)
            + (1.0 - float(rule.alpha or 0.0)) * global_row
        )
    raise ValueError(f"Unknown aggregation rule {rule.name!r}")  # pragma: no cover


def aggregate_chunk_similarities(
    similarities: np.ndarray,
    groups: Sequence[Sequence[int]],
    rule: Aggregation,
    *,
    global_similarities: np.ndarray | None = None,
) -> np.ndarray:
    """Fold a chunk x column similarity matrix into one row per group.

    Args:
        similarities: The ``(n_chunk_rows, n_columns)`` similarity matrix.
        groups: The row indices of every output group (one entry per text:
            the rows of its chunks, in order).
        rule: The aggregation rule to apply.
        global_similarities: The full-text score matrix (one row per text, in
            the same order as ``groups``) — the second input of the
            local+global blend (``exp06``).

    Returns:
        The ``(n_groups, n_columns)`` matrix of aggregated scores.

    Raises:
        ValueError: If a group is empty, a row index is out of range, the rule
            is unknown, or the rule needs the full-text score and none was
            given.
    """
    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("similarities must be a 2-D matrix")
    if global_similarities is not None:
        global_matrix = np.asarray(global_similarities, dtype=np.float64)
        if global_matrix.ndim != 2 or global_matrix.shape[0] != len(groups):
            raise ValueError(
                "the full-text score matrix must have one row per group "
                f"({len(groups)}), got shape {global_matrix.shape}"
            )
    else:
        global_matrix = None
    aggregated = np.zeros((len(groups), matrix.shape[1]), dtype=np.float64)
    for position, group in enumerate(groups):
        indices = np.asarray(list(group), dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"aggregation group {position} is empty")
        if indices.min() < 0 or indices.max() >= matrix.shape[0]:
            raise ValueError(f"aggregation group {position} points outside the matrix")
        global_row = None if global_matrix is None else global_matrix[position]
        aggregated[position] = _apply_rule(matrix[indices], rule, global_row)
    return aggregated


def describe_aggregation(rule: Aggregation) -> Mapping[str, Any]:
    """A JSON-friendly description of a rule (recorded in the artifacts)."""
    return {
        "name": rule.name,
        "k": rule.k,
        "decay": rule.decay,
        "spread": rule.spread,
        "alpha": rule.alpha,
        "margin": rule.margin,
        "local": None if rule.local is None else dict(describe_aggregation(rule.local)),
        "spec": rule.spec,
    }
