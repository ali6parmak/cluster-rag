"""Aggregation rules: how chunk-level scores fold into one pair score.

The track's Stage 1 measures *structural decomposition* of the paragraph side
(TEXT_PROCESSING_PLAN.md section 3, Stage 1.1a): a paragraph is split into
atoms, every atom is embedded, and the atom x question similarity matrix has
to be folded into **one** score per (paragraph, question) pair. This module
holds the folding rules. It is a pure core (numpy only — no model clients, no
I/O), so a rule is testable by hand.

The three rules of the Stage-1 change set:

- ``max`` — the best chunk match. It takes a maximum over *more* candidates
  than the full-text score does, so its scale is not comparable to the
  full-text scale: the control side of the artifact is what makes that
  inflation visible (the plan's reporting rule).
- ``top_k_mean`` — the mean of the *k* best chunk matches of a pair (the
  exp07 lineage without the decay): less peak-sensitive than ``max``.
- ``mean`` — the mean over all the chunk matches of a pair (a "coverage"
  score).

The rules are pure functions of a similarity matrix and a grouping (which
chunk rows belong to which text), so nothing here depends on the corpus, the
labels or the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# The rule names (the config-facing spelling of a rule).
AGGREGATION_MAX = "max"
AGGREGATION_MEAN = "mean"
AGGREGATION_TOP_K_MEAN = "top_k_mean"

AGGREGATION_NAMES: tuple[str, ...] = (
    AGGREGATION_MAX,
    AGGREGATION_MEAN,
    AGGREGATION_TOP_K_MEAN,
)


@dataclass(frozen=True)
class Aggregation:
    """One aggregation rule: its name and its parameter (``k`` for top_k_mean).

    Attributes:
        name: One of :data:`AGGREGATION_NAMES`.
        k: The number of best chunk matches a ``top_k_mean`` averages.
    """

    name: str
    k: int | None = None

    @property
    def spec(self) -> str:
        """The config-facing spelling of the rule (``"top_k_mean:2"``)."""
        return self.name if self.k is None else f"{self.name}:{self.k}"

    @property
    def slug(self) -> str:
        """A filesystem/label-safe spelling of the rule (``"top_k_mean_2"``)."""
        return self.spec.replace(":", "_")


def parse_aggregation(spec: str) -> Aggregation:
    """Parse a config-facing aggregation spec into a rule.

    The spelling is ``"max"``, ``"mean"`` or ``"top_k_mean:k"`` — the ``k``
    is explicit, so no rule carries a hidden default.

    Args:
        spec: The rule's spelling.

    Returns:
        The parsed :class:`Aggregation`.

    Raises:
        ValueError: If the rule name is unknown or the ``k`` is missing or
            not a positive integer.
    """
    name, _, parameter = spec.partition(":")
    name = name.strip()
    if name not in AGGREGATION_NAMES:
        raise ValueError(
            f"Unknown aggregation {spec!r}; expected one of "
            f"{list(AGGREGATION_NAMES)} (top_k_mean needs an explicit k, "
            "e.g. 'top_k_mean:2')"
        )
    if name != AGGREGATION_TOP_K_MEAN:
        if parameter.strip():
            raise ValueError(f"The aggregation {name!r} takes no parameter: {spec!r}")
        return Aggregation(name=name)
    if not parameter.strip():
        raise ValueError(
            f"The aggregation 'top_k_mean' needs an explicit k: {spec!r} (e.g. 'top_k_mean:2')"
        )
    try:
        k = int(parameter)
    except ValueError as exc:
        raise ValueError(f"The aggregation 'top_k_mean' needs an integer k: {spec!r}") from exc
    if k < 1:
        raise ValueError(f"The aggregation 'top_k_mean' needs k >= 1: {spec!r}")
    return Aggregation(name=name, k=k)


def parse_aggregations(specs: Sequence[str]) -> tuple[Aggregation, ...]:
    """Parse a list of aggregation specs (order preserved, duplicates kept)."""
    return tuple(parse_aggregation(spec) for spec in specs)


def aggregate_chunk_similarities(
    similarities: np.ndarray,
    groups: Sequence[Sequence[int]],
    rule: Aggregation,
) -> np.ndarray:
    """Fold a chunk x column similarity matrix into one row per group.

    Args:
        similarities: The ``(n_chunk_rows, n_columns)`` similarity matrix.
        groups: The row indices of every output group (one entry per text:
            the rows of its chunks, in order).
        rule: The aggregation rule to apply.

    Returns:
        The ``(n_groups, n_columns)`` matrix of aggregated scores.

    Raises:
        ValueError: If a group is empty, a row index is out of range, or the
            rule is unknown.
    """
    matrix = np.asarray(similarities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("similarities must be a 2-D matrix")
    aggregated = np.zeros((len(groups), matrix.shape[1]), dtype=np.float64)
    for position, group in enumerate(groups):
        indices = np.asarray(list(group), dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"aggregation group {position} is empty")
        if indices.min() < 0 or indices.max() >= matrix.shape[0]:
            raise ValueError(f"aggregation group {position} points outside the matrix")
        rows = matrix[indices]
        if rule.name == AGGREGATION_MAX:
            aggregated[position] = rows.max(axis=0)
        elif rule.name == AGGREGATION_MEAN:
            aggregated[position] = rows.mean(axis=0)
        elif rule.name == AGGREGATION_TOP_K_MEAN:
            k = min(int(rule.k or 1), rows.shape[0])
            if k == rows.shape[0]:
                aggregated[position] = rows.mean(axis=0)
            else:
                best = np.partition(rows, rows.shape[0] - k, axis=0)[rows.shape[0] - k :]
                aggregated[position] = best.mean(axis=0)
        else:  # pragma: no cover - parse_aggregation rejects other names
            raise ValueError(f"Unknown aggregation rule {rule.name!r}")
    return aggregated


def describe_aggregation(rule: Aggregation) -> Mapping[str, object]:
    """A JSON-friendly description of a rule (recorded in the artifacts)."""
    return {"name": rule.name, "k": rule.k, "spec": rule.spec}
