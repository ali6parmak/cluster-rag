"""The control set (subset B): the same-document grid and its seeded sample.

The OMCT pairwise file covers a small, **rank-biased** part of the
same-document grid: the judged pairs are the ones a model put on top and a
reviewer looked at (TEXT_PROCESSING_PLAN.md section 0.1, item 1). The rest of
the grid — the *unjudged* cells — is the only rank-unbiased part of it: the
part nobody's ranking chose. This module turns that part into a reproducible
sample, the "control", and defines the measurement the track's reporting rule
needs next to every AUC: a pair's **rank percentile** inside its paragraph's
own same-document candidate list.

Terms (binding, TEXT_PROCESSING_PLAN.md section 3, Stage 0.1):

- subset **A** — the reviewer-verified hard negatives (ground truth, hard by
  construction): the file's label-0 records;
- subset **C** — the verified positives: the file's label-1 records;
- subset **B** — this control: sampled **unjudged** cells. It is *not* a
  negative set; it is "presumed-negative" only where a label-shaped number is
  unavoidable, and it is never pooled with subset A.

Pure core (AGENTS.md dependency direction): numpy and the standard library
only — no model clients, no I/O. The sampling is deterministic: the same data
and the same seed give the same cells, so a later stage can re-use exactly the
cells recorded in the run's artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# The one sampling rule of Stage 0: per-document quotas proportional to the
# number of unjudged cells, drawn with the largest-remainder method, then a
# seeded draw without replacement per document (documents in sorted order).
SAMPLING_RULE_PROPORTIONAL = "proportional"
SAMPLING_RULES: tuple[str, ...] = (SAMPLING_RULE_PROPORTIONAL,)

SAMPLING_RULE_DESCRIPTIONS: dict[str, str] = {
    SAMPLING_RULE_PROPORTIONAL: (
        "per-document quotas proportional to the number of unjudged same-document "
        "cells (largest-remainder method), then a seeded draw without replacement "
        "per document with numpy.random.default_rng(seed), documents in sorted order"
    ),
}


@dataclass(frozen=True)
class DocumentGrid:
    """One document's same-document cell grid: its paragraphs x its questions.

    A cell is the pair (paragraph row, question column) of one document. The
    grid holds every cell of that document; ``judged`` holds the cells with a
    record in the pairwise file (the rest are the unjudged cells the control
    is drawn from). Rows and columns are **global** indices into the run's
    paragraph and question vectors, so a cell is comparable across documents.
    """

    document: str
    paragraph_rows: tuple[int, ...]
    question_cols: tuple[int, ...]
    judged: frozenset[tuple[int, int]]

    @property
    def n_cells(self) -> int:
        """Number of cells in this document's grid."""
        return len(self.paragraph_rows) * len(self.question_cols)

    @property
    def n_judged(self) -> int:
        """Number of cells with a record in the pairwise file."""
        return len(self.judged)

    @property
    def n_unjudged(self) -> int:
        """Number of cells without a record (the control's sampling pool)."""
        return self.n_cells - self.n_judged

    def unjudged_cells(self) -> list[tuple[int, int]]:
        """The unjudged cells, in a deterministic order (rows, then columns)."""
        judged = self.judged
        return [
            (row, col)
            for row in self.paragraph_rows
            for col in self.question_cols
            if (row, col) not in judged
        ]


def allocate_quotas(available: Mapping[str, int], size: int) -> dict[str, int]:
    """Split ``size`` slots over the strata proportionally to their sizes.

    Largest-remainder method: every stratum gets its floor share of
    ``size * available[stratum] / total`` and the leftover slots go to the
    largest fractional parts (ties broken by the stratum name, so the
    allocation is deterministic). A stratum never receives more slots than it
    has cells.

    Args:
        available: Stratum name -> number of cells available in it.
        size: Total number of slots to distribute.

    Returns:
        The per-stratum quota (sums to ``size``).

    Raises:
        ValueError: If ``size`` is negative, no stratum has cells, or
            ``size`` exceeds the number of available cells.
    """
    if size < 0:
        raise ValueError(f"control size must not be negative, got {size}")
    total = sum(available.values())
    if total <= 0:
        raise ValueError("no unjudged cells to sample from")
    if size > total:
        raise ValueError(
            f"cannot draw {size} control cells from {total} unjudged cells "
            "(lower analysis.control_size)"
        )
    exact = {name: size * count / total for name, count in available.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    leftover = size - sum(quotas.values())
    order = sorted(available, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:leftover]:
        quotas[name] += 1
    for name, quota in quotas.items():
        if quota > available[name]:  # pragma: no cover - impossible for size <= total
            raise ValueError(
                f"quota {quota} for stratum {name!r} exceeds its {available[name]} cells"
            )
    return quotas


@dataclass(frozen=True)
class ControlSample:
    """The drawn control cells and the provenance of the draw.

    ``cells`` holds ``(document, paragraph row, question column)`` triples in
    the draw order (documents sorted, cells sorted inside a document);
    ``quotas`` and ``available`` record how many cells each document
    contributed and how many it could have contributed.
    """

    cells: tuple[tuple[str, int, int], ...] = ()
    quotas: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    size: int = 0
    seed: int = 0
    rule: str = SAMPLING_RULE_PROPORTIONAL

    @property
    def n_cells(self) -> int:
        """Number of drawn control cells."""
        return len(self.cells)

    def describe(self) -> dict[str, object]:
        """JSON-serializable provenance of the draw (recorded in the artifact)."""
        return {
            "size_requested": self.size,
            "size_drawn": self.n_cells,
            "seed": self.seed,
            "sampling_rule": self.rule,
            "sampling_rule_description": SAMPLING_RULE_DESCRIPTIONS.get(self.rule, self.rule),
            "per_document": {
                document: {
                    "unjudged_cells_available": self.available.get(document, 0),
                    "quota": self.quotas.get(document, 0),
                    "n_drawn": sum(1 for cell in self.cells if cell[0] == document),
                }
                for document in sorted(self.quotas)
            },
        }


def sample_control(
    grids: Sequence[DocumentGrid],
    *,
    size: int,
    seed: int,
    rule: str = SAMPLING_RULE_PROPORTIONAL,
) -> ControlSample:
    """Draw the control: a seeded, proportional sample of the unjudged cells.

    The draw is deterministic: the same grids, size and seed give the same
    cells, so a later stage re-uses exactly the cells recorded in the run's
    artifact (the cell ids are in the artifact as well).

    Args:
        grids: The per-document grids (the unjudged cells are the pool).
        size: Number of cells to draw.
        seed: The seed of the numpy generator (from the run's config).
        rule: The sampling rule (only the Stage-0 rule exists so far).

    Returns:
        The :class:`ControlSample` with the drawn cells and their provenance.

    Raises:
        ValueError: If the rule is unknown, there is no grid, or the size
            exceeds the number of unjudged cells.
    """
    if rule not in SAMPLING_RULES:
        raise ValueError(f"Unknown sampling rule {rule!r}; expected one of {SAMPLING_RULES}")
    if not grids:
        raise ValueError("no document grid to sample from")
    available = {grid.document: grid.n_unjudged for grid in grids}
    quotas = allocate_quotas(available, size)
    rng = np.random.default_rng(seed)
    cells: list[tuple[str, int, int]] = []
    for grid in sorted(grids, key=lambda item: item.document):
        quota = quotas[grid.document]
        if quota == 0:
            continue
        pool = grid.unjudged_cells()
        chosen = rng.choice(len(pool), size=quota, replace=False)
        for index in sorted(int(index) for index in chosen):
            row, col = pool[index]
            cells.append((grid.document, row, col))
    return ControlSample(
        cells=tuple(cells),
        quotas=quotas,
        available=available,
        size=size,
        seed=seed,
        rule=rule,
    )


def rank_positions(
    similarities: np.ndarray,
    rows: Sequence[int],
    columns: Sequence[int],
    candidate_columns: Sequence[Sequence[int]],
) -> np.ndarray:
    """The rank index of each measured pair inside its paragraph's candidate list.

    The candidate list of a paragraph row is the questions of its document(s)
    (``candidate_columns[row]``) — the same list the retrieval scope uses, so
    the judged pairs and the control cells are measured on one scale.

    **Convention:** the index counts from the top: ``0`` is the best-scoring
    candidate. Ties keep the candidates' own order (a stable sort), so the
    number is reproducible. The track's *rank percentile* is this index
    divided by the candidate count, and a pair's *hit@k* is this index below
    ``k``.

    Args:
        similarities: The (paragraphs x questions) score matrix.
        rows: The paragraph row of each measured pair.
        columns: The question column of each measured pair.
        candidate_columns: The candidate columns per paragraph row.

    Returns:
        The rank index of each pair, in the order of ``rows``.
    """
    positions = np.zeros(len(rows), dtype=np.int64)
    ranks_by_row: dict[int, dict[int, int]] = {}
    for position, (row, column) in enumerate(zip(rows, columns, strict=True)):
        ranks = ranks_by_row.get(row)
        if ranks is None:
            candidates = list(candidate_columns[row])
            order = sorted(candidates, key=lambda col: -float(similarities[row, col]))
            ranks = {col: index for index, col in enumerate(order)}
            ranks_by_row[row] = ranks
        positions[position] = ranks[column]
    return positions


def percentiles_from_positions(
    positions: np.ndarray,
    rows: Sequence[int],
    candidate_columns: Sequence[Sequence[int]],
) -> np.ndarray:
    """The rank percentiles of given rank positions.

    The percentile of a position is the position divided by the size of the
    paragraph's candidate list (see :func:`rank_percentiles` for the
    convention).

    Args:
        positions: The rank index of each measured pair (0 = the best
            candidate).
        rows: The paragraph row of each measured pair.
        candidate_columns: The candidate columns per paragraph row.

    Returns:
        The percentile of each pair, in the order of ``rows``.
    """
    sizes = np.asarray([len(candidate_columns[row]) for row in rows], dtype=np.float64)
    return np.asarray(positions, dtype=np.float64) / sizes


def rank_percentiles(
    similarities: np.ndarray,
    rows: Sequence[int],
    columns: Sequence[int],
    candidate_columns: Sequence[Sequence[int]],
) -> np.ndarray:
    """The rank percentile of each pair inside its paragraph's candidate list.

    The candidate list of a paragraph row is the questions of its document(s)
    (``candidate_columns[row]``) — the same list the retrieval scope uses, so
    the judged pairs and the control cells are measured on one scale.

    **Convention:** the percentile is the *position from the top* of that
    list: ``0.0`` is the best-scoring candidate, ``1.0`` the worst (the
    bottom candidate of a list of n sits at ``(n - 1) / n``). A random
    candidate therefore sits at about ``0.5``. Ties keep the candidates' own
    order (a stable sort), so the number is reproducible.

    Args:
        similarities: The (paragraphs x questions) score matrix.
        rows: The paragraph row of each measured pair.
        columns: The question column of each measured pair.
        candidate_columns: The candidate columns per paragraph row.

    Returns:
        The percentile of each pair, in the order of ``rows``.
    """
    positions = rank_positions(similarities, rows, columns, candidate_columns)
    return percentiles_from_positions(positions, rows, candidate_columns)
