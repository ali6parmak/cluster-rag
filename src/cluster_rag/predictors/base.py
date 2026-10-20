"""Predictor protocol (Phase 3): per-label score matrices for the harness contract.

A predictor maps texts to one score per label in the fixed label space. The
label space (the sorted taxonomy columns) and the train-only context are fixed
at construction, so `score` is a pure function of the input texts — trivial
baselines see train counts only, no val/test gold (leakage boundary).

Two views share one matrix: `score(...) >= threshold` rows (threshold view,
``eval.multilabel.apply_decision_rule``) and the descending argsort (top-k
view). Phase 4 predictors (knn / prototype / zero-shot / fusion) implement
this same port; embedding-dependent ones additionally declare
``REQUIRES_EMBEDDINGS = True`` and ``REQUIRED_KINDS`` (e.g. ``("document",
"query")``) and receive the aligned query vectors at ``score`` time plus their
train document vectors through :class:`PredictorContext`. Pure module: no
model clients, no I/O (AGENTS.md dependency-direction rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from cluster_rag.data import Record


class Predictor(Protocol):
    """The port every label scorer implements (baselines and Phase 4 alike)."""

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Score every text against every label.

        Args:
            texts: The texts being classified (query side).
            query_vectors: The aligned query-kind embeddings of ``texts``
                (row ``i`` is ``texts[i]``'s query vector), wired by the
                evaluate command from a prior embedding run
                (``eval.embedding_run_id``). Predictors with
                ``REQUIRES_EMBEDDINGS = True`` need them and fail with a clear
                message when ``None``; embedding-free predictors ignore them.

        Returns:
            A ``(n_texts, n_labels)`` float matrix, higher = more likely,
            arbitrary scale (thresholds are relative to the score semantics).
            Column order is the construction-time label space.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-serializable lineage (name, params, versions) for run artifacts."""
        ...


@dataclass(frozen=True)
class PredictorContext:
    """Construction inputs shared by all predictors.

    Attributes:
        labels: The score-column convention: every label the predictor may
            output, including labels with zero train examples (they score
            according to the predictor's fall-back, e.g. 0 for frequencies).
        train_records: The train split only — the leakage boundary. No val/
            test record ever reaches a predictor at construction time.
        train_document_vectors: Document-kind embeddings of ``train_records``
            (row ``i`` is ``train_records[i]``), required by predictors with
            ``REQUIRES_EMBEDDINGS = True`` — the evaluate command aligns them
            from a prior embedding run (``eval.embedding_run_id``). ``None``
            for embedding-free predictors. Still the train split only: no
            val/test document vector ever reaches construction.
    """

    labels: tuple[str, ...]
    train_records: tuple[Record, ...] = field(default=())
    train_document_vectors: np.ndarray | None = None

    def require_train_vectors(self) -> np.ndarray:
        """Return the validated train document vectors (embedding predictors).

        Checks the embedding contract (float32, 2-D, finite, unit rows) and
        the row-per-record alignment. The evaluate command performs the same
        checks on the full artifact; this re-checks the predictor boundary so
        a smuggled or shifted matrix fails loudly instead of scoring against
        the wrong examples.

        Raises:
            ValueError: If no vectors were provided or any contract/alignment
                check fails.
        """
        if self.train_document_vectors is None:
            raise ValueError(
                "this predictor requires train document embeddings; the evaluate "
                "command wires them from eval.embedding_run_id (a prior upr-embed-* "
                "run_id) — see PLAN.md Phase 4 opening D"
            )
        return check_vectors(
            self.train_document_vectors,
            name="train_document_vectors",
            n_rows=len(self.train_records),
        )


def check_no_extra_params(params: Mapping[str, Any], allowed: Mapping[str, Any], name: str) -> None:
    """Reject parameter keys a predictor does not support (fail before the run)."""
    unknown = [key for key in params if key not in allowed]
    if unknown:
        raise ValueError(
            f"predictor {name!r} does not support parameter(s) {unknown}; "
            f"supported: {sorted(allowed)}"
        )


UNIT_NORM_ATOL = 1e-3  # stored float32 unit vectors sit ~1e-7 from 1.0; generous margin


def check_vectors(matrix: np.ndarray, *, name: str, n_rows: int | None = None) -> np.ndarray:
    """Assert the embedding contract on an already-materialized matrix.

    Unlike :func:`cluster_rag.embed.base.ensure_embedding_contract` (which
    repairs by re-normalizing), this *rejects*: predictor/index boundaries
    must fail loudly on a non-unit or misaligned matrix rather than silently
    scoring against different geometry.

    Args:
        matrix: Candidate embedding matrix.
        name: Boundary name for error messages (e.g. ``"train_document_vectors"``).
        n_rows: Expected row count, when the matrix must align with records.

    Returns:
        The validated matrix itself (unmodified float32 array).

    Raises:
        ValueError: On shape, dtype, finiteness, unit-norm or row-count
            violations.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d); got shape {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"{name} must be float32; got {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")
    norms = np.linalg.norm(matrix, axis=1)
    bad = np.flatnonzero(np.abs(norms - 1.0) > UNIT_NORM_ATOL)
    if bad.size:
        row = int(bad[0])
        raise ValueError(
            f"{name} rows must be L2-normalized (norm within {UNIT_NORM_ATOL} of 1.0); "
            f"row {row} has norm {norms[row]:.6f} — check the embedding source "
            "(embeddings are unit-norm at every boundary per AGENTS.md)"
        )
    if n_rows is not None and matrix.shape[0] != n_rows:
        raise ValueError(
            f"{name} must align row-by-row with its records: got {matrix.shape[0]} "
            f"rows for {n_rows} record(s) — the evaluate command aligns vectors to "
            "records by id; passing an unaligned matrix would score the wrong examples"
        )
    return matrix
