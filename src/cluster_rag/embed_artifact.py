"""Resolve and validate a prior ``embed`` run's vector artifact (Phase 4 opening D).

The evaluate command points at a previous embedding run through
``eval.embedding_run_id`` (a ``upr-embed-*`` run_id); this module loads
``results/<run_id>/embeddings.npz`` + ``meta.json``, validates them hard
(existence, provenance, required encode kinds, the embedding contract) and
aligns vector rows to record ids. File/numpy I/O like ``data.py`` — no model
clients (AGENTS.md dependency-direction rule); cli.py orchestrates.

Everything fails loudly: a missing artifact, a kind the predictor needs but the
run never encoded, a source-sha mismatch (the dataset changed since the
vectors were made), or a matrix that violates the embedding contract each
raise with a message naming the fix — silently scoring against wrong or
shifted vectors is the one failure mode here that must be impossible.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_UNIT_NORM_ATOL = 1e-3  # same tolerance as predictors/base.check_vectors


@dataclass(frozen=True)
class EmbeddingArtifact:
    """A validated ``embed`` run artifact, aligned-selectable by record id."""

    run_id: str
    ids: tuple[str, ...]
    meta: dict[str, Any]
    vectors: dict[str, np.ndarray] = field(repr=False)

    @property
    def lineage(self) -> dict[str, Any]:
        """Provenance block for the consuming run's metrics.json (eval lineage)."""
        embedder = self.meta["embedder"]
        preprocess = self.meta["preprocess"]
        return {
            "run_id": self.run_id,
            "backend": embedder["backend"],
            "model_id": embedder["model_id"],
            "revision_or_digest": embedder["revision_or_digest"],
            "dimension": embedder["dimension"],
            "context_tokens": embedder.get("context_tokens"),
            "prefixes": dict(embedder["prefixes"]),
            "preprocess": {
                "variant": preprocess["variant"],
                "version": preprocess["version"],
            },
            "source_sha256": self.meta["source_sha256"],
            "kinds": list(self.vectors),
        }

    def select(self, kind: str, record_ids: Sequence[str]) -> np.ndarray:
        """Rows of ``vectors_<kind>`` in ``record_ids`` order (id-aligned).

        Args:
            kind: Encode kind validated at load time (KeyError impossible
                through the loader path).
            record_ids: Record ids to align; duplicates select the same row
                twice, order is preserved.

        Returns:
            ``(len(record_ids), d)`` float32 matrix.

        Raises:
            ValueError: If a record id is absent from the artifact (a split
                from a different dataset generation would score the wrong
                vectors).
        """
        matrix = self.vectors[kind]
        position = {rid: row for row, rid in enumerate(self.ids)}
        missing = [rid for rid in record_ids if rid not in position]
        if missing:
            raise ValueError(
                f"embedding run {self.run_id!r} is missing record id(s) {list(missing[:5])} "
                f"({len(missing)} total) — the artifact and the dataset disagree; "
                "re-run the embedding for the current data"
            )
        if not record_ids:
            return np.zeros((0, matrix.shape[1]), dtype=np.float32)
        rows = np.stack([matrix[position[rid]] for rid in record_ids])
        return rows.astype(np.float32, copy=False)


def _required(mapping: dict[str, Any], key_path: str) -> Any:
    """Fetch a ``meta.json`` entry, failing with the full path in the message."""
    node: Any = mapping
    for key in key_path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(
                f"embedding run meta.json is missing {key_path!r} — not a recognizable "
                "embed-run artifact (meta.json is written by the embed command)"
            )
        node = node[key]
    return node


def load_embedding_artifact(
    results_root: Path,
    run_id: str,
    *,
    required_kinds: Sequence[str],
    source_sha256: str,
) -> EmbeddingArtifact:
    """Load and hard-validate ``results/run_id/embeddings.npz`` (+ meta.json).

    Args:
        results_root: The configured ``paths.results`` directory.
        run_id: The embedding run's id (``eval.embedding_run_id``).
        required_kinds: Encode kinds the consuming predictor needs
            (``document``/``query``/``label``); all must be present.
        source_sha256: The sha256 of the current raw dataset (the same value
            the split provenance records) — the artifact must have been built
            from exactly this data.

    Returns:
        The validated :class:`EmbeddingArtifact`.

    Raises:
        ValueError: On any existence, provenance, kind, id or embedding-
            contract violation, with the fix named in the message.
    """
    if not required_kinds:
        raise ValueError("load_embedding_artifact requires at least one encode kind")
    artifact_dir = Path(results_root) / run_id
    npz_path = artifact_dir / "embeddings.npz"
    meta_path = artifact_dir / "meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise ValueError(
            f"embedding run {run_id!r} not found: expected {npz_path} plus meta.json. "
            "Run the embedding first (cluster-rag embed, or the upr-embed-* sweep "
            "cells), then point eval.embedding_run_id at that run_id."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("command") != "embed":
        raise ValueError(
            f"run {run_id!r} meta.json says command={meta.get('command')!r}; "
            "eval.embedding_run_id must reference an embed run (one with an "
            "embeddings.npz artifact)"
        )
    recorded_sha = _required(meta, "source_sha256")
    if recorded_sha != source_sha256:
        raise ValueError(
            f"embedding run {run_id!r} was built from a different dataset "
            f"(source_sha256 mismatch): the vectors and the current data disagree; "
            "re-run the embedding — do not evaluate against stale vectors."
        )

    meta_kinds = _required(meta, "kinds")
    missing = [kind for kind in required_kinds if kind not in meta_kinds]
    if missing:
        raise ValueError(
            f"embedding run {run_id!r} lacks encode kind(s) {missing} "
            f"(kinds recorded: {meta_kinds}); re-run the embed command with "
            "--kinds document,query (the artifact holds one matrix per kind)"
        )

    dimension = _required(meta, "embedder.dimension")
    _required(meta, "embedder.backend")
    _required(meta, "embedder.model_id")
    _required(meta, "embedder.revision_or_digest")
    _required(meta, "embedder.prefixes")
    _required(meta, "preprocess.variant")
    _required(meta, "preprocess.version")

    with np.load(npz_path) as data:
        ids = np.asarray(data["ids"])
        vectors: dict[str, np.ndarray] = {}
        for kind in required_kinds:
            matrix = np.asarray(data[f"vectors_{kind}"])
            if matrix.shape[0] != ids.shape[0]:
                raise ValueError(
                    f"embedding run {run_id!r} is inconsistent: matrix "
                    f"vectors_{kind} has {matrix.shape[0]} rows but ids has "
                    f"{ids.shape[0]} entries"
                )
            _check_artifact_matrix(matrix, run_id=run_id, kind=kind, dimension=dimension)
            vectors[kind] = matrix.astype(np.float32, copy=False)

    id_tuple = tuple(str(rid) for rid in ids)
    if len(set(id_tuple)) != len(id_tuple):
        raise ValueError(
            f"embedding run {run_id!r} has duplicate ids in its ids array — "
            "id alignment would be ambiguous; re-run the embedding"
        )
    return EmbeddingArtifact(run_id=run_id, ids=id_tuple, meta=meta, vectors=vectors)


def _check_artifact_matrix(matrix: np.ndarray, *, run_id: str, kind: str, dimension: int) -> None:
    """Artifact-side embedding-contract check (reject, don't repair)."""
    if matrix.ndim != 2:
        raise ValueError(
            f"embedding run {run_id!r}: vectors_{kind} must be 2-D; got {matrix.shape}"
        )
    if matrix.dtype != np.float32:
        raise ValueError(
            f"embedding run {run_id!r}: vectors_{kind} must be float32; got {matrix.dtype}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"embedding run {run_id!r}: vectors_{kind} contains non-finite values")
    if matrix.shape[1] != dimension:
        raise ValueError(
            f"embedding run {run_id!r}: vectors_{kind} dimension {matrix.shape[1]} "
            f"disagrees with meta.json dimension {dimension}"
        )
    norms = np.linalg.norm(matrix, axis=1)
    bad = np.flatnonzero(np.abs(norms - 1.0) > _UNIT_NORM_ATOL)
    if bad.size:
        row = int(bad[0])
        raise ValueError(
            f"embedding run {run_id!r}: vectors_{kind} rows must be unit-norm; "
            f"row {row} has norm {norms[row]:.6f} — the artifact is corrupted or "
            "was written outside the embed command"
        )
