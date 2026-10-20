"""Unit tests for the embed-artifact loader (Phase 4 opening D wiring).

Synthesizes ``results/upr-embed-fix/`` (npz + meta.json) in tmp dirs and checks
the loader's happy path plus every hard-failure contract: existence, provenance
sha, kinds, dimension, dtype, unit rows, id alignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cluster_rag.embed.base import l2_normalize
from cluster_rag.embed_artifact import EmbeddingArtifact, load_embedding_artifact

SOURCE_SHA = "93ba79cc0104c4cb1529864a4053e1c18497050ab135ec7ddb2d067649e9e50f"
IDS = ["r0", "r1", "r2", "r3"]
DIM = 2


def _meta(kinds=("document", "query"), dimension=DIM, **overrides):
    meta = {
        "run_id": "upr-embed-fix",
        "command": "embed",
        "kinds": list(kinds),
        "n_records": len(IDS),
        "source_sha256": SOURCE_SHA,
        "embedder": {
            "backend": "fake",
            "model_id": "fake-embedder",
            "dimension": dimension,
            "revision_or_digest": "fake-revision",
            "context_tokens": None,
            "prefixes": {"query": "", "document": "", "label": ""},
            "extra": {},
        },
        "cache": {"enabled": True},
        "preprocess": {"variant": "raw", "version": "raw-v1", "n_changed": 0},
        "kind_stats": {},
        "total_seconds": 1.0,
    }
    for key, value in overrides.items():
        meta[key] = value
    return meta


def _write_run(
    tmp_path: Path,
    run_id: str = "upr-embed-fix",
    *,
    kinds=("document", "query"),
    n_ids: int = 4,
    dim: int = DIM,
    meta: dict | None = None,
    matrices: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a synthetic embed run; returns its results root."""
    ids = IDS[:n_ids]
    rng = np.random.default_rng(0)
    base = l2_normalize(rng.uniform(size=(max(n_ids, 1), dim)).astype(np.float32))
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {"ids": np.asarray(ids, dtype="<U8")}
    if matrices is None:
        for kind in kinds:
            arrays[f"vectors_{kind}"] = base
    else:  # caller-supplied full npz array names (may mismatch sizes deliberately)
        arrays.update(matrices)
    np.savez_compressed(run_dir / "embeddings.npz", **arrays)
    (run_dir / "meta.json").write_text(json.dumps(meta or _meta(kinds=kinds)), encoding="utf-8")
    return tmp_path / "results"


def _load(results_root: Path, run_id="upr-embed-fix", kinds=("document", "query")):
    return load_embedding_artifact(
        results_root,
        run_id,
        required_kinds=kinds,
        source_sha256=SOURCE_SHA,
    )


def test_load_validates_and_lineage_is_recordable(tmp_path: Path) -> None:
    """Happy path: vectors load, lineage carries the full embed-run provenance."""
    results_root = _write_run(tmp_path)
    artifact = _load(results_root)
    assert isinstance(artifact, EmbeddingArtifact)
    assert artifact.ids == tuple(IDS)
    assert set(artifact.vectors) == {"document", "query"}
    assert artifact.vectors["document"].dtype == np.float32
    assert artifact.lineage == {
        "run_id": "upr-embed-fix",
        "backend": "fake",
        "model_id": "fake-embedder",
        "revision_or_digest": "fake-revision",
        "dimension": DIM,
        "context_tokens": None,
        "prefixes": {"query": "", "document": "", "label": ""},
        "preprocess": {"variant": "raw", "version": "raw-v1"},
        "source_sha256": SOURCE_SHA,
        "kinds": ["document", "query"],
    }


def test_select_aligns_rows_by_id(tmp_path: Path) -> None:
    """Rows follow the requested id order; subsets and duplicates stay aligned."""
    results_root = _write_run(tmp_path)
    artifact = _load(results_root)
    selected = artifact.select("document", ["r2", "r0", "r2"])
    assert selected.shape == (3, DIM) and selected.dtype == np.float32
    np.testing.assert_array_equal(selected[0], artifact.vectors["document"][2])
    np.testing.assert_array_equal(selected[1], artifact.vectors["document"][0])
    np.testing.assert_array_equal(selected[2], artifact.vectors["document"][2])
    empty = artifact.select("query", [])
    assert empty.shape == (0, DIM)
    with pytest.raises(ValueError, match="missing record id"):
        artifact.select("document", ["r0", "ghost"])


def test_missing_run_fails_with_the_fix(tmp_path: Path) -> None:
    """A run-id with no artifact names the directory and the repair command."""
    results_root = _write_run(tmp_path)
    with pytest.raises(ValueError, match="not found.*upr-embed-other"):
        _load(results_root, run_id="upr-embed-other")


def test_non_embed_run_is_rejected(tmp_path: Path) -> None:
    """A results folder without command=embed (e.g. an eval run) is refused."""
    results_root = _write_run(tmp_path, meta=_meta(command="evaluate"))
    with pytest.raises(ValueError, match="must reference an embed run"):
        _load(results_root)


def test_source_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    """Vectors from a different dataset generation must never score an eval."""
    results_root = _write_run(tmp_path, meta=_meta(source_sha256="other-sha"))
    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        _load(results_root)


def test_missing_kind_is_rejected_with_the_rerun_hint(tmp_path: Path) -> None:
    """A run that never encoded the query kind fails with --kinds guidance."""
    results_root = _write_run(tmp_path, kinds=("document",))
    with pytest.raises(ValueError, match="--kinds document,query"):
        _load(results_root)


def test_missing_meta_section_is_rejected(tmp_path: Path) -> None:
    """An unrecognizable meta.json (loader-written file required) fails loudly."""
    meta = _meta()
    del meta["preprocess"]
    results_root = _write_run(tmp_path, meta=meta)
    with pytest.raises(ValueError, match="missing 'preprocess\\.variant'"):
        _load(results_root)


def test_dimension_mismatch_between_meta_and_matrix_is_rejected(tmp_path: Path) -> None:
    """meta says 256, matrix says 2: the artifact is not trustworthy."""
    results_root = _write_run(tmp_path, meta=_meta(dimension=256))
    with pytest.raises(ValueError, match="dimension 2 disagrees"):
        _load(results_root)


def test_non_unit_or_wrong_dtype_matrices_are_rejected(tmp_path: Path) -> None:
    """The artifact-side embedding contract rejects (never repairs)."""
    zeros = np.zeros((4, DIM), dtype=np.float32)  # zero rows: |0 - 1| far above atol
    results_root = _write_run(
        tmp_path, matrices={"vectors_document": zeros, "vectors_query": zeros}
    )
    with pytest.raises(ValueError, match="unit-norm"):
        _load(results_root)


def test_ids_row_count_mismatch_and_duplicate_ids(tmp_path: Path) -> None:
    """Inconsistent ids arrays fail before any scoring can be misaligned."""
    base = l2_normalize(np.random.default_rng(1).uniform(size=(4, DIM)).astype(np.float32))
    results_root = _write_run(
        tmp_path,
        n_ids=3,
        matrices={"vectors_document": base, "vectors_query": base},
    )
    with pytest.raises(ValueError, match="4 rows but ids has 3"):
        _load(results_root)

    duplicated = np.asarray(["r0", "r0", "r2", "r3"], dtype="<U8")
    run_dir = tmp_path / "results" / "upr-embed-dup"
    run_dir.mkdir()
    base = l2_normalize(np.random.default_rng(1).uniform(size=(4, DIM)).astype(np.float32))
    arrays = {"ids": duplicated, "vectors_document": base, "vectors_query": base}
    np.savez_compressed(run_dir / "embeddings.npz", **arrays)
    (run_dir / "meta.json").write_text(json.dumps(_meta()), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate ids"):
        _load(tmp_path / "results", run_id="upr-embed-dup")
