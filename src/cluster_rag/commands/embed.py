"""The ``embed`` and ``labelembed`` commands.

``embed`` writes the dataset's text embeddings through the disk cache
(Phase 2); ``labelembed`` writes the one-text-per-label artifact the
zero-shot predictor and the fusion's c-term consume (Phase 4).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np

from cluster_rag.commands._common import (
    _append_command_rows,
    _require_run_id,
    _upr_label_catalog,
    _warn_upr_prefix,
    _write_resolved_config,
)
from cluster_rag.config import AppConfig
from cluster_rag.data import load_jsonl
from cluster_rag.dataset_stats import file_sha256
from cluster_rag.embed import KINDS, build_embedder
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.experiment_table import EMBED_COLUMNS, EMBED_KEY_COLUMNS
from cluster_rag.preprocess import build_preprocess_spec

logger = logging.getLogger(__name__)


def _parse_kinds(raw: str) -> list[str]:
    """Parse and validate the --kinds flag ("document" or "document,query")."""
    kinds = [part.strip() for part in raw.split(",") if part.strip()]
    if not kinds:
        raise ValueError("--kinds must name at least one encode kind")
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"Unknown encode kind {kind!r}; expected one of {KINDS}")
    return kinds


def _cmd_embed(config: AppConfig, kinds: Sequence[str]) -> int:
    """Run the embed command: preprocess, embed through the disk cache, persist.

    Writes ``results/<run_id>/embeddings.npz`` (one ``vectors_<kind>`` matrix
    plus ``ids``) and ``meta.json`` (model/digest/revision, prefixes,
    preprocessing diagnostics, cache hit/miss statistics, timings). A rerun
    with the same config must report ``cache_hits == n_records``.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.embed is None:
        raise ValueError(
            "the embed: section (backend, model_id, prefixes) is required for the `embed` command"
        )
    report = load_jsonl(config.paths.data_raw)
    if report.errors:
        logger.warning("%d malformed line(s) dropped before embedding", report.n_invalid)
    ids = [record.id for record in report.records]
    raw_texts = [record.text for record in report.records]

    spec = build_preprocess_spec(config.preprocess)
    processed = spec.apply(raw_texts)
    preprocess_meta = spec.describe(raw_texts)

    embedder = build_embedder(config.embed)
    info = embedder.info
    if config.cache.enabled:
        embedder = CachedEmbedder(embedder, config.paths.cache, preprocessing_version=spec.version)

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)

    arrays: dict[str, Any] = {"ids": np.asarray(ids)}
    kind_stats: dict[str, dict[str, Any]] = {}

    def _cache_counters() -> tuple[int, int, int]:
        return (
            getattr(embedder, "cache_hits", 0),
            getattr(embedder, "cache_misses", 0),
            getattr(embedder, "n_stored", 0),
        )

    for kind in kinds:
        started_kind = time.perf_counter()
        before = _cache_counters()
        vectors = embedder.encode(processed, kind)
        after = _cache_counters()
        kind_stats[kind] = {
            "n_vectors": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "l2_rows": float(np.linalg.norm(vectors, axis=1).mean()) if len(vectors) else 1.0,
            "cache_hits": after[0] - before[0],
            "cache_misses": after[1] - before[1],
            "n_stored": after[2] - before[2],
            "seconds": round(time.perf_counter() - started_kind, 3),
        }
        arrays[f"vectors_{kind}"] = vectors
    np.savez_compressed(results_dir / "embeddings.npz", **arrays)

    totals = _cache_counters()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "command": "embed",
        "kinds": list(kinds),
        "n_records": len(ids),
        "n_malformed_lines": report.n_invalid,
        "source_sha256": file_sha256(config.paths.data_raw),
        "embedder": {
            "backend": info.backend,
            "model_id": info.model_id,
            "dimension": info.dimension,
            "revision_or_digest": info.revision_or_digest,
            "context_tokens": info.context_tokens,
            "prefixes": dict(info.prefixes),
            "extra": info.extra,
        },
        "cache": {
            "enabled": config.cache.enabled,
            "path": str(config.paths.cache) if config.cache.enabled else None,
            "cache_hits": totals[0],
            "cache_misses": totals[1],
            "n_stored": totals[2],
        },
        "preprocess": preprocess_meta,
        "kind_stats": kind_stats,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (results_dir / "meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _append_command_rows(
        config,
        "embed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[
            {
                "run_id": run_id,
                "date": date.today().isoformat(),
                "backend": info.backend,
                "model": info.model_id,
                "model_version": (info.revision_or_digest or "")[:12],
                "dim": info.dimension,
                "ctx": info.context_tokens,
                "kinds": ",".join(kinds),
                "preprocess": spec.variant,
                "n_texts": len(ids),
                "seconds": payload["total_seconds"],
                "cache_hits": totals[0],
                "cache_misses": totals[1],
            }
        ],
    )

    print(f"Embedded {len(ids)} record(s) with {info.backend}:{info.model_id} -> {results_dir}")
    print(
        f"  kinds: {list(kinds)}, dimension: {info.dimension}, "
        f"preprocess: {spec.variant} ({preprocess_meta['n_changed']} text(s) changed)"
    )
    print(
        f"  cache: {totals[0]} hit(s), {totals[1]} miss(es), {totals[2]} stored, "
        f"{payload['total_seconds']}s total"
    )
    return 0


def _cmd_labelembed(config: AppConfig) -> int:
    """Run labelembed: embed one text per dataset label (zero-shot artifact, Phase 4).

    Labels are the dataset's sorted unique labels (no catalog file, PLAN.md
    section 7); the per-label texts come from the config's ``label_texts:``
    section (variant ``name`` or a ``{label}`` template). Rendered texts go
    through the same preprocessing spec and disk cache as dataset texts, under
    the model's card-verified ``label`` prefix. Writes
    ``results/<run_id>/label_embeddings.npz`` (``labels`` + ``vectors_label``,
    row order = sorted label order) plus ``meta.json`` with the full lineage;
    consumed through ``eval.label_embedding_run_id``.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.embed is None:
        raise ValueError(
            "the embed: section (backend, model_id, prefixes) is required for "
            "the `labelembed` command"
        )
    if config.label_texts is None:
        raise ValueError(
            "the label_texts: section (variant 'name' or a {label} template) is "
            "required for the `labelembed` command"
        )
    report = load_jsonl(config.paths.data_raw)
    if report.errors:
        logger.warning("%d malformed line(s) dropped before label embedding", report.n_invalid)
    labels = sorted({label for record in report.records for label in record.labels})
    if not labels:
        raise ValueError("the dataset has no labels; nothing to embed")
    catalog = _upr_label_catalog(config) if config.label_texts.variant == "description" else {}
    texts: list[str] = []
    n_fallback = 0
    for label in labels:
        entry = catalog.get(label)
        description = entry.description if entry else None
        if config.label_texts.variant == "description" and not description:
            description = label
            n_fallback += 1
        texts.append(config.label_texts.render_text(label, description))

    spec = build_preprocess_spec(config.preprocess)
    processed = spec.apply(texts)
    preprocess_meta = spec.describe(texts)

    embedder = build_embedder(config.embed)
    info = embedder.info
    if config.cache.enabled:
        embedder = CachedEmbedder(embedder, config.paths.cache, preprocessing_version=spec.version)

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)

    def _cache_counters() -> tuple[int, int, int]:
        return (
            getattr(embedder, "cache_hits", 0),
            getattr(embedder, "cache_misses", 0),
            getattr(embedder, "n_stored", 0),
        )

    vectors = embedder.encode(processed, "label")
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(labels)
        or (info.dimension is not None and vectors.shape[1] != info.dimension)
    ):
        raise ValueError(
            f"label embedding produced shape {vectors.shape}, expected "
            f"({len(labels)}, {info.dimension}); refusing to write the artifact"
        )
    np.savez_compressed(
        results_dir / "label_embeddings.npz",
        labels=np.asarray(labels),
        vectors_label=vectors,
    )

    totals = _cache_counters()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "command": "labelembed",
        "n_labels": len(labels),
        "label_texts": {
            "variant": config.label_texts.variant,
            "template": config.label_texts.template,
            "label_space": (
                "the dataset's sorted unique labels"
                + (
                    ""
                    if config.label_texts.variant != "description"
                    else " (the catalog's descriptions, with the name as the fallback)"
                )
            ),
            "rendered_example": texts[0],
            "rendered_texts": texts,
            "catalog_path": (
                str(config.paths.label_catalog)
                if config.label_texts.variant == "description" and config.paths.label_catalog
                else None
            ),
            "n_fallback_to_name": n_fallback,
        },
        "source_sha256": file_sha256(config.paths.data_raw),
        "n_malformed_lines": report.n_invalid,
        "embedder": {
            "backend": info.backend,
            "model_id": info.model_id,
            "dimension": info.dimension,
            "revision_or_digest": info.revision_or_digest,
            "context_tokens": info.context_tokens,
            "prefixes": dict(info.prefixes),
            "extra": info.extra,
        },
        "cache": {
            "enabled": config.cache.enabled,
            "path": str(config.paths.cache) if config.cache.enabled else None,
            "cache_hits": totals[0],
            "cache_misses": totals[1],
            "n_stored": totals[2],
        },
        "preprocess": preprocess_meta,
        "label_stats": {
            "dimension": int(vectors.shape[1]),
            "l2_rows": float(np.linalg.norm(vectors, axis=1).mean()) if len(vectors) else 1.0,
        },
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (results_dir / "meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _append_command_rows(
        config,
        "labelembed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[
            {
                "run_id": run_id,
                "date": date.today().isoformat(),
                "backend": info.backend,
                "model": info.model_id,
                "model_version": (info.revision_or_digest or "")[:12],
                "dim": info.dimension,
                "ctx": info.context_tokens,
                "kinds": "label",
                "preprocess": spec.variant,
                "n_texts": len(labels),
                "seconds": payload["total_seconds"],
                "cache_hits": totals[0],
                "cache_misses": totals[1],
            }
        ],
    )

    print(
        f"Embedded {len(labels)} label text(s) with {info.backend}:{info.model_id} -> {results_dir}"
    )
    print(
        f"  label_texts: {config.label_texts.variant}"
        + (f" ({config.label_texts.template})" if config.label_texts.template else "")
        + f", dimension: {info.dimension}, preprocess: {spec.variant}"
    )
    print(
        f"  cache: {totals[0]} hit(s), {totals[1]} miss(es), {totals[2]} stored, "
        f"{payload['total_seconds']}s total"
    )
    return 0
