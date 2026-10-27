"""The rerankers' disk cache (a **separate** cache, never the embedding one).

A reranker score is not a vector, so the reranker cache never mixes with the
embedding cache (TEXT_PROCESSING_PLAN.md section 2). It has its own root
under ``cache/`` (``cache/rerank/``) and its own key:

    (model id, revision, the family, the score's shape, the request)

The key covers everything the score depends on:

- the **model id and its pinned revision** (a re-pulled model never reads the
  old build's scores);
- the **family** (the cross-encoder, the yes/no head and the listwise model
  score differently) and the **request shape** (the pair template, the input
  budget, the block size);
- the **request itself**: the query and the ordered candidate documents.

The **document order** is part of the key on purpose: a listwise reranker
reads the whole candidate list in one context, so a document's score depends
on its neighbours' positions and the request must mean exactly one thing.

The track's hygiene: the scores are cached, so a re-run is reproducible
modulo model drift and never re-scores a pair.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cluster_rag.rerank.base import RerankInfo, RerankRequest

logger = logging.getLogger(__name__)

# The cache's own format version (its layout, not the embedding cache's).
CACHE_FORMAT_VERSION = "v1"

# The directory name of the reranker cache under the run's cache root.
RERANK_CACHE_DIRNAME = "rerank"

_SLUG_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _slug(value: str) -> str:
    """Readable path component (``[A-Za-z0-9._-]``); empty collapses to "-"."""
    slug = "".join(character if character in _SLUG_SAFE else "_" for character in value)
    return slug or "-"


def _model_dir(model_id: str) -> str:
    """The model's cache directory: a readable slug plus a uniqueness prefix."""
    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{digest}-{_slug(model_id)}"


def rerank_cache_key(
    *,
    model_id: str,
    revision: str,
    kind: str,
    request: RerankRequest,
    shape: dict[str, Any],
) -> str:
    """The cache key of one request: the model's identity, its shape, the request.

    The key is the canonical JSON of the tuple, hashed — so the same request
    always lands on the same file and a different request never does.
    """
    payload = {
        "model_id": model_id,
        "revision": revision,
        "kind": kind,
        "shape": shape,
        "request": {"query": request.query, "documents": list(request.documents)},
    }
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def rerank_cache_path(
    cache_root: Path | str,
    *,
    model_id: str,
    revision: str,
    kind: str,
    key: str,
    cache_format_version: str = CACHE_FORMAT_VERSION,
) -> Path:
    """The cache file of one request (sharded by the key's prefix)."""
    return (
        Path(cache_root)
        .joinpath(
            RERANK_CACHE_DIRNAME,
            cache_format_version,
            _model_dir(model_id),
            _slug(kind),
            revision[:12],
            key[:2],
        )
        .joinpath(f"{key}.json")
    )


def _store(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one score (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class CachedReranker:
    """Reranker decorator that serves scores from disk and delegates the misses.

    The wrapped reranker's provenance is passed through, so the artifact
    records the real model's id and revision even when the scores come from
    the cache.
    """

    def __init__(
        self,
        reranker: Any,
        cache_root: Path | str,
        *,
        shape: dict[str, Any] | None = None,
        cache_format_version: str = CACHE_FORMAT_VERSION,
    ) -> None:
        """Wrap ``reranker``; the shape is what the scores depend on."""
        self._reranker = reranker
        self._root = Path(cache_root)
        self._shape = dict(shape or {})
        self._cache_format_version = cache_format_version
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_stored = 0

    @property
    def info(self) -> RerankInfo:
        """The wrapped reranker's provenance plus the cache's location."""
        info = self._reranker.info
        extra = {
            **info.extra,
            "cache_root": str(self._root),
            "cache_format_version": self._cache_format_version,
        }
        return RerankInfo(
            model_id=info.model_id,
            revision=info.revision,
            kind=info.kind,
            backend=info.backend,
            extra=extra,
        )

    @property
    def usage(self) -> dict[str, float]:
        """The wrapped reranker's counters (the calls that actually ran)."""
        return dict(getattr(self._reranker, "usage", {}))

    def benchmark(self, requests: Sequence[RerankRequest], *, repeats: int = 2) -> dict[str, float]:
        """Time a forward pass over the given requests (the cache is bypassed).

        The wrapper delegates: the benchmark must measure the **model**, not
        the cache (the run's own wall time is not a cost measurement when the
        cache served part of the work).
        """
        return self._reranker.benchmark(requests, repeats=repeats)

    def score(self, requests: Sequence[RerankRequest]) -> list[list[float]]:
        """Score every request: the cache first, the model on a miss."""
        if not requests:
            return []
        info = self._reranker.info
        results: list[list[float]] = []
        misses: list[tuple[int, RerankRequest, str]] = []
        for index, request in enumerate(requests):
            key = rerank_cache_key(
                model_id=info.model_id,
                revision=info.revision,
                kind=info.kind,
                request=request,
                shape=self._shape,
            )
            path = rerank_cache_path(
                self._root,
                model_id=info.model_id,
                revision=info.revision,
                kind=info.kind,
                key=key,
                cache_format_version=self._cache_format_version,
            )
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                results.append([float(value) for value in payload["scores"]])
                self.cache_hits += 1
                continue
            misses.append((index, request, key))
            results.append([])
        if misses:
            scored = self._reranker.score([request for _, request, _ in misses])
            for (index, _request, key), scores in zip(misses, scored, strict=True):
                results[index] = [float(value) for value in scores]
                path = rerank_cache_path(
                    self._root,
                    model_id=info.model_id,
                    revision=info.revision,
                    kind=info.kind,
                    key=key,
                    cache_format_version=self._cache_format_version,
                )
                _store(
                    path,
                    {
                        "key": {
                            "model_id": info.model_id,
                            "revision": info.revision,
                            "kind": info.kind,
                            "key": key,
                        },
                        "scores": [float(value) for value in scores],
                    },
                )
                self.cache_misses += 1
                self.n_stored += 1
        return results
