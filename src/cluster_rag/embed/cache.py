"""Content-addressed disk cache for embeddings (Phase 2, PLAN.md).

Key, exactly as planned: ``(model_id, prefix, preprocessing_version,
sha1(text))`` — implemented on disk as

    <cache>/v1/<model-slug>/<preprocessing_version>/<prefix-slug>/<sha1(text)>.npy

Path components are readable slugs: characters outside ``[A-Za-z0-9._-]``
become ``_`` (an empty value collapses to "-"); the model directory carries a
short ``sha1(model_id)`` prefix so ids that slugify identically stay distinct
("BAAI/bge-m3" → ``<sha8>-BAAI_bge-m3``, "a/b" vs "a_b"). One file per vector:
writes are atomic (tmp + rename), so a crashed run never corrupts an entry,
and hits never reach the model at all (proven by tests). Only ``embed/``
touches the disk, per AGENTS.md.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from cluster_rag.embed.base import (
    Embedder,
    EmbedderInfo,
    EncodeKind,
    TokenEmbedding,
    empty_embeddings,
    ensure_embedding_contract,
    resolve_prefix,
)

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = "v1"

# Slug alphabet for path components: strictly ASCII so unicode is not leaked
# in by isalnum(); everything else becomes "_".
_SLUG_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _slug(value: str) -> str:
    """Readable path component (``[A-Za-z0-9._-]``); empty collapses to "-"."""
    slug = "".join(char if char in _SLUG_SAFE else "_" for char in value)
    return slug or "-"


def _model_dir(model_id: str) -> str:
    """Model cache directory: readable slug plus a uniqueness prefix.

    The short ``sha1(model_id)`` prefix keeps distinct model ids apart when
    their slugs would collide ("a/b" and "a_b" both slugify to "a_b").
    """
    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{digest}-{_slug(model_id)}"


def cache_key_path(
    cache_root: Path | str,
    *,
    model_id: str,
    prefix: str,
    preprocessing_version: str,
    text: str,
    cache_format_version: str = CACHE_FORMAT_VERSION,
) -> Path:
    """The cache file path for one (model, prefix, prep-version, text) tuple."""
    text_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return (
        Path(cache_root)
        .joinpath(
            cache_format_version,
            _model_dir(model_id),
            _slug(preprocessing_version),
            _slug(prefix),
            text_sha1,
        )
        .with_suffix(".npy")
    )


def _load_vector(path: Path, dimension: int) -> np.ndarray:
    """Load one cached vector and enforce the stored shape.

    Raises:
        RuntimeError: On a corrupt/stale cache entry (clear message instead of
            a silently misaligned vector matrix).
    """
    try:
        vector = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    except Exception as exc:  # noqa: BLE001 -- any decode failure means "stale/corrupt"
        raise RuntimeError(
            f"Unreadable cache file {path} — delete the cache directory: {exc}"
        ) from exc
    if vector.shape != (dimension,):
        raise RuntimeError(
            f"Cache file {path} has shape {vector.shape}, expected ({dimension},) — "
            "the cache may predate the model; delete the cache directory to re-embed"
        )
    return vector


def _store_vector(path: Path, vector: np.ndarray) -> None:
    """Atomically write one vector (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with tmp.open("wb") as handle:  # file handle: np.save must not rewrite the name
        np.save(handle, np.asarray(vector, dtype=np.float32))
    os.replace(tmp, path)


class CachedEmbedder:
    """Embedder decorator that serves vectors from disk and delegates misses.

    Wraps any :class:`~cluster_rag.embed.base.Embedder`; the cache key uses the
    backend's configured per-kind prefix (``backend.info.prefixes``), the
    caller-supplied preprocessing version and the text hash. Hit counts and
    miss counts are kept for run metadata (``meta.json``), so "cache hits skip
    re-encoding" is verifiable per run.
    """

    def __init__(
        self,
        backend: Embedder,
        cache_root: Path | str,
        *,
        preprocessing_version: str,
        cache_format_version: str = CACHE_FORMAT_VERSION,
    ) -> None:
        """Wrap ``backend``; version strings come from preprocess.py."""
        self._backend = backend
        self._root = Path(cache_root)
        self._preprocessing_version = preprocessing_version
        self._cache_format_version = cache_format_version
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_stored = 0

    @property
    def info(self) -> EmbedderInfo:
        """Provenance of the wrapped backend plus the cache location/version."""
        extra = {
            **self._backend.info.extra,
            "cache_root": str(self._root),
            "cache_format_version": self._cache_format_version,
            "preprocessing_version": self._preprocessing_version,
        }
        return replace(self._backend.info, extra=extra)

    def encode(self, texts: Sequence[str], kind: EncodeKind) -> np.ndarray:
        """Embed texts, cache-first: hits load from disk, misses delegate+store.

        Texts repeated inside one batch are encoded and stored once.
        """
        texts = list(texts)
        if not texts:
            return empty_embeddings(self._backend.info.dimension)
        prefix = resolve_prefix(self._backend.info.prefixes, kind)

        # group occurrences by identical text (-> identical key)
        unique_paths: list[Path] = []
        group_of_path: dict[Path, int] = {}
        group_of_index: list[int] = []
        members: list[list[int]] = []
        for index, text in enumerate(texts):
            path = cache_key_path(
                self._root,
                model_id=self._backend.info.model_id,
                prefix=prefix,
                preprocessing_version=self._preprocessing_version,
                text=text,
                cache_format_version=self._cache_format_version,
            )
            group = group_of_path.get(path)
            if group is None:
                group = len(unique_paths)
                unique_paths.append(path)
                group_of_path[path] = group
                members.append([])
            members[group].append(index)
            group_of_index.append(group)

        group_vectors: dict[int, np.ndarray] = {}
        missing_groups: list[int] = []
        for group, path in enumerate(unique_paths):
            if path.is_file():
                group_vectors[group] = _load_vector(path, self._backend.info.dimension)
                self.cache_hits += len(members[group])
            else:
                missing_groups.append(group)
                self.cache_misses += len(members[group])

        if missing_groups:
            fresh = self._backend.encode(
                [texts[members[group][0]] for group in missing_groups], kind
            )
            for slot, group in enumerate(missing_groups):
                vector = fresh[slot]
                group_vectors[group] = vector
                _store_vector(unique_paths[group], vector)
                self.n_stored += 1

        if self.cache_hits and not missing_groups and logger.isEnabledFor(logging.DEBUG):
            logger.debug("EmbeddingCache: all %d vectors from cache", len(texts))
        return ensure_embedding_contract(
            np.stack([group_vectors[group] for group in group_of_index]),
            dimension=self._backend.info.dimension,
        )

    @property
    def supports_tokens(self) -> bool:
        """Whether the wrapped backend can return token-level embeddings.

        The capability is the backend's, not the cache's: a cached Ollama
        backend still cannot return token-level vectors, and the late-chunking
        arm must say so instead of failing inside the run.
        """
        return bool(
            getattr(self._backend, "supports_tokens", hasattr(self._backend, "encode_tokens"))
        )

    def encode_tokens(self, texts: Sequence[str], kind: EncodeKind) -> list[TokenEmbedding]:
        """The backend's token-level embeddings — **not** cached.

        The token-level vectors are deliberately left out of the cache: the
        plan's section 2 fixes the cache-key layout
        ``(model, prefix, version, sha1(text))``, and a late-chunked vector is
        a function of the **whole document**, not of one chunk — caching it
        under the chunk's key would serve a wrong vector. The pass is one
        forward pass per text, so it is re-run on every run that needs it.
        """
        if not self.supports_tokens:
            raise ValueError(
                "The wrapped backend cannot return token-level embeddings; the "
                "token-level path (the late chunking) is not available for it"
            )
        return self._backend.encode_tokens(list(texts), kind)  # type: ignore[attr-defined]
