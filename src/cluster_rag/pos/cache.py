"""The part-of-speech tagger's **disk cache** (a separate cache, never the
embedding one).

A tag is not a vector, so the tagger's cache never mixes with the embedding
cache (TEXT_PROCESSING_PLAN.md section 2). It has its own root under
``cache/`` (``cache/pos/``) and its own key:

    (model id, digest, the text's sha1)

The digest is part of the key, so a re-downloaded model never reads the old
build's tags, and the text's sha1 is the text's own hash, so a re-run never
re-tags a text it has already tagged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from cluster_rag.pos.base import TaggerInfo

logger = logging.getLogger(__name__)

# The cache's own format version (its layout, not the embedding cache's).
CACHE_FORMAT_VERSION = "v1"

# The directory name of the tagger cache under the run's cache root.
POS_CACHE_DIRNAME = "pos"

_SLUG_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _slug(value: str) -> str:
    """Readable path component (``[A-Za-z0-9._-]``); empty collapses to "-"."""
    slug = "".join(character if character in _SLUG_SAFE else "_" for character in value)
    return slug or "-"


def _model_dir(model_id: str) -> str:
    """The model's cache directory: a readable slug plus a uniqueness prefix."""
    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{digest}-{_slug(model_id)}"


def tagger_cache_key(*, model_id: str, digest: str, text: str) -> str:
    """The cache key of one text's tags: the model's identity and the text."""
    payload = {
        "model_id": model_id,
        "digest": digest,
        "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
    }
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def tagger_cache_path(
    cache_root: Path | str,
    *,
    model_id: str,
    key: str,
    cache_format_version: str = CACHE_FORMAT_VERSION,
) -> Path:
    """The cache file of one text's tags."""
    return (
        Path(cache_root)
        .joinpath(
            POS_CACHE_DIRNAME,
            cache_format_version,
            _model_dir(model_id),
            key,
        )
        .with_suffix(".json")
    )


def _store(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one text's tags (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class CachedTagger:
    """Tagger decorator that serves tags from disk and delegates the misses.

    The wrapped tagger's provenance is passed through, so the artifact records
    the real model's id and digest even when the tags come from the cache.
    """

    def __init__(
        self,
        tagger: Any,
        cache_root: Path | str,
        *,
        cache_format_version: str = CACHE_FORMAT_VERSION,
    ) -> None:
        """Wrap ``tagger``; the cache root is the run's own."""
        self._tagger = tagger
        self._root = Path(cache_root)
        self._cache_format_version = cache_format_version
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_stored = 0

    @property
    def info(self) -> TaggerInfo:
        """The wrapped tagger's provenance plus the cache's location."""
        info = self._tagger.info
        extra = {
            **info.extra,
            "cache_root": str(self._root),
            "cache_format_version": self._cache_format_version,
        }
        return TaggerInfo(
            model_id=info.model_id,
            digest=info.digest,
            backend=info.backend,
            extra=extra,
        )

    def tag(self, text: str) -> list[tuple[str, str]]:
        """The text's tokens with their tags: the cache first, the tagger on a miss."""
        info = self.info
        key = tagger_cache_key(model_id=info.model_id, digest=info.digest, text=text)
        path = tagger_cache_path(
            self._root,
            model_id=info.model_id,
            key=key,
            cache_format_version=self._cache_format_version,
        )
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return [(str(token), str(tag)) for token, tag in payload["tags"]]
        self.cache_misses += 1
        tags = self._tagger.tag(text)
        _store(
            path,
            {
                "key": {
                    "model_id": info.model_id,
                    "digest": info.digest,
                    "key": key,
                },
                "tags": [[token, tag] for token, tag in tags],
            },
        )
        self.n_stored += 1
        return list(tags)
