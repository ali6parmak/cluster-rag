"""Ollama embedding backend: batched ``POST /api/embed`` (Phase 2).

Errors are translated into actionable messages exactly as AGENTS.md demands:
connection problems ask whether ``ollama serve`` is running, 404s ask for
``ollama pull <model>``. At construction the backend records the model's
Ollama digest(es) and context length via ``/api/tags`` / ``/api/show``, so
every run's meta.json pins the exact server-side model.

The embedding contract (float32, ``(n, d)``, L2-normalized) is enforced by
``ensure_embedding_contract``; the model-specific prefixes/templates come from
the config and are applied per kind before the request.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from cluster_rag.config import EmbedderConfig
from cluster_rag.embed.base import (
    EmbedderInfo,
    EncodeKind,
    apply_prefix,
    empty_embeddings,
    ensure_embedding_contract,
    resolve_prefix,
    validate_texts,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def normalize_ollama_host(host: str) -> str:
    """Accept hosts with or without a scheme (``0.0.0.0:11434`` -> ``http://...``)."""
    host = host.strip()
    if not host:
        return DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def _resolve_host(config: EmbedderConfig) -> str:
    """Host precedence: explicitly-set config value, then env ``OLLAMA_HOST``, then default.

    (Both config and env may express the same default in ``EmbedderConfig``.)
    """
    if "host" in config.model_fields_set:
        return normalize_ollama_host(config.host)
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        return normalize_ollama_host(env_host)
    return normalize_ollama_host(config.host)


class OllamaEmbedder:
    """Embedder over an Ollama server's ``/api/embed`` endpoint, sent in batches."""

    def __init__(self, config: EmbedderConfig, *, client: httpx.Client | None = None) -> None:
        """Connect and record server-side model info.

        Args:
            config: Validated :class:`~cluster_rag.config.EmbedderConfig`.
            client: Optional pre-built ``httpx.Client`` (tests inject a
                ``MockTransport``-backed client here instead of a real server).

        Raises:
            RuntimeError: The server is unreachable, or the model id is not
                installed on it (message points at ``ollama serve`` /
                ``ollama pull``).
        """
        self._config = config
        self._model_id = config.model_id
        self._host = _resolve_host(config)
        self._client = client or httpx.Client(base_url=self._host, timeout=config.timeout)
        digests, parameter_size = self._fetch_digests()
        api_dimension, api_context_tokens = self._fetch_model_info()
        if config.dimension is not None:
            if api_dimension is not None and api_dimension != config.dimension:
                logger.warning(
                    "/api/show reports embedding_length %d for %r but the config pins "
                    "dimension %d; trusting the config (some Ollama packages report the "
                    "pre-projection size) — the encode-time contract check still hard-fails "
                    "on any returned vector that mismatches",
                    api_dimension,
                    self._model_id,
                    config.dimension,
                )
                dimension, dimension_source = config.dimension, "config_override"
            elif api_dimension is None:
                logger.warning(
                    "/api/show did not expose an embedding length for %r; using the config "
                    "fallback dimension %d (no server-side confirmation)",
                    self._model_id,
                    config.dimension,
                )
                dimension, dimension_source = config.dimension, "config_fallback"
            else:
                dimension, dimension_source = api_dimension, "ollama_api"
        else:
            dimension, dimension_source = api_dimension, "ollama_api"
        context_tokens = config.context_tokens or api_context_tokens
        if dimension is None:
            raise RuntimeError(
                f"Could not discover the embedding dimension of {self._model_id!r}: neither "
                "`/api/show` model_info nor the `embed.dimension` config fallback provided it"
            )
        self._info = EmbedderInfo(
            model_id=self._model_id,
            backend="ollama",
            dimension=dimension,
            revision_or_digest=digests[0] if digests else None,
            prefixes=config.prefixes.model_dump(),
            context_tokens=context_tokens,
            extra={
                "host": self._host,
                "digests": digests,
                "parameter_size": parameter_size,
                "host_from_env": "OLLAMA_HOST" in os.environ
                and "host" not in config.model_fields_set,
                "dimension_source": dimension_source,
                "api_embedding_length": api_dimension,
                "discovered_context_tokens": api_context_tokens,
                "configured_context_tokens": config.context_tokens,
            },
        )
        logger.info(
            "Ollama embedder ready: %s (dimension %d, context %s, digest %s)",
            self._model_id,
            dimension,
            context_tokens,
            digests,
        )

    @property
    def info(self) -> EmbedderInfo:
        """Provenance facts for run metadata (digest, context, server details)."""
        return self._info

    # -- request plumbing ------------------------------------------------

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        """Send JSON and return the parsed body, with actionable error translation.

        Raises:
            RuntimeError: Unreachable server (``ollama serve`` hint), unknown
                model (``ollama pull`` hint), or any other HTTP/decode error.
        """
        try:
            response = self._client.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._host} — is `ollama serve` running? "
                f"(also check `ollama pull {self._model_id}`) Original error: {exc}"
            ) from exc
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{self._model_id}' was not found at {self._host} — "
                f"run `ollama pull {self._model_id}` (installed models: `ollama list`)"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama {path} returned HTTP {response.status_code} for model "
                f"'{self._model_id}': {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Ollama {path} returned a non-JSON response: {response.text[:300]}"
            ) from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST wrapper around :meth:`_request`."""
        return self._request("POST", path, payload)

    def _fetch_digests(self) -> tuple[list[str], str | None]:
        """Model digests + parameter size from ``/api/tags`` (multiple entries possible)."""
        data = self._request("GET", "/api/tags", None)
        matching: list[dict[str, Any]] = []
        for entry in data.get("models", []):
            name = entry.get("model") or entry.get("name") or ""
            if self._matches_tag(name):
                matching.append(entry)
        if not matching:
            raise RuntimeError(
                f"Ollama model '{self._model_id}' is not installed at {self._host} — "
                f"run `ollama pull {self._model_id}` (installed models: `ollama list`)"
            )
        digests = sorted({entry["digest"] for entry in matching if entry.get("digest")})
        parameter_size = matching[0].get("details", {}).get("parameter_size")
        return digests, parameter_size

    def _matches_tag(self, tag: str) -> bool:
        """True when the /api/tags entry refers to the configured model id.

        A model id without an explicit tag refers to the server's default
        (":latest"), so bare names resolve against the tagged entries too.
        """
        if tag == self._model_id:
            return True
        if ":" in self._model_id:
            return False  # explicit tag configured: only an exact match counts
        return tag.rsplit(":", 1)[0] == self._model_id

    def _fetch_model_info(self) -> tuple[int | None, int | None]:
        """(embedding dimension, context length) from ``/api/show`` model_info."""
        data = self._post("/api/show", {"model": self._model_id})
        model_info = data.get("model_info", {})
        dimension_values = {
            value for key, value in model_info.items() if key.endswith("embedding_length")
        }
        context_values = {
            value for key, value in model_info.items() if key.endswith("context_length")
        }
        if len(dimension_values) > 1:
            raise RuntimeError(
                f"/api/show reports several embedding lengths for {self._model_id!r} "
                f"({sorted(dimension_values)}); set embed.dimension in the config explicitly"
            )
        dimension = int(next(iter(dimension_values))) if len(dimension_values) == 1 else None
        context = int(next(iter(context_values))) if len(context_values) == 1 else None
        return dimension, context

    # -- protocol --------------------------------------------------------

    def encode(self, texts: Sequence[str], kind: EncodeKind) -> np.ndarray:
        """Embed texts in batches of ``embed.batch_size``; contract-enforced output.

        A batch is a single ``POST /api/embed`` with ``{"model", "input"}``;
        the number of returned embeddings must match the request.
        """
        prefix = resolve_prefix(self._info.prefixes, kind)
        inputs = apply_prefix(prefix, validate_texts(texts))
        if not inputs:
            return empty_embeddings(self._info.dimension)
        out = np.zeros((len(inputs), self._info.dimension), dtype=np.float32)
        row = 0
        for start in range(0, len(inputs), self._config.batch_size):
            batch = inputs[start : start + self._config.batch_size]
            data = self._post("/api/embed", {"model": self._model_id, "input": batch})
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                count = len(embeddings) if isinstance(embeddings, list) else "missing"
                raise RuntimeError(
                    f"Ollama /api/embed returned {count} embeddings for a batch of "
                    f"{len(batch)} inputs (model {self._model_id!r}, host {self._host})"
                )
            vectors = np.asarray(embeddings, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != self._info.dimension:
                width = vectors.shape[1] if vectors.ndim == 2 else "non-matrix"
                raise RuntimeError(
                    f"Ollama /api/embed returned {width}-dim embeddings, but the "
                    f"model/config dimension is {self._info.dimension} "
                    f"(model {self._model_id!r}; check the embed.dimension config)"
                )
            out[row : row + len(batch)] = vectors
            row += len(batch)
        matrix = out[:row]
        if matrix.size and not np.isfinite(matrix).all():
            raise RuntimeError(f"Ollama returned non-finite embeddings for {self._model_id!r}")
        return ensure_embedding_contract(matrix, dimension=self._info.dimension)
