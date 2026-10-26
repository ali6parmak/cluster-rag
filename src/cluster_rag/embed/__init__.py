"""Embedding backends for Phase 2 (PLAN.md).

This package hosts the :class:`~cluster_rag.embed.base.Embedder` protocol and
the concrete backends, plus the disk cache (``embed/cache.py``).

Dependency rule (AGENTS.md): only modules under ``embed/`` touch model clients
(``torch``/``sentence-transformers``/``httpx``). The heavy imports are deferred
into the backend constructors, so importing this package stays cheap and the
rest of the pipeline keeps working without them.
"""

from __future__ import annotations

from cluster_rag.config import EmbedderConfig
from cluster_rag.embed.base import (
    CONTENT_PLACEHOLDER,
    KINDS,
    Embedder,
    EmbedderInfo,
    EncodeKind,
    apply_prefix,
    empty_embeddings,
    ensure_embedding_contract,
    l2_normalize,
    resolve_prefix,
    validate_texts,
)
from cluster_rag.embed.hf import HFEmbedder
from cluster_rag.embed.ollama import OllamaEmbedder, normalize_ollama_host
from cluster_rag.embed.random import RandomEmbedder

__all__ = [
    "CONTENT_PLACEHOLDER",
    "KINDS",
    "EncodeKind",
    "Embedder",
    "EmbedderInfo",
    "HFEmbedder",
    "OllamaEmbedder",
    "RandomEmbedder",
    "apply_prefix",
    "build_embedder",
    "empty_embeddings",
    "ensure_embedding_contract",
    "l2_normalize",
    "normalize_ollama_host",
    "resolve_prefix",
    "validate_texts",
]


def build_embedder(config: EmbedderConfig) -> Embedder:
    """Create the embedder named by the resolved config (cli/omct_probe entry point).

    The concrete imports are deferred into the branches: building one backend
    keeps the other backend's heavy dependencies (torch vs httpx) unloaded.

    Raises:
        ValueError: Unknown backend or a non-``EmbedderConfig`` argument.
    """
    if not isinstance(config, EmbedderConfig):
        raise ValueError(f"build_embedder expects an EmbedderConfig, got {type(config).__name__}")
    if config.backend == "hf":
        from cluster_rag.embed.hf import HFEmbedder as _HF

        return _HF(config)
    if config.backend == "ollama":
        from cluster_rag.embed.ollama import OllamaEmbedder as _Ollama

        return _Ollama(config)
    raise ValueError(
        f"Unknown embed backend {config.backend!r} (expected 'hf' or 'ollama') "
        f"for model {config.model_id!r}"
    )
