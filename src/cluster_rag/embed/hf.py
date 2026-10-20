"""sentence-transformers embedding backend (Phase 2).

Loads a Hugging Face model through sentence-transformers and enforces the
shared embedding contract (float32, ``(n, d)``, L2-normalized so the dot
product equals cosine similarity, AGENTS.md). ``normalize_embeddings=True`` is
always passed; config prefixes/templates are applied per kind before the call.

The heavy imports (``torch``, ``sentence_transformers``) happen inside the
constructor so importing this module is cheap (AGENTS.md dependency rule:
model clients stay inside ``embed/``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

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


def _resolved_revision(config: EmbedderConfig) -> str | None:
    """Commit hash of the snapshot actually on disk, without network access.

    Used only for run provenance when ``embed.revision`` is not pinned; failure
    is a warning, never an error.
    """
    try:
        import huggingface_hub

        snapshot = huggingface_hub.snapshot_download(
            config.model_id, revision=config.revision, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 -- provenance lookup must not break loading
        logger.warning(
            "Could not resolve the local snapshot revision for %s: %s", config.model_id, exc
        )
        return None
    return Path(snapshot).name


class HFEmbedder:
    """Embedder over a ``sentence_transformers.SentenceTransformer`` model."""

    def __init__(self, config: EmbedderConfig) -> None:
        """Load the model and record provenance (device, context, revision).

        Raises:
            ValueError: If the model has no discoverable embedding dimension.
        """
        from sentence_transformers import SentenceTransformer  # deferred heavy import

        self._config = config
        self._model = SentenceTransformer(
            config.model_id,
            revision=config.revision,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
        )
        # sentence-transformers 6 renamed the method; keep 5.x compatibility
        get_dimension = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        dimension = get_dimension()
        if not dimension:
            raise ValueError(
                f"Could not infer the embedding dimension of {config.model_id!r} "
                "(get_sentence_embedding_dimension() returned None)"
            )
        max_seq_length = getattr(self._model, "max_seq_length", None)
        resolved = config.revision or _resolved_revision(config) or None
        import sentence_transformers  # already imported above (deferred heavy import)

        self._info = EmbedderInfo(
            model_id=config.model_id,
            backend="hf",
            dimension=dimension,
            revision_or_digest=resolved,
            prefixes=config.prefixes.model_dump(),
            context_tokens=int(max_seq_length) if max_seq_length else None,
            extra={
                "device": str(self._model.device),
                "max_seq_length": max_seq_length,
                "requested_revision": config.revision,
                "resolved_revision": resolved,
                "sentence_transformers_version": sentence_transformers.__version__,
                "loader_summary": (
                    f"device: {self._model.device}; max_seq_length: {max_seq_length}; "
                    f"resolved_revision: {resolved}"
                ),
            },
        )
        logger.info(
            "HF embedder ready: %s (revision %s, dimension %d, max_seq_length %s, device %s)",
            config.model_id,
            resolved,
            dimension,
            max_seq_length,
            self._model.device,
        )

    @property
    def info(self) -> EmbedderInfo:
        """Provenance facts for run metadata (revision, device, tokenizer budget)."""
        return self._info

    def encode(self, texts: Sequence[str], kind: EncodeKind) -> np.ndarray:
        """Embed texts (config prefix/template per kind applied); contract-enforced."""
        prefix = resolve_prefix(self._info.prefixes, kind)
        inputs = apply_prefix(prefix, validate_texts(texts))
        if not inputs:
            return empty_embeddings(self._info.dimension)
        vectors = self._model.encode(
            inputs,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # AGENTS.md: always normalize (dot == cosine)
            show_progress_bar=False,
        )
        return ensure_embedding_contract(np.asarray(vectors), dimension=self._info.dimension)

    def token_counts(self, texts: Sequence[str]) -> np.ndarray:
        """Exact subword token counts including special tokens (truncation diagnostics).

        Ollama has no client-side tokenizer, so only HF embedders can report
        these exactly; the omct probe uses them to measure how many inputs
        exceed the model's ``max_seq_length`` and get truncated server-side.
        """
        encoded = self._model.tokenizer(list(texts))
        return np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=np.int64)
