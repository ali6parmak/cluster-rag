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
from typing import Any

import numpy as np

from cluster_rag.config import EmbedderConfig
from cluster_rag.embed.base import (
    EmbedderInfo,
    EncodeKind,
    TokenEmbedding,
    apply_prefix,
    empty_embeddings,
    ensure_embedding_contract,
    prefix_shift,
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
        # model_kwargs flows model-side constructor kwargs to the loaded
        # modules (e.g. jina-embeddings-v5's remote-code module requires
        # {"default_task": "retrieval"}; plain encode() raises without it).
        self._model = SentenceTransformer(
            config.model_id,
            revision=config.revision,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
            model_kwargs=dict(config.model_kwargs),
        )
        # sentence-transformers 6 renamed the method; keep 5.x compatibility
        get_dimension = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        discovered_dimension = int(d) if (d := get_dimension()) else None
        if config.dimension is not None:
            if discovered_dimension is not None and discovered_dimension != config.dimension:
                logger.warning(
                    "The model reports output dimension %d but the config pins dimension %d "
                    "for %r; trusting the config (some remote-code modules do not report "
                    "their true output size) — the encode-time contract check still "
                    "hard-fails on any returned vector that mismatches",
                    discovered_dimension,
                    config.dimension,
                    config.model_id,
                )
            elif discovered_dimension is None:
                logger.warning(
                    "The model does not report an embedding dimension; using the pinned "
                    "embed.dimension=%d for %r (no model-side confirmation)",
                    config.dimension,
                    config.model_id,
                )
            dimension = config.dimension
            dimension_source = (
                "config_pin" if discovered_dimension != config.dimension else "model_reported"
            )
        else:
            dimension = discovered_dimension
            dimension_source = "model_reported"
        if not dimension:
            raise ValueError(
                f"Could not infer the embedding dimension of {config.model_id!r} "
                "(get_sentence_embedding_dimension() returned None); pin embed.dimension "
                "in the config"
            )
        max_seq_length = getattr(self._model, "max_seq_length", None)
        context_tokens = int(max_seq_length) if max_seq_length else None
        if config.context_tokens is not None and context_tokens != config.context_tokens:
            logger.warning(
                "Overriding the discovered context %s of %r with the configured "
                "context_tokens=%d (both values are recorded in run metadata)",
                context_tokens,
                config.model_id,
                config.context_tokens,
            )
            context_tokens = config.context_tokens
        resolved = config.revision or _resolved_revision(config) or None
        import sentence_transformers  # already imported above (deferred heavy import)

        self._info = EmbedderInfo(
            model_id=config.model_id,
            backend="hf",
            dimension=dimension,
            revision_or_digest=resolved,
            prefixes=config.prefixes.model_dump(),
            context_tokens=context_tokens,
            extra={
                "device": str(self._model.device),
                "max_seq_length": max_seq_length,
                "requested_revision": config.revision,
                "resolved_revision": resolved,
                "model_kwargs": dict(config.model_kwargs),
                "dimension_source": dimension_source,
                "discovered_dimension": discovered_dimension,
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

    def _transformer(self) -> Any:
        """The model's transformer module (the token-level path needs it).

        Raises:
            ValueError: If the model does not expose its transformer, so the
                token-level path is not available for it.
        """
        modules = list(self._model)
        for module in modules:
            auto_model = getattr(module, "auto_model", None)
            if auto_model is not None:
                return auto_model
        raise ValueError(
            f"The model {self._info.model_id!r} does not expose its transformer "
            "(no module with an 'auto_model' attribute); the token-level path "
            "(the late chunking) is not available for it"
        )

    def _apply_token_projections(self, hidden: Any) -> Any:
        """Apply the model's per-token projection modules to the token vectors.

        A model whose projection lives in a separate sentence-transformers
        module (e5, bge-m3, ...) embeds its sentence vectors in a space the
        transformer does not produce, so the module has to be applied to the
        token vectors too — otherwise the pooled chunk vectors would live in
        the wrong space. A model whose projection is inside the transformer
        (voyage-4-nano's remote code projects 1024 -> 2048 per token) has no
        such module and nothing is applied.
        """
        for module in list(self._model):
            linear = getattr(module, "linear", None)
            if linear is None or not hasattr(linear, "weight"):
                continue
            projected = module.activation_function(linear(hidden))
            if getattr(module, "use_residual", False):
                residual = (
                    hidden if module.in_features == module.out_features else module.residual(hidden)
                )
                projected = projected + residual
            hidden = projected
        return hidden

    def encode_tokens(self, texts: Sequence[str], kind: EncodeKind) -> list[TokenEmbedding]:
        """The **token-level** embeddings of the texts (the late-chunking path).

        The tokenizer returns every token's character span and the transformer
        returns one vector per token, so a caller can pool the tokens of a
        chunk (Stage 2.7). The spans are mapped back onto the *text*: the
        tokenizer's offsets are offsets into the prefixed text, and the
        prefix's own tokens get an empty span (they carry no text and join no
        chunk).

        The vectors are the model's own embedding space: the transformer's
        output, plus the per-token projection of a model that has one (see
        :meth:`_apply_token_projections`).

        Raises:
            ValueError: If the model exposes no transformer.
        """
        import torch  # deferred heavy import (the backend's own dependency)

        prefix = resolve_prefix(self._info.prefixes, kind)
        inputs = apply_prefix(prefix, validate_texts(texts))
        if not inputs:
            return []
        shift = prefix_shift(prefix)
        encoded = self._model.tokenizer(
            inputs,
            padding=True,
            truncation="longest_first",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        device = self._model.device
        with torch.no_grad():
            output = self._transformer()(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
            hidden = self._apply_token_projections(output.last_hidden_state)
        hidden = hidden.float().cpu().numpy()
        offsets = np.asarray(encoded["offset_mapping"].cpu().numpy())
        embeddings: list[TokenEmbedding] = []
        for row in range(hidden.shape[0]):
            spans: list[tuple[int, int]] = []
            prefix_tokens = 0
            for start, end in offsets[row]:
                # The prefix's own tokens land before the text: an empty span.
                # They are the *leading* run of such tokens — the batch's
                # padding tokens come after the text and are not counted.
                if int(end) <= shift and prefix_tokens == len(spans):
                    prefix_tokens += 1
                shifted_start = max(int(start) - shift, 0)
                shifted_end = max(int(end) - shift, 0)
                spans.append((shifted_start, shifted_end))
            embeddings.append(
                TokenEmbedding(
                    vectors=hidden[row],
                    spans=tuple(spans),
                    prefix_tokens=prefix_tokens,
                )
            )
        return embeddings
