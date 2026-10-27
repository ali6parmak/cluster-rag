"""The Hugging Face reranker client (the track's CS-15, PLAN's Phase 6).

The client loads a reranker and scores requests. It covers the three
families the port knows about, and each one is scored the way **its own model
card** scores it:

- ``cross_encoder`` (``jina-reranker-v2-base-multilingual``): the pair is
  tokenized as a pair and the score is the **sigmoid** of the classification
  logit (the card's ``compute_score``);
- ``llm_yesno`` (``nvidia/llama-nemotron-rerank-1b-v2``): the pair is
  rendered into a **single sequence** by the config's template (the card's
  ``"question:... passage:..."`` format) and the score is the **raw**
  relevance logit (the card's own output);
- ``listwise`` (``jinaai/jina-reranker-v3``): the whole candidate list is
  read in **one context** through the card's ``rerank`` method and the score
  is the card's ``relevance_score``.

**The hygiene**: the model is pinned by id **and revision** (the tokenizer
too — the remote code would otherwise load an unpinned tokenizer from the
hub), the scores are cached by the decorator in
:mod:`cluster_rag.rerank.cache` (a reranker score is not a vector, so it
never enters the embedding cache), and a **shortened input is never silent**:
the client counts the inputs it had to truncate and reports it.

**The transformers-5 compatibility shim.** The jina-reranker-v2 checkpoint
ships custom modelling code that imports
``create_position_ids_from_input_ids`` from
``transformers.models.xlm_roberta.modeling_xlm_roberta`` — a helper
transformers 5 removed. The shim below installs the transformers-4.x
implementation of that **one** helper before the dynamic module is loaded.
A transformers *downgrade* is deliberately **not** the fix: the repo's
embedding artifacts were produced with transformers 5, so a downgrade would
change the embeddings and break the UPR guard
(``tools/verify_upr_baseline.py``).

The heavy imports (``torch``, ``transformers``) happen inside the
constructor, so importing this module is cheap (AGENTS.md dependency rule:
model clients stay inside the client packages).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from cluster_rag.config import RerankerModelConfig
from cluster_rag.rerank.base import (
    KIND_CROSS_ENCODER,
    KIND_LISTWISE,
    KIND_LLM_YESNO,
    RerankInfo,
    RerankRequest,
    render_template,
)

logger = logging.getLogger(__name__)

# The score transform of each family (recorded in the artifact).
SCORE_TRANSFORMS: dict[str, str] = {
    KIND_CROSS_ENCODER: "sigmoid(logit)",
    KIND_LLM_YESNO: "raw_logit",
    KIND_LISTWISE: "card_relevance_score",
}


def ensure_transformers5_compat() -> bool:
    """Install the one transformers-5 compatibility shim the rerankers need.

    The jina-reranker-v2 checkpoint's custom modelling code imports
    ``create_position_ids_from_input_ids`` from
    ``transformers.models.xlm_roberta.modeling_xlm_roberta``; transformers 5
    removed that helper. This installs the transformers-4.x implementation of
    it (the same "position numbers begin at padding_idx + 1" rule) when it is
    missing.

    Returns:
        Whether the shim was installed by this call (``False`` when the
        helper is already there, i.e. on transformers 4.x).

    Raises:
        ImportError: If transformers itself is not installed.
    """
    import torch
    from transformers.models.xlm_roberta import modeling_xlm_roberta

    if hasattr(modeling_xlm_roberta, "create_position_ids_from_input_ids"):
        return False

    def create_position_ids_from_input_ids(
        input_ids: Any, padding_idx: int, past_key_values_length: int = 0
    ) -> Any:
        """The transformers-4.x helper: the position numbers, padding excluded."""
        mask = input_ids.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    modeling_xlm_roberta.create_position_ids_from_input_ids = create_position_ids_from_input_ids
    logger.info(
        "Installed the transformers-5 compatibility shim for "
        "create_position_ids_from_input_ids (the jina-reranker-v2 custom code needs it)"
    )
    return True


def _resolve_dtype(torch: Any, name: str) -> Any:
    """The torch dtype of a config value (``auto`` keeps the checkpoint's own)."""
    if name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(
            f"unknown reranker dtype {name!r}; expected one of {sorted(mapping) + ['auto']}"
        )
    return mapping[name]


class HFReranker:
    """A reranker over Hugging Face models, scored the way each card scores."""

    def __init__(self, config: RerankerModelConfig) -> None:
        """Load the model (and its pinned tokenizer) and record the provenance.

        Raises:
            ValueError: If the model's family is unknown or the pad token
                cannot be resolved (a model without one cannot be batched).
        """
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        ensure_transformers5_compat()

        self._config = config
        self._torch = torch
        dtype = _resolve_dtype(torch, config.dtype)
        loader_kwargs: dict[str, Any] = {
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "dtype": dtype,
            **config.model_kwargs,
        }
        if config.kind == KIND_LISTWISE:
            self._model = AutoModel.from_pretrained(config.model_id, **loader_kwargs)
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                config.model_id, **loader_kwargs
            )
        if config.device and config.device != "auto":
            self._model = self._model.to(config.device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            padding_side=config.padding_side,
        )
        self._pad_token_id = self._resolve_pad_token()
        if config.kind == KIND_LISTWISE:
            # The remote code's `rerank` loads its own tokenizer from the hub
            # when it has none; the pinned one is handed over instead, so the
            # tokenizer's revision is pinned like the model's.
            self._model._tokenizer = self._tokenizer

        self._usage: dict[str, float] = {
            "n_requests": 0,
            "n_pairs": 0,
            "n_calls": 0,
            "seconds": 0.0,
            "peak_gpu_mb": 0.0,
            "n_truncated_inputs": 0,
        }
        self._info = RerankInfo(
            model_id=config.model_id,
            revision=config.revision,
            kind=config.kind,
            backend="huggingface",
            extra={
                "name": config.name,
                "score_transform": SCORE_TRANSFORMS[config.kind],
                "max_length": config.max_length,
                "batch_size": config.batch_size,
                "max_documents": config.max_documents,
                "max_query_length": config.max_query_length,
                "max_doc_length": config.max_doc_length,
                "padding_side": config.padding_side,
                "device": config.device,
                "dtype": config.dtype,
                "template": config.template,
                "model_kwargs": dict(config.model_kwargs),
                "torch_version": torch.__version__,
            },
        )
        logger.info(
            "Reranker ready: %s (%s, revision %s, device %s)",
            config.model_id,
            config.kind,
            config.revision,
            config.device,
        )

    def _resolve_pad_token(self) -> int:
        """The pad token's id (set once, on the tokenizer and on the config).

        Raises:
            ValueError: If the tokenizer has no pad token and no eos/unk
                token to fall back on (the model cannot be batched).
        """
        tokenizer = self._tokenizer
        if tokenizer.pad_token is None:
            for fallback in ("eos_token", "unk_token"):
                token = getattr(tokenizer, fallback, None)
                if token:
                    tokenizer.pad_token = token
                    break
        if tokenizer.pad_token is None:
            raise ValueError(
                f"the tokenizer of {self._config.model_id!r} has no pad token and no "
                "eos/unk token to fall back on: the model cannot be batched"
            )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        config = getattr(self._model, "config", None)
        if config is not None and getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = pad_token_id
        return int(pad_token_id)

    @property
    def info(self) -> RerankInfo:
        """The provenance of the model (its id, its pinned revision, its family)."""
        return self._info

    @property
    def usage(self) -> dict[str, float]:
        """The run-time counters (the requests, the pairs, the calls, the time)."""
        return dict(self._usage)

    def score(self, requests: Sequence[RerankRequest]) -> list[list[float]]:
        """Score every request's documents; one list of scores per request.

        Raises:
            ValueError: If a request carries no document.
        """
        if not requests:
            return []
        for request in requests:
            if not request.documents:
                raise ValueError("a rerank request must carry at least one document")
        if self._config.kind == KIND_LISTWISE:
            scores = self._score_listwise(requests)
        else:
            scores = self._score_pairs(requests)
        self._usage["n_requests"] += len(requests)
        self._usage["n_pairs"] += sum(len(request.documents) for request in requests)
        return scores

    def benchmark(self, requests: Sequence[RerankRequest], *, repeats: int = 2) -> dict[str, float]:
        """Time a forward pass over the given requests (the cache is bypassed).

        The cost of a reranker is part of the deliverable (PLAN's Phase 6:
        "quality gain vs latency and cost"), and the run's own wall time is
        not a cost measurement when the cache served part of the work. This
        method measures the **throughput** directly: the first pass is a
        warm-up (the CUDA allocator and the kernels), the last pass is the
        timed one.

        Args:
            requests: The requests to time.
            repeats: How many passes to run (the first is the warm-up).

        Returns:
            The block: the requests, the pairs, the seconds, the pairs per
            second and the peak GPU memory.
        """
        if not requests:
            return {"n_requests": 0, "n_pairs": 0, "seconds": 0.0, "pairs_per_second": 0.0}
        seconds = 0.0
        for index in range(max(1, repeats)):
            started = time.perf_counter()
            self.score(requests)
            elapsed = time.perf_counter() - started
            if index == max(1, repeats) - 1:
                seconds = elapsed
        n_pairs = sum(len(request.documents) for request in requests)
        self._record_peak_memory()
        return {
            "n_requests": float(len(requests)),
            "n_pairs": float(n_pairs),
            "seconds": float(seconds),
            "pairs_per_second": (n_pairs / seconds) if seconds > 0 else 0.0,
            "peak_gpu_mb": float(self._usage["peak_gpu_mb"]),
        }

    # --- the pair-based families (one forward pass per pair) ---

    def _score_pairs(self, requests: Sequence[RerankRequest]) -> list[list[float]]:
        """The pair-based families: one forward pass per (query, document).

        The pairs are batched **by length** (a stable sort), so a batch holds
        inputs of similar size: the padding inside a batch stays small, and
        with it the numerical wobble the padding causes. The wobble is real
        and measured (the smoke test's batch-size check): a batch's padding
        moves a score by up to 0.007 (the 278M cross-encoder, on a [0, 1]
        scale) and by up to 0.3 logits (the 1B yes/no head), which is enough
        to flip a **near-tie**. Sorting by length makes the batches nearly
        homogeneous, and the run's batch size is pinned in the config, so a
        re-run reproduces the same scores (the cache makes that exact).
        """
        torch = self._torch
        flat: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for request in requests:
            start = len(flat)
            flat.extend((request.query, document) for document in request.documents)
            spans.append((start, len(flat)))
        first = [pair[0] for pair in flat]
        second = [pair[1] for pair in flat]
        if self._config.kind == KIND_LLM_YESNO:
            template = self._config.template or ""
            rendered = [render_template(template, a, b) for a, b in flat]
            first, second = rendered, None  # type: ignore[assignment]

        lengths = self._measure_lengths(first, second)
        truncated = sum(1 for length in lengths if length > self._config.max_length)

        started = time.perf_counter()
        values: list[float] = [0.0] * len(flat)
        batch_size = self._config.batch_size
        order = sorted(range(len(flat)), key=lambda index: lengths[index])
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            encoded = self._encode(
                [first[index] for index in chunk],
                None if second is None else [second[index] for index in chunk],
            )
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            chunk_values = self._read_logits(logits)
            for position, index in enumerate(chunk):
                values[index] = chunk_values[position]
            self._usage["n_calls"] += 1
        self._usage["n_truncated_inputs"] += truncated
        self._usage["seconds"] += time.perf_counter() - started
        self._record_peak_memory()

        out: list[list[float]] = []
        for begin, end in spans:
            out.append(values[begin:end])
        return out

    def _measure_lengths(self, first: Sequence[str], second: Sequence[str] | None) -> list[int]:
        """Every input's token count, without truncation (the sort key and the
        truncation report).

        The truncation is **counted, never silent**: every input over the
        budget is counted (the track's rule: a shortened input is never
        silent).
        """
        if second is None:
            full = self._tokenizer(list(first), padding=False, truncation=False)
        else:
            full = self._tokenizer(list(first), list(second), padding=False, truncation=False)
        return [len(ids) for ids in full["input_ids"]]

    def _encode(self, first: Sequence[str], second: Sequence[str] | None) -> dict[str, Any]:
        """Tokenize a batch (with the truncation) and move it to the device."""
        kwargs: dict[str, Any] = {
            "padding": True,
            "truncation": True,
            "max_length": self._config.max_length,
            "return_tensors": "pt",
        }
        if second is None:
            encoded = self._tokenizer(list(first), **kwargs)
        else:
            encoded = self._tokenizer(list(first), list(second), **kwargs)
        device = getattr(self._model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return dict(encoded)

    def _read_logits(self, logits: Any) -> list[float]:
        """The batch's scores, per the family's own score transform."""
        values = logits.view(-1).float()
        if self._config.kind == KIND_CROSS_ENCODER:
            values = self._torch.sigmoid(values)
        return [float(value) for value in values.cpu().tolist()]

    # --- the listwise family (one forward pass per candidate list) ---

    def _score_listwise(self, requests: Sequence[RerankRequest]) -> list[list[float]]:
        """The listwise family: the whole candidate list in one context."""
        started = time.perf_counter()
        out: list[list[float]] = []
        truncated = 0
        max_documents = self._config.max_documents
        for request in requests:
            query_length = len(self._tokenizer(request.query)["input_ids"])
            truncated += int(
                self._config.max_query_length is not None
                and query_length > self._config.max_query_length
            )
            scores: list[float] = []
            for start in range(0, len(request.documents), max_documents):
                block = request.documents[start : start + max_documents]
                for document in block:
                    length = len(self._tokenizer(document)["input_ids"])
                    truncated += int(
                        self._config.max_doc_length is not None
                        and length > self._config.max_doc_length
                    )
                result = self._model.rerank(
                    request.query,
                    list(block),
                    max_query_length=self._config.max_query_length,
                    max_doc_length=self._config.max_doc_length,
                )
                block_scores = [0.0] * len(block)
                for item in result:
                    index = int(item["index"])
                    block_scores[index] = float(item["relevance_score"])
                scores.extend(block_scores)
                self._usage["n_calls"] += 1
            out.append(scores)
        self._usage["n_truncated_inputs"] += truncated
        self._usage["seconds"] += time.perf_counter() - started
        self._record_peak_memory()
        return out

    def _record_peak_memory(self) -> None:
        """Record the peak GPU memory of the run (in MB)."""
        torch = self._torch
        if not torch.cuda.is_available():
            return
        peak = float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
        self._usage["peak_gpu_mb"] = max(self._usage["peak_gpu_mb"], peak)
