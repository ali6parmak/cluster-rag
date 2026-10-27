"""The rerankers at the edge (the text-processing track's CS-15, PLAN's Phase 6).

A reranker re-orders a candidate list: it is the component that PLAN's Phase 6
asks for ("Cross-encoder reranker on (text, label name/description) pairs.
Zero-shot first, then fine-tuned on the auxiliary pairwise data").

Three modules:

- :mod:`cluster_rag.rerank.base` — the :class:`~cluster_rag.rerank.base.Reranker`
  port, the request shape and the request builders (no model client, no I/O);
- :mod:`cluster_rag.rerank.hf` — the Hugging Face client (the three families:
  the cross-encoder, the yes/no head and the listwise model);
- :mod:`cluster_rag.rerank.cache` — the rerankers' **own** disk cache (keyed
  by the model id, the revision, the family, the score's shape and the
  request), which never mixes with the embedding cache.

The pure side of the track (the candidate sets, the rank fusion, the
measurement) lives in :mod:`cluster_rag.text_processors.rerank`.
"""

from cluster_rag.rerank.base import (
    KIND_CROSS_ENCODER,
    KIND_LISTWISE,
    KIND_LLM_YESNO,
    PAIR_KINDS,
    RERANK_KINDS,
    Reranker,
    RerankInfo,
    RerankRequest,
    listwise_requests,
    pair_requests,
    render_template,
)
from cluster_rag.rerank.cache import (
    CachedReranker,
    rerank_cache_key,
    rerank_cache_path,
)
from cluster_rag.rerank.hf import HFReranker

__all__ = [
    "KIND_CROSS_ENCODER",
    "KIND_LISTWISE",
    "KIND_LLM_YESNO",
    "PAIR_KINDS",
    "RERANK_KINDS",
    "CachedReranker",
    "HFReranker",
    "RerankInfo",
    "RerankRequest",
    "Reranker",
    "listwise_requests",
    "pair_requests",
    "rerank_cache_key",
    "rerank_cache_path",
    "render_template",
]
