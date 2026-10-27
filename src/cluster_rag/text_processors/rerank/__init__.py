"""The rerankers' pure side (the text-processing track's CS-15).

One pure module:

- :mod:`cluster_rag.text_processors.rerank.upr_rerank` — the candidate sets
  and their **ceiling**, the label side and the **label-blind counterpart**,
  the **rank fusion** and the measurement of a re-ordered arm.

Pure core (AGENTS.md dependency direction): numpy and the standard library
only — no torch, sentence-transformers, httpx or sqlite3. The model client
(the Hugging Face reranker) lives in :mod:`cluster_rag.rerank`, at the edge,
and the CLI wires it in.
"""

from cluster_rag.text_processors.rerank.upr_rerank import (
    LABEL_DOCUMENT_DESCRIPTION,
    LABEL_DOCUMENT_NAME,
    LABEL_DOCUMENT_VARIANTS,
    ORDER_LABEL_FIRST,
    ORDER_TEXT_FIRST,
    ORDER_VARIANTS,
    candidate_ceiling,
    candidate_columns,
    fuse_candidates,
    label_documents,
    measure_reranked_arm,
    rerank_matrix,
    shifted_documents,
)

__all__ = [
    "LABEL_DOCUMENT_DESCRIPTION",
    "LABEL_DOCUMENT_NAME",
    "LABEL_DOCUMENT_VARIANTS",
    "ORDER_LABEL_FIRST",
    "ORDER_TEXT_FIRST",
    "ORDER_VARIANTS",
    "candidate_ceiling",
    "candidate_columns",
    "fuse_candidates",
    "label_documents",
    "measure_reranked_arm",
    "rerank_matrix",
    "shifted_documents",
]
