"""The reranker's port (the text-processing track's CS-15 / PLAN's Phase 6).

A **reranker** scores a *candidate list* against a *query*: it is the
component that re-orders an existing set of candidates, never one that
enlarges it. The port is deliberately small, and it is shaped around the
**request** (one query, one candidate list) because the three reranker
families differ in what one forward pass covers:

- the **cross-encoder** (e.g. ``jina-reranker-v2-base-multilingual``): one
  forward pass per **(query, document) pair**;
- the **LLM reranker with a relevance head** (e.g.
  ``nvidia/llama-nemotron-rerank-1b-v2``): one forward pass per pair as well,
  but the pair is rendered into a **single sequence** by a prompt template;
- the **listwise reranker** (e.g. ``jinaai/jina-reranker-v3``): one forward
  pass per **(query, up to N documents)** — the whole candidate list in one
  context, which is also why its score for a document depends on the *other*
  documents of the request.

This module holds the port's data shapes (the :class:`RerankRequest`, the
:class:`RerankInfo` and the :class:`Reranker` protocol) — no model client, no
I/O. The client lives in :mod:`cluster_rag.rerank.hf` and the cache in
:mod:`cluster_rag.rerank.cache`.

**Why a port at all** (AGENTS.md dependency direction): the pure side of the
track (the candidate sets, the rank fusion, the measurement) must be testable
without a GPU, so the pure core talks to a ``Reranker`` protocol and the CLI
wires the real client in.

**The hygiene the port carries** (TEXT_PROCESSING_PLAN.md section 2): the
model is pinned by id **and revision**, the scores are **cached** (a reranker
score is not a vector, so it never enters the embedding cache), and a
shortened input is never silent (the client reports its truncation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# The reranker families (what one forward pass covers).
KIND_CROSS_ENCODER = "cross_encoder"
KIND_LLM_YESNO = "llm_yesno"
KIND_LISTWISE = "listwise"
RERANK_KINDS: tuple[str, ...] = (KIND_CROSS_ENCODER, KIND_LLM_YESNO, KIND_LISTWISE)

# The families that score one (query, document) pair per forward pass.
PAIR_KINDS: tuple[str, ...] = (KIND_CROSS_ENCODER, KIND_LLM_YESNO)


@dataclass(frozen=True)
class RerankRequest:
    """One scoring request: a query and the candidate documents to score.

    The **order** of the documents is part of the request: a listwise
    reranker reads them in one context, so a document's score depends on its
    neighbours' positions. The pair-based families score each document on its
    own, so the order is irrelevant to them (and the cache key keeps it
    anyway, so a request always means the same thing).

    Attributes:
        query: The query side of the request (the text, for the pipeline's
            retrieval framing).
        documents: The candidate documents (the label sides), in the order
            the caller wants them read.
    """

    query: str
    documents: tuple[str, ...]


@dataclass(frozen=True)
class RerankInfo:
    """The provenance of a reranker: the id, the pinned revision, the family.

    ``extra`` carries the run-time facts (the device, the dtype, the model's
    own context budget, the score transform) and is written into the run's
    artifact.
    """

    model_id: str
    revision: str
    kind: str
    backend: str = "huggingface"
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Reranker(Protocol):
    """A reranker: a query plus candidate documents -> a score per document."""

    @property
    def info(self) -> RerankInfo:
        """The provenance of the model (its id, its pinned revision, its family)."""

    def score(self, requests: Sequence[RerankRequest]) -> list[list[float]]:
        """Score every request's documents; one list of scores per request."""


def pair_requests(
    queries: Sequence[str], documents: Sequence[Sequence[str]]
) -> list[RerankRequest]:
    """One request per (query, document) pair (the pair-based families).

    A pair-based reranker scores each document on its own, so the request
    carries a single document. The caller then gets one score per candidate,
    and the candidate set's order is the caller's business.

    Args:
        queries: The query side of every request (one per text).
        documents: The candidate documents of every query.

    Returns:
        The requests, in the (query, document) order.

    Raises:
        ValueError: If the queries and the document lists disagree.
    """
    if len(queries) != len(documents):
        raise ValueError(
            f"the queries and the document lists must have the same length, got "
            f"{len(queries)} and {len(documents)}"
        )
    requests: list[RerankRequest] = []
    for query, candidates in zip(queries, documents, strict=True):
        for document in candidates:
            requests.append(RerankRequest(query=query, documents=(document,)))
    return requests


def listwise_requests(
    queries: Sequence[str], documents: Sequence[Sequence[str]]
) -> list[RerankRequest]:
    """One request per query, carrying its whole candidate list (the listwise family).

    The **order** of the documents is the caller's (the incumbent's order
    inside the candidate set): a listwise reranker reads them in one context,
    so the order is part of the input, not a detail.

    Args:
        queries: The query side of every request (one per text).
        documents: The candidate documents of every query, in the order they
            are read.

    Returns:
        The requests, one per query.

    Raises:
        ValueError: If the queries and the document lists disagree.
    """
    if len(queries) != len(documents):
        raise ValueError(
            f"the queries and the document lists must have the same length, got "
            f"{len(queries)} and {len(documents)}"
        )
    return [
        RerankRequest(query=query, documents=tuple(candidates))
        for query, candidates in zip(queries, documents, strict=True)
    ]


def render_template(template: str, query: str, document: str) -> str:
    """Render a pair template (``{query}`` / ``{document}``) into one sequence.

    The LLM rerankers with a relevance head are trained on a **single**
    sequence per pair (``"question:... passage:..."``), so the template is the
    model's own prompt format and it is config, never hard-coded.

    Args:
        template: The template with the ``{query}`` and ``{document}``
            placeholders.
        query: The query side.
        document: The document side.

    Returns:
        The rendered sequence.

    Raises:
        ValueError: If the template carries an unknown placeholder.
    """
    try:
        rendered = template.format(query=query, document=document)
    except (KeyError, IndexError) as exc:
        raise ValueError(
            "the pair template must use the {query} and {document} placeholders "
            f"only, got {template!r} ({exc})"
        ) from exc
    return rendered
