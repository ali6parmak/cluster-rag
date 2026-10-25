"""Semantic boundaries (exp18(ii) lineage): cut where the topic changes.

The text is split into **atoms** (sentences, optionally clause atoms), every
atom is embedded, and a boundary is cut **between two adjacent atoms whose
cosine similarity drops below a threshold**. The atoms between two cuts form
one chunk.

This is the one chunker of the track that is not a pure text transform: it
needs the *atom* embeddings to find its boundaries. The plan's section 2
("inject an Embedder port - NO model client inside") settles how: the
chunker takes the :class:`~cluster_rag.embed.base.Embedder` **port** as a
constructor argument, and the CLI wires the (cached) backend in. The package
itself still imports no model client - only the protocol and numpy.

The threshold is a **swept** parameter: the plan's Stage 2.3 sweeps it on
fit-side documents only, and the driver's cross-fitting chooses one threshold
per document rotation (see :mod:`cluster_rag.omct_analysis`). The threshold
is part of the chunker's cache-facing version, so two thresholds can never
share cached chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.embed.base import Embedder
from cluster_rag.text_processors.base import BaseChunker
from cluster_rag.text_processors.chunkers.atoms import (
    DEFAULT_ABBREVIATIONS,
    split_clauses,
    split_sentences,
)


@dataclass(frozen=True)
class SemanticChunkerConfig:
    """Parameters of the semantic-boundary chunker (exp18(ii) lineage).

    Attributes:
        threshold: A boundary is cut between two adjacent atoms whose cosine
            similarity is **below** this value. It is the swept parameter of
            the Stage-2.3 run.
        atom_split_clauses: Split each sentence further into clause atoms
            before the boundary search.
        min_chunk_atoms: Chunks with fewer atoms are dropped (1 = keep all).
        abbreviations: Tokens that end in a period without ending a sentence.
        embed_kind: The encode kind of the atom embeddings (the paragraph side
            is embedded as a ``document``, the question side as a ``query``).
    """

    threshold: float = 0.5
    atom_split_clauses: bool = False
    min_chunk_atoms: int = 1
    abbreviations: tuple[str, ...] = DEFAULT_ABBREVIATIONS
    embed_kind: str = "document"

    def __post_init__(self) -> None:
        """Normalize the sequence parameter and reject a degenerate setting.

        Raises:
            ValueError: If the threshold is outside [-1, 1] or the minimum
                chunk size is not positive.
        """
        object.__setattr__(self, "abbreviations", tuple(self.abbreviations))
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {self.threshold}")
        if self.min_chunk_atoms < 1:
            raise ValueError(f"min_chunk_atoms must be >= 1, got {self.min_chunk_atoms}")

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (hashed into the version)."""
        return {
            "threshold": self.threshold,
            "atom_split_clauses": self.atom_split_clauses,
            "min_chunk_atoms": self.min_chunk_atoms,
            "abbreviations": list(self.abbreviations),
            "embed_kind": self.embed_kind,
        }


class SemanticChunker(BaseChunker):
    """Cuts a text where the adjacent-atom similarity drops below a threshold.

    The chunker owns no model client: it is handed the :class:`Embedder`
    port and calls it for the atoms. The vectors are normalized defensively,
    so the boundary test is a cosine similarity whatever the backend returns.
    """

    name = "semantic"
    requires_embedder = True

    def __init__(self, config: SemanticChunkerConfig, embedder: Embedder) -> None:
        self.config = config
        self.atom_embedder = embedder

    def version_payload(self) -> dict[str, Any]:
        """The parameters that define this chunker's behaviour."""
        return self.config.payload()

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text into its semantic chunks.

        The atoms of **all** texts are embedded in one batch (the port is
        called once), then every text is cut at its own boundaries.
        """
        atom_lists = [self.atoms(text) for text in texts]
        flat = [atom for atoms in atom_lists for atom in atoms]
        vectors = (
            self.atom_embedder.encode(flat, self.config.embed_kind)  # type: ignore[arg-type]
            if flat
            else np.zeros((0, 1), dtype=np.float32)
        )
        unit = _unit_rows(vectors)
        chunks: list[list[str]] = []
        cursor = 0
        for atoms in atom_lists:
            count = len(atoms)
            if count == 0:
                chunks.append([])
                continue
            similarities = unit[cursor : cursor + count] @ unit[cursor : cursor + count].T
            chunks.append(self._chunks(atoms, similarities))
            cursor += count
        return chunks

    def atoms(self, text: str) -> list[str]:
        """The atoms the boundary search runs on."""
        sentences = split_sentences(text, abbreviations=self.config.abbreviations)
        if not self.config.atom_split_clauses:
            return sentences
        return [piece for sentence in sentences for piece in split_clauses(sentence)]

    def _chunks(self, atoms: Sequence[str], similarities: np.ndarray) -> list[str]:
        """Group the atoms into chunks, cutting at the low-similarity gaps."""
        groups: list[list[str]] = [[atoms[0]]]
        for index in range(1, len(atoms)):
            if similarities[index - 1, index] < self.config.threshold:
                groups.append([])
            groups[-1].append(atoms[index])
        return [" ".join(group) for group in groups if len(group) >= self.config.min_chunk_atoms]


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize the rows (the boundary test is a cosine similarity)."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe
