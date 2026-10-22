"""The track's chunkers: one file per method, one class per file.

``CHUNKERS`` maps a config-facing name to the chunker class, so the CLI can
build a chunker from a config section without importing the modules by hand.
The registry grows with the track's methods (word/sentence windows, semantic
boundaries, question parts — TEXT_PROCESSING_PLAN.md section 2).
"""

from cluster_rag.text_processors.base import BaseChunker
from cluster_rag.text_processors.chunkers.atoms import (
    DEFAULT_ABBREVIATIONS,
    AtomChunker,
    AtomChunkerConfig,
    split_clauses,
    split_sentences,
)
from cluster_rag.text_processors.chunkers.full_text import FullTextChunker

CHUNKERS: dict[str, type[BaseChunker]] = {
    "full_text": FullTextChunker,
    "atoms": AtomChunker,
}

__all__ = [
    "CHUNKERS",
    "DEFAULT_ABBREVIATIONS",
    "AtomChunker",
    "AtomChunkerConfig",
    "BaseChunker",
    "FullTextChunker",
    "split_clauses",
    "split_sentences",
]
