"""The no-op baseline chunker: one chunk per text, unchanged.

The baseline every other chunker is compared against (exp01 in the reference
lineage, and the "full text vs full text" row of the track's Stage 2 ladder).

Identity chunking has one useful consequence: the chunk *is* the preprocessed
text, so the embedding-cache version slot stays the preprocessing version
verbatim — the warm cache of the 14 pool models is reused as is, with no
re-encoding.
"""

from __future__ import annotations

from collections.abc import Sequence

from cluster_rag.text_processors.base import BaseChunker


class FullTextChunker(BaseChunker):
    """Keeps the whole text as a single chunk."""

    name = "full_text"

    @property
    def version(self) -> str:
        """Stable version string (the chunker has no parameters)."""
        return "full_text-v1"

    def cache_version(self, base_version: str) -> str:
        """Identity chunking: the preprocessing version is used unchanged.

        The chunked text is byte-identical to the preprocessed text, so the
        cache entries of the unchunked run are the right ones — reusing them
        is what keeps the track's runs warm (no re-encoding).
        """
        return base_version

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """One chunk per non-blank text (a blank text yields no chunks)."""
        return [[text.strip()] if text.strip() else [] for text in texts]
