"""Overlapping sentence windows (exp02 lineage): the paragraph side of Stage 2.

A text is split into its sentences (the track's rule-based splitter, shared
with the atom chunker) and the sentences are grouped into **overlapping
windows** of a fixed number of sentences. The plan's Stage 2.2 uses two
sentences with a step of one, so every sentence appears in two neighbouring
windows.

**Window layout.** The windows cover the whole text:

- a text of at most ``n_sentences`` sentences is one window;
- otherwise the window starts are ``0, step, 2 * step, ...`` while the window
  still fits, plus one **tail window** that ends at the last sentence (so the
  last sentences are never a one-sentence fragment).

For example, 5 sentences with ``n_sentences=2`` and ``step=1`` give 4 windows
(starts 0, 1, 2 and 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cluster_rag.text_processors.base import BaseChunker
from cluster_rag.text_processors.chunkers.atoms import DEFAULT_ABBREVIATIONS, split_sentences


@dataclass(frozen=True)
class SentenceWindowsConfig:
    """Parameters of the sentence-window chunker (exp02 lineage).

    Attributes:
        n_sentences: The window size in sentences.
        step: The step between two windows, in sentences (1 = every sentence
            starts a window, so consecutive windows overlap).
        min_chunk_sentences: Windows with fewer sentences are dropped
            (1 = keep all).
        abbreviations: Tokens that end in a period without ending a sentence.
    """

    n_sentences: int = 2
    step: int = 1
    min_chunk_sentences: int = 1
    abbreviations: tuple[str, ...] = DEFAULT_ABBREVIATIONS

    def __post_init__(self) -> None:
        """Normalize the sequence parameter and reject a degenerate geometry.

        Raises:
            ValueError: If the window size, the step or the minimum chunk size
                is not positive.
        """
        object.__setattr__(self, "abbreviations", tuple(self.abbreviations))
        if self.n_sentences < 1:
            raise ValueError(f"n_sentences must be >= 1, got {self.n_sentences}")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        if self.min_chunk_sentences < 1:
            raise ValueError(f"min_chunk_sentences must be >= 1, got {self.min_chunk_sentences}")

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (hashed into the version)."""
        return {
            "n_sentences": self.n_sentences,
            "step": self.step,
            "min_chunk_sentences": self.min_chunk_sentences,
            "abbreviations": list(self.abbreviations),
        }


class SentenceWindowsChunker(BaseChunker):
    """Splits a text into overlapping windows of sentences."""

    name = "sentence_windows"

    def __init__(self, config: SentenceWindowsConfig | None = None) -> None:
        self.config = config or SentenceWindowsConfig()

    @property
    def slug(self) -> str:
        """A parameter-aware handle (``"sent2s1"``) that names the arm."""
        return f"sent{self.config.n_sentences}s{self.config.step}"

    def version_payload(self) -> dict[str, Any]:
        """The parameters that define this chunker's behaviour."""
        return self.config.payload()

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text into its sentence windows."""
        return [self.windows(text) for text in texts]

    def windows(self, text: str) -> list[str]:
        """The sentence windows of one text (empty for a blank text)."""
        sentences = split_sentences(text, abbreviations=self.config.abbreviations)
        if not sentences:
            return []
        size = self.config.n_sentences
        if len(sentences) <= size:
            candidates = [sentences]
        else:
            starts = list(range(0, len(sentences) - size + 1, self.config.step))
            tail = len(sentences) - size
            if starts[-1] != tail:
                starts.append(tail)
            candidates = [sentences[start : start + size] for start in starts]
        return [
            " ".join(window)
            for window in candidates
            if len(window) >= self.config.min_chunk_sentences
        ]
