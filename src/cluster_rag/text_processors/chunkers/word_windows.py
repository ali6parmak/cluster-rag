"""Fixed word windows (exp03 lineage): the paragraph side of Stage 2.

A text is split into overlapping windows of a fixed number of **words**. The
plan's Stage 2.1 sweeps the window size (N in {15, 25, 40}) with a stride of
about N/2, so a window overlaps its neighbour and no sentence is cut in half by
the window boundary.

The splitter is a deterministic rule set (no NLP dependency) and every rule
that can change the output is a parameter of :class:`WordWindowsConfig`, so
the parameters are hashed into the chunker's cache-facing version.

**Window layout.** The windows are full-size (``n_words`` words each) and
cover the whole text:

- a text of at most ``n_words`` words is one window;
- otherwise the window starts are ``0, stride, 2 * stride, ...`` while the
  window still fits, plus one **tail window** that ends at the last word (so
  the last words of a text are never a one-word fragment).

For example, 71 words with ``n_words=15`` and ``stride=7`` give 9 windows
(starts 0, 7, ..., 56); with ``n_words=40`` and ``stride=20`` they give 3
(starts 0, 20 and the tail window at 31).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cluster_rag.text_processors.base import BaseChunker


@dataclass(frozen=True)
class WordWindowsConfig:
    """Parameters of the fixed-window chunker (exp03 lineage).

    Attributes:
        n_words: The window size in words.
        stride: The step between two windows, in words (the plan uses about
            ``n_words / 2``, so consecutive windows overlap).
        min_chunk_words: Windows with fewer words are dropped (1 = keep all).
    """

    n_words: int = 20
    stride: int = 10
    min_chunk_words: int = 1

    def __post_init__(self) -> None:
        """Reject a degenerate window geometry.

        Raises:
            ValueError: If the window size or the stride is not positive, or
                the minimum chunk size is not positive.
        """
        if self.n_words < 1:
            raise ValueError(f"n_words must be >= 1, got {self.n_words}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.min_chunk_words < 1:
            raise ValueError(f"min_chunk_words must be >= 1, got {self.min_chunk_words}")

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (hashed into the version)."""
        return {
            "n_words": self.n_words,
            "stride": self.stride,
            "min_chunk_words": self.min_chunk_words,
        }


class WordWindowsChunker(BaseChunker):
    """Splits a text into fixed-size word windows."""

    name = "word_windows"

    def __init__(self, config: WordWindowsConfig | None = None) -> None:
        self.config = config or WordWindowsConfig()

    @property
    def slug(self) -> str:
        """A parameter-aware handle (``"words15"``) that names the arm."""
        return f"words{self.config.n_words}"

    def version_payload(self) -> dict[str, Any]:
        """The parameters that define this chunker's behaviour."""
        return self.config.payload()

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text into its word windows."""
        return [self.windows(text) for text in texts]

    def window_starts(self, n_words: int) -> list[int]:
        """The start index of every window of a text with ``n_words`` words.

        The layout is the one :meth:`windows` uses: a text of at most
        ``n_words`` words is one window; otherwise the window starts are
        ``0, stride, 2 * stride, ...`` while the window still fits, plus one
        **tail window** that ends at the last word.
        """
        size = self.config.n_words
        if n_words <= size:
            return [0]
        starts = list(range(0, n_words - size + 1, self.config.stride))
        tail = n_words - size
        if starts[-1] != tail:
            starts.append(tail)
        return starts

    def windows(self, text: str) -> list[str]:
        """The word windows of one text (empty for a blank text)."""
        words = text.split()
        if not words:
            return []
        size = self.config.n_words
        return [
            " ".join(words[start : start + size])
            for start in self.window_starts(len(words))
            if len(words[start : start + size]) >= self.config.min_chunk_words
        ]

    def segment_spans(self, texts: Sequence[str]) -> list[list[tuple[int, int]]]:
        """The **exact** character span of every window (the words' own offsets).

        The word windows are the reference chunker of the late-chunking arm
        (Stage 2.7), so their spans are computed from the words' own character
        offsets instead of the generic "find the chunk in the text" fallback:
        the windows are the words joined by single spaces, which is not
        necessarily a substring of the original text.
        """
        segmented: list[list[tuple[int, int]]] = []
        for text in texts:
            words = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
            if not words:
                segmented.append([])
                continue
            size = self.config.n_words
            spans: list[tuple[int, int]] = []
            for start in self.window_starts(len(words)):
                window = words[start : start + size]
                if len(window) < self.config.min_chunk_words:
                    continue
                spans.append((window[0][0], window[-1][1]))
            segmented.append(spans)
        return segmented
