"""The chunk header (Stage 2.6): the paragraph's **own first sentence**.

The classic "contextual chunk header" idea, in the one form that is legal on
this data: the header is the text's **own first sentence** — pair-local,
label-free, no document identity — and it keeps one embedding per chunked
text (a *question-stem* header would need one embedding per (chunk, question)
pair: ~175,000 instead of ~500 per model).

The other headings are ruled out by the track's leakage rules: the **document
title** is document identity (rule ii) and the **question number** is a corpus
statistic (rule i — 186/294 question numbers are judged in more than one
document, 101 with conflicting labels).

The inner chunker (the windows) does the splitting; this chunker only
prefixes its chunks with the header. **Two modes, both measured** (the
reviewer's rule: measure the alternatives rather than picking one):

- ``"others"`` (the default): the header is prepended to every chunk
  **except the first** one. The first chunk already starts with the header's
  text, so prefixing it would only duplicate it — and for a one-sentence
  paragraph (35 of the corpus's 225) it would duplicate the whole text. This
  mode measures the header's *context* and nothing else: the chunks' own
  content is unchanged.
- ``"all"``: the header is prepended to every chunk (the literal reading of
  the idea). The first chunk then carries its first sentence twice.

The header is the text's first sentence, split by the track's deterministic
sentence splitter (:func:`split_sentences`) — no model, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cluster_rag.text_processors.base import BaseChunker
from cluster_rag.text_processors.chunkers.atoms import split_sentences

# The two modes: the header on every chunk, or on every chunk but the first.
HEADER_MODE_ALL = "all"
HEADER_MODE_OTHERS = "others"
HEADER_MODES: tuple[str, ...] = (HEADER_MODE_OTHERS, HEADER_MODE_ALL)


@dataclass(frozen=True)
class HeaderConfig:
    """Parameters of the header chunker.

    Attributes:
        chunker: The registry name of the inner chunker (the one that splits).
        chunker_params: The inner chunker's parameters.
        mode: ``"others"`` (the header on every chunk but the first) or
            ``"all"`` (the header on every chunk).
        separator: What separates the header from the chunk.
    """

    chunker: str = "word_windows"
    chunker_params: dict[str, Any] = field(default_factory=dict)
    mode: str = HEADER_MODE_OTHERS
    separator: str = "\n"

    def __post_init__(self) -> None:
        """Reject an unknown mode.

        Raises:
            ValueError: If the mode is not one of the two.
        """
        if self.mode not in HEADER_MODES:
            raise ValueError(
                f"Unknown header mode {self.mode!r}; expected one of {list(HEADER_MODES)}"
            )

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (hashed into the version)."""
        return {
            "chunker": self.chunker,
            "chunker_params": dict(self.chunker_params),
            "mode": self.mode,
            "separator": self.separator,
        }


class HeaderChunker(BaseChunker):
    """Prefixes every chunk of a text with the text's own first sentence."""

    name = "header"

    def __init__(self, config: HeaderConfig | None = None) -> None:
        self.config = config or HeaderConfig()
        # A local import: the registry imports this module, so a top-level
        # import of the registry here would be a cycle.
        from cluster_rag.text_processors.chunkers import build_chunker

        self.inner = build_chunker(self.config.chunker, self.config.chunker_params)

    @property
    def slug(self) -> str:
        """A parameter-aware handle (``"words25_header"``) that names the arm."""
        return f"{self.inner.slug}_header"

    def version_payload(self) -> dict[str, Any]:
        """The parameters that define this chunker's behaviour."""
        return self.config.payload()

    def header_of(self, text: str) -> str:
        """The text's first sentence (the header), or an empty string."""
        sentences = split_sentences(text)
        return sentences[0] if sentences else ""

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text with the inner chunker and prefix its chunks."""
        chunk_lists = self.inner.segment(list(texts))
        segmented: list[list[str]] = []
        for text, chunks in zip(texts, chunk_lists, strict=True):
            header = self.header_of(text)
            prefixed: list[str] = []
            for index, chunk in enumerate(chunks):
                if not header or (self.config.mode == HEADER_MODE_OTHERS and index == 0):
                    prefixed.append(chunk)
                else:
                    prefixed.append(f"{header}{self.config.separator}{chunk}")
            segmented.append(prefixed)
        return segmented
