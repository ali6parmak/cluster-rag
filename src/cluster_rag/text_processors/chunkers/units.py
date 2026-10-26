"""The chunk profile's units: sentences and word pieces (exp17's granularities).

The Stage-3.3 chunk profile needs **both** sides of a pair cut into small
pieces, so that a pair's local matches can be *ranked* instead of collapsed
into a single maximum. exp17 (the reference lineage) cuts both sides into
**sentences** and into **5-word pieces inside the sentences**; this module is
that cut.

Two kinds, one class:

- ``sentence`` — the track's rule-based sentence splitter (shared with the
  atom chunker), so a sentence is the same unit everywhere in the track;
- ``ngram`` — the non-overlapping ``n_words``-word pieces **inside** a
  sentence. A piece never spans a sentence boundary, and the last piece of a
  sentence may be shorter than ``n_words`` (the pieces *tile* the sentence, so
  no text is dropped).

The splitter is a deterministic rule set (no NLP dependency) and every rule
that can change the output is a parameter of :class:`UnitSplitterConfig`, so
the parameters are hashed into the chunker's cache-facing version: two
differently configured unit splitters can never share cached units.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from cluster_rag.text_processors.base import BaseChunker
from cluster_rag.text_processors.chunkers.atoms import DEFAULT_ABBREVIATIONS, split_sentences

# The kinds of unit the profile cuts a text into.
UNIT_KINDS: tuple[str, ...] = ("sentence", "ngram")


@dataclass(frozen=True)
class UnitSplitterConfig:
    """Parameters of the unit splitter (the profile's two granularities).

    Attributes:
        kind: ``sentence`` (the track's sentence splitter) or ``ngram`` (the
            word pieces inside the sentences).
        n_words: The nominal size of a word piece (the ``ngram`` kind).
        min_unit_chars: Units shorter than this are dropped (1 = keep all).
        abbreviations: Tokens that end in a period without ending a sentence.
    """

    kind: Literal["sentence", "ngram"] = "sentence"
    n_words: int = 5
    min_unit_chars: int = 1
    abbreviations: tuple[str, ...] = DEFAULT_ABBREVIATIONS

    def __post_init__(self) -> None:
        """Normalize the sequence parameter and reject a degenerate setting.

        Raises:
            ValueError: If the kind is unknown, the piece size is too small
                or the minimum unit size is not positive.
        """
        object.__setattr__(self, "abbreviations", tuple(self.abbreviations))
        if self.kind not in UNIT_KINDS:
            raise ValueError(f"the unit kind must be one of {list(UNIT_KINDS)}, got {self.kind!r}")
        if self.kind == "ngram" and self.n_words < 2:
            raise ValueError(f"n_words must be >= 2, got {self.n_words}")
        if self.min_unit_chars < 1:
            raise ValueError(f"min_unit_chars must be >= 1, got {self.min_unit_chars}")

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (hashed into the version)."""
        return {
            "kind": self.kind,
            "n_words": self.n_words,
            "min_unit_chars": self.min_unit_chars,
            "abbreviations": list(self.abbreviations),
        }


class UnitSplitter(BaseChunker):
    """Splits a text into the profile's units (sentences or word pieces)."""

    name = "units"

    def __init__(self, config: UnitSplitterConfig | None = None) -> None:
        self.config = config or UnitSplitterConfig()

    @property
    def slug(self) -> str:
        """A parameter-aware handle (``"units-sent"`` / ``"units-5w"``)."""
        if self.config.kind == "sentence":
            return "units-sent"
        return f"units-{self.config.n_words}w"

    def version_payload(self) -> dict[str, Any]:
        """The parameters that define the chunker's behaviour."""
        return self.config.payload()

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text into its units."""
        return [self.units(text) for text in texts]

    def units(self, text: str) -> list[str]:
        """The units of one text (empty for a blank text)."""
        sentences = split_sentences(text, abbreviations=self.config.abbreviations)
        if self.config.kind == "sentence":
            pieces: list[str] = sentences
        else:
            pieces = [piece for sentence in sentences for piece in self._word_pieces(sentence)]
        return [piece for piece in pieces if len(piece) >= self.config.min_unit_chars]

    def _word_pieces(self, sentence: str) -> list[str]:
        """The word pieces of a sentence (they tile it; the last one may be short)."""
        words = sentence.split()
        size = self.config.n_words
        return [
            " ".join(words[start : start + size])
            for start in range(0, len(words), size)
            if words[start : start + size]
        ]
