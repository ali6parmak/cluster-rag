"""Sentence/clause atoms (exp18(i) lineage): the paragraph side of Stage 1.

A paragraph is split into sentence atoms; with ``split_clauses`` each
sentence is split further at the configured clause separators (commas and
semicolons by default), so a long sentence contributes several small atoms
instead of one.

The splitter is a deterministic rule set — no NLP dependency (a new
dependency needs evidence first, AGENTS.md section 3) — and every rule that
can change the output is a parameter of :class:`AtomChunkerConfig`, so the
parameters are hashed into the chunker's cache-facing version.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cluster_rag.text_processors.base import BaseChunker

# Tokens that end in a period without ending a sentence. Deliberately short
# and conservative: a token that is a normal word ("no", "al") would swallow
# real sentence boundaries in prose, which is worse than a rare miss.
DEFAULT_ABBREVIATIONS: tuple[str, ...] = (
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "para",
    "paras",
    "sect",
    "cf",
    "approx",
    "inc",
    "ltd",
    "u.s",
    "u.k",
)

# A sentence end: a terminator, optional closing quotes/brackets, then space.
_BOUNDARY = re.compile(r"[.!?…]+[\"')\]]*\s+")
_TRAILING_TOKEN = re.compile(r"[^\s]+$")


def _trailing_token(before: str) -> str:
    """The last whitespace-delimited token of ``before``, without leading quotes."""
    if not before.strip():
        return ""
    match = _TRAILING_TOKEN.search(before)
    token = match.group(0) if match else ""
    return token.strip("([\"'“‘«")


def _is_sentence_end(text: str, match: re.Match[str], abbreviations: frozenset[str]) -> bool:
    """Whether a candidate boundary really ends a sentence.

    Two false positives are filtered: a decimal point ("3. 5") and a token
    that ends in a period without ending a sentence (an abbreviation or a
    single-letter initial, "e.g." / "J.").
    """
    before = text[: match.start()]
    after = text[match.end() :].lstrip()[:1]
    if before[-1:].isdigit() and after[:1].isdigit():
        return False
    token = _trailing_token(before)
    if not token:
        return True
    if token.lower() in abbreviations:
        return False
    # an initial ("J. Smith") is not a sentence end either
    return not (len(token) == 1 and token.isalpha())


def split_sentences(text: str, abbreviations: Sequence[str] = DEFAULT_ABBREVIATIONS) -> list[str]:
    """Split a text into sentences (deterministic, rule-based).

    Args:
        text: The text to split; blank text yields no sentences.
        abbreviations: Tokens that end in a period without ending a sentence.

    Returns:
        The sentences, in order, each stripped of surrounding whitespace and
        keeping its terminator.
    """
    stripped = text.strip()
    if not stripped:
        return []
    known = frozenset(abbreviation.lower() for abbreviation in abbreviations)
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(stripped):
        if not _is_sentence_end(stripped, match, known):
            continue
        chunk = stripped[start : match.end()].strip()
        if chunk:
            sentences.append(chunk)
        start = match.end()
    tail = stripped[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _next_separator(
    text: str, index: int, ordered_separators: Sequence[str]
) -> tuple[int, int] | None:
    """The earliest separator at or after ``index``, as (begin, end), or None.

    ``ordered_separators`` holds the separators longest-first, so the longest
    one wins a tie on the same position.
    """
    best: tuple[int, int] | None = None
    for separator in ordered_separators:
        position = text.find(separator, index)
        if position == -1:
            continue
        if best is None or position < best[0]:
            best = (position, position + len(separator))
    return best


def _is_digit_group_separator(text: str, begin: int, end: int) -> bool:
    """Whether the separator sits between digits (a thousands separator).

    "49,002 people" must not become "49" + "002 people": the comma belongs to
    the number, it is not a clause boundary.
    """
    before = text[begin - 1] if begin > 0 else ""
    after = text[end] if end < len(text) else ""
    return before.isdigit() and after.isdigit()


def split_clauses(sentence: str, separators: Sequence[str] = (",", ";")) -> list[str]:
    """Split one sentence at the given clause separators (separators dropped).

    A separator between two digits is part of a number ("49,002") and is not
    a clause boundary.

    Args:
        sentence: The sentence to split.
        separators: The strings that start a new clause.

    Returns:
        The clause fragments, stripped and without empty pieces.
    """
    if not sentence.strip():
        return []
    if not separators:
        return [sentence.strip()]
    ordered = sorted(separators, key=len, reverse=True)
    pieces: list[str] = []
    start = 0
    index = 0
    while index < len(sentence):
        found = _next_separator(sentence, index, ordered)
        if found is None:
            break
        begin, end = found
        if _is_digit_group_separator(sentence, begin, end):
            index = end  # a thousands separator: keep scanning
            continue
        pieces.append(sentence[start:begin])
        start = end
        index = end
    pieces.append(sentence[start:])
    return [piece.strip() for piece in pieces if piece.strip()]


@dataclass(frozen=True)
class AtomChunkerConfig:
    """Parameters of the atom chunker (exp18(i) lineage).

    Attributes:
        split_clauses: Split each sentence further at ``clause_separators``.
        clause_separators: The strings that start a new clause.
        min_atom_chars: Atoms shorter than this are dropped (1 = keep all).
        abbreviations: Tokens that end in a period without ending a sentence.
    """

    split_clauses: bool = True
    clause_separators: tuple[str, ...] = (",", ";")
    min_atom_chars: int = 1
    abbreviations: tuple[str, ...] = DEFAULT_ABBREVIATIONS

    def __post_init__(self) -> None:
        """Normalize the sequence parameters to tuples.

        A config file carries them as lists, so the version hash (and the
        splitter's behaviour) must not depend on the container type.
        """
        object.__setattr__(self, "clause_separators", tuple(self.clause_separators))
        object.__setattr__(self, "abbreviations", tuple(self.abbreviations))

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (hashed into the version)."""
        return {
            "split_clauses": self.split_clauses,
            "clause_separators": list(self.clause_separators),
            "min_atom_chars": self.min_atom_chars,
            "abbreviations": list(self.abbreviations),
        }


class AtomChunker(BaseChunker):
    """Splits a text into sentence atoms (optionally clause atoms)."""

    name = "atoms"

    def __init__(self, config: AtomChunkerConfig | None = None) -> None:
        self.config = config or AtomChunkerConfig()

    def version_payload(self) -> dict[str, Any]:
        """The parameters that define this chunker's behaviour."""
        return self.config.payload()

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text into its atoms."""
        return [self.atoms(text) for text in texts]

    def atoms(self, text: str) -> list[str]:
        """The atoms of one text: sentences, optionally split into clauses."""
        sentences = split_sentences(text, abbreviations=self.config.abbreviations)
        if not self.config.split_clauses:
            pieces = sentences
        else:
            pieces = [
                piece
                for sentence in sentences
                for piece in split_clauses(sentence, self.config.clause_separators)
            ]
        return [piece for piece in pieces if len(piece) >= self.config.min_atom_chars]
