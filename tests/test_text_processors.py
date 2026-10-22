"""Tests for the text-processing track's pure core (TEXT_PROCESSING_PLAN.md).

Contracts under test:

- the chunker ports behave (pure, deterministic, one chunk list per text);
- the chunking version plugs into the **existing** embedding-cache version
  slot, and the identity chunker keeps the warm cache valid;
- the atom chunker splits sentences (and clauses) with the documented rules;
- the package stays a pure core: no torch / sentence-transformers / httpx /
  sqlite3 import anywhere in it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from cluster_rag.text_processors import (
    BaseChunker,
    Chunker,
    Combiner,
    FeatureExtractor,
    TextPair,
    config_version,
)
from cluster_rag.text_processors.chunkers import (
    CHUNKERS,
    AtomChunker,
    AtomChunkerConfig,
    FullTextChunker,
    split_clauses,
    split_sentences,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "cluster_rag" / "text_processors"
FORBIDDEN_IMPORTS = ("torch", "sentence_transformers", "httpx", "sqlite3", "requests")


def test_full_text_chunker_keeps_the_whole_text() -> None:
    """The baseline chunker returns exactly one chunk per non-blank text."""
    chunker = FullTextChunker()
    chunks = chunker.segment(["  The Committee recalls.  ", "Second paragraph.", "   "])
    assert chunks == [["The Committee recalls."], ["Second paragraph."], []]
    assert len(chunks) == 3  # one list per input text, always


def test_full_text_chunker_keeps_the_warm_cache_valid() -> None:
    """Identity chunking reuses the preprocessing version slot unchanged."""
    chunker = FullTextChunker()
    assert chunker.cache_version("raw-v1") == "raw-v1"
    assert chunker.version == "full_text-v1"


def test_chunker_version_is_stable_and_parameter_sensitive() -> None:
    """The version is a hash of the parameters: stable, and sensitive to them."""
    first = AtomChunker()
    second = AtomChunker()
    assert first.version == second.version
    assert first.version.startswith("atoms-v1-")

    changed = AtomChunker(AtomChunkerConfig(split_clauses=False))
    assert changed.version != first.version
    assert changed.cache_version("raw-v1") != first.cache_version("raw-v1")
    # the chunking version goes into the existing version slot
    assert first.cache_version("raw-v1").startswith("raw-v1+")


def test_config_version_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        config_version("", {"a": 1})


def test_split_sentences_basic() -> None:
    """Sentences split on terminators, each keeping its terminator."""
    text = "The Committee recalls the measure. It notes the trend! Does it matter? Yes."
    assert split_sentences(text) == [
        "The Committee recalls the measure.",
        "It notes the trend!",
        "Does it matter?",
        "Yes.",
    ]


def test_split_sentences_guards_abbreviations_decimals_and_initials() -> None:
    """A period that does not end a sentence is not a boundary."""
    text = "The figure is 3. 5 million. See e.g. the report. J. Smith spoke."
    sentences = split_sentences(text)
    assert sentences[0] == "The figure is 3. 5 million."  # decimal: not a boundary
    assert sentences[1] == "See e.g. the report."  # abbreviation: not a boundary
    assert sentences[2] == "J. Smith spoke."  # initial: not a boundary


def test_split_sentences_edge_cases() -> None:
    """Blank text yields no sentences; a single sentence stays one."""
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("One sentence without a terminator") == [
        "One sentence without a terminator"
    ]


def test_split_clauses_drops_the_separators() -> None:
    """Clauses split at the configured separators; empty pieces are dropped."""
    assert split_clauses("the Committee recalls, and the State agrees; nothing else") == [
        "the Committee recalls",
        "and the State agrees",
        "nothing else",
    ]
    assert split_clauses("no separator here", ()) == ["no separator here"]
    assert split_clauses("   ", (",",)) == []


def test_split_clauses_keeps_thousands_separators_intact() -> None:
    """A comma between digits belongs to the number, not to a clause boundary."""
    assert split_clauses("affecting at least 49,002 people") == ["affecting at least 49,002 people"]
    assert split_clauses("in December, affecting 5 million") == [
        "in December",
        "affecting 5 million",
    ]


def test_atom_chunker_splits_sentences_and_clauses() -> None:
    """The atom chunker produces sentence atoms, then clause atoms."""
    chunker = AtomChunker()
    text = "The Committee recalls the measure, and the State agrees. It notes the trend."
    assert chunker.segment([text]) == [
        [
            "The Committee recalls the measure",
            "and the State agrees.",
            "It notes the trend.",
        ]
    ]


def test_atom_chunker_without_clause_split() -> None:
    """With clause splitting off, the atoms are the sentences."""
    chunker = AtomChunker(AtomChunkerConfig(split_clauses=False))
    assert chunker.segment(["A first, and a second. Next."]) == [
        ["A first, and a second.", "Next."]
    ]


def test_atom_chunker_drops_short_atoms() -> None:
    """Atoms below the minimum length are dropped."""
    chunker = AtomChunker(AtomChunkerConfig(min_atom_chars=5))
    assert chunker.segment(["Hi, there friend."]) == [["there friend."]]


def test_atom_chunker_is_deterministic_and_pure() -> None:
    """The same input gives the same atoms; the input is not modified."""
    chunker = AtomChunker()
    text = "First sentence. Second sentence."
    first = chunker.segment([text])
    second = chunker.segment([text])
    assert first == second
    assert text == "First sentence. Second sentence."


def test_blank_text_yields_no_chunks() -> None:
    """A blank text cannot be embedded, so it yields no chunk."""
    assert FullTextChunker().segment([""])[0] == []
    assert AtomChunker().segment(["   "])[0] == []


def test_chunkers_registry_holds_the_track_methods() -> None:
    """The registry maps config-facing names to the chunker classes."""
    assert CHUNKERS["full_text"] is FullTextChunker
    assert CHUNKERS["atoms"] is AtomChunker
    for name, chunker_class in CHUNKERS.items():
        chunker = chunker_class()
        assert isinstance(chunker, Chunker)
        assert chunker.version
        assert name in chunker.version


def test_protocols_are_runtime_checkable() -> None:
    """The ports are protocols: a duck-typed implementation conforms."""

    class _FakeFeatures:
        @property
        def feature_names(self) -> tuple[str, ...]:
            return ("cos",)

        def extract(self, pairs: list[TextPair]) -> np.ndarray:
            return np.zeros((len(pairs), 1), dtype=np.float32)

    class _FakeCombiner:
        def fit(self, features: np.ndarray, labels: list[int]) -> _FakeCombiner:
            return self

        def score(self, features: np.ndarray) -> np.ndarray:
            return np.zeros(len(features), dtype=np.float32)

    assert isinstance(_FakeFeatures(), FeatureExtractor)
    assert isinstance(_FakeCombiner(), Combiner)
    assert not isinstance(object(), Chunker)
    assert not isinstance(object(), BaseChunker)


def test_text_pair_carries_text_only() -> None:
    """A pair has no label, no document name: leakage rules hold by construction."""
    pair = TextPair(paragraph_text="p", question_text="q")
    assert pair.paragraph_text == "p"
    assert pair.question_text == "q"
    assert not hasattr(pair, "label")


def test_package_imports_no_model_clients() -> None:
    """The track's package is a pure core: no torch/ST/httpx/sqlite3 import."""
    pattern = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(FORBIDDEN_IMPORTS) + r")\b")
    for module_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for line in source.splitlines():
            assert not pattern.match(line), f"{module_path.name} imports a model client: {line}"
