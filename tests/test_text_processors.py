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

import ast
import re
import sys
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
from cluster_rag.text_processors.aggregation import (
    aggregate_chunk_similarities,
    parse_aggregation,
    parse_aggregations,
)
from cluster_rag.text_processors.chunkers import (
    CHUNKERS,
    AtomChunker,
    AtomChunkerConfig,
    FullTextChunker,
    SemanticChunker,
    SemanticChunkerConfig,
    SentenceWindowsChunker,
    SentenceWindowsConfig,
    WordWindowsChunker,
    WordWindowsConfig,
    build_chunker,
    split_clauses,
    split_sentences,
)
from fake_embedder import FakeEmbedder

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "cluster_rag" / "text_processors"
FORBIDDEN_IMPORTS = ("torch", "sentence_transformers", "httpx", "sqlite3", "requests")


class ScriptedEmbedder:
    """An embedder that returns a prescribed vector per text.

    The semantic chunker finds its boundaries from the atom embeddings, so a
    test needs to *choose* the adjacent similarities instead of hashing them.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = {
            text: np.asarray(vector, dtype=np.float32) for text, vector in vectors.items()
        }
        self.dimension = len(next(iter(self.vectors.values())))

    @property
    def info(self):
        """The interface facts of this scripted embedder."""
        from cluster_rag.embed.base import EmbedderInfo

        return EmbedderInfo(
            model_id="scripted-embedder",
            backend="fake",
            dimension=self.dimension,
            revision_or_digest="scripted",
        )

    def encode(self, texts, kind) -> np.ndarray:
        """The prescribed vectors, in input order."""
        return np.vstack([self.vectors[text] for text in texts]).astype(np.float32)


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
    assert CHUNKERS["word_windows"] is WordWindowsChunker
    assert CHUNKERS["sentence_windows"] is SentenceWindowsChunker
    assert CHUNKERS["semantic"] is SemanticChunker
    for name, chunker_class in CHUNKERS.items():
        if getattr(chunker_class, "requires_embedder", False):
            chunker = chunker_class(SemanticChunkerConfig(), FakeEmbedder(dimension=4))
        else:
            chunker = chunker_class()
        assert isinstance(chunker, Chunker)
        assert chunker.version
        assert name in chunker.version


def test_word_windows_split_into_fixed_overlapping_windows() -> None:
    """The fixed windows: the plan's N with a stride of about N/2."""
    chunker = WordWindowsChunker(WordWindowsConfig(n_words=4, stride=2))
    windows = chunker.segment(["a b c d e f g h i"])[0]
    # 9 words, 4 per window, stride 2: starts 0, 2, 4 and the tail window at 5
    assert windows == ["a b c d", "c d e f", "e f g h", "f g h i"]
    # a text shorter than the window is one chunk
    assert chunker.segment(["a b"])[0] == ["a b"]
    # a blank text has no chunk
    assert chunker.segment([""])[0] == []
    # the parameters are part of the cache-facing version
    assert chunker.version != WordWindowsChunker(WordWindowsConfig(n_words=4, stride=4)).version
    assert chunker.slug == "words4"


def test_word_windows_cover_the_tail_without_a_fragment() -> None:
    """The last window ends at the last word (no one-word fragment)."""
    chunker = WordWindowsChunker(WordWindowsConfig(n_words=5, stride=3))
    windows = chunker.segment(["w1 w2 w3 w4 w5 w6 w7 w8"])[0]
    assert windows[-1].endswith("w8")
    assert all(len(window.split()) == 5 for window in windows)


def test_sentence_windows_group_sentences_with_a_step() -> None:
    """The overlapping sentence windows (exp02: two sentences, step one)."""
    chunker = SentenceWindowsChunker(SentenceWindowsConfig(n_sentences=2, step=1))
    windows = chunker.segment(["One. Two. Three. Four."])[0]
    assert windows == ["One. Two.", "Two. Three.", "Three. Four."]
    # a single sentence is one chunk
    assert chunker.segment(["Only one."])[0] == ["Only one."]
    assert chunker.slug == "sent2s1"
    assert (
        chunker.version
        != SentenceWindowsChunker(SentenceWindowsConfig(n_sentences=3, step=1)).version
    )


def test_semantic_chunker_cuts_where_the_adjacent_similarity_drops() -> None:
    """The boundary is cut between two atoms whose similarity is too low."""
    embedder = ScriptedEmbedder(
        {
            "First.": [1.0, 0.0],
            "Second.": [0.9, 0.1],
            "Third.": [0.0, 1.0],
        }
    )
    chunker = SemanticChunker(SemanticChunkerConfig(threshold=0.5), embedder)
    chunks = chunker.segment(["First. Second. Third."])[0]
    # cos(First, Second) = 0.9 (no cut), cos(Second, Third) = 0.1 (a cut)
    assert chunks == ["First. Second.", "Third."]
    # a lower threshold cuts nothing
    low = SemanticChunker(SemanticChunkerConfig(threshold=0.05), embedder)
    assert low.segment(["First. Second. Third."])[0] == ["First. Second. Third."]
    # the threshold is part of the cache-facing version
    assert chunker.version != low.version


def test_semantic_chunker_needs_the_embedder_port() -> None:
    """The chunker that needs embeddings refuses to be built without them."""
    with pytest.raises(ValueError, match="needs an Embedder port"):
        build_chunker("semantic", {"threshold": 0.5})
    chunker = build_chunker("semantic", {"threshold": 0.5}, embedder=FakeEmbedder(dimension=4))
    assert isinstance(chunker, SemanticChunker)
    # a chunker that does not need the port refuses one
    with pytest.raises(ValueError, match="does not take an embedder"):
        build_chunker("atoms", {}, embedder=FakeEmbedder(dimension=4))


def test_build_chunker_takes_its_parameters() -> None:
    """A chunker is built from its name and its parameters."""
    chunker = build_chunker("atoms", {"split_clauses": False})
    assert isinstance(chunker, AtomChunker)
    assert chunker.config.split_clauses is False
    # the parameters are part of the cache-facing version
    assert chunker.version != build_chunker("atoms").version
    assert chunker.version != build_chunker("atoms", {"split_clauses": True}).version
    # a chunker without parameters refuses any
    assert isinstance(build_chunker("full_text"), FullTextChunker)
    with pytest.raises(ValueError, match="takes no parameters"):
        build_chunker("full_text", {"split_clauses": True})
    with pytest.raises(ValueError, match="Unknown parameter"):
        build_chunker("atoms", {"not_a_parameter": 1})
    with pytest.raises(ValueError, match="Unknown chunker"):
        build_chunker("does-not-exist")


def test_atom_chunker_config_normalizes_its_sequences() -> None:
    """A config file carries lists; the version must not depend on the type."""
    from_list = AtomChunkerConfig(clause_separators=[",", ";"], abbreviations=["mr"])
    assert from_list.clause_separators == (",", ";")
    assert from_list.abbreviations == ("mr",)


def test_aggregation_parsing() -> None:
    """The rule spellings: the plain ones and the explicit top_k_mean k."""
    assert parse_aggregation("max") == parse_aggregation("max")
    assert parse_aggregation("max").spec == "max"
    assert parse_aggregation("mean").spec == "mean"
    top = parse_aggregation("top_k_mean:2")
    assert top.name == "top_k_mean" and top.k == 2
    assert top.spec == "top_k_mean:2" and top.slug == "top_k_mean_2"
    assert parse_aggregations(["max", "mean"]) == (
        parse_aggregation("max"),
        parse_aggregation("mean"),
    )
    with pytest.raises(ValueError, match="Unknown aggregation"):
        parse_aggregation("median")
    with pytest.raises(ValueError, match="explicit k"):
        parse_aggregation("top_k_mean")
    with pytest.raises(ValueError, match="integer k"):
        parse_aggregation("top_k_mean:two")
    with pytest.raises(ValueError, match="k >= 1"):
        parse_aggregation("top_k_mean:0")
    with pytest.raises(ValueError, match="takes no parameter"):
        parse_aggregation("max:2")


def test_stage2_aggregation_parsing() -> None:
    """The Stage-2 rules: their spellings and their parameter ranges."""
    decayed = parse_aggregation("decayed_top_k:3:0.5:0.25")
    assert decayed.name == "decayed_top_k"
    assert (decayed.k, decayed.decay, decayed.spread) == (3, 0.5, 0.25)
    assert decayed.spec == "decayed_top_k:3:0.5:0.25"
    blend = parse_aggregation("local_global:0.5:max")
    assert blend.name == "local_global" and blend.alpha == 0.5
    assert blend.local == parse_aggregation("max")
    assert blend.spec == "local_global:0.5:max"
    # the local rule may carry its own parameters
    nested = parse_aggregation("local_global:0.75:top_k_mean:3")
    assert nested.local == parse_aggregation("top_k_mean:3")
    assert nested.spec == "local_global:0.75:top_k_mean:3"
    peak = parse_aggregation("peak:3:0.05")
    assert (peak.k, peak.margin) == (3, 0.05)
    assert peak.spec == "peak:3:0.05"
    with pytest.raises(ValueError, match="needs k, decay and spread"):
        parse_aggregation("decayed_top_k:3:0.5")
    with pytest.raises(ValueError, match="decay of"):
        parse_aggregation("decayed_top_k:3:1.5:0.25")
    with pytest.raises(ValueError, match="spread penalty"):
        parse_aggregation("decayed_top_k:3:0.5:-1")
    with pytest.raises(ValueError, match="alpha of"):
        parse_aggregation("local_global:1.5:max")
    with pytest.raises(ValueError, match="needs alpha and a local rule"):
        parse_aggregation("local_global:0.5")
    with pytest.raises(ValueError, match="margin"):
        parse_aggregation("peak:3:-0.1")


def test_stage2_aggregations_fold_the_chunk_matrix_by_hand() -> None:
    """The Stage-2 folds: the decayed average, the blend and the peak."""
    matrix = np.array(
        [
            [0.2, 0.9],
            [0.8, 0.4],
            [0.5, 0.5],
        ]
    )
    groups = [[0, 1], [2]]
    # exp07: the decayed average of the two best, minus the spread penalty.
    # column 0: the two best are 0.8 and 0.2 -> decayed (0.8 + 0.5*0.2)/1.5,
    # the spread of the two is 0.3 -> the penalty is 0.25 * 0.3.
    decayed = aggregate_chunk_similarities(
        matrix, groups, parse_aggregation("decayed_top_k:2:0.5:0.25")
    )
    expected = (0.8 + 0.5 * 0.2) / 1.5 - 0.25 * 0.3
    assert decayed[0, 0] == pytest.approx(expected)
    # column 1: the two best are 0.9 and 0.4
    expected_one = (0.9 + 0.5 * 0.4) / 1.5 - 0.25 * 0.25
    assert decayed[0, 1] == pytest.approx(expected_one)
    # exp06: alpha * local + (1 - alpha) * full text
    global_matrix = np.array([[0.1, 0.1], [0.2, 0.2]])
    blend = aggregate_chunk_similarities(
        matrix,
        groups,
        parse_aggregation("local_global:0.5:max"),
        global_similarities=global_matrix,
    )
    assert blend[0].tolist() == pytest.approx([0.45, 0.5])
    assert blend[1].tolist() == pytest.approx([0.35, 0.35])
    # the blend needs the full-text matrix
    with pytest.raises(ValueError, match="needs the full-text score matrix"):
        aggregate_chunk_similarities(matrix, groups, parse_aggregation("local_global:0.5:max"))
    # exp08: the peak shortcut. The broad evidence of group 0, column 0, is
    # (0.8 + 0.2)/2 = 0.5 and the peak is 0.8 -> with a margin of 0.2 the peak
    # wins (0.8 - 0.5 = 0.3 >= 0.2); with a margin of 0.5 it does not.
    peak = aggregate_chunk_similarities(matrix, groups, parse_aggregation("peak:2:0.2"))
    assert peak[0, 0] == pytest.approx(0.8)
    strict = aggregate_chunk_similarities(matrix, groups, parse_aggregation("peak:2:0.5"))
    assert strict[0, 0] == pytest.approx(0.5)


def test_aggregation_folds_the_chunk_matrix_by_hand() -> None:
    """The fold: max, mean and the mean of the k best chunk matches."""
    # two groups of chunk rows: group 0 = rows 0,1; group 1 = row 2
    matrix = np.array(
        [
            [0.2, 0.9],
            [0.8, 0.4],
            [0.5, 0.5],
        ]
    )
    groups = [[0, 1], [2]]
    maximum = aggregate_chunk_similarities(matrix, groups, parse_aggregation("max"))
    assert maximum.tolist() == [[0.8, 0.9], [0.5, 0.5]]
    mean = aggregate_chunk_similarities(matrix, groups, parse_aggregation("mean"))
    assert mean[0].tolist() == pytest.approx([0.5, 0.65])
    assert mean[1].tolist() == pytest.approx([0.5, 0.5])
    top1 = aggregate_chunk_similarities(matrix, groups, parse_aggregation("top_k_mean:1"))
    assert top1[0].tolist() == pytest.approx([0.8, 0.9])
    top2 = aggregate_chunk_similarities(matrix, groups, parse_aggregation("top_k_mean:2"))
    assert top2[0].tolist() == pytest.approx([0.5, 0.65])  # both chunks averaged
    # a k larger than the group size falls back to the group mean
    top5 = aggregate_chunk_similarities(matrix, groups, parse_aggregation("top_k_mean:5"))
    assert top5[0].tolist() == pytest.approx([0.5, 0.65])
    with pytest.raises(ValueError, match="empty"):
        aggregate_chunk_similarities(matrix, [[]], parse_aggregation("max"))
    with pytest.raises(ValueError, match="outside the matrix"):
        aggregate_chunk_similarities(matrix, [[0, 9]], parse_aggregation("max"))


def test_protocols_are_runtime_checkable() -> None:
    """The ports are protocols: a duck-typed implementation conforms."""

    class _FakeFeatures:
        @property
        def feature_names(self) -> tuple[str, ...]:
            return ("cos",)

        @property
        def all_feature_names(self) -> tuple[str, ...]:
            return ("cos",)

        @property
        def selected_indices(self) -> tuple[int, ...]:
            return (0,)

        def extract(self, pairs: list[TextPair]) -> np.ndarray:
            return np.zeros((len(pairs), 1), dtype=np.float32)

        def extract_all(self, pairs: list[TextPair]) -> np.ndarray:
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


def test_the_only_heavy_import_is_the_combiner_scikit_learn() -> None:
    """The package's one exception (TEXT_PROCESSING_PLAN.md section 2).

    The combiner may import scikit-learn (a core dependency of the main
    pipeline: no model download, no network, no I/O). Nothing else in the
    package may import a third-party library beyond numpy.
    """
    allowed = {"numpy", "sklearn"}
    stdlib = set(sys.stdlib_module_names)
    for module_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        for module in sorted(imported):
            if module in stdlib or module in allowed or module.startswith("cluster_rag"):
                continue
            raise AssertionError(
                f"{module_path.name} imports {module}: the track's package may only "
                "import numpy, scikit-learn (the combiner) and cluster_rag"
            )
    # the exception is used where it is documented
    assert "sklearn" in (PACKAGE_ROOT / "combining" / "logistic.py").read_text(encoding="utf-8")
    assert "sklearn" in (PACKAGE_ROOT / "combining" / "histgb.py").read_text(encoding="utf-8")
