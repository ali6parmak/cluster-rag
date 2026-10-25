"""Feature pack v0: one row of numbers per (paragraph, question) pair.

TEXT_PROCESSING_PLAN.md section 3 (Stage 3.1) replaces the hand-written
scoring rules with a **table of numbers per pair**: the combiner then learns
how to weigh them. The v0 feature list is the plan's section-2 list (minus the
length feature, which the reviewer's decision leaves out because length is a
document proxy on this data — NU's paragraphs are long, AI's are short, so a
length feature can memorize the document instead of the topic):

===============================  ==================================================
feature                          what it measures
===============================  ==================================================
``cosine_full``                  the full-text cosine (the Stage-0 baseline's score)
``chunk_cosine_best``            the best paragraph-chunk cosine
``chunk_cosine_top2_mean``       the mean of the two best chunk cosines
``chunk_cosine_topK_mean``       the mean of the K best chunk cosines (K = ``top_k``)
``chunk_cosine_spread``          the standard deviation of the K best chunk cosines
``chunk_peak_margin``            the best chunk cosine minus the K-best mean
``lex_question_coverage``        the question stem's soft coverage by the paragraph
``lex_paragraph_coverage``       the paragraph's soft coverage by the question stem
``lex_char_ngram_cosine``        the char-3-gram cosine of the two texts
``qpart_options_coverage``       the answer options' soft coverage by the paragraph
``qtype_has_options``            1 when the question carries an options block
``qtype_has_no_data_option``     1 when an option says "no data / n/a"
``qtype_yes_no``                 1 when the question is a yes/no question
``qtype_frequency``              1 when the question asks about frequency
``qtype_option_count``           the number of answer options
===============================  ==================================================

**Pair-local by construction.** The pack sees one pair at a time and derives
everything from that pair's two texts: no corpus statistic, no question-level
prior, no document identity, no label. (The Stage-3 leakage rules of the
plan: 186/294 question numbers are judged in more than one document and 101
are positive in one document and negative in another, so a question-level
statistic would leak across the document holdout.)

**The question's two parts.** The plan's Stage 1.1b: the answer options are
scales ("Very often", "Yes"), i.e. a *type* signal, not topical content. The
pack therefore splits the question into its **stem** (the topical part) and its
**options** (the type part), uses the stem for the lexical overlap, and turns
the options into the type one-hots.

**The feature groups (the CS-6a ablation's unit).** The pack's features fall
into four **groups**, each one a block of the same kind of signal:

=====================  =====================================================
group                  what it holds
=====================  =====================================================
``embedding``          the full-text cosine + the five chunk features
``lexical``            the soft lexical overlap (both ways) + the char n-grams
``question_part``      the answer options' soft coverage
``question_type``      the five question-type one-hots
=====================  =====================================================

The group of a feature is decided by its **name prefix**, so the
K-dependent name needs no special case, and the four groups partition the
pack exactly (a test enforces it). The extractor keeps only the groups it is
given (``groups``), which is how the plan's feature-group ablation (Stage
3.1b) measures which block carries the gain: the pack itself never changes,
only the columns the combiner sees.

**Where the embeddings come from.** The pack takes the ``Embedder`` **port**
(``cluster_rag.embed.base``) through :class:`FeatureEmbedders` — the same
precedent as the semantic chunker: the package never imports a model client,
the CLI wires the cached backends in. Three embedding sides are needed and each
one has its own cache-version slot, so the ports are separate:

- ``paragraph``: the whole paragraph (the baseline arm's cache slot);
- ``question``: the whole question (the baseline arm's question slot);
- ``chunks``: the paragraph's chunks (the chunker's cache slot).

Pure core: numpy + the standard library + ``cluster_rag.embed.base`` only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.embed.base import Embedder
from cluster_rag.text_processors.base import Chunker, TextPair

# The v0 feature names, in column order. The K of the top-K features is
# configurable, so the two K-dependent names are built per instance (see
# ``PairFeatureExtractor.feature_names``); the tuple below is the default
# (K = 3) and is what the artifact records.
FEATURE_NAMES: tuple[str, ...] = (
    "cosine_full",
    "chunk_cosine_best",
    "chunk_cosine_top2_mean",
    "chunk_cosine_top3_mean",
    "chunk_cosine_spread",
    "chunk_peak_margin",
    "lex_question_coverage",
    "lex_paragraph_coverage",
    "lex_char_ngram_cosine",
    "qpart_options_coverage",
    "qtype_has_options",
    "qtype_has_no_data_option",
    "qtype_yes_no",
    "qtype_frequency",
    "qtype_option_count",
)

# The feature groups of the v0 pack (the Stage-3.1b ablation's unit). The
# group of a feature is decided by its **name prefix**, so the K-dependent
# name needs no special case; the four groups partition the pack exactly (a
# test enforces it).
FEATURE_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "embedding": ("cosine_full", "chunk_"),
    "lexical": ("lex_",),
    "question_part": ("qpart_",),
    "question_type": ("qtype_",),
}

# The group names, in the canonical order.
FEATURE_GROUPS: tuple[str, ...] = tuple(FEATURE_GROUP_PREFIXES)


def feature_group(name: str) -> str | None:
    """The group a feature belongs to (``None`` when it belongs to none).

    Args:
        name: The feature's name (e.g. ``"chunk_cosine_best"``).

    Returns:
        The group's name, or ``None`` when no group claims the feature.
    """
    for group, prefixes in FEATURE_GROUP_PREFIXES.items():
        if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
            return group
    return None


# The stopwords of the lexical coverage: the coverage of a *stopword* would
# be trivially high (every English text contains "the"), so the coverage
# features use the content tokens only. The list is a property of the pack
# (recorded in the artifact), not a threshold fitted on the data.
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "at",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "their",
        "they",
        "them",
        "we",
        "us",
        "our",
        "you",
        "your",
        "which",
        "what",
        "who",
        "whom",
        "whose",
        "how",
        "when",
        "where",
        "why",
        "not",
        "no",
        "yes",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "any",
        "all",
        "each",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "then",
        "so",
        "up",
        "out",
        "about",
        "into",
        "over",
        "under",
        "also",
        "very",
        "s",
        "t",
    ]
)

# The words that mark an answer option as "no data" (the plan's Stage 1.1b
# question-type feature: 261/294 questions carry such an option).
NO_DATA_PATTERNS: tuple[str, ...] = (
    "no data",
    "no information",
    "no answer",
    "no response",
    "no relevant data",
    "not applicable",
    "not available",
    "not reported",
    "not specified",
    "not collected",
    "n/a",
    "na",
    "none",
    "unknown",
)

# The yes/no option family (the plan's question-type feature).
YES_NO_OPTIONS: frozenset[str] = frozenset(
    {"yes", "no", "true", "false", "yes/no", "yes, no", "no, yes"}
)

# The frequency-scale option family and the stem's frequency words.
FREQUENCY_OPTION_TERMS: tuple[str, ...] = (
    "very often",
    "very frequently",
    "frequently",
    "often",
    "sometimes",
    "occasionally",
    "rarely",
    "seldom",
    "never",
    "always",
    "not at all",
    "not never",
    "very rarely",
    "regularly",
    "never/rarely/sometimes/often/always",
)

FREQUENCY_STEM_TERMS: tuple[str, ...] = (
    "how often",
    "how frequently",
    "how frequently do",
    "frequency",
    "regularity",
)

# The auxiliary verbs that start a yes/no question.
YES_NO_STEM_PREFIXES: tuple[str, ...] = (
    "did ",
    "does ",
    "do ",
    "is ",
    "are ",
    "was ",
    "were ",
    "has ",
    "have ",
    "had ",
    "can ",
    "could ",
    "will ",
    "would ",
    "should ",
    "may ",
    "might ",
)

# The "Question:" label of the OMCT question texts (the same pattern the
# pipeline's question transform strips).
QUESTION_PREFIX_PATTERN = re.compile(r"^\s*Question:\s*")

# The character n-gram size of the char-level features.
NGRAM_SIZE = 3

# The token pattern of the lexical features.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# The option bullets (the OMCT answer choices are "- option" lines).
_OPTION_BULLETS = "-*•–—"


@dataclass(frozen=True)
class QuestionParts:
    """A question split into its topical **stem** and its typed **options**.

    The stem is the question text before the options block (the "Question:"
    label removed); the options are the bullet lines of the block. A question
    without an options block has an empty ``options`` tuple.
    """

    stem: str
    options: tuple[str, ...] = ()

    @property
    def has_options(self) -> bool:
        """Whether the question carries an options block."""
        return bool(self.options)


def _option_lines(block: str) -> tuple[str, ...]:
    """The option texts of a block (the non-empty lines, bullets removed)."""
    options: list[str] = []
    for line in block.splitlines():
        cleaned = line.strip().lstrip(_OPTION_BULLETS).strip()
        if cleaned:
            options.append(cleaned)
    return tuple(options)


def split_question_parts(
    text: str, markers: Sequence[str] = ("Answer choices:", "Notes:")
) -> QuestionParts:
    """Split a question into its stem and its answer options.

    The options block starts at the first marker that is followed by at least
    one bulleted line; the stem is everything before it (the "Question:" label
    removed). A later marker (e.g. "Notes:") ends the block.

    Example: ``"Question: How often?\\n\\nAnswer choices:\\n- Often\\n- No data"``
    gives the stem ``"How often?"`` and the options ``("Often", "No data")``.

    Args:
        text: The question text (as it is in the pairwise file).
        markers: The markers of the options block and of the trailing blocks.

    Returns:
        The :class:`QuestionParts` of the question.
    """
    stripped = text.strip()
    stem = stripped
    block = ""
    for marker in markers:
        position = stripped.find(marker)
        if position <= 0:
            continue
        candidate = stripped[position + len(marker) :]
        if _option_lines(candidate):
            stem, block = stripped[:position], candidate
            break
    if not block:
        for marker in markers:
            position = stripped.find(marker)
            if position > 0:
                stem, block = stripped[:position], stripped[position + len(marker) :]
                break
    for marker in markers:
        position = block.find(marker)
        if position > 0:
            block = block[:position]
    stem = QUESTION_PREFIX_PATTERN.sub("", stem, count=1).strip()
    return QuestionParts(stem=stem, options=_option_lines(block))


def tokenize(text: str) -> list[str]:
    """The lower-cased alphanumeric tokens of a text."""
    return TOKEN_PATTERN.findall(text.lower())


def content_tokens(text: str, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> list[str]:
    """The tokens of a text without the stopwords (the coverage's unit)."""
    return [token for token in tokenize(text) if token not in stopwords and len(token) > 1]


def char_ngrams(text: str, size: int = NGRAM_SIZE) -> frozenset[str]:
    """The character n-grams of a text (space-padded, so the edges count)."""
    padded = f" {text.lower()} "
    if len(padded) < size:
        return frozenset({padded})
    return frozenset(padded[index : index + size] for index in range(len(padded) - size + 1))


def _token_ngrams(token: str, size: int = NGRAM_SIZE) -> frozenset[str]:
    """The character n-grams of one token (space-padded like a text)."""
    return char_ngrams(token, size)


def _soft_coverage(token_ngrams: Sequence[frozenset[str]], other_ngrams: frozenset[str]) -> float:
    """The share of the tokens' n-grams that the other text contains.

    A token counts as covered in proportion to how many of its character
    n-grams the other text has: "frequently" is half covered by a text that
    says "frequency" (the shared 3-grams), fully covered by one that says
    "frequently". The value is in [0, 1] and is 0 when there is no token.
    """
    if not token_ngrams:
        return 0.0
    total = 0.0
    for grams in token_ngrams:
        if not grams:
            continue
        hits = sum(1 for gram in grams if gram in other_ngrams)
        total += hits / len(grams)
    return total / len(token_ngrams)


def _char_cosine(
    counts: Mapping[str, int], other_counts: Mapping[str, int], norm: float, other_norm: float
) -> float:
    """The cosine of two char-n-gram count vectors (the exact, sparse dot)."""
    if norm <= 0.0 or other_norm <= 0.0:
        return 0.0
    if len(counts) > len(other_counts):
        counts, other_counts = other_counts, counts
    dot = 0
    for gram, count in counts.items():
        other = other_counts.get(gram)
        if other:
            dot += count * other
    return float(dot / (norm * other_norm))


def _has_no_data_option(options: Sequence[str]) -> bool:
    """Whether one of the options says "no data / n/a"."""
    for option in options:
        lowered = option.lower()
        for pattern in NO_DATA_PATTERNS:
            if re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", lowered):
                return True
    return False


def _is_yes_no(parts: QuestionParts) -> bool:
    """Whether the question is a yes/no question."""
    if parts.options:
        lowered = {option.lower().strip(" .") for option in parts.options}
        if lowered and lowered <= YES_NO_OPTIONS:
            return True
    stem = parts.stem.lower()
    return any(stem.startswith(prefix) for prefix in YES_NO_STEM_PREFIXES)


def _is_frequency(parts: QuestionParts) -> bool:
    """Whether the question asks about frequency."""
    stem = parts.stem.lower()
    if any(term in stem for term in FREQUENCY_STEM_TERMS):
        return True
    return any(
        term in option.lower() for option in parts.options for term in FREQUENCY_OPTION_TERMS
    )


@dataclass(frozen=True)
class FeatureEmbedders:
    """The embedding sides the feature pack needs (the ``Embedder`` ports).

    The three sides have three different cache-version slots, so they are three
    ports and not one: the whole paragraph lives in the baseline arm's
    paragraph slot, the whole question in the baseline arm's question slot
    (the question transform's version) and the chunks in the chunker's slot.
    The CLI wires the cached backends in; the pack itself never imports a
    model client.

    Attributes:
        paragraph: The whole paragraph (encoded with the ``document`` kind).
        question: The whole question (encoded with the ``query`` kind).
        chunks: The paragraph's chunks (encoded with the ``document`` kind).
        question_chunks: The question's chunks, when the question side is
            chunked too (encoded with the ``query`` kind); ``None`` means the
            question side is whole, which is the plan's Stage-2 verdict.
    """

    paragraph: Embedder
    question: Embedder
    chunks: Embedder
    question_chunks: Embedder | None = None


@dataclass
class _TextFacts:
    """The per-text facts the pack derives once and re-uses for every pair."""

    ngrams: frozenset[str]
    gram_counts: dict[str, int]
    gram_norm: float
    token_ngrams: list[frozenset[str]]


class PairFeatureExtractor:
    """The v0 feature pack: a pair in, a row of numbers out.

    The extractor is **label-free and pair-local**: it derives every feature
    from the pair's own two texts, keeps no corpus-level statistic and never
    sees a label or a document name. The embeddings it needs come through the
    :class:`FeatureEmbedders` ports (the CLI wires the cached backends in).

    Args:
        embedders: The embedding sides (the ports).
        paragraph_chunker: The chunker of the paragraph side (the Stage-2
            verdict: chunk the paragraph, keep the question whole).
        top_k: The K of the top-K chunk similarities (the spread and the peak
            margin are measured over the same K best chunk cosines).
        question_embedding_texts: The text that is embedded for a question,
            keyed by the question's own text. The driver fills it with the
            question transform's output (e.g. the "Answer choices:" block
            stripped), so the pack's embedding features match the baseline
            arm's cosine exactly while the lexical and type features still
            see the question's parts.
        groups: The **feature groups** to keep (the Stage-3.1b ablation's
            unit). ``None`` keeps every group, i.e. the whole v0 pack. The
            selected features keep the pack's canonical order, so a group's
            columns are the same whatever else is selected.
        stopwords: The stopwords of the lexical coverage.

    Raises:
        ValueError: If ``top_k`` is not positive, if a group name is unknown,
            if a group is repeated, or if the selection is empty.
    """

    def __init__(
        self,
        embedders: FeatureEmbedders,
        *,
        paragraph_chunker: Chunker,
        top_k: int = 3,
        question_embedding_texts: Mapping[str, str] | None = None,
        groups: Sequence[str] | None = None,
        stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.embedders = embedders
        self.paragraph_chunker = paragraph_chunker
        self.top_k = top_k
        self.question_embedding_texts = dict(question_embedding_texts or {})
        self.stopwords = frozenset(stopwords)
        self.groups = self._checked_groups(groups)
        self._canonical_names = self._canonical_feature_names()
        self._selected = tuple(
            index
            for index, name in enumerate(self._canonical_names)
            if feature_group(name) in self.groups
        )
        if not self._selected:
            raise ValueError(
                f"the feature groups {list(self.groups)} select no feature of the v0 pack"
            )

    @staticmethod
    def _checked_groups(groups: Sequence[str] | None) -> tuple[str, ...]:
        """The group selection, checked (known names, no repeats, not empty)."""
        if groups is None:
            return FEATURE_GROUPS
        selected = tuple(groups)
        if not selected:
            raise ValueError("the feature groups must name at least one group")
        unknown = [group for group in selected if group not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown feature group(s) {unknown}; expected values in {list(FEATURE_GROUPS)}"
            )
        if len(set(selected)) != len(selected):
            raise ValueError(f"the feature groups must not repeat a group: {list(selected)}")
        return selected

    def _canonical_feature_names(self) -> tuple[str, ...]:
        """The pack's feature names, in column order (the K-dependent ones included)."""
        return (
            "cosine_full",
            "chunk_cosine_best",
            "chunk_cosine_top2_mean",
            f"chunk_cosine_top{self.top_k}_mean",
            "chunk_cosine_spread",
            "chunk_peak_margin",
            "lex_question_coverage",
            "lex_paragraph_coverage",
            "lex_char_ngram_cosine",
            "qpart_options_coverage",
            "qtype_has_options",
            "qtype_has_no_data_option",
            "qtype_yes_no",
            "qtype_frequency",
            "qtype_option_count",
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The selected feature names, in column order (the K-dependent ones included)."""
        return tuple(self._canonical_names[index] for index in self._selected)

    @property
    def dropped_feature_names(self) -> tuple[str, ...]:
        """The canonical features the current group selection leaves out."""
        return tuple(
            name for index, name in enumerate(self._canonical_names) if index not in self._selected
        )

    def describe(self) -> dict[str, Any]:
        """A JSON-friendly description of the pack (recorded in the artifact)."""
        return {
            "name": "v0",
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "groups": list(self.groups),
            "dropped_features": list(self.dropped_feature_names),
            "paragraph_chunker": self.paragraph_chunker.version,
            "top_k": self.top_k,
            "stopwords": sorted(self.stopwords),
            "embedding_sides": {
                side: {
                    "cache_hits": getattr(embedder, "cache_hits", None),
                    "cache_misses": getattr(embedder, "cache_misses", None),
                }
                for side, embedder in (
                    ("paragraph", self.embedders.paragraph),
                    ("question", self.embedders.question),
                    ("chunks", self.embedders.chunks),
                    ("question_chunks", self.embedders.question_chunks),
                )
                if embedder is not None
            },
            "note": (
                "the pack is label-free and pair-local: no corpus-level question "
                "statistic, no document-identity feature, and no length feature "
                "(length is a document proxy on this data; the lengths are "
                "recorded in the artifact instead). 'groups' is the feature-group "
                "selection of the Stage-3.1b ablation: the pack computes every "
                "feature and the combiner sees only the selected groups' columns."
            ),
        }

    # -- the text side -----------------------------------------------------------------

    def _facts(self, text: str) -> _TextFacts:
        """The per-text facts (the n-grams and the content tokens' n-grams)."""
        counts: dict[str, int] = {}
        for gram in char_ngrams(text):
            counts[gram] = counts.get(gram, 0) + 1
        norm = float(sum(count * count for count in counts.values())) ** 0.5
        tokens = content_tokens(text, self.stopwords)
        return _TextFacts(
            ngrams=frozenset(counts),
            gram_counts=counts,
            gram_norm=norm,
            token_ngrams=[_token_ngrams(token) for token in tokens],
        )

    def _question_parts(self, text: str) -> QuestionParts:
        """The question's parts (the stem and the options)."""
        return split_question_parts(text)

    # -- the extraction ----------------------------------------------------------------

    def extract(self, pairs: Sequence[TextPair]) -> np.ndarray:
        """The feature matrix of the pairs, shape ``(n_pairs, n_features)``.

        The embeddings are done in **one batch per side**: the unique
        paragraphs, the unique questions and the unique chunks are encoded
        once each, and every pair's features are derived from them. That is a
        memoisation of a deterministic function of a text (the same text always
        embeds to the same vector), so the features stay pair-local.

        Args:
            pairs: The pairs to featurize (the text only — never a label).

        Returns:
            The float32 feature matrix, one row per pair.
        """
        items = list(pairs)
        matrix = np.zeros((len(items), len(self.feature_names)), dtype=np.float32)
        if not items:
            return matrix

        paragraph_texts = list(dict.fromkeys(pair.paragraph_text for pair in items))
        question_texts = list(dict.fromkeys(pair.question_text for pair in items))
        paragraph_index = {text: position for position, text in enumerate(paragraph_texts)}
        question_index = {text: position for position, text in enumerate(question_texts)}

        # The embeddings: one batch per side.
        paragraph_vectors = np.asarray(
            self.embedders.paragraph.encode(paragraph_texts, "document"), dtype=np.float32
        )
        embedding_texts = [self.question_embedding_texts.get(text, text) for text in question_texts]
        question_vectors = np.asarray(
            self.embedders.question.encode(embedding_texts, "query"), dtype=np.float32
        )
        chunk_lists = self.paragraph_chunker.segment(paragraph_texts)
        chunk_texts: list[str] = []
        chunk_rows: list[tuple[int, ...]] = []
        for chunk_list in chunk_lists:
            start = len(chunk_texts)
            chunk_texts.extend(chunk_list)
            chunk_rows.append(tuple(range(start, len(chunk_texts))))
        chunk_vectors = (
            np.asarray(self.embedders.chunks.encode(chunk_texts, "document"), dtype=np.float32)
            if chunk_texts
            else np.zeros((0, paragraph_vectors.shape[1]), dtype=np.float32)
        )

        # The per-text facts (the n-grams and the content tokens). The question
        # side is decomposed into its parts once per question: the stem carries
        # the topical content (the lexical features) and the options carry the
        # type signal.
        paragraph_facts = [self._facts(text) for text in paragraph_texts]
        question_parts = [self._question_parts(text) for text in question_texts]
        question_stem_facts = [self._facts(parts.stem) for parts in question_parts]
        question_option_facts = [
            self._facts(" ".join(parts.options)) if parts.options else None
            for parts in question_parts
        ]

        top_k = self.top_k
        selected = list(self._selected)
        for position, pair in enumerate(items):
            row = paragraph_index[pair.paragraph_text]
            column = question_index[pair.question_text]
            paragraph_vector = paragraph_vectors[row]
            question_vector = question_vectors[column]
            # the canonical row: every feature is computed, and the combiner
            # sees only the selected groups' columns
            values = np.zeros(len(self._canonical_names), dtype=np.float32)

            # 1. the global (full-text) cosine
            values[0] = float(paragraph_vector @ question_vector)

            # 2. the chunk-level cosines
            rows = chunk_rows[row]
            if rows:
                similarities = chunk_vectors[list(rows)] @ question_vector
                best = np.sort(similarities)[::-1]
                values[1] = float(best[0])
                values[2] = float(best[: min(2, best.size)].mean())
                top = best[: min(top_k, best.size)]
                values[3] = float(top.mean())
                values[4] = float(top.std()) if top.size > 1 else 0.0
                values[5] = float(best[0] - top.mean())

            # 3. the lexical overlap
            paragraph_fact = paragraph_facts[row]
            stem_fact = question_stem_facts[column]
            parts = question_parts[column]
            values[6] = _soft_coverage(stem_fact.token_ngrams, paragraph_fact.ngrams)
            values[7] = _soft_coverage(paragraph_fact.token_ngrams, stem_fact.ngrams)
            values[8] = _char_cosine(
                paragraph_fact.gram_counts,
                stem_fact.gram_counts,
                paragraph_fact.gram_norm,
                stem_fact.gram_norm,
            )

            # 4. the question parts and the question type
            options_fact = question_option_facts[column]
            values[9] = (
                _soft_coverage(options_fact.token_ngrams, paragraph_fact.ngrams)
                if options_fact is not None
                else 0.0
            )
            values[10] = 1.0 if parts.has_options else 0.0
            values[11] = 1.0 if _has_no_data_option(parts.options) else 0.0
            values[12] = 1.0 if _is_yes_no(parts) else 0.0
            values[13] = 1.0 if _is_frequency(parts) else 0.0
            values[14] = float(len(parts.options))
            # the combiner sees the selected groups' columns only
            matrix[position] = values[selected]
        return matrix


def build_feature_extractor(
    embedders: FeatureEmbedders,
    *,
    paragraph_chunker: Chunker,
    top_k: int = 3,
    question_embedding_texts: Mapping[str, str] | None = None,
    groups: Sequence[str] | None = None,
) -> PairFeatureExtractor:
    """Build the v0 feature pack (the config-facing entry point)."""
    return PairFeatureExtractor(
        embedders,
        paragraph_chunker=paragraph_chunker,
        top_k=top_k,
        question_embedding_texts=question_embedding_texts,
        groups=groups,
    )
