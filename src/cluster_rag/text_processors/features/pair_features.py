"""Feature pack: one row of numbers per (paragraph, question) pair.

TEXT_PROCESSING_PLAN.md section 3 (Stage 3.1) replaces the hand-written
scoring rules with a **table of numbers per pair**: the combiner then learns
how to weigh them. The v0 feature list is the plan's section-2 list (minus the
length feature, which the reviewer's decision leaves out because length is a
document proxy on this data — NU's paragraphs are long, AI's are short, so a
length feature can memorize the document instead of the topic):

=============================  ==================================================
feature                         what it measures
=============================  ==================================================
``cosine_full``                 the full-text cosine (the Stage-0 baseline's score)
``chunk_cosine_best``           the best paragraph-chunk cosine
``chunk_cosine_top2_mean``      the mean of the two best chunk cosines
``chunk_cosine_topK_mean``      the mean of the K best chunk cosines (K = ``top_k``)
``chunk_cosine_spread``         the standard deviation of the K best chunk cosines
``chunk_peak_margin``           the best chunk cosine minus the K-best mean
``lex_question_coverage``       the question stem's soft coverage by the paragraph
``lex_paragraph_coverage``      the paragraph's soft coverage by the question stem
``lex_char_ngram_cosine``       the char-3-gram cosine of the two texts
``qpart_options_coverage``      the answer options' soft coverage by the paragraph
``qtype_has_options``           1 when the question carries an options block
``qtype_has_no_data_option``    1 when an option says "no data / n/a"
``qtype_yes_no``                1 when the question is a yes/no question
``qtype_frequency``             1 when the question asks about frequency
``qtype_option_count``          the number of answer options
=============================  ==================================================

The Stage-3.3 **chunk profile** (the plan's 3.3, exp17/15/16's lineage) adds
a fifth block. It says **where** the paragraph and the question match, not
how similar they are on average:

====================================  ==========================================
feature                                what it measures
====================================  ==========================================
``cprof_sent_top1..topK``              the ranked cross-chunk match profile, sentence granularity
``cprof_ngram_top1..topK``             the ranked cross-chunk match profile, word-piece granularity
``cprof_align_sent``                   1 when the best alignment is a sentence match
``cprof_align_ngram``                  1 when the best alignment is a word-piece match
``cprof_align_lexical``                1 when the two aligned units share a content word
``cprof_align_semantic_only``          1 when they share none
====================================  ==========================================

**The profile.** exp17 cuts **both** sides of a pair into small pieces — the
sentences and the word pieces inside the sentences — computes every
piece×piece cosine, sorts them and keeps the top ranks. So the profile is
the *shape* of the local matches: one strong peak (a paragraph that matches
the question in one place) or several (a paragraph that matches it in many
places). The alignment kinds say which granularity produced the best match
and whether it was a lexical match (the two units share a content word) or a
semantic-only one.

**Pair-local by construction.** The pack sees one pair at a time and derives
everything from that pair's two texts: no corpus statistic, no question-level
prior, no document identity, no label. (The Stage-3 leakage rules of the
plan: 186/294 question numbers are judged in more than one document and 101
are positive in one document and negative in another, so a question-level
statistic would leak across the document holdout.) exp16's ``log(1+df)`` of
the matched phrase — a statistic of the *labeled corpus* — is therefore **not**
a feature here; the reviewer's decision (2026-10-26) was to drop it and to
keep the pair-local alignment kinds instead.

**The question's two parts.** The plan's Stage 1.1b: the answer options are
scales ("Very often", "Yes"), i.e. a *type* signal, not topical content. The
pack therefore splits the question into its **stem** (the topical part) and its
**options** (the type part), uses the stem for the lexical overlap, and turns
the options into the type one-hots. The profile's question side follows the
embedding side: the units are cut from the **transformed** question (the
"Answer choices:" block stripped), so the scale words stay out of the
cross-chunk matches.

**The feature groups (the CS-6a ablation's unit).** The pack's features fall
into five **groups**, each one a block of the same kind of signal:

=====================  =====================================================
group                  what it holds
=====================  =====================================================
``embedding``          the full-text cosine + the five chunk features
``lexical``            the soft lexical overlap (both ways) + the char n-grams
``question_part``      the answer options' soft coverage
``question_type``      the five question-type one-hots
``chunk_profile``      the ranked cross-chunk profile + the alignment kinds
=====================  =====================================================

The group of a feature is decided by its **name prefix**, so the
K-dependent names need no special case, and the five groups partition the
pack exactly (a test enforces it). The extractor keeps only the groups it is
given (``groups``), which is how the plan's feature-group ablation (Stage
3.1b) measures which block carries the gain: the pack itself never changes,
only the columns the combiner sees. The fifth group was added by the Stage-3.3
change set, so the four earlier groups keep their exact meaning (and their
runs keep their exact feature sets).

**Where the embeddings come from.** The pack takes the ``Embedder`` **port**
(``cluster_rag.embed.base``) through :class:`FeatureEmbedders` — the same
precedent as the semantic chunker: the package never imports a model client,
the CLI wires the cached backends in. Every embedding side has its own
cache-version slot, so the ports are separate:

- ``paragraph``: the whole paragraph (the baseline arm's cache slot);
- ``question``: the whole question (the baseline arm's question slot);
- ``chunks``: the paragraph's chunks (the chunker's cache slot);
- ``paragraph_sentences`` / ``question_sentences``: the profile's sentence units;
- ``paragraph_ngrams`` / ``question_ngrams``: the profile's word-piece units.

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

# The v0 feature names, in column order (K = 3). The K of the top-K features
# is configurable, so the two K-dependent names are built per instance (see
# ``PairFeatureExtractor.feature_names``); the tuple below is the default
# (K = 3) and is what the artifact records. This is the **Stage-3.1**
# feature set — the one the CS-6a ablation's arms describe.
V0_FEATURE_NAMES: tuple[str, ...] = (
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

# The chunk profile's feature names, in column order (the default ranks).
PROFILE_FEATURE_NAMES: tuple[str, ...] = (
    "cprof_sent_top1",
    "cprof_sent_top2",
    "cprof_sent_top3",
    "cprof_ngram_top1",
    "cprof_ngram_top2",
    "cprof_ngram_top3",
    "cprof_align_sent",
    "cprof_align_ngram",
    "cprof_align_lexical",
    "cprof_align_semantic_only",
)

# The pack's full feature list, in column order (the defaults: K = 3 chunk
# similarities and 3 profile ranks per granularity).
FEATURE_NAMES: tuple[str, ...] = V0_FEATURE_NAMES + PROFILE_FEATURE_NAMES

# The feature groups of the pack (the Stage-3.1b ablation's unit). The group
# of a feature is decided by its **name prefix**, so the K-dependent names
# need no special case; the five groups partition the pack exactly (a test
# enforces it).
FEATURE_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "embedding": ("cosine_full", "chunk_"),
    "lexical": ("lex_",),
    "question_part": ("qpart_",),
    "question_type": ("qtype_",),
    "chunk_profile": ("cprof_",),
}

# The group names, in the canonical order.
FEATURE_GROUPS: tuple[str, ...] = tuple(FEATURE_GROUP_PREFIXES)

# The **default** selection: the v0 pack's four groups (the Stage-3.1
# behaviour). The Stage-3.3 chunk profile is opt-in, so the default keeps
# the earlier arms' feature sets exactly; the whole pack is every group.
DEFAULT_FEATURE_GROUPS: tuple[str, ...] = (
    "embedding",
    "lexical",
    "question_part",
    "question_type",
)


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


def soft_coverage(token_ngrams: Sequence[frozenset[str]], other_ngrams: frozenset[str]) -> float:
    """The share of the tokens' n-grams that the other text contains.

    A token counts as covered in proportion to how many of its character
    n-grams the other text has: "frequently" is half covered by a text that
    says "frequency" (the shared 3-grams), fully covered by one that says
    "frequently". The value is in [0, 1] and is 0 when there is no token.

    The helper is public because the UPR pack (CS-8d) measures the same soft
    coverage between a text and a **label name** — the same statistic, a
    different pair.
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


def char_cosine(
    counts: Mapping[str, int], other_counts: Mapping[str, int], norm: float, other_norm: float
) -> float:
    """The cosine of two char-n-gram count vectors (the exact, sparse dot).

    Public because the UPR pack (CS-8d) measures the same char-n-gram cosine
    between a text and a label name.
    """
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

    Every side has its own cache-version slot, so they are separate ports and
    not one: the whole paragraph lives in the baseline arm's paragraph slot,
    the whole question in the baseline arm's question slot (the question
    transform's version), the chunks in the chunker's slot and the profile's
    four unit sides in their own slots. The CLI wires the cached backends in;
    the pack itself never imports a model client.

    Attributes:
        paragraph: The whole paragraph (encoded with the ``document`` kind).
        question: The whole question (encoded with the ``query`` kind).
        chunks: The paragraph's chunks (encoded with the ``document`` kind).
        question_chunks: The question's chunks, when the question side is
            chunked too (encoded with the ``query`` kind); ``None`` means the
            question side is whole, which is the plan's Stage-2 verdict.
        paragraph_sentences: The profile's sentence units of the paragraph
            side (the ``document`` kind); ``None`` disables the profile.
        question_sentences: The profile's sentence units of the question side
            (the ``query`` kind).
        paragraph_ngrams: The profile's word-piece units of the paragraph
            side (the ``document`` kind).
        question_ngrams: The profile's word-piece units of the question side
            (the ``query`` kind).
    """

    paragraph: Embedder
    question: Embedder
    chunks: Embedder
    question_chunks: Embedder | None = None
    paragraph_sentences: Embedder | None = None
    question_sentences: Embedder | None = None
    paragraph_ngrams: Embedder | None = None
    question_ngrams: Embedder | None = None


@dataclass(frozen=True)
class ProfileUnits:
    """The chunk profile's unit splitters (the two granularities).

    The same splitters are used on both sides of a pair; only the embedding
    side differs (the paragraph's units are documents, the question's are
    queries). The CLI builds them and wires the matching cached backends in.
    """

    sentences: Chunker
    ngrams: Chunker


@dataclass
class _TextFacts:
    """The per-text facts the pack derives once and re-uses for every pair."""

    ngrams: frozenset[str]
    gram_counts: dict[str, int]
    gram_norm: float
    token_ngrams: list[frozenset[str]]


@dataclass(frozen=True)
class _UnitSide:
    """One side's profile units: the unit texts and their vectors.

    ``rows`` gives the vector rows of every item's units (the units of an
    item are consecutive rows, in text order), and ``ranges`` the column
    range of every item in the *other* side's concatenated vectors.
    """

    texts: tuple[tuple[str, ...], ...]
    vectors: np.ndarray
    rows: tuple[tuple[int, ...], ...]
    ranges: tuple[tuple[int, int], ...]


def _encode_units(
    embedder: Embedder,
    chunker: Chunker,
    texts: Sequence[str],
    kind: str,
    dimension: int,
) -> _UnitSide:
    """Cut every text into its units and embed them in one batch."""
    chunk_lists = chunker.segment(list(texts))
    flat: list[str] = []
    rows: list[tuple[int, ...]] = []
    for chunk_list in chunk_lists:
        start = len(flat)
        flat.extend(chunk_list)
        rows.append(tuple(range(start, len(flat))))
    vectors = (
        np.asarray(embedder.encode(flat, kind), dtype=np.float32)  # type: ignore[arg-type]
        if flat
        else np.zeros((0, dimension), dtype=np.float32)
    )
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for chunk_list in chunk_lists:
        ranges.append((cursor, cursor + len(chunk_list)))
        cursor += len(chunk_list)
    return _UnitSide(
        texts=tuple(tuple(chunk_list) for chunk_list in chunk_lists),
        vectors=vectors,
        rows=tuple(rows),
        ranges=tuple(ranges),
    )


def _unit_rows(vectors: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    """The rows of the given items (an empty selection gives no rows)."""
    if not rows:
        return np.zeros((0, vectors.shape[1]), dtype=np.float32)
    return vectors[np.asarray(rows, dtype=np.int64)]


def _cross_blocks(paragraph_side: _UnitSide, question_side: _UnitSide) -> list[np.ndarray]:
    """The cross-similarity block of every paragraph against all question units.

    The block of a paragraph is its units' vectors times **all** the question
    side's unit vectors, so one matrix serves every question of that
    paragraph: the per-question slice is a column range.
    """
    blocks: list[np.ndarray] = []
    for rows in paragraph_side.rows:
        blocks.append(_unit_rows(paragraph_side.vectors, rows) @ question_side.vectors.T)
    return blocks


def _ranked_profile(block: np.ndarray, top_k: int) -> tuple[np.ndarray, float, int, int]:
    """The ranked cross-chunk profile of one block.

    Returns the top values (descending and **zero-padded** when the block
    holds fewer than ``top_k`` cosines — the lineage's convention), the best
    value and the indices of the best pair (the paragraph's unit, the
    question's unit).
    """
    if block.size == 0:
        return np.zeros(top_k, dtype=np.float32), float("-inf"), 0, 0
    flat = block.reshape(-1)
    count = min(top_k, flat.size)
    order = np.argpartition(flat, -count)[-count:]
    top = np.zeros(top_k, dtype=np.float32)
    top[:count] = np.sort(flat[order])[::-1]
    best = int(np.argmax(flat))
    best_paragraph, best_question = divmod(best, block.shape[1])
    return top, float(flat[best]), best_paragraph, best_question


class PairFeatureExtractor:
    """The feature pack: a pair in, a row of numbers out.

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
            unit). ``None`` keeps the default selection — the v0 pack's four
            groups, i.e. the Stage-3.1 behaviour; the whole pack is an
            explicit selection. The selected features keep the pack's
            canonical order, so a group's columns are the same whatever else
            is selected.
        profile_units: The chunk profile's unit splitters (the sentences and
            the word pieces). Needed when the ``chunk_profile`` group is
            selected; without them the profile's columns stay 0.
        profile_top_k: How many ranks of the cross-chunk profile are kept per
            granularity.
        stopwords: The stopwords of the lexical coverage.

    Raises:
        ValueError: If ``top_k`` is not positive, if a group name is unknown,
            if a group is repeated, if the selection is empty, if
            ``profile_top_k`` is not positive, or if the ``chunk_profile``
            group is selected without the profile's unit splitters.
    """

    def __init__(
        self,
        embedders: FeatureEmbedders,
        *,
        paragraph_chunker: Chunker,
        top_k: int = 3,
        question_embedding_texts: Mapping[str, str] | None = None,
        groups: Sequence[str] | None = None,
        profile_units: ProfileUnits | None = None,
        profile_top_k: int = 3,
        stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if profile_top_k < 1:
            raise ValueError(f"profile_top_k must be >= 1, got {profile_top_k}")
        self.embedders = embedders
        self.paragraph_chunker = paragraph_chunker
        self.top_k = top_k
        self.question_embedding_texts = dict(question_embedding_texts or {})
        self.stopwords = frozenset(stopwords)
        self.profile_units = profile_units
        self.profile_top_k = profile_top_k
        self.groups = self._checked_groups(groups)
        self._canonical_names = self._canonical_feature_names()
        self._selected = tuple(
            index
            for index, name in enumerate(self._canonical_names)
            if feature_group(name) in self.groups
        )
        if not self._selected:
            raise ValueError(
                f"the feature groups {list(self.groups)} select no feature of the pack"
            )
        if "chunk_profile" in self.groups and profile_units is None:
            raise ValueError(
                "the chunk_profile group needs the profile's unit splitters (profile_units=...)"
            )

    @staticmethod
    def _checked_groups(groups: Sequence[str] | None) -> tuple[str, ...]:
        """The group selection, checked (known names, no repeats, not empty).

        ``None`` keeps the **default** selection — the v0 pack's four groups,
        i.e. the Stage-3.1 behaviour. The whole pack (every group, the
        Stage-3.3 chunk profile included) is an explicit selection.
        """
        if groups is None:
            return DEFAULT_FEATURE_GROUPS
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
        names = [
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
        ]
        for granularity in ("sent", "ngram"):
            names.extend(
                f"cprof_{granularity}_top{rank}" for rank in range(1, self.profile_top_k + 1)
            )
        names.extend(
            (
                "cprof_align_sent",
                "cprof_align_ngram",
                "cprof_align_lexical",
                "cprof_align_semantic_only",
            )
        )
        return tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The selected feature names, in column order (the K-dependent ones included)."""
        return tuple(self._canonical_names[index] for index in self._selected)

    @property
    def all_feature_names(self) -> tuple[str, ...]:
        """Every feature the pack computes, in column order."""
        return self._canonical_names

    @property
    def selected_indices(self) -> tuple[int, ...]:
        """The column indices of the selected features (in the full matrix)."""
        return self._selected

    @property
    def dropped_feature_names(self) -> tuple[str, ...]:
        """The canonical features the current group selection leaves out."""
        return tuple(
            name for index, name in enumerate(self._canonical_names) if index not in self._selected
        )

    def describe(self) -> dict[str, Any]:
        """A JSON-friendly description of the pack (recorded in the artifact)."""
        return {
            "name": "v0+chunk_profile",
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "all_feature_names": list(self._canonical_names),
            "groups": list(self.groups),
            "dropped_features": list(self.dropped_feature_names),
            "paragraph_chunker": self.paragraph_chunker.version,
            "top_k": self.top_k,
            "profile_top_k": self.profile_top_k,
            "profile_units": (
                None
                if self.profile_units is None
                else {
                    "sentences": self.profile_units.sentences.version,
                    "ngrams": self.profile_units.ngrams.version,
                }
            ),
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
                    ("paragraph_sentences", self.embedders.paragraph_sentences),
                    ("question_sentences", self.embedders.question_sentences),
                    ("paragraph_ngrams", self.embedders.paragraph_ngrams),
                    ("question_ngrams", self.embedders.question_ngrams),
                )
                if embedder is not None
            },
            "note": (
                "the pack is label-free and pair-local: no corpus-level question "
                "statistic, no document-identity feature, and no length feature "
                "(length is a document proxy on this data; the lengths are "
                "recorded in the artifact instead). 'groups' is the feature-group "
                "selection of the Stage-3.1b ablation: the pack computes every "
                "feature and the combiner sees only the selected groups' columns. "
                "The chunk_profile group is the Stage-3.3 block (exp17's ranked "
                "cross-chunk match profile and exp16's alignment kinds); exp16's "
                "corpus-level df was dropped by the reviewer's decision, because "
                "it would break the pack's pair-locality."
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

        The columns are the **selected** features (the groups the extractor
        was given); the pack computes every feature, the combiner sees the
        selection.
        """
        return self.extract_all(pairs)[:, np.asarray(self._selected, dtype=np.int64)]

    def extract_all(self, pairs: Sequence[TextPair]) -> np.ndarray:
        """The full feature matrix of the pairs (every canonical column).

        The embeddings are done in **one batch per side**: the unique
        paragraphs, the unique questions and the unique chunks are encoded
        once each, and every pair's features are derived from them. That is a
        memoisation of a deterministic function of a text (the same text always
        embeds to the same vector), so the features stay pair-local.

        Args:
            pairs: The pairs to featurize (the text only — never a label).

        Returns:
            The float32 feature matrix, one row per pair, with every
            canonical column (the selected ones included).
        """
        items = list(pairs)
        matrix = np.zeros((len(items), len(self._canonical_names)), dtype=np.float32)
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

        # The chunk profile's units (the sentences and the word pieces), on both
        # sides. The question side follows the embedding side: the units are
        # cut from the transformed question (the "Answer choices:" block
        # stripped), so the scale words stay out of the cross-chunk matches.
        profile = self._profile_sides(paragraph_texts, embedding_texts, paragraph_vectors.shape[1])

        top_k = self.top_k
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
            values[6] = soft_coverage(stem_fact.token_ngrams, paragraph_fact.ngrams)
            values[7] = soft_coverage(paragraph_fact.token_ngrams, stem_fact.ngrams)
            values[8] = char_cosine(
                paragraph_fact.gram_counts,
                stem_fact.gram_counts,
                paragraph_fact.gram_norm,
                stem_fact.gram_norm,
            )

            # 4. the question parts and the question type
            options_fact = question_option_facts[column]
            values[9] = (
                soft_coverage(options_fact.token_ngrams, paragraph_fact.ngrams)
                if options_fact is not None
                else 0.0
            )
            values[10] = 1.0 if parts.has_options else 0.0
            values[11] = 1.0 if _has_no_data_option(parts.options) else 0.0
            values[12] = 1.0 if _is_yes_no(parts) else 0.0
            values[13] = 1.0 if _is_frequency(parts) else 0.0
            values[14] = float(len(parts.options))

            # 5. the chunk profile (the Stage-3.3 block)
            if profile is not None:
                self._profile_values(values, profile, row, column)

            # every canonical column is written; the combiner sees the
            # selected groups' columns only (``extract`` slices them)
            matrix[position] = values
        return matrix

    def _profile_sides(
        self,
        paragraph_texts: Sequence[str],
        question_texts: Sequence[str],
        dimension: int,
    ) -> dict[str, Any] | None:
        """The profile's two sides and their cross-similarity blocks.

        ``None`` when the pack has no unit splitters (the profile's columns
        then stay 0 — the pack still computes every other feature).
        """
        if self.profile_units is None:
            return None
        paragraph_sentences = _encode_units(
            self.embedders.paragraph_sentences,  # type: ignore[arg-type]
            self.profile_units.sentences,
            paragraph_texts,
            "document",
            dimension,
        )
        question_sentences = _encode_units(
            self.embedders.question_sentences,  # type: ignore[arg-type]
            self.profile_units.sentences,
            question_texts,
            "query",
            dimension,
        )
        paragraph_ngrams = _encode_units(
            self.embedders.paragraph_ngrams,  # type: ignore[arg-type]
            self.profile_units.ngrams,
            paragraph_texts,
            "document",
            dimension,
        )
        question_ngrams = _encode_units(
            self.embedders.question_ngrams,  # type: ignore[arg-type]
            self.profile_units.ngrams,
            question_texts,
            "query",
            dimension,
        )
        return {
            "paragraph_sentences": paragraph_sentences,
            "question_sentences": question_sentences,
            "paragraph_ngrams": paragraph_ngrams,
            "question_ngrams": question_ngrams,
            "sentence_blocks": _cross_blocks(paragraph_sentences, question_sentences),
            "ngram_blocks": _cross_blocks(paragraph_ngrams, question_ngrams),
        }

    def _profile_values(
        self,
        values: np.ndarray,
        profile: dict[str, Any],
        row: int,
        column: int,
    ) -> None:
        """The profile's columns of one pair (the ranked profile and the kinds).

        The layout of the profile's columns: the sentence granularity's ranks,
        the word-piece granularity's ranks, then the four alignment kinds.
        """
        first = len(V0_FEATURE_NAMES)
        ranks = self.profile_top_k
        sentence_side: _UnitSide = profile["question_sentences"]
        ngram_side: _UnitSide = profile["question_ngrams"]
        paragraph_sentences: _UnitSide = profile["paragraph_sentences"]
        paragraph_ngrams: _UnitSide = profile["paragraph_ngrams"]

        sentence_start, sentence_end = sentence_side.ranges[column]
        ngram_start, ngram_end = ngram_side.ranges[column]
        sentence_block = profile["sentence_blocks"][row][:, sentence_start:sentence_end]
        ngram_block = profile["ngram_blocks"][row][:, ngram_start:ngram_end]

        sentence_top, sentence_best, sentence_p, sentence_q = _ranked_profile(sentence_block, ranks)
        ngram_top, ngram_best, ngram_p, ngram_q = _ranked_profile(ngram_block, ranks)

        values[first : first + ranks] = sentence_top
        values[first + ranks : first + 2 * ranks] = ngram_top

        # the best alignment over both granularities (a tie goes to the
        # sentences, so the kind is deterministic)
        if sentence_best >= ngram_best:
            values[first + 2 * ranks] = 1.0
            paragraph_unit = paragraph_sentences.texts[row][sentence_p]
            question_unit = sentence_side.texts[column][sentence_q]
        else:
            values[first + 2 * ranks + 1] = 1.0
            paragraph_unit = paragraph_ngrams.texts[row][ngram_p]
            question_unit = ngram_side.texts[column][ngram_q]
        # the lexicalness: do the two aligned units share a content word?
        shared = set(content_tokens(paragraph_unit, self.stopwords)) & set(
            content_tokens(question_unit, self.stopwords)
        )
        if shared:
            values[first + 2 * ranks + 2] = 1.0
        else:
            values[first + 2 * ranks + 3] = 1.0


def build_feature_extractor(
    embedders: FeatureEmbedders,
    *,
    paragraph_chunker: Chunker,
    top_k: int = 3,
    question_embedding_texts: Mapping[str, str] | None = None,
    groups: Sequence[str] | None = None,
    profile_units: ProfileUnits | None = None,
    profile_top_k: int = 3,
) -> PairFeatureExtractor:
    """Build the feature pack (the config-facing entry point)."""
    return PairFeatureExtractor(
        embedders,
        paragraph_chunker=paragraph_chunker,
        top_k=top_k,
        question_embedding_texts=question_embedding_texts,
        groups=groups,
        profile_units=profile_units,
        profile_top_k=profile_top_k,
    )
