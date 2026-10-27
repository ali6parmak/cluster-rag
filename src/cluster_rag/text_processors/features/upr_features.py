"""The UPR feature pack: one row of numbers per (text, label) cell.

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8d) asks for the OMCT track's
feature pack on the **UPR** data: "the global cosine, the best-chunk cosine,
the mean of the 2 and of the K=3 best chunk cosines, their spread and the peak
margin, the soft lexical overlap in both directions (text <-> label name) and
the char-n-gram cosine, the label-description coverage, the label-type
features and the chunk-profile block".

The UPR pack differs from the OMCT one in two structural ways, and both
follow from the data:

- **The pair is (text, label), not (paragraph, question).** The UPR grid is
  dense: every text is scored against every label, so the pack's natural
  output is a **cube** ``(n_texts, n_labels, n_features)`` — the grid's cells,
  not a list of pairs. The combiner then sees the flattened grid.
- **The label side has no parts.** A UPR label is a taxonomy term ("Detention",
  "Enforced disappearances"): it has no answer options, no "no data" option
  and no yes/no structure, so OMCT's question-part and question-type blocks
  have **no clean UPR analogue**. Their label-side counterparts (the label's
  description and the label's parent, i.e. its type) need a **label
  catalog** — a ``labels.jsonl`` with a name / description / parent per label
  (``paths.label_catalog``). The repo has none (``label_catalog: null``), so
  the two groups are **catalog-gated**: selecting them without a catalog is a
  config-time error, and a run without a catalog records the gap in its
  artifact. The CS-8d run therefore packs the ``embedding`` and ``lexical``
  groups only.

**The features** (the canonical order; the group of a feature is decided by
its name prefix, so the pack's groups partition it exactly):

==========================  ==================================================
feature                     what it measures
==========================  ==================================================
``cosine_full``             the whole text's cosine against the label vector
``chunk_cosine_best``       the best chunk's cosine against the label vector
``chunk_cosine_top2_mean``  the mean of the two best chunk cosines
``chunk_cosine_topK_mean``  the mean of the K best chunk cosines
``chunk_cosine_spread``     the standard deviation of the K best chunk cosines
``chunk_peak_margin``       the best chunk cosine minus the K-best mean
``lex_label_coverage``      the label name's soft coverage by the text
``lex_text_coverage``       the text's soft coverage by the label name
``lex_char_ngram_cosine``   the char-3-gram cosine of the text and the name
``labelpart_description_``  the label description's soft coverage by the text
``labeltype_<parent>``      the label's parent (its type), as a one-hot
==========================  ==================================================

**No length feature.** The length is *measurable* on UPR, so it is recorded
and reported as a diagnostic (the CS-6c precedent) and never fed to the model:
the text's length tracks the label's frequency and the record's nature, i.e.
it is a document proxy. The **label frequency** is a corpus statistic and is
not a feature either (it is a fit-side-only quantity by construction, and the
pack's spec has no frequency column).

**Pair-local by construction.** The pack sees one (text, label) cell at a
time and derives everything from that cell's own two texts: no corpus
statistic, no document identity, no length. The embeddings come through the
:class:`UprFeatureEmbedders` **ports** (the same precedent as the OMCT pack
and the semantic chunker): the CLI wires the cached backends in, and the
random-vector floor (CS-8d's third floor) swaps them for seeded noise.

Pure core: numpy + the standard library + ``cluster_rag.embed.base`` only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.embed.base import Embedder
from cluster_rag.text_processors.base import Chunker
from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    ProfileUnits,
    char_cosine,
    char_ngrams,
    content_tokens,
    soft_coverage,
)

# The UPR pack's feature groups. The two catalog-gated groups need a label
# catalog (a name / description / parent per label); without one they cannot
# be selected (the config refuses the selection, the artifact records the gap).
UPR_FEATURE_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "embedding": ("cosine_full", "chunk_"),
    "lexical": ("lex_",),
    "label_part": ("labelpart_",),
    "label_type": ("labeltype_",),
    "chunk_profile": ("cprof_",),
    "decision": ("dec_",),
}

# The group names, in the canonical order.
UPR_FEATURE_GROUPS: tuple[str, ...] = tuple(UPR_FEATURE_GROUP_PREFIXES)

# The groups that need a label catalog.
CATALOG_GROUPS: tuple[str, ...] = ("label_part", "label_type")

# The **default** selection: the two catalog-free groups. The CS-8d run uses
# exactly these (the repo has no label catalog), and the artifact records the
# gap the two gated groups leave.
DEFAULT_UPR_FEATURE_GROUPS: tuple[str, ...] = ("embedding", "lexical")

# The K of the top-K chunk similarities (the spread and the peak margin are
# measured over the same K best chunk cosines).
DEFAULT_UPR_TOP_K = 3

# The name prefixes of the dynamic columns: the label's description coverage
# and the label-type one-hots.
DESCRIPTION_FEATURE_PREFIX = "labelpart_description_coverage"
LABEL_TYPE_FEATURE_PREFIX = "labeltype_"

# The **decision** group's column (the track's CS-13): the decision model's
# probability for the (text, label) cell. The pack does not compute it -- the
# model is called at the edge and the matrix is passed in -- so the column is
# **cell-local** by construction.
DECISION_FEATURE_NAME = "dec_probability"

# The chunk profile's feature names (the CS-8e group): the ranked cross-unit
# match profile at the two granularities and the alignment kinds.
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


def upr_feature_group(name: str) -> str | None:
    """The group a feature belongs to (``None`` when it belongs to none).

    Args:
        name: The feature's name (e.g. ``"chunk_cosine_best"``).

    Returns:
        The group's name, or ``None`` when no group claims the feature.
    """
    for group, prefixes in UPR_FEATURE_GROUP_PREFIXES.items():
        if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
            return group
    return None


@dataclass(frozen=True)
class UprLabel:
    """One label's side of the pack.

    Attributes:
        name: The label's **name** (the taxonomy term, e.g. ``"Detention"``).
            The lexical features compare the **name** with the text: it is the
            label's topical content.
        text: The text that is **embedded** for this label (the zero-shot
            label vector's own text — the bare name or the config template,
            e.g. ``"This recommendation is about Detention."``). It defaults to
            the name.
        description: The label's description (the catalog's field); ``None``
            when there is no catalog.
        parent: The label's parent category (its **type** in the taxonomy);
            ``None`` when there is no catalog.
    """

    name: str
    text: str = ""
    description: str | None = None
    parent: str | None = None

    @property
    def embedded_text(self) -> str:
        """The text that is embedded for this label (the name, when unset)."""
        return self.text or self.name


@dataclass(frozen=True)
class UprFeatureEmbedders:
    """The embedding sides the UPR feature pack needs (the ``Embedder`` ports).

    Every side has its own cache-version slot, so they are separate ports:

    - ``text``: the whole text (the ``query`` kind — the UPR pipeline's query
      side is the text side);
    - ``chunks``: the text's chunks (the ``query`` kind, the chunker's own
      cache slot);
    - ``labels``: the label texts (the ``label`` kind — the zero-shot label
      vectors' own slot);
    - ``text_sentences`` / ``text_ngrams``: the text's **profile units** (the
      ``query`` kind, the chunk profile group's own slots). ``None`` disables
      the chunk-profile group.

    The CLI wires the cached backends in; the random-vector floor swaps them
    for the seeded :class:`~cluster_rag.embed.random.RandomEmbedder`.
    """

    text: Embedder
    chunks: Embedder
    labels: Embedder
    text_sentences: Embedder | None = None
    text_ngrams: Embedder | None = None


@dataclass
class _TextFacts:
    """The per-text facts the pack derives once and re-uses for every label."""

    ngrams: frozenset[str]
    gram_counts: dict[str, int]
    gram_norm: float
    token_ngrams: list[frozenset[str]]


@dataclass(frozen=True)
class _LabelFacts:
    """The per-label facts the pack derives once (the name's side)."""

    name_ngrams: frozenset[str]
    name_gram_counts: dict[str, int]
    name_gram_norm: float
    name_token_ngrams: list[frozenset[str]]
    description_token_ngrams: list[frozenset[str]]


class UprFeatureExtractor:
    """The UPR feature pack: a text and a label in, a row of numbers out.

    The extractor is **label-free and cell-local**: it derives every feature
    from one (text, label) cell's own two texts, keeps no corpus-level
    statistic and never sees a gold label. The embeddings it needs come
    through the :class:`UprFeatureEmbedders` ports (the CLI wires the cached
    backends in).

    Args:
        embedders: The embedding sides (the ports).
        labels: The label space (one :class:`UprLabel` per label, in the
            harness's column order).
        text_chunker: The chunker whose chunks the chunk-level features are
            measured on (the text side only — the label side is never
            chunked: the UPR labels are 1-3 words, so the label side is the
            identity, the CS-8b verdict).
        top_k: The K of the top-K chunk similarities (the spread and the peak
            margin are measured over the same K best chunk cosines).
        groups: The **feature groups** to keep (the CS-8e's ablation unit).
            ``None`` keeps the default selection — the two catalog-free
            groups. The selected features keep the pack's canonical order.
        catalog_available: Whether a label catalog was available. The
            catalog-gated groups can only be selected when it is.
        stopwords: The stopwords of the lexical coverage.

    Raises:
        ValueError: If the label space is empty, if a group name is unknown,
            if the selection is empty or repeats a group, if a catalog-gated
            group is selected without a catalog, if ``top_k`` is not positive
            or if it would duplicate the top-2 mean column.
    """

    def __init__(
        self,
        embedders: UprFeatureEmbedders,
        *,
        labels: Sequence[UprLabel],
        text_chunker: Chunker,
        top_k: int = DEFAULT_UPR_TOP_K,
        groups: Sequence[str] | None = None,
        catalog_available: bool = False,
        profile_units: ProfileUnits | None = None,
        profile_top_k: int = 3,
        stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    ) -> None:
        if not labels:
            raise ValueError("the UPR feature pack needs a non-empty label space")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k == 2:
            raise ValueError("top_k=2 would duplicate the top-2 mean column: use 1 or a value >= 3")
        if profile_top_k < 1:
            raise ValueError(f"profile_top_k must be >= 1, got {profile_top_k}")
        self.embedders = embedders
        self.labels = tuple(labels)
        self.text_chunker = text_chunker
        self.top_k = top_k
        self.stopwords = frozenset(stopwords)
        self.catalog_available = bool(catalog_available)
        self.profile_units = profile_units
        self.profile_top_k = profile_top_k
        self.groups = self._checked_groups(groups)
        gated = [group for group in CATALOG_GROUPS if group in self.groups]
        if gated and not self.catalog_available:
            raise ValueError(
                f"the feature group(s) {gated} need a label catalog (a name / "
                "description / parent per label, paths.label_catalog): the repo "
                "has none, so the pack runs without them"
            )
        if "chunk_profile" in self.groups and profile_units is None:
            raise ValueError(
                "the chunk_profile group needs the profile's unit splitters (profile_units=...)"
            )
        self._canonical_names = self._canonical_feature_names()
        self._selected = tuple(
            index
            for index, name in enumerate(self._canonical_names)
            if upr_feature_group(name) in self.groups
        )
        if not self._selected:
            raise ValueError(
                f"the feature groups {list(self.groups)} select no feature of the pack"
            )

    @staticmethod
    def _checked_groups(groups: Sequence[str] | None) -> tuple[str, ...]:
        """The group selection, checked (known names, no repeats, not empty).

        ``None`` keeps the **default** selection — the two catalog-free
        groups, i.e. the CS-8d behaviour.
        """
        if groups is None:
            return DEFAULT_UPR_FEATURE_GROUPS
        selected = tuple(groups)
        if not selected:
            raise ValueError("the feature groups must name at least one group")
        unknown = [group for group in selected if group not in UPR_FEATURE_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown feature group(s) {unknown}; expected values in {list(UPR_FEATURE_GROUPS)}"
            )
        if len(set(selected)) != len(selected):
            raise ValueError(f"the feature groups must not repeat a group: {list(selected)}")
        return selected

    def _canonical_feature_names(self) -> tuple[str, ...]:
        """The pack's feature names, in column order (the dynamic ones included)."""
        names = [
            "cosine_full",
            "chunk_cosine_best",
            "chunk_cosine_top2_mean",
            f"chunk_cosine_top{self.top_k}_mean",
            "chunk_cosine_spread",
            "chunk_peak_margin",
            "lex_label_coverage",
            "lex_text_coverage",
            "lex_char_ngram_cosine",
        ]
        if "label_part" in self.groups:
            names.append(DESCRIPTION_FEATURE_PREFIX)
        if "label_type" in self.groups:
            names.extend(f"{LABEL_TYPE_FEATURE_PREFIX}{parent}" for parent in self._parent_names())
        if "decision" in self.groups:
            names.append(DECISION_FEATURE_NAME)
        if "chunk_profile" in self.groups:
            names.extend(self._profile_feature_names())
        return tuple(names)

    def _profile_feature_names(self) -> tuple[str, ...]:
        """The chunk profile's feature names (the K-dependent ones included)."""
        names: list[str] = []
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

    def _parent_names(self) -> tuple[str, ...]:
        """The catalog's parents (the label types), in sorted order."""
        parents = {label.parent for label in self.labels if label.parent}
        return tuple(sorted(parents))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The selected feature names, in column order."""
        return tuple(self._canonical_names[index] for index in self._selected)

    @property
    def all_feature_names(self) -> tuple[str, ...]:
        """Every feature the pack computes, in column order."""
        return self._canonical_names

    @property
    def selected_indices(self) -> tuple[int, ...]:
        """The column indices of the selected features (in the full cube)."""
        return self._selected

    @property
    def dropped_feature_names(self) -> tuple[str, ...]:
        """The canonical features the current group selection leaves out."""
        return tuple(
            name for index, name in enumerate(self._canonical_names) if index not in self._selected
        )

    @property
    def n_labels(self) -> int:
        """The size of the label space."""
        return len(self.labels)

    def describe(self) -> dict[str, Any]:
        """A JSON-friendly description of the pack (recorded in the artifact)."""
        gated = [group for group in CATALOG_GROUPS if group not in self.groups]
        return {
            "name": "upr",
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "all_feature_names": list(self._canonical_names),
            "groups": list(self.groups),
            "dropped_features": list(self.dropped_feature_names),
            "text_chunker": self.text_chunker.version,
            "top_k": self.top_k,
            "stopwords": sorted(self.stopwords),
            "n_labels": self.n_labels,
            "label_catalog": {
                "available": self.catalog_available,
                "gated_groups": list(CATALOG_GROUPS),
                "groups_not_selected": gated,
                "note": (
                    "the label-description coverage and the label-type features "
                    "need a label catalog (a name / description / parent per "
                    "label, paths.label_catalog). The repo has none, so the "
                    "pack runs without them and the gap is stated here: the "
                    "label's description and its parent category are not part "
                    "of this arm's features."
                ),
            },
            "embedding_sides": {
                side: {
                    "cache_hits": getattr(embedder, "cache_hits", None),
                    "cache_misses": getattr(embedder, "cache_misses", None),
                }
                for side, embedder in (
                    ("text", self.embedders.text),
                    ("chunks", self.embedders.chunks),
                    ("labels", self.embedders.labels),
                    ("text_sentences", self.embedders.text_sentences),
                    ("text_ngrams", self.embedders.text_ngrams),
                )
            },
            "profile": {
                "enabled": "chunk_profile" in self.groups,
                "profile_top_k": self.profile_top_k,
                "units": (
                    None
                    if self.profile_units is None
                    else {
                        "sentences": self.profile_units.sentences.version,
                        "ngrams": self.profile_units.ngrams.version,
                    }
                ),
                "note": (
                    "the chunk profile: the text is cut into its sentence units "
                    "and its word pieces, and every unit's cosine against the "
                    "label is ranked. The LABEL side is one unit (the UPR labels "
                    "are 1-3 words, so the label side is the identity — the "
                    "CS-8b verdict), so the profile says how well the label "
                    "matches the text's finest units. The alignment kinds say "
                    "which granularity produced the best match and whether the "
                    "aligned unit and the label's name share a content word."
                ),
            },
            "note": (
                "the pack is label-free and cell-local: no corpus statistic, no "
                "document-identity feature and no length feature (the length is "
                "measurable on UPR, so it is recorded as a diagnostic and never "
                "fed to the model — the CS-6c precedent). The lexical features "
                "compare the text with the label's bare NAME; the embedding "
                "features use the label's embedded text (the zero-shot label "
                "vector's own text)."
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

    def _label_facts(self, label: UprLabel) -> _LabelFacts:
        """The per-label facts (the name's n-grams and its tokens')."""
        counts: dict[str, int] = {}
        for gram in char_ngrams(label.name):
            counts[gram] = counts.get(gram, 0) + 1
        norm = float(sum(count * count for count in counts.values())) ** 0.5
        description_tokens = (
            content_tokens(label.description, self.stopwords) if label.description else []
        )
        return _LabelFacts(
            name_ngrams=frozenset(counts),
            name_gram_counts=counts,
            name_gram_norm=norm,
            name_token_ngrams=[
                _token_ngrams(token) for token in content_tokens(label.name, self.stopwords)
            ],
            description_token_ngrams=[_token_ngrams(token) for token in description_tokens],
        )

    # -- the extraction ----------------------------------------------------------------

    def chunks_per_text(self, texts: Sequence[str]) -> tuple[int, ...]:
        """The number of chunks every text contributes (the artifact's lineage)."""
        return tuple(len(chunks) for chunks in self.text_chunker.segment(list(texts)))

    def extract(
        self, texts: Sequence[str], decision_scores: np.ndarray | None = None
    ) -> np.ndarray:
        """The feature cube of the texts, shape ``(n_texts, n_labels, n_features)``.

        The columns are the **selected** features (the groups the extractor
        was given); the pack computes every feature, the combiner sees the
        selection.

        Args:
            texts: The texts to featurize.
            decision_scores: The decision model's probability matrix for the
                cells (the ``decision`` group's column; the CS-13 feature).
        """
        selected = np.asarray(self._selected, dtype=np.int64)
        return self.extract_all(texts, decision_scores)[:, :, selected]

    def extract_all(
        self, texts: Sequence[str], decision_scores: np.ndarray | None = None
    ) -> np.ndarray:
        """The full feature cube (every canonical column).

        The embeddings are done in **one batch per side**: the texts, their
        chunks and the label texts are encoded once each, and every cell's
        features are derived from them. That is a memoisation of a
        deterministic function of a text (the same text always embeds to the
        same vector), so the features stay cell-local.

        Args:
            texts: The texts to featurize (the text only — never a label).
            decision_scores: The decision model's probability matrix for the
                cells (the ``decision`` group's column; the CS-13 feature).
                Required when the group is selected.

        Returns:
            The float32 feature cube, one row per (text, label) cell.

        Raises:
            ValueError: If the ``decision`` group is selected and the matrix
                is missing or has the wrong shape.
        """
        items = list(texts)
        n_texts = len(items)
        n_labels = self.n_labels
        n_features = len(self._canonical_names)
        matrix = np.zeros((n_texts, n_labels, n_features), dtype=np.float32)
        if not items:
            return matrix
        decision_index = (
            self._canonical_names.index(DECISION_FEATURE_NAME)
            if "decision" in self.groups
            else None
        )
        if decision_index is not None:
            if decision_scores is None:
                raise ValueError(
                    "the 'decision' feature group needs the decision model's "
                    "probability matrix (the caller scores the cells at the edge)"
                )
            probabilities = np.asarray(decision_scores, dtype=np.float32)
            if probabilities.shape != (n_texts, n_labels):
                raise ValueError(
                    f"the decision matrix must have shape {(n_texts, n_labels)}, "
                    f"got {probabilities.shape}"
                )

        text_vectors = np.asarray(self.embedders.text.encode(items, "query"), dtype=np.float32)
        label_vectors = np.asarray(
            self.embedders.labels.encode([label.embedded_text for label in self.labels], "label"),
            dtype=np.float32,
        )
        chunk_lists = self.text_chunker.segment(items)
        chunk_texts: list[str] = []
        chunk_rows: list[tuple[int, ...]] = []
        for chunk_list in chunk_lists:
            start = len(chunk_texts)
            chunk_texts.extend(chunk_list)
            chunk_rows.append(tuple(range(start, len(chunk_texts))))
        chunk_vectors = (
            np.asarray(self.embedders.chunks.encode(chunk_texts, "query"), dtype=np.float32)
            if chunk_texts
            else np.zeros((0, text_vectors.shape[1]), dtype=np.float32)
        )

        label_facts = [self._label_facts(label) for label in self.labels]
        description_index = (
            self._canonical_names.index(DESCRIPTION_FEATURE_PREFIX)
            if "label_part" in self.groups
            else None
        )
        type_index = (
            {
                parent: self._canonical_names.index(f"{LABEL_TYPE_FEATURE_PREFIX}{parent}")
                for parent in self._parent_names()
            }
            if "label_type" in self.groups
            else {}
        )
        # The chunk profile's units (the sentences and the word pieces) on the
        # text side; the label side is one unit (the label's own vector).
        profile = self._profile_sides(items, label_vectors, text_vectors.shape[1])

        top_k = self.top_k
        for position, text in enumerate(items):
            facts = self._facts(text)
            values = matrix[position]
            # 1. the global (whole text) cosine
            values[:, 0] = text_vectors[position] @ label_vectors.T
            # 2. the chunk-level cosines
            rows = chunk_rows[position]
            if rows:
                similarities = np.sort(chunk_vectors[list(rows)] @ label_vectors.T, axis=0)[::-1]
                values[:, 1] = similarities[0]
                values[:, 2] = similarities[: min(2, similarities.shape[0])].mean(axis=0)
                top = similarities[: min(top_k, similarities.shape[0])]
                values[:, 3] = top.mean(axis=0)
                values[:, 4] = top.std(axis=0) if top.shape[0] > 1 else 0.0
                values[:, 5] = similarities[0] - values[:, 3]
            # 3. the lexical overlap (the label's bare name against the text)
            for column, label_facts_row in enumerate(label_facts):
                values[column, 6] = soft_coverage(label_facts_row.name_token_ngrams, facts.ngrams)
                values[column, 7] = soft_coverage(facts.token_ngrams, label_facts_row.name_ngrams)
                values[column, 8] = char_cosine(
                    facts.gram_counts,
                    label_facts_row.name_gram_counts,
                    facts.gram_norm,
                    label_facts_row.name_gram_norm,
                )
                # 4. the label's description coverage (the catalog-gated group)
                if description_index is not None and label_facts_row.description_token_ngrams:
                    values[column, description_index] = soft_coverage(
                        label_facts_row.description_token_ngrams, facts.ngrams
                    )
            # 5. the label's type (the catalog-gated group)
            if type_index:
                for column, label in enumerate(self.labels):
                    if label.parent is not None and label.parent in type_index:
                        values[column, type_index[label.parent]] = 1.0
            # 6. the chunk profile (the CS-8e group)
            if profile is not None:
                self._profile_values(values, profile, position)
            # 7. the decision model's probability (the CS-13 group)
            if decision_index is not None:
                values[:, decision_index] = probabilities[position]
        return matrix

    def _profile_sides(
        self,
        texts: Sequence[str],
        label_vectors: np.ndarray,
        dimension: int,
    ) -> dict[str, Any] | None:
        """The profile's two unit sides and their cross-similarity blocks.

        ``None`` when the pack has no unit splitters (the profile's columns
        then stay 0 — the pack still computes every other feature).
        """
        if self.profile_units is None:
            return None
        sentence_texts, sentence_vectors = self._encode_units(
            self.embedders.text_sentences,  # type: ignore[arg-type]
            self.profile_units.sentences,
            texts,
            "query",
            dimension,
        )
        ngram_texts, ngram_vectors = self._encode_units(
            self.embedders.text_ngrams,  # type: ignore[arg-type]
            self.profile_units.ngrams,
            texts,
            "query",
            dimension,
        )
        return {
            "sentence_texts": sentence_texts,
            "ngram_texts": ngram_texts,
            "sentence_blocks": [units @ label_vectors.T for units in sentence_vectors],
            "ngram_blocks": [units @ label_vectors.T for units in ngram_vectors],
        }

    def _encode_units(
        self,
        embedder: Embedder,
        chunker: Chunker,
        texts: Sequence[str],
        kind: str,
        dimension: int,
    ) -> tuple[list[tuple[str, ...]], list[np.ndarray]]:
        """Cut every text into its units and embed them in one batch.

        Returns:
            The units' texts and their vectors, one entry per text.
        """
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
        unit_texts = [tuple(chunk_list) for chunk_list in chunk_lists]
        unit_vectors = [
            vectors[np.asarray(row, dtype=np.int64)]
            if row
            else np.zeros((0, dimension), dtype=np.float32)
            for row in rows
        ]
        return unit_texts, unit_vectors

    @staticmethod
    def _ranked_rows(sorted_block: np.ndarray, ranks: int) -> np.ndarray:
        """The top ranks of a sorted block, one row per label (zero-padded).

        The block is ``(n_units, n_labels)`` (the units down, the labels
        across); the result is its transpose, so a text with fewer units than
        the rank count is zero-padded (the OMCT pack's convention).
        """
        n_labels = sorted_block.shape[1]
        out = np.zeros((n_labels, ranks), dtype=np.float64)
        take = min(ranks, sorted_block.shape[0])
        if take:
            out[:, :take] = sorted_block[:take].T
        return out

    def _profile_values(
        self,
        values: np.ndarray,
        profile: dict[str, Any],
        position: int,
    ) -> None:
        """The profile's columns of one text (the ranked profile and the kinds).

        The layout of the profile's columns: the sentence granularity's ranks,
        the word-piece granularity's ranks, then the four alignment kinds.
        """
        first = len(self._canonical_names) - len(self._profile_feature_names())
        ranks = self.profile_top_k
        sentence_texts: list[tuple[str, ...]] = profile["sentence_texts"]
        ngram_texts: list[tuple[str, ...]] = profile["ngram_texts"]
        sentence_block = profile["sentence_blocks"][position]
        ngram_block = profile["ngram_blocks"][position]
        if sentence_block.size == 0 and ngram_block.size == 0:
            return
        sentence_sorted = np.sort(sentence_block, axis=0)[::-1]
        ngram_sorted = np.sort(ngram_block, axis=0)[::-1]
        values[:, first : first + ranks] = self._ranked_rows(sentence_sorted, ranks)
        values[:, first + ranks : first + 2 * ranks] = self._ranked_rows(ngram_sorted, ranks)

        # the best alignment over both granularities (a tie goes to the
        # sentences, so the kind is deterministic)
        empty = np.full(values.shape[0], -np.inf)
        sentence_best = sentence_sorted[0] if sentence_sorted.size else empty
        ngram_best = ngram_sorted[0] if ngram_sorted.size else empty
        sentence_index = np.argmax(sentence_block, axis=0) if sentence_block.size else None
        ngram_index = np.argmax(ngram_block, axis=0) if ngram_block.size else None
        for column in range(values.shape[0]):
            if sentence_best[column] >= ngram_best[column]:
                values[column, first + 2 * ranks] = 1.0
                unit = (
                    sentence_texts[position][int(sentence_index[column])]
                    if sentence_index is not None
                    else ""
                )
            else:
                values[column, first + 2 * ranks + 1] = 1.0
                unit = (
                    ngram_texts[position][int(ngram_index[column])]
                    if ngram_index is not None
                    else ""
                )
            # the lexicalness: does the aligned unit share a content word with
            # the label's name?
            shared = set(content_tokens(unit, self.stopwords)) & set(
                content_tokens(self.labels[column].name, self.stopwords)
            )
            if shared:
                values[column, first + 2 * ranks + 2] = 1.0
            else:
                values[column, first + 2 * ranks + 3] = 1.0


def _token_ngrams(token: str, size: int = 3) -> frozenset[str]:
    """The character n-grams of one token (space-padded like a text)."""
    return char_ngrams(token, size)
