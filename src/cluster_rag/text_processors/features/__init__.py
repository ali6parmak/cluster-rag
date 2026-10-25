"""The track's feature packs: a row of numbers per (paragraph, question) pair.

``pair_features`` holds the **v0** pack of TEXT_PROCESSING_PLAN.md section 3
(Stage 3.1): the global and best chunk cosines, the top-K chunk similarities
and their spread, the soft lexical overlap and the char n-grams, the
question-part coverage and the question-type one-hots.

The pack is a **pure core** in the AGENTS.md sense: it imports numpy, the
standard library and ``cluster_rag.embed.base`` (a ``typing.Protocol``) — no
model client, no I/O. The embeddings it needs come through the
:class:`~cluster_rag.text_processors.features.pair_features.FeatureEmbedders`
ports, which the CLI wires to the cached backends (the same precedent as the
semantic chunker, which takes the ``Embedder`` port).

The pack is **label-free and pair-local by construction**: it sees one
(paragraph, question) pair at a time, keeps no corpus-level statistic, and
never sees a label, a document name or a length feature (the plan's Stage-3
leakage rules: no corpus-level question statistics, no document-identity
feature — and length is a document proxy, so it is recorded in the artifact
but never fed to the model).

The pack's features fall into four **groups** (``embedding``, ``lexical``,
``question_part``, ``question_type``) — the unit of the plan's Stage-3.1b
feature-group ablation. The extractor keeps the groups it is given, so the
ablation changes the combiner's columns and nothing else.
"""

from __future__ import annotations

from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    FEATURE_GROUP_PREFIXES,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FeatureEmbedders,
    PairFeatureExtractor,
    QuestionParts,
    feature_group,
    split_question_parts,
)

__all__ = [
    "DEFAULT_STOPWORDS",
    "FEATURE_GROUPS",
    "FEATURE_GROUP_PREFIXES",
    "FEATURE_NAMES",
    "FeatureEmbedders",
    "PairFeatureExtractor",
    "QuestionParts",
    "feature_group",
    "split_question_parts",
]
