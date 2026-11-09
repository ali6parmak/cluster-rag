"""The paragraph-level holdout on OMCT (CS-60): the missing generalization test.

CS-59 held out **questions** but shared **paragraphs** — a held-out question's
pairs sit on paragraphs the fit side has seen under other questions, so 0.6309
is "new question, known paragraphs", not "new text". This change set holds out
the other side: **5 folds over the 225 paragraph texts**
(``default_rng([42, 3])``), each fold's table from the other folds'
paragraphs' pairs, the held-out paragraphs' pairs evaluated (the questions are
shared — exactly what a routing rule needs).

The **neutral fill is the primary encoding** (the shipped rule recorded in
§7.11); the floor is reported beside it for comparability with CS-59's 0.6309,
and the fill's delta is **paired-tested** here (the caveat CS-59 left open).
The **fallback set** — the pairs whose per-question map has no table in their
fold — is the routing rule's use case and has its own registered test.

One cell: ``analyze-voyage-wpairs-paragraph`` ->
``results/omct-analyze-voyage-wpairs-paragraph/``. **0 model calls** (the
cosine is the Stage-0 cache), the artifact's block at
``models[kind=dictionary].readout.word_pairs_paragraph``.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    DictionaryConfig,
)
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

REFERENCE_MODEL = "voyage-4-nano"
CONTROL_SIZE = 4000
PER_QUESTION = "sum_zscore_positive"
WORD_PAIR = "sum_zscore_positive"

cells = [
    AnalyzeOmctCell(
        label="analyze-voyage-wpairs-paragraph",
        analysis=AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name=REFERENCE_MODEL, embed=MODELS[REFERENCE_MODEL]),
            ),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(
                include_map=False,
                folds=5,
                scorings=(PER_QUESTION,),
                primary=PER_QUESTION,
                blind=True,
                word_pair_paragraph_holdout=True,
                word_pair_scorings=(WORD_PAIR,),
                word_pair_primary=WORD_PAIR,
            ),
        ),
    )
]
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_wordpairs_paragraph")
