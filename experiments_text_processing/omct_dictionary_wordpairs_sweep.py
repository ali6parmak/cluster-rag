"""The word-pair table on OMCT (CS-59): one global table, two evaluations.

The per-question dictionary (CS-48 … CS-57) is a **fixed questionnaire**: a
question gets a table only after a positive judged pair, so **85 of 294**
questions have none and **518 pairs (34.08 %)** are floored under the pair
cross-fit. A new question is silent. This change set moves the unit: **one
global table** over the words a paragraph and a question **share**, learned
from all 1,520 judged pairs, with **no question in the key** (the diagonal of
the paragraph-word x question-word matrix — the plan's option (a)).

Two registered evaluations:

- the **pair-level** cross-fit (seed 42, the round's scheme): all questions
  seen, compared with the per-question z arm (0.7020) and the cosine (0.5238);
- the **question-level** cross-fit (5 folds over the 294 question texts,
  ``default_rng([42, 2])``): each fold's table from the other folds'
  questions, the held-out questions' pairs evaluated against the cosine
  (0.5238) — the number that matters, because the per-question dictionary is
  100 % floored there.

One cell: ``analyze-voyage-wpairs`` ->
``results/omct-analyze-voyage-wpairs/``. **0 model calls** (the cosine is the
Stage-0 cache), the artifact's block at
``models[kind=dictionary].readout.word_pairs``.
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
        label="analyze-voyage-wpairs",
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
                abstention_fills=("neutral",),
                word_pairs=True,
                word_pair_scorings=(WORD_PAIR, "support_count"),
                word_pair_primary=WORD_PAIR,
            ),
        ),
    )
]
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_wordpairs")
