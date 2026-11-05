"""The label side's own words (CS-55): the question text as a channel.

CS-35/45's label-side channel, ported to the pair unit: OMCT has no catalog, so
the label's "name" and "description" are the same object — the question text.
The channel reads the question's content words against the paragraph (no pair
label, no map, no embedding), the register is the questionnaire's shared
vocabulary (a label-side word in at least a quarter of the 294 questions'
texts), and the arms are `all` (the primary), `idf`, `topic`-only and
`register`-only, each against the cosine and the primary against the map's z
arm. The blind counterpart permutes the questions across the pairs.

One cell: ``analyze-voyage-dict-lexicon`` ->
``results/omct-analyze-voyage-dict-lexicon/``.
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
PRIMARY = "sum_zscore_positive"

cells = [
    AnalyzeOmctCell(
        label="analyze-voyage-dict-lexicon",
        analysis=AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name=REFERENCE_MODEL, embed=MODELS[REFERENCE_MODEL]),
            ),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(
                include_map=True,
                folds=5,
                scorings=(PRIMARY,),
                primary=PRIMARY,
                blind=False,
                question_lexicon=True,
            ),
        ),
    )
]
# ``analysis.dictionary.question_lexicon`` is the channel's on-switch; the map
# arm (the bar) is the run's own primary z arm.
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_lexicon")
