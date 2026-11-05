"""The phrases (CS-53): the unit change — the phrase table and the union arm.

The map rebuilt over the reviewed paragraphs' **consecutive bigrams** (the
same builder via its ``units`` parameter, the full token stream with the
stopwords kept — CS-39's settled wording), with the same folds, the same
estimator and the same primary arm as CS-50. Two things are measured beside
the phrase arm's own number:

- the **union arm** = the word z arm + the phrase z arm (CS-39's composition,
  ported cell for cell), which is the registered test: it must beat the
  **word** z arm by >= 0.05 on the same pairs, with a paired interval
  excluding zero, for "phrases work";
- the two halves' **count** arms (CS-49's measure-free instrument, kept).

CS-39 was a null on UPR for rarity (82.8 % hapax, the blind union gained
more); the registered prediction here is a null for **redundancy** (a bigram
occurs only where both its words do), with the phrase table's prior measured
and reported beside it.

One cell: ``analyze-voyage-dict-phrases`` ->
``results/omct-analyze-voyage-dict-phrases/``. The reference cosine is
voyage-4-nano's own (the Stage-0 cache hits); no blind arm (the increment is
between two arms built from the same labels), as the pre-registration states.
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
        label="analyze-voyage-dict-phrases",
        analysis=AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name=REFERENCE_MODEL, embed=MODELS[REFERENCE_MODEL]),
            ),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(
                include_map=True,
                folds=5,
                units="phrases",
                scorings=(PRIMARY, "support_count"),
                primary=PRIMARY,
                blind=False,
                union=True,
            ),
        ),
    )
]
# The phrase table's unit (CS-39's bigrams) rides through
# ``analysis.dictionary.units``; the union arm through
# ``analysis.dictionary.union`` (valid only with the phrase unit).
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_phrases")
