"""The grammatical axis (CS-52): the POS filter on the pair unit.

CS-34's five Penn classes on the reviewed paragraphs (the class of each word's
modal tag, the tagger at the edge with its answers cached), the readout on the
z-score, and the two POS-restricted z arms (``noun``, ``verb``) measured
against the full z arm — the same increment test CS-51 used for the class
filter.

CS-34 was a null on UPR (the nouns and the verbs were nearly identical, the
noun-only map materially worse) and CS-51 found the hard class filter a small
cost on top of CS-50's soft prior; the pre-registered prediction is a null
again, with its own bar (the noun-only arm beating the full arm by >= 0.05).

One cell: ``analyze-voyage-dict-pos`` -> ``results/omct-analyze-voyage-dict-pos/``.
The ``pos:`` section is on (the tagger is enabled); the model is already in the
cache from the UPR CS-34 run, and the reviewed set's 225 paragraphs are tagged
once (cached).
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
        label="analyze-voyage-dict-pos",
        analysis=AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name=REFERENCE_MODEL, embed=MODELS[REFERENCE_MODEL]),
            ),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(
                include_map=True,
                folds=5,
                scorings=(
                    "sum_logodds",
                    "sum_positive",
                    "max_logodds",
                    "support_count",
                    "sum_zscore",
                    "sum_zscore_positive",
                ),
                primary=PRIMARY,
                pos_restrictions=("noun", "verb"),
            ),
        ),
    )
]
# The tagger (CS-34's): the same nltk model, the paragraph corpus tagged
# once. ``analysis.dictionary.pos_restrictions`` is the tagger's on-switch for
# this run (the command enables it; the ``pos:`` section's own ``enabled``
# stays the UPR default).
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_pos")
