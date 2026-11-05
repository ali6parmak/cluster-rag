"""The word classes (CS-51): the hard filter against the soft prior.

CS-50's z-score already sends the common register words to `z ~ 0` — a soft
class restriction. CS-51 ports CS-33's three classes (a word's class is how
many of the **labelled** tables it sits in: distinctive 1-2, shared 3-52,
generic >= 53 of the 209) and measures the two **hard** filters with the
primary z arm behind them, each compared with the **full** z arm on the same
pairs (the increment test):

- ``nogeneric`` — keep distinctive + shared (drop the register);
- ``distinctive`` — keep only the topic's own vocabulary.

The pre-registered prediction is that the hard filter adds little (the two are
substitutes); "it works" = the "no generic" arm beats the full arm by >= 0.05
with a paired interval excluding zero. The run is one cell:
``analyze-voyage-dict-classes`` -> ``results/omct-analyze-voyage-dict-classes/``.
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

# The full z arm (CS-50's winner) with the two hard class filters beside it.
SCORINGS = (
    "sum_logodds",
    "sum_positive",
    "max_logodds",
    "support_count",
    "sum_zscore",
    "sum_zscore_positive",
)
PRIMARY = "sum_zscore_positive"

cells = [
    AnalyzeOmctCell(
        label="analyze-voyage-dict-classes",
        analysis=AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name=REFERENCE_MODEL, embed=MODELS[REFERENCE_MODEL]),
            ),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(
                include_map=True,
                folds=5,
                scorings=SCORINGS,
                primary=PRIMARY,
                class_restrictions=("nogeneric", "distinctive"),
            ),
        ),
    )
]

SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_classes")
