"""The stability of the pair unit (CS-54): which links survive resampling.

CS-36's bootstrap, ported to the reviewed pairs: the map's links' survival at
the two registered thresholds, the per-question and head readouts, and the
stability-filtered **z arm** measured against the full z arm — the reference is
CS-50's arm (0.7020, non-floor 0.8158), not the raw log-odds.

Each fold's filter is estimated on that fold's *own* training pairs (CS-36's
leakage discipline), so a judged pair is scored by a map and a filter that both
never saw it; the unjudged control's filter is the full reviewed-set estimate.

One cell: ``analyze-voyage-dict-stability`` ->
``results/omct-analyze-voyage-dict-stability/``.
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
        label="analyze-voyage-dict-stability",
        analysis=AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name=REFERENCE_MODEL, embed=MODELS[REFERENCE_MODEL]),
            ),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(
                include_map=True,
                folds=5,
                scorings=(PRIMARY, "support_count"),
                primary=PRIMARY,
                blind=False,
                stability=True,
                stability_resamples=500,
                stability_fold_resamples=100,
                stability_thresholds=(0.75, 0.90),
            ),
        ),
    )
]
# ``analysis.dictionary.stability`` is the bootstrap's on-switch; the two
# thresholds are the arms' registration, read off the survival curve.
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_stability")
