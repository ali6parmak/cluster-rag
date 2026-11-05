"""Abstention on the pair unit (CS-56): the empty-table fills.

CS-37's abstention, ported to the pair unit. On UPR the abstention ("no
evidence" as NaN instead of the floor) was a null: the "impossible" verdict was
never in the encoding. On OMCT the silence is per question and informative (the
floor pairs are 77.8 % negative), so the flip is the opposite one: the floor's
"impossible" should win.

Three fills of the empty-table cells only, on the primary z arm:

- the ``floor`` (the run's own primary grid: the row's smallest seen score
  minus one — the reference);
- ``neutral`` (the row's median of the seen cells — "no evidence");
- ``max`` (the row's maximum — "possible").

One cell: ``analyze-voyage-dict-abstain`` ->
``results/omct-analyze-voyage-dict-abstain/``.
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
        label="analyze-voyage-dict-abstain",
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
                abstention_fills=("neutral", "max"),
            ),
        ),
    )
]
# ``analysis.dictionary.abstention_fills`` names the fills; the floor is the
# run's own primary grid and the comparison's reference.
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_abstain")
