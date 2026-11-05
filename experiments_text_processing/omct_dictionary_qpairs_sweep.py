"""Question pairs on OMCT (CS-57): the questionnaire's own relation.

CS-38's label-pair map, adapted: there is no catalog on OMCT, so the taxonomy's
claim cannot be the test. The relation is read against the reviewed set's own
structure — do questions that share a reviewed paragraph (the same document's
questionnaire) share vocabulary? — plus the closest-pairs readout and the
vocabulary-less class. A description: no pair label, no embedding, no model.

One cell: ``analyze-voyage-dict-qpairs`` ->
``results/omct-analyze-voyage-dict-qpairs/``.
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
        label="analyze-voyage-dict-qpairs",
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
                question_pairs=True,
            ),
        ),
    )
]
# ``analysis.dictionary.question_pairs`` is the description's on-switch.
SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_qpairs")
