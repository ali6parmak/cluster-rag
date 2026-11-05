"""The dictionary estimator (CS-50): the z-scored arms on OMCT.

CS-50 ports CS-31's measures (the informative-prior log-odds, the standard
error, the z-score, G², the conditional smoothing) to the pair unit and tests
the reading CS-49 opened: the z-score is computed **within** a question's own
table (the prior is the word's global df, the standard error the same table's),
so it can shrink a question's noisy weights but **cannot** make two questions'
weights comparable. The pre-registration registers the null and its refutation.

The sweep is CS-49's run plus the two z-scored arms, with
`sum_zscore_positive` fixed as the pre-registered primary (the UPR CS-31
fit-side selection, ported as the prior); the five CS-30 arms stay in the
readout as the spread, CS-49's numbers reproduced.

The run is one cell: `analyze-voyage-dict-z` →
`results/omct-analyze-voyage-dict-z/`; the artifact carries the `dictionary`
block (with the six measures and the estimator readout) and the
`kind: dictionary` model entry (the six arms, the comparisons, the readout).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    DictionaryConfig,
)
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

# The reference model: the same Stage-0 HF model as CS-49 (the cosine is the
# run's own; its embeddings are the Stage-0 cache hits).
REFERENCE_MODEL = "voyage-4-nano"

# The control size: the plan's default.
CONTROL_SIZE = 4000

# CS-30's four arms (the spread) plus CS-31's two z-scored arms.
SCORINGS = (
    "sum_logodds",
    "sum_positive",
    "max_logodds",
    "support_count",
    "sum_zscore",
    "sum_zscore_positive",
)

# The pre-registered primary arm (the UPR CS-31 fit-side selection, ported).
PRIMARY = "sum_zscore_positive"

cells = [
    AnalyzeOmctCell(
        label="analyze-voyage-dict-z",
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
            ),
        ),
    )
]

SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_estimator")
