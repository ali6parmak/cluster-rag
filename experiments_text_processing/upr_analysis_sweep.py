"""The UPR side's first sweep: the whole-text reference arm (CS-8a).

`TEXT_PROCESSING_PLAN.md` section 7 (CS-8a) measures the UPR corpus with the
text-processing track's own diagnostics, using the **main pipeline's**
standing champion as the scoring method: embeddinggemma, the fusion
(a=0.5 knn / b=1.0 prototype / c=1.0 zero-shot, knn k=10) and the
`top_k k=2` decision rule — the config that reached the val micro-F1 0.5479
of Phase 4.

The cell is one run (`upr-analyze-champion`). Its whole-text **reference arm**
is the pipeline's own scoring of the val split, so the run must reproduce the
matching `evaluate` run (`upr-eval-fusionc-embeddinggemma-k10-c1`) field by
field — the guard, the UPR analogue of `tools/verify_analysis_baseline.py`.

The embedding runs are named explicitly (never derived): the champion's
`upr-embed-embeddinggemma` (the document + query matrices) and the zero-shot
label vectors' `upr-labelembed-embeddinggemma-about` (the label texts are the
`about` template — the Phase-4 champion's label side).
"""

from __future__ import annotations

from cluster_rag.config import UprAnalysisConfig
from cluster_rag.sweep import AnalyzeUprCell, define_sweep

# The standing champion (PLAN.md Phase 4): embeddinggemma, the fusion of the
# three retrieval signals, the kNN's k=10 and the `top_k k=2` decision.
CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"

# The run whose numbers the whole-text reference arm must reproduce.
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"

# The text-length buckets (characters): the axis the chunking hypothesis is
# about on UPR — a pooled number hides the long-text tail. The edges split the
# val split's texts into 155 / 366 / 114 / 25 (the corpus averages 151
# characters, so the first two buckets carry most of it and the last one is
# the long-text tail the hypothesis is about).
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)

cells = [
    AnalyzeUprCell(
        label="upr-analyze-champion",
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        analysis=UprAnalysisConfig(
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=CHAMPION_EVAL_RUN,
        ),
    )
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_analysis")
