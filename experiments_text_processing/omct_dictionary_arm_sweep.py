"""The dictionary arm (CS-49): the question map as a pair scorer on OMCT.

The map of CS-48 is a description; this sweep scores every reviewed pair with
it — the sum of the question's map weights over the paragraph's words — behind
the same 5-fold cross-fit, and compares the **primary arm** (`sum_positive`,
fixed a priori by CS-30's UPR prior) with the run's own plain full-text cosine.
The four CS-30 arms are all scored, so the primary's spread is visible; the
**label-blind counterpart** (the questions permuted across the pairs) is scored
beside it.

**One model, on purpose.** The arm is model-independent (it reads no
embedding), but the comparison needs a reference cosine, and
`analysis.dictionary.scorings` is validated to need exactly one embedder. The
model is `voyage-4-nano` — the Stage-0 model whose backend is HF, so the run
works without an Ollama server; its Stage-0 embeddings are in the cache (225
paragraphs + 294 questions), so the run costs **0 new model calls**.

The run is one cell: `analyze-voyage-dict` →
`results/omct-analyze-voyage-4-nano-dict/`, the artifact `analysis.json` with
the `dictionary` block (the map, reproduced) and the `kind: dictionary` model
entry (the four arms, the comparison, the readout).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    DictionaryConfig,
)
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

# The reference model: the Stage-0 model with an HF backend (the cache is warm
# from the Stage-0 voyage-4-nano run).
REFERENCE_MODEL = "voyage-4-nano"

# The control size: the plan's default, the same seeded proportional sample of
# the unjudged same-document cells every track run draws.
CONTROL_SIZE = 4000

# CS-30's four arms, in the report's order.
SCORINGS = ("sum_logodds", "sum_positive", "max_logodds", "support_count")

cells = [
    AnalyzeOmctCell(
        label="analyze-voyage-dict",
        analysis=AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name=REFERENCE_MODEL, embed=MODELS[REFERENCE_MODEL]),
            ),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(
                include_map=True,
                folds=5,
                scorings=SCORINGS,
            ),
        ),
    )
]

SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary_arm")
