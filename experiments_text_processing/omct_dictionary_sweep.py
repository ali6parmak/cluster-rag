"""The dictionary port to OMCT (CS-48): the question map, model-free.

`TEXT_PROCESSING_PLAN.md` §7.9 ports the UPR dictionary chain to OMCT. The
first change set is the **question map**: the reviewed pairs' own
word↔question association, built from subset A only (the unjudged control,
subset B, never enters it), with the counts, the lift and the 0.5-smoothed
log-odds behind every (question, word) pair — plus the map's cross-fit
availability (the share of the 1,520 pairs whose question has an out-of-fold
table and the share that receives at least one positive word).

**Why the cell has no embedder.** The map is a *description* of the reviewed
pairs: it reads no embedding and calls no model. The command's
`analysis.embedders` may therefore be empty when the run carries the
`analysis.dictionary` section — the artifact's `dictionary` block is
model-independent, and a run that scores embedding arms (a later change set)
brings its own cell. The control set is still drawn (the same seeded 4,000
unjudged cells as every other run): it is recorded in the artifact and is
**not** handed to the map.

The run is one cell: `analyze-dictmap` → `results/omct-analyze-dictmap/`, the
artifact `analysis.json` with the top-level `dictionary` block.
"""

from __future__ import annotations

from cluster_rag.config import AnalysisConfig, DictionaryConfig
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

# The control size: the plan's default, the same seeded proportional sample of
# the unjudged same-document cells every track run draws.
CONTROL_SIZE = 4000

cells = [
    AnalyzeOmctCell(
        label="analyze-dictmap",
        analysis=AnalysisConfig(
            embedders=(),
            control_size=CONTROL_SIZE,
            dictionary=DictionaryConfig(),
        ),
    )
]

SWEEP = define_sweep(cells, dataset="omct", label="omct_dictionary")
