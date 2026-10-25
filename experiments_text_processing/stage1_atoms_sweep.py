"""Stage-1 atoms sweep: the paragraph side's structural decomposition (CS-3).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 1.1a) measures the paragraph side's
internal structure: a paragraph is split into sentence/clause **atoms**, every
atom is embedded, and the atom-level scores are folded into one pair score by
an aggregation rule (``max``, ``top_k_mean:2``, ``top_k_mean:3``, ``mean``).
The question side stays unchunked (the question-side *chunker* is out of
scope) and the free ``keep`` vs ``strip_block`` question-transform comparison
runs as one more arm of the same run.

Every cell is its own run (`omct-analyze-<model>-atoms`) and holds **all** the
arms: the full-text baseline arm (the Stage-0 scoring, the reference every
other arm is compared with) and the chunked arms. The pairing is therefore by
construction — same pairs, same control cells, one run — and the artifact
carries the paired bootstrap of the AUC difference next to the control-side
counterpart (the track's reporting rule).

The models are the three Stage-0 models (`embeddinggemma`, `voyage-4-nano`,
`arctic2`), so the new arms sit directly next to the Stage-0 artifacts and
the "three symptoms at once" argument keeps its n = 3. Their configs come from
the shared pool (`experiments/embedder_pool.py`), so the track's runs use the
same model ids, prefixes and pinned revisions as the main pipeline's sweeps.

The control set is model-independent and seeded, so the three runs draw
**exactly the same control cells** as the Stage-0 runs (the cell ids are in
every artifact).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    ChunkerSpec,
    PairedBootstrapConfig,
)
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

# The Stage-0 models (the same three the Stage-0 artifact used).
STAGE1_MODELS: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: the seeded, proportional sample of the unjudged
# same-document cells (the plan's default; the AI stratum is the binding
# constraint). The same size and seed as the Stage-0 runs, so the control
# cells are the same cells.
CONTROL_SIZE = 4000

# The paragraph side's chunker: sentence/clause atoms (the exp18(i) lineage).
CHUNKER = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The aggregation rules of the chunked arms (the plan's Stage 1.1a set).
AGGREGATIONS: tuple[str, ...] = ("max", "top_k_mean:2", "top_k_mean:3", "mean")

# The paired bootstrap of the arm-vs-baseline AUC difference (Stage 2.5's
# statistic, needed already at Stage 1: without it a +/-0.03 AUC effect is
# unresolvable).
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{name}-atoms",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=name, embed=MODELS[name]),),
            control_size=CONTROL_SIZE,
            chunker=CHUNKER,
            aggregations=AGGREGATIONS,
            question_transforms=("keep",),
            bootstrap=BOOTSTRAP,
        ),
    )
    for name in STAGE1_MODELS
]

SWEEP = define_sweep(cells, dataset="omct", label="stage1_atoms")
