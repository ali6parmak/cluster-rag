"""Stage-2 ladder sweep: the whole segmentation ladder (CS-4).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 2) measures the **segmentation
ladder** on the OMCT pairwise data: the fixed word windows (exp03), the
overlapping sentence windows (exp02), the semantic boundaries (exp18(ii)) and
the two symmetric variants (exp05, exp10 — both sides chunked). The reviewer
decided to run the **whole ladder**, not a reduced grid (the plan's Stage 2.4:
"the whole ladder runs on both, not just the winner"), and to carry the three
Stage-0 models, so the ladder's arms sit next to the Stage-0 and Stage-1
artifacts and the n = 3 argument keeps holding.

Every cell is its own run (`omct-analyze-<model>-<variant>`) and holds **all**
the arms: the full-text baseline arm (the Stage-0 scoring, the reference
every other arm is compared with) and the chunked arms, one per aggregation
rule. The pairing is therefore by construction — same pairs, same control
cells, one run — and the artifact carries the paired bootstrap of the AUC
difference next to the control-side counterpart (the track's reporting rule).

**The aggregation set** (the plan's Stage-2 menu plus the Stage-1
continuity):

- ``max`` — the best chunk match (also exp06's local rule);
- ``top_k_mean:2`` — the mean of the two best chunk matches (the Stage-1
  continuity: the broad evidence without a decay);
- ``decayed_top_k:3:0.5:0.25`` — exp07, a decaying average of the three best
  chunk matches minus a spread penalty;
- ``local_global:0.5:max`` and ``local_global:0.75:max`` — exp06, the
  local+global blend; with the baseline (alpha = 0) and the ``max`` arm
  (alpha = 1) the two ends of the alpha path;
- ``peak:3:0.05`` — exp08, the peak shortcut (a margin over the pair's own
  broad evidence, never an absolute cut: the track forbids absolute score
  bars on this data).

**The swept parameter.** The semantic chunker's threshold is not a fixed
value: the plan's Stage 2.3 sweeps it **on fit-side documents only**. The
driver chooses it per document by the document-level cross-fit (the threshold
of a document comes from the rotation that holds that document out), so no
document ever sees its own fitted value. The artifact records the grid, the
per-rotation fit-side numbers and the chosen value.

The models are the three Stage-0 models (`embeddinggemma`, `voyage-4-nano`,
`arctic2`); their configs come from the shared pool
(`experiments/embedder_pool.py`), so the track's runs use the same model ids,
prefixes and pinned revisions as the main pipeline's sweeps.

The control set is model-independent and seeded, so every run draws **exactly
the same control cells** as the Stage-0 and Stage-1 runs (the cell ids are in
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

# The Stage-0 models (the same three the Stage-0 and Stage-1 artifacts used).
STAGE2_MODELS: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: the seeded, proportional sample of the unjudged
# same-document cells (the plan's default). The same size and seed as the
# Stage-0 runs, so the control cells are the same cells.
CONTROL_SIZE = 4000

# The aggregation rules every ladder variant is scored with (the plan's
# Stage-2 menu plus the Stage-1 continuity).
AGGREGATIONS: tuple[str, ...] = (
    "max",
    "top_k_mean:2",
    "decayed_top_k:3:0.5:0.25",
    "local_global:0.5:max",
    "local_global:0.75:max",
    "peak:3:0.05",
)

# The swept grid of the semantic chunker's threshold. The boundary is cut
# where the adjacent atoms' cosine similarity drops **below** the threshold,
# so a low threshold cuts almost everywhere and a high one hardly at all.
SEMANTIC_THRESHOLDS: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

# The paired bootstrap of the arm-vs-baseline AUC difference (Stage 2.5's
# statistic: without it a +/-0.03 AUC effect is unresolvable).
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

# The ladder: one chunker variant per run. The symmetric variants chunk the
# question side too (exp05: both sides sentence-windowed, exp10: both sides
# word-chunked).
LADDER: tuple[tuple[str, ChunkerSpec, ChunkerSpec | None], ...] = (
    ("words15", ChunkerSpec(name="word_windows", params={"n_words": 15, "stride": 7}), None),
    ("words25", ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12}), None),
    ("words40", ChunkerSpec(name="word_windows", params={"n_words": 40, "stride": 20}), None),
    ("sent2", ChunkerSpec(name="sentence_windows", params={"n_sentences": 2, "step": 1}), None),
    (
        "semantic",
        ChunkerSpec(name="semantic", params={"threshold": 0.5}),
        None,
    ),
    (
        "symwords25",
        ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12}),
        ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12}),
    ),
    (
        "symsent2",
        ChunkerSpec(name="sentence_windows", params={"n_sentences": 2, "step": 1}),
        ChunkerSpec(name="sentence_windows", params={"n_sentences": 2, "step": 1}),
    ),
)

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-{variant}",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=model, embed=MODELS[model]),),
            control_size=CONTROL_SIZE,
            chunker=chunker,
            question_chunker=question_chunker,
            aggregations=AGGREGATIONS,
            semantic_thresholds=SEMANTIC_THRESHOLDS if variant == "semantic" else (),
            bootstrap=BOOTSTRAP,
        ),
    )
    for model in STAGE2_MODELS
    for variant, chunker, question_chunker in LADDER
]

SWEEP = define_sweep(cells, dataset="omct", label="stage2_ladder")
