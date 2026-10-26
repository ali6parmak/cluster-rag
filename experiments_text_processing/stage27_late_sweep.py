"""Stage-2.7 sweep (CS-7e): the late chunking.

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 2.7): instead of embedding each
chunk separately, embed the **whole document once** and pool its
**token-level** embeddings per chunk, so every chunk carries the document's
context.

**What was run.** Two runs, both on `voyage-4-nano`:

- `analyze-voyage-4-nano-late-words25` — the paragraph side late-chunked
  (the question side stays whole, as the early-chunked reference does);
- `analyze-voyage-4-nano-late-symwords25` — the symmetric reading: both
  sides late-chunked.

Each run carries the **early-chunked arm of the same run** beside the late
one (the Stage-2.6 precedent), so the late-vs-early comparison is **paired
by construction**: the same cells, the same candidate lists, the same
control, the same seeds. The published early reference
(`omct-analyze-voyage-4-nano-words25`, the Stage-2 ladder run) is the
cross-run check of that arm: it must reproduce it exactly.

**Why only `voyage-4-nano`.** The arm needs **token-level** embeddings, and
Ollama's `/api/embed` cannot return them: the Stage-0 models are
`embeddinggemma` (Ollama), `voyage-4-nano` (HF) and `arctic2` (Ollama), so
the arm is HF-only and the comparison is **within-model** (the plan's cost
(a)) — the late-chunked voyage-4-nano against the early-chunked
voyage-4-nano, not against the other two models' numbers.

**The geometry.** The late side uses the **same** windows as the early
reference (25 words, stride 12 — the Stage-2 verdict's paragraph-side
window), so the only difference between the two arms is *where the chunk's
vector comes from*: its own embedding (early) or the document's token-level
embeddings pooled over the chunk's character span (late).

**The pooling rules, both measured** (the reviewer's rule: measure the
alternatives rather than pick one):

- `mean` — the mean of the chunk's token vectors, then L2-normalized: the
  model's own formula (`Normalize(mean(tokens))`), restricted to the chunk;
- `norm` — every token vector is L2-normalized *before* the mean, so each
  token weighs the same whatever its magnitude.

**The aggregation rules** are the two the plan names as the early-chunked
reference: `max` (the best chunk match) and `peak:3:0.05` (the peak
shortcut).

**The pre-registered hypothesis** (the plan's, the same as Stage 2.6): the
segmentation geometry carries no label-driven **ranking** signal, so the
expected within-document AUC gain is ≈ 0; the plausible payoff is the
**retrieval** side (the positives' hit@1). And the yardstick is no longer
the embedding arms' own numbers: the decision model's arm reaches the
within-document AUC 0.7701 and the shortlist rerank the positives' hit@1
0.1565.
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

# The only Stage-0 model whose backend can return token-level embeddings
# (HF): the arm is HF-only (the plan's Stage 2.7, cost (a)).
MODEL = "voyage-4-nano"

# The control size: the seeded, proportional sample of the unjudged
# same-document cells — the same cells the Stage-0 runs draw.
CONTROL_SIZE = 4000

# The Stage-2 verdict's paragraph-side window (25 words, stride 12) — the
# same geometry on both sides of the comparison, so the only difference
# between the arms is early vs late pooling.
WINDOW_PARAMS: dict[str, object] = {"n_words": 25, "stride": 12}
WINDOWS = ChunkerSpec(name="word_windows", params=dict(WINDOW_PARAMS))

# The two aggregation rules the plan names as the early-chunked reference.
AGGREGATIONS: tuple[str, ...] = ("max", "peak:3:0.05")

# The paired bootstrap of the arm-vs-baseline AUC difference.
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

# The two pooling rules (both measured).
POOLINGS: tuple[str, ...] = ("mean", "norm")

# The two variants: the paragraph side only, and the symmetric reading
# (both sides late-chunked).
VARIANTS: tuple[tuple[str, ChunkerSpec | None], ...] = (
    ("words25", None),
    ("symwords25", WINDOWS),
)

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{MODEL}-late-{variant}",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=MODEL, embed=MODELS[MODEL]),),
            control_size=CONTROL_SIZE,
            # The early-chunked side of this run: the reference the late arm
            # is measured against (paired by construction).
            chunker=WINDOWS,
            question_chunker=question_chunker,
            # The late-chunked side: the same windows, the document embedded
            # once and the token-level embeddings pooled per chunk.
            late_chunker=WINDOWS,
            late_question_chunker=question_chunker,
            late_pooling=POOLINGS,
            aggregations=AGGREGATIONS,
            bootstrap=BOOTSTRAP,
        ),
    )
    for variant, question_chunker in VARIANTS
]

SWEEP = define_sweep(cells, dataset="omct", label="stage27_late")
