"""Stage-2.6 sweep (CS-7d): the chunk header.

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 2.6): the classic "contextual chunk
header" idea, in the one form that is legal on this data — the header is the
paragraph's **own first sentence** (pair-local, label-free, no document
identity, and it keeps one embedding per chunked text instead of the ~175,000
a question-stem header would need).

**What was run.** Six runs (3 Stage-0 models x 2 header modes) with the
Stage-2 verdict's paragraph-side window (25 words, stride 12) and two
aggregation rules (`max` and `peak:3:0.05`), each with the **no-header** arm
of the same run beside it — so the header is measured against the no-header
arm **paired by construction** (the same cells, the same candidate lists).

**The two modes, both measured** (the reviewer's rule: measure the
alternatives rather than picking one):

- `others`: the header is prepended to every chunk **except the first** (the
  first chunk already starts with the header's text, so prefixing it would
  only duplicate it) — the chunks' own content is unchanged, so the arm
  measures the header's *context* and nothing else;
- `all`: the header is prepended to every chunk (the literal reading), so the
  first chunk carries its first sentence twice.

**The pre-registered hypothesis** (the plan's): the chunk geometry carries no
label-driven **ranking** signal (CS-6d's random-vector floor), so the
expected within-document AUC gain is ≈ 0; the plausible payoff is the
**retrieval** side (the positives' hit@1). And the yardstick is now much
higher than the embedding arms' own numbers: the decision model's arm reaches
the within-document AUC 0.7701 and the shortlist rerank the positives' hit@1
0.1565, so a header that cannot beat the cosine's 0.1050-0.1355 is not
worth shipping.
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

# The Stage-0 models (the cosine's reference and the header's own arm).
STAGE0_MODELS: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: the seeded, proportional sample of the unjudged
# same-document cells — the same cells the Stage-0 runs draw.
CONTROL_SIZE = 4000

# The Stage-2 verdict's paragraph-side window (25 words, stride 12).
WINDOW_PARAMS: dict[str, object] = {"n_words": 25, "stride": 12}
WINDOWS = ChunkerSpec(name="word_windows", params=dict(WINDOW_PARAMS))

# The two aggregation rules the header is measured with.
AGGREGATIONS: tuple[str, ...] = ("max", "peak:3:0.05")

# The paired bootstrap of the arm-vs-baseline AUC difference.
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

# The two header modes (both measured).
HEADER_MODES: tuple[str, ...] = ("others", "all")


def _header_chunker(mode: str) -> ChunkerSpec:
    """The header chunker for one mode (it wraps the same windows)."""
    return ChunkerSpec(
        name="header",
        params={
            "chunker": "word_windows",
            "chunker_params": dict(WINDOW_PARAMS),
            "mode": mode,
        },
    )


cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-header-{mode}",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=model, embed=MODELS[model]),),
            control_size=CONTROL_SIZE,
            chunker=WINDOWS,
            header_chunker=_header_chunker(mode),
            aggregations=AGGREGATIONS,
            bootstrap=BOOTSTRAP,
        ),
    )
    for model in STAGE0_MODELS
    for mode in HEADER_MODES
]

SWEEP = define_sweep(cells, dataset="omct", label="stage26_header")
