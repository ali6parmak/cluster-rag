"""The UPR segmentation ladder: the track's whole ladder on UPR (CS-8b).

`TEXT_PROCESSING_PLAN.md` section 7 (CS-8b) runs the **whole** segmentation
ladder on the UPR data — the methods the track measured on OMCT, including the
ones that failed there, so that "it does not work here" is a measurement and
not an assumption.

**The pre-registered expectation** (written down before the run, in the
track's plan §7.1): the chunking axis is a **null on the ranking side** (the
UPR texts average 153 characters and the track's smallest window was 40 words
≈ 250 characters, so a text is usually one chunk and the fold is the
identity), with a possible small win on the **retrieval** side for the
long-text tail. The length buckets are therefore reported separately: a
pooled UPR number would hide exactly the effect the hypothesis is about.

**The three families** (the reviewer named two; the third is the null-check):

- `pair` — the **pair-scoring** family: the text's chunks against the
  zero-shot label vectors (`score(text, label) = aggregate over the chunks
  of cos(chunk, label_vector)`), the `zeroshot` predictor's geometry;
- `retr` — the **retrieval** family: every chunk is a query against the
  train-only index (the champion's `fusion`: a=0.5 knn / b=1.0 prototype /
  c=1.0 zero-shot, knn k=10), and the chunk-level scores are folded with
  the same rules;
- `sym` — the **symmetric** family: the **label** side is chunked too (the
  null-check — the ladder's verdict says never chunk the question, i.e.
  never chunk the label, but it is run anyway because it is cheap, and the
  UPR labels are 1-3 words, so the artifact records the degeneracy).

**The chunkers** (the ladder's text side): the atoms, the fixed word windows
(N = 15 / 25 / 40, stride ~N/2), the overlapping sentence windows (2
sentences, step 1) and the semantic boundaries. The semantic chunker's
**threshold is selected on the fit side** (the train split, with a 3-fold
cross-fit over the train records reporting the transfer); the grid, the
chosen threshold and the per-fold numbers go into the artifact.

**The folding rules** (the plan's Stage-2 menu plus the Stage-1 continuity):
`max`, `top_k_mean:2`, `top_k_mean:3`, `mean`, `decayed_top_k:3:0.5:0.25`,
`local_global:0.5:max`, `local_global:0.75|max` and `peak:3:0.05`.

**The reference arm** (the whole-text champion) is computed in the same run,
so the pairing is by construction: same texts, same split, same decision
rule, one artifact.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    UprAnalysisConfig,
    UprArmConfig,
    UprWeightGridConfig,
)
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
# about on UPR — a pooled number hides the long-text tail.
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)

# The folding rules of the ladder.
AGGREGATIONS: tuple[str, ...] = (
    "max",
    "top_k_mean:2",
    "top_k_mean:3",
    "mean",
    "decayed_top_k:3:0.5:0.25",
    "local_global:0.5:max",
    "local_global:0.75:max",
    "peak:3:0.05",
)

# The ladder's chunkers. The text side is chunked; the label side (the
# symmetric family) uses the same word windows, whose label side is
# degenerate by construction (the labels are 1-3 words).
WORD_WINDOWS = ChunkerSpec(name="word_windows", params={"n_words": 15, "stride": 7})
WORD_WINDOWS_25 = ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12})
WORD_WINDOWS_40 = ChunkerSpec(name="word_windows", params={"n_words": 40, "stride": 20})
SENTENCE_WINDOWS = ChunkerSpec(name="sentence_windows", params={"n_sentences": 2, "step": 1})
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})
# The semantic chunker's atoms are embedded with the *query* kind (the text
# side of the UPR pipeline is the query side).
SEMANTIC = ChunkerSpec(name="semantic", params={"embed_kind": "query"})

CHUNKERS: tuple[ChunkerSpec, ...] = (
    ATOMS,
    WORD_WINDOWS,
    WORD_WINDOWS_25,
    WORD_WINDOWS_40,
    SENTENCE_WINDOWS,
    SEMANTIC,
)

# The swept grid of the semantic chunker's threshold (the fit side chooses
# one; the grid is recorded in the artifact).
SEMANTIC_THRESHOLDS: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

# The three families of the ladder: the pair-scoring family, the retrieval
# family and the symmetric (label-side chunked) null-check. The family name
# carries the chunker, so the arm names read "retr_words15_max".
FAMILIES: tuple[UprArmConfig, ...] = tuple(
    [
        UprArmConfig(
            name=f"pair_{slug}",
            chunker=chunker,
            aggregations=AGGREGATIONS,
            predictor="zeroshot",
        )
        for slug, chunker in (
            ("atoms", ATOMS),
            ("words15", WORD_WINDOWS),
            ("words25", WORD_WINDOWS_25),
            ("words40", WORD_WINDOWS_40),
            ("sent2", SENTENCE_WINDOWS),
            ("semantic", SEMANTIC),
        )
    ]
    + [
        UprArmConfig(
            name=f"retr_{slug}",
            chunker=chunker,
            aggregations=AGGREGATIONS,
            predictor=CHAMPION_PREDICTOR,
            params=CHAMPION_PARAMS,
        )
        for slug, chunker in (
            ("atoms", ATOMS),
            ("words15", WORD_WINDOWS),
            ("words25", WORD_WINDOWS_25),
            ("words40", WORD_WINDOWS_40),
            ("sent2", SENTENCE_WINDOWS),
            ("semantic", SEMANTIC),
        )
    ]
    + [
        UprArmConfig(
            name="sym_words15",
            chunker=WORD_WINDOWS,
            aggregations=AGGREGATIONS,
            predictor="zeroshot",
            label_chunker=WORD_WINDOWS,
        )
    ]
)

cells = [
    AnalyzeUprCell(
        label="upr-analyze-ladder",
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        embed=MODELS["embeddinggemma"],
        analysis=UprAnalysisConfig(
            arms=FAMILIES,
            weight_grid=UprWeightGridConfig(enabled=True),
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=CHAMPION_EVAL_RUN,
            semantic_thresholds=SEMANTIC_THRESHOLDS,
            semantic_folds=3,
        ),
    )
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_ladder")
