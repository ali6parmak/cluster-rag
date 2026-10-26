"""The UPR text-side extras and the label-side transform (CS-8c).

`TEXT_PROCESSING_PLAN.md` section 7 (CS-8c) runs the track's remaining
text-side methods on UPR — including the ones that failed on OMCT, so that
"it does not work here" is a measurement and not an assumption.

**Three things, in two runs:**

1. **The chunk header** (CS-7d on OMCT, where it did **not** ship): the
   text's own first sentence, prepended to every chunk, in both modes
   (`others` — the header on every chunk but the first — and `all` — the
   literal reading). The other headings stay ruled out: the document title is
   document identity and a corpus-level statistic is forbidden.
2. **The label-side transform** (the analogue of OMCT's `keep` vs
   `strip_block`): the label text as the **bare name** vs the config
   **template** ("This recommendation is about {label}."). Both variants are
   in the same run (the family's own prior label run), so the comparison is
   paired, with the non-gold labels' level and rank shift beside it.
3. **The late chunking** (CS-7e on OMCT, the ladder's one winner): the text
   embedded **once** and its token-level embeddings pooled per chunk, with
   the model's own `Normalize(mean(tokens))` formula and the per-token-
   normalized variant. It needs **token-level** embeddings, which Ollama's
   `/api/embed` cannot return, so it is **HF-only**: the run uses
   `voyage-4-nano` and the comparison is **within-model** — the late-chunked
   arm against the early-chunked arm of the same run (the reference arm is
   the voyage champion). The CLI says so before the run (the family's
   `late` block in the artifact records it too).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import ChunkerSpec, UprAnalysisConfig, UprArmConfig
from cluster_rag.sweep import AnalyzeUprCell, define_sweep

# The standing champion (PLAN.md Phase 4): embeddinggemma, the fusion of the
# three retrieval signals, the kNN's k=10 and the `top_k k=2` decision.
CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"

# The label-side transform's second variant: the bare label name (the
# champion's label side is the "about" template).
NAME_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-name"

# The late chunking's model: voyage-4-nano (HF — the only pool model with
# token-level embeddings), with its own champion runs.
VOYAGE_EMBEDDING_RUN = "upr-embed-voyage-4-nano"
VOYAGE_LABEL_EMBEDDING_RUN = "upr-labelembed-voyage-4-nano-about"
VOYAGE_EVAL_RUN = "upr-eval-fusionc-voyage-4-nano-k10-c1"

LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)

# The folding rules of the arms (the plan's Stage-2 menu).
AGGREGATIONS: tuple[str, ...] = (
    "max",
    "top_k_mean:2",
    "mean",
    "local_global:0.75:max",
    "peak:3:0.05",
)

# The chunk header's inner chunker: the atoms (the sentences — the CS-8b
# ladder's best chunker on UPR) and the word windows, so the header is
# measured on the geometry that carries the gain.
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})
WORD_WINDOWS = ChunkerSpec(name="word_windows", params={"n_words": 15, "stride": 7})
HEADER_OTHERS = ChunkerSpec(
    name="header",
    params={"chunker": "atoms", "chunker_params": {"split_clauses": True}, "mode": "others"},
)
HEADER_ALL = ChunkerSpec(
    name="header",
    params={"chunker": "atoms", "chunker_params": {"split_clauses": True}, "mode": "all"},
)

# The families of the champion's run: the chunk header (both modes) and the
# label-side transform (the two label variants, in the pair-scoring family
# where the label side carries the whole signal).
CHAMPION_FAMILIES: tuple[UprArmConfig, ...] = (
    UprArmConfig(
        name="header_atoms_others",
        chunker=HEADER_OTHERS,
        aggregations=AGGREGATIONS,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
    ),
    UprArmConfig(
        name="header_atoms_all",
        chunker=HEADER_ALL,
        aggregations=AGGREGATIONS,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
    ),
    UprArmConfig(
        name="labeltext_about",
        chunker=ATOMS,
        aggregations=AGGREGATIONS,
        predictor="zeroshot",
        label_embedding_run_id=CHAMPION_LABEL_EMBEDDING_RUN,
    ),
    UprArmConfig(
        name="labeltext_name",
        chunker=ATOMS,
        aggregations=AGGREGATIONS,
        predictor="zeroshot",
        label_embedding_run_id=NAME_LABEL_EMBEDDING_RUN,
    ),
)

# The late chunking's families (the voyage run): the early-chunked arm (the
# within-model reference) and the two late poolings.
VOYAGE_FAMILIES: tuple[UprArmConfig, ...] = (
    UprArmConfig(
        name="late_early_words25",
        chunker=ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12}),
        aggregations=AGGREGATIONS,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
    ),
    UprArmConfig(
        name="late_mean_words25",
        chunker=ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12}),
        aggregations=AGGREGATIONS,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        late_pooling="mean",
    ),
    UprArmConfig(
        name="late_norm_words25",
        chunker=ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12}),
        aggregations=AGGREGATIONS,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        late_pooling="norm",
    ),
)

cells = [
    AnalyzeUprCell(
        label="upr-analyze-extras",
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        embed=MODELS["embeddinggemma"],
        analysis=UprAnalysisConfig(
            arms=CHAMPION_FAMILIES,
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=CHAMPION_EVAL_RUN,
        ),
    ),
    AnalyzeUprCell(
        label="upr-analyze-late",
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=VOYAGE_EMBEDDING_RUN,
        label_embedding_run=VOYAGE_LABEL_EMBEDDING_RUN,
        embed=MODELS["voyage-4-nano"],
        analysis=UprAnalysisConfig(
            arms=VOYAGE_FAMILIES,
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=VOYAGE_EVAL_RUN,
        ),
    ),
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_extras")
