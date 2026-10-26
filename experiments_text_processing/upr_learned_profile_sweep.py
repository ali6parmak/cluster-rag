"""The UPR chunk profile: the pack's fifth group (CS-8e).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8e) re-runs the OMCT track's Stage
3.3 on the UPR data: the **chunk profile** as the pack's fifth group, with the
same three arms as on OMCT —

- `base` — the CS-8d pack (the embedding and the lexical groups), the control;
- `full` — the base plus the chunk profile;
- `profile` — the chunk profile **alone**.

The profile says **where** the text and the label match, not how similar they
are on average: the text is cut into its sentence units and its word pieces,
every unit's cosine against the label is ranked, and the alignment kinds say
which granularity produced the best match and whether the aligned unit and
the label's name share a content word. The **label side is one unit** (the UPR
labels are 1-3 words, so the label side is the identity — the CS-8b verdict),
so the profile says how well the label matches the text's finest units.

The pre-registered expectation (from the OMCT verdict, CS-6c): the chunk
profile is **not the lever** — on OMCT the block alone was indistinguishable
from the full-text cosine and bought a small gain in the full pack on one
model of three. On UPR the same question is re-asked, and the answer is
measured.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    UprAnalysisConfig,
    UprLearnedConfig,
    UprLearnedFloorsConfig,
)
from cluster_rag.sweep import AnalyzeUprCell, define_sweep

CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The three arms: the base pack, the full pack and the profile alone.
ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("base", ("embedding", "lexical")),
    ("full", ("embedding", "lexical", "chunk_profile")),
    ("profile", ("chunk_profile",)),
)

cells = [
    AnalyzeUprCell(
        label=f"upr-analyze-learned-cp-{name}",
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        embed=MODELS["embeddinggemma"],
        analysis=UprAnalysisConfig(
            arms=(),
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=CHAMPION_EVAL_RUN,
            learned=UprLearnedConfig(
                enabled=True,
                groups=groups,
                top_k=3,
                chunker=ATOMS,
                profile_top_k=3,
                profile_ngram_size=5,
                combiner=CombinerSpec(name="logistic"),
                control_weight=0.3,
                floors=UprLearnedFloorsConfig(
                    chance=True,
                    control_role=True,
                    random_vectors=True,
                    permutation_resamples=5,
                    random_seed=42,
                ),
                seed=42,
            ),
        ),
    )
    for name, groups in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_profile")
