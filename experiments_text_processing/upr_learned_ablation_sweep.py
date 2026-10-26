"""The UPR feature-group ablation: which block carries the signal (CS-8e).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8e) re-asks the OMCT track's
Stage-3.1b question on the UPR vocabulary: **the lexical block's transfer is
the headline**, because on OMCT the learned pack's gain was the *lexical*
block's and not the embedding geometry's.

The arms are the feature-group sets (the groups are a config knob, so the
pack never changes — only the columns the combiner sees):

- `embed` — the embedding geometry alone (the champion's own view);
- `lex` — the soft lexical overlap and the char n-grams alone;
- `noembed` — the lexical block plus the chunk profile (no embedding at all);
- `nolex` — the embedding block plus the chunk profile (no lexical block).

The `base` (the CS-8d pack), `full` and `profile` arms are in the chunk
profile's sweep (`upr_learned_profile_sweep.py`), so the whole grid is read
across the two sweeps.

The floors: the label-permutation chance floor and the control-role floor on
every arm (they are the "no label information" references), and the
random-vector floor only where the arm has embedding features (for the
lexical-only arms the random pack *is* the real pack, so the floor would be
the arm itself).
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

# The ablation's arms: the group set and whether the random-vector floor
# applies (it needs an embedding feature to replace).
ARMS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("embed", ("embedding",), True),
    ("lex", ("lexical",), False),
    ("noembed", ("lexical", "chunk_profile"), False),
    ("nolex", ("embedding", "chunk_profile"), True),
)

cells = [
    AnalyzeUprCell(
        label=f"upr-analyze-learned-abl-{name}",
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
                    random_vectors=random_vectors,
                    permutation_resamples=5,
                    random_seed=42,
                ),
                seed=42,
            ),
        ),
    )
    for name, groups, random_vectors in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_ablation")
