"""The UPR gradient-boosted combiner: is the learner the lever? (CS-8e).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8e) re-runs the OMCT track's Stage
3.2 on the UPR data: the **gradient-boosted combiner** (scikit-learn's
``HistGradientBoostingClassifier``, the pinned defaults, **no grid**) behind
the same ``Combiner`` port, with the feature-group selection as the knob.

The pre-registered expectation (from the OMCT verdict, CS-6b): the learner is
**not the lever** — on OMCT the booster was below the logistic regression on
8 of the 9 arms and ahead in 5 of the 27 per-rotation paired comparisons, and
its gain was diffuse and leaned harder on the embedding block (the block the
ablation showed carries no label-driven signal).

The arms: the same three group sets as on OMCT — `full` (the whole pack the
repo can build), `lex-only` and `embed-only` — so the two learners are
compared on the same feature sets.
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

# The arms: the group set and whether the random-vector floor applies.
ARMS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("full", ("embedding", "lexical", "chunk_profile"), True),
    ("lex-only", ("lexical",), False),
    ("embed-only", ("embedding",), True),
)

cells = [
    AnalyzeUprCell(
        label=f"upr-analyze-learned-histgb-{name}",
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
                combiner=CombinerSpec(name="histgb"),
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

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_histgb")
