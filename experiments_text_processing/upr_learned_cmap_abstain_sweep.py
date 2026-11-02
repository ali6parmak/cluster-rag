"""The corpus map's **abstention** in the UPR learned pack (the track's CS-37).

The map's per-cell score has two states today: evidence (a number) and no
evidence (the **floor** — the row's smallest seen score minus one, which ranks
the label last in every row). The floor is what makes CS-32's pack lose the
eight held-out labels outright (the unseen bucket 0.4867 → **0.0000**): the
combiner learns to trust the channel, and the channel says "impossible" where
it should say "I don't know".

This run measures the fix: a label with **no table at all** **abstains** — its
cells carry ``NaN`` ("no evidence") instead of the floor. The value is
*absent*, so the combiner can tell the two apart; the gradient-boosted
combiner takes NaN natively.

**Five arms in one artifact**, so every comparison is paired by construction:

- the **map's arm** (the map's column with the floor) — CS-32's arm;
- the **control** (the same pack without the group);
- the **label-blind counterpart** (the blind map with the floor);
- the **abstention** (the map's column with the abstention) — the new arm;
- the **abstention's label-blind counterpart** (the blind map with the
  abstention) — the new control. The blind map's eight labels have empty
  tables too, so this arm gets the same recovery of the held-out labels:
  that is the abstention's doing, not a label-driven gain.

**The guard.** The pack is fitted on train and the map is built from train, so
the fit side's map is **cross-fit** (5 folds over the train split, each fold's
cells scored against a map built from the other folds), and the abstention
rides into it: a label with no example in a fold's training texts abstains in
that fold.

The run is the ``analyze-upr`` command (the track's UPR artifact), so the row
lands in the **track's** UPR table (``docs/experiments_table_text_processing_upr.md``)
under the sweep's section.

    uv run cluster-rag sweep --file experiments_text_processing/upr_learned_cmap_abstain_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    UprAnalysisConfig,
    UprCorpusMapFeatureConfig,
    UprLearnedConfig,
    UprLearnedFloorsConfig,
    UprLearnedReportingConfig,
)
from cluster_rag.sweep import AnalyzeUprCell, define_sweep

# The champion's own scoring (the reference arm every number is read against).
CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"

# The run whose numbers the whole-text reference arm must reproduce (the
# track's guard: the reference arm IS the main pipeline's own scoring).
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"

# The pack's groups: CS-32's configuration, unchanged (the abstention changes
# the map's column, not the pack).
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CMAP_PACK: tuple[str, ...] = (*FULL_PACK, "corpus_map")

# The pack's chunker: the **atoms** (the CS-8b ladder's best chunker on UPR).
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The corpus map's feature: the z-scored "evidence for" score (CS-31's
# fit-side selection), the fit side's map cross-fit over 5 folds, the
# label-blind counterpart beside it and -- the change set's question -- the
# **abstention** (CS-37: a label with no table at all carries NaN instead of
# the floor).
CMAP_FEATURE = UprCorpusMapFeatureConfig(
    enabled=True,
    scoring="sum_zscore_positive",
    folds=5,
    label_blind=True,
    abstain=True,
)

cells = [
    AnalyzeUprCell(
        label="upr-analyze-cmap-abstain",
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        embed=MODELS["embeddinggemma"],
        analysis=UprAnalysisConfig(
            arms=(),
            reference_eval_run=CHAMPION_EVAL_RUN,
            learned=UprLearnedConfig(
                enabled=True,
                groups=CMAP_PACK,
                top_k=3,
                chunker=ATOMS,
                profile_top_k=3,
                profile_ngram_size=5,
                combiner=CombinerSpec(name="histgb"),
                control_weight=0.3,
                floors=UprLearnedFloorsConfig(
                    chance=True,
                    control_role=True,
                    random_vectors=True,
                    permutation_resamples=5,
                    random_seed=42,
                ),
                reporting=UprLearnedReportingConfig(
                    enabled=True,
                    bins=10,
                    threshold_quantiles=99,
                ),
                corpus_map_feature=CMAP_FEATURE,
                seed=42,
            ),
        ),
    )
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_cmap_abstain")
