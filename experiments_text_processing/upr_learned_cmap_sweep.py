"""The corpus map in the UPR learned pack (the track's CS-32): one arm.

The map's per-cell score becomes a **feature group** of the UPR pack, so the
learned combiner can use the corpus's word statistics beside the embedding
and lexical features. The run carries three arms in one artifact:

- the **real** arm (the pack with the ``corpus_map`` group);
- the **control** (the same pack without the group) — the difference against
  the real arm is paired by construction;
- the **label-blind counterpart** (the same pack, but the map is built with
  the labels permuted among the train records) — it says how much of the
  feature's contribution is the map's *construction* and not the corpus's
  word↔label association.

**The guard.** The pack is fitted on train and the map is built from train, so
the fit side's map is **cross-fit** (each fold's cells scored against a map
built from the other folds): otherwise the feature would be optimistically
biased exactly where the combiner learns its weight.

The run is the ``analyze-upr`` command (the track's UPR artifact), so the row
lands in the **track's** UPR table (``docs/experiments_table_text_processing_upr.md``)
under the sweep's section.

    uv run cluster-rag sweep --file experiments_text_processing/upr_learned_cmap_sweep.py
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

# The pack's groups: the CS-13 pack's groups plus the new one. The chunk
# profile is in (the CS-8e grid's best arm) and the catalog's groups are out
# (the catalog is not this change set's question).
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CMAP_PACK: tuple[str, ...] = (*FULL_PACK, "corpus_map")

# The pack's chunker: the **atoms** (the CS-8b ladder's best chunker on UPR).
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The corpus map's feature: the z-scored "evidence for" score (CS-31's
# fit-side selection), the fit side's map cross-fit over 5 folds, and the
# label-blind counterpart beside it.
CMAP_FEATURE = UprCorpusMapFeatureConfig(
    enabled=True,
    scoring="sum_zscore_positive",
    folds=5,
    label_blind=True,
)

cells = [
    AnalyzeUprCell(
        label="upr-analyze-cmap",
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

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_cmap")
