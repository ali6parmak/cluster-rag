"""The **fit-side mask** on the corpus map, in the UPR learned pack (the track's CS-42).

The map's per-cell score has a defect the encoding cannot fix (CS-37): on the
**fit side**, "the dictionary is silent about this label" and "this label is
never positive" are the same fact, so the learned combiner reads the map's
floor as "impossible" — and with the map's column in the pack the eight
held-out labels' bucket collapses (0.4469 -> 0.0000, CS-32).

This run measures the fix: during the pack's **fit pass only**, the tables of a
fixed, **seed-chosen** set of **seen** labels are emptied, so their gold rows
carry the floor exactly like the held-out labels' rows do. The combiner then
sees "positive with the channel at the floor" and cannot use the floor as a
proxy for impossibility. The **evaluation** is unchanged: the val split is
scored with the full train map.

The masked labels are never chosen by hand (a hand-picked list could be chosen
on val): candidates are the **train** map's labels with a non-empty table (60 of
the 68), sorted by name; `np.random.default_rng(seed).choice(..., 8,
replace=False)` with seed 42. The eight held-out labels are ineligible by
construction (they have no table). The chosen names and their train counts are
recorded in the artifact.

**Four arms in one artifact**, so every comparison is paired by construction:

- the **real arm** (the map's column with the floor) — CS-32's arm;
- the **control** (the same pack without the group);
- the **label-blind counterpart** (the map built with the labels permuted
  among the train records);
- the **fit-mask arm** (the map's column emptied for the masked labels on the
  fit side only) — the new arm.

**The guard.** The pack is fitted on train and the map is built from train, so
the fit side's map is **cross-fit** (5 folds over the train split, each fold's
cells scored against a map built from the other folds), and the mask rides into
it: each fold's mask empties the masked labels' tables in **that fold's** map.
The evaluated split's column is the real arm's own, byte-identical.

**The falsification, registered before the call.** (i) The fit-mask arm's unseen
bucket must recover to **at least 0.45** (the control's own number, 0.4469,
rounded up) with the column still in the pack; if the eight still collapse, the
silence proxy is not the binding cause. (ii) The seen buckets must keep their
gain: the paired seen-view difference (fit-mask - control) excludes zero, its
point estimate stays at or above +0.10, and the pooled micro-F1 stays at or
above 0.5502. (iii) The label-blind control still buys nothing.

**The cost.** One run, about 2-3 minutes (cache hits, no model calls); no new
dependency. The run is the ``analyze-upr`` command (the track's UPR artifact),
so the row lands in the **track's** UPR table
(``docs/experiments_table_text_processing_upr.md``) under the sweep's section.

    uv run cluster-rag sweep --file experiments_text_processing/upr_learned_cmap_fitmask_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    UprAnalysisConfig,
    UprCorpusMapFeatureConfig,
    UprFitMaskConfig,
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

# The pack's groups: CS-32's configuration, unchanged (the mask changes the
# fit side's column, not the pack).
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CMAP_PACK: tuple[str, ...] = (*FULL_PACK, "corpus_map")

# The pack's chunker: the **atoms** (the CS-8b ladder's best chunker on UPR).
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The corpus map's feature: the z-scored "evidence for" score (CS-31's
# fit-side selection), the fit side's map cross-fit over 5 folds, the
# label-blind counterpart beside it and -- the change set's question -- the
# **fit-side mask** (CS-42: 8 seed-chosen SEEN labels' tables emptied on the
# fit side only; the evaluation side is unchanged).
CMAP_FEATURE = UprCorpusMapFeatureConfig(
    enabled=True,
    scoring="sum_zscore_positive",
    folds=5,
    label_blind=True,
    fit_mask=UprFitMaskConfig(enabled=True, n_labels=8, seed=42),
)

cells = [
    AnalyzeUprCell(
        label="upr-analyze-cmap-fitmask",
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

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_cmap_fitmask")
