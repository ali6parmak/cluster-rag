"""The **borrowed words** for the five thin labels (the track's CS-44).

Five labels have (almost) no vocabulary of their own: their dictionary tables
lie inside a partner's (CS-38's containment class — *Sexual and reproductive
rights* 0.935 inside *Women's rights*; *Freedom of assembly* 0.909 inside
*Detention*; *Statelessness* 0.864 inside *Rights of the Child*; *CP rights -
general* 0.843 inside *International instruments*; *Trafficking* 0.808 inside
*Rights of the Child*). So the map's column is evidence-poor for them exactly
where it is evidence-rich for the partner: a text about sexual and reproductive
rights is scored high on *Women's rights* and, for *Sexual and reproductive
rights* itself, only on the words the two share.

This run measures the **lending**: the thin label's column also reads the
partner's **evidence-for** words (positive z-score) that its own table does not
carry, each at a **discount** (0.5 × the partner's weight). The words are lent,
not copied, and the partner's own column is untouched. The lending relation is
computed once from the **full train map** and applied to every fold's own
weights on the fit side (the cross-fit: the pairs are a train-side fact, the
weights are the fold's).

**Five arms in one artifact**, so every comparison is paired by construction:

- the **real arm** (the plain column) — CS-32's arm;
- the **control** (the same pack without the group);
- the **label-blind counterpart** (the blind map);
- the **borrow arm** (the lent column on both sides) — the new arm;
- the random-vector floor. The chance and control-role floors ride along.

The run declares **label groups** (the five thin labels, their partners, and the
ten labels individually), so the artifact carries each arm's own per-group
sections and the borrow arm's paired differences on the pair's own cells.

**The falsification, registered before the call.** (i) The thin set's paired Δ
(borrow − plain) is positive with an interval excluding zero, and at least 4 of
the 5 individual signs are positive (the power is low — 63 thin gold cells — so
a positive point with an interval including zero is *unresolved*, not a pass).
(ii) The partners' set does not fall materially: its point estimate and its
interval's lower bound at or above −0.02. (iii) The per-pair swap readout is
descriptive.

**The cost.** One run, ~2-3 minutes (cache hits, no model calls); no new
dependency. The run is the ``analyze-upr`` command (the track's UPR artifact), so
the row lands in the **track's** UPR table
(``docs/experiments_table_text_processing_upr.md``) under the sweep's section.

    uv run cluster-rag sweep --file experiments_text_processing/upr_learned_cmap_borrow_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    UprAnalysisConfig,
    UprBorrowConfig,
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

# The pack's groups: CS-32's configuration, unchanged (the lending changes the
# map's tables, not the pack).
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CMAP_PACK: tuple[str, ...] = (*FULL_PACK, "corpus_map")

# The pack's chunker: the **atoms** (the CS-8b ladder's best chunker on UPR).
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The corpus map's feature: CS-32's column plus -- the change set's question --
# the **lent tables** (CS-44: the five thin labels also read their partner's
# evidence-for words at the discount 0.5, at CS-38's 0.80 containment).
CMAP_FEATURE = UprCorpusMapFeatureConfig(
    enabled=True,
    scoring="sum_zscore_positive",
    folds=5,
    label_blind=True,
    borrow=UprBorrowConfig(enabled=True, discount=0.5, threshold=0.8),
)

cells = [
    AnalyzeUprCell(
        label="upr-analyze-cmap-borrow",
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

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_cmap_borrow")
