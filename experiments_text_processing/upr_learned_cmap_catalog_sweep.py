"""The **dictionary + the catalog description** in one pack (the track's CS-43).

The corpus map's column is silent for the eight held-out labels (their tables
are empty), and CS-32/CS-37 measured what that costs: the combiner learns
"floor ⇒ impossible" and the unseen bucket collapses to **0.0000**. The
catalog's description is the one channel that is **non-silent for all 68
labels**: alone it lifts the unseen bucket to **0.4690** (query half,
`upr-analyze-learned-cat-desc`) and **0.4824** (feature half,
`upr-analyze-learned-cat-feat`). This run puts the two channels in **one pack**
for the first time and asks whether the description's presence stops the
dictionary's floor from being read as a veto.

**Two cells** (the second is the registered robustness arm):

- ``upr-analyze-cmap-catalog`` — CS-11's ``CATALOG_PACK`` (the description's
  **feature half**: ``label_part`` + ``label_type``) plus the ``corpus_map``
  group, the champion's label embeddings and the champion's reference eval run;
- ``upr-analyze-cmap-catalog-desc`` — CS-11's ``FULL_PACK`` (the description's
  **query half**, the description in place of the label name) plus the
  ``corpus_map`` group, the description's label embeddings (cached) and the
  description label side's own reference eval run.

Both cells carry CS-32's dictionary settings and the **fit-side mask OFF**
(CS-42's instrument is not combined here: one thing at a time). Each artifact
carries four arms — the real arm (with the column), the **control** (the same
catalog pack without the column: the description's own level), the label-blind
counterpart and the random-vector floor — so every comparison is paired by
construction.

**The guard.** Each cell's control IS CS-11's pack (the same groups, the same
label embeddings, the same knobs), so it must reproduce the matching CS-11 arm
field for field: ``upr-analyze-learned-cat-feat``'s 0.5076 pooled / 0.4824
unseen (cell 1) and ``upr-analyze-learned-cat-desc``'s 0.5084 / 0.4690
(cell 2). A mismatch stops the change set.

**The falsification, registered before the call.** (i) The real arm's unseen
micro-F1 must be at least **0.45** (the control's own level is 0.4824/0.4690,
so the bar allows the column a cost of ~0.03 against the description pack); if
it stays at **0.00**, the dictionary overrules the description and CS-42's
fit-side route is the only one left. (ii) The real arm's **pooled** micro-F1
must stay at or above **0.55**. (iii) The label-blind control still buys
nothing. The verdict is cell 1's; cell 2 is read the same way.

**The cost.** Two runs, ~2-3 minutes each (cache hits; no model calls); no code
change; no new dependency. The run is the ``analyze-upr`` command (the track's
UPR artifact), so the rows land in the **track's** UPR table
(``docs/experiments_table_text_processing_upr.md``) under the sweep's section.

    uv run cluster-rag sweep --file experiments_text_processing/upr_learned_cmap_catalog_sweep.py
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

# The guard's reference: the champion, and the champion with the **description**
# label vectors (the query half changes the whole-text reference arm's label
# side, so it has its own reference run — CS-11's own wiring).
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"
DESC_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1-desc"

# CS-11's two packs, unchanged.
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CATALOG_PACK: tuple[str, ...] = (*FULL_PACK, "label_part", "label_type")

# The label-side query texts (CS-11's own artifacts).
LABEL_RUNS: dict[str, str] = {
    "about": "upr-labelembed-embeddinggemma-about",
    "desc": "upr-labelembed-embeddinggemma-desc",
}

# The pack's chunker: the **atoms** (the CS-8b ladder's best chunker on UPR).
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})


def _cmap_feature() -> UprCorpusMapFeatureConfig:
    """CS-32's dictionary column: the z-scored "evidence for" score, cross-fit."""
    return UprCorpusMapFeatureConfig(
        enabled=True,
        scoring="sum_zscore_positive",
        folds=5,
        label_blind=True,
    )


def _learned(*, groups: tuple[str, ...]) -> UprLearnedConfig:
    """CS-11's pack, plus the dictionary's column (CS-42's mask stays off)."""
    return UprLearnedConfig(
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
            random_vectors=True,
            permutation_resamples=5,
            random_seed=42,
        ),
        reporting=UprLearnedReportingConfig(
            enabled=True,
            bins=10,
            threshold_quantiles=99,
        ),
        corpus_map_feature=_cmap_feature(),
        seed=42,
    )


# The cells: (the run's label, the label-side query, the pack's groups, the
# whole-text reference run). Cell 1 is the primary (the feature half); cell 2 is
# the registered robustness arm (the query half).
ARMS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "upr-analyze-cmap-catalog",
        "about",
        (*CATALOG_PACK, "corpus_map"),
        CHAMPION_EVAL_RUN,
    ),
    (
        "upr-analyze-cmap-catalog-desc",
        "desc",
        (*FULL_PACK, "corpus_map"),
        DESC_EVAL_RUN,
    ),
)

cells = [
    AnalyzeUprCell(
        label=label,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=LABEL_RUNS[variant],
        embed=MODELS["embeddinggemma"],
        analysis=UprAnalysisConfig(
            arms=(),
            reference_eval_run=reference_run,
            learned=_learned(groups=groups),
        ),
    )
    for label, variant, groups, reference_run in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_cmap_catalog")
