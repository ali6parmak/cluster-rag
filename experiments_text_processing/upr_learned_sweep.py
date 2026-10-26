"""The UPR feature pack + the logistic combiner + the three floors (CS-8d).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8d) ports the OMCT track's learned
pair scorer to the UPR data: the **feature pack** (the global cosine, the
chunk cosines, the soft lexical overlap and the char n-grams), the **logistic
combiner** behind the track's ``Combiner`` port, and the **three floors** (the
label-permutation chance floor, the control-role floor and the random-vector
floor) on the same statistic as the headline.

**The protocol** is the main pipeline's: the combiner is fitted on **train**
(the train gold cells plus the train non-gold cells as down-weighted presumed
negatives, weight 0.3 as on OMCT) and evaluated on **val**. The whole-text
reference arm (the standing champion) is computed in the same run, so the
pairing is by construction and the guard holds.

**The pack's groups**: the two catalog-free ones (``embedding`` and
``lexical``). The label-description coverage and the label-type features need
a **label catalog** (a name / description / parent per label); the repo has
none (``paths.label_catalog: null``), so the pack runs without them and the
gap is stated in the artifact and in the track log.

**The chunker** of the pack's chunk-level features: the **atoms** — the CS-8b
ladder's best chunker on UPR (the sentence boundaries beat the word windows,
the opposite of the OMCT verdict).
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

# The pack's chunker: the atoms (the CS-8b ladder's best chunker on UPR).
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The pack's feature groups: the two catalog-free ones.
PACK_GROUPS: tuple[str, ...] = ("embedding", "lexical")

cells = [
    AnalyzeUprCell(
        label="upr-analyze-learned",
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
                groups=PACK_GROUPS,
                top_k=3,
                chunker=ATOMS,
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
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned")
