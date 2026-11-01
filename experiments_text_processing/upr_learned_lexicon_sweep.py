"""The label lexicon as a label-side channel (the track's CS-35): the arms.

The **corpus map** (CS-23/CS-31) *is* the label→word lexicon. This change set
puts it to work as a **label-side channel**: a label-side representation built
from the corpus-mined words (the PIFA idea) instead of from the name and the
description, plus the **expansion route** for the eight labels that have no
train example (the label's name matched against the corpus, with no labels
involved).

Seven runs, all on UPR **val**, on the CS-8f grid's best arm (the
gradient-boosted combiner on the full pack, the reporting blocks on), so every
comparison is paired (the same texts, the same fit side, the same seeds):

- the **control** (the champion's label side, the plain pack) — CS-11's
  ``cat-control`` re-run, which doubles as the reproducibility check;
- the name+description channel's two halves, re-run for a contemporaneous
  comparison: the **description query** (CS-11's ``cat-desc``) and the
  **catalog-gated groups** (CS-11's ``cat-feat``);
- the mined-words channel's two halves: the **mined query** (the label side is
  the label's corpus-mined words) and the **mined group** (the pack's
  ``label_mined`` column: the mined words' coverage by the text).

**The guard.** The pack is fitted on train and the mined words come from train,
so the fit side's label side is **cross-fit** (each fold's cells are extracted
with the label side built from the other folds' records): otherwise the
channel would be optimistically biased exactly where the combiner learns its
weight. The runs whose label side is not train-derived are not cross-fitted,
so their artifacts must reproduce CS-11's exactly.

    uv run cluster-rag sweep --file experiments_text_processing/upr_learned_lexicon_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    LabelTextsConfig,
    UprAnalysisConfig,
    UprLabelLexiconConfig,
    UprLearnedConfig,
    UprLearnedFloorsConfig,
    UprLearnedReportingConfig,
)
from cluster_rag.sweep import AnalyzeUprCell, EvalCell, LabelEmbedCell, define_sweep

CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"
# The guard's reference for the arms whose **label side** is not the name's: the
# guard compares the run's whole-text reference arm against an `evaluate` run
# with the SAME label vectors, so each label side needs its own reference run
# (the champion with the description vectors, and the champion with the mined
# ones).
DESC_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1-desc"
MINED_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1-mined"
REFERENCE_EVAL_RUNS: dict[str, str] = {
    "desc": DESC_EVAL_RUN,
    "mined": MINED_EVAL_RUN,
}
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CATALOG_PACK: tuple[str, ...] = (*FULL_PACK, "label_part", "label_type")
MINED_PACK: tuple[str, ...] = (*FULL_PACK, "label_mined")

# The label-side query texts. The champion's is the template; the description
# channel swaps the label's NAME for its DESCRIPTION inside the same template
# (CS-11), and the mined-words channel swaps it for the label's corpus-mined
# words inside the same template (CS-35).
LABEL_RUNS: dict[str, str] = {
    "about": "upr-labelembed-embeddinggemma-about",
    "desc": "upr-labelembed-embeddinggemma-desc",
    "mined": "upr-labelembed-embeddinggemma-mined",
}

# The mined-words channel's own knobs (the plan's registered numbers).
LEXICON = UprLabelLexiconConfig(
    enabled=True,
    top_k=10,
    expansion_min_prefix=4,
    folds=5,
)


def _learned(
    *,
    groups: tuple[str, ...] = FULL_PACK,
    lexicon: bool = False,
) -> UprLearnedConfig:
    """The CS-8f grid's best arm, with the channel's knobs switched on."""
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
        label_lexicon=LEXICON if lexicon else UprLabelLexiconConfig(),
        seed=42,
    )


# The arms: (the run's label, the label-side query, the pack, the lexicon).
ARMS: tuple[tuple[str, str, tuple[str, ...], bool], ...] = (
    # the control, re-run in this sweep: the cross-run paired comparison needs
    # both sides' per-text blocks, and the re-run doubles as the
    # reproducibility check against CS-11's artifact
    ("upr-analyze-learned-lexicon-control", "about", FULL_PACK, False),
    # the name+description channel (CS-11), re-run in this sweep
    ("upr-analyze-learned-lexicon-desc", "desc", FULL_PACK, False),
    ("upr-analyze-learned-lexicon-catfeat", "about", CATALOG_PACK, False),
    # the mined-words channel (CS-35): the query half and the feature half
    ("upr-analyze-learned-lexicon-mined", "mined", FULL_PACK, True),
    ("upr-analyze-learned-lexicon-minedfeat", "about", MINED_PACK, True),
)

cells: list[LabelEmbedCell | EvalCell | AnalyzeUprCell] = [
    # the mined-words label artifact (the input of the mined query arm)
    LabelEmbedCell(
        label="upr-labelembed-embeddinggemma-mined",
        embed=MODELS["embeddinggemma"],
        label_texts=LabelTextsConfig(
            variant="mined_words",
            template="This recommendation is about {label}.",
            top_k=LEXICON.top_k,
            expansion_min_prefix=LEXICON.expansion_min_prefix,
        ),
    ),
    # the guard's reference for the mined label side: the champion with the
    # mined label vectors (the guard compares the run's whole-text reference
    # arm against an `evaluate` run with the SAME label vectors)
    EvalCell(
        label=MINED_EVAL_RUN,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=LABEL_RUNS["mined"],
    ),
]
cells.extend(
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
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=REFERENCE_EVAL_RUNS.get(variant, CHAMPION_EVAL_RUN),
            learned=_learned(groups=groups, lexicon=lexicon),
        ),
    )
    for label, variant, groups, lexicon in ARMS
)

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_lexicon")
