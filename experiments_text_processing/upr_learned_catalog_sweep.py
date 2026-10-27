"""The UPR **label catalog**: the description channel, the parent backoff (CS-11).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-11) puts the repo's **label
catalog** (a name / description / parent per label) to work. The catalog is
**LLM-written** from the 68 label names and the public taxonomy only, frozen
and committed before these runs (`catalogs/upr_labels.jsonl` + its meta file).

The three channels the catalog opens, one arm each (plus the combination):

- **the description as its own channel** — the label's description in place of
  its name in the label-side query ("This recommendation is about
  <description>." against the champion's "This recommendation is about
  <name>."). The bare name is the third point of the comparison: the
  champion's own label text is the template, and the CS-8c measured the bare
  name against it (-0.15 of micro-F1 in the champion's geometry);
- **the parent as a sibling-max backoff** — a label with no fit-side support
  inherits the score of a taxonomy sibling, with the **control** beside it
  (the same rule on the fit side's rarest *supported* labels);
- **the catalog-gated feature groups** (`label_part`: the label-description
  coverage; `label_type`: the label-type one-hots), which the catalog unlocks
  for free — the code existed; only the input was missing.

Every arm runs on the **val** split, on the CS-8f grid's best arm (the
gradient-boosted combiner on the full pack, with the reporting blocks on), so
the control is `upr-analyze-learned-report-histgb` and every comparison is
paired (the same texts, the same fit side, the same seeds).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    LabelTextsConfig,
    UprAnalysisConfig,
    UprBackoffConfig,
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
# The guard's reference for the arms whose **label side** is the description
# variant: the guard compares the run's whole-text reference arm against an
# `evaluate` run with the **same** label vectors, so the description label
# side needs its own reference run (the champion with the description labels).
DESC_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1-desc"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CATALOG_PACK: tuple[str, ...] = (*FULL_PACK, "label_part", "label_type")

# The label-side query texts. The champion's is the template; the description
# channel swaps the label's NAME for its DESCRIPTION inside the same template
# (one variable changes).
LABEL_RUNS: dict[str, str] = {
    "about": "upr-labelembed-embeddinggemma-about",
    "desc": "upr-labelembed-embeddinggemma-desc",
}


def _learned(
    *,
    groups: tuple[str, ...] = FULL_PACK,
    backoff: bool = False,
) -> UprLearnedConfig:
    """The CS-8f grid's best arm, with the catalog's channels switched on."""
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
        backoff=UprBackoffConfig(enabled=backoff, lams=(0.5, 1.0), control=True),
        seed=42,
    )


# The arms: (the run's label, the label-side query, the pack, the backoff).
# The **bare name** is not an arm here: the name is the pack's *lexical* block's
# text (it is in every pack), so the single-variable change is the **query**
# (the template against the description), and a name-only query would need a
# new main-pipeline `evaluate` run to guard its reference arm.
ARMS: tuple[tuple[str, str, tuple[str, ...], bool], ...] = (
    # the control, re-run in this sweep: the cross-run paired comparison needs
    # both sides' per-text blocks, and the re-run doubles as a reproducibility
    # check against the CS-8f artifact (the same config, the same numbers)
    ("upr-analyze-learned-cat-control", "about", FULL_PACK, False),
    ("upr-analyze-learned-cat-desc", "desc", FULL_PACK, False),
    ("upr-analyze-learned-cat-feat", "about", CATALOG_PACK, False),
    ("upr-analyze-learned-cat-backoff", "about", FULL_PACK, True),
    ("upr-analyze-learned-cat-desc-backoff", "desc", FULL_PACK, True),
)

cells: list[LabelEmbedCell | EvalCell | AnalyzeUprCell] = [
    # the description query's label artifact (the input of two arms)
    LabelEmbedCell(
        label="upr-labelembed-embeddinggemma-desc",
        embed=MODELS["embeddinggemma"],
        label_texts=LabelTextsConfig(
            variant="description",
            template="This recommendation is about {label}.",
        ),
    ),
    # the guard's reference for the description label side: the champion with
    # the description label vectors (the guard compares the run's whole-text
    # reference arm against an `evaluate` run with the SAME label vectors)
    EvalCell(
        label=DESC_EVAL_RUN,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=LABEL_RUNS["desc"],
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
            reference_eval_run=DESC_EVAL_RUN if variant == "desc" else CHAMPION_EVAL_RUN,
            learned=_learned(groups=groups, backoff=backoff),
        ),
    )
    for label, variant, groups, backoff in ARMS
)

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_catalog")
