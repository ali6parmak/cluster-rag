"""The **distinctive** lexicon (the track's CS-45): the mined words without the register.

CS-35's mined words are the label's top-10 PIFA words, and they carry the
corpus's **register** ("ensure", "right", "system") beside its topic
("corruption", "asylum"). CS-33 measured the classes: a **distinctive** word
sits in one or two of the labelled tables, a **shared** one in 3-14, a
**generic** one in at least a quarter. This change set builds the label-side
representation **from the non-generic words only** — the class is the map's own,
the filter applies to the candidate pool **before** the top-K cut, and it
applies to both routes (an expansion label left without a word falls back to its
name). K stays 10, the expansion prefix 4, the folds 5: nothing else is tuned.

**The cells** (the two halves CS-35 measured, its own comparison targets):

- `upr-labelembed-embeddinggemma-mined-distinctive` — the label artifact: the 68
  label texts from the **filtered** lexicon;
- `upr-eval-fusionc-embeddinggemma-k10-c1-mined-distinctive` — the guard's
  reference for the filtered label side (the champion with those vectors);
- **cell 1 (the primary — the feature half)**
  `upr-analyze-learned-lexicon-distinctive-feat`: the pack's ``label_mined``
  group, the champion's label side, the lexicon filtered;
- **cell 2 (the query half)** `upr-analyze-learned-lexicon-distinctive`: the
  filtered mined label side, groups `embedding + lexical + chunk_profile`, the
  lexicon filtered.

**The comparison.** The name+description channel's numbers are CS-35's own
artifacts (`upr-analyze-learned-lexicon-catfeat` and `-desc`), read by the same
cross-run paired instrument (`tools/compare_upr_arms.py`, the per-text counts,
2,000 resamples, seed 42; the "seen" view is the head, the torso and the tail
pooled). The guard: `upr-analyze-learned-lexicon-catfeat` is re-run and must
reproduce its artifact field for field (the wall clock aside).

**The falsification, registered before the call.** The CS-35 bar unchanged: the
head-to-head (the filtered arm minus the name+description arm) must be positive
with an interval excluding zero on the seen view **and** on at least two of the
three seen buckets. The verdict is cell 1's; cell 2 is read the same way.

**The cost.** Four runs (the analyze ones ~2-3 minutes each), plus the catfeat
re-run; no model calls; no new dependency. The analyze rows land in the
**track's** UPR table under the sweep's section.

    uv run cluster-rag sweep --file \
        experiments_text_processing/upr_learned_lexicon_distinctive_sweep.py
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

# The filtered mined label side's own artifacts (the query half's input and its
# guard's reference).
MINED_LABEL_RUN = "upr-labelembed-embeddinggemma-mined-distinctive"
MINED_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1-mined-distinctive"

ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
MINED_PACK: tuple[str, ...] = (*FULL_PACK, "label_mined")

# CS-33's own classes, minus the register (CS-45's registered rule).
KEPT_CLASSES: tuple[str, ...] = ("distinctive", "shared")

# The lexicon's knobs: CS-35's K, prefix and folds, plus the class filter.
LEXICON = UprLabelLexiconConfig(
    enabled=True,
    top_k=10,
    expansion_min_prefix=4,
    folds=5,
    word_classes=KEPT_CLASSES,
)

# The label-side query texts: the mined words inside the champion's template,
# filtered by the same classes.
LABEL_TEXTS = LabelTextsConfig(
    variant="mined_words",
    template="This recommendation is about {label}.",
    top_k=LEXICON.top_k,
    expansion_min_prefix=LEXICON.expansion_min_prefix,
    word_classes=KEPT_CLASSES,
)


def _learned(*, groups: tuple[str, ...], lexicon: bool) -> UprLearnedConfig:
    """The CS-8f grid's best arm, with the filtered channel switched on."""
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


# The cells: (the run's label, the label-side query, the pack, the lexicon's
# switch, the whole-text reference run).
ARMS: tuple[tuple[str, str, tuple[str, ...], bool, str], ...] = (
    # cell 1, the primary: the feature half (the champion's label side)
    (
        "upr-analyze-learned-lexicon-distinctive-feat",
        "about",
        MINED_PACK,
        True,
        CHAMPION_EVAL_RUN,
    ),
    # cell 2: the query half (the filtered mined label side)
    (
        "upr-analyze-learned-lexicon-distinctive",
        "mined-distinctive",
        FULL_PACK,
        True,
        MINED_EVAL_RUN,
    ),
)

LABEL_RUNS: dict[str, str] = {
    "about": "upr-labelembed-embeddinggemma-about",
    "mined-distinctive": MINED_LABEL_RUN,
}

cells: list[LabelEmbedCell | EvalCell | AnalyzeUprCell] = [
    # the filtered mined label artifact (the input of the query half)
    LabelEmbedCell(
        label=MINED_LABEL_RUN,
        embed=MODELS["embeddinggemma"],
        label_texts=LABEL_TEXTS,
    ),
    # the guard's reference for the filtered label side (the champion with the
    # SAME label vectors)
    EvalCell(
        label=MINED_EVAL_RUN,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=MINED_LABEL_RUN,
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
            reference_eval_run=reference_run,
            learned=_learned(groups=groups, lexicon=lexicon),
        ),
    )
    for label, variant, groups, lexicon, reference_run in ARMS
)

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_lexicon_distinctive")
