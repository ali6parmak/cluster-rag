"""The **corpus map** (the track's CS-23): the train split's own word↔label
association.

The hit map's four modes (CS-18's topic-side decomposition + the causal
ablation, CS-19's text-side decomposition + the surrogate attribution) all
answer a question about the **model** or about a **feature**. This run answers
the plain statistical question the map never measured: *which words go with
which labels in the corpus itself?*

The map is a description of the **data**, not a measurement of the model:

- the corpus is the **train** split (1,676 texts), so no val text and no val
  label enters the counts and `test` is not touched;
- every (label, word) pair that occurs is stored, with the counts behind it
  (`n_with`, `df`), the lift and the 0.5-smoothed log-odds — **no
  thresholds**: the noise is handled at display time, never by dropping data;
- the eight **held-out** labels have no train example at all, so their tables
  are empty **by construction** — reported as empty, never fabricated.

**The cost: 0 new model calls.** The run is a `hitmap-upr` run, so the intact
texts' grid is scored as always — and those calls are the CS-9b run's cache
hits (the same arm, the same prompt, the same cell layout). The ablation and
the surrogate are switched off: the corpus map needs neither.

**The run** (fixed in the plan's §7.2 before the first call): the run id
`upr-hitmap-corpus`, the artifact `results/upr-hitmap-corpus/hitmap.json` (the
block `corpus_map`), the sweep label `upr_hitmap_corpus` (so the table's
section is `## UPR — upr_hitmap_corpus`) and the log section
`## CS-23 — the corpus map` in `docs/experiments_text_processing.md`.
"""

from __future__ import annotations

from embedder_pool import MODELS  # noqa: F401  (the shared model pool stays importable)

from cluster_rag.config import (
    DecisionModelConfig,
    PosConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
    UprHitMapConfig,
)
from cluster_rag.sweep import HitMapUprCell, define_sweep

CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)

# The UPR-native **task** instruction (the same one the CS-9b grid run used, so
# the intact-text calls are that run's cache hits).
INSTRUCTIONS = "Is the text about the topic in the question?"

# The pinned digest (read from the server on 2026-10-26) and the model's
# measured input budget (the server's hard context).
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
TEV1_MAX_INPUT_TOKENS = 2500


def _model() -> DecisionModelConfig:
    """The decision model the map reads (the champion model, the paragraph shape).

    The map has no model of its own: it reads the CS-9b run's pinned arm, so the
    map, the CS-13 feature and the CS-18 round are read against one model.
    """
    return DecisionModelConfig(
        name="tev1-paragraph-noul",
        model_id="tev1:latest",
        digest=TEV1_DIGEST,
        state="paragraph",
        question_type="noul",
        instructions=INSTRUCTIONS,
        options=(),
        criteria={},
        positive_option=None,
        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
    )


cells = [
    HitMapUprCell(
        label="upr-hitmap-corpus",
        k=2,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        # the part-of-speech tagger (the track's CS-34): the map's own block
        # carries the grammatical readout beside the frequency one.
        pos=PosConfig(enabled=True),
        analysis=UprAnalysisConfig(
            arms=(),
            length_buckets=LENGTH_BUCKETS,
            decision=UprDecisionConfig(
                models=(_model(),),
                scope="grid",
                seed=42,
                competence_control=False,
                competence_pairs=0,
                prompt_version="v1",
                host="",
                timeout=600.0,
                max_questions_per_call=64,
                bins=10,
                mismatched_control=True,
            ),
            hitmap=UprHitMapConfig(
                unit="clause",
                # the corpus map is the only block this run computes: it needs
                # neither the lexical decompositions, nor the surrogate, nor
                # the ablation.
                include_lexical_map=False,
                include_text_side_map=False,
                include_surrogate=False,
                include_ablation=False,
                include_label_blind=True,
                include_corpus_map=True,
            ),
        ),
    )
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_hitmap_corpus")
