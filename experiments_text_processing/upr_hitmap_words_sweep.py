"""The hit map's two **cheap modes** (the track's CS-19): the text-side word map
and the surrogate attribution.

CS-18 (the plan archive's section 7.1.4) measured the map's first two versions:
the topic-side decomposition of the word-overlap feature (mode A) and the
decision model's clause-level ablation (mode D, the causal one). A colleague
then asked for the **text side** of the word map — "split the text into words,
check which one contributes more for each label" — which is the pack's *second*
lexical feature (``lex_text_coverage``) decomposed the same exact way.

This sweep adds two modes, both of which cost **no new model call**:

- **mode B — the text-side word map** (`upr-hitmap-text`): the text's own
  words and their share of every cell's ``lex_text_coverage``. The per-document
  view ("which word of *this* recommendation did the matching") and the
  corpus-level view (the long tail of matching words).
- **mode C — the surrogate attribution** (`upr-hitmap-surrogate`): a ridge
  surrogate of the decision model's own probability, fitted on the **train**
  split and read as a per-topic word attribution. It is an *attribution*, never
  a causal claim — so it is validated three ways: its out-of-sample quality, the
  topic-faithfulness check (the label-blind counterpart's attribution follows
  the *shifted* topic), and the **agreement with the causal map** (do the
  surrogate's top words fall inside the clause the CS-18 ablation identified
  as the carrier?).

The decision model is the very same pinned `tev1` the CS-13 feature and the
CS-18 map used, and every call it makes is a **cache hit** (the val grid and
the train-side fit grid are the CS-9b run's).
"""

from __future__ import annotations

from embedder_pool import MODELS  # noqa: F401  (the shared model pool stays importable)

from cluster_rag.config import (
    DecisionModelConfig,
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

# The two modes, as the runs that measure them.
MODES: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "text",
        {
            "include_lexical_map": False,
            "include_text_side_map": True,
            "include_surrogate": False,
            "include_ablation": False,
        },
    ),
    (
        "surrogate",
        {
            "include_lexical_map": False,
            "include_text_side_map": False,
            "include_surrogate": True,
            # the surrogate's cross-check against the causal map needs the
            # carrier clauses, which come from the ablation's units
            "include_ablation": True,
        },
    ),
)


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
        label=f"upr-hitmap-{mode}",
        k=2,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
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
            hitmap=UprHitMapConfig(unit="clause", **mode_config),  # type: ignore[arg-type]
        ),
    )
    for mode, mode_config in MODES
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_hitmap_words")
