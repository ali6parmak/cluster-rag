"""The reranker's score as a **pack feature** on UPR (the track's CS-16).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-16) adds one more column to the
UPR pack's ``(n_texts, n_labels, n_features)`` cube: the reranker's score
for the (text, label) cell.

**The CS-13 precedent is the reason.** The decision probability as a
**stand-alone scorer** was worth +0.0023 of micro-F1 over the champion, and
the *same* probability as a pack **feature** was worth **+0.1147**: a feature
can be *combined* with the pack's other columns, and a stand-alone scorer
cannot. So the reranker's score is measured as a column, in **two shapes**:

- the **full grid** (68 pairs per text, no fill value needed);
- the **candidate-restricted** feature (the decision model's top-10, 10
  pairs per text) with the **pre-registered fill value**: the row's minimum
  scored value (the fill never lifts a non-candidate above a candidate and
  never manufactures a separation the reranker did not produce).

**The hypothesis.** The pack's lexical block finds five of the eight unseen
labels because the **name-vs-text** mechanism is identical for a seen and an
unseen label, so its fitted weight transfers; the decision probability has
the same property and its weight transferred (CS-13). The reranker's score
has it too — it is computed from the label's **description**, identically for
a held-out label — so its fitted weight should transfer as well. The fit
side has **zero** positives for the eight unseen labels, so the transfer is
the whole story.

**The arms** are the two shapes, each with the real feature and with the
**label-blind counterpart** (the same model, the wrong pairing — the yardstick
that says how much of the feature's contribution is the model's own
label-blind signal). The control is the CS-13 arm (the same pack and
combiner **without** the reranker's column), re-run in this sweep so the
cross-run paired comparison has both sides' per-text blocks.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    DecisionModelConfig,
    RerankerModelConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
    UprDecisionFeatureConfig,
    UprLearnedConfig,
    UprLearnedFloorsConfig,
    UprLearnedReportingConfig,
    UprRerankFeatureConfig,
)
from cluster_rag.sweep import AnalyzeUprCell, define_sweep

CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})
BASE_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile", "decision")
RERANK_PACK: tuple[str, ...] = (*BASE_PACK, "rerank")

# The pinned decision model (the CS-9 probe's champion model: the paragraph
# shape, the `noul` question type) and its measured input budget. The
# restricted feature's candidate set is its own top-k.
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
TEV1_MAX_INPUT_TOKENS = 2500
INSTRUCTIONS = "Is the text about the topic in the question?"

# The three rerankers, each pinned by id **and** revision (the same configs
# the CS-15 probe used; float32, because in bfloat16 a score depends on the
# batch's composition).
JINA_V2 = RerankerModelConfig(
    name="jina-v2",
    model_id="jinaai/jina-reranker-v2-base-multilingual",
    revision="9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
    kind="cross_encoder",
    dtype="float32",
    max_length=512,
    batch_size=32,
    device="cuda",
    trust_remote_code=True,
    model_kwargs={"use_flash_attn": False},
)
NEMOTRON = RerankerModelConfig(
    name="nemotron-1b",
    model_id="nvidia/llama-nemotron-rerank-1b-v2",
    revision="828765652b05bd439c9789d2a6d093db1caa1443",
    kind="llm_yesno",
    template="question:{query} \n \n passage:{document}",
    dtype="float32",
    max_length=512,
    batch_size=16,
    device="cuda",
    trust_remote_code=True,
    padding_side="left",
)
JINA_V3 = RerankerModelConfig(
    name="jina-v3",
    model_id="jinaai/jina-reranker-v3",
    revision="d7d7e73b6ea138ced340b83865931b5dfb6c97aa",
    kind="listwise",
    max_documents=64,
    max_query_length=512,
    max_doc_length=512,
    device="cuda",
    trust_remote_code=True,
)
MODELS_POOL: tuple[RerankerModelConfig, ...] = (JINA_V2, NEMOTRON, JINA_V3)


def _decision_model() -> DecisionModelConfig:
    """The decision model whose probability is in the pack and whose top-k is
    the restricted feature's candidate set."""
    return DecisionModelConfig(
        name="tev1-paragraph-noul",
        model_id="tev1:latest",
        digest=TEV1_DIGEST,
        state="paragraph",
        question_type="noul",
        instructions=INSTRUCTIONS,
        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
    )


def _learned(
    groups: tuple[str, ...],
    *,
    rerank: UprRerankFeatureConfig | None = None,
) -> UprLearnedConfig:
    """The CS-13 grid's best arm, with the rerank feature when it is on."""
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
        decision_feature=UprDecisionFeatureConfig(
            enabled=True,
            model=_decision_model(),
            prompt_version="v1",
            max_questions_per_call=64,
            label_blind=False,
        ),
        rerank_feature=rerank or UprRerankFeatureConfig(),
        seed=42,
    )


def _analysis(learned: UprLearnedConfig) -> UprAnalysisConfig:
    """The run's analysis section (the learned arm and the decision source)."""
    return UprAnalysisConfig(
        arms=(),
        length_buckets=LENGTH_BUCKETS,
        reference_eval_run=CHAMPION_EVAL_RUN,
        learned=learned,
        decision=UprDecisionConfig(
            models=(_decision_model(),),
            scope="grid",
            seed=42,
            competence_control=False,
            prompt_version="v1",
            host="",
            timeout=600.0,
            max_questions_per_call=64,
            bins=10,
            mismatched_control=False,
        ),
    )


def _cell(label: str, learned: UprLearnedConfig) -> AnalyzeUprCell:
    """One arm of the sweep."""
    return AnalyzeUprCell(
        label=label,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        embed=MODELS["embeddinggemma"],
        analysis=_analysis(learned),
    )


# The arms: the control (the CS-13 pack, re-run for the pairing) and the two
# shapes per model, each with its label-blind counterpart.
cells = [_cell("upr-analyze-learned-rerank-control", _learned(BASE_PACK))]
for _model in MODELS_POOL:
    for _shape in ("full", "restricted"):
        for _blind in (False, True):
            suffix = "blind" if _blind else "real"
            cells.append(
                _cell(
                    f"upr-analyze-learned-rerank-{_model.name}-{_shape}-{suffix}",
                    _learned(
                        RERANK_PACK,
                        rerank=UprRerankFeatureConfig(
                            enabled=True,
                            model=_model,
                            shape=_shape,
                            candidate_set="decision",
                            candidate_k=10,
                            label_document="description",
                            order="text_first",
                            label_blind=_blind,
                        ),
                    ),
                )
            )

SWEEP = define_sweep(cells, dataset="upr", label="upr_rerank_feature")
