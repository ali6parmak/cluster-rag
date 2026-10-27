"""The zero-shot rerankers on UPR, as **scorers** over a fixed candidate set.

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-15) measures three Hugging Face
rerankers on the UPR data. A reranker can **re-order** a candidate set, never
enlarge it, so the **candidate set** is the experiment's independent variable
and the run is configured with the set it re-orders:

- the **decision model's** top-10 (the primary: the track's best candidate
  generator, the shortlist R@50 0.9987);
- the **union's** top-10 (the per-row z-max of the champion's and the
  decision model's scores, the recall@10 0.9120);
- the **champion's** top-10 (the **control**: the lowest ceiling, the
  recall@10 0.7126 — a reranker cannot beat it by re-ordering).

Every comparison holds the candidate set fixed and changes only the
ordering: the incumbent is the ordering that produced the set.

**The three models** (each pinned by id **and** revision, as the repo's rule
requires):

- `jina-reranker-v2-base-multilingual` (278M): a cross-encoder, one forward
  pass per (query, document) pair;
- `llama-nemotron-rerank-1b-v2` (1B): a cross-encoder with a relevance head,
  the pair rendered into a single sequence by the card's own template;
- `jina-reranker-v3` (0.6B): a **listwise** reranker, the whole candidate
  list in one context.

**The label side** is the catalog's description (the frozen, label-side
knowledge the CS-11 catalog holds), with the label's own name as the cheap
ablation. The **order** of the pair is the text first (the pipeline's
retrieval framing), with the reverse measured in the decision-set run.

**The cost** is part of the deliverable: every arm reports its pairs per
text, its pairs per second, its wall time and its peak GPU memory, against
the decision model's own cost (2 calls per text, ~13 minutes for the 660 val
texts at the measured ~100 calls/min).
"""

from __future__ import annotations

from embedder_pool import MODELS  # noqa: F401  (the shared model pool stays importable)

from cluster_rag.config import (
    DecisionModelConfig,
    RerankerModelConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
    UprRerankConfig,
)
from cluster_rag.sweep import RerankUprCell, define_sweep

CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
CANDIDATE_K = 10

# The pinned decision model (the CS-9 probe's champion model: the paragraph
# shape, the `noul` question type) and its measured input budget. The
# `decision` and `union` candidate sets are its own.
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
TEV1_MAX_INPUT_TOKENS = 2500
INSTRUCTIONS = "Is the text about the topic in the question?"

# The three rerankers, each pinned by id **and** revision.
JINA_V2 = RerankerModelConfig(
    name="jina-v2",
    model_id="jinaai/jina-reranker-v2-base-multilingual",
    revision="9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
    kind="cross_encoder",
    # float32, not the checkpoint's bfloat16: in bfloat16 a score depends on
    # the batch's composition (the padding and the kernels' accumulation
    # order), by up to 0.007 on this model's [0, 1] scale — enough to flip a
    # near-tie, i.e. to let the GPU's rounding decide a measurement. In
    # float32 the score is batch-independent (the measured wobble is 1e-6).
    dtype="float32",
    max_length=512,
    batch_size=32,
    device="cuda",
    trust_remote_code=True,
    # The card's flash attention needs a package this environment does not
    # have; the standard attention is the documented fallback.
    model_kwargs={"use_flash_attn": False},
)
NEMOTRON = RerankerModelConfig(
    name="nemotron-1b",
    model_id="nvidia/llama-nemotron-rerank-1b-v2",
    revision="828765652b05bd439c9789d2a6d093db1caa1443",
    kind="llm_yesno",
    # The card's own prompt format: one sequence per pair.
    template="question:{query} \n \n passage:{document}",
    # float32, for the same reason as the cross-encoder above: in bfloat16
    # this model's score moves by up to 0.3 logits with the batch's
    # composition, and in float32 the move is 7e-5.
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

# The candidate sets: (the set's name, the label documents, the orders).
CANDIDATE_SETS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # The primary: the decision model's own candidates. The label side's and
    # the order's variants are measured here (the cheap ablations).
    ("decision", ("description", "name"), ("text_first", "label_first")),
    # The union's candidates: the description and the primary order only.
    ("union", ("description",), ("text_first",)),
    # The champion's candidates (the control): the description and the
    # primary order only.
    ("champion", ("description",), ("text_first",)),
)


def _decision_section() -> UprDecisionConfig:
    """The decision model's config (the candidate sets' source).

    The calls are cache hits: the whole val grid was scored by CS-9b, and
    the probe asks for the very same cells with the very same arm.
    """
    return UprDecisionConfig(
        models=(
            DecisionModelConfig(
                name="tev1-paragraph-noul",
                model_id="tev1:latest",
                digest=TEV1_DIGEST,
                state="paragraph",
                question_type="noul",
                instructions=INSTRUCTIONS,
                max_input_tokens=TEV1_MAX_INPUT_TOKENS,
            ),
        ),
        scope="grid",
        seed=42,
        competence_control=False,
        prompt_version="v1",
        host="",
        timeout=600.0,
        max_questions_per_call=64,
        bins=10,
        mismatched_control=False,
    )


def _analysis(
    candidate_set: str,
    label_documents: tuple[str, ...],
    orders: tuple[str, ...],
) -> UprAnalysisConfig:
    """The run's analysis section (the rerank stage and the decision source)."""
    return UprAnalysisConfig(
        arms=(),
        length_buckets=LENGTH_BUCKETS,
        reference_eval_run=CHAMPION_EVAL_RUN,
        decision=_decision_section(),
        rerank=UprRerankConfig(
            enabled=True,
            models=MODELS_POOL,
            candidate_set=candidate_set,
            candidate_k=CANDIDATE_K,
            label_documents=label_documents,
            orders=orders,
            label_blind=True,
            benchmark_requests=64,
        ),
    )


def _label(candidate_set: str, model: str) -> str:
    """The run id: the model and the candidate set."""
    return f"upr-rerank-{model}-{candidate_set}"


cells = [
    RerankUprCell(
        label=_label(candidate_set, model.name),
        k=2,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        analysis=_analysis(candidate_set, label_documents, orders),
    )
    for candidate_set, label_documents, orders in CANDIDATE_SETS
    for model in MODELS_POOL
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_rerank")
