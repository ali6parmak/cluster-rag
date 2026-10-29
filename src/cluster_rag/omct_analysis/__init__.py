"""The text-processing track's OMCT analysis driver, as a package.

The public surface is unchanged: every name the importers use is
re-exported here, so ``from cluster_rag.omct_analysis import ...`` keeps
working exactly as before the split. The modules underneath are layered:

    helpers  the small pure helpers (the statistics, the naming, the ids)
      view   the dataset view and the encoding
      arms   the ladder's arms and the comparison against the baseline
    stage3   the learned feature pack and its reporting
  decision   the decision models and the safe-negative rule
    report   the artifact one run writes
"""

from cluster_rag.omct_analysis.arms import (
    ArmScores,
    BaselineArm,
    ChunkedSide,
    LateSide,
    compare_arms,
    measure_arm,
    score_arm,
    score_model_arms,
)
from cluster_rag.omct_analysis.decision import (
    decision_rule_block,
    score_decision_arm,
    score_stage3_model,
    shortlist_rerank_block,
)
from cluster_rag.omct_analysis.helpers import (
    BASELINE_ARM,
    BASELINE_CHUNKER,
    COSINE_FEATURE_NAME,
    DECISION_ARM,
    DECISION_FIXED_CUT,
    DECISION_RULE_FIT_THRESHOLD,
    DECISION_RULE_FIXED_CUT,
    DECISION_RULE_MARGIN,
    DECISION_RULE_VARIANTS,
    DISTRIBUTION_QUANTILES,
    FULL_TEXT_AGGREGATION,
    HIT_KS,
    RULE_VARIANT_ARM,
    RULE_VARIANT_BOTH,
    RULE_VARIANT_COSINE,
    RULE_VARIANTS,
    SEMANTIC_CRITERIA,
    SEMANTIC_CRITERION_AUC,
    SEMANTIC_CRITERION_FLOOR_GAP,
    STAGE3_ARM,
    document_code,
    length_correlation,
    logger,
)
from cluster_rag.omct_analysis.report import (
    build_analysis_report,
)
from cluster_rag.omct_analysis.stage3 import (
    Stage3Context,
    Stage3Grid,
    build_stage3_grid,
    compare_stage3_arm,
)
from cluster_rag.omct_analysis.view import (
    DatasetView,
    ReportingSpec,
    SideEncoding,
    build_dataset_view,
    encode_late_side,
    encode_side,
    fold_question_chunks,
    paragraph_length_report,
    side_cache_versions,
    side_similarities,
)

__all__ = [
    "ArmScores",
    "BASELINE_ARM",
    "BASELINE_CHUNKER",
    "BaselineArm",
    "COSINE_FEATURE_NAME",
    "ChunkedSide",
    "DECISION_ARM",
    "DECISION_FIXED_CUT",
    "DECISION_RULE_FIT_THRESHOLD",
    "DECISION_RULE_FIXED_CUT",
    "DECISION_RULE_MARGIN",
    "DECISION_RULE_VARIANTS",
    "DISTRIBUTION_QUANTILES",
    "DatasetView",
    "FULL_TEXT_AGGREGATION",
    "HIT_KS",
    "LateSide",
    "RULE_VARIANTS",
    "RULE_VARIANT_ARM",
    "RULE_VARIANT_BOTH",
    "RULE_VARIANT_COSINE",
    "ReportingSpec",
    "SEMANTIC_CRITERIA",
    "SEMANTIC_CRITERION_AUC",
    "SEMANTIC_CRITERION_FLOOR_GAP",
    "STAGE3_ARM",
    "SideEncoding",
    "Stage3Context",
    "Stage3Grid",
    "build_analysis_report",
    "build_dataset_view",
    "build_stage3_grid",
    "compare_arms",
    "compare_stage3_arm",
    "decision_rule_block",
    "document_code",
    "encode_late_side",
    "encode_side",
    "fold_question_chunks",
    "length_correlation",
    "logger",
    "measure_arm",
    "paragraph_length_report",
    "score_arm",
    "score_decision_arm",
    "score_model_arms",
    "score_stage3_model",
    "shortlist_rerank_block",
    "side_cache_versions",
    "side_similarities",
]
