"""Configuration loading: YAML files validated with pydantic.

Every CLI command takes ``--config`` pointing at a YAML file (see
``configs/base.yaml``). Values live in the config, never in the code, so the
same code can be reused across experiments (``experiments/*.py`` sweep files
resolve configs in Python; ``configs/experiments/*.yaml`` was retired once
sweeps covered every experiment — the frozen ``results/<run_id>/config.yaml``
is every run's record).

Path resolution: relative paths in a config are resolved against the current
working directory. Run the CLI from the repository root, as in AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cluster_rag.eval.multilabel import DECISION_RULES

# The encode kinds (embed/backends' protocol contract). Owned here because
# kinds are schema-level values (embed_kinds config, PrefixesConfig keys);
# cluster_rag.embed.base re-exports them for the backend API surface.
EncodeKind = Literal["query", "document", "label"]
KINDS: tuple[EncodeKind, ...] = ("query", "document", "label")

# Default formulaic openers for the ``strip_leading_formulaic`` preprocessing
# variant (Phase 2). Derived from the Phase 1 prefix analysis (results/base/
# stats.json): "continue" opens 17.8% of UPR texts, the top-8 verbs ~49%.
# Longest match wins at encode time; override per experiment in YAML.
DEFAULT_STRIP_OPENERS: tuple[str, ...] = (
    "continue to",
    "continue its efforts to",
    "continue its efforts",
    "continue its",
    "continue efforts to",
    "continue efforts",
    "continue",
    "ensure that the",
    "ensure that",
    "ensure the",
    "ensure",
    "take all necessary measures to",
    "take all necessary measures",
    "take measures to",
    "take measures",
    "take steps to",
    "take steps",
    "take",
    "strengthen",
    "adopt",
    "consider",
    "ratify",
    "establish",
    "implement",
    "further",
    "increase",
    "review",
    "develop",
    "intensify",
)


class PathsConfig(BaseModel):
    """Filesystem locations used across the pipeline."""

    model_config = ConfigDict(extra="forbid")

    data_raw: Path
    data_processed: Path
    data_splits: Path
    cache: Path
    results: Path
    # Per-dataset markdown comparison tables, auto-appended by CLI commands
    # (Phase 3.5; command → dataset decides the file); unset in throwaway
    # configs -> the append is skipped.
    experiment_table_upr: Path | None = None
    experiment_table_omct: Path | None = None
    label_catalog: Path | None = None
    data_external: Path | None = None  # auxiliary pairwise dataset (PLAN.md section 2)


class StatsConfig(BaseModel):
    """Parameters of the ``stats`` command (Phase 1 descriptive statistics)."""

    model_config = ConfigDict(extra="forbid")

    frequency_head_min: int = 100  # min examples for a label to count as head
    frequency_torso_min: int = 40  # min examples for torso; below is tail
    prefix_max_words: int = 3
    prefix_lowercase: bool = True
    top_n_labels: int = 15
    top_n_pairs: int = 10
    top_n_prefixes: int = 10


class LabelHoldoutConfig(BaseModel):
    """Which labels are held out of train entirely (zero-shot discovery test)."""

    model_config = ConfigDict(extra="forbid")

    n_labels: int = 8
    min_examples: int = 30  # eligibility: total examples within [min, max]
    max_examples: int = 80


class SplitsConfig(BaseModel):
    """Parameters of the ``split`` command (Phase 1 fixed splits).

    ``ratios`` are (train, val, test) proportions that must sum to 1.
    """

    model_config = ConfigDict(extra="forbid")

    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    coverage_min_total: int = 3
    label_holdout: LabelHoldoutConfig = Field(default_factory=LabelHoldoutConfig)

    @field_validator("ratios")
    @classmethod
    def _ratios_are_valid(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(ratio <= 0 for ratio in value):
            raise ValueError("split ratios (train, val, test) must be positive")
        if abs(sum(value) - 1.0) > 1e-6:
            raise ValueError(f"split ratios {value} must sum to 1")
        return value


class PrefixesConfig(BaseModel):
    """Per-kind embedding prefixes applied before the model sees the text.

    Keys are the encode kinds ("query" = the text being classified, "document" =
    indexed examples, "label" = label names/descriptions). Values are prepended
    verbatim, or used as templates when they contain ``{content}`` (e.g. the
    documented EmbeddingGemma prompts). Empty string = no prefix. The right
    values per model live in configs/ (AGENTS.md: never hard-coded in logic);
    check them against the model card before running (e.g. e5 "query: " /
    "passage: ", nomic "classification:" / "search_document:").
    """

    model_config = ConfigDict(extra="forbid")

    query: str = ""
    document: str = ""
    label: str = ""


class EmbedderConfig(BaseModel):
    """Which embedding backend and model to use, and how to call it.

    ``host``/``timeout`` only apply to the Ollama backend; ``device``/
    ``revision``/``trust_remote_code``/``model_kwargs`` only to the HF backend.
    Irrelevant keys are ignored with the backend that does not use them.
    ``context_tokens`` (both backends) is an explicit model-input-context
    override for run provenance and truncation diagnostics — some servers
    report the loader context (Ollama /api/show for embeddinggemma-2: the 256K
    modality loader) instead of the text window (8K per the model card); when
    set, it replaces the discovered value, which is still recorded in the
    embedder's ``extra`` metadata.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str  # "hf" (sentence-transformers) or "ollama" (/api/embed)
    model_id: str = Field(min_length=1)
    dimension: int | None = Field(default=None, ge=1)  # set = pinned output dimension
    prefixes: PrefixesConfig = Field(default_factory=PrefixesConfig)
    batch_size: int = Field(default=64, ge=1)
    device: str | None = None  # HF only: null = let sentence-transformers pick (cuda if present)
    revision: str | None = None  # HF only: pin the model revision (AGENTS.md)
    trust_remote_code: bool = False  # HF only
    # Both HF-only constructor passthroughs: model_kwargs carries extra
    # SentenceTransformer(...) kwargs (e.g. remote-code modules requiring
    # {"default_task": "retrieval"}). context_tokens is an explicit context
    # override for provenance/diagnostics (None = backend-discovered value).
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    context_tokens: int | None = Field(default=None, ge=1)
    host: str = "http://localhost:11434"  # Ollama only; env OLLAMA_HOST overrides an unset value
    timeout: float = Field(default=120.0, gt=0)  # Ollama HTTP timeout in seconds


class CacheConfig(BaseModel):
    """Disk cache for embeddings (key includes model, prefix, preprocessing)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class PreprocessConfig(BaseModel):
    """Preprocessing applied to texts before embedding (Phase 2).

    Variants: ``raw`` (identity), ``lowercase``, ``strip_leading_formulaic``
    (remove formulaic openers like "Continue to" from UPR recommendations —
    17.8% start with "continue"; see PLAN.md section 6). The opener list is
    config, never code; changing it changes the preprocessing version and thus
    the cache key, so stale embeddings cannot leak through.
    """

    model_config = ConfigDict(extra="forbid")

    variant: str = "raw"  # raw | lowercase | strip_leading_formulaic
    strip_openers: tuple[str, ...] = DEFAULT_STRIP_OPENERS


class EvalPredictorConfig(BaseModel):
    """Which predictor the ``evaluate`` command runs (name + its parameters).

    The name resolves through the registry in
    :mod:`cluster_rag.predictors` (``most_frequent`` / ``random`` in Phase 3);
    ``params`` are validated by the predictor class itself.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class EvalDecisionConfig(BaseModel):
    """The operating point turning scores into F1/coverage decisions.

    Rules (the Phase 4 thresholding options, defined once in Phase 3):
    ``top_k`` predicts the k best-ranked labels per record; ``score_threshold``
    predicts every label with ``score >= threshold``; ``top_k_min_score``
    predicts at most k labels, each additionally passing ``min_score``.
    Parameter completeness is validated here so a bad config fails at load.
    """

    model_config = ConfigDict(extra="forbid")

    rule: str = "top_k"  # one of eval.multilabel.DECISION_RULES
    k: int = Field(default=2, ge=1)  # mean gold labels/record = 1.91 (Phase 1 stats)
    threshold: float | None = None
    min_score: float | None = None

    @field_validator("rule")
    @classmethod
    def _rule_is_known(cls, value: str) -> str:
        """Reject unknown rules at config time (fail before any run)."""
        if value not in DECISION_RULES:
            raise ValueError(f"Unknown decision rule {value!r}; expected one of {DECISION_RULES}")
        return value

    @model_validator(mode="after")
    def _params_complete_for_rule(self) -> EvalDecisionConfig:
        """Each rule needs exactly its own parameters, none of the others."""
        if self.rule == "score_threshold" and self.threshold is None:
            raise ValueError("decision rule 'score_threshold' requires decision.threshold")
        if self.rule == "top_k_min_score" and self.min_score is None:
            raise ValueError("decision rule 'top_k_min_score' requires decision.min_score")
        return self


class EvalMetricsConfig(BaseModel):
    """Grids for the rank-based metrics (k) and shortlist recall (m)."""

    model_config = ConfigDict(extra="forbid")

    ks: tuple[int, ...] = (1, 3, 5, 10)
    shortlist_ms: tuple[int, ...] = (20, 50, 100)

    @field_validator("ks", "shortlist_ms")
    @classmethod
    def _values_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Non-empty grids of positive integers only."""
        if not value or any(k < 1 for k in value):
            raise ValueError(f"metric grids must be non-empty with positive values, got {value}")
        return value


class EvalConfig(BaseModel):
    """Phase 3 evaluation section: predictor, decision rule and metric grids.

    ``split`` names the evaluated fixed split (``val`` default; ``test`` only
    for final reported numbers) — the CLI ``--split`` flag and sweep overrides
    win over this default. Only val/test are allowed: train is an index source,
    never an evaluation split, per the AGENTS.md leakage discipline.
    """

    model_config = ConfigDict(extra="forbid")

    predictor: EvalPredictorConfig
    decision: EvalDecisionConfig = Field(default_factory=EvalDecisionConfig)
    metrics: EvalMetricsConfig = Field(default_factory=EvalMetricsConfig)
    split: str = "val"

    @field_validator("split")
    @classmethod
    def _split_is_evaluable(cls, value: str) -> str:
        """Restrict to val/test (evaluating on train would only re-measure leakage)."""
        if value not in ("val", "test"):
            raise ValueError("eval.split must be 'val' or 'test' (the fixed evaluation splits)")
        return value


class ProbeConfig(BaseModel):
    """Parameters of the Phase 2 auxiliary probe on the pairwise dataset.

    ``question_transforms``: the probe's default question-side transform is
    ``strip_block`` (the "Question:"/"Answer choices:"/Notes:" furniture is
    mostly noise — see PLAN.md section 7 and the Phase 2 findings); ``keep``
    (raw question text) stays available for an explicit comparison by listing
    both in one config, e.g. ``[strip_block, keep]``.
    """

    model_config = ConfigDict(extra="forbid")

    question_transforms: tuple[str, ...] = ("strip_block",)
    question_block_markers: tuple[str, ...] = ("Answer choices:", "Notes:")
    recall_ks: tuple[int, ...] = (1, 10)

    @field_validator("question_transforms")
    @classmethod
    def _transforms_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unknown transforms at config time (fail before any run)."""
        known = ("keep", "strip_block")
        if not value:
            raise ValueError("question_transforms must name at least one transform")
        unknown = [transform for transform in value if transform not in known]
        if unknown:
            raise ValueError(f"Unknown question transform(s) {unknown}; expected values in {known}")
        return value


class TableConfig(BaseModel):
    """Experiment-table bookkeeping carried into runs (Phase 3.5).

    ``sweep_label`` names the sweep that produced the run; the CLI commands
    append their comparison-table row under that sweep's section heading
    (``## <UPR|OMCT> — <sweep_label>``) instead of the standalone
    ``## <UPR|OMCT> — <command>`` heading. Only :func:`~cluster_rag.sweep.define_sweep`
    sets it (sweep cells may not override it); single-run configs leave it unset.
    """

    model_config = ConfigDict(extra="forbid")

    sweep_label: str | None = None


class AppConfig(BaseModel):
    """Root configuration validated from a YAML file.

    Unknown keys are rejected (``extra="forbid"``) so typos in a config file
    fail immediately instead of silently changing behavior. ``extends`` names
    a parent YAML (path relative to this file) merged underneath this one:
    child values win in a recursive dict merge, so experiment configs only
    override what they change.
    """

    model_config = ConfigDict(extra="forbid")

    extends: str | None = None
    run_id: str | None = None
    seed: int = 42
    paths: PathsConfig
    stats: StatsConfig = Field(default_factory=StatsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    embed: EmbedderConfig | None = None
    # Kinds the `embed` command encodes for (one matrix per kind; one per
    # prefix in the cache). Sweeps carry it per cell (EmbedCell.kinds); the
    # CLI ``--kinds`` flag overrides a config value.
    embed_kinds: tuple[str, ...] = ("document",)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    eval: EvalConfig | None = None  # Phase 3: the `evaluate` command requires it
    table: TableConfig = Field(default_factory=TableConfig)  # Phase 3.5: sweep bookkeeping

    @field_validator("embed_kinds")
    @classmethod
    def _kinds_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Restrict the embed command's kinds to the known encode kinds."""
        if not value:
            raise ValueError("embed_kinds must name at least one encode kind")
        unknown = [kind for kind in value if kind not in KINDS]
        if unknown:
            raise ValueError(
                f"Unknown encode kind(s) {unknown} in embed_kinds; expected values in {KINDS}"
            )
        return value


def _parse_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file and require a mapping at the top level."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return raw


def merge_configs(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge the child experiment over its parent config (child wins)."""
    merged: dict[str, Any] = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_chain(path: Path, *, _depth: int = 0) -> dict[str, Any]:
    """Read a config and merge its ``extends`` parent chain underneath it.

    Each file may extend one parent (path relative to the file, or absolute);
    children override parents in a recursive dict merge. The ``extends`` key
    itself is removed from the merged payload.
    """
    if _depth > 8:
        raise ValueError(f"config extends chain is too deep at {path} (possible cycle)")
    raw = _parse_yaml_mapping(path)
    extends = raw.pop("extends", None)
    if extends is not None:
        if not isinstance(extends, str) or not extends.strip():
            raise ValueError(f"extends must be a path string in {path}")
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        raw = merge_configs(_load_raw_chain(parent_path, _depth=_depth + 1), raw)
    return raw


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file into an :class:`AppConfig`.

    An ``extends: <parent.yaml>`` key merges the parent chain underneath this
    config before validation (see :func:`_load_raw_chain`).

    Raises:
        FileNotFoundError: If the config file (or a parent) does not exist.
        ValueError: If a file is not valid YAML or does not match the schema
            (pydantic ``ValidationError`` is a ``ValueError`` subclass).
    """
    return AppConfig.model_validate(_load_raw_chain(Path(path)))
