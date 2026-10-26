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

import re
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

# The decision models' prompt template's placeholder for the question's text
# (the text-processing track's Stage 3.5; the placeholder is rendered by
# ``cluster_rag.text_processors.decision.questions.render_instructions``).
DECISION_QUESTION_PLACEHOLDER = "{question}"

# The decision models' state shapes: ``pair`` (the paragraph and the question
# in one state) and ``paragraph`` (the paragraph alone as the state, with the
# question in the instructions — the shape that allows 64 questions per call).
DECISION_STATE_PAIR = "pair"
DECISION_STATE_PARAGRAPH = "paragraph"

# The decision models' question types.
DECISION_QUESTION_NOUL = "noul"
DECISION_QUESTION_CHOICE = "choice"

# What a decision probe scores: the reviewed set plus the unjudged-proxy
# control, or the whole same-document grid.
DECISION_SCOPE_JUDGED = "judged"
DECISION_SCOPE_GRID = "grid"


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
    # The text-processing track's own pair of comparison tables
    # (TEXT_PROCESSING_PLAN.md section 2): the track's command family owns
    # them, so track rows never mix with the main pipeline's tables. The
    # command → (dataset, track) mapping is in experiment_table.py.
    experiment_table_text_processing_omct: Path | None = None
    experiment_table_text_processing_upr: Path | None = None
    label_catalog: Path | None = None
    data_external: Path | None = None  # auxiliary pairwise dataset (PLAN.md section 2)


class LabelTextsConfig(BaseModel):
    """What text stands in for each label zero-shot (Phase 4 label-embedding artifact).

    The 68 UPR labels are known from annotations only (no catalog file with
    descriptions, PLAN.md section 7), so the artifact embeds one configurable
    text per label under the model's ``prefixes.label``:

    - ``variant: name`` — the bare label name ("Discrimination");
    - ``variant: template`` — a config-provided template with a ``{label}``
      placeholder (e.g. ``"This recommendation is about {label}."``), rendered
      per label.

    A later ``descriptions`` variant would need the absent label catalog; it is
    deliberately not implemented ahead of that file existing.
    """

    model_config = ConfigDict(extra="forbid")

    variant: Literal["name", "template"] = "name"
    template: str | None = None

    @model_validator(mode="after")
    def _template_matches_variant(self) -> LabelTextsConfig:
        """Template variant needs a template with the placeholder; name forbids one."""
        if self.variant == "name":
            if self.template is not None:
                raise ValueError("label_texts variant 'name' must not set template")
        elif not self.template:
            raise ValueError("label_texts variant 'template' requires template")
        elif "{label}" not in self.template:
            raise ValueError("label_texts template must contain the {label} placeholder")
        return self

    def render_text(self, label: str) -> str:
        """The embeddable text for one label (deterministic, config-driven)."""
        if self.variant == "name":
            return label
        assert self.template is not None  # guaranteed by the validator
        return self.template.replace("{label}", label)


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
    :mod:`cluster_rag.predictors` (``most_frequent``/``random`` in Phase 3,
    ``knn``/``prototype`` in Phase 4); ``params`` are validated by the
    predictor class itself.
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


EMBEDDING_RUN_ID_PREFIX = "upr-embed-"  # AGENTS.md run-id grammar for embed commands


def validate_embedding_run_id(value: str) -> str:
    """Validate a reference to a prior UPR embedding run (Phase 4 opening, item D).

    ``eval.embedding_run_id`` names the ``upr-embed-*`` run whose
    ``results/<run_id>/embeddings.npz`` the evaluate command consumes. The
    ``upr-embed-`` prefix is enforced at config time: an ``omct-`` reference
    would mix the auxiliary OMCT line into a UPR metric, and a non-embed run
    id carries no vectors (the artifact loader gives the run's own message for
    files that exist but are not embed artifacts). Validation here is purely
    syntactic — existence is checked at run time.

    Raises:
        ValueError: If the value is not filesystem-safe or lacks the prefix.
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(
            f"embedding_run_id must be filesystem-safe ([A-Za-z0-9._-]+), got {value!r}"
        )
    if not value.startswith(EMBEDDING_RUN_ID_PREFIX):
        raise ValueError(
            f"embedding_run_id {value!r} must name a UPR embedding run and start with "
            f"{EMBEDDING_RUN_ID_PREFIX!r} (AGENTS.md run-id grammar; e.g. "
            "'upr-embed-e5-base-v2') — an omct- or non-embed run cannot feed an eval"
        )
    return value


LABEL_EMBEDDING_RUN_ID_PREFIX = "upr-labelembed-"  # run-id grammar for labelembed commands


def validate_label_embedding_run_id(value: str) -> str:
    """Validate a reference to a prior zero-shot label-embedding run (Phase 4).

    ``eval.label_embedding_run_id`` names the ``upr-labelembed-*`` run whose
    ``results/<run_id>/label_embeddings.npz`` a zero-shot-capable predictor
    consumes (the ``zeroshot`` predictor, or ``fusion`` with ``c > 0``). Same
    discipline as :func:`validate_embedding_run_id`: syntactic here
    (filesystem-safe + the prefix), existence and consistency (label space,
    model/dimension match with the primary embedding run) at run time.

    Raises:
        ValueError: If the value is not filesystem-safe or lacks the prefix.
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(
            f"label_embedding_run_id must be filesystem-safe ([A-Za-z0-9._-]+), got {value!r}"
        )
    if not value.startswith(LABEL_EMBEDDING_RUN_ID_PREFIX):
        raise ValueError(
            f"label_embedding_run_id {value!r} must name a UPR label-embedding run and "
            f"start with {LABEL_EMBEDDING_RUN_ID_PREFIX!r} (AGENTS.md run-id grammar; "
            "e.g. 'upr-labelembed-jina-v5-about') — dataset records and label texts "
            "live in different artifacts"
        )
    return value


class EvalConfig(BaseModel):
    """Phase 3 evaluation section: predictor, decision rule and metric grids.

    ``split`` names the evaluated fixed split (``val`` default; ``test`` only
    for final reported numbers) — the CLI ``--split`` flag and sweep overrides
    win over this default. Only val/test are allowed: train is an index source,
    never an evaluation split, per the AGENTS.md leakage discipline.

    ``embedding_run_id`` (Phase 4 opening, item D) is None for embedding-free
    predictors and names a prior ``upr-embed-*`` run for embedding-dependent
    ones: the evaluate command resolves, validates and id-aligns that run's
    ``embeddings.npz`` and records its lineage in metrics.json.
    """

    model_config = ConfigDict(extra="forbid")

    predictor: EvalPredictorConfig
    decision: EvalDecisionConfig = Field(default_factory=EvalDecisionConfig)
    metrics: EvalMetricsConfig = Field(default_factory=EvalMetricsConfig)
    split: str = "val"
    embedding_run_id: str | None = None
    # Zero-shot label vectors (Phase 4): a prior `upr-labelembed-*` run, needed
    # by the `zeroshot` predictor and by `fusion` with `c > 0` only.
    label_embedding_run_id: str | None = None

    @field_validator("split")
    @classmethod
    def _split_is_evaluable(cls, value: str) -> str:
        """Restrict to val/test (evaluating on train would only re-measure leakage)."""
        if value not in ("val", "test"):
            raise ValueError("eval.split must be 'val' or 'test' (the fixed evaluation splits)")
        return value

    @field_validator("embedding_run_id")
    @classmethod
    def _embedding_run_id_is_valid(cls, value: str | None) -> str | None:
        """Syntactic check of the embedding-run reference (fail at config time)."""
        return None if value is None else validate_embedding_run_id(value)

    @field_validator("label_embedding_run_id")
    @classmethod
    def _label_embedding_run_id_is_valid(cls, value: str | None) -> str | None:
        """Syntactic check of the label-embedding-run reference."""
        return None if value is None else validate_label_embedding_run_id(value)


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


class ChunkerSpec(BaseModel):
    """A chunker of the text-processing track's registry: a name and its parameters.

    ``name`` is the registry key (e.g. ``"atoms"``); ``params`` are the
    chunker's own parameters (e.g. ``{"split_clauses": false}``). The
    parameters are part of the chunker's cache-facing version, so two
    differently configured chunkers never share cached chunks.

    A bare string is accepted for ``name`` (``chunker: atoms``), so the
    Stage-0 style config keeps loading.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "full_text"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _bare_name_is_accepted(cls, value: Any) -> Any:
        """``"atoms"`` is read as ``{"name": "atoms"}``."""
        if isinstance(value, str):
            return {"name": value}
        return value

    @field_validator("name")
    @classmethod
    def _name_is_known(cls, value: str) -> str:
        """Reject unknown chunker names at config time (fail before any run)."""
        from cluster_rag.text_processors.chunkers import chunker_config_class

        chunker_config_class(value)  # raises for an unknown name
        return value

    @model_validator(mode="after")
    def _params_match_the_chunker(self) -> ChunkerSpec:
        """Check the parameters against the chunker's config class."""
        from cluster_rag.text_processors.chunkers import chunker_config_class

        config_class = chunker_config_class(self.name)
        if config_class is None:
            if self.params:
                raise ValueError(
                    f"The chunker {self.name!r} takes no parameters: {sorted(self.params)}"
                )
            return self
        known = {field.name for field in config_class.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(self.params) - known)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {unknown} for the chunker {self.name!r}; "
                f"expected a subset of {sorted(known)}"
            )
        return self


class PairedBootstrapConfig(BaseModel):
    """The paired-bootstrap settings of a comparison (Stage 2.5's statistic).

    ``resamples`` is the number of bootstrap resamples; ``seed`` seeds the
    generator (``None`` = the run's own seed, so the statistic is reproducible
    from the run's config alone). The effective seed is recorded in the
    artifact.
    """

    model_config = ConfigDict(extra="forbid")

    resamples: int = Field(default=2000, ge=1)
    seed: int | None = None


class AnalysisEmbedderConfig(BaseModel):
    """One embedder the Stage-0 analysis scores with.

    ``name`` is the short handle the run's table row and the artifact use
    (e.g. ``embeddinggemma``); ``embed`` is the full, card-verified embedder
    config (model id, prefixes, revision / digest) — the same object the main
    pipeline's sweeps use, so the analysis and the pool never drift apart.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    embed: EmbedderConfig


class CombinerSpec(BaseModel):
    """A combiner of the text-processing track's registry: a name and its parameters.

    ``name`` is the registry key (``logistic``); ``params`` are the
    combiner's own parameters (e.g. ``{"C": 1.0}``). The parameters are
    validated against the combiner's config class at config time, so a typo
    fails before a run starts.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "logistic"
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_known(cls, value: str) -> str:
        """Reject unknown combiner names at config time (fail before any run)."""
        from cluster_rag.text_processors.combining import combiner_config_class

        combiner_config_class(value)  # raises for an unknown name
        return value

    @model_validator(mode="after")
    def _params_match_the_combiner(self) -> CombinerSpec:
        """Check the parameters against the combiner's config class."""
        from cluster_rag.text_processors.combining import combiner_config_class

        config_class = combiner_config_class(self.name)
        if config_class is None:
            if self.params:
                raise ValueError(
                    f"The combiner {self.name!r} takes no parameters: {sorted(self.params)}"
                )
            return self
        known = {field.name for field in config_class.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(self.params) - known)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {unknown} for the combiner {self.name!r}; "
                f"expected a subset of {sorted(known)}"
            )
        return self


class FeaturePackConfig(BaseModel):
    """The options of the track's feature pack (Stage 3.1, feature pack v0).

    - ``paragraph_chunker``: the chunker whose chunks the chunk-level
      features are measured on. The default is the Stage-2 verdict's
      paragraph-side window (25 words, stride 12); the question side stays
      whole.
    - ``question_chunker``: the chunker of the question side. ``None`` (the
      default) keeps the question whole — the Stage-2 verdict ("chunk the
      paragraph, never the question").
    - ``top_k``: the K of the top-K chunk similarities (the spread and the
      peak margin are measured over the same K best chunk cosines).
    - ``groups``: the **feature groups** to keep (Stage 3.1b's ablation
      unit). The pack's features fall into five groups — ``embedding`` (the
      full-text cosine and the five chunk features), ``lexical`` (the soft
      overlap in both ways and the char n-grams), ``question_part`` (the
      answer options' coverage), ``question_type`` (the five question-type
      one-hots) and ``chunk_profile`` (the Stage-3.3 chunk profile: the
      ranked cross-chunk match profile and the alignment kinds) — and the
      five partition the pack exactly. The default keeps the **four v0
      groups**, i.e. the Stage-3.1 behaviour (the fifth group is opt-in, so
      the earlier runs keep their exact feature sets); a shorter list is the
      ablation's arm. An unknown group name fails at config time, before any
      run.
    - ``profile_top_k``: how many ranks of the chunk profile are kept per
      granularity (the sentence profile and the word-piece profile).
    - ``profile_ngram_size``: the nominal size of the profile's word pieces
      (the pieces tile a sentence, so the last one may be shorter).
    """

    model_config = ConfigDict(extra="forbid")

    paragraph_chunker: ChunkerSpec = Field(
        default_factory=lambda: ChunkerSpec(
            name="word_windows", params={"n_words": 25, "stride": 12}
        )
    )
    question_chunker: ChunkerSpec | None = None
    top_k: int = Field(default=3, ge=1)
    groups: tuple[str, ...] = ("embedding", "lexical", "question_part", "question_type")
    profile_top_k: int = Field(default=3, ge=1)
    profile_ngram_size: int = Field(default=5, ge=2)

    @field_validator("groups")
    @classmethod
    def _groups_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an unknown, an empty or a repeated feature group at config time."""
        from cluster_rag.text_processors.features import FEATURE_GROUPS

        if not value:
            raise ValueError("analysis.stage3.features.groups must name at least one feature group")
        unknown = [group for group in value if group not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown feature group(s) {unknown} in analysis.stage3.features.groups; "
                f"expected values in {list(FEATURE_GROUPS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError(
                f"analysis.stage3.features.groups must not repeat a group: {list(value)}"
            )
        return value


class Stage3ReportingConfig(BaseModel):
    """The settings of the Stage-3 **reporting** (CS-6e).

    The plan's Stage-3 discipline asks for *"calibration curves + thresholds
    per document; pooled and per-document macro; both negative subsets
    separately"*, and CS-6e adds the **safe-negative margin** measured
    against the **control-role floor** (the track's durable deliverable).

    - ``enabled``: whether the run computes the reporting blocks at all. The
      default is ``False``, so every earlier run's artifact keeps its exact
      content (the same precedent as the ``embeddings`` knob below); the
      CS-6e sweep turns it on.
    - ``bins``: the number of the reliability diagram's bins (equal width over
      ``[0, 1]``).
    - ``threshold_criterion``: the **fit-side** criterion of the per-document
      threshold. ``"floor_gap"`` measures the separation between the verified
      positives and the unjudged control cells (the rule's target negative
      class); ``"youden"`` measures it between the verified positives and the
      reviewer-verified hard negatives (the reviewed set's own ground truth).
      Both are Youden separations (``TPR - FPR``) on the fit side; the
      threshold is then applied to the held-out document.
    - ``threshold_quantiles``: how many quantiles of the fit side's scores
      form the threshold's candidate grid.
    - ``margin_quantiles``: how many quantiles of the fit side's margins form
      the margin rule's candidate grid.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    bins: int = Field(default=10, ge=2, le=50)
    threshold_criterion: Literal["floor_gap", "youden"] = "floor_gap"
    threshold_quantiles: int = Field(default=99, ge=3, le=999)
    margin_quantiles: int = Field(default=49, ge=3, le=999)


class Stage3Config(BaseModel):
    """The settings of the track's learned pair scorer (Stage 3.1).

    - ``features``: the feature pack's options (the v0 pack).
    - ``combiner``: the learned scorer (the logistic-regression baseline).
    - ``control_weight``: the training weight of the fit side's **unjudged
      control cells**. They are presumed negatives (their true positive rate
      is unmeasured), so they are down-weighted: they must not dominate the
      fit next to the reviewer-verified labels.
    - ``permutation_resamples``: the number of label permutations of the
      floor check (the shuffled labels must land the AUC at chance).
    - ``embeddings``: where the feature pack's **embedding sides** get their
      vectors. ``"model"`` (the default) is the Stage-3.1 behaviour: the run's
      model, through the disk cache. ``"random"`` is the plan's second sanity
      floor (Stage 3.3's "random-vector baseline"): the embedding sides are
      **random unit vectors** of the model's dimension, drawn from the run's
      seed, so a "win" can be checked against what the *training design* alone
      buys. The pack's lexical, question-part and question-type features never
      used an embedding, so they stay real, and the run's baseline arm (the
      full-text cosine) keeps the model's embeddings. The default is
      ``"model"``, so every earlier run and artifact keeps its meaning.
    - ``reporting``: the settings of the Stage-3 reporting (CS-6e). The
      default is **off**, so the earlier runs' artifacts keep their exact
      content.
    """

    model_config = ConfigDict(extra="forbid")

    features: FeaturePackConfig = Field(default_factory=FeaturePackConfig)
    combiner: CombinerSpec = Field(default_factory=CombinerSpec)
    control_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    permutation_resamples: int = Field(default=5, ge=0)
    embeddings: Literal["model", "random"] = "model"
    reporting: Stage3ReportingConfig = Field(default_factory=Stage3ReportingConfig)


class DecisionModelConfig(BaseModel):
    """One decision model and the prompt it is asked with (Stage 3.5).

    A decision model is Ollama's new model *type*: a **state** (the text to
    judge) plus typed **questions**, answered with probabilities in one forward
    pass. The model only *scores* — it never relabels anything.

    - ``name``: the arm's handle (the run id's model part).
    - ``model_id``: the Ollama tag (e.g. ``"tev1:latest"``).
    - ``digest``: the **pinned** digest of that tag. The client reads the
      server's ``/api/tags`` and the run *fails* when the digest differs, so a
      re-pulled model can never be measured under the old arm's name.
    - ``state``: the state's shape — ``pair`` (the paragraph and the question
      in one state) or ``paragraph`` (the paragraph alone as the state, with
      the question in the instructions; that shape allows up to 64 questions
      per call and is the cheap one).
    - ``question_type``: ``noul`` (a boolean statement, answered with the
      probability that it is true) or ``choice`` (a pick from 2-26 options).
    - ``instructions``: the prompt's **task** instruction (e.g. "is the
      paragraph about the same topic as the question?"). It must **not** carry
      the ``{question}`` placeholder: the state's shape decides where the
      question goes (in the state for ``pair``, in the instructions for
      ``paragraph``), and a task that repeats the question would put it into
      the call twice — which biases the grid's comparison and breaks the
      smaller model's input budget.
    - ``options`` / ``criteria`` / ``positive_option``: the choice
      question's options, their descriptions and the option whose probability
      is the arm's score. They belong to the ``choice`` type only.
    - ``max_input_tokens``: the model's **input budget** in tokens (its hard
      context: measured 512 for laya and ~2,500 for tev1). The planner groups
      the cells into calls that fit the budget and truncates the state's
      paragraph when a single cell does not fit — the question and the
      instructions are never cut, and the truncation is recorded per cell.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    digest: str = Field(min_length=64, max_length=64)
    state: Literal["pair", "paragraph"] = DECISION_STATE_PAIR
    question_type: Literal["noul", "choice"] = DECISION_QUESTION_NOUL
    instructions: str = Field(min_length=1)
    options: tuple[str, ...] = ()
    criteria: dict[str, str] = Field(default_factory=dict)
    positive_option: str | None = None
    max_input_tokens: int | None = Field(default=None, ge=1)

    @field_validator("name")
    @classmethod
    def _name_is_run_id_safe(cls, value: str) -> str:
        """The arm's handle ends up in a run id and a table row: keep it safe."""
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(
                "the decision model's name must be filesystem-safe "
                f"([A-Za-z0-9._-]+), got {value!r}"
            )
        return value

    @field_validator("digest")
    @classmethod
    def _digest_is_hex(cls, value: str) -> str:
        """A digest is a sha256 hex string; anything else is a typo."""
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                "the decision model's digest must be a 64-character sha256 hex "
                f"string, got {value!r}"
            )
        return value

    @field_validator("instructions")
    @classmethod
    def _instructions_are_the_task_alone(cls, value: str) -> str:
        """The task instruction must not repeat the question (the shape's business)."""
        if DECISION_QUESTION_PLACEHOLDER in value:
            raise ValueError(
                f"the decision model's instructions must not carry the "
                f"{DECISION_QUESTION_PLACEHOLDER!r} placeholder: the state's shape "
                "decides where the question goes (in the state for 'pair', in the "
                "instructions for 'paragraph')"
            )
        return value

    @model_validator(mode="after")
    def _question_type_matches_its_options(self) -> DecisionModelConfig:
        """A choice question needs its options, criteria and positive option."""
        if self.question_type == "choice":
            if not 2 <= len(self.options) <= 26:
                raise ValueError(f"a choice question needs 2-26 options, got {len(self.options)}")
            if set(self.criteria) != set(self.options):
                raise ValueError(
                    "a choice question's criteria must describe exactly its options: "
                    f"criteria {sorted(self.criteria)} vs options {sorted(self.options)}"
                )
            if self.positive_option not in self.options:
                raise ValueError(
                    "a choice question needs a positive_option among its options; got "
                    f"{self.positive_option!r} for options {sorted(self.options)}"
                )
        elif self.options or self.criteria or self.positive_option is not None:
            raise ValueError(
                "a 'noul' question takes no options, no criteria and no positive option"
            )
        return self


class DecisionConfig(BaseModel):
    """The decision models' settings (the track's Stage 3.5, CS-7a).

    - ``models``: the arms to probe (each a
      :class:`DecisionModelConfig`); the prompt grid lives here, not in the
      code, so every variant is a config value the artifact records.
    - ``scope``: which cells the probe scores — ``judged`` (the reviewed set
      plus the unjudged-proxy control) or ``grid`` (every cell of the
      same-document grid).
    - ``competence_control``: whether the **synthetic competence control**
      runs beside the arm (constructed positive pairs against unrelated
      negatives). It is not optional: a chance-level result on the reviewed
      set is ambiguous without it.
    - ``competence_pairs``: how many positive/negative pairs the control holds.
    - ``prompt_version``: the prompt's version, recorded in the artifact and
      part of the decision cache's key (a re-prompted arm never reads the old
      arm's answers).
    - ``host`` / ``timeout``: the Ollama server (the first call pays the model
      load, so the timeout is generous).
    - ``max_questions_per_call``: how many questions one call may carry (the
      API accepts at most 64).
    - ``bins``: the reliability diagram's number of bins.
    """

    model_config = ConfigDict(extra="forbid")

    models: tuple[DecisionModelConfig, ...] = ()
    scope: Literal["judged", "grid"] = DECISION_SCOPE_JUDGED
    competence_control: bool = True
    competence_pairs: int = Field(default=100, ge=1)
    # The **label-blind counterpart** (CS-7b): the same model, the same
    # prompts, but every cell's question replaced by another question of the
    # same document. It is the control-role floor's counterpart for a model
    # that is not fitted on anything: it keeps the model's shape, its
    # calibration and its cost, and loses only the pairing.
    mismatched_control: bool = True
    # The shortlist rerank's K grid (CS-7c): the cosine's top-K candidates per
    # paragraph are re-decided by the decision model, and the positives' hit@1
    # is measured against the cosine's and the decision model's own.
    rerank_ks: tuple[int, ...] = (1, 5, 10, 20, 50)
    prompt_version: str = Field(default="v1", min_length=1)
    host: str = "http://localhost:11434"
    timeout: float = Field(default=600.0, gt=0)
    max_questions_per_call: int = Field(default=64, ge=1, le=64)
    bins: int = Field(default=10, ge=2, le=50)

    @field_validator("rerank_ks")
    @classmethod
    def _rerank_ks_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """The shortlist sizes must be positive integers."""
        if not value or any(k < 1 for k in value):
            raise ValueError(f"analysis.decision.rerank_ks must be positive, got {value}")
        return value

    @field_validator("models")
    @classmethod
    def _models_are_present_and_unique(
        cls, value: tuple[DecisionModelConfig, ...]
    ) -> tuple[DecisionModelConfig, ...]:
        """At least one arm, and the arm names are unique (they key the artifact)."""
        if not value:
            raise ValueError("analysis.decision.models must name at least one decision model")
        names = [model.name for model in value]
        if len(set(names)) != len(names):
            raise ValueError(f"the decision models' names must be unique, got {names}")
        return value


class AnalysisConfig(BaseModel):
    """The text-processing track's Stage-0 analysis settings.

    The ``analyze-omct`` command writes the Stage-0 artifact (the OMCT
    pair-score benchmark: per-pair scores, the control set, the per-document
    strata and the negative-enrichment measurement — TEXT_PROCESSING_PLAN.md
    section 3, Stage 0.1).

    - ``embedders``: the models to score with (each with its own card-verified
      embedder config; several models share one run and one artifact).
    - ``control_size``: the number of control cells (subset B) to draw from
      the *unjudged* same-document cells. The default (4,000) is the plan's;
      it is a plain config value, so a different size is a one-line change.
    - ``sampling_rule``: how the control is drawn per document (the Stage-0
      rule is proportional to the unjudged cells; the artifact records the
      rule, the seed and the per-document quotas).
    - ``question_transform`` / ``question_block_markers``: the question side's
      text transform (``strip_block`` is the default: the "Answer
      choices:"/"Notes:" furniture is mostly noise, PLAN.md section 7). The
      markers mirror ``probe.question_block_markers``; the transform version
      hashes them, so a difference can never pass unnoticed.
    - ``chunker``: the chunker of the track's registry used for the
      paragraph side of the **structural arms** (Stage 1.1a) — a name plus its
      parameters. The default (``full_text``) is the Stage-0 baseline: one
      chunk per text, so the embedding cache stays warm.
    - ``question_chunker``: the chunker of the **question** side (Stage 2's
      symmetric variants, exp05/exp10). ``None`` (the default) keeps the
      question whole.
    - ``aggregations``: the aggregation rules the structural arms fold the
      chunk-level scores with (e.g. ``max``, ``top_k_mean:2``). Empty means
      "no structural arm" — the Stage-0 behaviour.
    - ``question_transforms``: extra question-side transforms, each one an
      arm of its own (the free ``keep`` vs ``strip_block`` comparison of
      Stage 1.1a). The baseline transform is always computed as well.
    - ``bootstrap``: the paired-bootstrap settings of the arm-vs-baseline
      comparisons (resamples and seed; the seed defaults to the run's seed).
    - ``semantic_thresholds``: the swept grid of the **semantic** chunker's
      threshold (Stage 2.3). Empty = a fixed threshold (the chunker's own
      parameter); non-empty = the grid is swept and the threshold of every
      document is chosen on the **fit side** (the other two documents), so no
      document ever sees its own fitted value.
    - ``semantic_selection``: the fit-side criterion of that choice —
      ``auc`` (the reviewed set's AUC on the fit side) or ``floor_gap`` (the
      verified positives' mean minus the control's mean, on the fit side).
    - ``header_chunker``: the chunk **header** arm (Stage 2.6): a second
      chunked side whose chunks carry the paragraph's own first sentence (the
      one legal header on this data). It runs beside the plain chunked arms,
      so the header is measured against the no-header arm **in the same run**
      (the pairing is by construction).
    - ``stage3``: the settings of the **learned** pair scorer (Stage 3.1):
      the feature pack's options, the combiner and the control's training
      weight. ``None`` (the default) means the run scores the Stage-0/1/2
      arms only, so the earlier runs are unaffected.
    - ``decision``: the settings of the **decision models** (Stage 3.5): the
      arms (the model, the pinned digest, the state's shape and the prompt),
      the scope and the synthetic competence control. ``None`` (the default)
      means the run does not touch a decision model.
    """

    model_config = ConfigDict(extra="forbid")

    embedders: tuple[AnalysisEmbedderConfig, ...] = ()
    control_size: int = Field(default=4000, ge=1)
    sampling_rule: Literal["proportional"] = "proportional"
    question_transform: Literal["keep", "strip_block"] = "strip_block"
    question_block_markers: tuple[str, ...] = ("Answer choices:", "Notes:")
    chunker: ChunkerSpec = Field(default_factory=ChunkerSpec)
    question_chunker: ChunkerSpec | None = None
    header_chunker: ChunkerSpec | None = None
    aggregations: tuple[str, ...] = ()
    question_transforms: tuple[str, ...] = ()
    semantic_thresholds: tuple[float, ...] = ()
    semantic_selection: Literal["auc", "floor_gap"] = "auc"
    bootstrap: PairedBootstrapConfig = Field(default_factory=PairedBootstrapConfig)
    stage3: Stage3Config | None = None
    decision: DecisionConfig | None = None

    @field_validator("aggregations")
    @classmethod
    def _aggregations_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unknown aggregation rules at config time (fail before any run)."""
        from cluster_rag.text_processors.aggregation import parse_aggregation

        for spec in value:
            parse_aggregation(spec)
        return value

    @field_validator("semantic_thresholds")
    @classmethod
    def _thresholds_are_usable(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject a threshold outside the cosine range or a repeated grid point."""
        for threshold in value:
            if not -1.0 <= threshold <= 1.0:
                raise ValueError(
                    f"analysis.semantic_thresholds must be in [-1, 1], got {threshold}"
                )
        if len(set(value)) != len(value):
            raise ValueError(f"analysis.semantic_thresholds must not repeat a value: {value}")
        return value

    @field_validator("question_transforms")
    @classmethod
    def _question_transforms_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unknown question transforms at config time."""
        known = ("keep", "strip_block")
        unknown = [transform for transform in value if transform not in known]
        if unknown:
            raise ValueError(f"Unknown question transform(s) {unknown}; expected values in {known}")
        return value

    @model_validator(mode="after")
    def _arms_are_consistent(self) -> AnalysisConfig:
        """The arms must be meaningful: no duplicate question transform, and
        aggregations need a chunker that actually splits a text."""
        if len(set(self.question_transforms)) != len(self.question_transforms):
            raise ValueError(
                f"analysis.question_transforms must not repeat a transform: "
                f"{list(self.question_transforms)}"
            )
        if self.question_transform in self.question_transforms:
            raise ValueError(
                f"analysis.question_transforms must not repeat the baseline transform "
                f"({self.question_transform!r}): it is always computed"
            )
        if self.aggregations and self.chunker.name == "full_text":
            raise ValueError(
                "analysis.aggregations need a chunker that splits a text; "
                "'full_text' has one chunk per text (its arms would be the "
                "baseline arm)"
            )
        if self.header_chunker is not None and not self.aggregations:
            raise ValueError(
                "analysis.header_chunker needs aggregation rules: the header's "
                "chunks are folded into one pair score by an arm's rule"
            )
        if self.question_chunker is not None and not self.aggregations:
            raise ValueError(
                "analysis.question_chunker needs aggregation rules: a chunked "
                "question side is only scored through the arms' rules"
            )
        if self.semantic_thresholds and self.chunker.name != "semantic":
            raise ValueError(
                "analysis.semantic_thresholds only applies to the 'semantic' "
                f"chunker, but the chunker is {self.chunker.name!r}"
            )
        return self


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
    # The text-processing track's Stage-0 analysis (the `analyze-omct`
    # command): the models to score with and the control-set design.
    analysis: AnalysisConfig | None = None
    table: TableConfig = Field(default_factory=TableConfig)  # Phase 3.5: sweep bookkeeping
    # Zero-shot label texts (Phase 4): required by the `labelembed` command,
    # which embeds one text per dataset label into the label-embedding artifact.
    label_texts: LabelTextsConfig | None = None

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
