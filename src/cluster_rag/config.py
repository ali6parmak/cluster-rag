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
from collections.abc import Sequence
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

    The 68 UPR labels are known from annotations only (PLAN.md section 7), so
    the artifact embeds one configurable text per label under the model's
    ``prefixes.label``:

    - ``variant: name`` — the bare label name ("Discrimination");
    - ``variant: template`` — a config-provided template with a ``{label}``
      placeholder (e.g. ``"This recommendation is about {label}."``), rendered
      per label;
    - ``variant: description`` — the label's **catalog description** (the
      text-processing track's CS-11 catalog, ``paths.label_catalog``), either
      bare or through the same template. The label without a catalog entry
      falls back to its name, and the artifact records the count;
    - ``variant: mined_words`` — the label's **corpus-mined words** (the
      text-processing track's CS-35: the label-side representation built from
      the train split's corpus map, the PIFA idea), either bare or through the
      same template. The words come from the **train** split, so the variant
      needs the split's fixed ids; the command builds the map and the lexicon
      and renders every label's text.

    The ``description`` variant needs the catalog file to exist; without it the
    command fails loudly rather than embedding a half-empty catalog.
    """

    model_config = ConfigDict(extra="forbid")

    variant: Literal["name", "template", "description", "mined_words"] = "name"
    template: str | None = None
    # The mined-words variant's own knobs (CS-35): how many mined words a
    # label keeps, and the expansion route's minimum shared prefix; and the
    # word-class filter (CS-45: the classes to keep, ``None`` = every class).
    top_k: int = Field(default=10, ge=1)
    expansion_min_prefix: int = Field(default=4, ge=1)
    word_classes: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _template_matches_variant(self) -> LabelTextsConfig:
        """The template's presence must match the variant."""
        if self.variant == "name":
            if self.template is not None:
                raise ValueError("label_texts variant 'name' must not set template")
        elif self.variant == "template":
            if not self.template:
                raise ValueError("label_texts variant 'template' requires template")
            elif "{label}" not in self.template:
                raise ValueError("label_texts template must contain the {label} placeholder")
        elif self.template is not None and "{label}" not in self.template:
            raise ValueError("label_texts template must contain the {label} placeholder")
        return self

    def render_text(
        self,
        label: str,
        description: str | None = None,
        mined_words: Sequence[str] | None = None,
    ) -> str:
        """The embeddable text for one label (deterministic, config-driven).

        Args:
            label: The label's name.
            description: The label's catalog description (the
                ``description`` variant only).
            mined_words: The label's **mined words** (the ``mined_words``
                variant only — the corpus-mined representation of CS-35).

        Returns:
            The text to embed.

        Raises:
            ValueError: If the variant needs a description or mined words and
                none is given.
        """
        if self.variant == "name":
            return label
        if self.variant == "description":
            if not description:
                raise ValueError(
                    f"the label_texts variant 'description' needs a description for "
                    f"the label {label!r}: the catalog has none (or the file is absent)"
                )
            if self.template:
                return self.template.replace("{label}", description)
            return description
        if self.variant == "mined_words":
            if not mined_words:
                raise ValueError(
                    f"the label_texts variant 'mined_words' needs the mined words "
                    f"for the label {label!r}: the lexicon has none (or was not built)"
                )
            words = " ".join(mined_words)
            if self.template:
                return self.template.replace("{label}", words)
            return words
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


class PosConfig(BaseModel):
    """The **part-of-speech tagger** (the text-processing track's CS-34).

    A **model client**: the tagger downloads its model (``nltk``'s
    ``averaged_perceptron_tagger_eng``), so it lives at the edge
    (``src/cluster_rag/pos/``) and the track's pure package imports the port
    only. The model's files live under the run's own cache root
    (``cache/pos/nltk_data/``) and the model's **digest** is recorded in the
    run's artifact.

    **Off by default**: every earlier run's artifact keeps its exact content,
    and a run that does not need the grammatical classes never pays for the
    download.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: Literal["nltk"] = "nltk"
    model_id: str = "averaged_perceptron_tagger_eng"
    language: str = "eng"


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


class DictionaryConfig(BaseModel):
    """The dictionary port's settings (CS-48): the question map on the reviewed pairs.

    The map is a **description** of the reviewed set, so it needs no embedder
    and no model: a run that carries only this section may leave
    ``analysis.embedders`` empty. Its one structural knob is the **cross-fit**:
    the folds are over the reviewed **pairs** (each fold's map is built from the
    other folds), and the availability readout reads every pair against a map
    that never saw it.

    - ``include_map``: whether the run builds the map and writes the
      ``dictionary`` block (the CS-48 change set). The switch stays so a later
      dictionary change set can carry its own block without the map.
    - ``folds``: how many folds the cross-fit splits the reviewed pairs into
      (at least 2; the same scheme as the UPR cross-fit).
    - ``seed``: the fold split's seed. ``None`` (the default) uses the run's
      seed, so a re-run of the same config reproduces the same folds.
    - ``scorings``: the arms of the **pair scorer** (the CS-49 change set) —
      CS-30's four scoring functions, in the UPR report's order. Empty (the
      default) means the run writes no arm; the arm needs exactly one
      reference embedder (the run's own full-text cosine).
    - ``blind``: whether the run also scores the **label-blind counterpart**
      (the questions permuted across the reviewed pairs), the arm's
      label-driven check (CS-49's registered condition 2).
    - ``primary``: the **pre-registered primary arm** (CS-49's ``sum_positive``
      by default; CS-50 fixes ``sum_zscore_positive``). It is the arm whose
      comparison with the cosine is the registered test; the other arms are the
      spread.
    - ``class_restrictions``: CS-51's class-filtered arms, each scored with the
      **primary** arm's scoring and compared with the **full** primary arm (the
      increment test): ``nogeneric`` (drop the register) and ``distinctive``
      (keep only the topic's own vocabulary). Empty (the default) runs none.
    - ``pos_restrictions``: CS-52's grammatical arms (``noun``, ``verb``), the
      same increment test on the POS axis; it needs the ``pos:`` section on
      (the tagger supplies the classes).
    - ``units``: the map's **unit** (CS-53): ``words`` (the default, the
      track's content-word tokenizer) or ``phrases`` (the reviewed paragraphs'
      consecutive bigrams over the full token stream, the stopwords kept —
      CS-39's wording). The phrase table is built with the same builder, so
      every readout and every arm works unchanged.
    - ``union``: whether the run also scores the **union arm** (CS-53): the
      word map's z-scored score plus the phrase map's own, both out-of-fold
      behind the same folds (CS-39's composition, ported). It needs
      ``units: phrases`` (the union sums the two halves) and one reference
      embedder.
    - ``stability``: whether the run estimates the map's **stability** (CS-54
      — the bootstrap over the reviewed pairs, every link's `sign_stability`)
      and scores the stability-filtered arms against the full arm. The readout
      rides in the `dictionary` block's `stability` sub-block.
    - ``stability_resamples``: the bootstrap's resamples for the readout (and
      the unjudged cells' filter).
    - ``stability_fold_resamples``: the bootstrap's resamples for **each
      fold's** own filter (the judged cells are scored by the filter estimated
      on their fold's training pairs only).
    - ``stability_thresholds``: the survival thresholds the filtered arms are
      built at (CS-54 registers 0.75; 0.90 is reported beside it).
    - ``question_lexicon``: whether the run also measures the **label side's
      own words** (CS-55): the question text's vocabulary as a channel, the
      register split and the blind counterpart. It needs one reference
      embedder (the cosine comparison) and reads no pair label.
    """

    model_config = ConfigDict(extra="forbid")

    include_map: bool = True
    folds: int = Field(default=5, ge=2)
    seed: int | None = None
    scorings: tuple[str, ...] = ()
    blind: bool = True
    primary: str | None = None
    class_restrictions: tuple[str, ...] = ()
    pos_restrictions: tuple[str, ...] = ()
    units: str = "words"
    union: bool = False
    stability: bool = False
    stability_resamples: int = Field(default=500, ge=1)
    stability_fold_resamples: int = Field(default=100, ge=1)
    stability_thresholds: tuple[float, ...] = (0.75, 0.90)
    question_lexicon: bool = False

    @field_validator("stability_thresholds")
    @classmethod
    def _thresholds_are_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """The survival thresholds are shares in (0, 1], distinct and sorted."""
        if not value:
            raise ValueError("analysis.dictionary.stability_thresholds must not be empty")
        if any(not (0.0 < threshold <= 1.0) for threshold in value):
            raise ValueError(
                "analysis.dictionary.stability_thresholds must be shares in (0, 1], "
                f"got {list(value)}"
            )
        if len(set(value)) != len(value):
            raise ValueError(
                "analysis.dictionary.stability_thresholds must not repeat a "
                f"threshold: {list(value)}"
            )
        return tuple(sorted(value))

    @field_validator("units")
    @classmethod
    def _units_are_known(cls, value: str) -> str:
        """Reject an unknown unit at config time (CS-53)."""
        from cluster_rag.text_processors.features.question_map import (
            UNIT_PHRASES,
            UNIT_WORDS,
        )

        known = (UNIT_WORDS, UNIT_PHRASES)
        if value not in known:
            raise ValueError(f"analysis.dictionary.units must be one of {known}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _union_needs_the_phrase_unit(self) -> DictionaryConfig:
        """The union arm sums the word and the phrase halves (CS-53)."""
        if self.union:
            from cluster_rag.text_processors.features.question_map import UNIT_PHRASES

            if self.units != UNIT_PHRASES:
                raise ValueError(
                    "analysis.dictionary.union needs analysis.dictionary.units: "
                    f"{UNIT_PHRASES!r}: the union is the word map's score plus the "
                    "phrase map's own, so the run's map must be the phrase table"
                )
        if self.question_lexicon and not self.scorings:
            raise ValueError(
                "analysis.dictionary.question_lexicon needs analysis.dictionary.scorings: "
                "the channel is compared with the map's primary arm"
            )
        if self.stability and not self.scorings:
            raise ValueError(
                "analysis.dictionary.stability needs analysis.dictionary.scorings: "
                "the filtered arms are measured against the full primary arm"
            )
        return self

    @field_validator("scorings")
    @classmethod
    def _scorings_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an unknown arm at config time (fail before any run)."""
        if not value:
            return value
        from cluster_rag.text_processors.lexical.question_map_score import SCORING_FUNCTIONS

        unknown = [name for name in value if name not in SCORING_FUNCTIONS]
        if unknown:
            raise ValueError(
                f"analysis.dictionary.scorings has unknown arm(s) {unknown}; "
                f"expected some of {list(SCORING_FUNCTIONS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError(f"analysis.dictionary.scorings must not repeat an arm: {list(value)}")
        return value

    @field_validator("class_restrictions")
    @classmethod
    def _class_restrictions_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an unknown class restriction at config time."""
        if not value:
            return value
        from cluster_rag.text_processors.lexical.question_map_score import (
            CLASS_RESTRICTIONS,
        )

        unknown = [name for name in value if name not in CLASS_RESTRICTIONS]
        if unknown:
            raise ValueError(
                f"analysis.dictionary.class_restrictions has unknown restriction(s) "
                f"{unknown}; expected some of {list(CLASS_RESTRICTIONS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError(
                "analysis.dictionary.class_restrictions must not repeat a "
                f"restriction: {list(value)}"
            )
        return value

    @field_validator("pos_restrictions")
    @classmethod
    def _pos_restrictions_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an unknown POS restriction at config time."""
        if not value:
            return value
        from cluster_rag.text_processors.lexical.question_map_score import (
            POS_RESTRICTIONS,
        )

        unknown = [name for name in value if name not in POS_RESTRICTIONS]
        if unknown:
            raise ValueError(
                f"analysis.dictionary.pos_restrictions has unknown restriction(s) "
                f"{unknown}; expected some of {list(POS_RESTRICTIONS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError(
                f"analysis.dictionary.pos_restrictions must not repeat a restriction: {list(value)}"
            )
        return value

    @field_validator("primary")
    @classmethod
    def _primary_is_known(cls, value: str | None) -> str | None:
        """Reject an unknown primary arm at config time."""
        if value is None:
            return value
        from cluster_rag.text_processors.lexical.question_map_score import SCORING_FUNCTIONS

        if value not in SCORING_FUNCTIONS:
            raise ValueError(
                f"analysis.dictionary.primary is not a known arm: {value!r} "
                f"(expected one of {list(SCORING_FUNCTIONS)})"
            )
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
    - ``late_chunker``: the **late-chunked** arm (Stage 2.7): the document is
      embedded once and the token-level embeddings are pooled per chunk, so
      every chunk carries the document's context. It needs a backend that can
      return token-level embeddings (HF only: Ollama's ``/api/embed`` returns
      one vector per text), so the arm is within-model — the late-chunked
      voyage-4-nano against the early-chunked voyage-4-nano of the same run.
    - ``late_question_chunker``: the late-chunked **question** side (the
      symmetric reading of Stage 2.7). ``None`` (the default) keeps the
      question side whole, as the early-chunked reference does.
    - ``late_pooling``: the pooling rules of the late-chunked arm (each one
      an arm family of its own): ``"mean"`` (the model's own formula,
      ``Normalize(mean(tokens))``) and ``"norm"`` (every token normalized
      before the mean).
    - ``stage3``: the settings of the **learned** pair scorer (Stage 3.1):
      the feature pack's options, the combiner and the control's training
      weight. ``None`` (the default) means the run scores the Stage-0/1/2
      arms only, so the earlier runs are unaffected.
    - ``decision``: the settings of the **decision models** (Stage 3.5): the
      arms (the model, the pinned digest, the state's shape and the prompt),
      the scope and the synthetic competence control. ``None`` (the default)
      means the run does not touch a decision model.
    - ``dictionary``: the settings of the **question map** (CS-48): the map of
      the reviewed pairs' own word↔question association and its cross-fit
      availability. ``None`` (the default) means the run writes no dictionary
      block; a run that carries it may have **no embedders** (the map is a
      description of the data, not a model).
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
    late_chunker: ChunkerSpec | None = None
    late_question_chunker: ChunkerSpec | None = None
    late_pooling: tuple[Literal["mean", "norm"], ...] = ("mean", "norm")
    aggregations: tuple[str, ...] = ()
    question_transforms: tuple[str, ...] = ()
    semantic_thresholds: tuple[float, ...] = ()
    semantic_selection: Literal["auc", "floor_gap"] = "auc"
    bootstrap: PairedBootstrapConfig = Field(default_factory=PairedBootstrapConfig)
    stage3: Stage3Config | None = None
    decision: DecisionConfig | None = None
    dictionary: DictionaryConfig | None = None

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
        if self.late_chunker is not None and not self.aggregations:
            raise ValueError(
                "analysis.late_chunker needs aggregation rules: the late-chunked "
                "chunks are folded into one pair score by an arm's rule"
            )
        if self.late_question_chunker is not None and not self.aggregations:
            raise ValueError(
                "analysis.late_question_chunker needs aggregation rules: a "
                "late-chunked question side is only scored through the arms' rules"
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
        # The dictionary arm (CS-49) is scored against the run's own full-text
        # cosine, so the run needs exactly one reference model.
        if self.dictionary is not None and self.dictionary.scorings and len(self.embedders) != 1:
            raise ValueError(
                "analysis.dictionary.scorings needs exactly one reference embedder "
                f"(the arm is compared with that model's cosine), got {len(self.embedders)}"
            )
        return self


class UprWeightGridConfig(BaseModel):
    """The fusion weights re-fitted on val for the chunked components (CS-8b).

    The Phase-4 convention is kept: ``b = 1`` (the prototype weight is the
    fixed reference) and the two other weights explore the **ratios** around
    it. The grid is applied to the chunk-level components *before* they are
    folded, so each combination is a different folding, not a different
    linear combination of the folded scores.

    The selection happens on **val** (the plan's protocol: val is for
    tuning), so a re-fitted arm's val numbers are optimistically biased — the
    artifact records the grid, the criterion and the number of combinations,
    and the arm's name carries the ``_refit`` suffix.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    a: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    c: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

    @model_validator(mode="after")
    def _weights_are_usable(self) -> UprWeightGridConfig:
        """Reject a negative weight or a grid with a single point."""
        for name, values in (("a", self.a), ("c", self.c)):
            if any(weight < 0 for weight in values):
                raise ValueError(f"upr_analysis.weight_grid.{name} must be >= 0, got {values}")
        if self.enabled and (not self.a or not self.c):
            raise ValueError(
                "upr_analysis.weight_grid is enabled but one of the weight grids is empty"
            )
        return self


class UprArmConfig(BaseModel):
    """One chunked **family** of the UPR analysis (``analyze-upr``, CS-8b).

    A family is a chunker plus the aggregation rules it is folded with; the
    command builds one **arm** per aggregation rule, named
    ``<family>_<chunker slug>_<aggregation slug>``.

    Attributes:
        name: The family's name (the arm names' prefix).
        chunker: The **text** side's chunker.
        aggregations: The folding rules (the track's aggregation registry).
        predictor: The scoring geometry, from the main pipeline's predictor
            registry. ``zeroshot`` is the **pair-scoring** family (the text's
            chunks against the zero-shot label vectors); ``knn`` /
            ``prototype`` / ``fusion`` are the **retrieval** family (each chunk
            is a query against the train-only index).
        params: The predictor's own parameters (e.g. the fusion weights).
        label_chunker: The **label** side's chunker, for the **symmetric**
            family (``None`` = the label stays whole). The UPR labels are 1-3
            words, so the symmetric arms are expected to be nulls — the
            null-check, and the degeneracy is recorded in the artifact.
        label_aggregation: The label side's folding rule (the maximum, as on
            OMCT: a label keeps its best chunk).
        label_embedding_run_id: A **prior zero-shot label run** for this
            family (``None`` = the run's own ``eval.label_embedding_run_id``).
            It is the UPR analogue of OMCT's free question transform: the
            label-side text (the bare name vs the config template) is a
            *config* choice, and this knob lets one run compare the two
            variants side by side.
        late_pooling: The late chunking's pooling rule (``"mean"`` = the
            model's own ``Normalize(mean(tokens))`` formula, ``"norm"`` = the
            per-token-normalized variant). ``None`` = the arm is not
            late-chunked. The late chunking is **HF-only** (Ollama's
            ``/api/embed`` cannot return token-level embeddings), so the
            comparison is within-model, against the early-chunked arm of the
            same run.
        semantic_threshold: The semantic chunker's boundary threshold. The
            threshold is **selected on the fit side** (the train split); the
            selected value and the per-fold numbers go into the artifact.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    chunker: ChunkerSpec
    aggregations: tuple[str, ...] = ("max",)
    predictor: str = Field(default="fusion", min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    label_chunker: ChunkerSpec | None = None
    label_aggregation: str = "max"
    label_embedding_run_id: str | None = None
    late_pooling: Literal["mean", "norm"] | None = None
    semantic_threshold: float | None = None

    @field_validator("aggregations", "label_aggregation")
    @classmethod
    def _aggregation_is_known(cls, value: Any) -> Any:
        """Reject unknown aggregation rules at config time (fail before the run)."""
        from cluster_rag.text_processors.aggregation import parse_aggregation

        if isinstance(value, str):
            parse_aggregation(value)
            return value
        for spec in value:
            parse_aggregation(spec)
        return value

    @model_validator(mode="after")
    def _family_is_consistent(self) -> UprArmConfig:
        """The family must be meaningful: a known predictor, a real chunker and
        a semantic threshold only on the semantic chunker."""
        from cluster_rag.predictors import PREDICTORS

        if self.predictor not in PREDICTORS:
            raise ValueError(
                f"Unknown predictor {self.predictor!r}; expected one of {sorted(PREDICTORS)}"
            )
        if not self.aggregations:
            raise ValueError(f"the UPR arm family {self.name!r} needs an aggregation rule")
        if self.chunker.name == "full_text":
            raise ValueError(
                f"the UPR arm family {self.name!r} uses the 'full_text' chunker: that is "
                "the whole-text reference arm, not a chunked family"
            )
        if self.semantic_threshold is not None and self.chunker.name != "semantic":
            raise ValueError(
                f"the UPR arm family {self.name!r} sets a semantic_threshold, but its "
                f"chunker is {self.chunker.name!r} (only 'semantic' takes one)"
            )
        if self.label_chunker is not None and self.label_chunker.name == "full_text":
            raise ValueError(
                f"the UPR arm family {self.name!r} sets a label_chunker, but the "
                "'full_text' chunker is the whole label (drop the field)"
            )
        return self


class UprLearnedFloorsConfig(BaseModel):
    """The three sanity floors of the UPR learned arm (CS-8d).

    The plan's Stage-3 discipline asks for floors, so that a "win" can be
    checked against chance. On UPR all three are measured on the **same
    statistic as the headline** (the decision view's micro-F1 and the ranking
    view's nDCG@5 / shortlist recall), not on an AUC:

    - ``chance``: the **label-permutation** floor. The fit side's whole label
      vector (the gold cells and the non-gold cells together) is permuted: the
      class counts and the training weights stay exactly as they are and only
      the assignment changes, so the model has no label information at all.
    - ``control_role``: the **control-role** floor. The fit side's gold cells'
      positive labels are relocated into the non-gold space (the non-gold
      cells keep their presumed-negative role as the negative class and the
      original gold cells become negatives). The gap to the chance floor is
      what the training design's own prior buys.
    - ``random_vectors``: the **random-vector** floor. The feature pack's
      embedding sides are replaced with seeded random unit vectors, so the
      arm's lexical features stay real and the embedding geometry is gone.
    """

    model_config = ConfigDict(extra="forbid")

    chance: bool = True
    control_role: bool = True
    random_vectors: bool = True
    permutation_resamples: int = Field(default=5, ge=0)
    random_seed: int = Field(default=42)


class UprDecisionRuleConfig(BaseModel):
    """The UPR safe-negative rule and the shortlist rerank (CS-9b).

    The rule's three forms (the **fixed cut**, the **fit-side threshold** and
    the **margin rule**) and the **label-blind counterpart** beside each; the
    rule's fit side is the **train** split, so no operating point is fitted on
    the evaluated split. The **shortlist rerank** re-orders the champion's
    top-K labels per text with the decision model's own probability.

    Attributes:
        enabled: Whether the run computes the rule and the rerank at all. The
            default is ``False``, so the CS-9 runs' artifacts keep their exact
            content.
        fixed_cut: The fixed cut of the first form (the calibrated model's own
            operating point).
        threshold_quantiles: How many quantiles of the fit side's scores form
            the threshold's candidate grid.
        margin_quantiles: How many quantiles of the fit side's margins form
            the margin rule's candidate grid.
        rerank_ks: The shortlist rerank's K grid.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fixed_cut: float = 0.5
    threshold_quantiles: int = Field(default=99, ge=3, le=999)
    margin_quantiles: int = Field(default=49, ge=3, le=999)
    rerank_ks: tuple[int, ...] = (5, 10, 20, 50)

    @field_validator("rerank_ks")
    @classmethod
    def _rerank_ks_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject a non-positive K."""
        if not value or any(k < 1 for k in value):
            raise ValueError(
                f"upr_analysis.decision.rule.rerank_ks must be positive, got {list(value)}"
            )
        return value


class UprPhraseMapConfig(BaseModel):
    """The **phrase table**'s readout (the track's CS-39).

    The corpus map read over the corpus's **bigrams**: the same relation
    between labels and units, one unit finer. The readout carries the table's
    shape (how many phrases, how many pairs, how fragile they are), the class
    split over it (CS-33's own machinery, applied to phrases) and the **pair
    interaction** (how much more a phrase says than its two words).

    Attributes:
        enabled: Whether the run computes the phrase table at all. The default
            is ``False``, so the earlier runs' artifacts keep their exact
            content.
        n_top: How many interactions the readout lists.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    n_top: int = Field(default=20, ge=1)


class UprLabelPairsConfig(BaseModel):
    """The **label-pair relation**'s readout (the track's CS-38).

    The corpus map read the other way round: a relation between **labels** —
    the vocabulary two labels share and how often they co-occur in the same
    train text. The readout names the lexical near-duplicates, the neighbours
    and the labels with **no vocabulary of their own**, and compares the
    corpus's own geometry with the catalog's (the **sibling** relation: the
    pairs of labels that share a catalog parent).

    Attributes:
        enabled: Whether the run computes the relation at all. The default is
            ``False``, so the earlier runs' artifacts keep their exact content.
        near_duplicate: The vocabulary **Jaccard** at which two labels count
            as lexical near-duplicates.
        containment: The share of a label's own table that must lie inside one
            partner's table for the label to count as having **no vocabulary of
            its own**.
        n_neighbours: How many pairs the readout lists as the **neighbours**
            (the strongest co-occurrence lifts).
        bootstrap_resamples: How many bootstrap resamples the **separation**
            between the catalog's sibling pairs and the rest rests on (the
            resampling unit is the text: the map is rebuilt on every resample).
        bootstrap_seed: The resampling's seed (deterministic).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    near_duplicate: float = Field(default=0.5, ge=0.0, le=1.0)
    containment: float = Field(default=0.8, ge=0.0, le=1.0)
    n_neighbours: int = Field(default=20, ge=1)
    bootstrap_resamples: int = Field(default=2000, ge=0)
    bootstrap_seed: int = 42


class UprHitMapConfig(BaseModel):
    """The **hit map**'s settings (the track's CS-18).

    The map has two versions: the **lexical map** (Version A — the lexical
    feature's own per-word decomposition, a pure computation over the inputs the
    pack already has) and the **ablation map** (Version B — the decision model's
    own sentence- or clause-level ablation, the only version that perturbs the
    input and therefore the only one that says anything causal).

    The decision model is **not** configured here: the map uses the very same
    pinned model as the CS-13 feature (the ``upr_analysis.decision.models``
    entry), so the map and the feature are read against the same model.

    Attributes:
        unit: The ablation unit — ``"sentence"`` (the track's own sentence
            splitter) or ``"clause"`` (the track's atom chunker with the clause
            splitting on). The clause unit is the finer one: the val split's
            texts average ~152 characters and 98.9% of them are a single
            sentence, so the sentence unit can ablate almost nothing.
        include_lexical_map: Whether the run computes the **lexical map**
            (Version A — the topic side of the word-overlap feature's own
            decomposition). It is free, so it is on by default.
        include_text_side_map: Whether the run computes the **text-side map**
            (the mirror decomposition: which words *of the text* matched the
            label's name). It is free, so it is on by default.
        include_surrogate: Whether the run computes the **surrogate
            attribution** (a ridge surrogate of the decision model's own
            probability, fitted on the train split, read as a per-topic word
            attribution). It needs the model's cached probabilities and no new
            model call, so it is on by default.
        include_ablation: Whether the run computes the **ablation map** (the
            decision model's own sentence- or clause-level ablation). It is
            needed by the surrogate's cross-check against the causal map, and
            it is off for a run that only wants the cheap modes.
        include_label_blind: Whether the **label-blind counterpart** runs
            beside the ablation (the yardstick that says whether the
            attribution is label-driven). Off by default would make the map
            say nothing, so it is on by default.
        include_corpus_map: Whether the run computes the **corpus map**
            (CS-23): the **train** split's own word↔label association — every
            word for every label, with the counts (``n_with``, ``df``), the
            lift and the 0.5-smoothed log-odds. It is a description of the
            data, not a measurement of the model, so it needs no model call of
            its own; it is **on** by default (the reviewer's decision of
            2026-11-01), so every ``hitmap-upr`` run carries it and a re-run
            of the earlier sweeps grows the artifact by a few MB.
        stability_resamples: How many **bootstrap resamples** the map's own
            **stability** (CS-36) is estimated on: the train split's texts are
            resampled with replacement, the map is rebuilt on each resample,
            and every (label, word) pair's survival is read off. **0** (the
            default) measures nothing — the stability block then says so.
        stability_head_k: How many of a label's top words by log-odds count as
            its **head words** (the second stability statistic: does the pair
            keep its place among them?).
        phrase_map: The **phrase table**'s readout (the track's CS-39): the
            corpus map read over the corpus's bigrams, with the class split and
            the pair interaction. Off by default.
        label_pairs: The **label-pair relation**'s readout (the track's CS-38):
            the corpus's own geometry over the label space (the vocabulary
            overlap and the co-occurrence of every pair of labels), read
            against the catalog's own relation. Off by default.
        catch_all_labels: The generic catch-alls the map reports beside the
            held-out labels.
        surrogate_lambda: The ridge penalty of the surrogate (the larger, the
            more its coefficients are shrunk towards zero).
        surrogate_min_df: The smallest number of **train** texts a word must
            appear in to become a surrogate feature.
        surrogate_top_words: How many words each topic's attribution carries on
            each side (the positive ones and the negative ones).
        bootstrap: The paired bootstrap's settings (the resampling unit is the
            text).
    """

    model_config = ConfigDict(extra="forbid")

    unit: Literal["sentence", "clause"] = "clause"
    include_lexical_map: bool = True
    include_text_side_map: bool = True
    include_surrogate: bool = True
    include_ablation: bool = True
    include_label_blind: bool = True
    include_corpus_map: bool = True
    stability_resamples: int = Field(default=0, ge=0)
    stability_head_k: int = Field(default=10, ge=1)
    phrase_map: UprPhraseMapConfig = Field(default_factory=UprPhraseMapConfig)
    label_pairs: UprLabelPairsConfig = Field(default_factory=UprLabelPairsConfig)
    catch_all_labels: tuple[str, ...] = ("General", "Other")
    surrogate_lambda: float = Field(default=1.0, ge=0.0)
    surrogate_min_df: int = Field(default=3, ge=1)
    surrogate_top_words: int = Field(default=15, ge=1)
    bootstrap: PairedBootstrapConfig = Field(default_factory=PairedBootstrapConfig)


class UprListwiseConfig(BaseModel):
    """The **listwise** second decision stage (the track's CS-12).

    A second round on top of the pointwise arm: **one** call per text asking
    "which of these N topics is the text about?", the options being the
    round-1 top-N labels. The hypothesis: the decision model is good at
    **finding** the candidates and weak at **ordering** them, and the ordering
    is where the pipeline's ``top_k k=2`` lives.

    Attributes:
        enabled: Whether the run computes the listwise stage at all. The
            default is ``False``, so the earlier runs' artifacts keep their
            exact content.
        candidates: How many round-1 candidates the second round offers.
        criteria: The wording of each option: ``"catalog"`` (the CS-11
            catalog's description) or ``"name"`` (the label's own name — the
            control that isolates the *listwise mechanism* from the
            *descriptions*).
        question_text: The question's wording.
        instructions: The task instruction (the question's placement is the
            shape's business, so the task must not repeat the question).
        prompt_version: The listwise prompt's version (part of the decision
            cache's key, so the second round's calls never collide with the
            pointwise ones).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    candidates: int = Field(default=20, ge=2, le=26)
    criteria: Literal["catalog", "name"] = "catalog"
    question_text: str = "Which of these topics is the text about?"
    instructions: str = "Pick the topic that the text is about."
    prompt_version: str = "v2"

    @field_validator("question_text", "instructions")
    @classmethod
    def _wording_is_present(cls, value: str) -> str:
        """Reject an empty wording (the call would carry no question)."""
        if not value.strip():
            raise ValueError("the listwise question's wording must not be empty")
        return value


class UprDecisionConfig(BaseModel):
    """The UPR decision probe's settings (the track's CS-9).

    The decision models are **measurement instruments** on the UPR side: they
    answer "is this text about *<label>*?" per label, and the probe measures
    what their probability buys. They never enter the pipeline.

    Attributes:
        models: The models to probe (each with its pinned id **and** digest,
            its state shape and its question type — the same
            :class:`DecisionModelConfig` the OMCT probe uses).
        scope: ``"sample"`` (a seeded sample of the evaluated split's texts)
            or ``"grid"`` (every text of the split). The grid scope is
            expensive: the ``paragraph`` shape is ~2 calls per text and the
            ``pair`` shape one call per (text, label) cell, so the full val
            grid is ~1,300 calls per model for the former and ~44,900 for the
            latter.
        sample_size: How many texts the sample scope takes.
        seed: The seed of the sample (the same texts for every arm, so the
            arms are paired).
        competence_control: Whether the synthetic competence control runs
            beside every arm.
        competence_pairs: How many positive/negative pairs the competence
            control builds.
        prompt_version: The prompt's version (part of the decision cache's
            key).
        host: The Ollama host (the empty value uses ``OLLAMA_HOST`` or the
            default).
        timeout: The call timeout in seconds.
        max_questions_per_call: How many questions one call may carry (the
            API's limit is 64).
        bins: The number of the reliability diagram's bins.
        mismatched_control: Whether the **label-blind counterpart** runs beside
            every arm (the yardstick that turns the cost into a quality
            claim).
        rule: The **safe-negative rule** and the **shortlist rerank**
            (CS-9b). Off by default, so the CS-9 runs' artifacts keep their
            exact content.
        listwise: The **listwise** second stage (CS-12). Off by default, so
            the earlier runs' artifacts keep their exact content.
        union: Whether the run measures the **union** candidate generator
            (CS-14). It is a measurement, not a component: the union is not
            built and enters no pipeline. Off by default.
    """

    model_config = ConfigDict(extra="forbid")

    models: tuple[DecisionModelConfig, ...] = ()
    scope: Literal["sample", "grid"] = "sample"
    sample_size: int = Field(default=150, ge=1)
    seed: int = 42
    competence_control: bool = True
    competence_pairs: int = Field(default=50, ge=0)
    prompt_version: str = "v1"
    host: str = ""
    timeout: float = 600.0
    max_questions_per_call: int = Field(default=64, ge=1, le=64)
    bins: int = Field(default=10, ge=2, le=50)
    mismatched_control: bool = True
    rule: UprDecisionRuleConfig = Field(default_factory=UprDecisionRuleConfig)
    listwise: UprListwiseConfig = Field(default_factory=UprListwiseConfig)
    union: bool = False


class RerankerModelConfig(BaseModel):
    """One reranker and how it is called (the track's CS-15, PLAN's Phase 6).

    A reranker re-orders an existing candidate list. The three families
    differ in what one forward pass covers, so the family (``kind``) is part
    of the model, not of the caller:

    - ``cross_encoder``: one forward pass per **(query, document) pair**, the
      score is the sigmoid of the classification logit (the jina-reranker-v2
      card's own score);
    - ``llm_yesno``: one forward pass per pair as well, but the pair is
      rendered into a **single sequence** by the ``template`` (the
      llama-nemotron card's ``"question:... passage:..."`` format), and the
      score is the raw relevance logit;
    - ``listwise``: one forward pass per **(query, up to ``max_documents``)**
      — the whole candidate list in one context (the jina-reranker-v3 card's
      ``rerank``), and the score is the card's relevance score.

    Attributes:
        name: The arm's handle (the run id's model part).
        model_id: The Hugging Face model id.
        revision: The **pinned** commit hash of the model (the repo's rule:
            the model and its revision are pinned in the config).
        kind: The reranker family (see above).
        max_length: The input budget in tokens for the pair-based families.
        batch_size: How many pairs one forward pass carries.
        dtype: The torch dtype of the loaded weights (``auto`` keeps the
            checkpoint's own).
        device: The device the model runs on.
        trust_remote_code: Whether the model's custom modelling code is
            allowed (all three rerankers ship custom code).
        template: The pair template of the ``llm_yesno`` family (the
            ``{query}`` / ``{document}`` placeholders).
        padding_side: The tokenizer's padding side (the llama-nemotron card
            asks for ``left``; the cross-encoders take the default).
        max_query_length: The listwise family's query budget in tokens.
        max_doc_length: The listwise family's per-document budget in tokens.
        max_documents: The listwise family's block size (how many documents
            one forward pass may read; the card's own limit is 64).
        model_kwargs: Extra keyword arguments for the model loader (e.g.
            ``{"use_flash_attn": false}`` for the jina-reranker-v2, whose
            flash attention needs a package this environment does not have).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    kind: Literal["cross_encoder", "llm_yesno", "listwise"] = "cross_encoder"
    max_length: int = Field(default=512, ge=1)
    batch_size: int = Field(default=32, ge=1)
    dtype: str = "auto"
    device: str = "cuda"
    trust_remote_code: bool = True
    padding_side: Literal["left", "right"] = "right"
    template: str | None = None
    max_query_length: int | None = Field(default=None, ge=1)
    max_doc_length: int | None = Field(default=None, ge=1)
    max_documents: int = Field(default=64, ge=1)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_is_run_id_safe(cls, value: str) -> str:
        """The arm's handle ends up in a run id and a table row: keep it safe."""
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(
                f"the reranker's name must be filesystem-safe ([A-Za-z0-9._-]+), got {value!r}"
            )
        return value

    @field_validator("revision")
    @classmethod
    def _revision_is_a_commit(cls, value: str) -> str:
        """The revision is a commit hash; anything shorter is a typo."""
        if not re.fullmatch(r"[0-9a-f]{7,40}", value):
            raise ValueError(
                "the reranker's revision must be a commit hash (7-40 hex characters), "
                f"got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _template_matches_the_family(self) -> RerankerModelConfig:
        """The pair template belongs to the ``llm_yesno`` family alone."""
        if self.kind == "llm_yesno" and not self.template:
            raise ValueError(
                "an 'llm_yesno' reranker needs the pair template (the model's own "
                'prompt format, e.g. "question:{query} \\n \\n passage:{document}")'
            )
        if self.kind != "llm_yesno" and self.template:
            raise ValueError(
                f"a {self.kind!r} reranker takes no pair template (its pair is "
                "tokenized as a pair, not rendered into one sequence)"
            )
        return self


class UprRerankFeatureConfig(BaseModel):
    """The reranker's score as a **pack feature** (the track's CS-16).

    The CS-13 precedent is the reason this exists: the decision probability
    as a stand-alone scorer was worth +0.0023 of micro-F1 over the champion,
    and the *same* probability as a pack **feature** was worth +0.1147. So the
    reranker's score is measured as one more column of the pack's cube, in
    **two shapes**:

    - ``full``: the score of every (text, label) cell (68 pairs per text, no
      fill value needed);
    - ``restricted``: the score of the candidate set's cells only (10 pairs
      per text) with the **pre-registered fill value** — the row's minimum
      scored value — for the cells outside the set.

    The feature is **cell-local** by construction: the pack never calls the
    model (the cells are scored at the edge and the matrix is passed in), and
    the label side is the frozen catalog's description, so the fitted weight
    transfers to the labels the fit side never saw.

    Attributes:
        enabled: Whether the run computes the feature at all. The default is
            ``False``, so the earlier runs' artifacts keep their exact
            content.
        model: The reranker (pinned by id **and** revision).
        shape: ``"full"`` or ``"restricted"``.
        candidate_set: The candidate set of the ``restricted`` shape (the
            decision model's own top-k).
        candidate_k: How many labels per text the candidate set holds.
        label_document: The label side's variant (``"description"`` or
            ``"name"``).
        order: The pair order variant (``"text_first"`` or
            ``"label_first"``).
        label_blind: Whether the feature is the **label-blind counterpart's**
            score (the wrong pairing, the same model and the same cost) — the
            yardstick that says how much of the feature's contribution is the
            model's own label-blind signal.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: RerankerModelConfig | None = None
    shape: Literal["full", "restricted"] = "full"
    candidate_set: Literal["decision", "union", "champion"] = "decision"
    candidate_k: int = Field(default=10, ge=1)
    label_document: Literal["description", "name"] = "description"
    order: Literal["text_first", "label_first"] = "text_first"
    label_blind: bool = False

    @model_validator(mode="after")
    def _model_is_required(self) -> UprRerankFeatureConfig:
        """An enabled feature needs a pinned model."""
        if self.enabled and self.model is None:
            raise ValueError(
                "upr_analysis.learned.rerank_feature is enabled but carries no "
                "model: the feature needs a pinned reranker (id and revision)"
            )
        return self


class UprRerankConfig(BaseModel):
    """The rerankers on UPR, as **scorers** over a fixed candidate set (CS-15).

    A reranker can re-order a set, never enlarge it, so the **candidate set**
    is the experiment's independent variable: the run is configured with the
    set it re-orders (the decision model's top-10, the union's top-10, or the
    champion's top-10 as the control), and the run measures the reordering
    against **the ordering that produced the set**.

    Attributes:
        enabled: Whether the run measures the rerankers at all. The default is
            ``False``, so the earlier runs' artifacts keep their exact content.
        models: The rerankers to measure (each pinned by id **and**
            revision).
        candidate_set: Which set is re-ordered: ``"decision"`` (the decision
            model's own top-k), ``"union"`` (the per-row z-max of the
            champion's and the decision model's scores) or ``"champion"`` (the
            geometry's own top-k — the control, whose ceiling is the lowest).
        candidate_k: How many labels per text the candidate set holds.
        label_documents: The label side's variants: ``"description"`` (the
            catalog's description) and ``"name"`` (the label's own name).
        orders: The pair order variants: ``"text_first"`` (the pipeline's
            retrieval framing) and ``"label_first"`` (the cross-encoders'
            training orientation). The listwise family only admits the first.
        label_blind: Whether the **label-blind counterpart** runs beside every
            arm (the yardstick that says whether a gain is label-driven).
        benchmark_requests: How many requests the **throughput probe** times
            (the probe bypasses the cache, so the cost claim does not depend
            on the cache's state).
        bootstrap: The paired-bootstrap settings of the arm comparisons.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    models: tuple[RerankerModelConfig, ...] = ()
    candidate_set: Literal["decision", "union", "champion"] = "decision"
    candidate_k: int = Field(default=10, ge=1)
    label_documents: tuple[Literal["description", "name"], ...] = ("description",)
    orders: tuple[Literal["text_first", "label_first"], ...] = ("text_first",)
    label_blind: bool = True
    benchmark_requests: int = Field(default=64, ge=0)
    bootstrap: PairedBootstrapConfig = Field(default_factory=PairedBootstrapConfig)

    @field_validator("models")
    @classmethod
    def _model_names_are_unique(
        cls, value: tuple[RerankerModelConfig, ...]
    ) -> tuple[RerankerModelConfig, ...]:
        """Two arms of one run must not share a name (the arm names' prefix)."""
        names = [model.name for model in value]
        if len(set(names)) != len(names):
            raise ValueError(f"the rerankers must have unique names: {names}")
        return value

    @model_validator(mode="after")
    def _enabled_needs_models(self) -> UprRerankConfig:
        """An enabled rerank stage needs at least one pinned model."""
        if self.enabled and not self.models:
            raise ValueError(
                "upr_analysis.rerank is enabled but carries no model: the rerank "
                "stage needs at least one pinned reranker (id and revision)"
            )
        return self


class UprLearnedReportingConfig(BaseModel):
    """The UPR learned arm's reporting (CS-8f).

    The two blocks the plan asks for:

    - the **calibration**: the reliability diagram of the arm's probability
      (the expected calibration error, the Brier score and the per-bin
      records), pooled and per **label bucket** and per **text-length
      bucket**;
    - the **decision-rule comparison**: the three decision rules (``top_k``,
      ``score_threshold`` and ``top_k_min_score``) on the same arm. The
      threshold rules' cut is chosen on the **fit side** (the train split) by
      the Youden separation, so the evaluated split never picks it.

    Attributes:
        enabled: Whether the run computes the reporting blocks at all. The
            default is ``False``, so every earlier run's artifact keeps its
            exact content.
        bins: The number of the reliability diagram's bins (equal width over
            ``[0, 1]``).
        threshold_quantiles: How many quantiles of the fit side's scores form
            the threshold's candidate grid.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    bins: int = Field(default=10, ge=2, le=50)
    threshold_quantiles: int = Field(default=99, ge=3, le=999)


class UprDecisionFeatureConfig(BaseModel):
    """The decision probability as a **pack feature** (the track's CS-13).

    The decision model's probability for the (text, label) cell becomes one
    more column of the pack's ``(n_texts, n_labels, n_features)`` cube. The
    pack does not call the model (the model is a client, and the package is a
    pure core): the CLI scores the cells at the edge and passes the matrix
    in, so the feature is **cell-local** by construction — no corpus
    statistic, no document identity, no length.

    Attributes:
        enabled: Whether the run computes the feature at all. The default is
            ``False``, so the earlier runs' artifacts keep their exact
            content.
        model: The decision model (pinned by id **and** digest, like every
            other decision model in the track).
        prompt_version: The prompt's version (part of the decision cache's
            key).
        host: The Ollama host (the empty value uses ``OLLAMA_HOST`` or the
            default).
        timeout: The call timeout in seconds.
        max_questions_per_call: How many questions one call may carry.
        label_blind: Whether the feature is the **label-blind counterpart's**
            probability (the wrong pairing, the same model and the same
            cost) — the yardstick that says how much of the feature's
            contribution is the model's own label-blind signal.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: DecisionModelConfig | None = None
    prompt_version: str = "v1"
    host: str = ""
    timeout: float = 600.0
    max_questions_per_call: int = Field(default=64, ge=1, le=64)
    label_blind: bool = False

    @model_validator(mode="after")
    def _model_is_required(self) -> UprDecisionFeatureConfig:
        """An enabled feature needs a pinned model."""
        if self.enabled and self.model is None:
            raise ValueError(
                "upr_analysis.learned.decision_feature is enabled but carries no "
                "model: the feature needs a pinned decision model (id and digest)"
            )
        return self


class UprFitMaskConfig(BaseModel):
    """The **fit-side mask** on the corpus map's tables (the track's CS-42).

    During the pack's **fit pass only**, the map's tables of a fixed,
    seed-chosen set of **seen** labels are emptied, so their gold rows carry the
    floor in the dictionary's column exactly like the held-out labels' rows do.
    The combiner then cannot learn "the channel is at the floor ⇒ the label is
    out of the question" — the coincidence CS-37 measured on the fit side.

    The **evaluation** side is untouched (the val split is scored with the full
    train map): the mask is a fit-side instrument, a "pretend unseen" control.

    The masked labels are never listed by hand: they are drawn from the train
    map's own label list (the labels with a non-empty table), so no val
    statistic and no per-run choice can pick them. The seed is separate from
    ``upr_analysis.learned.seed``, so the draw is pinned independently of the
    fold split and of the floors' permutations.

    Attributes:
        enabled: Whether the run fits the masked arm at all. The default is
            ``False``, so every earlier run's artifact keeps its exact content.
        n_labels: How many seen labels are masked (at least 1; the train map
            must hold that many labels with a table).
        seed: The draw's seed. The candidates are sorted by name before the
            draw, so it does not depend on the label space's order.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    n_labels: int = Field(default=8, ge=1)
    seed: int = 42


class UprBorrowConfig(BaseModel):
    """The **lent tables** on the corpus map (the track's CS-44).

    Five labels have (almost) no vocabulary of their own: their dictionary
    tables lie inside a partner's (CS-38's containment class). With the borrow
    on, such a label's column also reads the partner's **evidence-for** words
    (positive z-score) that its own table does not carry, each at ``discount`` ×
    the partner's weight — the words are lent, not copied, and the partner's own
    column is untouched.

    The lending relation is computed once from the **full train map** and
    recorded in the artifact; on the fit side the same relation rides into every
    fold, so each fold's cells read their own map's partner weights (the
    cross-fit guard). The default is off, so every earlier run's artifact keeps
    its exact content.

    Attributes:
        enabled: Whether the run fits the borrow arm at all.
        discount: The share of the partner's weight a lent word carries
            (CS-44's registered 0.5; one value, not swept).
        threshold: The containment share that makes a label a borrower (CS-38's
            registered 0.80; one value, not swept).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    discount: float = Field(default=0.5, gt=0.0, le=1.0)
    threshold: float = Field(default=0.8, gt=0.0, le=1.0)


class UprCorpusMapFeatureConfig(BaseModel):
    """The **corpus map**'s per-cell score as a **pack feature** (the track's CS-32).

    The corpus map (CS-23) is the train split's own word↔label association.
    Its per-cell score becomes one more column of the pack's
    ``(n_texts, n_labels, n_features)`` cube, so the learned combiner can use
    the corpus's word statistics beside the embedding and lexical features.

    The pack does **not** build the map: the map is built at the edge (from the
    train split) and its score matrix is passed in, exactly like the decision
    probability (CS-13) and the reranker's score (CS-16), so the column is
    cell-local by construction.

    **The guard.** The pack is fitted on train and the map is built from
    train, so the fit side's map must be **cross-fit** (each fold's cells
    scored against a map built from the other folds) — otherwise the feature
    is optimistically biased exactly where the combiner learns its weight.

    Attributes:
        enabled: Whether the run computes the feature at all. The default is
            ``False``, so the earlier runs' artifacts keep their exact
            content.
        scoring: Which scoring function of the map's scorer weights the cells
            (the z-scored "evidence for" sum by default — the arm CS-31's fit
            side selected).
        folds: The cross-fit's number of folds (at least 2).
        label_blind: Whether the run also computes the **label-blind
            counterpart** (the map built with the labels permuted among the
            train records) — the yardstick that says how much of the feature's
            contribution is the map's construction rather than the corpus's
            word↔label association.
        abstain: Whether the run also computes the **abstention** (the track's
            CS-37): a label with **no table at all** carries ``NaN``
            ("no evidence") instead of the floor ("impossible"), so the
            combiner can learn that the channel is silent there rather than
            that the label is out of the question. The default is ``False``,
            so every earlier run's artifact keeps its exact content. The
            encoding is ``NaN``, which the gradient-boosted combiner takes
            natively.
        fit_mask: The **fit-side mask** (the track's CS-42): the map's tables
            of a fixed, seed-chosen set of **seen** labels are emptied during
            the pack's fit pass only, so the combiner sees "positive with the
            channel at the floor" and cannot read the floor as "impossible".
            The evaluation side is untouched. The default is off, so every
            earlier run's artifact keeps its exact content.
        borrow: The **lent tables** (the track's CS-44): the labels with no
            vocabulary of their own also read their partner's evidence-for
            words, at a discount. The default is off, so every earlier run's
            artifact keeps its exact content.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    scoring: str = "sum_zscore_positive"
    folds: int = Field(default=5, ge=2)
    label_blind: bool = True
    abstain: bool = False
    fit_mask: UprFitMaskConfig = Field(default_factory=UprFitMaskConfig)
    borrow: UprBorrowConfig = Field(default_factory=UprBorrowConfig)


class UprLabelLexiconConfig(BaseModel):
    """The **label lexicon** as the pack's channel (the track's CS-35).

    The corpus map (CS-23) *is* the label→word lexicon; this config turns it
    into a **label-side representation**: per label, the words that characterise
    it (the PIFA idea — the label's TF-IDF centroid over its positive
    instances), and for the labels with no train example the **expansion**
    route (the label's name matched against the corpus, with no labels
    involved).

    The pack does **not** build the lexicon: it is built at the edge (from the
    train split) and the words ride on the label side, exactly like the
    description and the parent.

    **The guard.** The pack is fitted on train and the mined words come from
    train, so on the fit side the channel would be optimistically biased
    exactly where the combiner learns its weight. The fit side is therefore
    **cross-fit**: each fold's cells are scored with the label side built from
    the **other** folds (CS-32's rule, applied to the label side).

    Attributes:
        enabled: Whether the run computes the mined-words channel at all. The
            default is ``False``, so the earlier runs' artifacts keep their
            exact content.
        top_k: How many mined words a label keeps (CS-35's K).
        expansion_min_prefix: The expansion route's minimum shared prefix (in
            characters).
        folds: The cross-fit's number of folds (at least 2).
        word_classes: The word classes the lexicon keeps (the track's CS-45:
            CS-33's classes — ``distinctive``, ``shared``, ``generic``). ``None``
            (the default) keeps every class, so every earlier artifact keeps
            its exact content. The class is the map's own; an unknown class
            fails at build time.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    top_k: int = Field(default=10, ge=1)
    expansion_min_prefix: int = Field(default=4, ge=1)
    folds: int = Field(default=5, ge=2)
    word_classes: tuple[str, ...] | None = None


class UprBackoffConfig(BaseModel):
    """The **sibling-max backoff** on the UPR pack (the track's CS-11).

    The catalog's **parent** channel: a label with no fit-side support (zero
    gold cells on the train split) inherits the score of a taxonomy sibling.
    The backoff is a **fallback**, and the **control** beside it is the point:
    the same rule applied to a label that *has* fit-side support must not
    help it, or the honest claim changes to "another correlated signal".

    Attributes:
        enabled: Whether the run computes the backoff arms at all. The default
            is ``False``, so the earlier runs' artifacts keep their exact
            content.
        lams: The pre-registered mixing weights (the backoff's score is
            ``lam`` times the siblings' best, and the label's own score is the
            floor). Nothing is fitted on the evaluated split.
        control: Whether the control arm (the same rule on the fit side's
            rarest *supported* labels) is measured beside the backoff arm.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    lams: tuple[float, ...] = (0.5, 1.0)
    control: bool = True

    @field_validator("lams")
    @classmethod
    def _lams_are_positive(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject a negative or empty weight grid."""
        if not value:
            raise ValueError("upr_analysis.learned.backoff.lams must not be empty")
        if any(lam < 0 for lam in value):
            raise ValueError(f"upr_analysis.learned.backoff.lams must be >= 0, got {list(value)}")
        return value


class UprLearnedConfig(BaseModel):
    """The UPR learned arm (CS-8d): the feature pack, the combiner, the floors.

    The arm is **fitted on train** (the train gold cells plus the train
    non-gold cells as down-weighted presumed negatives) and **evaluated on
    val** — the main pipeline's protocol (AGENTS.md: the indexes and the fit
    side come from train, val is for tuning, test is touched once).

    Attributes:
        enabled: Whether the run computes the learned arms at all. The default
            is ``False``, so every earlier run's artifact keeps its exact
            content (the same precedent as the OMCT track's ``embeddings``
            knob).
        groups: The **feature groups** the combiner sees (the CS-8e's ablation
            unit). The default is the two catalog-free groups.
        top_k: The K of the top-K chunk similarities.
        chunker: The chunker whose chunks the pack's chunk-level features are
            measured on (the text side only).
        profile_top_k: How many ranks of the chunk profile are kept per
            granularity (the sentence profile and the word-piece profile).
        profile_ngram_size: The nominal size of the profile's word pieces.
        combiner: The learned scorer (the logistic regression by default; the
            gradient-boosted one is a config choice).
        control_weight: The training weight of the non-gold cells (presumed
            negatives whose true positive rate is unmeasured).
        floors: The three sanity floors.
        reporting: The reporting blocks (the calibration and the decision-rule
            comparison); off by default.
        backoff: The **sibling-max backoff** (the catalog's parent channel);
            off by default.
        rerank_feature: The **reranker's score** as a pack feature (the
            track's CS-16; the CS-13 precedent's shape). Off by default.
        decision_feature: The **decision probability** as a pack feature
            (CS-13); off by default.
        corpus_map_feature: The **corpus map**'s per-cell score as a pack
            feature (CS-32); off by default.
        label_lexicon: The **label lexicon** (the corpus-mined words) as the
            pack's label-side channel (CS-35); off by default.
        seed: The seed of the floors' permutations and of the random-vector
            draw. ``None`` uses the run's seed.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    groups: tuple[str, ...] = ("embedding", "lexical")
    top_k: int = Field(default=3, ge=1)
    chunker: ChunkerSpec = Field(
        default_factory=lambda: ChunkerSpec(name="atoms", params={"split_clauses": True})
    )
    profile_top_k: int = Field(default=3, ge=1)
    profile_ngram_size: int = Field(default=5, ge=2)
    combiner: CombinerSpec = Field(default_factory=CombinerSpec)
    control_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    floors: UprLearnedFloorsConfig = Field(default_factory=UprLearnedFloorsConfig)
    reporting: UprLearnedReportingConfig = Field(default_factory=UprLearnedReportingConfig)
    backoff: UprBackoffConfig = Field(default_factory=UprBackoffConfig)
    decision_feature: UprDecisionFeatureConfig = Field(default_factory=UprDecisionFeatureConfig)
    rerank_feature: UprRerankFeatureConfig = Field(default_factory=UprRerankFeatureConfig)
    corpus_map_feature: UprCorpusMapFeatureConfig = Field(default_factory=UprCorpusMapFeatureConfig)
    label_lexicon: UprLabelLexiconConfig = Field(default_factory=UprLabelLexiconConfig)
    seed: int | None = None

    @field_validator("groups")
    @classmethod
    def _groups_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an unknown, an empty or a repeated feature group at config time."""
        from cluster_rag.text_processors.features.upr_features import UPR_FEATURE_GROUPS

        if not value:
            raise ValueError("upr_analysis.learned.groups must name at least one feature group")
        unknown = [group for group in value if group not in UPR_FEATURE_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown feature group(s) {unknown} in upr_analysis.learned.groups; "
                f"expected values in {list(UPR_FEATURE_GROUPS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError(f"upr_analysis.learned.groups must not repeat a group: {list(value)}")
        return value

    @model_validator(mode="after")
    def _top_k_is_usable(self) -> UprLearnedConfig:
        """``top_k = 2`` would duplicate the pack's top-2 mean column."""
        if self.top_k == 2:
            raise ValueError(
                "upr_analysis.learned.top_k=2 would duplicate the pack's top-2 mean column: "
                "use 1 or a value >= 3"
            )
        return self


class UprAnalysisConfig(BaseModel):
    """The text-processing track's UPR analysis (``analyze-upr``) settings.

    The UPR counterpart of :class:`AnalysisConfig`: the chunked arm families,
    the text-length buckets (the UPR texts' length is the axis the chunking
    question is really about) and the reference ``evaluate`` run the
    whole-text arm must reproduce.

    Attributes:
        arms: The chunked arm families. The whole-text **reference arm** is
            always computed (it is the pipeline's own scoring), so an empty
            list means "the reference arm only".
        weight_grid: The fusion weights re-fitted on val for the chunked
            components.
        length_buckets: The upper edges (in characters) of the text-length
            buckets. The texts are bucketed into
            ``[0, e1)``, ``[e1, e2)``, ..., ``[e_last, inf)``; the buckets are
            reported separately, because the pooled number hides the long-text
            tail the chunking hypothesis is about.
        reference_eval_run: The ``evaluate`` run whose numbers the whole-text
            reference arm must reproduce, field by field (the guard). ``None``
            disables the check (the run then records that it was not checked).
        semantic_thresholds: The grid of the semantic chunker's threshold. The
            threshold is **selected on the fit side** (the train split).
        semantic_folds: The number of the fit-side cross-fit folds over the
            train records (the UPR analogue of OMCT's document rotation).
        semantic_criterion: The fit-side criterion of the threshold selection.
        bootstrap: The paired-bootstrap settings of the arm comparisons.
        learned: The **learned arm** (CS-8d): the feature pack, the combiner
            and the three floors. Off by default, so every earlier run's
            artifact keeps its exact content.
        decision: The **decision probe** (CS-9): the models, the prompt grid
            and the scope. Off by default (no models = no probe).
        hitmap: The **hit map** (CS-18): the ablation unit and the map's own
            knobs. Off by default is expressed by the decision probe's own
            absence (the map has no model of its own: it reads the decision
            probe's pinned model).
        rerank: The **rerankers** (CS-15): the models, the candidate set they
            re-order and the label side's variants. Off by default, so the
            earlier runs' artifacts keep their exact content.
    """

    model_config = ConfigDict(extra="forbid")

    arms: tuple[UprArmConfig, ...] = ()
    weight_grid: UprWeightGridConfig = Field(default_factory=UprWeightGridConfig)
    length_buckets: tuple[int, ...] = (100, 200, 300)
    reference_eval_run: str | None = None
    semantic_thresholds: tuple[float, ...] = ()
    semantic_folds: int = Field(default=3, ge=2)
    semantic_criterion: Literal["micro_f1", "ndcg"] = "micro_f1"
    bootstrap: PairedBootstrapConfig = Field(default_factory=PairedBootstrapConfig)
    learned: UprLearnedConfig = Field(default_factory=UprLearnedConfig)
    decision: UprDecisionConfig = Field(default_factory=UprDecisionConfig)
    hitmap: UprHitMapConfig = Field(default_factory=UprHitMapConfig)
    rerank: UprRerankConfig = Field(default_factory=UprRerankConfig)

    @field_validator("length_buckets")
    @classmethod
    def _length_buckets_are_usable(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject a non-positive or repeated length-bucket edge."""
        if any(edge < 1 for edge in value):
            raise ValueError(f"upr_analysis.length_buckets must be positive, got {value}")
        if len(set(value)) != len(value):
            raise ValueError(f"upr_analysis.length_buckets must not repeat an edge: {value}")
        return tuple(sorted(value))

    @field_validator("semantic_thresholds")
    @classmethod
    def _thresholds_are_usable(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject a threshold outside the cosine range or a repeated grid point."""
        for threshold in value:
            if not -1.0 <= threshold <= 1.0:
                raise ValueError(
                    f"upr_analysis.semantic_thresholds must be in [-1, 1], got {threshold}"
                )
        if len(set(value)) != len(value):
            raise ValueError(f"upr_analysis.semantic_thresholds must not repeat a value: {value}")
        return value

    @model_validator(mode="after")
    def _family_names_are_unique(self) -> UprAnalysisConfig:
        """Two families of one run must not share a name (the arm names' prefix)."""
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"upr_analysis.arms must have unique names: {names}")
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
    # The part-of-speech tagger (the track's CS-34): a model client, off by
    # default. When it is on, the train corpus is tagged and the words'
    # grammatical classes ride into the corpus map and the pack.
    pos: PosConfig = Field(default_factory=PosConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    eval: EvalConfig | None = None  # Phase 3: the `evaluate` command requires it
    # The text-processing track's Stage-0 analysis (the `analyze-omct`
    # command): the models to score with and the control-set design.
    analysis: AnalysisConfig | None = None
    # The text-processing track's UPR analysis (the `analyze-upr` command): the
    # arms to score with, the text-length buckets and the guard's reference.
    upr_analysis: UprAnalysisConfig | None = None
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
