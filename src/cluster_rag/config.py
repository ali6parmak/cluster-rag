"""Configuration loading: YAML files validated with pydantic.

Every CLI command takes ``--config`` pointing at a YAML file (see
``configs/base.yaml``). Values live in the config, never in the code, so the
same code can be reused across experiments (``configs/experiments/*.yaml``).

Path resolution: relative paths in a config are resolved against the current
working directory. Run the CLI from the repository root, as in AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    ``revision``/``trust_remote_code`` only to the HF backend. Irrelevant keys
    are ignored with the backend that does not use them.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str  # "hf" (sentence-transformers) or "ollama" (/api/embed)
    model_id: str = Field(min_length=1)
    dimension: int | None = Field(default=None, ge=1)  # Ollama fallback if /api/show hides the size
    prefixes: PrefixesConfig = Field(default_factory=PrefixesConfig)
    batch_size: int = Field(default=64, ge=1)
    device: str | None = None  # HF only: null = let sentence-transformers pick (cuda if present)
    revision: str | None = None  # HF only: pin the model revision (AGENTS.md)
    trust_remote_code: bool = False  # HF only
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


class ProbeConfig(BaseModel):
    """Parameters of the Phase 2 auxiliary probe on the pairwise dataset."""

    model_config = ConfigDict(extra="forbid")

    question_transforms: tuple[str, ...] = ("keep", "strip_block")
    question_block_markers: tuple[str, ...] = ("Answer choices:", "Notes:")
    recall_ks: tuple[int, ...] = (1, 10)


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
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)


def _parse_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file and require a mapping at the top level."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return raw


def _merge_configs(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge the child experiment over its parent config."""
    merged: dict[str, Any] = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_configs(merged[key], value)
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
        raw = _merge_configs(_load_raw_chain(parent_path, _depth=_depth + 1), raw)
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
