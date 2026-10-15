"""Configuration loading: YAML files validated with pydantic.

Every CLI command takes ``--config`` pointing at a YAML file (see
``configs/base.yaml``). Values live in the config, never in the code, so the
same code can be reused across experiments (``configs/experiments/*.yaml``).

Path resolution: relative paths in a config are resolved against the current
working directory. Run the CLI from the repository root, as in AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathsConfig(BaseModel):
    """Filesystem locations used across the pipeline."""

    model_config = ConfigDict(extra="forbid")

    data_raw: Path
    data_processed: Path
    data_splits: Path
    cache: Path
    results: Path
    label_catalog: Path | None = None


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


class AppConfig(BaseModel):
    """Root configuration validated from a YAML file.

    Unknown keys are rejected (``extra="forbid"``) so typos in a config file
    fail immediately instead of silently changing behavior.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    seed: int = 42
    paths: PathsConfig
    stats: StatsConfig = Field(default_factory=StatsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file into an :class:`AppConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is not valid YAML or does not match the schema.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping at the top level.")
    return AppConfig.model_validate(raw)
