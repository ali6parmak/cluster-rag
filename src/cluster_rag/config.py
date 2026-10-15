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
from pydantic import BaseModel, ConfigDict


class PathsConfig(BaseModel):
    """Filesystem locations used across the pipeline."""

    model_config = ConfigDict(extra="forbid")

    data_raw: Path
    data_processed: Path
    data_splits: Path
    cache: Path
    results: Path
    label_catalog: Path | None = None


class AppConfig(BaseModel):
    """Root configuration validated from a YAML file.

    Unknown keys are rejected (``extra="forbid"``) so typos in a config file
    fail immediately instead of silently changing behavior.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    seed: int = 42
    paths: PathsConfig


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
