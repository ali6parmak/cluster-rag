"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_jsonl() -> Path:
    """Path to the tiny fixture dataset (~20 rows, see AGENTS.md testing rules)."""
    return FIXTURE_DIR / "sample.jsonl"
