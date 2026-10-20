"""Tests for preprocessing variants (Phase 2): raw/lowercase/strip + omct question-block strip."""

from __future__ import annotations

import pytest

from cluster_rag.config import DEFAULT_STRIP_OPENERS, PreprocessConfig
from cluster_rag.preprocess import (
    apply_question_transform,
    build_preprocess_spec,
    question_transform_version,
    strip_question_block,
)


def test_raw_is_identity() -> None:
    """raw must not touch anything (whitespace included)."""
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    assert spec.version == "raw-v1"
    assert spec.apply(["  Keep  spacing ", ""]) == ["  Keep  spacing ", ""]
    assert spec.describe(["a b"])["n_changed"] == 0


def test_lowercase_variant() -> None:
    """lowercase collapses case only, and reports its coverage."""
    spec = build_preprocess_spec(PreprocessConfig(variant="lowercase"))
    assert spec.version == "lowercase-v1"
    assert spec.apply(["Continue To Act", "OK"]) == ["continue to act", "ok"]
    meta = spec.describe(["Continue To Act", "ok"])
    assert meta["n_changed"] == 1 and abs(meta["share_changed"] - 0.5) < 1e-9


def test_strip_removes_longest_opener_match() -> None:
    """The opener list is matched longest-first with word boundaries."""
    spec = build_preprocess_spec(PreprocessConfig(variant="strip_leading_formulaic"))
    texts = [
        "Continue to implement the strategy",
        "Continue its cooperation with OHCHR",  # "continue its" wins over "continue"
        "Ensure that all children attend school",  # "ensure that"
        "Ratify the treaty",  # plain verb opener
        "Take measures to combat trafficking",  # "take measures to"
        "The government should act",  # untouched
        "continuous monitoring is different",  # boundary check: no strip
    ]
    out = spec.apply(texts)
    assert out == [
        "implement the strategy",
        "cooperation with OHCHR",
        "all children attend school",
        "the treaty",
        "combat trafficking",
        "The government should act",
        "continuous monitoring is different",
    ]


def test_strip_openers_come_from_config_not_code() -> None:
    """A custom opener list overrides the default and changes the version."""
    config = PreprocessConfig(variant="strip_leading_formulaic", strip_openers=("note that",))
    custom = build_preprocess_spec(config)
    assert custom.apply(["Note that this must stop"]) == ["this must stop"]
    default = build_preprocess_spec(PreprocessConfig(variant="strip_leading_formulaic"))
    assert custom.version != default.version  # cache honesty: different rule -> different key


def test_strip_version_follows_opener_list_content() -> None:
    """Same opener list -> same version; reordered list -> same version (set)."""
    variant = "strip_leading_formulaic"
    a = build_preprocess_spec(PreprocessConfig(variant=variant, strip_openers=("a", "b")))
    b = build_preprocess_spec(PreprocessConfig(variant=variant, strip_openers=("b", "a")))
    c = build_preprocess_spec(PreprocessConfig(variant=variant, strip_openers=("a", "c")))
    assert a.version == b.version and a.version != c.version
    assert a.version.startswith("strip_leading_formulaic-v1-")


def test_strip_never_returns_empty_text() -> None:
    """An opener-only text keeps the stripped original instead of becoming ''."""
    spec = build_preprocess_spec(PreprocessConfig(variant="strip_leading_formulaic"))
    assert spec.apply_one("continue to") == "continue to"
    assert spec.apply_one("Take.") == "Take."
    assert spec.apply_one("Continue,") == "Continue,"


def test_strip_trailing_opener_punctuation() -> None:
    """Opener punctuation is dropped ('Ratify, as soon as possible, ...')."""
    spec = build_preprocess_spec(
        PreprocessConfig(variant="strip_leading_formulaic", strip_openers=("ratify",))
    )
    assert spec.apply_one("Ratify, as soon as possible, the treaty") == (
        "as soon as possible, the treaty"
    )
    meta = spec.describe(["Ratify, as soon as possible, the treaty", "The verdict"])
    assert meta["n_changed"] == 1 and meta["mean_chars_removed"] > 0


def test_strip_reports_diagnostics() -> None:
    """describe() exposes coverage stats fed into run metadata."""
    spec = build_preprocess_spec(PreprocessConfig(variant="strip_leading_formulaic"))
    texts = ["Continue to act", "leave me alone", "ensure that it works"]
    meta = spec.describe(texts)
    assert meta["n_changed"] == 2 and meta["n_texts"] == 3
    assert abs(meta["share_changed"] - 2 / 3) < 1e-9
    assert list(meta["openers"]) == list(DEFAULT_STRIP_OPENERS)  # schema default in use


def test_strip_openers_default_covers_continue_share() -> None:
    """On the real UPR data the default openers strip ~49%+ of texts (Phase 1:
    "continue" alone opens 535/3000 = 17.8%; the default covers the top verbs)."""
    import json
    from pathlib import Path

    raw = Path(__file__).parents[1] / "data/raw/upr.jsonl"
    if not raw.exists():  # dataset not present in this checkout
        pytest.skip("real dataset not available")
    texts = [json.loads(line)["text"] for line in raw.open(encoding="utf-8")]
    spec = build_preprocess_spec(PreprocessConfig(variant="strip_leading_formulaic"))
    meta = spec.describe(texts)
    assert meta["n_texts"] == 3000
    assert meta["share_changed"] >= 0.49


def test_unknown_variant_raises() -> None:
    """Typos fail loudly instead of silently running raw."""
    with pytest.raises(ValueError, match="variant"):
        build_preprocess_spec(PreprocessConfig(variant="strip_leading_phrase"))


def test_strip_question_block_keeps_the_question_sentence() -> None:
    """Aux transform: drop 'Question:' label and the trailing blocks (PLAN §7)."""
    with_choices = (
        "Question: How often were people displaced?\n\nAnswer choices:\n"
        "- Very often\n- Often\n- Rarely"
    )
    assert (
        strip_question_block(with_choices, ("Answer choices:", "Notes:"))
        == "How often were people displaced?"
    )
    with_notes = "Question: Has the treaty been ratified?Notes:Check OHCHR records"
    assert strip_question_block(with_notes) == "Has the treaty been ratified?"
    clean = "Has the treaty been ratified?"
    assert strip_question_block(clean) == clean


def test_apply_question_transform_and_version() -> None:
    """keep = identity; strip_block applies to every text; versions differ."""
    markers = ("Answer choices:", "Notes:")
    texts = ["a", "Question: b\n\nAnswer choices:\n- c"]
    assert apply_question_transform(texts, "keep", markers) == texts
    assert apply_question_transform(texts, "strip_block", markers) == ["a", "b"]
    assert question_transform_version("keep", markers) == "question-transform-keep-v1"
    strip_version = question_transform_version("strip_block", markers)
    assert strip_version.startswith("question-transform-strip_block-v1-")
    # a different marker list changes the version (cache honesty)
    assert question_transform_version("strip_block", ("Notes:",)) != strip_version
    with pytest.raises(ValueError, match="transform"):
        question_transform_version("uppercase", markers)
