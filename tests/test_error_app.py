"""Smoke tests for the error viewer's Streamlit app (V1).

The app is exercised with **Streamlit's own AppTest** on the tiny synthetic run
from ``test_error_view`` — no browser, no network, no real artifact. The test
pins the two behaviours the meeting depends on: every view renders, and a failed
guard blocks the app instead of showing stale numbers.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
TESTS = Path(__file__).resolve().parent
VIEWS = ("Errors", "One text", "Aggregates", "Compare runs", "Compare hitmaps", "Export")


def _fixture_module() -> Any:
    """The data-layer tests' fixture builder (the tiny synthetic run)."""
    name = "test_error_view"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TESTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _point_at(monkeypatch: pytest.MonkeyPatch, config: Path, results: Path) -> None:
    """Point the app at the fixture (its own environment defaults)."""
    monkeypatch.setenv("ERROR_VIEW_CONFIG", str(config))
    monkeypatch.setenv("ERROR_VIEW_RESULTS", str(results))
    monkeypatch.setenv("ERROR_VIEW_SPLIT", "val")
    monkeypatch.setenv("ERROR_VIEW_CORPUS_RUN", "upr-hitmap-corpus")
    monkeypatch.setenv("ERROR_VIEW_EVAL_RUN", "upr-eval-corpus-map-sum-zscore-positive-val")
    monkeypatch.setenv("ERROR_VIEW_SURROGATE_RUN", "upr-hitmap-surrogate")


@pytest.fixture()
def fixture(tmp_path: Path) -> Any:
    """The tiny synthetic run."""
    for name in ("ERROR_VIEW_CONFIG", "ERROR_VIEW_RESULTS"):
        os.environ.pop(name, None)
    return _fixture_module()._build_fixture(tmp_path)


def _app(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The app, pointed at the fixture."""
    _point_at(monkeypatch, fixture.config, fixture.results)
    return AppTest.from_file(str(TOOLS / "error_app.py"), default_timeout=30)


def test_the_app_renders_every_view(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """All five views render, the guard passes, and no error banner appears."""
    app = _app(fixture, monkeypatch)
    app.run()
    assert not app.exception
    assert app.title[0].value.startswith("Error viewer")
    assert not app.error
    assert app.get_by_key("view") is not None
    for view in VIEWS:
        app.get_by_key("view").set_value(view).run()
        assert not app.exception, view
        assert not app.error, view
    # the guard's own report is on the screen
    assert any("reproduction: OK" in code.value for code in app.code)


def test_the_errors_view_lists_a_mistake(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default Errors view shows the hand-made mistakes and their words."""
    app = _app(fixture, monkeypatch)
    app.run()
    assert not app.exception
    shown = " ".join(str(element.value) for element in app.markdown)
    assert "ev-table" in shown, "the Errors view printed no table"
    assert "val-2" in shown
    assert (
        'data-columns="text|error|label|score|rank|gap_to_cut|gold|predicted|'
        'has_evidence|length_bucket|top_words|text_id"'
    ) in shown
    # the cells are the read-only diagnostics, not plain text
    assert "ev-text" in shown  # the text expands in place on hover
    assert "title=" not in shown  # no tooltip attribute
    assert "ev-pill" in shown  # the contributing words
    assert "ev-dot" in shown  # the booleans
    captions = " ".join(str(element.value) for element in app.caption)
    assert "scroll inside the table" in captions


def test_the_one_text_view_shows_the_matrix(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The One text view renders the ranking and the word x label matrix."""
    app = _app(fixture, monkeypatch)
    app.run()
    app.get_by_key("view").set_value("One text").run()
    assert not app.exception
    assert len(app.dataframe) >= 1  # the word x label matrix
    shown = " ".join(str(element.value) for element in (*app.markdown, *app.caption))
    assert "words x labels matrix" in shown
    assert "contrast B:" in shown
    assert ('data-columns="rank|label|score|gold|predicted|error|has_evidence|gap_to_cut"') in shown


def test_the_compare_view_pairs_the_two_synthetic_runs(
    fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Compare view joins the two series by their per-text counts."""
    app = _app(fixture, monkeypatch)
    app.run()
    app.get_by_key("view").set_value("Compare runs").run()
    assert not app.exception
    metric_labels = " ".join(str(element.label) for element in app.metric)
    assert "Δ micro-F1" in metric_labels
    values = " ".join(str(element.value) for element in app.metric)
    assert "+0.6667" in values
    shown = " ".join(str(element.value) for element in app.markdown)
    assert (
        'data-columns="text|delta_f1|f1_a|f1_b|direction|delta_tp|delta_fp|delta_fn|text_id"'
    ) in shown
    assert "ev-badge-improved" in shown


def test_the_compare_hitmaps_view(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hitmap view: one row per word, gold + FP + alternative label columns."""
    app = _app(fixture, monkeypatch)
    app.run()
    app.get_by_key("view").set_value("Compare hitmaps").run()
    assert not app.exception
    shown = " ".join(str(element.value) for element in app.markdown)
    # val-1 is gold A, predicted A + B; D is the strongest alternative with a table
    assert 'data-columns="word|n|A|B|D"' in shown
    assert "ev-h-gold" in shown and "ev-h-fp" in shown and "ev-h-alt" in shown
    assert "ev-z-pos" in shown and "ev-z-none" in shown
    # val-1's picks are A (gold, predicted) and B (fp): two header badges
    assert shown.count('<span class="ev-pick">') == 2
    assert "classifier's pick" in shown
    assert "alpha" in shown and "beta" in shown


def test_the_aggregates_tables_are_styled(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The aggregate tables carry the same HTML diagnostics as the Errors view."""
    app = _app(fixture, monkeypatch)
    app.run()
    app.get_by_key("view").set_value("Aggregates").run()
    assert not app.exception
    shown = " ".join(str(element.value) for element in app.markdown)
    assert (
        'data-columns="label|bucket|n_gold|n_pred|tp|fp|fn|near_miss|'
        'precision|recall|miss_rate|fp_rate"'
    ) in shown
    assert 'data-columns="kind|bucket|n_cells|tp|fp|fn|near_miss|micro_f1"' in shown
    assert "ev-chip" in shown
    assert "ev-meter" in shown


def test_the_export_view_offers_the_csv(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Export view prepares a CSV with the provenance block."""
    app = _app(fixture, monkeypatch)
    app.run()
    app.get_by_key("view").set_value("Export").run()
    assert not app.exception
    assert app.download_button, "the export offered no download"
    blocks = " ".join(str(element.value) for element in app.code)
    assert "# config:" in blocks
    # the compare table is exportable too (the two synthetic series)
    app.get_by_key("export_table").set_value("compare (A/B)").run()
    assert not app.exception
    assert app.download_button
    assert "delta_f1" in " ".join(str(element.value) for element in app.code)


def test_a_failed_guard_blocks_the_app(fixture: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale artifact is a loud failure: the views never render."""
    path = fixture.results / "upr-eval-corpus-map-sum-zscore-positive-val" / "metrics.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["report"]["overall"]["f1"]["micro_f1"] = 0.9
    path.write_text(json.dumps(payload), encoding="utf-8")
    app = _app(fixture, monkeypatch)
    app.run()
    assert not app.exception
    assert app.error
    assert "DISAGREE" in app.error[0].value
    assert not app.dataframe


def test_the_catppuccin_mocha_theme_is_configured() -> None:
    """The app's Streamlit theme is the Catppuccin Mocha palette."""
    theme = tomllib.loads(
        (Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml").read_text(
            encoding="utf-8"
        )
    )["theme"]
    assert theme["base"] == "dark"
    assert theme["backgroundColor"] == "#1e1e2e"  # base
    assert theme["secondaryBackgroundColor"] == "#181825"  # mantle
    assert theme["textColor"] == "#cdd6f4"  # text
    assert theme["primaryColor"] == "#cba6f7"  # mauve
    assert theme["baseFontSize"] == 15
    assert "sans-serif" in theme["font"]


def _app_module() -> Any:
    """The app module (for the pure HTML-cell helpers)."""
    name = "error_app_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / "error_app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_signed_value_colours_are_catppuccin() -> None:
    """The table's sign colours are Mocha's green, red and overlay0."""
    module = _app_module()
    assert module._sign_colour(1.0) == "color: #a6e3a1"
    assert module._sign_colour(-1.0) == "color: #f38ba8"
    assert module._sign_colour(0.0) == "color: #6c7086"
    assert module._sign_colour(float("nan")) == "color: #6c7086"


def test_the_cells_are_read_only_diagnostics() -> None:
    """The cells carry HTML, not checkbox-looking booleans or plain text."""
    module = _app_module()
    text = module._expandable_text("A long recommendation " * 20)
    assert text.startswith('<span class="ev-text"')
    assert "title=" not in text
    # the full text is in the cell: the CSS clips it, and hover expands it
    assert "A long recommendation A long recommendation" in text
    assert "…" not in text
    assert "★" in module._gold_dot(True)
    assert "ev-dot-off" in module._gold_dot(False)
    assert "○" in module._evidence_dot(False)
    assert "ev-badge-red" in module._error_badge("fp")
    assert "ev-badge-amber" in module._error_badge("near_miss")
    assert "ev-badge-green" in module._error_badge("tp")
    assert "ev-pill" in module._word_pills([("alpha", 2.0)], 2.0)
    assert "ev-chip" in module._chip(2.0, 4.0, tone="score")
    assert "ev-bar" in module._meter(2.0, 4.0)
    table = module._html_table(["text", "delta_f1"], [["x", "y"]], columns=["text", "delta_f1"])
    assert 'data-columns="text|delta_f1"' in table
    assert "ev-table" in table


def test_the_stylesheet_carries_the_table_look() -> None:
    """The CSS file is the Catppuccin Mocha palette and the table's own classes."""
    css = (TOOLS / "error_app.css").read_text(encoding="utf-8")
    assert "--ctp-base: #1e1e2e" in css
    assert ".ev-table" in css
    assert ".ev-pill" in css
    assert ".ev-chip" in css
    assert ".ev-dot" in css
    assert ".ev-text:hover" in css
    assert "white-space: normal" in css
    # the table scrolls inside its own frame, so the header can stay pinned
    assert "max-height: 72vh" in css
    assert "position: sticky" in css
    assert "z-index: 2" in css
    assert ".ev-z-pos" in css
    assert ".ev-h-gold" in css
    assert ".ev-pick" in css
    # the hitmap headers are top-aligned and one line each, so they line up
    assert ".ev-hitmap thead th { vertical-align: top; }" in css
    assert "white-space: nowrap" in css
