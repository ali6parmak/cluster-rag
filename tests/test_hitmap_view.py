"""Tests for the hit map's **terminal viewer** (the track's CS-20).

The viewer is a **tool**, so its tests are about the tool's own contract:

- the **pure** half: the per-word decomposition (the contributions sum to the
  feature's value), the two lexical rankings and the word removal;
- the **offline** mode: the viewer's own numbers are pinned against the
  artifacts (the map's own runs on the tiny fixture dataset), and a **stale
  artifact fails loudly**;
- the **live** and **what-if** modes: exercised against a **fake decider** (no
  server in the tests), and the output says which of its columns are
  attributions and which one is causal.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from cluster_rag.cli import main
from cluster_rag.decision.base import DecisionAnswer, DecisionInfo

TOOLS = Path(__file__).resolve().parent.parent / "tools"

# The hit map's config for the tiny fixture dataset: the pinned fake model and
# the two cheap modes the viewer reads.
BASE_EXTRA = """
eval:
  predictor:
    name: most_frequent
  decision:
    rule: top_k
    k: 2
upr_analysis:
  length_buckets: [40, 80]
  decision:
    models:
      - name: fake-paragraph-noul
        model_id: fake:latest
        digest: ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        state: paragraph
        question_type: noul
        instructions: Is the text about the topic in the question?
        max_input_tokens: 2500
    scope: grid
    prompt_version: v1
    mismatched_control: true
  hitmap:
    unit: clause
    include_label_blind: true
    catch_all_labels: [General, Other]
    surrogate_min_df: 1
"""

TEXT_MODE_EXTRA = (
    BASE_EXTRA
    + """    include_text_side_map: true
    include_surrogate: false
    include_ablation: false
"""
)

SURROGATE_MODE_EXTRA = (
    BASE_EXTRA
    + """    include_text_side_map: false
    include_surrogate: true
    include_ablation: true
"""
)

CORPUS_MODE_EXTRA = (
    BASE_EXTRA
    + """    include_lexical_map: false
    include_text_side_map: false
    include_surrogate: false
    include_ablation: false
    include_corpus_map: true
"""
)


def _load(name: str) -> Any:
    """Import a tool module from the repository's tools folder."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # the module is registered before it runs: the tool's dataclasses resolve
    # their own module through sys.modules
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeDecider:
    """A decider that answers from the state's text (no server, no model).

    The answer is the "the label's name is in the text" rule, so a text that
    loses a word really loses the probability: the what-if mode has something to
    measure.
    """

    def __init__(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_stored = 0
        self.calls: list[str] = []
        self._info = DecisionInfo(model_id="fake:latest", digest="f" * 64, backend="fake")

    @property
    def info(self) -> DecisionInfo:
        """The fake model's provenance."""
        return self._info

    def decide(self, state: Any, questions: Any) -> dict[str, DecisionAnswer]:
        """Answer each question with 0.9 when the label's words are in the text."""
        text = str(state.get("paragraph", "")).lower()
        self.calls.append(text)
        answers: dict[str, DecisionAnswer] = {}
        for name, spec in questions.items():
            label = spec.instructions.split("Question: ", 1)[-1].split("\n", 1)[0].lower()
            value = 0.9 if label and label in text else 0.1
            answers[name] = DecisionAnswer(name=name, type="noul", value=value)
        return answers


def _config(path: Path, data_raw: Path, tmp_path: Path, run_id: str, extra: str) -> Path:
    """Write a config for the fixture dataset (the paths and the mode's section)."""
    path.write_text(
        f"run_id: {run_id}\n"
        "paths:\n"
        f"  data_raw: {data_raw.as_posix()}\n"
        f"  data_processed: {(tmp_path / 'processed').as_posix()}\n"
        f"  data_splits: {(tmp_path / 'splits').as_posix()}\n"
        f"  cache: {(tmp_path / 'cache').as_posix()}\n"
        f"  results: {(tmp_path / 'results').as_posix()}\n" + extra,
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def viewer_module() -> Any:
    """The viewer tool, loaded from the repository's tools folder."""
    return _load("hitmap_view")


@pytest.fixture()
def fixture_artifacts(sample_jsonl, tmp_path, monkeypatch) -> dict[str, Any]:
    """The map's two runs on the tiny fixture dataset (the artifacts the viewer reads).

    The runs are the CS-19 pair (the text-side map and the surrogate), executed
    on the fixture dataset with a **fake decider** — so the whole fixture path
    needs no server and no model. The **corpus map** (CS-23) is a third run, on
    the same fixture.
    """
    from cluster_rag.commands import upr as upr_commands

    decider = FakeDecider()
    monkeypatch.setattr(upr_commands, "build_upr_decider", lambda model_config, config: decider)
    split_config = _config(tmp_path / "split.yaml", sample_jsonl, tmp_path, "upr-split", BASE_EXTRA)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(split_config)]) == 0
    text_config = _config(
        tmp_path / "text.yaml", sample_jsonl, tmp_path, "upr-hitmap-fixture-text", TEXT_MODE_EXTRA
    )
    assert main(["hitmap-upr", "--config", str(text_config)]) == 0
    surrogate_config = _config(
        tmp_path / "surrogate.yaml",
        sample_jsonl,
        tmp_path,
        "upr-hitmap-fixture-surrogate",
        SURROGATE_MODE_EXTRA,
    )
    assert main(["hitmap-upr", "--config", str(surrogate_config)]) == 0
    corpus_config = _config(
        tmp_path / "corpus.yaml",
        sample_jsonl,
        tmp_path,
        "upr-hitmap-fixture-corpus",
        CORPUS_MODE_EXTRA,
    )
    assert main(["hitmap-upr", "--config", str(corpus_config)]) == 0
    ids = json.loads((tmp_path / "splits" / "val_ids.json").read_text("utf-8"))["ids"]
    return {
        "config": tmp_path / "text.yaml",
        "results": tmp_path / "results",
        "text_run": "upr-hitmap-fixture-text",
        "surrogate_run": "upr-hitmap-fixture-surrogate",
        "corpus_run": "upr-hitmap-fixture-corpus",
        "val_ids": ids,
    }


def _view_args(artifacts: dict[str, Any], *extra: str) -> list[str]:
    """The viewer's own arguments (the fixture's artifacts, one val text)."""
    return [
        "--config",
        str(artifacts["config"]),
        "--text-map-run",
        str(artifacts["text_run"]),
        "--surrogate-run",
        str(artifacts["surrogate_run"]),
        "--corpus-map-run",
        str(artifacts["corpus_run"]),
        "--no-colour",
        *extra,
    ]


def _coloured_args(artifacts: dict[str, Any], *extra: str) -> list[str]:
    """The same arguments with the colours forced on (the pipe case)."""
    args = _view_args(artifacts, *extra)
    args.remove("--no-colour")
    return [*args[: args.index("--text")], "--colour", *args[args.index("--text") :]]


# --- the pure half ---


def test_the_words_sum_to_the_feature(viewer_module: Any) -> None:
    """The viewer's per-word rows are the feature's own decomposition."""
    text = "Ensure safe working conditions and protect the labour rights of migrant workers"
    rows = viewer_module.word_rows(text, "Labour rights")
    assert sum(float(row["contribution"]) for row in rows) == pytest.approx(
        viewer_module.text_side_coverage(text, "Labour rights")
    )
    words = {str(row["word"]): row for row in rows}
    assert words["labour"]["coverage"] == pytest.approx(1.0)
    assert words["migrant"]["coverage"] == pytest.approx(0.0)
    # the strongest word comes first
    assert str(rows[0]["word"]) == "labour"


def test_the_two_rankings_are_the_maps_own(viewer_module: Any) -> None:
    """The text-side and the topic-side rankings are the map's two features."""
    text = "Ensure safe working conditions and protect the labour rights of migrant workers"
    labels = ["Labour rights", "Migrants", "Poverty"]
    text_side = viewer_module.lexical_ranking(text, labels)
    topic_side = viewer_module.topic_ranking(text, labels)
    assert text_side[0][0] == "Labour rights"
    assert topic_side[0][0] in {"Labour rights", "Migrants"}
    # the two features are not symmetric: the topic side sees the topic's words
    assert viewer_module.cell_coverage(text, "Labour rights") == pytest.approx(
        viewer_module.topic_ranking(text, labels)[0][1]
    )


def test_the_removal_takes_the_word_out(viewer_module: Any) -> None:
    """The causal removal: the word is gone and the rest of the text is kept."""
    text = "Take further concrete steps towards the ratification of the Convention"
    variant, count = viewer_module.remove_word(text, "ratification")
    assert count == 1
    assert "ratification" not in variant.lower()
    assert "concrete steps" in variant
    assert "Convention" in variant
    with pytest.raises(ValueError):
        viewer_module.remove_word(text, "missing")


def test_the_seeded_draw_is_deterministic(viewer_module: Any) -> None:
    """The same seed picks the same text (the track's own draw convention)."""
    records = [type("R", (), {"id": f"upr-{index:04d}"})() for index in range(20)]
    first = viewer_module.pick_row(records, text_id=None, sample=True, seed=7)
    again = viewer_module.pick_row(records, text_id=None, sample=True, seed=7)
    assert first == again
    assert viewer_module.pick_row(records, text_id="upr-0003", sample=False, seed=7) == 3
    with pytest.raises(ValueError):
        viewer_module.pick_row(records, text_id="upr-9999", sample=False, seed=7)


# --- the offline mode ---


def test_the_viewer_runs_offline_on_the_fixture(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The default mode reads the artifacts only: no model call, no server."""
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]))
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert "HIT MAP VIEWER" in out
    assert "UPR val" in out
    assert "WHAT IS TRUE" in out
    assert "THE FEATURE'S OWN SPLIT" in out
    assert "THE MODEL'S WORD HABITS" in out
    assert "THE TOPIC'S OWN WORDS" in out


def test_the_viewers_numbers_are_the_artifacts_own(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The viewer's ranking is the artifact's: the same label and the same value."""
    row = 0
    artifact = json.loads(
        (fixture_artifacts["results"] / fixture_artifacts["text_run"] / "hitmap.json").read_text(
            "utf-8"
        )
    )
    stored = artifact["text_side_map"]["per_text_top_label"][row]
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
    )
    ranking = viewer_module.lexical_ranking(context.records[row].text, context.labels)
    assert ranking[0][0] == stored["label"]
    assert ranking[0][1] == pytest.approx(float(stored["value"]), abs=1e-6)
    # and the cross-check finds nothing to complain about
    assert viewer_module.cross_check(context, row, context.records[row].text) == []
    # the screen shows the artifact's own value
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][row]))
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert f"{float(stored['value']):.4f}" in out


def test_a_stale_artifact_fails_loudly(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """An artifact that disagrees with the data is reported, not smoothed over."""
    path = fixture_artifacts["results"] / fixture_artifacts["text_run"] / "hitmap.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["text_side_map"]["per_text_top_label"][0]["value"] = 0.999
    payload["text_side_map"]["per_text_top_label"][0]["label"] = "Not a label of the split"
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]))
    assert viewer_module.main(args) == 1
    out = capsys.readouterr().out
    assert "THE ARTIFACTS AND THE DATA DISAGREE" in out


# --- the live and what-if modes ---


def test_the_live_mode_asks_the_model(
    viewer_module: Any,
    fixture_artifacts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The live mode shows the model's own ranking (against the fake decider)."""
    decider = FakeDecider()
    monkeypatch.setattr(viewer_module, "build_upr_decider", lambda model_config, config: decider)
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]), "--live")
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert "WHAT THE MODEL SAYS" in out
    assert decider.calls, "the model was not asked"
    assert "the tool's own reading, not an artifact's number" in out


def test_the_what_if_mode_removes_the_word_and_re_asks(
    viewer_module: Any,
    fixture_artifacts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The what-if mode perturbs the input: the model is asked about the ablated text."""
    decider = FakeDecider()
    monkeypatch.setattr(viewer_module, "build_upr_decider", lambda model_config, config: decider)
    record_id = str(fixture_artifacts["val_ids"][0])
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
    )
    text = context.records[0].text
    word = text.split()[0]
    args = _view_args(fixture_artifacts, "--text", record_id, "--what-if", word)
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert "the causal one" in out
    assert "NOT deterministic" in out
    # the model saw the intact text and the text without the word
    assert any(word.lower() in call for call in decider.calls)
    assert any(word.lower() not in call for call in decider.calls)


def test_the_output_separates_the_attributions_from_the_causal_column(
    viewer_module: Any,
    fixture_artifacts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The attribution/causation distinction is visible in the output itself."""
    decider = FakeDecider()
    monkeypatch.setattr(viewer_module, "build_upr_decider", lambda model_config, config: decider)
    record_id = str(fixture_artifacts["val_ids"][0])
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
        corpus_map_run=str(fixture_artifacts["corpus_run"]),
    )
    word = context.records[0].text.split()[0]
    args = _view_args(fixture_artifacts, "--text", record_id, "--live", "--what-if", word)
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    # the attributions say what they are ...
    assert "an attribution: the feature's own arithmetic" in out
    assert "an attribution of the model's own probability" in out
    assert "a description of the data" in out
    # ... and the causal one says so
    assert "WHAT IF THIS WORD IS REMOVED — the causal one, at word level" in out


# --- the corpus map (CS-23) on the screen ---


def test_the_viewer_shows_the_two_corpus_blocks(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The corpus map's two blocks are on the screen: the topic's words and the
    text's words."""
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]))
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert "THE CORPUS'S OWN WORDS FOR THIS TOPIC" in out
    assert "THIS TEXT'S WORDS AND THE LABELS THEY LEAN TO" in out
    # the counts sit next to the numbers (the noise is visible)
    assert "n_with" in out
    assert "log-odds" in out


def test_the_viewers_corpus_numbers_are_the_artifacts_own(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The viewer's corpus rows are the artifact's, and the guard finds nothing."""
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
        corpus_map_run=str(fixture_artifacts["corpus_run"]),
    )
    focus, _rule = viewer_module._default_focus(context.records[0], context.labels)
    words = list(dict.fromkeys(viewer_module.content_tokens(context.records[0].text)))
    assert viewer_module.corpus_cross_check(context, focus, words) == []
    # the screen shows the artifact's own counts for the topic's first word
    stored = viewer_module.corpus_topic_rows(context.corpus_map, focus)
    assert stored
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]))
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert str(stored[0]["word"]) in out
    assert f"{int(stored[0]['n_with'])}" in out


def test_a_stale_corpus_artifact_fails_loudly(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """A corpus map that disagrees with the data is reported, not smoothed over."""
    path = fixture_artifacts["results"] / fixture_artifacts["corpus_run"] / "hitmap.json"
    payload = json.loads(path.read_text("utf-8"))
    payload["corpus_map"]["per_label"][0]["words"][0]["n_with"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]))
    assert viewer_module.main(args) == 1
    out = capsys.readouterr().out
    assert "THE ARTIFACTS AND THE DATA DISAGREE" in out


def test_the_min_df_filter_only_changes_the_display(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The filter is a display-time filter: the artifact keeps every pair."""
    topic = "Children"  # a label with train examples in the fixture
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
        corpus_map_run=str(fixture_artifacts["corpus_run"]),
    )
    stored = viewer_module.corpus_topic_rows(context.corpus_map, topic)
    assert stored
    assert viewer_module.corpus_topic_rows(context.corpus_map, topic, min_df=1000) == []
    args = _view_args(
        fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]), "--topic", topic
    )
    assert viewer_module.main(args) == 0
    unfiltered = capsys.readouterr().out
    assert str(stored[0]["word"]) in unfiltered
    args = _view_args(
        fixture_artifacts,
        "--text",
        str(fixture_artifacts["val_ids"][0]),
        "--topic",
        topic,
        "--min-df",
        "1000",
    )
    assert viewer_module.main(args) == 0
    filtered = capsys.readouterr().out
    assert "filter hides every row" in filtered
    # the artifact itself is untouched: every pair that occurs is still stored
    payload = json.loads(
        (fixture_artifacts["results"] / fixture_artifacts["corpus_run"] / "hitmap.json").read_text(
            "utf-8"
        )
    )
    assert payload["corpus_map"]["corpus"]["n_pairs"] > 0


def test_the_viewer_runs_without_the_corpus_map(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The corpus map is optional: the screen says so and the tool still works."""
    args = [
        "--config",
        str(fixture_artifacts["config"]),
        "--text-map-run",
        str(fixture_artifacts["text_run"]),
        "--surrogate-run",
        str(fixture_artifacts["surrogate_run"]),
        "--corpus-map-run",
        "upr-hitmap-a-run-that-does-not-exist",
        "--no-colour",
        "--text",
        str(fixture_artifacts["val_ids"][0]),
    ]
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert "has not been made" in out
    assert "THE CORPUS'S OWN WORDS FOR THIS TOPIC" not in out


# --- the surrogate's full coefficient vector (CS-24) ---


def test_the_surrogate_column_has_a_number_for_every_word(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """With the full vector, every word of the model's vocabulary has a number."""
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
        corpus_map_run=str(fixture_artifacts["corpus_run"]),
    )
    _arm, surrogate = viewer_module.surrogate_block(context.surrogate, None)
    vector = viewer_module.surrogate_full_vector(surrogate)
    assert vector is not None, "the artifact carries no full vector"
    topic = context.records[0].labels[0]
    coefficients = viewer_module.surrogate_coefficients(surrogate, topic)
    assert coefficients, "the topic has no coefficients"
    # the vector's vocabulary is the model's vocabulary
    assert set(vector["vocabulary"]) == set(coefficients)
    text_words = {
        str(row["word"]) for row in viewer_module.word_rows(context.records[0].text, topic)
    }
    in_vocabulary = text_words & set(vector["vocabulary"])
    assert in_vocabulary, "none of the text's words is in the model's vocabulary"
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]))
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    # the old fallback note is gone: the column is complete
    assert "—' means 'not stored'" not in out
    assert "the model's own for every word of its vocabulary" in out
    # a word the vocabulary holds shows its number
    sample = sorted(in_vocabulary)[0]
    assert f"{coefficients[sample]:+.3f}" in out
    # a word it does not hold says so (the fixture's val-only words)
    assert "not in the model's vocabulary" in out


def test_a_word_outside_the_vocabulary_says_so(
    viewer_module: Any,
    fixture_artifacts: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A word the surrogate's vocabulary does not hold reads "not in the model's vocabulary"."""
    source = fixture_artifacts["results"] / fixture_artifacts["surrogate_run"] / "hitmap.json"
    payload = json.loads(source.read_text("utf-8"))
    arm = next(iter(payload["ablation_map"]["arms"].values()))
    vector = arm["surrogate"]["full_vector"]
    # a narrower vocabulary: only the first three features
    vector["vocabulary"] = vector["vocabulary"][:3]
    vector["coefficients"] = [row[:3] for row in vector["coefficients"]]
    narrow_run = "upr-hitmap-fixture-surrogate-narrow"
    target = fixture_artifacts["results"] / narrow_run
    target.mkdir(parents=True, exist_ok=True)
    (target / "hitmap.json").write_text(json.dumps(payload), encoding="utf-8")
    args = [
        "--config",
        str(fixture_artifacts["config"]),
        "--text-map-run",
        str(fixture_artifacts["text_run"]),
        "--surrogate-run",
        narrow_run,
        "--corpus-map-run",
        str(fixture_artifacts["corpus_run"]),
        "--no-colour",
        "--text",
        str(fixture_artifacts["val_ids"][0]),
    ]
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert "not in the model's vocabulary" in out
    assert "never zero" in out


# --- the clearer screen and the colour (CS-25) ---


def test_the_colour_carries_nothing_the_text_does_not(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The rule: the coloured screen, stripped of its codes, is the plain screen.

    Every colour is a reading of a number that is on the screen anyway, so
    ``--no-colour`` loses no information.
    """
    record_id = str(fixture_artifacts["val_ids"][0])
    assert viewer_module.main(_view_args(fixture_artifacts, "--text", record_id)) == 0
    plain = capsys.readouterr().out
    assert viewer_module.main(_coloured_args(fixture_artifacts, "--text", record_id)) == 0
    coloured = capsys.readouterr().out
    assert "\033[" in coloured, "the coloured screen carries no escape codes"
    stripped = re.sub(r"\033\[[0-9;]*m", "", coloured)
    assert stripped == plain


def test_the_colour_flag_forces_the_colour_when_piping(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """``--colour`` turns the colours on even when the output is piped."""
    record_id = str(fixture_artifacts["val_ids"][0])
    assert viewer_module.main(_coloured_args(fixture_artifacts, "--text", record_id)) == 0
    assert "\033[" in capsys.readouterr().out


def test_the_palette_lands_on_the_right_cells(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The coloured output is checked: the codes are on the cells they belong to."""
    record_id = str(fixture_artifacts["val_ids"][0])
    assert viewer_module.main(_coloured_args(fixture_artifacts, "--text", record_id)) == 0
    out = capsys.readouterr().out
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
        corpus_map_run=str(fixture_artifacts["corpus_run"]),
    )
    focus, _rule = viewer_module._default_focus(context.records[0], context.labels)
    own_words = viewer_module.topic_words(context.records[0].text, focus)
    cyan = "\033[38;5;51m"
    # the header's dataset and split are in cyan
    assert f"{cyan}UPR val" in out
    # a gold label is in bold green
    assert viewer_module.GOLD in out
    # the topic's own name words are in cyan
    assert own_words
    assert f"{cyan}{own_words[0]}" in out
    # the surrogate's positive and negative coefficients are on the two ramps
    assert any(code in out for code in viewer_module.GREEN_RAMP)
    assert any(code in out for code in viewer_module.RED_RAMP)
    # the notes are dim
    assert viewer_module.DIM in out


def test_the_mode_letters_leave_the_headings(
    viewer_module: Any, fixture_artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The headings name the question; the mode letters stay in the plan and the log."""
    args = _view_args(fixture_artifacts, "--text", str(fixture_artifacts["val_ids"][0]))
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    for letter in ("mode A", "mode B", "mode C", "mode D", "mode E"):
        assert letter not in out
    # the blocks name the question they answer
    assert "THE FEATURE'S OWN SPLIT — a spelling question:" in out
    assert "THE TOPIC'S OWN WORDS — a spelling question:" in out
    assert "THE CORPUS'S OWN WORDS FOR THIS TOPIC — a statistical question:" in out
    assert "THE MODEL'S WORD HABITS — a learned question:" in out


def test_the_noise_band_marks_the_deltas(
    viewer_module: Any,
    fixture_artifacts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A delta inside the model's noise band is dim and marked "noise"."""
    decider = FakeDecider()
    monkeypatch.setattr(viewer_module, "build_upr_decider", lambda model_config, config: decider)
    config = viewer_module.load_config(fixture_artifacts["config"])
    context = viewer_module.load_context(
        config,
        split="val",
        text_map_run=str(fixture_artifacts["text_run"]),
        surrogate_run=str(fixture_artifacts["surrogate_run"]),
        results=fixture_artifacts["results"],
        corpus_map_run=str(fixture_artifacts["corpus_run"]),
    )
    word = context.records[0].text.split()[0]
    args = _coloured_args(
        fixture_artifacts,
        "--text",
        str(fixture_artifacts["val_ids"][0]),
        "--live",
        "--what-if",
        word,
    )
    assert viewer_module.main(args) == 0
    out = capsys.readouterr().out
    assert "noise" in out
    assert f"±{viewer_module.NOISE_BAND:g}" in out
