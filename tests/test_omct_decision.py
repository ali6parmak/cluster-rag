"""Tests for the decision probe's CLI (``decide-omct``).

Contracts under test:

- the run writes the artifact (``decisions.json``) beside the frozen config,
  with the measurement blocks, the synthetic competence control and the
  provenance (the model id, the pinned digest, the prompt version, the cache's
  counters);
- the command adds the ``omct-`` dataset prefix to the run id;
- the row lands in the text-processing track's own table;
- the command refuses to run without an ``analysis.decision`` section.
"""

from __future__ import annotations

import json

import pytest

from cluster_rag import cli
from cluster_rag.cli import main
from cluster_rag.decision.base import DecisionAnswer, DecisionInfo
from cluster_rag.text_processors.decision.probe import GROUP_COMPETENCE, GROUP_CONTROL, GROUP_JUDGED
from fake_embedder import FakeEmbedder

DIGEST = "c" * 64

DECISION_CONFIG_EXTRA = f"""
analysis:
  control_size: 2
  sampling_rule: proportional
  question_transform: strip_block
  decision:
    models:
      - name: fake-decider
        model_id: fake:latest
        digest: {DIGEST}
        state: pair
        question_type: noul
        instructions: "Is the paragraph about the same topic as the question?"
    scope: judged
    competence_control: true
    competence_pairs: 2
    prompt_version: v1
"""


class FakeDecider:
    """A decider that answers with a deterministic probability (no server)."""

    def __init__(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_stored = 0
        self._info = DecisionInfo(model_id="fake:latest", digest=DIGEST, backend="fake")

    @property
    def info(self) -> DecisionInfo:
        """The fake model's provenance."""
        return self._info

    @property
    def usage(self) -> dict[str, int]:
        """The fake model's call counters."""
        return {"n_calls": 1, "input_tokens": 10, "output_tokens": 0}

    def decide(self, state, questions):
        """Answer every question with a probability that depends on the state."""
        answers = {}
        for name in questions:
            text = str(state.get("question", "")) if isinstance(state, dict) else str(state)
            value = 0.9 if "defen" in text or "evict" in text else 0.2
            answers[name] = DecisionAnswer(name=name, type="noul", value=value)
        return answers


@pytest.fixture()
def fake_decider(monkeypatch) -> FakeDecider:
    """Wire the CLI to a fake decider (no server, no model download)."""
    decider = FakeDecider()
    monkeypatch.setattr(cli, "build_decider", lambda model_config, config: decider)
    return decider


def test_decide_omct_writes_the_artifact(
    sample_pairs, write_config, fake_decider, tmp_path
) -> None:
    """A probe run: the artifact carries the measurement, the control and the provenance."""
    config_file = write_config(
        sample_pairs,
        run_id="decide-fake",
        data_external=sample_pairs,
        extra=DECISION_CONFIG_EXTRA,
    )
    assert main(["decide-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-decide-fake"
    payload = json.loads((results_dir / "decisions.json").read_text("utf-8"))
    assert payload["command"] == "decide-omct"
    assert payload["run_id"] == "omct-decide-fake"
    assert (results_dir / "config.yaml").is_file()

    model = payload["models"][0]
    assert model["model"]["model_id"] == "fake:latest"
    assert model["model"]["digest"] == DIGEST
    assert model["model"]["prompt_version"] == "v1"
    assert model["state"] == "pair"
    assert model["question_type"] == "noul"

    measurement = model["measurement"]
    assert measurement["pair_auc"]["overall"] is not None
    assert measurement["pair_auc_decomposition"]["within_group_auc"] is not None
    assert measurement["calibration"]["pooled"]["n"] == 6
    assert measurement["floor_gap"]["gap"] is not None

    groups = {cell["group"] for cell in model["cells"]}
    assert GROUP_JUDGED in groups
    assert GROUP_CONTROL in groups

    competence = model["competence_control"]
    assert competence["n_cells"] == 4
    assert competence["pair_auc"] is not None
    assert {cell["label"] for cell in competence["cells"]} == {0, 1}
    # the competence cells live in their own block, never in the arm's cells
    assert GROUP_COMPETENCE not in groups


ANALYZE_DECISION_CONFIG_EXTRA = f"""
analysis:
  embedders:
    - name: fake-model
      embed:
        backend: fake
        model_id: fake-embedder
        batch_size: 4
        prefixes:
          query: "query: "
          document: "passage: "
  control_size: 3
  sampling_rule: proportional
  question_transform: strip_block
  chunker: full_text
  bootstrap:
    resamples: 20
    seed: 7
  decision:
    models:
      - name: fake-decider
        model_id: fake:latest
        digest: {DIGEST}
        state: paragraph
        question_type: noul
        instructions: "Is the paragraph about the same topic as the question?"
        max_input_tokens: 512
    scope: grid
    competence_control: false
    mismatched_control: true
    prompt_version: v1
"""


def test_analyze_omct_adds_the_decision_arm(
    sample_pairs, write_config, fake_decider, tmp_path, monkeypatch
) -> None:
    """The decision model becomes an arm of the analysis (CS-7b)."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-decision",
        data_external=sample_pairs,
        extra=ANALYZE_DECISION_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-decision"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    decision = [model for model in payload["models"] if model.get("kind") == "decision"]
    assert len(decision) == 1
    entry = decision[0]
    assert entry["model"]["model_id"] == "fake:latest"
    assert entry["model"]["digest"] == DIGEST
    arm = entry["arms"]["decision"]
    assert arm["pair_auc"]["overall"] is not None
    assert arm["hit_at_k"]["label1"]["hit@1"] is not None
    # the comparison against the full-text cosine (the same run, the same cells)
    comparison = entry["comparisons"]["decision"]
    assert comparison["paired_auc"]["delta_auc"] is not None
    assert comparison["deshifted_auc"]["overall"] is not None
    # the shortlist rerank (CS-7c): the K grid and the two references
    rerank = entry["shortlist_rerank"]
    assert "1" in rerank["by_k"]
    assert rerank["references"]["cosine_hit1"] is not None
    assert rerank["references"]["decision_hit1"] is not None
    assert rerank["by_k"]["1"]["positives_hit1"] is not None
    # a rerank of a one-candidate shortlist cannot move anything
    assert rerank["by_k"]["1"]["positives_hit1"] == pytest.approx(
        rerank["references"]["cosine_hit1"], abs=1e-9
    )
    # the rule: the three variants and the label-blind counterpart
    rule = entry["rule"]
    assert set(rule["pooled"]) == {
        "decision_fixed_cut",
        "decision_fit_threshold",
        "decision_margin",
    }
    fixed = rule["pooled"]["decision_fixed_cut"]
    assert fixed["coverage"] is not None
    assert fixed["harm"] is not None
    # the label-blind counterpart ran (its harm is None only when a document
    # carries no verified positive to discard)
    assert "mean" in fixed["counterpart"]
    per_document = rule["per_document"]
    assert per_document
    for block in per_document.values():
        counterpart = block["variants"]["decision_fixed_cut"]["counterpart"]
        assert counterpart["n_draws"] == 1


def test_the_decision_arm_gets_its_own_table_row(
    sample_pairs, write_config, fake_decider, tmp_path, monkeypatch
) -> None:
    """The decision arm's row lands in the track's table (the arms shape)."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "track_table.md"
    config_file = write_config(
        sample_pairs,
        run_id="analyze-decision",
        data_external=sample_pairs,
        extra=ANALYZE_DECISION_CONFIG_EXTRA,
        experiment_table_text_processing_omct=table,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    text = table.read_text("utf-8")
    assert "## OMCT — analyze-omct" in text
    assert "| decision |" in text
    assert "fake:latest" in text


def test_decide_omct_appends_the_track_table_row(
    sample_pairs, write_config, fake_decider, tmp_path
) -> None:
    """The row lands in the track's own table file."""
    table = tmp_path / "track_table.md"
    config_file = write_config(
        sample_pairs,
        run_id="decide-fake",
        data_external=sample_pairs,
        extra=DECISION_CONFIG_EXTRA,
        experiment_table_text_processing_omct=table,
    )
    assert main(["decide-omct", "--config", str(config_file)]) == 0
    text = table.read_text("utf-8")
    assert "## OMCT — decide-omct" in text
    assert "omct-decide-fake" in text
    assert "within_auc" in text


def test_decide_omct_requires_the_decision_section(
    sample_pairs, write_config, fake_decider, caplog
) -> None:
    """Without an ``analysis.decision`` section the command refuses to run."""
    config_file = write_config(
        sample_pairs,
        run_id="decide-missing",
        data_external=sample_pairs,
        extra="analysis:\n  control_size: 2\n",
    )
    assert main(["decide-omct", "--config", str(config_file)]) == 1


def test_the_decision_config_validates_the_prompt_and_the_digest() -> None:
    """A bad digest, a task that repeats the question or an unknown option fails at config time."""
    from cluster_rag.config import DecisionConfig, DecisionModelConfig

    with pytest.raises(ValueError):
        DecisionModelConfig(name="x", model_id="m", digest="short", instructions="Same topic?")
    with pytest.raises(ValueError):
        DecisionModelConfig(
            name="x", model_id="m", digest="a" * 64, instructions="Question: {question}"
        )
    with pytest.raises(ValueError):
        DecisionConfig(
            models=(
                DecisionModelConfig(
                    name="x",
                    model_id="m",
                    digest="a" * 64,
                    question_type="choice",
                    instructions="Same topic?",
                    options=("yes", "no"),
                    criteria={"yes": "a"},
                ),
            )
        )
    with pytest.raises(ValueError):
        DecisionConfig(models=())
