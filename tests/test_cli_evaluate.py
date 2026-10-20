"""Tests for the `evaluate` CLI command (Phase 3), on the tiny fixture dataset.

Covers: split -> evaluate end-to-end (artifacts, bucket sections, table row),
seeded determinism on rerun, the --split flag, and the config-error contract.
Baselines are pure, so no fake embedder or network is involved.
"""

from __future__ import annotations

import json

import pytest
import yaml

from cluster_rag.cli import main

EVAL_CONFIG_EXTRA = """
eval:
  predictor:
    name: most_frequent
  decision:
    rule: top_k
    k: 2
"""


@pytest.fixture()
def evaluate_run(sample_jsonl, write_config, tmp_path):
    """Factory: run `split` then `evaluate` on the fixture; return the artifacts."""

    def run(args=None, run_id="evaluate-test", extra=EVAL_CONFIG_EXTRA, table=None):
        config_file = write_config(
            sample_jsonl, run_id=run_id, extra=extra, experiment_table_upr=table
        )
        assert main(["split", "--config", str(config_file)]) == 0
        argv = ["evaluate", "--config", str(config_file)]
        if args:
            argv.extend(args)
        code = main(argv)
        results_dir = tmp_path / "results" / run_id
        metrics_path = results_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text("utf-8")) if metrics_path.is_file() else None
        return code, metrics, config_file, results_dir

    return run


def test_evaluate_command_artifacts(evaluate_run, tmp_path) -> None:
    """Split then evaluate: metrics.json with per-bucket sections + frozen config."""
    code, metrics, _, results_dir = evaluate_run()
    assert code == 0
    val_ids = json.loads((tmp_path / "splits" / "val_ids.json").read_text("utf-8"))["ids"]
    assert metrics["command"] == "evaluate"
    assert metrics["split"] == "val" and metrics["n_records"] == len(val_ids)
    assert metrics["predictor"]["name"] == "most_frequent"
    assert metrics["decision"] == {
        "rule": "top_k",
        "k": 2,
        "threshold": None,
        "min_score": None,
    }
    assert metrics["metrics_config"]["ks"] == [1, 3, 5, 10]
    assert metrics["buckets"]["thresholds"] == {"head_min": 100, "torso_min": 40}
    # Bucket sizes cover every fixture label; the 100/40 thresholds leave only
    # tail/unseen non-empty (head/torso sections are omitted, sizes stay 0).
    sizes = metrics["buckets"]["n_labels_per_bucket"]
    assert set(sizes) == {"head", "torso", "tail", "unseen"}
    assert sizes["head"] == 0 and sizes["torso"] == 0
    assert sum(sizes.values()) == metrics["n_labels"]
    assert (
        sum(len(subset) for subset in metrics["buckets"]["labels_by_bucket"].values())
        == metrics["n_labels"]
    )

    overall = metrics["report"]["overall"]
    assert overall["n_records"] == len(val_ids)
    # Buckets partition the label space, so their gold-present counts sum to the overall.
    assert overall["n_labels_with_gold"] == sum(
        section["n_labels_with_gold"] for section in metrics["report"]["per_bucket"].values()
    )
    for block, keys in (
        ("ranking", ("precision@1", "recall@3", "ndcg@10")),
        ("f1", ("micro_f1", "macro_f1", "sample_f1", "micro_precision", "micro_recall")),
        ("coverage", ("coverage", "n_labels_suggested", "n_labels_covered")),
        ("shortlist", ("recall@20", "recall@50", "recall@100")),
    ):
        assert set(overall[block]) >= set(keys)
    assert set(metrics["report"]["per_bucket"]) == {"tail", "unseen"}
    unseen = metrics["report"]["per_bucket"]["unseen"]
    assert unseen["bucket"] == "unseen" and unseen["n_labels"] == sizes["unseen"]
    # most_frequent scores unseen labels 0 -> it never covers them at top-k=2.
    assert unseen["coverage"]["coverage"] == 0.0

    frozen = yaml.safe_load((results_dir / "config.yaml").read_text("utf-8"))
    assert frozen["eval"]["predictor"]["name"] == "most_frequent"


def test_evaluate_command_appends_table_row(evaluate_run, tmp_path) -> None:
    """The auto table lands under '## UPR — evaluate', upserted by run_id."""
    table = tmp_path / "tables.md"
    code, metrics, _, _ = evaluate_run(run_id="evaluate-table", table=table)
    assert code == 0
    content = table.read_text("utf-8")
    assert "## UPR — evaluate" in content
    assert "| evaluate-table |" in content
    assert content.count("| evaluate-table |") == 1
    decision_cell = f"top_k k={metrics['decision']['k']}"  # rendered from the run artifacts
    assert decision_cell in content
    # regrouped columns: the f1 decision block before the ranking block
    header = next(line for line in content.splitlines() if line.startswith("| run_id"))
    assert header.index("f1 decision") < header.index("micro-F1") < header.index("P@5")
    assert "nDCG@5" in header and "shortlist R@50" in header and "coverage" in header


def test_evaluate_command_deterministic_rerun(evaluate_run) -> None:
    """A second run of the same predictor/seed reproduces the same metrics..."""
    random_extra = """
eval:
  predictor:
    name: random
  decision:
    rule: top_k
    k: 2
"""
    _, first, config_file, _ = evaluate_run(run_id="evaluate-random", extra=random_extra)
    assert main(["evaluate", "--config", str(config_file)]) == 0
    results_dir = config_file.parent / "results" / "evaluate-random"
    second = json.loads((results_dir / "metrics.json").read_text("utf-8"))
    # RandomPredictor is seeded from config.seed, so scores (and every metric)
    # are identical across reruns; only wall-clock timing may differ.
    del first["total_seconds"], second["total_seconds"]
    assert first == second


def test_evaluate_split_flag_uses_test_ids(evaluate_run, tmp_path) -> None:
    """--split test evaluates the fixed test ids, not the config's val default."""
    code, metrics, _, _ = evaluate_run(args=["--split", "test"])
    assert code == 0
    test_ids = json.loads((tmp_path / "splits" / "test_ids.json").read_text("utf-8"))["ids"]
    assert metrics["split"] == "test" and metrics["n_records"] == len(test_ids)


def test_evaluate_requires_eval_section_and_run_id(sample_jsonl, write_config, tmp_path) -> None:
    """Missing eval: section or missing run_id exit with code 1, no artifacts."""

    config_file = write_config(sample_jsonl, run_id="evaluate-nosec", extra="")
    assert main(["split", "--config", str(config_file)]) == 0
    assert main(["evaluate", "--config", str(config_file)]) == 1
    assert not (tmp_path / "results" / "evaluate-nosec" / "metrics.json").exists()

    no_run_id = write_config(sample_jsonl, run_id=None, extra=EVAL_CONFIG_EXTRA)
    assert main(["split", "--config", str(no_run_id)]) == 1
    assert main(["evaluate", "--config", str(no_run_id)]) == 1


def test_evaluate_unknown_predictor_name_fails(evaluate_run) -> None:
    """An unknown predictor name exits with code 1 (registry check)."""
    bad_extra = """
eval:
  predictor:
    name: knn
  decision:
    rule: top_k
    k: 2
"""
    code, metrics, _, results_dir = evaluate_run(run_id="evaluate-unknown", extra=bad_extra)
    assert code == 1
    assert metrics is None
    assert not (results_dir / "metrics.json").exists()
