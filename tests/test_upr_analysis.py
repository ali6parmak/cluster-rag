"""Tests for the text-processing track's UPR side (CS-8a).

Covers: the UPR diagnostics (hand-computed small cases — the rank percentiles,
the hit@k, the provenance, the level/rank shift, the length buckets), the
driver's view and arm block, and the ``analyze-upr`` CLI command end-to-end on
the tiny fixture dataset (the artifact, the guard against the matching
``evaluate`` run, the table row and the config-error contract).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import yaml

from cluster_rag.cli import main
from cluster_rag.text_processors.upr_diagnostics import (
    cell_rank_percentiles,
    distribution,
    gold_hit_at_k,
    length_bucket_names,
    length_buckets,
    provenance,
    top_fifth_share,
)
from cluster_rag.upr_analysis import BASELINE_ARM, arm_block, build_upr_view

EVAL_EXTRA = """
eval:
  predictor:
    name: most_frequent
  decision:
    rule: top_k
    k: 2
"""

ANALYZE_EXTRA = """
eval:
  predictor:
    name: most_frequent
  decision:
    rule: top_k
    k: 2
upr_analysis:
  length_buckets: [40, 80]
  reference_eval_run: upr-eval-ref
"""


# --- the pure diagnostics (hand-computed cases) ---


def test_cell_rank_percentiles_are_positions_over_the_label_count() -> None:
    """0.0 is the best label of the text, (n-1)/n the worst."""
    scores = np.asarray([[0.1, 0.9, 0.5], [0.7, 0.2, 0.4]])
    percentiles = cell_rank_percentiles(scores)
    # row 0: 0.9 (col 1) is the best -> 0.0; 0.1 (col 0) is the worst -> 2/3
    assert percentiles[0, 1] == pytest.approx(0.0)
    assert percentiles[0, 2] == pytest.approx(1 / 3)
    assert percentiles[0, 0] == pytest.approx(2 / 3)
    # row 1: 0.7 (col 0) is the best
    assert percentiles[1, 0] == pytest.approx(0.0)
    assert percentiles[1, 2] == pytest.approx(1 / 3)
    assert percentiles[1, 1] == pytest.approx(2 / 3)


def test_cell_rank_percentiles_keep_ties_in_column_order() -> None:
    """Ties keep the labels' own order, so the number is reproducible."""
    scores = np.asarray([[0.5, 0.5, 0.5]])
    percentiles = cell_rank_percentiles(scores)
    assert list(percentiles[0]) == pytest.approx([0.0, 1 / 3, 2 / 3])


def test_cell_rank_percentiles_reject_non_finite_scores() -> None:
    """A NaN would corrupt every ranking: the matrix is refused."""
    with pytest.raises(ValueError, match="finite"):
        cell_rank_percentiles(np.asarray([[0.1, float("nan")]]))


def test_gold_hit_at_k_counts_the_gold_cells_in_the_top_k() -> None:
    """hit@k: is the gold label among the text's top-k labels?"""
    scores = np.asarray([[0.9, 0.8, 0.1, 0.0]])
    gold = np.asarray([[True, False, False, False]])
    result = gold_hit_at_k(scores, gold, (1, 2))
    assert result["n_gold_cells"] == 1
    assert result["hit@1"] == pytest.approx(1.0)
    assert result["hit@2"] == pytest.approx(1.0)
    # a gold label at rank 3 (index 2) is not in the top-2
    gold_low = np.asarray([[False, False, True, False]])
    assert gold_hit_at_k(scores, gold_low, (1, 2))["hit@2"] == pytest.approx(0.0)
    assert gold_hit_at_k(scores, gold_low, (3,))["hit@3"] == pytest.approx(1.0)


def test_gold_hit_at_k_reports_an_empty_gold_matrix() -> None:
    """No gold cells: the hit rates are zero, not an error."""
    result = gold_hit_at_k(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool), (1,))
    assert result["n_gold_cells"] == 0 and result["hit@1"] == 0.0


def test_provenance_counts_gold_and_non_gold_in_the_top_k() -> None:
    """The top-1 provenance: how many texts' best label is a gold label."""
    scores = np.asarray([[0.9, 0.8, 0.1], [0.9, 0.8, 0.1]])
    gold = np.asarray([[True, False, False], [False, True, False]])
    result = provenance(scores, gold, (1, 2))
    assert result["top1_gold"] == 1 and result["top1_not_gold"] == 1
    assert result["top1_gold_share"] == pytest.approx(0.5)
    assert result["mean_gold_in_top1"] == pytest.approx(0.5)
    assert result["mean_nongold_in_top1"] == pytest.approx(0.5)
    # both texts' second-best label is a non-gold one
    assert result["mean_gold_in_top2"] == pytest.approx(1.0)
    assert result["mean_nongold_in_top2"] == pytest.approx(1.0)


def test_top_fifth_share_counts_the_top_fifth() -> None:
    """The share of a group inside the top fifth of its list."""
    assert top_fifth_share(np.asarray([0.0, 0.1, 0.19, 0.2, 0.9])) == pytest.approx(3 / 5)
    assert top_fifth_share(np.asarray([])) is None


def test_level_and_rank_shift_reports_the_background_against_a_reference() -> None:
    """The background's level and rank shift beside every headline number."""
    from cluster_rag.text_processors.upr_diagnostics import level_and_rank_shift

    gold = np.asarray([[True, False, False, False]])
    reference = np.asarray([[0.9, 0.1, 0.2, 0.3]])
    arm = np.asarray([[0.9, 0.5, 0.6, 0.7]])  # the background is lifted
    shift = level_and_rank_shift(arm, gold, reference)
    assert shift["n_background_cells"] == 3 and shift["n_gold_cells"] == 1
    assert shift["background_mean"] == pytest.approx(0.6)
    assert shift["background_mean_delta"] == pytest.approx(0.6 - 0.2)
    # the gold cell stays the best in both arms: no rank shift
    assert shift["gold_rank_p50_delta"] == pytest.approx(0.0)
    # the background's rank positions move: the last cell falls to the bottom
    assert shift["background_rank_p50_delta"] is not None


def test_level_and_rank_shift_without_a_reference_has_no_deltas() -> None:
    """No reference arm: the levels are reported, the deltas are null."""
    from cluster_rag.text_processors.upr_diagnostics import level_and_rank_shift

    shift = level_and_rank_shift(np.asarray([[0.9, 0.1]]), np.asarray([[True, False]]))
    assert shift["background_mean_delta"] is None
    assert shift["background_rank_p50_delta"] is None
    assert shift["gold_rank_p50_delta"] is None


def test_distribution_summarizes_a_score_group() -> None:
    """The distribution block: the count, the mean and the quantiles."""
    summary = distribution(np.asarray([0.0, 0.5, 1.0]))
    assert summary["n"] == 3 and summary["mean"] == pytest.approx(0.5)
    assert summary["p50"] == pytest.approx(0.5)
    assert distribution(np.asarray([]))["n"] == 0


def test_length_buckets_split_the_texts_at_the_edges() -> None:
    """The length buckets: the upper edges in characters, the last one open."""
    assert length_bucket_names((100, 200, 300)) == ("0-99", "100-199", "200-299", "300+")
    assert length_buckets([50, 150, 250, 350], (100, 200, 300)) == [0, 1, 2, 3]
    assert length_buckets([50], ()) == [-1]


# --- the paired bootstrap of a metric difference ---


def test_paired_metric_bootstrap_reports_a_null_for_identical_scorers() -> None:
    """The same scorer against itself: exactly zero, every time."""
    from cluster_rag.eval.multilabel import paired_metric_bootstrap

    scores = np.asarray([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3], [0.5, 0.4, 0.6]])
    gold = np.asarray([[True, False, False], [False, True, False], [False, False, True]])
    result = paired_metric_bootstrap(
        scores,
        scores,
        gold,
        metric="micro_f1",
        decision={"rule": "top_k", "k": 1, "threshold": None, "min_score": None},
        resamples=50,
        seed=1,
    )
    assert result["delta"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["share_positive"] == 0.0
    assert result["n_texts"] == 3


def test_paired_metric_bootstrap_separates_a_better_scorer() -> None:
    """A scorer that is right on every text: the interval excludes zero."""
    from cluster_rag.eval.multilabel import paired_metric_bootstrap

    n_texts = 8
    gold = np.zeros((n_texts, 3), dtype=bool)
    gold[:, 0] = True
    worse = np.zeros((n_texts, 3))
    worse[:, 1] = 1.0  # a confident wrong label
    better = np.zeros((n_texts, 3))
    better[:, 0] = 1.0  # the gold label
    result = paired_metric_bootstrap(
        better,
        worse,
        gold,
        metric="micro_f1",
        decision={"rule": "top_k", "k": 1, "threshold": None, "min_score": None},
        resamples=200,
        seed=3,
    )
    assert result["delta"] == pytest.approx(1.0)
    assert result["ci_low"] > 0
    assert result["share_positive"] == 1.0


def test_paired_metric_bootstrap_rejects_a_missing_decision_rule() -> None:
    """The F1 metrics need a decision rule (an implicit one would be a guess)."""
    from cluster_rag.eval.multilabel import paired_metric_bootstrap

    scores = np.zeros((2, 2))
    gold = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="needs a decision rule"):
        paired_metric_bootstrap(scores, scores, gold, metric="micro_f1", resamples=10)


def test_paired_metric_bootstrap_rejects_an_unknown_metric() -> None:
    """A typo in the metric name fails loudly."""
    from cluster_rag.eval.multilabel import paired_metric_bootstrap

    with pytest.raises(ValueError, match="Unknown paired metric"):
        paired_metric_bootstrap(
            np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2), dtype=bool), metric="auc"
        )


# --- the chunked arms ---


def test_fold_label_chunks_keeps_the_best_chunk_of_every_label() -> None:
    """The symmetric family's label fold: a label keeps its best chunk."""
    from cluster_rag.upr_analysis import fold_label_chunks

    matrix = np.asarray([[0.1, 0.9, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]])
    folded = fold_label_chunks(matrix, ((0, 1), (2, 3)))
    assert folded.shape == (2, 2)
    assert folded[0, 0] == pytest.approx(0.9)  # label 0: the best of chunks 0 and 1
    assert folded[1, 1] == pytest.approx(0.7)  # label 1: the best of chunks 2 and 3


def test_chunked_arm_scores_folds_the_chunks_per_text() -> None:
    """The chunk-level scores are folded into one score per text."""
    from cluster_rag.upr_analysis import aggregation_of, chunked_arm_scores

    class _FakePredictor:
        def score(self, texts, *, query_vectors=None):
            # the first chunk of every text scores 0.9, the second 0.1
            rows = []
            for text in texts:
                rows.append([0.9 if text.endswith("a") else 0.1, 0.2])
            return np.asarray(rows, dtype=np.float32)

    from cluster_rag.upr_analysis import ChunkedSide

    side = ChunkedSide(
        chunk_lists=(("xa", "xb"), ("ya", "yb")),
        vectors=np.zeros((4, 2), dtype=np.float32),
    )
    scores = chunked_arm_scores(_FakePredictor(), side, aggregation_of("max"))
    assert scores.shape == (2, 2)
    assert scores[0, 0] == pytest.approx(0.9)  # the max over the text's chunks
    assert scores[0, 1] == pytest.approx(0.2)


def test_encode_chunks_refuses_a_text_without_a_chunk() -> None:
    """A chunker that drops a text fails loudly (never a silent empty side)."""
    from cluster_rag.upr_analysis import encode_chunks

    class _EmptyChunker:
        def segment(self, texts):
            return [[] for _ in texts]

    class _FakeEmbedder:
        def encode(self, texts, kind):
            return np.zeros((len(texts), 2), dtype=np.float32)

    with pytest.raises(ValueError, match="no chunk for a text"):
        encode_chunks(_FakeEmbedder(), _EmptyChunker(), ["a text"], "query")


# --- the driver ---


def _fixture_view():
    """A tiny view: two texts, three labels, one gold cell each."""
    from cluster_rag.data import Record

    records = (
        Record(id="r1", text="a" * 30, language="en", labels=["Alpha"]),
        Record(id="r2", text="b" * 200, language="en", labels=["Beta", "Gamma"]),
    )
    labels = ["Alpha", "Beta", "Gamma"]
    buckets = {"Alpha": "head", "Beta": "tail", "Gamma": "unseen"}
    return build_upr_view(records, labels, buckets, (100, 200))


def test_build_upr_view_counts_the_gold_and_the_background() -> None:
    """The gold cells are the record's own labels; the background is the rest."""
    view = _fixture_view()
    assert view.n_texts == 2 and view.n_labels == 3
    assert view.n_gold_cells == 3
    assert view.n_background_cells == 3  # 2 texts x 3 labels - 3 gold
    # the edges are the upper bounds: 30 chars -> bucket 0, 200 chars -> bucket 2
    assert view.length_bucket_index == (0, 2)
    assert view.length_bucket_names == ("0-99", "100-199", "200+")


def test_build_upr_view_rejects_a_label_without_a_bucket() -> None:
    """A label missing from the bucket mapping fails loudly."""
    from cluster_rag.data import Record

    with pytest.raises(ValueError, match="missing from the bucket mapping"):
        build_upr_view(
            (Record(id="r1", text="x", language="en", labels=["Alpha"]),),
            ["Alpha", "Beta"],
            {"Alpha": "head"},
            (100,),
        )


def test_arm_block_reports_the_metrics_and_the_background() -> None:
    """The arm block carries the UPR metrics, the gold side and the background."""
    view = _fixture_view()
    scores = np.asarray([[0.9, 0.2, 0.1], [0.1, 0.8, 0.7]])
    block = arm_block(
        view,
        scores,
        decision={"rule": "top_k", "k": 1, "threshold": None, "min_score": None},
        ks=(1, 3),
        shortlist_ms=(2,),
        predictor={"name": "fake"},
        chunker="full_text",
    )
    assert block["arm"] == BASELINE_ARM
    assert block["metrics"]["overall"]["n_records"] == 2
    assert block["gold"]["n_cells"] == 3
    assert block["background"]["n_cells"] == 3
    # the gold cells' rank percentiles: 0.0, 0.0 and 1/3
    assert block["gold"]["rank_p50"] == pytest.approx(0.0)
    # the background's cells sit at 1/3, 2/3 and 2/3 -> the median is 2/3
    assert block["background"]["rank_p50"] == pytest.approx(2 / 3)
    assert block["provenance"]["top1_gold"] == 2
    assert set(block["metrics"]["per_length_bucket"]) == {"0-99", "200+"}
    assert block["distributions"]["per_label"]["Alpha"]["bucket"] == "head"


# --- the CLI command ---


@pytest.fixture()
def analyze_upr_run(sample_jsonl, write_config, tmp_path):
    """Factory: split + evaluate (the guard's reference) + analyze-upr."""

    def run(run_id="upr-analyze-test", extra=ANALYZE_EXTRA, table=None):
        config_file = write_config(
            sample_jsonl,
            run_id=run_id,
            extra=extra,
            experiment_table_text_processing_upr=table,
        )
        if not (tmp_path / "splits" / "train_ids.json").is_file():
            assert main(["split", "--config", str(config_file)]) == 0
        code = main(["analyze-upr", "--config", str(config_file)])
        results_dir = tmp_path / "results" / run_id
        path = results_dir / "analysis.json"
        artifact = json.loads(path.read_text("utf-8")) if path.is_file() else None
        return code, artifact, config_file, results_dir

    return run


def _write_reference_eval(sample_jsonl, write_config, tmp_path, run_id="upr-eval-ref"):
    """The matching `evaluate` run (the guard's reference artifact)."""
    config_file = write_config(sample_jsonl, run_id=run_id, extra=EVAL_EXTRA)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["evaluate", "--config", str(config_file)]) == 0
    return tmp_path / "results" / run_id / "metrics.json"


def test_analyze_upr_command_artifacts(analyze_upr_run, sample_jsonl, write_config, tmp_path):
    """analyze-upr: the artifact, the reference arm and the guard's outcome."""
    _write_reference_eval(sample_jsonl, write_config, tmp_path)
    code, artifact, config_file, results_dir = analyze_upr_run()
    assert code == 0
    assert artifact["command"] == "analyze-upr"
    assert artifact["split"] == "val"
    assert artifact["dataset"]["n_texts"] > 0
    assert artifact["dataset"]["n_background_cells"] > 0
    # the background is fully enumerable: no sampling, no seed
    assert artifact["dataset"]["background"]["sampling"] is None
    assert artifact["dataset"]["background"]["seed"] is None
    # the reference arm is the pipeline's own scoring
    assert BASELINE_ARM in artifact["arms"]
    arm = artifact["arms"][BASELINE_ARM]
    assert arm["chunker"] == "full_text"
    assert arm["metrics"]["overall"]["n_records"] == artifact["dataset"]["n_texts"]
    assert "per_length_bucket" in arm["metrics"]
    assert arm["gold"]["n_cells"] == artifact["dataset"]["n_gold_cells"]
    assert arm["background"]["n_cells"] == artifact["dataset"]["n_background_cells"]
    # the guard passed: the reference arm reproduces the evaluate run
    assert artifact["guard"]["matches"] is True
    assert artifact["guard"]["reference_run"] == "upr-eval-ref"
    frozen = yaml.safe_load((results_dir / "config.yaml").read_text("utf-8"))
    assert frozen["upr_analysis"]["reference_eval_run"] == "upr-eval-ref"
    assert config_file.is_file()


def test_analyze_upr_guard_fails_on_a_mismatch(
    analyze_upr_run, sample_jsonl, write_config, tmp_path, caplog
):
    """A reference run with a different predictor fails the command, not a warning."""
    other = EVAL_EXTRA.replace("name: most_frequent", "name: random")
    config_file = write_config(sample_jsonl, run_id="upr-eval-other", extra=other)
    if not (tmp_path / "splits" / "train_ids.json").is_file():
        assert main(["split", "--config", str(config_file)]) == 0
    assert main(["evaluate", "--config", str(config_file)]) == 0
    with caplog.at_level(logging.ERROR):
        code, artifact, _, _ = analyze_upr_run(
            run_id="upr-analyze-mismatch",
            extra=ANALYZE_EXTRA.replace("upr-eval-ref", "upr-eval-other"),
        )
    assert code == 1 and artifact is None
    assert any("does not reproduce" in record.message for record in caplog.records)


def test_analyze_upr_requires_the_upr_analysis_section(analyze_upr_run) -> None:
    """Without upr_analysis: the command exits 1 and writes no artifact."""
    code, artifact, _, results_dir = analyze_upr_run(run_id="upr-analyze-nosec", extra=EVAL_EXTRA)
    assert code == 1 and artifact is None
    assert not (results_dir / "analysis.json").exists()


def test_analyze_upr_appends_the_track_table_row(
    analyze_upr_run, sample_jsonl, write_config, tmp_path
):
    """The row lands in the track's own UPR table, keyed by (run_id, arm)."""
    _write_reference_eval(sample_jsonl, write_config, tmp_path)
    table = tmp_path / "tp_upr.md"
    code, _, _, _ = analyze_upr_run(run_id="upr-analyze-table", table=table)
    assert code == 0
    content = table.read_text("utf-8")
    assert "## UPR — analyze-upr" in content
    assert "| upr-analyze-table |" in content
    assert "| full_text |" in content
    header = next(line for line in content.splitlines() if line.startswith("| run_id"))
    assert header.index("micro-F1") < header.index("P@5")
    assert "bg_rank_p50" in header and "bg_mean_delta" in header


def test_analyze_upr_is_deterministic(
    analyze_upr_run, sample_jsonl, write_config, tmp_path
) -> None:
    """A re-run reproduces the artifact (only the wall time may move)."""
    _write_reference_eval(sample_jsonl, write_config, tmp_path)
    _, first, config_file, _ = analyze_upr_run(run_id="upr-analyze-det")
    assert main(["analyze-upr", "--config", str(config_file)]) == 0
    second = json.loads(
        (config_file.parent / "results" / "upr-analyze-det" / "analysis.json").read_text("utf-8")
    )
    del first["total_seconds"], second["total_seconds"]
    assert first == second
