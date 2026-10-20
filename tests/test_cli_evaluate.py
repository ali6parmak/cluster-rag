"""Tests for the `evaluate` CLI command (Phase 3), on the tiny fixture dataset.

Covers: split -> evaluate end-to-end (artifacts, bucket sections, table row),
seeded determinism on rerun, the --split flag, the config-error contract, and
the Phase 4 embedding-run wiring (knn via a fabricated upr-embed-fix artifact,
with a fake embedder, no network).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from cluster_rag import cli
from cluster_rag.cli import main
from cluster_rag.data import load_jsonl
from cluster_rag.dataset_stats import file_sha256
from fake_embedder import FakeEmbedder

EVAL_CONFIG_EXTRA = """
eval:
  predictor:
    name: most_frequent
  decision:
    rule: top_k
    k: 2
"""

KNN_CONFIG_EXTRA = """
eval:
  predictor:
    name: knn
    params:
      k: 2
  embedding_run_id: upr-embed-fix
  decision:
    rule: top_k
    k: 2
"""


def _write_embedding_run(results_root: Path, sample_jsonl: Path) -> None:
    """Fabricate results/upr-embed-fix (npz + meta.json) for the fixture dataset.

    Mirrors what the embed command writes — ids in dataset order, one
    (n, d) float32 unit-norm matrix per kind, plus the lineage meta.
    """
    report = load_jsonl(sample_jsonl)
    ids = [record.id for record in report.records]
    texts = [record.text for record in report.records]
    embedder = FakeEmbedder(dimension=8)
    run_dir = results_root / "upr-embed-fix"
    run_dir.mkdir(parents=True)
    np.savez_compressed(
        run_dir / "embeddings.npz",
        ids=np.asarray(ids),
        vectors_document=embedder.encode(texts, "document"),
        vectors_query=embedder.encode(texts, "query"),
    )
    meta = {
        "run_id": "upr-embed-fix",
        "command": "embed",
        "kinds": ["document", "query"],
        "n_records": len(ids),
        "source_sha256": file_sha256(sample_jsonl),
        "embedder": {
            "backend": "fake",
            "model_id": "fake-embedder",
            "dimension": 8,
            "revision_or_digest": "fake-revision",
            "context_tokens": None,
            "prefixes": {"query": "", "document": "", "label": ""},
            "extra": {},
        },
        "cache": {"enabled": False},
        "preprocess": {"variant": "raw", "version": "raw-v1", "n_changed": 0},
        "kind_stats": {},
        "total_seconds": 0.1,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


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
    name: no_such_predictor
  decision:
    rule: top_k
    k: 2
"""
    code, metrics, _, results_dir = evaluate_run(run_id="evaluate-unknown", extra=bad_extra)
    assert code == 1
    assert metrics is None
    assert not (results_dir / "metrics.json").exists()


def test_evaluate_knn_consumes_the_embedding_run(evaluate_run, sample_jsonl, tmp_path) -> None:
    """Phase 4 wiring: knn resolves, validates and aligns the upr-embed artifact."""
    _write_embedding_run(tmp_path / "results", sample_jsonl)
    code, metrics, _, results_dir = evaluate_run(run_id="upr-eval-knn-fix", extra=KNN_CONFIG_EXTRA)
    assert code == 0
    assert metrics["predictor"]["name"] == "knn" and metrics["predictor"]["k"] == 2
    lineage = metrics["embedding_run"]
    assert lineage is not None
    assert lineage["run_id"] == "upr-embed-fix"
    assert lineage["backend"] == "fake" and lineage["model_id"] == "fake-embedder"
    assert lineage["revision_or_digest"] == "fake-revision" and lineage["dimension"] == 8
    assert lineage["kinds"] == ["document", "query"]
    frozen = yaml.safe_load((results_dir / "config.yaml").read_text("utf-8"))
    assert frozen["eval"]["embedding_run_id"] == "upr-embed-fix"


def test_evaluate_knn_without_embedding_run_id_fails(
    evaluate_run, sample_jsonl, tmp_path, caplog
) -> None:
    """A kNN config with no eval.embedding_run_id fails with the wiring message."""
    knn_only = KNN_CONFIG_EXTRA.replace("  embedding_run_id: upr-embed-fix\n", "")
    with caplog.at_level(logging.ERROR):
        code, metrics, _, results_dir = evaluate_run(run_id="upr-eval-knn-nowire", extra=knn_only)
    assert code == 1 and metrics is None
    assert not (results_dir / "metrics.json").exists()
    assert any("requires embeddings" in record.message for record in caplog.records)


def test_evaluate_knn_with_missing_artifact_fails(
    evaluate_run, sample_jsonl, tmp_path, caplog
) -> None:
    """A dangling embedding_run_id fails naming the artifact and the fix."""
    with caplog.at_level(logging.ERROR):
        code, metrics, _, _ = evaluate_run(run_id="upr-eval-knn-missing", extra=KNN_CONFIG_EXTRA)
    assert code == 1 and metrics is None
    assert any("not found" in record.message for record in caplog.records)


def test_evaluate_warns_and_ignores_embedding_run_for_free_predictors(
    evaluate_run, sample_jsonl, tmp_path, caplog
) -> None:
    """most_frequent + embedding_run_id: run proceeds, warning logged, lineage None."""
    _write_embedding_run(tmp_path / "results", sample_jsonl)
    mf_extra = EVAL_CONFIG_EXTRA + "  embedding_run_id: upr-embed-fix\n"
    with caplog.at_level(logging.WARNING):
        code, metrics, _, _ = evaluate_run(run_id="upr-eval-mf-warn", extra=mf_extra)
    assert code == 0
    assert metrics["embedding_run"] is None  # no lineage: the artifact was never read
    assert any(
        "does not use" in record.message and "ignoring it" in record.message
        for record in caplog.records
    )


def test_evaluate_table_row_carries_embedding_column(evaluate_run, sample_jsonl, tmp_path) -> None:
    """The auto table shows the embedding run's detail next to the predictor."""
    _write_embedding_run(tmp_path / "results", sample_jsonl)
    table = tmp_path / "tables.md"
    code, _, _, _ = evaluate_run(run_id="upr-eval-knn-table", extra=KNN_CONFIG_EXTRA, table=table)
    assert code == 0
    content = table.read_text("utf-8")
    header = next(line for line in content.splitlines() if line.startswith("| run_id"))
    assert header.index("predictor") < header.index("embedding") < header.index("f1 decision")
    assert "| upr-eval-knn-table |" in content and "| fix |" in content


# --- Phase 4 zero-shot: the labelembed artifact and its consumers ---

LABEL_RUN_ID = "upr-labelembed-fake-name"
LABEL_RUN_EXTRA = """
embed:
  backend: fake
  model_id: fake-embedder
label_texts:
  variant: name
"""


def _sorted_fixture_labels(sample_jsonl) -> list[str]:
    from cluster_rag.data import load_jsonl

    return sorted({label for record in load_jsonl(sample_jsonl).records for label in record.labels})


def _run_labelembed(
    sample_jsonl, write_config, tmp_path, monkeypatch, run_id=LABEL_RUN_ID, extra=LABEL_RUN_EXTRA
):
    """Run the labelembed command with a fake embedder; returns (code, artifacts)."""
    created: list[FakeEmbedder] = []

    def factory(config):
        fake = FakeEmbedder(model_id="fake-embedder", dimension=8)
        created.append(fake)
        return fake

    monkeypatch.setattr(cli, "build_embedder", factory)
    config_file = write_config(sample_jsonl, run_id=run_id, extra=extra)
    code = main(["labelembed", "--config", str(config_file)])
    results_dir = tmp_path / "results" / run_id
    npz_path = results_dir / "label_embeddings.npz"
    npz = np.load(npz_path) if npz_path.is_file() else None
    meta = (
        json.loads((results_dir / "meta.json").read_text("utf-8"))
        if (results_dir / "meta.json").is_file()
        else None
    )
    return code, npz, meta, created, config_file


def test_labelembed_command_artifacts(sample_jsonl, write_config, tmp_path, monkeypatch) -> None:
    """label_embeddings.npz: sorted labels, unit rows, meta lineage, table row."""
    code, npz, meta, _, _ = _run_labelembed(sample_jsonl, write_config, tmp_path, monkeypatch)
    assert code == 0
    labels = _sorted_fixture_labels(sample_jsonl)
    assert [str(item) for item in npz["labels"]] == labels
    vectors = np.asarray(npz["vectors_label"])
    assert vectors.shape == (len(labels), 8) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert meta["command"] == "labelembed" and meta["n_labels"] == len(labels)
    assert meta["label_texts"]["variant"] == "name"
    assert meta["embedder"]["model_id"] == "fake-embedder"
    assert meta["embedder"]["revision_or_digest"] == "fake-revision"
    # the vectors are exactly the fake embedder's label-kind hashes in label order
    fake = FakeEmbedder(model_id="fake-embedder", dimension=8)
    expected = fake.encode(labels, "label")
    assert np.allclose(vectors, expected)


def test_labelembed_requires_embed_and_label_texts_sections(
    sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """Missing label_texts: or embed: exits 1 without artifacts."""
    code, npz, meta, _, _ = _run_labelembed(
        sample_jsonl, write_config, tmp_path, monkeypatch, extra="embed:\n  backend: fake\n"
    )
    assert code == 1 and npz is None and meta is None


def _zero_shot_setup(sample_jsonl, write_config, tmp_path, monkeypatch):
    """Pre-create the primary embedding artifact + the (fake-backend) label artifact.

    No split here: evaluate_run splits itself, and the split command refuses a
    silent overwrite.
    """
    _write_embedding_run(tmp_path / "results", sample_jsonl)
    _, _, _, _, label_config = _run_labelembed(sample_jsonl, write_config, tmp_path, monkeypatch)
    return label_config


def test_evaluate_zeroshot_consumes_both_artifacts(
    evaluate_run, sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """zeroshot reads eval.embedding_run_id + eval.label_embedding_run_id."""
    _zero_shot_setup(sample_jsonl, write_config, tmp_path, monkeypatch)
    zeroshot_extra = """
eval:
  predictor:
    name: zeroshot
  embedding_run_id: upr-embed-fix
  label_embedding_run_id: upr-labelembed-fake-name
  decision:
    rule: top_k
    k: 2
"""
    code, metrics, _, _ = evaluate_run(run_id="upr-eval-zeroshot-fix", extra=zeroshot_extra)
    assert code == 0
    assert metrics["predictor"]["name"] == "zeroshot"
    assert metrics["embedding_run"]["run_id"] == "upr-embed-fix"
    label_lineage = metrics["label_embedding_run"]
    assert label_lineage is not None
    assert label_lineage["run_id"] == "upr-labelembed-fake-name"
    assert label_lineage["label_texts"]["variant"] == "name"
    assert label_lineage["n_labels"] == metrics["n_labels"]


def test_evaluate_fusion_c_term_end_to_end(
    evaluate_run, sample_jsonl, write_config, tmp_path, monkeypatch
) -> None:
    """fusion with c > 0 wires both runs and reports the full weight set."""
    _zero_shot_setup(sample_jsonl, write_config, tmp_path, monkeypatch)
    fusion_extra = """
eval:
  predictor:
    name: fusion
    params:
      a: 1.0
      b: 1.0
      c: 0.5
      k: 2
  embedding_run_id: upr-embed-fix
  label_embedding_run_id: upr-labelembed-fake-name
  decision:
    rule: top_k
    k: 2
"""
    code, metrics, _, _ = evaluate_run(run_id="upr-eval-fusion-c-fix", extra=fusion_extra)
    assert code == 0
    assert metrics["predictor"]["c"] == 0.5
    assert metrics["label_embedding_run"]["run_id"] == "upr-labelembed-fake-name"


def test_evaluate_zeroshot_without_label_run_fails(
    evaluate_run, sample_jsonl, write_config, tmp_path, monkeypatch, caplog
) -> None:
    """A zeroshot config with no eval.label_embedding_run_id fails, naming the key."""
    _zero_shot_setup(sample_jsonl, write_config, tmp_path, monkeypatch)
    zeroshot_extra = """
eval:
  predictor:
    name: zeroshot
  embedding_run_id: upr-embed-fix
  decision:
    rule: top_k
    k: 2
"""
    with caplog.at_level(logging.ERROR):
        code, metrics, _, results_dir = evaluate_run(
            run_id="upr-eval-zeroshot-nowire", extra=zeroshot_extra
        )
    assert code == 1 and metrics is None
    assert not (results_dir / "metrics.json").exists()
    assert any("requires zero-shot label vectors" in record.message for record in caplog.records)


def test_evaluate_label_space_mismatch_fails(
    evaluate_run, sample_jsonl, write_config, tmp_path, monkeypatch, caplog
) -> None:
    """A tampered label artifact (label dropped) is refused, never mis-scored."""
    _zero_shot_setup(sample_jsonl, write_config, tmp_path, monkeypatch)
    npz_path = tmp_path / "results" / "upr-labelembed-fake-name" / "label_embeddings.npz"
    with np.load(npz_path) as data:
        labels = [str(item) for item in np.asarray(data["labels"])]
        vectors = np.asarray(data["vectors_label"])
    np.savez_compressed(
        npz_path,
        labels=np.asarray(labels[:-1]),
        vectors_label=vectors[:-1],
    )
    zeroshot_extra = """
eval:
  predictor:
    name: zeroshot
  embedding_run_id: upr-embed-fix
  label_embedding_run_id: upr-labelembed-fake-name
  decision:
    rule: top_k
    k: 2
"""
    with caplog.at_level(logging.ERROR):
        code, metrics, _, _ = evaluate_run(
            run_id="upr-eval-zeroshot-tampered", extra=zeroshot_extra
        )
    assert code == 1 and metrics is None
    assert any("label space mismatches" in record.message for record in caplog.records)


def test_evaluate_warns_and_ignores_label_run_for_free_predictors(
    evaluate_run, sample_jsonl, write_config, tmp_path, monkeypatch, caplog
) -> None:
    """most_frequent + label_embedding_run_id: runs proceed, warning logged, lineage None."""
    _zero_shot_setup(sample_jsonl, write_config, tmp_path, monkeypatch)
    mf_extra = (
        EVAL_CONFIG_EXTRA
        + "  embedding_run_id: upr-embed-fix\n"
        + "  label_embedding_run_id: upr-labelembed-fake-name\n"
    )
    with caplog.at_level(logging.WARNING):
        code, metrics, _, _ = evaluate_run(run_id="upr-eval-mf-labelwarn", extra=mf_extra)
    assert code == 0
    assert metrics["label_embedding_run"] is None
    assert any(
        "zero-shot label vectors" in record.message and "ignoring it" in record.message
        for record in caplog.records
    )
