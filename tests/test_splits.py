"""Tests for the multi-label split machinery (Phase 1)."""

from __future__ import annotations

import json
import logging

import numpy as np

from cluster_rag.cli import main
from cluster_rag.config import LabelHoldoutConfig, SplitsConfig
from cluster_rag.data import (
    Record,
    build_label_matrix,
    build_splits,
    load_jsonl,
    rebalance_split_coverage,
    select_label_holdout,
    stratified_split_indices,
)


def _rec(rec_id: int, labels: list[str], text: str = "some text") -> Record:
    return Record.model_validate(
        {"id": str(rec_id), "text": text, "language": "en", "labels": labels}
    )


def test_build_label_matrix() -> None:
    """Multi-hot construction with a stable, externally given label order."""
    records = [_rec(1, ["A", "B"]), _rec(2, ["B", "C"])]
    matrix = build_label_matrix(records, ["A", "B", "C"])
    assert matrix.dtype == np.bool_
    assert matrix.shape == (2, 3)
    assert matrix.tolist() == [[True, True, False], [False, True, True]]


def test_stratified_split_partition_and_determinism() -> None:
    """Every sample is assigned exactly once; the same seed reproduces it."""
    records = [_rec(i, [["A", "B", "C"][i % 3], ["D", "E"][i % 2]]) for i in range(60)]
    matrix = build_label_matrix(records, ["A", "B", "C", "D", "E"])

    split_of = stratified_split_indices(matrix, (0.7, 0.15, 0.15), seed=7)
    counts = [int((split_of == s).sum()) for s in (0, 1, 2)]
    assert counts[0] == 42  # 0.7 * 60 exactly
    assert 7 <= counts[1] <= 11  # 0.15 * 60 = 9
    assert 7 <= counts[2] <= 11
    assert ((split_of == 0) | (split_of == 1) | (split_of == 2)).all()

    again = stratified_split_indices(matrix, (0.7, 0.15, 0.15), seed=7)
    assert (again == split_of).all()
    different = stratified_split_indices(matrix, (0.7, 0.15, 0.15), seed=8)
    assert (different != split_of).any()


def test_rebalance_moves_missing_label_into_every_split() -> None:
    """Label C (train/train/val) gains a test example; B (1 example) is untouched."""
    # columns: A=0, B=1, C=2 ; rows 0,1,2 train, row 3 val, row 4 test
    matrix = np.array(
        [
            [True, False, False],
            [False, False, True],
            [False, False, True],
            [False, True, True],
            [True, False, False],
        ],
        dtype=bool,
    )
    split_of = np.array([0, 0, 0, 1, 2], dtype=np.int8)
    moves = rebalance_split_coverage(matrix, split_of, min_total=3)

    assert len(moves) == 1
    assert moves[0]["label_index"] == 2
    assert moves[0]["to"] == "test"
    assert matrix[moves[0]["row"], 2]  # the moved row contains C
    assert int(matrix[split_of == 0, 2].sum()) == 1  # one C stayed in train
    assert int(matrix[split_of == 1, 2].sum()) == 1
    assert int(matrix[split_of == 2, 2].sum()) == 1
    assert int(matrix[split_of == 0, 1].sum()) == 0  # B never moved (too rare)


def test_rebalance_respects_min_total() -> None:
    """A label with fewer than min_total examples is left untouched."""
    matrix = np.array([[True, True], [True, True], [False, False]], dtype=bool)
    split_of = np.array([0, 0, 2], dtype=np.int8)
    moves = rebalance_split_coverage(matrix, split_of, min_total=3)
    assert moves == []  # B has 2 examples, both train: reported, not churned
    assert (split_of == np.array([0, 0, 2], dtype=np.int8)).all()


def test_rebalance_avoids_emptying_the_donor_split() -> None:
    """Donor preference: give from the split with surplus, not the last copy."""
    # column C: train 1, val 2, test 0 -> the moved example must come from val
    matrix = np.array(
        [
            [True, False],  # A   train
            [False, True],  # C   train
            [False, True],  # C   val
            [False, True],  # C   val
            [True, False],  # A   test
        ],
        dtype=bool,
    )
    split_of = np.array([0, 0, 1, 1, 2], dtype=np.int8)
    moves = rebalance_split_coverage(matrix, split_of, min_total=3)

    moved = [move for move in moves if move["label_index"] == 1]
    assert len(moved) == 1
    assert moved[0]["from"] == "val" and moved[0]["to"] == "test"
    assert int(matrix[split_of == 0, 1].sum()) == 1  # train kept its copy


def test_select_label_holdout_prefers_low_collateral() -> None:
    """Greedy pick: least collateral first; held-out labels vanish from train."""
    # columns: A=0, B=1, H=2, X=3 ; rows 0-3,6 train, row 7 val, rows 4,5 test
    matrix = np.array(
        [
            [False, False, True, True],  # H+X  train
            [False, False, True, False],  # H    train
            [False, False, True, False],  # H    train
            [True, False, False, False],  # A    train
            [False, False, True, False],  # H    test
            [True, False, False, False],  # A    test
            [True, False, False, False],  # A    train
            [True, False, False, False],  # A    val
        ],
        dtype=bool,
    )
    split_of = np.array([0, 0, 0, 0, 2, 2, 0, 1], dtype=np.int8)
    picked = select_label_holdout(
        matrix, split_of, ["A", "B", "H", "X"], n_labels=2, min_examples=4, max_examples=10
    )

    # A (total 4, collateral 0) first; H second (total 4, removing it kills X in train)
    assert [entry["label"] for entry in picked] == ["A", "H"]
    assert picked[0]["collateral_labels_removed_from_train"] == []
    assert picked[1]["collateral_labels_removed_from_train"] == ["X"]
    assert picked[1]["total_examples"] == 4
    assert picked[1]["train_records_removed"] == 3


def test_build_splits_deterministic_partition_and_holdout(sample_jsonl) -> None:
    """End-to-end on the fixture: reproducible, a partition, holdout out of train."""
    records = load_jsonl(sample_jsonl).records
    config = SplitsConfig(
        ratios=(0.7, 0.15, 0.15),
        coverage_min_total=2,
        label_holdout=LabelHoldoutConfig(n_labels=2, min_examples=1, max_examples=5),
    )
    first = build_splits(records, config, seed=42)
    assert first == build_splits(records, config, seed=42)

    all_ids = {record.id for record in records}
    train_ids = set(first["ids"]["train"])
    val_ids = set(first["ids"]["val"])
    test_ids = set(first["ids"]["test"])
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == all_ids

    labels_of = {record.id: set(record.labels) for record in records}
    holdout = set(first["label_holdout"])
    assert len(holdout) == 2
    for rec_id in train_ids:
        assert not labels_of[rec_id] & holdout, "holdout label leaked into train"
    totals: dict[str, int] = {}
    for record in records:
        for label in record.labels:
            totals[label] = totals.get(label, 0) + 1
    for label in holdout:
        in_val = first["per_split_label_counts"]["val"][label]
        in_test = first["per_split_label_counts"]["test"][label]
        if totals[label] >= config.coverage_min_total:
            assert in_val >= 1 and in_test >= 1, f"{label}: eval coverage incomplete"
        else:  # singleton labels can only ever land in one split
            assert in_val + in_test >= 1, f"{label}: vanished from the evaluation pools"

    # per-label conservation: occurrences across the three splits add up
    total: dict[str, int] = {}
    for record in records:
        for label in record.labels:
            total[label] = total.get(label, 0) + 1
    for label, count in total.items():
        summed = sum(
            first["per_split_label_counts"][split][label] for split in ("train", "val", "test")
        )
        assert summed == count


def test_build_splits_holdout_eligibility_respected(sample_jsonl) -> None:
    """With min=max=4 examples, only the label with exactly 4 examples is chosen."""
    records = load_jsonl(sample_jsonl).records
    config = SplitsConfig(
        ratios=(0.7, 0.15, 0.15),
        label_holdout=LabelHoldoutConfig(n_labels=1, min_examples=4, max_examples=4),
    )
    manifest = build_splits(records, config, seed=42)
    assert manifest["label_holdout"] == ["Children"]  # the only fixture label with 4 examples
    assert "Children" in manifest["labels_missing_from_train"]
    # co-occurring labels may collateral-ly vanish from train on 20 rows; the
    # manifest must name them explicitly instead of hiding the damage
    assert set(manifest["labels_missing_from_train"]) >= {"Children"}


def test_split_command_end_to_end(sample_jsonl, write_config, tmp_path, caplog) -> None:
    """CLI end-to-end: artifacts written, overwrite-protected, reproducible."""
    config_file = write_config(sample_jsonl, run_id="split-test")

    assert main(["split", "--config", str(config_file)]) == 0
    manifest_path = tmp_path / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for name in ("train", "val", "test"):
        ids_file = tmp_path / "splits" / f"{name}_ids.json"
        ids = json.loads(ids_file.read_text("utf-8"))
        assert ids["split"] == name and len(ids["ids"]) == manifest["sizes"][name]
    holdout_file = json.loads((tmp_path / "splits" / "label_holdout.json").read_text("utf-8"))
    assert holdout_file["labels"] == manifest["label_holdout"]
    assert (tmp_path / "results" / "split-test" / "config.yaml").is_file()
    assert (tmp_path / "results" / "split-test" / "split_report.json").is_file()

    # rerun without --force: refused
    with caplog.at_level(logging.ERROR):
        assert main(["split", "--config", str(config_file)]) == 1
    assert any("--force" in record.message for record in caplog.records)

    # rerun with --force: same ids for the same seed
    assert main(["split", "--config", str(config_file), "--force"]) == 0
    assert json.loads(manifest_path.read_text("utf-8"))["ids"] == manifest["ids"]


def test_split_command_rejects_bad_ratios(sample_jsonl, write_config, caplog) -> None:
    """Ratios that do not sum to 1 are a config error, not a silent split."""
    config_file = write_config(
        sample_jsonl, run_id="bad-ratios", extra="splits:\n  ratios: [0.5, 0.2, 0.2]\n"
    )
    with caplog.at_level(logging.ERROR):
        assert main(["split", "--config", str(config_file)]) == 1
    assert any("sum to 1" in record.message for record in caplog.records)
