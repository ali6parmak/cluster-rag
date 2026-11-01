"""Compare two lexical arms of the corpus map **paired**, on the same texts.

The track's rule: *no verdict rests on two independent numbers*. The two arms
of the corpus map (CS-30's four, CS-31's two, CS-33's two class-restricted
ones) are scored on the **same** texts, with the same map and the same
operating point, so the difference between them is paired — and the paired
bootstrap over the texts (the resampling unit) puts an interval on it.

The tool **rebuilds** both arms' score matrices from the runs' own frozen
configs (the same pure functions the CLI uses) and **checks the rebuild against
the runs' artifacts**: the recomputed micro-F1 must equal the artifact's, or
the tool exits 1. So the interval is never about a different computation than
the one the runs measured.

Usage:

    uv run python tools/lexical_compare.py upr-eval-corpus-map-sum-positive-val \\
        upr-eval-corpus-map-sum-zscore-positive-val

    uv run python tools/lexical_compare.py --resamples 5000 <run> <run>

Exit code 0 when the rebuild reproduces both runs and the comparison is
printed; 1 when it does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cluster_rag.config import load_config
from cluster_rag.data import build_label_matrix, load_jsonl, load_split_ids, select_records_by_ids
from cluster_rag.eval.multilabel import (
    apply_decision_rule,
    decision_f1_metrics,
    paired_metric_bootstrap,
)
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.corpus_map_score import score_cells

DEFAULT_RESULTS = Path("results")


def _load_yaml(path: Path) -> dict[str, Any]:
    """One YAML file as a dict."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _split_ids(config: dict[str, Any], split: str) -> list[str]:
    """The fixed split's ids (the splits' own provenance check)."""
    ids, _ = load_split_ids(Path(config["paths"]["data_splits"]), split)
    return list(ids)


def _arm_scores(run_id: str, results: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild one run's score matrix (and the gold) from its frozen config.

    Args:
        run_id: The run whose artifact and frozen config are read.
        results: The results directory.

    Returns:
        The ``(n_texts, n_labels)`` score matrix, the gold matrix and the run's
        artifact.

    Raises:
        ValueError: If the run's config or artifact is missing, or the run is
            not a ``corpus_map`` evaluation.
    """
    run_dir = results / run_id
    config = _load_yaml(run_dir / "config.yaml")
    artifact = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    eval_config = config.get("eval") or {}
    predictor = eval_config.get("predictor") or {}
    if predictor.get("name") != "corpus_map":
        raise ValueError(
            f"{run_id}: this tool compares the corpus_map arms, got {predictor.get('name')!r}"
        )
    split = eval_config.get("split", "val")
    paths = config["paths"]
    report = load_jsonl(paths["data_raw"])
    ids_by_split = {name: _split_ids(config, name) for name in (split, "train")}
    records = select_records_by_ids(report.records, ids_by_split[split])
    train_records = select_records_by_ids(report.records, ids_by_split["train"])
    labels = sorted({label for record in report.records for label in record.labels})
    scoring = (predictor.get("params") or {}).get("scoring", "sum_logodds")
    restrict = (predictor.get("params") or {}).get("restrict")
    restrict_pos = (predictor.get("params") or {}).get("restrict_pos")
    word_pos: dict[str, str] = {}
    if restrict_pos is not None:
        # the grammatical classes (CS-34): the train corpus is tagged with the
        # same edge client the run used, so the rebuild reads the same map.
        from cluster_rag.commands.upr import pos_word_classes

        word_pos, _distribution, _provenance = pos_word_classes(
            load_config(run_dir / "config.yaml"),
            [record.text for record in train_records],
        )
    corpus_map = build_corpus_map(
        [record.text for record in train_records],
        [record.labels for record in train_records],
        labels,
        word_pos=word_pos or None,
    )
    scores = score_cells(
        [record.text for record in records],
        corpus_map,
        scoring=scoring,
        restrict=restrict,
        restrict_pos=restrict_pos,
    )
    gold = build_label_matrix(records, labels)
    return scores, gold, artifact


def compare(left_id: str, right_id: str, results: Path, resamples: int) -> int:
    """The paired comparison of two arms; 0 when the rebuild reproduces both runs."""
    left_scores, gold, left_artifact = _arm_scores(left_id, results)
    right_scores, _, right_artifact = _arm_scores(right_id, results)
    decision = left_artifact.get("decision") or {}
    rule = decision.get("rule", "top_k")
    k = decision.get("k", 2)
    threshold = decision.get("threshold")
    min_score = decision.get("min_score")

    problems: list[str] = []
    for run_id, artifact, scores in (
        (left_id, left_artifact, left_scores),
        (right_id, right_artifact, right_scores),
    ):
        predicted = apply_decision_rule(
            scores, rule=rule, k=k, threshold=threshold, min_score=min_score
        )
        recomputed = decision_f1_metrics(predicted, gold)["micro_f1"]
        reported = artifact["report"]["overall"]["f1"]["micro_f1"]
        if abs(recomputed - reported) > 1e-9:
            problems.append(
                f"{run_id}: the rebuild's micro-F1 {recomputed:.6f} does not match the "
                f"artifact's {reported:.6f}"
            )
    if problems:
        for problem in problems:
            print(problem)
        print("THE REBUILD DOES NOT REPRODUCE THE RUNS -- no comparison is reported")
        return 1

    comparison = paired_metric_bootstrap(
        left_scores,
        right_scores,
        gold,
        metric="micro_f1",
        k=k,
        decision={"rule": rule, "k": k, "threshold": threshold, "min_score": min_score},
        resamples=resamples,
    )
    print(f"the rebuild reproduces both runs: {left_id} and {right_id}")
    print(f"  the decision rule: {rule} k={k}")
    print(
        f"  {left_id}: micro-F1 {comparison['left']:.4f}   "
        f"{right_id}: micro-F1 {comparison['right']:.4f}"
    )
    print(
        f"  the paired difference ({left_id} - {right_id}): "
        f"{comparison['delta']:+.4f} "
        f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}] "
        f"(the {resamples} resamples over the texts; the share of texts where the "
        f"left is better: {comparison['share_positive']:.3f})"
    )
    return 0


def main() -> int:
    """The CLI: two run ids, the results directory and the resample count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, help="the two run ids to compare")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    return compare(args.runs[0], args.runs[1], args.results, args.resamples)


if __name__ == "__main__":
    raise SystemExit(main())
