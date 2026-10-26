"""Compare two arms (the paired difference between them).

The Stage-2.6 chunk-header arm is measured against the **no-header arm of the
same run** (the same cells, the same candidate lists, the same labels), so the
comparison is paired by construction. The harness's own ``comparisons`` block
compares every arm with the run's **baseline** (the full-text cosine), so
this tool adds the arm-vs-arm pairing the header's question needs.

With ``--run-b`` the second arm may come from **another run**: the Stage-2.7
late-chunked arm's early-chunked counterpart lives in the same run, but the
*published* early reference is an earlier run, and the two must reproduce
each other exactly (the cross-run reproducibility check).

It reads the run's artifact (never the raw pair file) and reports:

- the two arms' pooled and within-document AUCs and the **paired bootstrap**
  of the difference (the same statistics the harness uses);
- the positives' **hit@1** (the top-1 candidate, read from the per-pair rank
  percentiles) for both arms and the paired counts (the pairs where one arm
  hits and the other does not);
- the positives' median rank percentile for both arms.

Usage:

    uv run python tools/compare_arms_in_run.py <run_id> <arm_a> <arm_b>
    uv run python tools/compare_arms_in_run.py <run_a> <arm_a> <arm_b> --run-b <run_b>

The arms are named as in the artifact (``words25_max``,
``words25_header_max``, ``late_mean_words25_max``, ...).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cluster_rag.eval.metrics import (
    grouped_auc_decomposition,
    paired_auc_bootstrap,
    paired_within_group_auc_bootstrap,
)

DEFAULT_RESULTS = Path("results")


def load_artifact(results: Path, run_id: str) -> dict[str, Any]:
    """One run's analysis artifact.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = results / run_id / "analysis.json"
    if not path.is_file():
        raise FileNotFoundError(f"no analysis artifact for {run_id}: {path}")
    return json.loads(path.read_text("utf-8"))


def find_arm(artifact: dict[str, Any], arm_name: str) -> dict[str, Any]:
    """The arm's block (the baseline arm is the model's own top-level block).

    Raises:
        ValueError: If no arm carries that name.
    """
    for model in artifact.get("models", []):
        arms = model.get("arms") or {}
        if arm_name in arms:
            return arms[arm_name]
        if model.get("name") == arm_name:
            return model
    raise ValueError(f"no arm named {arm_name!r} in the artifact")


def _pairs(arm: dict[str, Any]) -> tuple[list[str], list[int], list[float], list[float]]:
    """The arm's per-pair records: the documents, the labels, the scores, the percentiles."""
    documents: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    percentiles: list[float] = []
    for record in arm.get("judged_pairs", []):
        documents.append(str(record["document"]))
        labels.append(int(record["label"]))
        scores.append(float(record["score"]))
        percentiles.append(float(record["rank_percentile"]))
    return documents, labels, scores, percentiles


def compare(
    artifact_a: dict[str, Any],
    arm_a: str,
    artifact_b: dict[str, Any],
    arm_b: str,
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """The paired comparison of two arms (of one run, or of two runs)."""
    block_a = find_arm(artifact_a, arm_a)
    block_b = find_arm(artifact_b, arm_b)
    documents_a, labels_a, scores_a, percentiles_a = _pairs(block_a)
    documents_b, labels_b, scores_b, percentiles_b = _pairs(block_b)
    if documents_a != documents_b:
        raise ValueError("the two arms do not cover the same pairs")
    labels = np.asarray(labels_a, dtype=np.int64)
    scores_a_array = np.asarray(scores_a, dtype=np.float64)
    scores_b_array = np.asarray(scores_b, dtype=np.float64)
    groups = np.asarray(documents_a)

    paired = paired_auc_bootstrap(
        labels.tolist(),
        scores_a_array.tolist(),
        scores_b_array.tolist(),
        n_resamples=resamples,
        seed=seed,
    )
    within_paired = paired_within_group_auc_bootstrap(
        labels.tolist(),
        scores_a_array.tolist(),
        scores_b_array.tolist(),
        groups.tolist(),
        n_resamples=resamples,
        seed=seed,
    )
    decomposition_a = grouped_auc_decomposition(
        labels.tolist(), scores_a_array.tolist(), groups.tolist()
    )
    decomposition_b = grouped_auc_decomposition(
        labels.tolist(), scores_b_array.tolist(), groups.tolist()
    )

    positive = labels == 1
    hit_a = np.asarray(percentiles_a)[positive] == 0.0
    hit_b = np.asarray(percentiles_b)[positive] == 0.0
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "pooled_auc": {
            "a": paired["auc_reference"],
            "b": paired["auc_arm"],
            "delta": paired["delta_auc"],
            "ci": [paired["ci_low"], paired["ci_high"]],
            "p_delta_positive": paired["p_delta_positive"],
        },
        "within_document_auc": {
            "a": decomposition_a["within_group_auc"],
            "b": decomposition_b["within_group_auc"],
            "delta": within_paired["delta_auc"],
            "ci": [within_paired["ci_low"], within_paired["ci_high"]],
            "p_delta_positive": within_paired["p_delta_positive"],
        },
        "positives_hit1": {
            "a": float(hit_a.mean()) if hit_a.size else None,
            "b": float(hit_b.mean()) if hit_b.size else None,
            "n_positives": int(hit_a.size),
            "a_only": int((hit_a & ~hit_b).sum()),
            "b_only": int((hit_b & ~hit_a).sum()),
            "both": int((hit_a & hit_b).sum()),
            "neither": int((~hit_a & ~hit_b).sum()),
        },
        "positives_rank_p50": {
            "a": float(np.median(np.asarray(percentiles_a)[positive])) if positive.any() else None,
            "b": float(np.median(np.asarray(percentiles_b)[positive])) if positive.any() else None,
        },
        "n_resamples": resamples,
        "seed": seed,
    }


def main() -> int:
    """Compare two arms and print the paired numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("arm_a")
    parser.add_argument("arm_b")
    parser.add_argument(
        "--run-b",
        default=None,
        help="the run of the second arm (default: the first run, i.e. both arms "
        "of one run; a second run makes it a cross-run comparison)",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    artifact_a = load_artifact(args.results, args.run_id)
    run_b = args.run_b or args.run_id
    artifact_b = artifact_a if run_b == args.run_id else load_artifact(args.results, run_b)
    report = compare(
        artifact_a, args.arm_a, artifact_b, args.arm_b, resamples=args.resamples, seed=42
    )
    report["run_a"] = args.run_id
    report["run_b"] = run_b
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
