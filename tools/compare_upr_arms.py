"""Compare two UPR arms that live in **two different runs** (paired).

The text-processing track's UPR side: two arms of the same question often
live in two different runs (the catalog's arms use a different label side, the
decision-feature arm uses a different pack), so their score matrices are not
in memory together. The runs' artifacts carry each arm's **per-text** summary
(the decision view's (tp, fp, fn) counts and the ranking view's per-text
nDCG), and that is enough for a **paired** comparison: the resampling unit is
the text, so both arms are judged on the same resampled texts.

The tool reads two artifacts (never the raw dataset):

    results/<left_run_id>/analysis.json
    results/<right_run_id>/analysis.json

Usage:

    uv run python tools/compare_upr_arms.py <left_run_id> <right_run_id> \\
        --left-arm learned_histgb --right-arm learned_histgb

Exit code 0 when the comparison was made (the interval's sign is a result,
not a failure), 1 on a usage or data error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cluster_rag.eval.multilabel import paired_counts_bootstrap, paired_values_bootstrap

DEFAULT_RESULTS = Path("results")


def load_artifact(results: Path, run_id: str) -> dict[str, Any]:
    """One run's artifact (``analysis.json`` or ``decisions.json``).

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    for name in ("analysis.json", "decisions.json"):
        path = results / run_id / name
        if path.is_file():
            return json.loads(path.read_text("utf-8"))
    raise FileNotFoundError(f"no analysis.json or decisions.json for {run_id}: {results / run_id}")


def arm_block(artifact: dict[str, Any], arm: str) -> dict[str, Any]:
    """One arm's block of an artifact.

    Raises:
        KeyError: If the artifact has no such arm.
    """
    arms = artifact.get("arms", {})
    if arm not in arms:
        raise KeyError(f"the artifact has no arm {arm!r}; it has {sorted(arms)}")
    return arms[arm]


def _counts(block: dict[str, Any], bucket: str | None) -> np.ndarray:
    """The per-text counts of an arm (overall or of one label bucket)."""
    per_text = block.get("per_text")
    if not per_text:
        raise KeyError("the arm has no per_text block (the run predates it)")
    if bucket is None:
        rows = per_text["f1_counts"]
    else:
        by_bucket = per_text.get("by_bucket", {})
        if bucket not in by_bucket:
            raise KeyError(f"the arm has no bucket {bucket!r}; it has {sorted(by_bucket)}")
        rows = by_bucket[bucket]["f1_counts"]
    return np.asarray(rows, dtype=np.float64)


def _values(block: dict[str, Any], bucket: str | None) -> np.ndarray:
    """The per-text nDCG values of an arm (overall or of one label bucket)."""
    per_text = block.get("per_text")
    if not per_text:
        raise KeyError("the arm has no per_text block (the run predates it)")
    if bucket is None:
        payload = per_text["ndcg"]
    else:
        by_bucket = per_text.get("by_bucket", {})
        if bucket not in by_bucket:
            raise KeyError(f"the arm has no bucket {bucket!r}; it has {sorted(by_bucket)}")
        payload = by_bucket[bucket]["ndcg"]
    return np.asarray(
        [np.nan if value is None else float(value) for value in payload["values"]],
        dtype=np.float64,
    )


def compare(
    left_block: dict[str, Any],
    right_block: dict[str, Any],
    *,
    buckets: list[str],
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """The paired comparison of two arms (pooled and per label bucket).

    Returns:
        The block: the micro-F1 and the nDCG comparisons, pooled and per
        bucket.
    """
    result: dict[str, Any] = {"pooled": {}, "per_bucket": {}}
    scopes: list[tuple[str, str | None]] = [("overall", None)]
    scopes.extend((bucket, bucket) for bucket in buckets)
    for bucket in buckets:
        f1 = paired_counts_bootstrap(
            _counts(left_block, bucket),
            _counts(right_block, bucket),
            resamples=resamples,
            seed=seed,
        )
        ndcg = paired_values_bootstrap(
            _values(left_block, bucket),
            _values(right_block, bucket),
            resamples=resamples,
            seed=seed,
        )
        result["per_bucket"][bucket] = {"micro_f1": f1, "ndcg": ndcg}
    f1 = paired_counts_bootstrap(
        _counts(left_block, None),
        _counts(right_block, None),
        resamples=resamples,
        seed=seed,
    )
    ndcg = paired_values_bootstrap(
        _values(left_block, None),
        _values(right_block, None),
        resamples=resamples,
        seed=seed,
    )
    result["pooled"] = {"micro_f1": f1, "ndcg": ndcg}
    return result


def main() -> int:
    """Compare the two arms and print the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_run_id")
    parser.add_argument("right_run_id")
    parser.add_argument("--left-arm", default="learned_histgb")
    parser.add_argument("--right-arm", default="learned_histgb")
    parser.add_argument("--buckets", nargs="*", default=["head", "torso", "tail", "unseen"])
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    try:
        left_artifact = load_artifact(args.results, args.left_run_id)
        right_artifact = load_artifact(args.results, args.right_run_id)
        left_block = arm_block(left_artifact, args.left_arm)
        right_block = arm_block(right_artifact, args.right_arm)
        outcome = compare(
            left_block,
            right_block,
            buckets=list(args.buckets),
            resamples=args.resamples,
            seed=args.seed,
        )
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"{args.left_run_id}[{args.left_arm}] vs {args.right_run_id}[{args.right_arm}]")
    header = (
        f"{'scope':10s} {'metric':10s} {'left':>9s} {'right':>9s} "
        f"{'delta':>9s} {'ci_low':>9s} {'ci_high':>9s}"
    )
    print(header)
    for scope, payload in [("pooled", outcome["pooled"])] + [
        (bucket, outcome["per_bucket"][bucket]) for bucket in outcome["per_bucket"]
    ]:
        for metric, block in payload.items():
            print(
                f"{scope:10s} {metric:10s} {block['left']:9.4f} {block['right']:9.4f} "
                f"{block['delta']:+9.4f} {block['ci_low']:+9.4f} {block['ci_high']:+9.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
