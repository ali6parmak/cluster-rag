"""Compare two Stage-3 arms with a paired per-pair bootstrap.

The text-processing track's Stage-3 runs hold one arm per run (the
feature-group ablation, the learner comparison, the chunk profile), so two
arms of different runs live in two artifacts. The artifacts carry each
arm's **per-rotation** AUCs and each arm's own paired interval against the
common reference (the full-text cosine), but a **paired per-pair** interval
between the two arms needs their per-pair out-of-fold scores — which the
Stage-3 block now carries (``out_of_fold_scores``).

This tool reads two run artifacts (never the raw pair file) and reports, per
model, the **within-document AUC difference** between the two arms with the
paired bootstrap's percentile interval (the same statistic and the same
resampling as the arm-vs-cosine comparison inside a run: stratified by
document and class, so every document keeps its size and its class balance).

Usage:

    uv run python tools/compare_stage3_arms.py \\
        omct-analyze-embeddinggemma-stage3-cp-base omct-analyze-embeddinggemma-stage3-cp-full

    uv run python tools/compare_stage3_arms.py a b --resamples 2000 --seed 42

Exit code 0 when every comparison runs, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cluster_rag.eval.metrics import paired_within_group_auc_bootstrap

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


def model_scores(artifact: dict[str, Any], model_name: str) -> dict[str, Any]:
    """The judged pairs' labels, documents and the arm's out-of-fold scores.

    Raises:
        KeyError: If the model is not in the artifact, or the artifact has no
            out-of-fold scores (a run from before the Stage-3.3 change set).
    """
    for model in artifact.get("models", []):
        if model.get("name") != model_name:
            continue
        stage3 = model.get("stage3") or {}
        scores = stage3.get("out_of_fold_scores")
        if scores is None:
            raise KeyError(
                f"the run has no out_of_fold_scores block (model {model_name}): "
                "re-run it with the Stage-3.3 code"
            )
        judged = model["judged_pairs"]
        if len(scores["judged_pairs"]) != len(judged):
            raise KeyError("the out-of-fold scores and the judged pairs do not line up")
        return {
            "labels": [int(pair["label"]) for pair in judged],
            "documents": [str(pair["document"]) for pair in judged],
            "scores": [float(value) for value in scores["judged_pairs"]],
        }
    raise KeyError(f"no model {model_name!r} in the artifact")


def compare(
    reference: dict[str, Any],
    arm: dict[str, Any],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """The paired within-document AUC difference between two arms."""
    return paired_within_group_auc_bootstrap(
        reference["labels"],
        reference["scores"],
        arm["scores"],
        reference["documents"],
        n_resamples=resamples,
        seed=seed,
    )


def main() -> int:
    """Compare the two runs and report the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="the reference run id (the first arm)")
    parser.add_argument("arm", help="the compared run id (the second arm)")
    parser.add_argument(
        "--model",
        default=None,
        help="the model to compare (default: every model of the reference run)",
    )
    parser.add_argument(
        "--resamples", type=int, default=2000, help="the bootstrap resamples (default: 2000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="the bootstrap seed (default: 42)")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="the results folder (default: results)",
    )
    args = parser.parse_args()

    reference_artifact = load_artifact(args.results, args.reference)
    arm_artifact = load_artifact(args.results, args.arm)
    names = [
        model["name"]
        for model in reference_artifact.get("models", [])
        if args.model is None or model["name"] == args.model
    ]
    if not names:
        print(f"no model {args.model!r} in {args.reference}")
        return 1
    failures = 0
    for name in names:
        try:
            reference = model_scores(reference_artifact, name)
            arm = model_scores(arm_artifact, name)
        except KeyError as exc:
            print(f"{name}: FAIL ({exc})")
            failures += 1
            continue
        result = compare(reference, arm, resamples=args.resamples, seed=args.seed)
        print(
            f"{name}: within-document AUC {result['auc_reference']:.4f} -> "
            f"{result['auc_arm']:.4f} (delta {result['delta_auc']:+.4f}, 95% CI "
            f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], "
            f"P(delta>0) {result['p_delta_positive']:.3f})"
        )
    print(
        f"{len(names)} model(s) compared: {args.reference} -> {args.arm} "
        f"({args.resamples} resamples, seed {args.seed})"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
