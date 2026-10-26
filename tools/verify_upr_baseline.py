"""Verify that a run's whole-text reference arm reproduces the ``evaluate`` run.

The text-processing track's UPR guard: every ``analyze-upr`` run scores the
**whole-text reference arm** — the main pipeline's own scoring of the split,
with the same predictor, the same embedding runs and the same decision rule —
and that arm must reproduce the matching ``evaluate`` run's report **field by
field** (the overall section, the per-bucket sections, the record count and
the label count). A mismatch means the run's numbers cannot be compared with
the pipeline's.

The tool compares two *artifacts* (never the raw dataset):

    results/upr-eval-<...>/metrics.json      (the `evaluate` run, the reference)
    results/<run_id>/analysis.json           (the `analyze-upr` run to check)

The reference run is taken from the analysis artifact's own ``guard`` block
(the config's ``upr_analysis.reference_eval_run``), so the tool checks what
the run itself claims to have checked — and a run without a guard block is
reported as unchecked, not as passing.

Usage:

    uv run python tools/verify_upr_baseline.py upr-analyze-champion
    uv run python tools/verify_upr_baseline.py "upr-analyze-*"

Exit code 0 when every checked run reproduces its reference, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_RESULTS = Path("results")


def load_artifact(results: Path, run_id: str, name: str) -> dict[str, Any]:
    """One run's artifact.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    path = results / run_id / name
    if not path.is_file():
        raise FileNotFoundError(f"no {name} for {run_id}: {path}")
    return json.loads(path.read_text("utf-8"))


def compare_reports(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """The differences between the two reports (empty = identical)."""
    problems: list[str] = []
    if reference.get("overall") != candidate.get("overall"):
        problems.append("the overall report differs")
    if reference.get("per_bucket") != candidate.get("per_bucket"):
        problems.append("the per-bucket report differs")
    for key in ("n_records", "n_labels"):
        if reference.get(key) != candidate.get(key):
            problems.append(f"{key} differs")
    return problems


def check_run(results: Path, run_id: str) -> list[str]:
    """The problems of one run (empty = the reference arm reproduces the run).

    Raises:
        FileNotFoundError: If one of the artifacts is missing.
    """
    analysis = load_artifact(results, run_id, "analysis.json")
    guard = analysis.get("guard")
    if not guard:
        return [f"{run_id}: no guard block — the run did not check its reference arm"]
    reference_run = guard.get("reference_run")
    if not reference_run:
        return [f"{run_id}: the guard block names no reference run"]
    metrics = load_artifact(results, reference_run, "metrics.json")
    arms = analysis.get("arms", {})
    reference_arm = arms.get("full_text")
    if reference_arm is None:
        return [f"{run_id}: the artifact has no 'full_text' reference arm"]
    problems = compare_reports(metrics.get("report", {}), reference_arm.get("metrics", {}))
    return [f"[{reference_run}] {problem}" for problem in problems]


def expand(results: Path, patterns: list[str]) -> list[str]:
    """The run ids of the patterns (a glob is matched against the run folders)."""
    run_ids: list[str] = []
    for pattern in patterns:
        if any(character in pattern for character in "*?["):
            run_ids.extend(
                sorted(
                    path.name
                    for path in results.glob(pattern)
                    if (path / "analysis.json").is_file()
                )
            )
        else:
            run_ids.append(pattern)
    return run_ids


def main() -> int:
    """Check every requested run and report the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run ids (globs allowed)")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="the results folder (default: results)",
    )
    args = parser.parse_args()

    failures = 0
    checked = 0
    for run_id in expand(args.results, args.runs):
        checked += 1
        try:
            problems = check_run(args.results, run_id)
        except FileNotFoundError as exc:
            problems = [str(exc)]
        status = "OK" if not problems else f"FAIL ({len(problems)} problems)"
        print(f"{run_id}: {status}")
        for problem in problems[:10]:
            print(f"  - {problem}")
        failures += len(problems)
    print(
        f"{checked} run(s) checked; the whole-text reference arm reproduces the evaluate runs"
        if not failures
        else "MISMATCH"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
