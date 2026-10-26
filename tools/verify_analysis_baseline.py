"""Verify that a run's full-text baseline arm reproduces the Stage-0 artifact.

The text-processing track's guard: every new run of the ``analyze-omct``
command scores the **full-text baseline arm** (the Stage-0 scoring) beside its
new arms, and that arm must reproduce the Stage-0 artifact field by field —
the same control cells in the same order, the same per-pair and per-cell
scores and rank percentiles, the same AUCs, score distributions and
negative-enrichment numbers. A mismatch means the run's numbers cannot be
compared with the Stage-0 ones (or with the other runs').

The tool compares two *artifacts* (never the raw pair file):

    results/omct-analyze-<model>/analysis.json      (Stage 0, the reference)
    results/<run_id>/analysis.json                   (the run to check)

Usage:

    uv run python tools/verify_analysis_baseline.py omct-analyze-arctic2-words15
    uv run python tools/verify_analysis_baseline.py "omct-analyze-*-words*"

Exit code 0 when every checked run reproduces the reference, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def compare_models(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """The differences between two models' baseline arms (empty = identical)."""
    problems: list[str] = []

    # 1. the control cells: the same ids, the same order
    ids_reference = [cell["cell_id"] for cell in reference["control_cells"]]
    ids_candidate = [cell["cell_id"] for cell in candidate["control_cells"]]
    if ids_reference != ids_candidate:
        problems.append("control cell ids differ")
    documents_reference = [cell["document"] for cell in reference["control_cells"]]
    documents_candidate = [cell["document"] for cell in candidate["control_cells"]]
    if documents_reference != documents_candidate:
        problems.append("control cell documents differ")

    # 2. the judged pairs
    pairs_reference = {pair["pair_id"]: pair for pair in reference["judged_pairs"]}
    pairs_candidate = {pair["pair_id"]: pair for pair in candidate["judged_pairs"]}
    if set(pairs_reference) != set(pairs_candidate):
        problems.append("judged pair ids differ")
    for pair_id, pair in pairs_reference.items():
        other = pairs_candidate.get(pair_id)
        if other is None:
            continue
        if pair["score"] != other["score"]:
            problems.append(f"pair score differs: {pair_id}")
        if pair["rank_percentile"] != other["rank_percentile"]:
            problems.append(f"pair rank percentile differs: {pair_id}")

    # 3. the control cells' scores
    cells_reference = {cell["cell_id"]: cell for cell in reference["control_cells"]}
    cells_candidate = {cell["cell_id"]: cell for cell in candidate["control_cells"]}
    for cell_id, cell in cells_reference.items():
        other = cells_candidate.get(cell_id)
        if other is None:
            continue
        if cell["score"] != other["score"]:
            problems.append(f"control score differs: {cell_id}")
        if cell["rank_percentile"] != other["rank_percentile"]:
            problems.append(f"control rank percentile differs: {cell_id}")

    # 4. the aggregate numbers
    if reference["pair_auc"]["overall"] != candidate["pair_auc"]["overall"]:
        problems.append("pair_auc.overall differs")
    if reference["pair_auc"]["by_document"] != candidate["pair_auc"]["by_document"]:
        problems.append("pair_auc.by_document differs")
    for group in ("label1", "label0", "control"):
        if reference["score_distributions"][group] != candidate["score_distributions"][group]:
            problems.append(f"score_distributions.{group} differs")
    if reference["negative_enrichment"] != candidate["negative_enrichment"]:
        problems.append("negative_enrichment differs")
    if reference["top1_candidate_provenance"] != candidate["top1_candidate_provenance"]:
        problems.append("top1_candidate_provenance differs")
    return problems


def reference_run_id(model_name: str) -> str:
    """The Stage-0 run of a model (the reference of the guard)."""
    return f"omct-analyze-{model_name}"


def check_run(results: Path, run_id: str) -> list[str]:
    """The problems of one run (empty = the baseline arm reproduces Stage 0)."""
    artifact = load_artifact(results, run_id)
    problems: list[str] = []
    for model in artifact.get("models", []):
        if model.get("kind") == "decision":
            # The decision arm (CS-7b) is not an embedding model: it has no
            # Stage-0 full-text baseline arm to reproduce (it is compared with
            # the cosine inside the run, as an arm).
            continue
        name = model.get("name", "unknown")
        try:
            reference = load_artifact(results, reference_run_id(name))["models"][0]
        except FileNotFoundError as exc:
            problems.append(str(exc))
            continue
        problems.extend(f"[{name}] {problem}" for problem in compare_models(reference, model))
    return problems


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
        problems = check_run(args.results, run_id)
        status = "OK" if not problems else f"FAIL ({len(problems)} problems)"
        print(f"{run_id}: {status}")
        for problem in problems[:10]:
            print(f"  - {problem}")
        failures += len(problems)
    print(
        f"{checked} run(s) checked; the full-text baseline arm reproduces the Stage-0 artifacts"
        if not failures
        else "MISMATCH"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
