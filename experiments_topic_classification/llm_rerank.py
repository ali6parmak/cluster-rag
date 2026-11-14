"""TC-18 = menu item 12: the LLM rerank of the best pack's shortlist (tev1).

One decision-model call per dev document scores the **TC-17 fine-tuned pack's
top-64** candidate titles as 64 ``noul`` questions ("is the document about the
following topic?"); the candidates are re-ordered by those probabilities and
the ranking is evaluated exactly like every other arm.  Stage 1 is the real
pairing; only if it clears the pre-registered ordering bar does Stage 2 run —
the **label-blind counterpart**, the same 64 questions with another document's
text as the state (a seeded derangement), to certify that the gain is
label-driven.

Pre-registered in ``docs/topic_classification.md`` section 25 before this script
runs.  Artifact: ``results/eurlex-llm-rerank/metrics.json``.  Splits: train =
fit (the TC-17 pack), dev = evaluation (6,000); test is never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))
sys.path.insert(0, str(ROOT / "experiments"))

from audit_eurlex import title_skeleton  # noqa: E402
from candidate_pack import bootstrap_ranked, slice_view  # noqa: E402
from decision_pool import DECISION_MODELS  # noqa: E402
from shortlist_baselines import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    full_text,
    label_buckets,
    load_split,
    near_duplicate_cosines,
)

from cluster_rag.decision import CachedDecider, SystemOneDecider  # noqa: E402
from cluster_rag.text_processors.decision.probe import (  # noqa: E402
    CHARS_PER_TOKEN,
    DecisionCell,
    plan_calls,
)

OUT = ROOT / "results" / "eurlex-llm-rerank"
RUN_ID = "eurlex-llm-rerank"
TC17_ARTIFACT = ROOT / "results" / "eurlex-embed-finetune" / "metrics.json"
TC17_RANKING = ROOT / "results" / "eurlex-embed-finetune" / "ranked_dev.npz"
PROMPT_VERSION = "eurlex-tc18-v1"
TASK = "Is the document about the following topic?"
MODEL = "tev1"
CANDIDATES = 64
MAX_INPUT_TOKENS = 2_500
DEPTH = 50
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
ORDERING_BAR = 0.01
SHORTLIST_READING = 0.005
LOSS_TOLERANCE = 0.01
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")


def load_data() -> dict[str, Any]:
    """The dev side, the pack's ranking and the three slices."""
    train_records = load_split("train")
    dev_records = load_split("dev")
    all_records = train_records + dev_records
    labels = sorted({str(label) for record in all_records for label in record["eurovoc_concepts"]})
    label_index = {label: index for index, label in enumerate(labels)}
    train_labels = [
        [str(label) for label in record["eurovoc_concepts"]] for record in train_records
    ]
    dev_labels = [[str(label) for label in record["eurovoc_concepts"]] for record in dev_records]
    dev_gold = [{label_index[label] for label in record_labels} for record_labels in dev_labels]
    bucket_of_label, bucket_counts = label_buckets(labels, train_labels)

    vocabulary = {
        str(record["id"]): record["title"]
        for record in (
            json.loads(line)
            for line in (ROOT / "data" / "eurlex_4k" / "eurovoc_concepts.jsonl").open(
                encoding="utf-8"
            )
        )
    }
    label_titles = [vocabulary[label] for label in labels]
    dev_texts = [full_text(record) for record in dev_records]

    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array(
        [title_skeleton(record["title"]) not in train_families for record in dev_records]
    )
    max_cosines = near_duplicate_cosines(train_records, dev_records)
    clean = max_cosines < NEAR_DUP_THRESHOLD
    slices = {
        "all_dev": np.ones(len(dev_records), dtype=bool),
        "family_new": family_new,
        "family_new_clean": family_new & clean,
    }
    packed = np.load(TC17_RANKING)
    ranked = np.asarray(packed["ranked"], dtype=np.int32)
    if ranked.shape != (len(dev_records), len(labels)):
        raise ValueError(f"pack ranking shape {ranked.shape} does not match the dev rows")
    return {
        "train_records": train_records,
        "dev_records": dev_records,
        "labels": labels,
        "label_titles": label_titles,
        "dev_texts": dev_texts,
        "dev_gold": dev_gold,
        "bucket_of_label": bucket_of_label,
        "bucket_counts": bucket_counts,
        "slices": slices,
        "pack_ranked": ranked,
    }


def build_cells(
    data: dict[str, Any], *, derangement: np.ndarray | None = None
) -> list[DecisionCell]:
    """The (document, candidate) cells; the blind pass swaps the state's document."""
    pack_ranked = data["pack_ranked"]
    cells: list[DecisionCell] = []
    for row in range(len(data["dev_records"])):
        state_text = (
            data["dev_texts"][int(derangement[row])]
            if derangement is not None
            else data["dev_texts"][row]
        )
        gold = data["dev_gold"][row]
        for position, candidate in enumerate(pack_ranked[row, :CANDIDATES]):
            cells.append(
                DecisionCell(
                    cell_id=f"eurlex-{row}-{position}",
                    group="gold" if int(candidate) in gold else "background",
                    document="eurlex",
                    paragraph_row=row,
                    question_col=position,
                    paragraph_text=state_text,
                    question_text=data["label_titles"][int(candidate)],
                    label=int(int(candidate) in gold),
                )
            )
    return cells


def plan(
    cells: list[DecisionCell],
) -> list[Any]:
    """One call per document with its 64 questions, the document cut to the budget.

    ``plan_calls``' own chunker tests the *untruncated* document against the
    token budget, so a 64-question call over a long document is split even
    though the same call would truncate the document anyway (measured: 31,993
    calls for the 6,000 documents). The grouping is therefore done without the
    budget (``max_questions_per_call`` still caps every call at 64) and the
    planner's own truncation rule — the budget in characters at
    ``CHARS_PER_TOKEN`` minus the rendered questions — is applied per call.
    Only the document is ever cut; the questions and the task are not.
    """
    calls = plan_calls(
        cells,
        state_shape="paragraph",
        instructions=TASK,
        question_type="noul",
        max_questions_per_call=CANDIDATES,
        max_input_tokens=None,
    )
    budget = int(MAX_INPUT_TOKENS * CHARS_PER_TOKEN)
    for call in calls:
        fixed = sum(len(spec.instructions) for spec in call.specs)
        keep = max(budget - fixed, 0)
        call.state["paragraph"] = call.state["paragraph"][:keep]
    return calls


def run_calls(
    decider: CachedDecider,
    calls: list[Any],
    cells: list[DecisionCell],
    *,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute the plan; return one probability per cell and the call diagnostics."""
    probabilities = np.full(len(cells), np.nan, dtype=np.float64)
    truncated = 0
    started = time.perf_counter()
    for number, call in enumerate(calls):
        answers = decider.decide(
            call.state, dict(zip(call.question_names, call.specs, strict=True))
        )
        for cell_index, name in zip(call.cell_indexes, call.question_names, strict=True):
            answer = answers[name]
            if answer.value is None:
                raise RuntimeError(f"{label}: no probability for cell {cells[cell_index].cell_id}")
            probabilities[cell_index] = float(answer.value)
        paragraph = call.state["paragraph"]
        if len(paragraph) < len(cells[call.cell_indexes[0]].paragraph_text):
            truncated += 1
        if (number + 1) % 500 == 0:
            elapsed = time.perf_counter() - started
            rate = (number + 1) / elapsed
            print(
                f"  {label}: {number + 1}/{len(calls)} calls, "
                f"{rate * 60:.0f} calls/min, {elapsed / 60:.1f} min",
                flush=True,
            )
    if np.isnan(probabilities).any():
        raise RuntimeError(f"{label}: {int(np.isnan(probabilities).sum())} cells unanswered")
    diagnostics = {
        "calls": len(calls),
        "truncated_calls": int(truncated),
        "cache_hits": int(decider.cache_hits),
        "cache_misses": int(decider.cache_misses),
        "usage": decider.usage,
        "seconds": round(time.perf_counter() - started, 1),
    }
    return probabilities, diagnostics


def rerank(data: dict[str, Any], probabilities: np.ndarray) -> np.ndarray:
    """Order each document's 64 candidates by probability, then the pack's rest."""
    pack_ranked = data["pack_ranked"]
    reranked = np.empty_like(pack_ranked)
    for row in range(len(data["dev_records"])):
        start = row * CANDIDATES
        candidates = pack_ranked[row, :CANDIDATES]
        scores = probabilities[start : start + CANDIDATES]
        order = np.lexsort((np.arange(CANDIDATES), -scores))
        reranked[row, :CANDIDATES] = candidates[order]
        reranked[row, CANDIDATES:] = pack_ranked[row, CANDIDATES:]
    return reranked


def compare(
    baseline: np.ndarray,
    treatment: np.ndarray,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Paired bootstrap of the rerank minus the pack on every slice and metric."""
    return {
        name: {
            "r50": bootstrap_ranked(baseline, treatment, data["dev_gold"], mask, metric="r50"),
            "r_precision": bootstrap_ranked(
                baseline, treatment, data["dev_gold"], mask, metric="r_precision"
            ),
            "ndcg": bootstrap_ranked(baseline, treatment, data["dev_gold"], mask, metric="ndcg"),
        }
        for name, mask in data["slices"].items()
    }


def evaluate(ranked: np.ndarray, data: dict[str, Any]) -> dict[str, Any]:
    """All primary/secondary metrics on the three slices."""
    return {
        name: slice_view(ranked, data["dev_gold"], mask, data["bucket_of_label"])
        for name, mask in data["slices"].items()
    }


def main() -> None:
    """Run Stage 1 (and Stage 2 only on a positive) and write the artifact."""
    started = time.perf_counter()
    data = load_data()
    slices = data["slices"]
    print(
        f"dev {len(data['dev_records'])}, labels {len(data['labels'])}, slices "
        + ", ".join(f"{name} {int(mask.sum())}" for name, mask in data["slices"].items()),
        flush=True,
    )
    baseline = evaluate(data["pack_ranked"], data)
    reference = json.loads(TC17_ARTIFACT.read_text(encoding="utf-8"))
    reference_metrics = reference["methods"]["finetuned_pack"]["slices"]["family_new"]["primary"]
    reproduced = {
        metric: {
            "tc17": reference_metrics[metric],
            "baseline": baseline["family_new"]["primary"][metric],
            "delta": round(
                baseline["family_new"]["primary"][metric] - reference_metrics[metric], 9
            ),
        }
        for metric in ("recall@50", "r_precision", "ndcg@10")
    }
    # The candidate set is regenerated on the GPU (the TC-17 fold models), so it is
    # not bit-exact; the pre-registered tolerance is 0.005 on every metric.
    if any(abs(entry["delta"]) > 0.005 for entry in reproduced.values()):
        raise ValueError(f"the pack baseline does not reproduce TC-17: {reproduced}")

    config = DECISION_MODELS[MODEL]
    decider = CachedDecider(
        SystemOneDecider(config, timeout=600.0),
        ROOT / "cache",
        prompt_version=PROMPT_VERSION,
    )
    print(f"model {decider.info.model_id} digest {decider.info.digest}", flush=True)

    cells = build_cells(data)
    calls = plan(cells)
    print(
        f"Stage 1: {len(cells)} cells in {len(calls)} calls "
        f"({len(cells) / len(calls):.1f} questions/call)",
        flush=True,
    )
    probabilities, stage1_calls = run_calls(decider, calls, cells, label="stage1")
    stage1_ranked = rerank(data, probabilities)
    stage1_metrics = evaluate(stage1_ranked, data)
    comparison = compare(data["pack_ranked"], stage1_ranked, data)
    primary = comparison["family_new"]
    ordering_conditions = {
        "ndcg_gain_at_least_0.01": primary["ndcg"]["delta"] >= ORDERING_BAR,
        "ndcg_ci_excludes_zero": primary["ndcg"]["bootstrap"]["ci_low"] > 0.0,
        "r50_loss_at_most_0.01": primary["r50"]["delta"] >= -LOSS_TOLERANCE,
    }
    shortlist_reading = {
        "r50_gain_at_least_0.005": primary["r50"]["delta"] >= SHORTLIST_READING,
        "r50_ci_excludes_zero": primary["r50"]["bootstrap"]["ci_low"] > 0.0,
    }
    stage1_pass = all(ordering_conditions.values())

    def coverage_at_64(ranked: np.ndarray) -> float:
        return round(
            float(
                np.mean(
                    [
                        bool(set(ranked[row, :CANDIDATES].tolist()) & data["dev_gold"][row])
                        for row in range(len(data["dev_records"]))
                    ]
                )
            ),
            6,
        )

    artifact: dict[str, Any] = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-18 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 12)",
        "config": {
            "model": MODEL,
            "model_id": decider.info.model_id,
            "digest": decider.info.digest,
            "prompt_version": PROMPT_VERSION,
            "task": TASK,
            "question_type": "noul",
            "question_text": "the candidate label's own EUROVOC title",
            "candidate_set": (
                f"the TC-17 fine-tuned pack's dev top-{CANDIDATES}, exported by "
                "experiments_topic_classification/export_tc17_ranking.py"
            ),
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_questions_per_call": CANDIDATES,
            "state": "the document's full_text (title + header + recitals + main body)",
            "rerank": (
                "candidates by probability desc (ties by the pack order), then the pack's rest"
            ),
            "stage2_derangement": (
                "numpy.random.default_rng(42), a permutation with no fixed point, "
                "the state's document replaced"
            ),
            "slices": list(SLICE_NAMES),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed": SEED,
        },
        "slices": {"documents": {name: int(mask.sum()) for name, mask in slices.items()}},
        "candidate_set": {
            "coverage@64": coverage_at_64(data["pack_ranked"]),
            "labels": len(data["labels"]),
        },
        "reproduction": reproduced,
        "baseline": {
            "kind": "TC-17 fine-tuned pack (the candidate set's own ranking)",
            "slices": baseline,
        },
        "stage1": {
            "label_driven_check": "the real pairing",
            "calls": stage1_calls,
            "metrics": stage1_metrics,
            "comparison": comparison,
            "conditions": {
                "ordering": ordering_conditions,
                "shortlist_reading": shortlist_reading,
            },
            "passed": stage1_pass,
        },
    }

    if not stage1_pass:
        artifact["stage2"] = {
            "status": "skipped",
            "reason": "Stage 1 did not clear the pre-registered ordering bar",
        }
        artifact["verdict"] = {
            "outcome": "null",
            "stage1_ordering_conditions": ordering_conditions,
            "family_new_r50_delta": primary["r50"]["delta"],
            "family_new_r_precision_delta": primary["r_precision"]["delta"],
            "family_new_ndcg@10_delta": primary["ndcg"]["delta"],
            "family_new_ndcg@10_95ci": [
                primary["ndcg"]["bootstrap"]["ci_low"],
                primary["ndcg"]["bootstrap"]["ci_high"],
            ],
            "shortlist_reading": shortlist_reading,
        }
    else:
        print("Stage 1 cleared the bar; running the label-blind counterpart ...", flush=True)
        rng = np.random.default_rng(SEED)
        while True:
            permutation = rng.permutation(len(data["dev_records"]))
            if not np.any(permutation == np.arange(len(data["dev_records"]))):
                break
        blind_cells = build_cells(data, derangement=permutation)
        blind_calls = plan(blind_cells)
        print(f"Stage 2: {len(blind_cells)} cells in {len(blind_calls)} calls", flush=True)
        blind_probabilities, stage2_calls = run_calls(
            decider, blind_calls, blind_cells, label="stage2"
        )
        blind_ranked = rerank(data, blind_probabilities)
        blind_metrics = evaluate(blind_ranked, data)
        blind_comparison = compare(data["pack_ranked"], blind_ranked, data)
        blind_primary = blind_comparison["family_new"]
        reproduced_gain = (
            blind_primary["ndcg"]["delta"] >= 0.5 * primary["ndcg"]["delta"]
            and blind_primary["ndcg"]["bootstrap"]["ci_low"] > 0.0
        )
        artifact["stage2"] = {
            "label_driven_check": "the state's document replaced (seeded derangement)",
            "calls": stage2_calls,
            "metrics": blind_metrics,
            "comparison": blind_comparison,
            "derangement": {
                "seed": SEED,
                "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
            },
            "reproduced_gain": bool(reproduced_gain),
        }
        artifact["verdict"] = {
            "outcome": "null" if reproduced_gain else "win",
            "stage1_ordering_conditions": ordering_conditions,
            "stage2_certified": not reproduced_gain,
            "family_new_r50_delta": primary["r50"]["delta"],
            "family_new_r_precision_delta": primary["r_precision"]["delta"],
            "family_new_ndcg@10_delta": primary["ndcg"]["delta"],
            "family_new_ndcg@10_95ci": [
                primary["ndcg"]["bootstrap"]["ci_low"],
                primary["ndcg"]["bootstrap"]["ci_high"],
            ],
            "blind_family_new_ndcg@10_delta": blind_primary["ndcg"]["delta"],
            "shortlist_reading": shortlist_reading,
        }

    artifact["cost"] = {
        "seconds_total": round(time.perf_counter() - started, 1),
        "llm_calls": stage1_calls["calls"]
        + (artifact["stage2"].get("calls", {}).get("calls", 0) if stage1_pass else 0),
        "stage1_seconds": stage1_calls["seconds"],
        "stage2_seconds": (
            artifact["stage2"].get("calls", {}).get("seconds", 0.0) if stage1_pass else 0.0
        ),
    }
    artifact["notes"] = {
        "leakage": (
            "the pack was fitted on train; dev is evaluation only; the decision model is "
            "zero-shot and the prompts carry only the document text and the candidate titles, "
            "never a gold indication"
        ),
        "bar": (
            "family_new nDCG@10 +0.01 over the TC-17 pack with the interval excluding 0 and "
            "R@50 loss <= 0.01; Stage 2 certifies the gain is not the model's label prior"
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "family_new": {
                    "pack": baseline["family_new"]["primary"],
                    "rerank": stage1_metrics["family_new"]["primary"],
                },
                "comparison": primary,
                "verdict": artifact["verdict"],
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
