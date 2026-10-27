"""A throwaway smoke script for the **reranker shapes** (NOT a harness run).

Three shapes have never been exercised on this machine, and the real run
(CS-15) is 9 full-val runs: the **cross-encoder** pair, the **yes/no head**
pair (one sequence per pair, the card's template) and the **listwise**
request (one query, many documents). The script therefore checks, on a
handful of pairs:

- the model loads and scores (the pinned revision, the custom modelling
  code, the shim);
- the **query/document order** matters (both orders are scored, so the
  choice is a measurement and not a belief);
- the **batch size** does not change the scores (a determinism check);
- a **long text** is truncated visibly (never silently).

It writes no table row, no run folder and no number: a feasibility check,
not a measurement.

Usage:

    uv run python tools/smoke_upr_rerank.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cluster_rag.cli import _load_sweep_runs
from cluster_rag.rerank.base import (
    KIND_LISTWISE,
    listwise_requests,
    pair_requests,
)
from cluster_rag.rerank.hf import HFReranker

SWEEP = "experiments_text_processing/upr_rerank_sweep.py"
DATA = Path("data/raw/upr.jsonl")


def _sample() -> tuple[list[str], list[str]]:
    """Three texts and three labels (the smoke test's sample)."""
    records = [json.loads(line) for line in DATA.read_text("utf-8").splitlines() if line]
    texts = [record["text"] for record in records[:3]]
    labels = sorted({label for record in records for label in record["labels"]})[:3]
    return texts, labels


def main() -> int:
    """Run the smoke test and print what each shape does."""
    runs = _load_sweep_runs(Path(SWEEP))
    config = runs[0].config
    models = config.upr_analysis.rerank.models
    texts, labels = _sample()
    documents = [f"This recommendation is about {label}." for label in labels]
    longest = max(texts, key=len)

    for model in models:
        client = HFReranker(model)
        print(f"\n[{model.name}] {model.model_id} @ {model.revision[:12]} ({model.kind})")
        if model.kind == KIND_LISTWISE:
            requests = listwise_requests(texts, [documents] * len(texts))
            scores = client.score(requests)
            for text, row in zip(texts, scores, strict=True):
                print(f"  {text[:40]!r} -> {[round(value, 4) for value in row]}")
        else:
            for order, queries, docs in (
                ("text_first", texts, documents),
                ("label_first", documents, texts),
            ):
                requests = pair_requests(queries, [docs] * len(queries))
                scores = client.score(requests)
                flat = [value for row in scores for value in row]
                print(f"  {order}: {[round(value, 4) for value in flat]}")
        # The batch size must not change the scores.
        if model.kind != KIND_LISTWISE:
            small = HFReranker(model.model_copy(update={"batch_size": 1}))
            requests = pair_requests(texts, [documents] * len(texts))
            one = [value for row in small.score(requests) for value in row]
            many = [value for row in client.score(requests) for value in row]
            same = all(abs(a - b) < 1e-4 for a, b in zip(one, many, strict=True))
            print(f"  the batch size does not change the scores: {same}")
        # A long text must be truncated visibly.
        before = dict(client.usage)
        long_requests = (
            listwise_requests([longest], [documents])
            if model.kind == KIND_LISTWISE
            else pair_requests([longest], [documents])
        )
        client.score(long_requests)
        after = client.usage
        print(
            "  the longest of the three texts: "
            f"{after['n_truncated_inputs'] - before.get('n_truncated_inputs', 0)} truncated "
            f"inputs (the count is reported, never silent)"
        )
        usage = client.usage
        print(
            f"  cost: {usage['n_pairs']:.0f} pairs, {usage['n_calls']:.0f} forward "
            f"passes, {usage['seconds']:.2f}s, {usage['peak_gpu_mb']:.0f} MB of peak VRAM"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
