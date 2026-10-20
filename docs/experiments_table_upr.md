# UPR experiment tables

Comparison tables keyed by run — auto-appended by the CLI commands (one file per dataset). The append-only narrative log is `docs/experiments.md` (one row per run, with full revisions/digests). Values come from each run's own artifacts; counts sit next to numbers; "n/a" = not reported by the run.

## UPR — migration

| run_id | date | backend | model | model_version | dim | ctx | kinds | preprocess | n_texts | seconds | cache_hits | cache_misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-embed-e5-base-v2 | 2026-10-20 | hf | intfloat/e5-base-v2 | f52bf8ec8c71 | 768 | 512 | document | raw | 3000 | 5.533 | 3000 | 0 |
| upr-embed-arctic2 | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | document | raw | 3000 | 0.492 | 3000 | 0 |

## UPR — baselines

| run_id | date | split | predictor | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-eval-mf-topk1-val | 2026-10-20 | val | most_frequent | top_k k=1 | 0.0862 | 0.0034 | 0.0828 | 0.015 | 0.0936 | 0.3023 | 0.1854 | 0.6764 | 660 | 0.017 |
| upr-eval-mf-topk2-val | 2026-10-20 | val | most_frequent | top_k k=2 | 0.1413 | 0.0073 | 0.1327 | 0.029 | 0.0936 | 0.3023 | 0.1854 | 0.6764 | 660 | 0.015 |
| upr-eval-mf-topk3-val | 2026-10-20 | val | most_frequent | top_k k=3 | 0.1543 | 0.0100 | 0.1441 | 0.044 | 0.0936 | 0.3023 | 0.1854 | 0.6764 | 660 | 0.015 |
| upr-eval-random-topk1-val | 2026-10-20 | val | random | top_k k=1 | 0.0183 | 0.0170 | 0.0172 | 1.000 | 0.0294 | 0.1496 | 0.0533 | 0.7792 | 660 | 0.04 |
| upr-eval-random-topk2-val | 2026-10-20 | val | random | top_k k=2 | 0.0304 | 0.0248 | 0.0290 | 1.000 | 0.0294 | 0.1496 | 0.0533 | 0.7792 | 660 | 0.019 |
| upr-eval-random-topk3-val | 2026-10-20 | val | random | top_k k=3 | 0.0346 | 0.0294 | 0.0333 | 1.000 | 0.0294 | 0.1496 | 0.0533 | 0.7792 | 660 | 0.019 |
| upr-eval-mf-topk2-test | 2026-10-20 | test | most_frequent | top_k k=2 | 0.1433 | 0.0074 | 0.1283 | 0.029 | 0.0940 | 0.2954 | 0.1762 | 0.6808 | 664 | 0.015 |
| upr-eval-random-topk2-test | 2026-10-20 | test | random | top_k k=2 | 0.0234 | 0.0196 | 0.0209 | 1.000 | 0.0274 | 0.1438 | 0.0463 | 0.7308 | 664 | 0.019 |
