# Experiment tables

Comparison tables keyed by run — one per (phase, command) — auto-appended by
the CLI commands. The append-only narrative log is `docs/experiments.md`
(one row per run, with full revisions/digests). Values come from each run's
own artifacts; counts sit next to numbers; "n/a" = not reported by the run.

## Phase 2 — embed

| run_id | date | backend | model | model_version | dim | ctx | kinds | preprocess | n_texts | seconds | cache_hits | cache_misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| embed-e5-base-v2 | 2026-10-19 | hf | intfloat/e5-base-v2 | f52bf8ec8c71 | 768 | 512 | document | raw | 3000 | 3.467 | 3000 | 0 |
| embed-arctic2 | 2026-10-19 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | document | raw | 3000 | 0.422 | 3000 | 0 |

## Phase 2 — probe-aux

| run_id | date | backend | model | model_version | dim | ctx | transform | pair_auc | same-doc hit@1 | same-doc hit@10 | same-doc recall@10 | paras>ctx | n_paras | n_pairs(pos) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aux-probe2-minilm | 2026-10-19 | hf | sentence-transformers/all-MiniLM-L6-v2 | c9745ed1d9f2 | 384 | 256 | strip_block | 0.4526 | 0.381 | 0.748 | 0.508 | 1.8% | 225 | 1520 (524) | 0.272 |
| aux-probe2-e5-base-v2 | 2026-10-19 | hf | intfloat/e5-base-v2 | f52bf8ec8c71 | 768 | 512 | strip_block | 0.5793 | 0.374 | 0.774 | 0.540 | 0.0% | 225 | 1520 (524) | 0.814 |
| aux-probe2-bge-m3 | 2026-10-19 | hf | BAAI/bge-m3 | 9a0624b896d8 | 1024 | 8192 | strip_block | 0.4687 | 0.361 | 0.748 | 0.476 | 0.0% | 225 | 1520 (524) | 2.215 |
| aux-probe2-arctic2 | 2026-10-19 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | strip_block | 0.4680 | 0.355 | 0.813 | 0.570 | 0.0% | 225 | 1520 (524) | 3.106 |
| aux-probe2-embeddinggemma | 2026-10-19 | ollama | embeddinggemma:latest | 3a2d262bdb75 | 768 | 2048 | strip_block | 0.5371 | 0.458 | 0.794 | 0.582 | 0.0% | 225 | 1520 (524) | 4.227 |
| aux-probe2-mxbai | 2026-10-19 | ollama | mxbai-embed-large:latest | 468836162de7 | 1024 | 512 | strip_block | 0.5552 | 0.413 | 0.768 | 0.558 | 0.4% | 225 | 1520 (524) | 3.432 |

## Phase 2 — probe-aux groups

| run_id | transform | document | source_csv | n_pairs | pos | neg | pair_auc | same-doc hit@1 | same-doc hit@10 | n_rows_with_gold |
|---|---|---|---|---|---|---|---|---|---|---|
| aux-probe2-minilm | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6753 | 0.400 | 0.857 | 35 |
| aux-probe2-minilm | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5377 | 0.378 | 0.743 | 74 |
| aux-probe2-minilm | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4235 | 0.370 | 0.674 | 46 |
| aux-probe2-e5-base-v2 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.7108 | 0.457 | 0.886 | 35 |
| aux-probe2-e5-base-v2 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6388 | 0.378 | 0.730 | 74 |
| aux-probe2-e5-base-v2 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5670 | 0.304 | 0.761 | 46 |
| aux-probe2-bge-m3 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.5241 | 0.429 | 0.857 | 35 |
| aux-probe2-bge-m3 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5339 | 0.351 | 0.730 | 74 |
| aux-probe2-bge-m3 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4804 | 0.326 | 0.696 | 46 |
| aux-probe2-arctic2 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6209 | 0.371 | 0.943 | 35 |
| aux-probe2-arctic2 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5408 | 0.351 | 0.797 | 74 |
| aux-probe2-arctic2 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4773 | 0.348 | 0.739 | 46 |
| aux-probe2-embeddinggemma | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.7633 | 0.571 | 0.971 | 35 |
| aux-probe2-embeddinggemma | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6263 | 0.459 | 0.770 | 74 |
| aux-probe2-embeddinggemma | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5231 | 0.370 | 0.696 | 46 |
| aux-probe2-mxbai | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6886 | 0.543 | 0.886 | 35 |
| aux-probe2-mxbai | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6569 | 0.351 | 0.743 | 74 |
| aux-probe2-mxbai | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5074 | 0.413 | 0.717 | 46 |
