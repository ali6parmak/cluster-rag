# OMCT experiment tables

Comparison tables keyed by run — auto-appended by the CLI commands (one file per dataset). The append-only narrative log is `docs/experiments.md` (one row per run, with full revisions/digests). Values come from each run's own artifacts; counts sit next to numbers; "n/a" = not reported by the run.

## OMCT — probe2

| run_id | date | backend | model | model_version | dim | ctx | transform | pair_auc | same-doc hit@1 | same-doc hit@10 | same-doc recall@10 | paras>ctx | n_paras | n_pairs(pos) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| omct-probe2-minilm | 2026-10-20 | hf | sentence-transformers/all-MiniLM-L6-v2 | c9745ed1d9f2 | 384 | 256 | strip_block | 0.4526 | 0.381 | 0.748 | 0.508 | 1.8% | 225 | 1520 (524) | 0.063 |
| omct-probe2-e5-base-v2 | 2026-10-20 | hf | intfloat/e5-base-v2 | f52bf8ec8c71 | 768 | 512 | strip_block | 0.5793 | 0.374 | 0.774 | 0.540 | 0.0% | 225 | 1520 (524) | 0.066 |
| omct-probe2-bge-m3 | 2026-10-20 | hf | BAAI/bge-m3 | 9a0624b896d8 | 1024 | 8192 | strip_block | 0.4687 | 0.361 | 0.748 | 0.476 | 0.0% | 225 | 1520 (524) | 0.083 |
| omct-probe2-arctic2 | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | strip_block | 0.4680 | 0.355 | 0.813 | 0.570 | 0.0% | 225 | 1520 (524) | 0.046 |
| omct-probe2-embeddinggemma | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | strip_block | 0.5371 | 0.458 | 0.794 | 0.582 | 0.0% | 225 | 1520 (524) | 0.041 |
| omct-probe2-mxbai | 2026-10-20 | ollama | mxbai-embed-large:latest | 468836162de7 | 1024 | 512 | strip_block | 0.5552 | 0.413 | 0.768 | 0.558 | 0.4% | 225 | 1520 (524) | 0.047 |
| omct-probe2-e5-large-v2 | 2026-10-20 | hf | intfloat/e5-large-v2 | f169b11e22de | 1024 | 512 | strip_block | 0.5112 | 0.400 | 0.826 | 0.594 | 0.0% | 225 | 1520 (524) | 2.751 |
| omct-probe2-mpnet | 2026-10-20 | hf | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf54 | 768 | 384 | strip_block | 0.4523 | 0.355 | 0.703 | 0.458 | 0.4% | 225 | 1520 (524) | 1.028 |
| omct-probe2-nomic-v2-moe | 2026-10-20 | ollama | nomic-embed-text-v2-moe:latest | ff9c2f10ef5e | 768 | 512 | strip_block | 0.4492 | 0.400 | 0.703 | 0.463 | 0.4% | 225 | 1520 (524) | 7.832 |
| omct-probe2-jina-v5 | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | strip_block | 0.5394 | 0.465 | 0.813 | 0.589 | 0.0% | 225 | 1520 (524) | 2.577 |
| omct-probe2-voyage-4-nano | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | strip_block | 0.5004 | 0.355 | 0.716 | 0.474 | 0.0% | 225 | 1520 (524) | 0.765 |
| omct-probe2-granite-english-r2 | 2026-10-20 | hf | ibm-granite/granite-embedding-english-r2 | 47ea694b257b | 768 | 8192 | strip_block | 0.5130 | 0.406 | 0.781 | 0.557 | 0.0% | 225 | 1520 (524) | 0.479 |
| omct-probe2-f2llm-80m | 2026-10-20 | hf | codefuse-ai/F2LLM-v2-80M | ad88d7a12671 | 320 | 40960 | strip_block | 0.5986 | 0.297 | 0.697 | 0.475 | 0.0% | 225 | 1520 (524) | 0.337 |
| omct-probe2-embeddinggemma2 | 2026-10-20 | ollama | embeddinggemma-2:latest | f4981a15b998 | 768 | 8192 | strip_block | 0.5815 | 0.374 | 0.755 | 0.564 | 0.0% | 225 | 1520 (524) | 4.463 |

## OMCT — probe2 - groups

| run_id | transform | document | source_csv | n_pairs | pos | neg | pair_auc | same-doc hit@1 | same-doc hit@10 | n_rows_with_gold |
|---|---|---|---|---|---|---|---|---|---|---|
| omct-probe2-minilm | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6753 | 0.400 | 0.857 | 35 |
| omct-probe2-minilm | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5377 | 0.378 | 0.743 | 74 |
| omct-probe2-minilm | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4235 | 0.370 | 0.674 | 46 |
| omct-probe2-e5-base-v2 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.7108 | 0.457 | 0.886 | 35 |
| omct-probe2-e5-base-v2 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6388 | 0.378 | 0.730 | 74 |
| omct-probe2-e5-base-v2 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5670 | 0.304 | 0.761 | 46 |
| omct-probe2-bge-m3 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.5241 | 0.429 | 0.857 | 35 |
| omct-probe2-bge-m3 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5339 | 0.351 | 0.730 | 74 |
| omct-probe2-bge-m3 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4804 | 0.326 | 0.696 | 46 |
| omct-probe2-arctic2 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6209 | 0.371 | 0.943 | 35 |
| omct-probe2-arctic2 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5408 | 0.351 | 0.797 | 74 |
| omct-probe2-arctic2 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4773 | 0.348 | 0.739 | 46 |
| omct-probe2-embeddinggemma | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.7633 | 0.571 | 0.971 | 35 |
| omct-probe2-embeddinggemma | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6263 | 0.459 | 0.770 | 74 |
| omct-probe2-embeddinggemma | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5231 | 0.370 | 0.696 | 46 |
| omct-probe2-mxbai | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6886 | 0.543 | 0.886 | 35 |
| omct-probe2-mxbai | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6569 | 0.351 | 0.743 | 74 |
| omct-probe2-mxbai | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5074 | 0.413 | 0.717 | 46 |
| omct-probe2-e5-large-v2 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6658 | 0.543 | 0.914 | 35 |
| omct-probe2-e5-large-v2 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6053 | 0.378 | 0.824 | 74 |
| omct-probe2-e5-large-v2 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4867 | 0.326 | 0.761 | 46 |
| omct-probe2-mpnet | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.5956 | 0.343 | 0.857 | 35 |
| omct-probe2-mpnet | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5435 | 0.392 | 0.689 | 74 |
| omct-probe2-mpnet | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4190 | 0.304 | 0.609 | 46 |
| omct-probe2-nomic-v2-moe | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.5038 | 0.457 | 0.857 | 35 |
| omct-probe2-nomic-v2-moe | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5272 | 0.392 | 0.703 | 74 |
| omct-probe2-nomic-v2-moe | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4393 | 0.370 | 0.587 | 46 |
| omct-probe2-jina-v5 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.7146 | 0.486 | 0.914 | 35 |
| omct-probe2-jina-v5 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6054 | 0.419 | 0.797 | 74 |
| omct-probe2-jina-v5 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5467 | 0.522 | 0.761 | 46 |
| omct-probe2-voyage-4-nano | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6165 | 0.457 | 0.857 | 35 |
| omct-probe2-voyage-4-nano | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5573 | 0.338 | 0.716 | 74 |
| omct-probe2-voyage-4-nano | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5027 | 0.304 | 0.609 | 46 |
| omct-probe2-granite-english-r2 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.7139 | 0.543 | 0.886 | 35 |
| omct-probe2-granite-english-r2 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.5955 | 0.378 | 0.757 | 74 |
| omct-probe2-granite-english-r2 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.4863 | 0.348 | 0.739 | 46 |
| omct-probe2-f2llm-80m | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6259 | 0.429 | 0.914 | 35 |
| omct-probe2-f2llm-80m | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6171 | 0.324 | 0.689 | 74 |
| omct-probe2-f2llm-80m | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5260 | 0.152 | 0.543 | 46 |
| omct-probe2-embeddinggemma2 | strip_block | Human rights in Colombia Amnesty International | OMCT GTI - Test - Reports 01 - 3 AI.csv | 99 | 79 | 20 | 0.6987 | 0.429 | 0.857 | 35 |
| omct-probe2-embeddinggemma2 | strip_block | Observaciones preliminares colombia 2024 | OMCT GTI - Test - Reports 01 - 2 CIDH.csv | 576 | 234 | 342 | 0.6294 | 0.405 | 0.770 | 74 |
| omct-probe2-embeddinggemma2 | strip_block | Concluding observations on the sixth periodic report of Colombia** | OMCT GTI - Test - Reports 01 - 1 NU.csv | 845 | 211 | 634 | 0.5585 | 0.283 | 0.652 | 46 |
