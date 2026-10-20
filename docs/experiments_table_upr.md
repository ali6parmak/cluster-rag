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

## UPR — embed_pool

| run_id | date | backend | model | model_version | dim | ctx | kinds | preprocess | n_texts | seconds | cache_hits | cache_misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-embed-minilm | 2026-10-20 | hf | sentence-transformers/all-MiniLM-L6-v2 | c9745ed1d9f2 | 384 | 256 | document,query | raw | 3000 | 3.723 | 3000 | 3000 |
| upr-embed-e5-base-v2 | 2026-10-20 | hf | intfloat/e5-base-v2 | f52bf8ec8c71 | 768 | 512 | document,query | raw | 3000 | 2.947 | 3000 | 3000 |
| upr-embed-bge-m3 | 2026-10-20 | hf | BAAI/bge-m3 | 9a0624b896d8 | 1024 | 8192 | document,query | raw | 3000 | 7.335 | 3000 | 3000 |
| upr-embed-arctic2 | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | document,query | raw | 3000 | 19.569 | 3000 | 3000 |
| upr-embed-embeddinggemma | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | document,query | raw | 3000 | 27.523 | 0 | 6000 |
| upr-embed-mxbai | 2026-10-20 | ollama | mxbai-embed-large:latest | 468836162de7 | 1024 | 512 | document,query | raw | 3000 | 34.764 | 0 | 6000 |
| upr-embed-e5-large-v2 | 2026-10-20 | hf | intfloat/e5-large-v2 | f169b11e22de | 1024 | 512 | document,query | raw | 3000 | 10.295 | 0 | 6000 |
| upr-embed-mpnet | 2026-10-20 | hf | sentence-transformers/all-mpnet-base-v2 | e8c3b32edf54 | 768 | 384 | document,query | raw | 3000 | 2.901 | 3000 | 3000 |
| upr-embed-nomic-v2-moe | 2026-10-20 | ollama | nomic-embed-text-v2-moe:latest | ff9c2f10ef5e | 768 | 512 | document,query | raw | 3000 | 28.768 | 0 | 6000 |
| upr-embed-jina-v5 | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | document,query | raw | 3000 | 21.639 | 0 | 6000 |
| upr-embed-voyage-4-nano | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | document,query | raw | 3000 | 7.254 | 0 | 6000 |
| upr-embed-granite-english-r2 | 2026-10-20 | hf | ibm-granite/granite-embedding-english-r2 | 47ea694b257b | 768 | 8192 | document,query | raw | 3000 | 2.615 | 3000 | 3000 |
| upr-embed-f2llm-80m | 2026-10-20 | hf | codefuse-ai/F2LLM-v2-80M | ad88d7a12671 | 320 | 40960 | document,query | raw | 3000 | 4.447 | 0 | 6000 |
| upr-embed-embeddinggemma2 | 2026-10-20 | ollama | embeddinggemma-2:latest | f4981a15b998 | 768 | 8192 | document,query | raw | 3000 | 41.567 | 0 | 6000 |
