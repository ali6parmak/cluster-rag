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

## UPR — pool_cut

| run_id | date | split | predictor | embedding | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-eval-knn-minilm-k1 | 2026-10-20 | val | knn | minilm | top_k k=2 | 0.4134 | 0.3905 | 0.4024 | 0.868 | 0.2082 | 0.5499 | 0.5140 | 0.9037 | 660 | 0.09 |
| upr-eval-knn-minilm-k3 | 2026-10-20 | val | knn | minilm | top_k k=2 | 0.4780 | 0.4122 | 0.4622 | 0.868 | 0.2388 | 0.6311 | 0.6001 | 0.9288 | 660 | 0.092 |
| upr-eval-knn-minilm-k5 | 2026-10-20 | val | knn | minilm | top_k k=2 | 0.4977 | 0.4285 | 0.4799 | 0.868 | 0.2452 | 0.6615 | 0.6155 | 0.9409 | 660 | 0.113 |
| upr-eval-knn-minilm-k10 | 2026-10-20 | val | knn | minilm | top_k k=2 | 0.5076 | 0.4341 | 0.4905 | 0.868 | 0.2521 | 0.6712 | 0.6288 | 0.9491 | 660 | 0.098 |
| upr-eval-knn-minilm-k20 | 2026-10-20 | val | knn | minilm | top_k k=2 | 0.5167 | 0.4472 | 0.4993 | 0.868 | 0.2603 | 0.6779 | 0.6373 | 0.9567 | 660 | 0.094 |
| upr-eval-proto-minilm | 2026-10-20 | val | prototype | minilm | top_k k=2 | 0.5122 | 0.4926 | 0.4954 | 0.868 | 0.2694 | 0.7009 | 0.6456 | 0.7337 | 660 | 0.058 |
| upr-eval-maxsim-minilm | 2026-10-20 | val | maxsim | minilm | top_k k=2 | 0.4422 | 0.3831 | 0.4292 | 0.868 | 0.2485 | 0.6739 | 0.5675 | 0.7302 | 660 | 0.061 |
| upr-eval-knn-e5-base-v2-k1 | 2026-10-20 | val | knn | e5-base-v2 | top_k k=2 | 0.3944 | 0.3805 | 0.3838 | 0.853 | 0.1933 | 0.5158 | 0.4856 | 0.8918 | 660 | 0.125 |
| upr-eval-knn-e5-base-v2-k3 | 2026-10-20 | val | knn | e5-base-v2 | top_k k=2 | 0.4780 | 0.4262 | 0.4610 | 0.853 | 0.2433 | 0.6403 | 0.6015 | 0.9303 | 660 | 0.119 |
| upr-eval-knn-e5-base-v2-k5 | 2026-10-20 | val | knn | e5-base-v2 | top_k k=2 | 0.5053 | 0.4434 | 0.4874 | 0.868 | 0.2515 | 0.6688 | 0.6245 | 0.9381 | 660 | 0.119 |
| upr-eval-knn-e5-base-v2-k10 | 2026-10-20 | val | knn | e5-base-v2 | top_k k=2 | 0.5205 | 0.4710 | 0.5025 | 0.868 | 0.2606 | 0.6723 | 0.6407 | 0.9481 | 660 | 0.14 |
| upr-eval-knn-e5-base-v2-k20 | 2026-10-20 | val | knn | e5-base-v2 | top_k k=2 | 0.5152 | 0.4561 | 0.4971 | 0.868 | 0.2658 | 0.6881 | 0.6407 | 0.9518 | 660 | 0.12 |
| upr-eval-proto-e5-base-v2 | 2026-10-20 | val | prototype | e5-base-v2 | top_k k=2 | 0.5182 | 0.5087 | 0.5002 | 0.868 | 0.2761 | 0.7107 | 0.6553 | 0.7275 | 660 | 0.09 |
| upr-eval-maxsim-e5-base-v2 | 2026-10-20 | val | maxsim | e5-base-v2 | top_k k=2 | 0.4453 | 0.3996 | 0.4307 | 0.868 | 0.2552 | 0.6811 | 0.5702 | 0.7321 | 660 | 0.091 |
| upr-eval-knn-bge-m3-k1 | 2026-10-20 | val | knn | bge-m3 | top_k k=2 | 0.4020 | 0.3716 | 0.3907 | 0.868 | 0.2030 | 0.5306 | 0.5018 | 0.9029 | 660 | 0.14 |
| upr-eval-knn-bge-m3-k3 | 2026-10-20 | val | knn | bge-m3 | top_k k=2 | 0.4787 | 0.4007 | 0.4618 | 0.868 | 0.2370 | 0.6212 | 0.5957 | 0.9256 | 660 | 0.136 |
| upr-eval-knn-bge-m3-k5 | 2026-10-20 | val | knn | bge-m3 | top_k k=2 | 0.4954 | 0.4234 | 0.4759 | 0.868 | 0.2464 | 0.6616 | 0.6121 | 0.9380 | 660 | 0.138 |
| upr-eval-knn-bge-m3-k10 | 2026-10-20 | val | knn | bge-m3 | top_k k=2 | 0.5099 | 0.4307 | 0.4910 | 0.868 | 0.2576 | 0.6788 | 0.6322 | 0.9524 | 660 | 0.136 |
| upr-eval-knn-bge-m3-k20 | 2026-10-20 | val | knn | bge-m3 | top_k k=2 | 0.5084 | 0.4334 | 0.4880 | 0.868 | 0.2606 | 0.6769 | 0.6292 | 0.9509 | 660 | 0.157 |
| upr-eval-proto-bge-m3 | 2026-10-20 | val | prototype | bge-m3 | top_k k=2 | 0.5205 | 0.5034 | 0.5023 | 0.868 | 0.2752 | 0.7139 | 0.6565 | 0.7303 | 660 | 0.101 |
| upr-eval-maxsim-bge-m3 | 2026-10-20 | val | maxsim | bge-m3 | top_k k=2 | 0.4286 | 0.3620 | 0.4156 | 0.868 | 0.2491 | 0.6835 | 0.5624 | 0.7321 | 660 | 0.109 |
| upr-eval-knn-arctic2-k1 | 2026-10-20 | val | knn | arctic2 | top_k k=2 | 0.4331 | 0.4193 | 0.4211 | 0.882 | 0.2167 | 0.5682 | 0.5356 | 0.9154 | 660 | 0.14 |
| upr-eval-knn-arctic2-k3 | 2026-10-20 | val | knn | arctic2 | top_k k=2 | 0.5129 | 0.4528 | 0.4952 | 0.882 | 0.2539 | 0.6563 | 0.6341 | 0.9417 | 660 | 0.136 |
| upr-eval-knn-arctic2-k5 | 2026-10-20 | val | knn | arctic2 | top_k k=2 | 0.5334 | 0.4664 | 0.5134 | 0.882 | 0.2588 | 0.6819 | 0.6434 | 0.9472 | 660 | 0.136 |
| upr-eval-knn-arctic2-k10 | 2026-10-20 | val | knn | arctic2 | top_k k=2 | 0.5372 | 0.4767 | 0.5176 | 0.868 | 0.2667 | 0.6958 | 0.6556 | 0.9540 | 660 | 0.137 |
| upr-eval-knn-arctic2-k20 | 2026-10-20 | val | knn | arctic2 | top_k k=2 | 0.5304 | 0.4690 | 0.5118 | 0.868 | 0.2685 | 0.6984 | 0.6559 | 0.9567 | 660 | 0.139 |
| upr-eval-proto-arctic2 | 2026-10-20 | val | prototype | arctic2 | top_k k=2 | 0.5312 | 0.5235 | 0.5117 | 0.882 | 0.2824 | 0.7205 | 0.6682 | 0.7337 | 660 | 0.124 |
| upr-eval-maxsim-arctic2 | 2026-10-20 | val | maxsim | arctic2 | top_k k=2 | 0.4605 | 0.4083 | 0.4481 | 0.882 | 0.2621 | 0.6994 | 0.5936 | 0.7337 | 660 | 0.109 |
| upr-eval-knn-embeddinggemma-k1 | 2026-10-20 | val | knn | embeddinggemma | top_k k=2 | 0.4179 | 0.4041 | 0.4063 | 0.882 | 0.2148 | 0.5659 | 0.5194 | 0.9072 | 660 | 0.135 |
| upr-eval-knn-embeddinggemma-k3 | 2026-10-20 | val | knn | embeddinggemma | top_k k=2 | 0.5000 | 0.4511 | 0.4834 | 0.868 | 0.2497 | 0.6590 | 0.6235 | 0.9408 | 660 | 0.118 |
| upr-eval-knn-embeddinggemma-k5 | 2026-10-20 | val | knn | embeddinggemma | top_k k=2 | 0.5236 | 0.4699 | 0.5064 | 0.868 | 0.2573 | 0.6778 | 0.6447 | 0.9499 | 660 | 0.119 |
| upr-eval-knn-embeddinggemma-k10 | 2026-10-20 | val | knn | embeddinggemma | top_k k=2 | 0.5388 | 0.4796 | 0.5186 | 0.853 | 0.2664 | 0.6933 | 0.6589 | 0.9576 | 660 | 0.119 |
| upr-eval-knn-embeddinggemma-k20 | 2026-10-20 | val | knn | embeddinggemma | top_k k=2 | 0.5327 | 0.4777 | 0.5131 | 0.853 | 0.2703 | 0.6981 | 0.6578 | 0.9580 | 660 | 0.123 |
| upr-eval-proto-embeddinggemma | 2026-10-20 | val | prototype | embeddinggemma | top_k k=2 | 0.5228 | 0.5209 | 0.5048 | 0.868 | 0.2745 | 0.7096 | 0.6566 | 0.7337 | 660 | 0.087 |
| upr-eval-maxsim-embeddinggemma | 2026-10-20 | val | maxsim | embeddinggemma | top_k k=2 | 0.4453 | 0.4058 | 0.4324 | 0.868 | 0.2570 | 0.6902 | 0.5759 | 0.7333 | 660 | 0.11 |
| upr-eval-knn-mxbai-k1 | 2026-10-20 | val | knn | mxbai | top_k k=2 | 0.4195 | 0.3997 | 0.4066 | 0.868 | 0.2112 | 0.5563 | 0.5218 | 0.9100 | 660 | 0.14 |
| upr-eval-knn-mxbai-k3 | 2026-10-20 | val | knn | mxbai | top_k k=2 | 0.4886 | 0.4284 | 0.4716 | 0.868 | 0.2506 | 0.6503 | 0.6201 | 0.9352 | 660 | 0.137 |
| upr-eval-knn-mxbai-k5 | 2026-10-20 | val | knn | mxbai | top_k k=2 | 0.5152 | 0.4604 | 0.4958 | 0.868 | 0.2555 | 0.6778 | 0.6357 | 0.9461 | 660 | 0.137 |
| upr-eval-knn-mxbai-k10 | 2026-10-20 | val | knn | mxbai | top_k k=2 | 0.5327 | 0.4718 | 0.5119 | 0.868 | 0.2630 | 0.6915 | 0.6446 | 0.9545 | 660 | 0.136 |
| upr-eval-knn-mxbai-k20 | 2026-10-20 | val | knn | mxbai | top_k k=2 | 0.5274 | 0.4638 | 0.5078 | 0.868 | 0.2685 | 0.6936 | 0.6509 | 0.9577 | 660 | 0.14 |
| upr-eval-proto-mxbai | 2026-10-20 | val | prototype | mxbai | top_k k=2 | 0.5289 | 0.5154 | 0.5098 | 0.882 | 0.2761 | 0.7125 | 0.6586 | 0.7337 | 660 | 0.102 |
| upr-eval-maxsim-mxbai | 2026-10-20 | val | maxsim | mxbai | top_k k=2 | 0.4498 | 0.3956 | 0.4341 | 0.868 | 0.2588 | 0.6922 | 0.5846 | 0.7330 | 660 | 0.131 |
| upr-eval-knn-e5-large-v2-k1 | 2026-10-20 | val | knn | e5-large-v2 | top_k k=2 | 0.4096 | 0.4117 | 0.3986 | 0.868 | 0.2027 | 0.5309 | 0.5126 | 0.8946 | 660 | 0.139 |
| upr-eval-knn-e5-large-v2-k3 | 2026-10-20 | val | knn | e5-large-v2 | top_k k=2 | 0.4848 | 0.4339 | 0.4670 | 0.868 | 0.2436 | 0.6323 | 0.6060 | 0.9313 | 660 | 0.139 |
| upr-eval-knn-e5-large-v2-k5 | 2026-10-20 | val | knn | e5-large-v2 | top_k k=2 | 0.5099 | 0.4536 | 0.4904 | 0.868 | 0.2542 | 0.6736 | 0.6292 | 0.9396 | 660 | 0.136 |
| upr-eval-knn-e5-large-v2-k10 | 2026-10-20 | val | knn | e5-large-v2 | top_k k=2 | 0.5274 | 0.4786 | 0.5075 | 0.853 | 0.2664 | 0.6937 | 0.6504 | 0.9479 | 660 | 0.137 |
| upr-eval-knn-e5-large-v2-k20 | 2026-10-20 | val | knn | e5-large-v2 | top_k k=2 | 0.5205 | 0.4653 | 0.5018 | 0.868 | 0.2691 | 0.6908 | 0.6533 | 0.9539 | 660 | 0.141 |
| upr-eval-proto-e5-large-v2 | 2026-10-20 | val | prototype | e5-large-v2 | top_k k=2 | 0.5372 | 0.5286 | 0.5177 | 0.868 | 0.2794 | 0.7120 | 0.6699 | 0.7268 | 660 | 0.103 |
| upr-eval-maxsim-e5-large-v2 | 2026-10-20 | val | maxsim | e5-large-v2 | top_k k=2 | 0.4498 | 0.4060 | 0.4352 | 0.868 | 0.2579 | 0.6909 | 0.5875 | 0.7333 | 660 | 0.108 |
| upr-eval-knn-mpnet-k1 | 2026-10-20 | val | knn | mpnet | top_k k=2 | 0.4210 | 0.4042 | 0.4104 | 0.882 | 0.2100 | 0.5545 | 0.5231 | 0.9069 | 660 | 0.144 |
| upr-eval-knn-mpnet-k3 | 2026-10-20 | val | knn | mpnet | top_k k=2 | 0.4787 | 0.4221 | 0.4645 | 0.882 | 0.2403 | 0.6389 | 0.6058 | 0.9343 | 660 | 0.12 |
| upr-eval-knn-mpnet-k5 | 2026-10-20 | val | knn | mpnet | top_k k=2 | 0.4947 | 0.4320 | 0.4774 | 0.868 | 0.2442 | 0.6581 | 0.6111 | 0.9419 | 660 | 0.123 |
| upr-eval-knn-mpnet-k10 | 2026-10-20 | val | knn | mpnet | top_k k=2 | 0.5023 | 0.4421 | 0.4871 | 0.853 | 0.2521 | 0.6686 | 0.6257 | 0.9448 | 660 | 0.119 |
| upr-eval-knn-mpnet-k20 | 2026-10-20 | val | knn | mpnet | top_k k=2 | 0.5061 | 0.4452 | 0.4895 | 0.853 | 0.2573 | 0.6751 | 0.6286 | 0.9557 | 660 | 0.122 |
| upr-eval-proto-mpnet | 2026-10-20 | val | prototype | mpnet | top_k k=2 | 0.5053 | 0.4929 | 0.4903 | 0.882 | 0.2667 | 0.7003 | 0.6405 | 0.7324 | 660 | 0.084 |
| upr-eval-maxsim-mpnet | 2026-10-20 | val | maxsim | mpnet | top_k k=2 | 0.4384 | 0.3881 | 0.4279 | 0.882 | 0.2476 | 0.6697 | 0.5706 | 0.7319 | 660 | 0.091 |
| upr-eval-knn-nomic-v2-moe-k1 | 2026-10-20 | val | knn | nomic-v2-moe | top_k k=2 | 0.4164 | 0.3840 | 0.4041 | 0.868 | 0.2127 | 0.5521 | 0.5126 | 0.9105 | 660 | 0.123 |
| upr-eval-knn-nomic-v2-moe-k3 | 2026-10-20 | val | knn | nomic-v2-moe | top_k k=2 | 0.4932 | 0.4198 | 0.4767 | 0.868 | 0.2503 | 0.6509 | 0.6177 | 0.9386 | 660 | 0.14 |
| upr-eval-knn-nomic-v2-moe-k5 | 2026-10-20 | val | knn | nomic-v2-moe | top_k k=2 | 0.5198 | 0.4466 | 0.4994 | 0.868 | 0.2545 | 0.6760 | 0.6339 | 0.9437 | 660 | 0.121 |
| upr-eval-knn-nomic-v2-moe-k10 | 2026-10-20 | val | knn | nomic-v2-moe | top_k k=2 | 0.5327 | 0.4642 | 0.5127 | 0.868 | 0.2600 | 0.6754 | 0.6442 | 0.9514 | 660 | 0.123 |
| upr-eval-knn-nomic-v2-moe-k20 | 2026-10-20 | val | knn | nomic-v2-moe | top_k k=2 | 0.5296 | 0.4702 | 0.5097 | 0.868 | 0.2648 | 0.6811 | 0.6492 | 0.9514 | 660 | 0.119 |
| upr-eval-proto-nomic-v2-moe | 2026-10-20 | val | prototype | nomic-v2-moe | top_k k=2 | 0.5015 | 0.4916 | 0.4818 | 0.882 | 0.2709 | 0.7130 | 0.6367 | 0.7333 | 660 | 0.084 |
| upr-eval-maxsim-nomic-v2-moe | 2026-10-20 | val | maxsim | nomic-v2-moe | top_k k=2 | 0.4400 | 0.3745 | 0.4265 | 0.868 | 0.2555 | 0.6788 | 0.5689 | 0.7326 | 660 | 0.091 |
| upr-eval-knn-jina-v5-k1 | 2026-10-20 | val | knn | jina-v5 | top_k k=2 | 0.4179 | 0.4213 | 0.4077 | 0.868 | 0.2136 | 0.5601 | 0.5299 | 0.9107 | 660 | 0.168 |
| upr-eval-knn-jina-v5-k3 | 2026-10-20 | val | knn | jina-v5 | top_k k=2 | 0.5038 | 0.4500 | 0.4881 | 0.868 | 0.2500 | 0.6514 | 0.6297 | 0.9415 | 660 | 0.186 |
| upr-eval-knn-jina-v5-k5 | 2026-10-20 | val | knn | jina-v5 | top_k k=2 | 0.5228 | 0.4634 | 0.5043 | 0.868 | 0.2582 | 0.6838 | 0.6472 | 0.9476 | 660 | 0.186 |
| upr-eval-knn-jina-v5-k10 | 2026-10-20 | val | knn | jina-v5 | top_k k=2 | 0.5304 | 0.4655 | 0.5111 | 0.868 | 0.2673 | 0.6997 | 0.6577 | 0.9558 | 660 | 0.163 |
| upr-eval-knn-jina-v5-k20 | 2026-10-20 | val | knn | jina-v5 | top_k k=2 | 0.5380 | 0.4760 | 0.5183 | 0.868 | 0.2736 | 0.6953 | 0.6634 | 0.9605 | 660 | 0.165 |
| upr-eval-proto-jina-v5 | 2026-10-20 | val | prototype | jina-v5 | top_k k=2 | 0.5380 | 0.5281 | 0.5194 | 0.882 | 0.2788 | 0.7172 | 0.6721 | 0.7337 | 660 | 0.131 |
| upr-eval-maxsim-jina-v5 | 2026-10-20 | val | maxsim | jina-v5 | top_k k=2 | 0.4559 | 0.4148 | 0.4436 | 0.868 | 0.2627 | 0.6947 | 0.5970 | 0.7337 | 660 | 0.139 |
| upr-eval-knn-voyage-4-nano-k1 | 2026-10-20 | val | knn | voyage-4-nano | top_k k=2 | 0.4400 | 0.4450 | 0.4281 | 0.868 | 0.2197 | 0.5761 | 0.5430 | 0.9105 | 660 | 0.236 |
| upr-eval-knn-voyage-4-nano-k3 | 2026-10-20 | val | knn | voyage-4-nano | top_k k=2 | 0.5198 | 0.4795 | 0.5025 | 0.868 | 0.2536 | 0.6578 | 0.6402 | 0.9415 | 660 | 0.226 |
| upr-eval-knn-voyage-4-nano-k5 | 2026-10-20 | val | knn | voyage-4-nano | top_k k=2 | 0.5312 | 0.4825 | 0.5125 | 0.868 | 0.2576 | 0.6732 | 0.6509 | 0.9479 | 660 | 0.228 |
| upr-eval-knn-voyage-4-nano-k10 | 2026-10-20 | val | knn | voyage-4-nano | top_k k=2 | 0.5327 | 0.4741 | 0.5123 | 0.868 | 0.2630 | 0.6894 | 0.6533 | 0.9549 | 660 | 0.246 |
| upr-eval-knn-voyage-4-nano-k20 | 2026-10-20 | val | knn | voyage-4-nano | top_k k=2 | 0.5380 | 0.4932 | 0.5178 | 0.868 | 0.2676 | 0.6980 | 0.6589 | 0.9597 | 660 | 0.23 |
| upr-eval-proto-voyage-4-nano | 2026-10-20 | val | prototype | voyage-4-nano | top_k k=2 | 0.5182 | 0.5171 | 0.4978 | 0.882 | 0.2745 | 0.7144 | 0.6525 | 0.7337 | 660 | 0.195 |
| upr-eval-maxsim-voyage-4-nano | 2026-10-20 | val | maxsim | voyage-4-nano | top_k k=2 | 0.4711 | 0.4332 | 0.4573 | 0.868 | 0.2597 | 0.6980 | 0.5966 | 0.7337 | 660 | 0.203 |
| upr-eval-knn-granite-english-r2-k1 | 2026-10-20 | val | knn | granite-english-r2 | top_k k=2 | 0.4255 | 0.4233 | 0.4151 | 0.868 | 0.2124 | 0.5602 | 0.5324 | 0.9033 | 660 | 0.142 |
| upr-eval-knn-granite-english-r2-k3 | 2026-10-20 | val | knn | granite-english-r2 | top_k k=2 | 0.5061 | 0.4667 | 0.4896 | 0.882 | 0.2467 | 0.6504 | 0.6261 | 0.9356 | 660 | 0.137 |
| upr-eval-knn-granite-english-r2-k5 | 2026-10-20 | val | knn | granite-english-r2 | top_k k=2 | 0.5205 | 0.4741 | 0.5034 | 0.882 | 0.2548 | 0.6773 | 0.6414 | 0.9467 | 660 | 0.135 |
| upr-eval-knn-granite-english-r2-k10 | 2026-10-20 | val | knn | granite-english-r2 | top_k k=2 | 0.5274 | 0.4761 | 0.5078 | 0.868 | 0.2639 | 0.6867 | 0.6492 | 0.9523 | 660 | 0.161 |
| upr-eval-knn-granite-english-r2-k20 | 2026-10-20 | val | knn | granite-english-r2 | top_k k=2 | 0.5319 | 0.4801 | 0.5132 | 0.868 | 0.2673 | 0.6905 | 0.6552 | 0.9568 | 660 | 0.152 |
| upr-eval-proto-granite-english-r2 | 2026-10-20 | val | prototype | granite-english-r2 | top_k k=2 | 0.5327 | 0.5259 | 0.5147 | 0.868 | 0.2779 | 0.7121 | 0.6694 | 0.7321 | 660 | 0.101 |
| upr-eval-maxsim-granite-english-r2 | 2026-10-20 | val | maxsim | granite-english-r2 | top_k k=2 | 0.4620 | 0.4281 | 0.4491 | 0.882 | 0.2600 | 0.6900 | 0.5969 | 0.7337 | 660 | 0.112 |
| upr-eval-knn-f2llm-80m-k1 | 2026-10-20 | val | knn | f2llm-80m | top_k k=2 | 0.4081 | 0.3819 | 0.3957 | 0.868 | 0.2048 | 0.5314 | 0.5072 | 0.9018 | 660 | 0.095 |
| upr-eval-knn-f2llm-80m-k3 | 2026-10-20 | val | knn | f2llm-80m | top_k k=2 | 0.4818 | 0.4130 | 0.4644 | 0.868 | 0.2415 | 0.6328 | 0.6008 | 0.9371 | 660 | 0.092 |
| upr-eval-knn-f2llm-80m-k5 | 2026-10-20 | val | knn | f2llm-80m | top_k k=2 | 0.5015 | 0.4262 | 0.4821 | 0.853 | 0.2488 | 0.6646 | 0.6111 | 0.9436 | 660 | 0.092 |
| upr-eval-knn-f2llm-80m-k10 | 2026-10-20 | val | knn | f2llm-80m | top_k k=2 | 0.5152 | 0.4482 | 0.4955 | 0.853 | 0.2576 | 0.6807 | 0.6279 | 0.9538 | 660 | 0.094 |
| upr-eval-knn-f2llm-80m-k20 | 2026-10-20 | val | knn | f2llm-80m | top_k k=2 | 0.5167 | 0.4497 | 0.4965 | 0.853 | 0.2615 | 0.6832 | 0.6323 | 0.9559 | 660 | 0.118 |
| upr-eval-proto-f2llm-80m | 2026-10-20 | val | prototype | f2llm-80m | top_k k=2 | 0.5038 | 0.4804 | 0.4853 | 0.882 | 0.2664 | 0.6977 | 0.6309 | 0.7345 | 660 | 0.063 |
| upr-eval-maxsim-f2llm-80m | 2026-10-20 | val | maxsim | f2llm-80m | top_k k=2 | 0.4407 | 0.3746 | 0.4261 | 0.868 | 0.2518 | 0.6816 | 0.5702 | 0.7301 | 660 | 0.067 |
| upr-eval-knn-embeddinggemma2-k1 | 2026-10-20 | val | knn | embeddinggemma2 | top_k k=2 | 0.4248 | 0.4208 | 0.4120 | 0.868 | 0.2073 | 0.5443 | 0.5214 | 0.9006 | 660 | 0.119 |
| upr-eval-knn-embeddinggemma2-k3 | 2026-10-20 | val | knn | embeddinggemma2 | top_k k=2 | 0.4992 | 0.4433 | 0.4816 | 0.868 | 0.2470 | 0.6391 | 0.6221 | 0.9303 | 660 | 0.117 |
| upr-eval-knn-embeddinggemma2-k5 | 2026-10-20 | val | knn | embeddinggemma2 | top_k k=2 | 0.5167 | 0.4592 | 0.4974 | 0.868 | 0.2548 | 0.6734 | 0.6372 | 0.9422 | 660 | 0.117 |
| upr-eval-knn-embeddinggemma2-k10 | 2026-10-20 | val | knn | embeddinggemma2 | top_k k=2 | 0.5319 | 0.4812 | 0.5122 | 0.868 | 0.2636 | 0.6812 | 0.6522 | 0.9511 | 660 | 0.118 |
| upr-eval-knn-embeddinggemma2-k20 | 2026-10-20 | val | knn | embeddinggemma2 | top_k k=2 | 0.5365 | 0.4956 | 0.5151 | 0.868 | 0.2691 | 0.6879 | 0.6579 | 0.9535 | 660 | 0.118 |
| upr-eval-proto-embeddinggemma2 | 2026-10-20 | val | prototype | embeddinggemma2 | top_k k=2 | 0.5289 | 0.5189 | 0.5085 | 0.868 | 0.2764 | 0.7120 | 0.6607 | 0.7302 | 660 | 0.104 |
| upr-eval-maxsim-embeddinggemma2 | 2026-10-20 | val | maxsim | embeddinggemma2 | top_k k=2 | 0.4658 | 0.4155 | 0.4507 | 0.868 | 0.2594 | 0.6900 | 0.5936 | 0.7330 | 660 | 0.089 |
