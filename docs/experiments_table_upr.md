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

## UPR — fusion_weights

| run_id | date | split | predictor | embedding | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-eval-fusion-embeddinggemma-k10-a0.25 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5456 | 0.4993 | 0.5254 | 0.868 | 0.2745 | 0.7081 | 0.6713 | 0.7337 | 660 | 0.139 |
| upr-eval-fusion-embeddinggemma-k10-a0.5 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5464 | 0.5000 | 0.5261 | 0.868 | 0.2739 | 0.7068 | 0.6698 | 0.7337 | 660 | 0.133 |
| upr-eval-fusion-embeddinggemma-k10-a1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5426 | 0.4873 | 0.5228 | 0.853 | 0.2739 | 0.7068 | 0.6681 | 0.7337 | 660 | 0.157 |
| upr-eval-fusion-embeddinggemma-k10-a2 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5441 | 0.4873 | 0.5237 | 0.853 | 0.2727 | 0.7064 | 0.6671 | 0.7337 | 660 | 0.134 |
| upr-eval-fusion-embeddinggemma-k10-a4 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5395 | 0.4798 | 0.5190 | 0.853 | 0.2709 | 0.7069 | 0.6649 | 0.7337 | 660 | 0.138 |
| upr-eval-fusion-embeddinggemma-k10-a8 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5388 | 0.4774 | 0.5184 | 0.853 | 0.2700 | 0.7066 | 0.6634 | 0.7337 | 660 | 0.135 |
| upr-eval-fusion-embeddinggemma-k20-a0.25 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5418 | 0.4956 | 0.5217 | 0.868 | 0.2764 | 0.7061 | 0.6667 | 0.7337 | 660 | 0.142 |
| upr-eval-fusion-embeddinggemma-k20-a0.5 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5403 | 0.4908 | 0.5201 | 0.868 | 0.2742 | 0.7057 | 0.6645 | 0.7337 | 660 | 0.147 |
| upr-eval-fusion-embeddinggemma-k20-a1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5403 | 0.4926 | 0.5200 | 0.868 | 0.2739 | 0.7057 | 0.6638 | 0.7337 | 660 | 0.143 |
| upr-eval-fusion-embeddinggemma-k20-a2 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5388 | 0.4864 | 0.5179 | 0.853 | 0.2730 | 0.7062 | 0.6621 | 0.7337 | 660 | 0.138 |
| upr-eval-fusion-embeddinggemma-k20-a4 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5372 | 0.4838 | 0.5168 | 0.853 | 0.2724 | 0.7071 | 0.6610 | 0.7337 | 660 | 0.163 |
| upr-eval-fusion-embeddinggemma-k20-a8 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5334 | 0.4779 | 0.5137 | 0.853 | 0.2718 | 0.7067 | 0.6599 | 0.7337 | 660 | 0.134 |
| upr-eval-fusion-jina-v5-k10-a0.25 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5403 | 0.4834 | 0.5207 | 0.868 | 0.2758 | 0.7110 | 0.6704 | 0.7337 | 660 | 0.178 |
| upr-eval-fusion-jina-v5-k10-a0.5 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5403 | 0.4835 | 0.5206 | 0.868 | 0.2761 | 0.7093 | 0.6704 | 0.7337 | 660 | 0.18 |
| upr-eval-fusion-jina-v5-k10-a1 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5403 | 0.4832 | 0.5206 | 0.868 | 0.2755 | 0.7093 | 0.6701 | 0.7337 | 660 | 0.177 |
| upr-eval-fusion-jina-v5-k10-a2 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5350 | 0.4742 | 0.5161 | 0.868 | 0.2748 | 0.7093 | 0.6677 | 0.7337 | 660 | 0.183 |
| upr-eval-fusion-jina-v5-k10-a4 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5334 | 0.4702 | 0.5147 | 0.868 | 0.2736 | 0.7093 | 0.6661 | 0.7337 | 660 | 0.192 |
| upr-eval-fusion-jina-v5-k10-a8 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5327 | 0.4704 | 0.5140 | 0.868 | 0.2727 | 0.7090 | 0.6649 | 0.7337 | 660 | 0.181 |
| upr-eval-fusion-jina-v5-k20-a0.25 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5410 | 0.4845 | 0.5215 | 0.868 | 0.2764 | 0.7119 | 0.6681 | 0.7337 | 660 | 0.198 |
| upr-eval-fusion-jina-v5-k20-a0.5 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5403 | 0.4842 | 0.5205 | 0.868 | 0.2764 | 0.7114 | 0.6671 | 0.7337 | 660 | 0.18 |
| upr-eval-fusion-jina-v5-k20-a1 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5395 | 0.4796 | 0.5197 | 0.868 | 0.2764 | 0.7112 | 0.6672 | 0.7337 | 660 | 0.182 |
| upr-eval-fusion-jina-v5-k20-a2 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5395 | 0.4789 | 0.5197 | 0.868 | 0.2755 | 0.7107 | 0.6659 | 0.7337 | 660 | 0.178 |
| upr-eval-fusion-jina-v5-k20-a4 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5380 | 0.4768 | 0.5185 | 0.868 | 0.2748 | 0.7092 | 0.6649 | 0.7337 | 660 | 0.183 |
| upr-eval-fusion-jina-v5-k20-a8 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5372 | 0.4740 | 0.5177 | 0.868 | 0.2742 | 0.7069 | 0.6642 | 0.7337 | 660 | 0.207 |
| upr-eval-fusion-voyage-4-nano-k10-a0.25 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5395 | 0.4893 | 0.5201 | 0.868 | 0.2730 | 0.7140 | 0.6654 | 0.7337 | 660 | 0.248 |
| upr-eval-fusion-voyage-4-nano-k10-a0.5 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5380 | 0.4842 | 0.5185 | 0.868 | 0.2721 | 0.7128 | 0.6641 | 0.7337 | 660 | 0.246 |
| upr-eval-fusion-voyage-4-nano-k10-a1 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5380 | 0.4803 | 0.5172 | 0.868 | 0.2724 | 0.7128 | 0.6650 | 0.7337 | 660 | 0.266 |
| upr-eval-fusion-voyage-4-nano-k10-a2 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5380 | 0.4809 | 0.5172 | 0.868 | 0.2715 | 0.7128 | 0.6637 | 0.7337 | 660 | 0.246 |
| upr-eval-fusion-voyage-4-nano-k10-a4 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5372 | 0.4783 | 0.5164 | 0.868 | 0.2700 | 0.7128 | 0.6622 | 0.7337 | 660 | 0.245 |
| upr-eval-fusion-voyage-4-nano-k10-a8 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5372 | 0.4796 | 0.5163 | 0.868 | 0.2688 | 0.7128 | 0.6608 | 0.7337 | 660 | 0.261 |
| upr-eval-fusion-voyage-4-nano-k20-a0.25 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5380 | 0.4951 | 0.5188 | 0.868 | 0.2736 | 0.7115 | 0.6669 | 0.7337 | 660 | 0.245 |
| upr-eval-fusion-voyage-4-nano-k20-a0.5 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5372 | 0.4922 | 0.5175 | 0.868 | 0.2718 | 0.7111 | 0.6637 | 0.7337 | 660 | 0.244 |
| upr-eval-fusion-voyage-4-nano-k20-a1 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5388 | 0.4940 | 0.5187 | 0.868 | 0.2712 | 0.7111 | 0.6632 | 0.7337 | 660 | 0.246 |
| upr-eval-fusion-voyage-4-nano-k20-a2 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5395 | 0.4931 | 0.5192 | 0.868 | 0.2709 | 0.7111 | 0.6628 | 0.7337 | 660 | 0.247 |
| upr-eval-fusion-voyage-4-nano-k20-a4 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5388 | 0.4915 | 0.5184 | 0.868 | 0.2691 | 0.7088 | 0.6609 | 0.7337 | 660 | 0.263 |
| upr-eval-fusion-voyage-4-nano-k20-a8 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5380 | 0.4912 | 0.5178 | 0.868 | 0.2688 | 0.7083 | 0.6606 | 0.7337 | 660 | 0.247 |
| upr-eval-fusion-arctic2-k10-a0.25 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5403 | 0.4887 | 0.5205 | 0.868 | 0.2733 | 0.7135 | 0.6650 | 0.7337 | 660 | 0.149 |
| upr-eval-fusion-arctic2-k10-a0.5 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5410 | 0.4867 | 0.5210 | 0.868 | 0.2724 | 0.7117 | 0.6637 | 0.7337 | 660 | 0.145 |
| upr-eval-fusion-arctic2-k10-a1 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5395 | 0.4855 | 0.5197 | 0.868 | 0.2715 | 0.7117 | 0.6628 | 0.7337 | 660 | 0.147 |
| upr-eval-fusion-arctic2-k10-a2 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5410 | 0.4877 | 0.5213 | 0.868 | 0.2709 | 0.7117 | 0.6623 | 0.7337 | 660 | 0.146 |
| upr-eval-fusion-arctic2-k10-a4 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5388 | 0.4812 | 0.5191 | 0.868 | 0.2703 | 0.7114 | 0.6610 | 0.7337 | 660 | 0.145 |
| upr-eval-fusion-arctic2-k10-a8 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5395 | 0.4798 | 0.5199 | 0.868 | 0.2694 | 0.7098 | 0.6599 | 0.7337 | 660 | 0.146 |
| upr-eval-fusion-arctic2-k20-a0.25 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5350 | 0.4725 | 0.5152 | 0.868 | 0.2742 | 0.7120 | 0.6632 | 0.7337 | 660 | 0.18 |
| upr-eval-fusion-arctic2-k20-a0.5 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5334 | 0.4705 | 0.5138 | 0.868 | 0.2730 | 0.7120 | 0.6618 | 0.7337 | 660 | 0.147 |
| upr-eval-fusion-arctic2-k20-a1 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5334 | 0.4670 | 0.5141 | 0.868 | 0.2730 | 0.7114 | 0.6616 | 0.7337 | 660 | 0.148 |
| upr-eval-fusion-arctic2-k20-a2 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5327 | 0.4668 | 0.5136 | 0.868 | 0.2712 | 0.7105 | 0.6593 | 0.7337 | 660 | 0.147 |
| upr-eval-fusion-arctic2-k20-a4 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5312 | 0.4652 | 0.5125 | 0.868 | 0.2706 | 0.7101 | 0.6584 | 0.7337 | 660 | 0.146 |
| upr-eval-fusion-arctic2-k20-a8 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5312 | 0.4688 | 0.5125 | 0.868 | 0.2703 | 0.7086 | 0.6584 | 0.7337 | 660 | 0.149 |

## UPR — zeroshot

| run_id | date | backend | model | model_version | dim | ctx | kinds | preprocess | n_texts | seconds | cache_hits | cache_misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-labelembed-embeddinggemma-name | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | label | raw | 68 | 1.836 | 0 | 68 |
| upr-eval-zeroshot-embeddinggemma-name | 2026-10-20 | val | zeroshot | embeddinggemma | top_k k=2 | 0.3875 | 0.4060 | 0.3795 | 0.897 | 0.2197 | 0.6559 | 0.5216 | 0.8615 | 660 | 0.054 |
| upr-labelembed-embeddinggemma-about | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | label | raw | 68 | 0.359 | 0 | 68 |
| upr-eval-zeroshot-embeddinggemma-about | 2026-10-20 | val | zeroshot | embeddinggemma | top_k k=2 | 0.4301 | 0.4439 | 0.4186 | 0.941 | 0.2455 | 0.7021 | 0.5752 | 0.8664 | 660 | 0.05 |
| upr-labelembed-embeddinggemma-topic | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | label | raw | 68 | 0.294 | 0 | 68 |
| upr-eval-zeroshot-embeddinggemma-topic | 2026-10-20 | val | zeroshot | embeddinggemma | top_k k=2 | 0.4407 | 0.4576 | 0.4286 | 0.971 | 0.2470 | 0.6942 | 0.5872 | 0.8556 | 660 | 0.053 |
| upr-labelembed-jina-v5-name | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | label | raw | 68 | 6.763 | 0 | 68 |
| upr-eval-zeroshot-jina-v5-name | 2026-10-20 | val | zeroshot | jina-v5 | top_k k=2 | 0.4704 | 0.4809 | 0.4617 | 0.926 | 0.2548 | 0.7220 | 0.6211 | 0.8847 | 660 | 0.072 |
| upr-labelembed-jina-v5-about | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | label | raw | 68 | 4.579 | 0 | 68 |
| upr-eval-zeroshot-jina-v5-about | 2026-10-20 | val | zeroshot | jina-v5 | top_k k=2 | 0.4894 | 0.5029 | 0.4806 | 0.941 | 0.2630 | 0.7447 | 0.6442 | 0.8899 | 660 | 0.074 |
| upr-labelembed-jina-v5-topic | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | label | raw | 68 | 4.696 | 0 | 68 |
| upr-eval-zeroshot-jina-v5-topic | 2026-10-20 | val | zeroshot | jina-v5 | top_k k=2 | 0.4848 | 0.5057 | 0.4741 | 0.941 | 0.2661 | 0.7359 | 0.6431 | 0.9018 | 660 | 0.076 |
| upr-labelembed-voyage-4-nano-name | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | label | raw | 68 | 1.189 | 0 | 68 |
| upr-eval-zeroshot-voyage-4-nano-name | 2026-10-20 | val | zeroshot | voyage-4-nano | top_k k=2 | 0.5015 | 0.5149 | 0.4877 | 0.956 | 0.2633 | 0.7357 | 0.6398 | 0.9068 | 660 | 0.253 |
| upr-labelembed-voyage-4-nano-about | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | label | raw | 68 | 1.15 | 0 | 68 |
| upr-eval-zeroshot-voyage-4-nano-about | 2026-10-20 | val | zeroshot | voyage-4-nano | top_k k=2 | 0.5122 | 0.5203 | 0.4998 | 0.956 | 0.2718 | 0.7490 | 0.6587 | 0.8842 | 660 | 0.119 |
| upr-labelembed-voyage-4-nano-topic | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | label | raw | 68 | 1.144 | 0 | 68 |
| upr-eval-zeroshot-voyage-4-nano-topic | 2026-10-20 | val | zeroshot | voyage-4-nano | top_k k=2 | 0.5023 | 0.5191 | 0.4875 | 0.971 | 0.2624 | 0.7419 | 0.6410 | 0.8837 | 660 | 0.118 |
| upr-labelembed-arctic2-name | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | label | raw | 68 | 1.481 | 0 | 68 |
| upr-eval-zeroshot-arctic2-name | 2026-10-20 | val | zeroshot | arctic2 | top_k k=2 | 0.4369 | 0.4412 | 0.4290 | 0.941 | 0.2330 | 0.6692 | 0.5688 | 0.8644 | 660 | 0.064 |
| upr-labelembed-arctic2-about | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | label | raw | 68 | 0.458 | 0 | 68 |
| upr-eval-zeroshot-arctic2-about | 2026-10-20 | val | zeroshot | arctic2 | top_k k=2 | 0.4635 | 0.4741 | 0.4537 | 0.956 | 0.2473 | 0.6995 | 0.6015 | 0.8625 | 660 | 0.062 |
| upr-labelembed-arctic2-topic | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | label | raw | 68 | 0.413 | 0 | 68 |
| upr-eval-zeroshot-arctic2-topic | 2026-10-20 | val | zeroshot | arctic2 | top_k k=2 | 0.4567 | 0.4783 | 0.4477 | 0.941 | 0.2430 | 0.7051 | 0.5918 | 0.8777 | 660 | 0.061 |

## UPR — fusion_c

| run_id | date | split | predictor | embedding | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-eval-fusionc-embeddinggemma-k10-c0.1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5471 | 0.4998 | 0.5267 | 0.868 | 0.2745 | 0.7073 | 0.6699 | 0.7337 | 660 | 0.131 |
| upr-eval-fusionc-embeddinggemma-k10-c0.25 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5479 | 0.4994 | 0.5272 | 0.868 | 0.2752 | 0.7105 | 0.6705 | 0.7337 | 660 | 0.138 |
| upr-eval-fusionc-embeddinggemma-k10-c0.5 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5456 | 0.4978 | 0.5245 | 0.868 | 0.2755 | 0.7115 | 0.6703 | 0.7337 | 660 | 0.157 |
| upr-eval-fusionc-embeddinggemma-k10-c1 | 2026-10-26 | val | fusion | embeddinggemma | top_k k=2 | 0.5479 | 0.5006 | 0.5269 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.129 |
| upr-eval-fusionc-embeddinggemma-k20-c0.1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5418 | 0.4922 | 0.5213 | 0.853 | 0.2742 | 0.7057 | 0.6648 | 0.7337 | 660 | 0.132 |
| upr-eval-fusionc-embeddinggemma-k20-c0.25 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5410 | 0.4956 | 0.5209 | 0.853 | 0.2748 | 0.7067 | 0.6649 | 0.7337 | 660 | 0.134 |
| upr-eval-fusionc-embeddinggemma-k20-c0.5 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5403 | 0.4950 | 0.5203 | 0.853 | 0.2764 | 0.7077 | 0.6662 | 0.7337 | 660 | 0.133 |
| upr-eval-fusionc-embeddinggemma-k20-c1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 | 0.5433 | 0.4980 | 0.5231 | 0.853 | 0.2764 | 0.7086 | 0.6674 | 0.7423 | 660 | 0.136 |
| upr-eval-fusionc-jina-v5-k10-c0.1 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5410 | 0.4844 | 0.5213 | 0.868 | 0.2758 | 0.7083 | 0.6698 | 0.7337 | 660 | 0.186 |
| upr-eval-fusionc-jina-v5-k10-c0.25 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5410 | 0.4854 | 0.5212 | 0.868 | 0.2758 | 0.7129 | 0.6699 | 0.7337 | 660 | 0.201 |
| upr-eval-fusionc-jina-v5-k10-c0.5 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5426 | 0.4876 | 0.5223 | 0.868 | 0.2758 | 0.7150 | 0.6701 | 0.7337 | 660 | 0.197 |
| upr-eval-fusionc-jina-v5-k10-c1 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5403 | 0.4856 | 0.5212 | 0.868 | 0.2773 | 0.7173 | 0.6714 | 0.7423 | 660 | 0.183 |
| upr-eval-fusionc-jina-v5-k20-c0.1 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5395 | 0.4811 | 0.5197 | 0.868 | 0.2764 | 0.7122 | 0.6672 | 0.7337 | 660 | 0.182 |
| upr-eval-fusionc-jina-v5-k20-c0.25 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5403 | 0.4812 | 0.5207 | 0.868 | 0.2773 | 0.7133 | 0.6691 | 0.7337 | 660 | 0.182 |
| upr-eval-fusionc-jina-v5-k20-c0.5 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5395 | 0.4791 | 0.5204 | 0.868 | 0.2779 | 0.7136 | 0.6696 | 0.7337 | 660 | 0.185 |
| upr-eval-fusionc-jina-v5-k20-c1 | 2026-10-20 | val | fusion | jina-v5 | top_k k=2 | 0.5418 | 0.4828 | 0.5228 | 0.868 | 0.2779 | 0.7141 | 0.6705 | 0.7423 | 660 | 0.188 |
| upr-eval-fusionc-voyage-4-nano-k10-c0.1 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5372 | 0.4836 | 0.5177 | 0.868 | 0.2721 | 0.7138 | 0.6640 | 0.7337 | 660 | 0.254 |
| upr-eval-fusionc-voyage-4-nano-k10-c0.25 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5380 | 0.4839 | 0.5182 | 0.868 | 0.2721 | 0.7134 | 0.6646 | 0.7337 | 660 | 0.245 |
| upr-eval-fusionc-voyage-4-nano-k10-c0.5 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5395 | 0.4872 | 0.5197 | 0.868 | 0.2724 | 0.7141 | 0.6660 | 0.7428 | 660 | 0.268 |
| upr-eval-fusionc-voyage-4-nano-k10-c1 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5395 | 0.4926 | 0.5197 | 0.868 | 0.2748 | 0.7256 | 0.6699 | 0.8193 | 660 | 0.272 |
| upr-eval-fusionc-voyage-4-nano-k20-c0.1 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5372 | 0.4923 | 0.5175 | 0.868 | 0.2721 | 0.7116 | 0.6643 | 0.7337 | 660 | 0.247 |
| upr-eval-fusionc-voyage-4-nano-k20-c0.25 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5388 | 0.4948 | 0.5193 | 0.868 | 0.2715 | 0.7115 | 0.6642 | 0.7337 | 660 | 0.251 |
| upr-eval-fusionc-voyage-4-nano-k20-c0.5 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5395 | 0.4970 | 0.5202 | 0.868 | 0.2721 | 0.7130 | 0.6654 | 0.7428 | 660 | 0.248 |
| upr-eval-fusionc-voyage-4-nano-k20-c1 | 2026-10-20 | val | fusion | voyage-4-nano | top_k k=2 | 0.5388 | 0.4979 | 0.5196 | 0.868 | 0.2745 | 0.7160 | 0.6680 | 0.8186 | 660 | 0.25 |
| upr-eval-fusionc-arctic2-k10-c0.1 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5418 | 0.4881 | 0.5216 | 0.868 | 0.2739 | 0.7117 | 0.6663 | 0.7337 | 660 | 0.154 |
| upr-eval-fusionc-arctic2-k10-c0.25 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5426 | 0.4900 | 0.5223 | 0.868 | 0.2733 | 0.7109 | 0.6653 | 0.7337 | 660 | 0.147 |
| upr-eval-fusionc-arctic2-k10-c0.5 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5403 | 0.4882 | 0.5205 | 0.868 | 0.2736 | 0.7119 | 0.6653 | 0.7337 | 660 | 0.174 |
| upr-eval-fusionc-arctic2-k10-c1 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5395 | 0.4878 | 0.5194 | 0.868 | 0.2745 | 0.7121 | 0.6663 | 0.7393 | 660 | 0.149 |
| upr-eval-fusionc-arctic2-k20-c0.1 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5380 | 0.4753 | 0.5177 | 0.868 | 0.2736 | 0.7120 | 0.6629 | 0.7337 | 660 | 0.151 |
| upr-eval-fusionc-arctic2-k20-c0.25 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5380 | 0.4745 | 0.5177 | 0.868 | 0.2742 | 0.7125 | 0.6632 | 0.7337 | 660 | 0.151 |
| upr-eval-fusionc-arctic2-k20-c0.5 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5380 | 0.4744 | 0.5177 | 0.868 | 0.2742 | 0.7125 | 0.6625 | 0.7337 | 660 | 0.146 |
| upr-eval-fusionc-arctic2-k20-c1 | 2026-10-20 | val | fusion | arctic2 | top_k k=2 | 0.5372 | 0.4745 | 0.5171 | 0.868 | 0.2745 | 0.7133 | 0.6631 | 0.7393 | 660 | 0.149 |

## UPR — thresholds

| run_id | date | split | predictor | embedding | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-eval-thr-embeddinggemma-tk1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=1 | 0.4878 | 0.4082 | 0.5004 | 0.794 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.163 |
| upr-eval-thr-embeddinggemma-tk3 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=3 | 0.5049 | 0.4646 | 0.4774 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.141 |
| upr-eval-thr-embeddinggemma-st0.75 | 2026-10-20 | val | fusion | embeddinggemma | thr=0.75 | 0.2614 | 0.2362 | 0.3047 | 0.882 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.143 |
| upr-eval-thr-embeddinggemma-tk2m0.75 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=0.75 | 0.5491 | 0.5029 | 0.5289 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.151 |
| upr-eval-thr-embeddinggemma-st0.9 | 2026-10-20 | val | fusion | embeddinggemma | thr=0.9 | 0.3779 | 0.3477 | 0.4046 | 0.882 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.131 |
| upr-eval-thr-embeddinggemma-tk2m0.9 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=0.9 | 0.5520 | 0.5081 | 0.5348 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.134 |
| upr-eval-thr-embeddinggemma-st1 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.0 | 0.4413 | 0.4142 | 0.4594 | 0.882 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.138 |
| upr-eval-thr-embeddinggemma-tk2m1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.0 | 0.5554 | 0.5133 | 0.5393 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.15 |
| upr-eval-thr-embeddinggemma-st1.1 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.1 | 0.4889 | 0.4581 | 0.4926 | 0.882 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.141 |
| upr-eval-thr-embeddinggemma-tk2m1.1 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.1 | 0.5597 | 0.5179 | 0.5431 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.152 |
| upr-eval-thr-embeddinggemma-st1.25 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.25 | 0.5484 | 0.5138 | 0.5357 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.182 |
| upr-eval-thr-embeddinggemma-tk2m1.25 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.25 | 0.5756 | 0.5334 | 0.5546 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.132 |
| upr-eval-thr-embeddinggemma-st1.4 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.4 | 0.5746 | 0.5263 | 0.5434 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.133 |
| upr-eval-thr-embeddinggemma-tk2m1.4 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.4 | 0.5778 | 0.5253 | 0.5484 | 0.853 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.142 |
| upr-eval-thr-embeddinggemma-st1.5 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.5 | 0.5917 | 0.5278 | 0.5486 | 0.853 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.145 |
| upr-eval-thr-embeddinggemma-tk2m1.5 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.5 | 0.5841 | 0.5229 | 0.5471 | 0.853 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.132 |
| upr-eval-thr-embeddinggemma-st1.6 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.6 | 0.5943 | 0.5210 | 0.5437 | 0.853 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.142 |
| upr-eval-thr-embeddinggemma-tk2m1.6 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.6 | 0.5852 | 0.5172 | 0.5417 | 0.853 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.146 |
| upr-eval-thr-embeddinggemma-st1.75 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.75 | 0.5988 | 0.5201 | 0.5379 | 0.853 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.156 |
| upr-eval-thr-embeddinggemma-tk2m1.75 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.75 | 0.5842 | 0.5106 | 0.5322 | 0.853 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.142 |
| upr-eval-thr-embeddinggemma-st1.9 | 2026-10-20 | val | fusion | embeddinggemma | thr=1.9 | 0.5824 | 0.4829 | 0.5113 | 0.838 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.151 |
| upr-eval-thr-embeddinggemma-tk2m1.9 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=1.9 | 0.5711 | 0.4752 | 0.5088 | 0.838 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.138 |
| upr-eval-thr-embeddinggemma-st2 | 2026-10-20 | val | fusion | embeddinggemma | thr=2.0 | 0.5772 | 0.4710 | 0.5018 | 0.794 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.146 |
| upr-eval-thr-embeddinggemma-tk2m2 | 2026-10-20 | val | fusion | embeddinggemma | top_k k=2 min=2.0 | 0.5645 | 0.4617 | 0.4982 | 0.794 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 660 | 0.146 |

## UPR — preprocess_ablation

| run_id | date | backend | model | model_version | dim | ctx | kinds | preprocess | n_texts | seconds | cache_hits | cache_misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-embed-embeddinggemma-strip | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | document,query | strip_leading_formulaic | 3000 | 26.317 | 0 | 6000 |
| upr-labelembed-embeddinggemma-strip-about | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | label | strip_leading_formulaic | 68 | 0.277 | 0 | 68 |
| upr-eval-abl-embeddinggemma-strip | 2026-10-20 | val | fusion | embeddinggemma-strip | top_k k=2 | 0.5388 | 0.4817 | 0.5171 | 0.853 | 0.2782 | 0.7133 | 0.6666 | 0.7514 | 660 | 0.14 |
| upr-embed-embeddinggemma-lower | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | document,query | lowercase | 3000 | 24.77 | 0 | 6000 |
| upr-labelembed-embeddinggemma-lower-about | 2026-10-20 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | label | lowercase | 68 | 0.289 | 0 | 68 |
| upr-eval-abl-embeddinggemma-lower | 2026-10-20 | val | fusion | embeddinggemma-lower | top_k k=2 | 0.5350 | 0.4866 | 0.5155 | 0.853 | 0.2748 | 0.7087 | 0.6662 | 0.7429 | 660 | 0.142 |
| upr-embed-jina-v5-strip | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | document,query | strip_leading_formulaic | 3000 | 21.944 | 0 | 6000 |
| upr-labelembed-jina-v5-strip-about | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | label | strip_leading_formulaic | 68 | 4.532 | 0 | 68 |
| upr-eval-abl-jina-v5-strip | 2026-10-20 | val | fusion | jina-v5-strip | top_k k=2 | 0.5486 | 0.4956 | 0.5275 | 0.868 | 0.2755 | 0.7155 | 0.6705 | 0.7337 | 660 | 0.183 |
| upr-embed-jina-v5-lower | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | document,query | lowercase | 3000 | 19.656 | 0 | 6000 |
| upr-labelembed-jina-v5-lower-about | 2026-10-20 | hf | jinaai/jina-embeddings-v5-text-small | dd76d535f544 | 1024 | 32768 | label | lowercase | 68 | 4.689 | 0 | 68 |
| upr-eval-abl-jina-v5-lower | 2026-10-20 | val | fusion | jina-v5-lower | top_k k=2 | 0.5395 | 0.4905 | 0.5197 | 0.868 | 0.2764 | 0.7170 | 0.6738 | 0.7337 | 660 | 0.185 |
| upr-embed-voyage-4-nano-strip | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | document,query | strip_leading_formulaic | 3000 | 7.356 | 0 | 6000 |
| upr-labelembed-voyage-4-nano-strip-about | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | label | strip_leading_formulaic | 68 | 1.164 | 0 | 68 |
| upr-eval-abl-voyage-4-nano-strip | 2026-10-20 | val | fusion | voyage-4-nano-strip | top_k k=2 | 0.5410 | 0.4958 | 0.5217 | 0.868 | 0.2761 | 0.7157 | 0.6713 | 0.7463 | 660 | 0.234 |
| upr-embed-voyage-4-nano-lower | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | document,query | lowercase | 3000 | 7.274 | 0 | 6000 |
| upr-labelembed-voyage-4-nano-lower-about | 2026-10-20 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | label | lowercase | 68 | 1.198 | 0 | 68 |
| upr-eval-abl-voyage-4-nano-lower | 2026-10-20 | val | fusion | voyage-4-nano-lower | top_k k=2 | 0.5350 | 0.4921 | 0.5167 | 0.868 | 0.2730 | 0.7120 | 0.6663 | 0.7345 | 660 | 0.24 |
| upr-embed-arctic2-strip | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | document,query | strip_leading_formulaic | 3000 | 32.367 | 0 | 6000 |
| upr-labelembed-arctic2-strip-about | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | label | strip_leading_formulaic | 68 | 0.361 | 0 | 68 |
| upr-eval-abl-arctic2-strip | 2026-10-20 | val | fusion | arctic2-strip | top_k k=2 | 0.5456 | 0.4851 | 0.5246 | 0.868 | 0.2733 | 0.7054 | 0.6694 | 0.7337 | 660 | 0.139 |
| upr-embed-arctic2-lower | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | document,query | lowercase | 3000 | 31.405 | 0 | 6000 |
| upr-labelembed-arctic2-lower-about | 2026-10-20 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | label | lowercase | 68 | 0.48 | 0 | 68 |
| upr-eval-abl-arctic2-lower | 2026-10-20 | val | fusion | arctic2-lower | top_k k=2 | 0.5479 | 0.4937 | 0.5262 | 0.868 | 0.2767 | 0.7135 | 0.6706 | 0.7337 | 660 | 0.149 |

## UPR — thresholds_refine

| run_id | date | split | predictor | embedding | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-eval-thr-jina-v5-st1.9 | 2026-10-20 | val | fusion | jina-v5-strip | thr=1.9 | 0.5880 | 0.5119 | 0.5455 | 0.868 | 0.2755 | 0.7155 | 0.6705 | 0.7337 | 660 | 0.186 |
| upr-eval-thr-jina-v5-st2.1 | 2026-10-20 | val | fusion | jina-v5-strip | thr=2.1 | 0.5906 | 0.5099 | 0.5372 | 0.838 | 0.2755 | 0.7155 | 0.6705 | 0.7337 | 660 | 0.205 |
| upr-eval-thr-jina-v5-st2.3 | 2026-10-20 | val | fusion | jina-v5-strip | thr=2.3 | 0.5836 | 0.4887 | 0.5265 | 0.824 | 0.2755 | 0.7155 | 0.6705 | 0.7337 | 660 | 0.193 |
| upr-eval-thr-jina-v5-tk2m1.75 | 2026-10-20 | val | fusion | jina-v5-strip | top_k k=2 min=1.75 | 0.5692 | 0.5003 | 0.5443 | 0.868 | 0.2755 | 0.7155 | 0.6705 | 0.7337 | 660 | 0.213 |
| upr-eval-thr-jina-v5-tk2m1.9 | 2026-10-20 | val | fusion | jina-v5-strip | top_k k=2 min=1.9 | 0.5726 | 0.4949 | 0.5406 | 0.853 | 0.2755 | 0.7155 | 0.6705 | 0.7337 | 660 | 0.188 |
| upr-eval-thr-jina-v5-tk2m2.1 | 2026-10-20 | val | fusion | jina-v5-strip | top_k k=2 min=2.1 | 0.5710 | 0.4908 | 0.5283 | 0.838 | 0.2755 | 0.7155 | 0.6705 | 0.7337 | 660 | 0.182 |

## UPR — phase4_final

| run_id | date | split | predictor | embedding | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-eval-best-embeddinggemma-st1.75-test | 2026-10-20 | test | fusion | embeddinggemma | thr=1.75 | 0.5979 | 0.5287 | 0.5430 | 0.853 | 0.2720 | 0.7110 | 0.6647 | 0.7378 | 664 | 0.134 |

## UPR — upr_learned_catalog

| run_id | date | split | predictor | embedding | f1 decision | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-labelembed-embeddinggemma-desc | 2026-10-27 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | label | raw | 68 | 0.079 | 68 | 0 |
| upr-eval-fusionc-embeddinggemma-k10-c1-desc | 2026-10-27 | val | fusion | embeddinggemma | top_k k=2 | 0.5456 | 0.4959 | 0.5248 | 0.868 | 0.2755 | 0.7093 | 0.6702 | 0.7352 | 660 | 0.182 |
