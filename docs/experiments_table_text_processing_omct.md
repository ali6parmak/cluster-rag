# OMCT text-processing experiment tables

Comparison tables for the text-processing track (`TEXT_PROCESSING_PLAN.md`), keyed by run — auto-appended by the track's CLI commands. The append-only narrative log is `docs/experiments_text_processing.md`. Values come from each run's own artifacts; counts sit next to numbers; "n/a" = not reported by the run.

## OMCT — analyze

| run_id | date | backend | model | model_version | dim | ctx | transform | chunker | pair_auc | ctrl_mean | ctrl_p50 | pos_rank_p50 | neg_rank_p50 | ctrl_rank_p50 | n_pairs | n_control | n_docs | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| omct-analyze-embeddinggemma | 2026-10-23 | ollama | embeddinggemma:latest | 82f094e4c0e1 | 768 | 2048 | strip_block | full_text | 0.5371 | 0.2194 | 0.2130 | 0.0783 | 0.2488 | 0.5042 | 1520 | 4000 | 3 | 0.123 |
| omct-analyze-voyage-4-nano | 2026-10-23 | hf | voyageai/voyage-4-nano | 433ece5e27a1 | 2048 | 32768 | strip_block | full_text | 0.5004 | 0.2351 | 0.2297 | 0.0980 | 0.2058 | 0.5023 | 1520 | 4000 | 3 | 3.377 |
| omct-analyze-arctic2 | 2026-10-23 | ollama | snowflake-arctic-embed2:latest | 30685f516da4 | 1024 | 8192 | strip_block | full_text | 0.4680 | 0.2436 | 0.2386 | 0.0968 | 0.2066 | 0.5021 | 1520 | 4000 | 3 | 0.089 |
