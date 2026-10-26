# UPR text-processing experiment tables

Comparison tables for the text-processing track (`TEXT_PROCESSING_PLAN.md`), keyed by run — auto-appended by the track's CLI commands. The append-only narrative log is `docs/experiments_text_processing.md`. Values come from each run's own artifacts; counts sit next to numbers; "n/a" = not reported by the run.

## UPR — upr_analysis

| run_id | date | split | arm | predictor | chunker | aggregation | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | hit@1 | hit@10 | gold_rank_p50 | bg_rank_p50 | bg_mean | bg_mean_delta | bg_rank_delta | chunks_per_text | n_texts | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| upr-analyze-champion | 2026-10-26 | val | full_text | fusion | full_text | n/a | 0.5479 | 0.5006 | 0.5269 | 0.868 | 0.2770 | 0.7126 | 0.6725 | 0.7423 | 0.3666 | 0.7462 | 0.0147 | 0.5000 | 0.5526 | n/a | n/a | n/a | 660 | 0.163 |
