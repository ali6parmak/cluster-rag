# Experiment log

Append-only, one row per run (see AGENTS.md). Never delete or edit old rows.
Every row must correspond to a `results/<run_id>/` folder containing the frozen
(resolved) config and the metrics. Record the embedder model id and revision /
digest in the notes.

| date | run_id | phase | embedder | preprocess | method | split | P@5 | R@10 | micro-F1 | macro-F1 | shortlist R@50 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|