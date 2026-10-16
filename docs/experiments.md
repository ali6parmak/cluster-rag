# Experiment log

Append-only, one row per run (see AGENTS.md). Never delete or edit old rows.
Every row must correspond to a `results/<run_id>/` folder containing the frozen
(resolved) config and the metrics. Record the embedder model id and revision /
digest in the notes.

| date | run_id | phase | embedder | preprocess | method | split | P@5 | R@10 | micro-F1 | macro-F1 | shortlist R@50 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Dataset statistics (Phase 1)

- **Run:** `base` (2026-10-15), report: `results/base/stats.json`
- **Source:** `data/raw/upr.jsonl`, sha256 `93ba79cc0104c4cb1529864a4053e1c18497050ab135ec7ddb2d067649e9e50f`, **3000 records, 0 malformed lines dropped**
- **Labels:** 68 unique, 5722 total occurrences, labels/record mean 1.91 (min 1, max 4)
- **Languages:** `en` only
- **Text length:** mean 151.3 chars (min 22, max 738)
- **Frequency buckets** (head_min=100, torso_min=40): head **16**, torso **34**, tail **18**. Not long-tailed: every label has >= 19 examples. Top: Women's rights (485), Rights of the Child (459), International instruments (345)
- **Top label co-occurrences:** Rights of the Child + Women's rights (204), Gender-based violence + Women's rights (70), Labour rights + Migrants (52)
- **Formulaic prefixes confirmed** (embedding-bias risk): "continue" opens 535 texts (17.8%), "continue to" 180, "ensure that" 83, "take measures to" 33
- Implication for PLAN.md section 2 assumptions: single language; label set is complete-ish (no zero/1-4-example labels), so zero-shot/new-label work is a controlled experiment via the label-holdout split, not a data reality

## Fixed splits (Phase 1)

- **Run:** `base` (2026-10-15), manifest: `data/splits/manifest.json` + `results/base/split_report.json`; ids in `data/splits/{train,val,test}_ids.json` (committed)
- **Method:** multi-label iterative stratification (`iterative-stratification` 0.1.9, two-stage via `MultilabelStratifiedShuffleSplit`, seeds `seed`, `seed+1`, holdout redistribution `seed+2`), 70/15/15, then deterministic coverage rebalance (min_total=3) and greedy min-collateral label-holdout selection
- **Sizes:** train **1676**, val **660**, test **664** of 3000. Nominal 70/15/15 is 2100/450/450; 424 records containing held-out labels were redistributed stratified-ly into val/test (so val/test intentionally exceed the nominal share). Coverage moves needed: **0** (every label present in val and test)
- **Label holdout** (8 labels, removed entirely from train; eligibility pool 30-80 examples, chosen by least collateral then most examples): Freedom of religion and belief (80), Gender-based violence (78), General (77), Freedom of association and peaceful assembly (76), Asylum-seekers - refugees (73), ESC rights - general (73), Other (73), Special procedures (72). Each lands ~50/50 in val/test (e.g. 40/40)
- **Integrity:** 0 holdout labels in train ids; **0 accidental labels absent from train** (holdout set == `labels_missing_from_train`); rerun with `--force` reproduces identical ids (same sha256 of the source file, seed 42). Note "General" and "Other" are catch-all categories picked procedurally; adjust `splits.label_holdout` bounds in config if only topical labels should be held out

## Auxiliary pairwise dataset statistics (Phase 1.5)

- **Run:** `aux-base` (2026-10-16), report: `results/aux-base/pairs_stats.json`
- **Source:** `data/external/omct_pairs.json` (`omct_gti_paragraph_question_golden_pairs`, collaborator-provided), sha256 `70a42ba2a42434f0...`
- **Pairs:** 1520 valid, 0 malformed; **524 positive / 996 negative (34.5% positive)**
- **Distinct questions:** 294 question numbers, 294 distinct texts (mean 650 chars; they include the "Question: ... Answer choices: ..." block — stripping to analyze, see open question)
- **Documents (3):** UN-style "Concluding observations ... Colombia" 845 pairs (211 positive, 25%), "Human rights in Colombia Amnesty International" 99 pairs (79 positive, 80%), "Observaciones preliminares colombia 2024" 576 pairs (234 positive, 41%) — per-document class balance differs strongly
- **Paragraph length:** mean 889.8 chars (min 32, max 2255) — much longer than UPR texts; embedding truncation must be checked in Phase 2
- **Zero duplicate ids; zero missing question texts**; labels are human-reviewed (metadata: positives from merged Questions column, negatives reviewer-confirmed non-links)