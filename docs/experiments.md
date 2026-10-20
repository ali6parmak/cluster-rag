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

## Phase 2: embedding layer (backends, cache, preprocessing)

### Full-dataset embed runs (Phase 2 exit criterion: ≥ 2 backends, cache hits skip re-encoding)

- **Run:** `embed-e5-base-v2` (2026-10-16, `results/embed-e5-base-v2/`, config `configs/experiments/embed_e5_base_v2.yaml`) — **HF backend**, `intfloat/e5-base-v2`, revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, dim 768, max_seq_length 512, device cuda:0, prefixes query=`query: ` / document=`passage: ` / label=`query: `. First pass: 3000 texts encoded in **4.94 s** (3000 misses/stored); **rerun: 3000 cache hits, 0 encoded, 3.45 s** (meta.json in the results folder is the rerun = the cache-hit proof artifact). Output: `embeddings.npz` (ids + `vectors_document`).
- **Run:** `embed-arctic2` (2026-10-16, `results/embed-arctic2/`, config `configs/experiments/embed_arctic2.yaml`) — **Ollama backend**, `snowflake-arctic-embed2:latest` (digest `30685f516da4...`), dim 1024, ctx 8192, prefix query=`query: ` / document="". 3000 texts via batched `POST /api/embed` in **16.27 s**; **rerun: 3000 cache hits, 0 encoded, 0.43 s** (meta.json shows the hit stats). The arctic-embed-v2 model card instructs query-prefix only; verified before the run.
- Cache layout: `cache/v1/<model_id>/<preprocessing_version>/<prefix>/<sha1(text)>.npy` (URL-quoted components), one atomic `.npy` per vector; cache-hit correctness proven by unit test (hits make zero model calls) and end-to-end above.

### Preprocessing variants
- `raw`, `lowercase`, `strip_leading_formulaic` implemented (`src/cluster_rag/preprocess.py`); openers config-driven and hashed into the preprocessing version (cache honesty). On the real UPR texts the default opener list (top-8 lead verbs + connector forms from the Phase 1 prefix analysis) **modifies 58.0% of records** (1739/3000, mean 10.7 chars), including all 535 "continue..." texts (17.8%). The effect on UPR quality itself is measured in Phase 4 ablations (embedding × preprocessing).

### Auxiliary probe (Phase 2, last checkbox): 6 candidate embedders × question transform

Scope: `data/external/omct_pairs.json` only (auxiliary, never a UPR metric). Corpus: 1520 judged pairs (524 positive / 996 negative), 225 distinct paragraphs (deduplicated by text), 294 distinct questions. Two question transforms compared per model: `keep` (raw question incl. "Question:/Answer choices:"/Notes: furniture) vs `strip_block` (drop it). Question = query side, paragraph = document side (models with asymmetric instructions prefixed accordingly, per model cards). Token counts: exact (HF tokenizer) or chars/4 estimate (Ollama, documented per report).

| run_id (aux-prefix) | model (backend) | dim | ctx | keep→strip pair-AUC | keep→strip hit@1 same-doc | keep→strip hit@10 same-doc | recall@10 best | paragraphs over ctx |
|---|---|---|---|---|---|---|---|---|
| `aux-probe-minilm` | all-MiniLM-L6-v2 (hf) | 384 | 256 | 0.4190 → 0.4526 | 0.355 → 0.381 | 0.723 → 0.748 | 0.508–0.529 | **4/225 (1.8%)** |
| `aux-probe-e5-base-v2` | intfloat/e5-base-v2 (hf) | 768 | 512 | 0.4921 → **0.5793** | 0.271 → 0.374 | 0.723 → 0.774 | 0.485–0.540 | 0 |
| `aux-probe-bge-m3` | BAAI/bge-m3 (hf) | 1024 | 8192 | 0.4501 → 0.4687 | 0.284 → 0.361 | 0.690 → 0.748 | 0.465–0.476 | 0 |
| `aux-probe-arctic2` | snowflake-arctic-embed2 (ollama) | 1024 | 8192 | 0.4512 → 0.4680 | 0.355 → 0.355 | **0.826** → 0.813 | 0.570–0.593 | 0 |
| `aux-probe-embeddinggemma` | embeddinggemma:latest (ollama, Q4_0 QAT) | 768 | 2048 | 0.5238 → 0.5371 | 0.400 → **0.458** | 0.826 → 0.794 | 0.582–0.650 | 0 |
| `aux-probe-mxbai` | mxbai-embed-large:latest (ollama) | 1024 | 512 | 0.4871 → 0.5552 | 0.426 → 0.413 | 0.781 → 0.768 | 0.558–0.582 | **1/225 (0.4%)** |

Digests/revisions recorded in each `results/aux-probe-*/probe_report.json` (ollama digests `30685f...` arctic2, `3a2d26...` gemma, `468836...` mxbai; HF revisions pinned in the configs).

**Findings:**
1. **Question furniture is mostly noise → `strip_block` adopted as the default for question-side text** (PLAN §7 open question answered): pair-AUC improves for **6/6** embedders (+0.013 … +0.087; strongest e5 +0.087, mxbai +0.068), same-doc hit@1 for 4/6 (embeddinggemma +0.058, e5 +0.103); a few same-doc hit@10 drop (arctic2 −0.013, embeddinggemma −0.032) so the block occasionally carries topical signal (answer-choice words), but the AUC direction is consistent everywhere.
2. **Truncation is not a blocker at current sizes**: max paragraph ≈ 564 est. tokens (chars/4) / 478 exact (bge-m3 tokenizer) — only MiniLM (256 ctx) truncates 1.8% of paragraphs and mxbai (512) 0.4%; even those over-context rows returned their gold question in the top-10 (n=4: hit@10 1.000; n=1: hit@10 1.000). Long-context models show 0% truncation.
3. **Absolute AUCs are modest (0.42–0.58)**: bi-encoder paragraph↔question discrimination on this corpus is hard (negatives are heavily same-document, topically adjacent); treat the numbers as *relative model evidence*, not absolute retrieval quality. Under `strip_block`: e5-base-v2 (0.579) and mxbai (0.555) lead on AUC; arctic2/embeddinggemma lead same-document hit@10 (0.81/0.79).
4. **Candidate pool (6, all already local — zero downloads; PLAN said "pick 3-5")**: MiniLM (fast baseline), e5-base-v2 (prefix asymmetry), bge-m3 (long-context HF), arctic2 / embeddinggemma / mxbai (Ollama). No single model dominates every column; the UPR `evaluate` harness (Phase 3/4) is the next filter. embeddinggemma is strong at hit@1 (best 0.458 same-doc) despite Q4 quantization.
5. 225 distinct paragraphs: dedup-by-text matters before embedding, but *within* documents only — paragraph texts do **not** repeat across documents (**0 of 225** appear in more than one document; verified on the data in the Phase 2.5 runs, counts recorded in the `aux-base` pairs_stats rerun). What repeats across documents are the questions: **186 of 294** distinct question texts are judged in more than one document — handled by one column per (transformed) text + per-document doc sets + per-paragraph golds, so the all-scope never counts them wrong. *(original wording "paragraph texts repeat across documents/paragraph numbers" was wrong; corrected 2026-10-19)*

**Next:** Phase 3 evaluation harness (metrics incl. the pairwise-retrieval functions above, shortlist recall, trivial baselines) before any Phase 4 modeling. *(the commit-able summary artifact originally planned here — `results/summaries/phase2_aux_probe.json` — was replaced in Phase 2.5 by the auto-appended comparison tables in `docs/experiments_table.md`)*

## Phase 2.5: review touch-ups (runs under the final code)

Datasets: `embed-*` rows embed the UPR full dataset (all 3000 records, no split); `aux-probe2-*`/`aux-base` rows are the auxiliary OMCT pairwise dataset (`aux-` prefix, never a UPR metric).

### Cache rebuild under the readable slug layout (item 1)

- Cache wiped and rebuilt: slugs are `[A-Za-z0-9._-]` with `_` for everything else, model dirs gain a short `sha1(model_id)` prefix (e.g. `BAAI/bge-m3` → `2022e1cf-BAAI_bge-m3`, `query: ` → `query__`); key semantics `(model_id, prefix, preprocessing_version, sha1(text))` unchanged, cache format stays `v1`.
- **Run:** `embed-e5-base-v2`, `configs/experiments/embed_e5_base_v2.yaml` — UPR dataset, 3000 texts: cold pass 3000 encoded **7.50 s** (GPU); warm rerun **3000 cache hits, 0 encoded, 3.47 s** (meta.json = the warm pass; same run_id as Phase 2, first-pass numbers there are the 4.9 s HF pass of 2026-10-16).
- **Run:** `embed-arctic2`, `configs/experiments/embed_arctic2.yaml` — Ollama `snowflake-arctic-embed2:latest` (digest `30685f516da4...`): cold 3000 via batched `POST /api/embed` **17.20 s**; warm **3000 hits, 0 encoded, 0.42 s**.
- Both warm passes prove the slug rewrite end-to-end: same texts, same content keys, hits under the new layout make zero model calls.

### Probe reruns: strip_block default + per-(document, source_csv) breakdown

Sweep via `cluster-rag sweep --file experiments/aux_probe_sweep.py` (6 cells → `results/aux-probe2-<model>/`); corpus 1520 judged pairs (524 positive / 996 negative), 225 paragraph rows, 294 question columns, strip_block transform. **All six overall numbers reproduce the Phase 2 strip_block results digit-for-digit** — pair-AUC minilm 0.4526 / e5 0.5793 / bge-m3 0.4687 / arctic2 0.4680 / embeddinggemma 0.5371 / mxbai 0.5552, same-doc hit@1 0.381 / 0.374 / 0.361 / 0.355 / 0.458 / 0.413, same-doc hit@10 0.748 / 0.774 / 0.748 / 0.813 / 0.794 / 0.768 — so the rebuilt vectors are identical to the Phase 2 runs; the comparison tables live in `docs/experiments_table.md` (probe-aux + probe-aux groups, counts next to numbers).

One per-group observation worth carrying forward (aux evidence, relative not absolute): the Amnesty source document (99 pairs, 79 positive = 80%) separates sharply for every model (AUC 0.52–0.76, same-doc hit@10 0.86–0.97), while the UN-style NU document (845 pairs, 211 positive = 25%) is where the "hard" same-document topically adjacent negatives live (AUC 0.42–0.48 for most models). This matches the Phase 2 decision: overall AUC is OMCT-pairs-only relative evidence, no gating at 0.5, model elimination happens on UPR val in Phase 3/4.

### Docs correction (item 6, finding 5 above)

Verified on `data/external/omct_pairs.json` and recorded in the `aux-base` rerun (`results/aux-base/pairs_stats.json`, field `n_paragraph_texts_in_multiple_documents` / `n_question_texts_in_multiple_documents`): **0 of 225** paragraph texts appear in more than one document (they repeat only within documents, against different questions); **186 of 294** question texts are judged in more than one document. `summarize_pairs` now carries both counts.

### Housekeeping

- Deleted `results/summaries/phase2_aux_probe.json` (its content moved to `docs/experiments_table.md` + this narrative; `docs/experiments.md` remains the append-only log).
- New CLI surface for Phase 3+: `cluster-rag sweep --file <sweep.py>` (typed cells → frozen per-run configs; cells only produce configs).

**Next:** Phase 3 evaluation harness, before any Phase 4 modeling.

## Phase 3: evaluation harness (built + trivial-baseline runs, 2026-10-19)

**Built (all pure/`numpy`, heavy imports only inside constructors; 43 new unit tests, suite 179 green, ruff clean):**
- `eval/multilabel.py`: stable-descending ranking (ties → earlier column, same convention as the pairwise metrics); `ranking_metrics` P@k / R@k / nDCG@k (binary gains 2^rel−1, log2 discount, IDCG from min(n_gold, k) ideal ones, raw score ranking); `apply_decision_rule` with one operating point per run (top_k | score_threshold | top_k_min_score); `decision_f1_metrics` micro/macro/sample-F1 (F1 = 0 on zero division, explicitly reported; macro averaged over labels with ≥1 gold in view, goldless ones counted, not averaged; micro TP/FP/FN over the full masked population so a record whose gold lies outside a bucket still contributes its in-bucket false positives); `shortlist_recall` gold-cell recall@m — **full-view**: the ranking is always global, `cell_mask` only restricts which gold cells count; `label_coverage` from the decision matrix (covered = ever suggested, denominator = labels with ≥1 gold in the split).
- `eval/buckets.py`: head/torso/tail from **full-dataset counts** (identical to the Phase 1 `stats` artifact, thresholds from the stats config) + `unseen` = absent from train, taking precedence.
- `eval/report.py`: one `overall` section (same code path, no mask) + one restricted-view section per bucket; every number carries n_records / n_rows_with_gold / n_gold_cells / n_labels_in_macro counts.
- `predictors/`: `Predictor` protocol (`score(texts) -> (n, n_labels)` matrix + `describe()` lineage; construction context carries the label space and **train-only records** — the leakage boundary), trivial baselines `MostFrequentPredictor` (train-count scores, constant across records, never covers unseen/tail below them) and `RandomPredictor` (seeded uniform), registry `build_predictor`; `eval:` config schema (predictor + decision + metrics + split). Sweep cells may run `evaluate`.
- Exit criterion: `cluster-rag evaluate --config <cfg> --split val|test` writes `results/<run_id>/{config.yaml, metrics.json}` + the auto "Phase 3 — evaluate" table row; split ids come from `data/splits/` with a source-sha256 consistency check (never re-split).

**Runs** (`experiments/baseline_eval_sweep.py`, 8 cells, pure CPU, 0.2 s total; `mf` = most_frequent): {mf, random} × top-k k∈{1,2,3} on val, then the k=2 pair on test once. Buckets on the evaluated splits: head 16 / torso 26 / tail 18 / unseen 8 labels (all 8 holdout labels were torso-range, 72–80 examples, so torso lost 8 to unseen). **All 68 labels have val/test gold** (coverage_min_total=3), so `n_labels_with_gold` = 68 per eval split.

| run_id | split | overall P@5 / R@10 / nDCG@5 | micro-F1 | macro-F1 | sample-F1 | shortlist R@50 | coverage | head/torso/tail/unseen micro-F1 |
|---|---|---|---|---|---|---|---|---|
| `ev-mf-topk1-val` | val | 0.0936 / 0.3023 / 0.1854 | 0.0862 | 0.0034 | 0.0828 | 0.6764 | 0.015 | 0.137 / 0 / 0 / 0 |
| `ev-mf-topk2-val` | val | 0.0936 / 0.3023 / 0.1854 | 0.1413 | 0.0073 | 0.1327 | 0.6764 | 0.029 | 0.196 / 0 / 0 / 0 |
| `ev-mf-topk3-val` | val | 0.0936 / 0.3023 / 0.1854 | 0.1543 | 0.0100 | 0.1441 | 0.6764 | 0.044 | 0.199 / 0 / 0 / 0 |
| `ev-random-topk1-val` | val | 0.0294 / 0.1496 / 0.0533 | 0.0183 | 0.0170 | 0.0172 | 0.7792 | 1.000 | 0.021 / 0.008 / 0.027 / 0.021 |
| `ev-random-topk2-val` | val | 0.0294 / 0.1496 / 0.0533 | 0.0304 | 0.0248 | 0.0290 | 0.7792 | 1.000 | 0.035 / 0.016 / 0.025 / 0.052 |
| `ev-random-topk3-val` | val | 0.0294 / 0.1496 / 0.0533 | 0.0346 | 0.0294 | 0.0333 | 0.7792 | 1.000 | 0.046 / 0.019 / 0.024 / 0.055 |
| `ev-mf-topk2-test` | test | 0.0940 / 0.2954 / 0.1762 | 0.1433 | 0.0074 | 0.1283 | 0.6808 | 0.029 | 0.199 / 0 / 0 / 0 |
| `ev-random-topk2-test` | test | 0.0274 / 0.1438 / 0.0463 | 0.0234 | 0.0196 | 0.0209 | 0.7308 | 1.000 | 0.033 / 0.010 / 0.008 / 0.043 |

**Harness findings (val unless stated; all from the runs above):**
1. The trivial baselines behave exactly as designed: `most_frequent` predicts the same k head labels for every record — head-only micro-F1, **zero** torso/tail/unseen F1 and coverage; at the val k=2 operating point P@1 0.1288 yet micro-F1 only 0.1413 — it over-predicts popular labels onto usually single-gold records.
2. `random` achieves coverage 1.000 — it suggests every label including all 8 unseen — at tiny F1/P@k; the coverage-vs-precision contrast is precisely what the two views are for.
3. F1 rises with k (0.086 → 0.154 for mf k=1→3) while P@k/R@k/nDCG@k are decision-rule-independent (same ranking) — top-k and threshold views separate cleanly as intended.
4. The unseen bucket has real support: 294 val / 296 test records contain at least one held-out label (test: 296), so Phase 4 zero-shot discovery is evaluable, not a footnote.
5. Shortlist R@50: mf 0.6764 (its constant ranking misses the 18 lowest-count labels entirely — 0.6764 = share of gold cells outside the missed 18 at m=50) vs random 0.7792 (m=50 < 68 leaves ~18 random labels out). These are the reranker-ceiling reference numbers for Phase 4.

Per-run bucket detail, thresholds and predictor lineage live in each `results/ev-*/metrics.json`; table rows auto-appended under "Phase 3 — evaluate".

**Next:** Phase 4 retrieval baselines (kNN / label prototypes / zero-shot label embeddings, fusion + thresholding) against this harness; model-pool decision uses val rows (see PLAN §7).

## Phase 3.5: naming + experiment UX (migration note, 2026-10-20)

Phase 3.5 settled the run-id grammar `<dataset>-<command>-<detail>[-<split>]` (datasets `upr`, `omct`), Python-only typed sweep files, and per-dataset comparison tables (`docs/experiments_table_upr.md`, `docs/experiments_table_omct.md`; see AGENTS.md "Runs and where things live"). All pre-3.5 results were re-run under the new ids — **all warm-cache/CPU, nothing re-encoded, no re-splitting** — and every new artifact was verified against the old one **field-for-field: numbers unchanged, verified** (script diff of the JSON artifacts; exemptions: `run_id`, `*seconds*` wall-clock timings, and the probe reports' `cache`/`probe.cache` hit-miss counters, which necessarily read 0→519 hits / 519→0 misses because the pre-3.5 probe2 passes built the aux cache entries and the migration passes are fully warm). The two embed runs' `embeddings.npz` id lists and vector matrices were additionally compared elementwise — identical.

Old → new mapping (old folders deleted after verification; the sweep file `experiments/aux_probe_sweep.py` was renamed to `experiments/omct_probe_sweep.py`):

| pre-3.5 run_id (folder deleted) | post-3.5 run_id | produced by / artifact |
|---|---|---|
| `base` (Phase 1) | `upr-stats-base` | `sweep --file experiments/upr_migration_sweep.py` → `stats.json` |
| `embed-e5-base-v2` | `upr-embed-e5-base-v2` | same sweep, warm: 3000 hits / 0 encoded → `meta.json` (+ `embeddings.npz`) |
| `embed-arctic2` | `upr-embed-arctic2` | same sweep, warm: 3000 hits / 0 encoded → `meta.json` (+ `embeddings.npz`); Ollama digest `30685f516da4...` re-read from the running server |
| `aux-base` | `omct-stats-base` | `sweep --file experiments/omct_stats_sweep.py` (cell label `stats-base`; the command adds `omct-`) → `pairs_stats.json` |
| `aux-probe2-{minilm,e5-base-v2,bge-m3,arctic2,embeddinggemma,mxbai}` | `omct-probe2-<model>` | `sweep --file experiments/omct_probe_sweep.py` → `probe_report.json`; strip_block numbers identical (e.g. AUC 0.4526/0.5793/0.4687/0.4680/0.5371/0.5552) |
| `ev-{mf,random}-topk{1,2,3}-val`, `ev-{mf,random}-topk2-test` | `upr-eval-{mf,random}-topk{1,2,3}-val`, `upr-eval-{mf,random}-topk2-test` | `sweep --file experiments/baseline_eval_sweep.py` (now typed `EvalCell`s) → `metrics.json` |

The 6 legacy Phase 2 probe folders (`aux-probe-*`, the pre-2.5 originals whose numbers live in the Phase 2 section above) were deleted with the rest — their surviving numbers are in this log, their `probe2-*` successors are the ones that re-ran. The frozen `config.yaml` inside each new folder differs from the pre-3.5 one only in the expected places: the `run_id`, the renamed table path keys (`paths.experiment_table_upr`/`_omct` instead of `paths.experiment_table`), and the `table.sweep_label` key injected by `define_sweep`.

**Legacy-id note:** log rows written before 2026-10-20 keep the ids they were written with (`base`, `embed-*`, `aux-*`, `ev-*`); never rewrite them. Use this mapping to move between the old names as they appear in old rows and the current `results/` folders. Phase 2.5's "Phase 2 — embed/probe-aux" and Phase 3's "Phase 3 — evaluate" headings from the retired single `docs/experiments_table.md` now live per dataset: embeds under `## UPR — migration` and the baselines under `## UPR — baselines` in `docs/experiments_table_upr.md`; the probe grid under `## OMCT — probe2` (+ `- groups`) in `docs/experiments_table_omct.md`.

## Phase 3.6: omct command naming (2026-10-20, same-day follow-up to Phase 3.5)

Reviewer decision: dataset-scoped commands carry the dataset name — `aux` and `pairs` must not name commands, flags or sweep commands. Renames (run-id grammar and prefixing unchanged):

| before (Phase 3.5) | after (Phase 3.6) |
|---|---|
| CLI command `probe-aux` | **`probe-omct`** |
| CLI flag `stats --pairs` | **`stats --omct`** |
| sweep commands `stats-pairs` / `probe-aux` | **`stats-omct` / `probe-omct`** |
| module `aux_probe.py`, test `test_aux_probe.py` | **`omct_probe.py`**, **`test_omct_probe.py`** (git renames) |
| table constants `PROBE_AUX_*` | **`PROBE_OMCT_*`** |
| omct stats artifact `pairs_stats.json` | **`stats.json`** (uniform one-artifact-per-command; `results/omct-stats-base/`) |

Kept deliberately: `pairs.py`/`PairRecord`/`load_pairs`/`summarize_pairs` and pair-count fields in reports (`n_pairs` etc.) — "pair" names the record type (a paragraph↔question pair), not the dataset; `paths.data_external` unchanged; "auxiliary" survives only as the English adjective (the OMCT data *is* auxiliary per PLAN §2). "Omct question transform" wording applies to `preprocess.py`'s `keep`/`strip_block` helpers; transform names unchanged.

Verification: the two omct sweeps re-ran warm (7 runs; frozen `config.yaml` + payload `command` fields refreshed to the new names; run ids `omct-probe2-*`/`omct-stats-base` and the `## OMCT — probe2` headings unchanged — labels own them) and every pre-3.6 artifact diffed **field-for-field: numbers unchanged, verified** (exempt: `run_id`, `*seconds*`, cache counters, the renamed payload `command` field). Historical log rows keep their old command names.

## 2026-10-20 — housekeeping: authored experiment YAMLs retired

`configs/experiments/*.yaml` (6 `probe_*`, 2 `embed_*`, `eval_baseline_most_frequent`) are deleted. They duplicated Python sweep cells since Phase 2.5/3.5 and nothing in code, tests or sweeps read them. `configs/base.yaml` remains the single authored config: the `define_sweep` resolution base and the stats/split standalone entry. No runs were executed and no metrics changed; nothing is recomputed — every run's settings stay reproducible from its frozen `results/<run_id>/config.yaml` (the 18 post-migration + later runs). AGENTS.md layout/"Running things" and the CLI `--config` help strings were updated to point at the sweep files.

## Phase 4 opening: pool widening → 14-embedder OMCT probe (2026-10-20)

Scope: `data/external/omct_pairs.json` only (auxiliary OMCT pairs, never a UPR metric). Corpus unchanged from Phase 2: 1520 judged pairs (524 positive / 996 negative), 225 distinct paragraphs, 294 distinct questions, question transform `strip_block` only (the Phase 2 default). Question = query kind, paragraph = document kind (per-model card prefixes; re-verified at wiring 2026-10-20). New pool entries were wired into `experiments/embedder_pool.py` (shared with the UPR embed sweep); per-model wiring notes at the end of this section. The 6 pre-existing cells re-ran warm: metrics/dims/contexts digit-for-digit identical vs the Phase 3.6-era table — **verified, 204/204 artifact checks** (exempt: `*seconds*`, cache counters). One provenance change among them: `embeddinggemma:latest` was re-pulled before the run and its Ollama digest changed `3a2d262bdb75` → `82f094e4c0e1`; metrics reproduce exactly because the disk cache (keyed by model id, prefix, preprocessing, text) served the Phase 2.5-era stored vectors — the digest field identifies server bytes, the metrics identify the cache lineage. *(Cache keys not including the Ollama digest is a known trade-off: a re-pulled model silently reuses old cached vectors. Fine while re-pull events are rare and logged; revisit if a digest drift produces a metrics change.)*

| run_id (omct-) | model (backend) | dim | ctx | pair-AUC | same-doc hit@1 | hit@10 | recall@10 | paras over ctx |
|---|---|---|---|---|---|---|---|---|
| `omct-probe2-minilm` | all-MiniLM-L6-v2 (hf) | 384 | 256 | 0.4526 | 0.381 | 0.748 | 0.508 | 4/225 (1.8%) |
| `omct-probe2-e5-base-v2` | intfloat/e5-base-v2 (hf) | 768 | 512 | 0.5793 | 0.374 | 0.774 | 0.540 | 0 |
| `omct-probe2-bge-m3` | BAAI/bge-m3 (hf) | 1024 | 8192 | 0.4687 | 0.361 | 0.748 | 0.476 | 0 |
| `omct-probe2-arctic2` | snowflake-arctic-embed2 (ollama) | 1024 | 8192 | 0.4680 | 0.355 | 0.813 | 0.570 | 0 |
| `omct-probe2-embeddinggemma` | embeddinggemma:latest (ollama, Q4 QAT) | 768 | 2048 | 0.5371 | 0.458 | 0.794 | 0.582 | 0 |
| `omct-probe2-mxbai` | mxbai-embed-large:latest (ollama) | 1024 | 512 | 0.5552 | 0.413 | 0.768 | 0.558 | 1/225 (0.4%) |
| `omct-probe2-e5-large-v2` | intfloat/e5-large-v2 (hf) | 1024 | 512 | 0.5112 | 0.400 | 0.826 | 0.594 | 0 |
| `omct-probe2-mpnet` | all-mpnet-base-v2 (hf) | 768 | 384 | 0.4523 | 0.355 | 0.703 | 0.458 | 1/225 (0.4%) |
| `omct-probe2-nomic-v2-moe` | nomic-embed-text-v2-moe:latest (ollama) | 768 | 512 | 0.4492 | 0.400 | 0.703 | 0.463 | 1/225 (0.4%) |
| `omct-probe2-jina-v5` | jinaai/jina-embeddings-v5-text-small (hf) | 1024 | 32768 | 0.5394 | 0.465 | 0.813 | 0.589 | 0 |
| `omct-probe2-voyage-4-nano` | voyageai/voyage-4-nano (hf) | 2048 | 32768 | 0.5004 | 0.355 | 0.716 | 0.474 | 0 |
| `omct-probe2-granite-english-r2` | granite-embedding-english-r2 (hf) | 768 | 8192 | 0.5130 | 0.406 | 0.781 | 0.557 | 0 |
| `omct-probe2-f2llm-80m` | codefuse-ai/F2LLM-v2-80M (hf) | 320 | 40960 | 0.5986 | 0.297 | 0.697 | 0.475 | 0 |
| `omct-probe2-embeddinggemma2` | embeddinggemma-2:latest (ollama) | 768 | 8192* | 0.5815 | 0.374 | 0.755 | 0.564 | 0 |

(8192* = the explicit config context override — see the wiring notes below.)
All-scope `pair_auc` / `same-document` hit@k / recall@k as in each run's
`probe_report.json`; "paras over ctx" = paragraphs exceeding the model's
context (truncation diagnostics).

**Findings (OMCT, auxiliary-only evidence):**
1. **All-scope AUC leader: f2llm-80m (0.5986)** — the smallest pool model, with its Instruct-prefixed queries; runner-ups embeddinggemma2 (0.5815) and e5-base-v2 (0.5793). But f2llm pays at the ranking top: same-doc hit@1 0.297 (weakest of the 14) — its strong AUC comes from whole-ranking separation, not from top placement.
2. **Same-document retrieval leaders: e5-large-v2** (hit@10 0.826, recall@10 0.594 — best of the pool on both) and **jina-v5** (0.813 / 0.589 with the best hit@1 0.465). embeddinggemma (v1, Q4) keeps the Phase 2 claim of a strong top-1 (0.458).
3. **Balance view: jina-v5 is the only model in the top 4 on all-scope AUC *and* a top-3 same-document retriever**; mpnet (0.452) and nomic-v2-moe (0.449) underdeliver on this corpus relative to reputation (nomic's multilingual-retrieval training; both are prefix-reliant recipes).
4. **Truncation stays immaterial on OMCT**: every model ≤1.8% (MiniLM's 256-ctx 1.8% is the unchanged Phase 2 value; nomic/mpnet 0.4%).
5. **Scope discipline**: modest absolute AUCs (0.45–0.60), as Phase 2 already found — these rank models relatively on a different domain; the pool cut uses UPR val classification evidence, not this table alone.

**Wiring notes (Phase 4 opening change set):** `peft` added as jina-v5's own loading mechanism (its checkpoint ships per-task adapters; `set_adapter("retrieval")` during encode; no generative model involved). jina-v5's custom ST module needs `model_kwargs={"default_task": "retrieval"}` at load (else every encode raises) and does not report its output dimension — pinned 1024 via `embed.dimension` (`dimension_source: config_pin` in meta, contract-enforced at encode). embeddinggemma-2: `/api/show` reports the pre-projection `embedding_length` 512 while `/api/embed` serves 768-d (projection inside the GGUF) — config pins `dimension: 768` (`dimension_source: config_override`); `/api/show` also reports the 256K modality-loader context, overridden to the card's 8192 text window (`configured_context_tokens: 8192`). nomic-v2-moe: `search_query: `/`search_document: ` prefixes REQUIRED per card, wired per kind. New `EmbedderConfig` fields carry both knobs: `model_kwargs` (HF constructor passthrough) and `context_tokens` (explicit context override; discovered values still recorded in meta.extra). Per-model exactly-once prefix checks (config-prefix encode vs manually-prefixed prefix-free encode ≈ identical) ran as 12 integration tests, all green before the sweep.

## Phase 4 opening: UPR kind sweep — 14 embedders × document,query (2026-10-20)

UPR dataset scope: the full dataset, 3000 records (no split, `raw` preprocessing), both encode kinds per model — `document` (the indexed side: labeled example texts) and `query` (the side a text being classified embeds on). No UPR metric exists in this section: these runs are the vector-factory layer Phase 4's retrieval predictors (kNN votes, prototypes, fusion) consume. Runs: `upr-embed-<model>` (the two Phase 2 run ids refresh in place), table section `## UPR — embed_pool`; the frozen per-run configs now record `embed_kinds` (EmbedCell.kinds, the Phase 4 opening checkbox B).

Cache behavior (verified against `kind_stats` + the .npz matrices): `upr-embed-e5-base-v2` / `upr-embed-arctic2` = document side warm (3000 hits, 0 stored — the Phase 2.5-era vectors) + query side fresh (3000 stored); the four **prefix-free** models (`minilm`, `bge-m3`, `mpnet`, `granite-english-r2`) share **one cache key across kinds** (both kinds' prefix is ""), so the query encode legally re-read the document-kind entries — meta shows query side 3000 hits / 0 stored and `vectors_document == vectors_query` elementwise, which is **correct**: a prefix-free model produces exactly one kind of vector (recorded as cache accounting, not a data issue). All prefix/template-carrying models encoded both kinds fresh (0 hits / 6000 stored).

Dimensions as served and verified: minilm 384, e5-base-v2 768, bge-m3 1024, arctic2 1024, embeddinggemma 768, mxbai 1024, e5-large-v2 **1024** (the settled-pool note's "768d" slip, corrected in PLAN), mpnet 768, nomic-v2-moe 768, jina-v5 1024, voyage-4-nano **2048** (default MRL head), granite-english-r2 768, f2llm-80m **320** (MRL), embeddinggemma-2 768 — all matrices (3000, d) float32, unit-norm; wall time 2.6–21.6 s per two-kind run (HF, GPU) and 19.6–41.6 s (Ollama HTTP).

**Next:** Phase 4 retrieval predictors (kNN voting, prototypes) over these artifacts on UPR val — pending the reviewer's OK on the eval↔embedding-artifact wiring (the `embedding_run` field, work-order item 5); the pool cut (checkbox C) follows that evidence.