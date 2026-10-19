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