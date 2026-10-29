# PLAN.md: Retrieval-Augmented Multi-Label Topic Classification with Feedback

## 1. Goal

Build a system that suggests topic labels for short texts (e.g. UPR-style human-rights recommendations). It should:

- work with **thousands of labels** (likely 2000+, long-tailed, and not all known yet),
- treat labeling as **retrieval**: find similar labeled examples and candidate labels, then score or rerank them,
- use **clustering** to explore the data and to narrow the search coarse-to-fine,
- **improve from user feedback** (accept / reject / add label) without full retraining,
- use **active learning** to choose which items a human should review next.

### Non-goals (for now)
- Production deployment, auth, multi-user scaling.
- Training large models from scratch.

### Satellite track: text processing (`TEXT_PROCESSING_PLAN.md`)

The text-processing layer — how text is chunked, segmented and structured before it is embedded — is the thinnest, least-explored layer of this pipeline. It is studied in a **satellite plan**: `TEXT_PROCESSING_PLAN.md` (revised 2026-10-23), OMCT-first, then a port of what survives to UPR.

The track's plan is split in two (2026-10-29): the **live** plan holds the open work, and `docs/text_processing_plan_archive.md` holds the executed pre-registrations and the questions they answered — moved verbatim, append-only. A tombstone in the live plan points at the archive for every moved block.

- **PLAN.md stays master.** A track result adopted downstream gets a note in the relevant phase here (e.g. the Phase 6 reranker slot, the Phase 9 fine-tuning pairs).
- The track's runs are logged in `docs/experiments_text_processing.md` and compared in `docs/experiments_table_text_processing_omct.md` / `..._upr.md` — the command → (dataset, track) mapping decides the file, so track rows never mix with the tables below.
- **OMCT findings annotate method choice; they are never a UPR metric.** The track's own leakage rules (no corpus-level question statistics, no document-identity feature, no pooling the two negative subsets) are stricter than the main plan's and bind every track artifact that is reused downstream.
- The track's named deliverable is a **characterized pair-relevance benchmark on OMCT** (verified hard negatives + unjudged-proxy control + per-document strata), consumable by Phase 6 and Phase 9.

## 2. Data

Input format is JSONL, one record per line:

    {"id": "...", "text": "...", "language": "en", "labels": ["Discrimination", "Sexual Orientation and Gender Identity"]}

Assumptions to verify in Phase 1:
- Multi-label, so a record can have 1..n labels.
- The label set in the dataset is **incomplete** compared with the real taxonomy.
- There may be several languages (the `language` field).
- Texts are formulaic ("Continue to...", "Ensure that..."), which may bias embeddings.

Optional extra resource: a label catalog file (`labels.jsonl`) with `name`, an optional `description`, and an optional `parent`. Even labels with zero examples can be retrieved through it.

### Auxiliary pairwise relevance data (Phase 1.5)

`data/external/omct_pairs.json` (`omct_gti_paragraph_question_golden_pairs`, provided by the collaborator): human-labeled paragraph ↔ GTI question pairs, each judged "same topic?" (label 0/1).

- Shape: `{"pairs": [{"pair_id", "source_csv", "document", "paragraph_number", "paragraph_text", "question_number", "question_text", "label", "label_origin"}], "n_pairs", "n_positive", "n_negative", "name", "description", "question_text_sources", ...}`.
- Formalized by the Phase 1.5 stats run (`aux-base`): 1520 pairs, 524 positive / 996 negative, 294 distinct question numbers/texts across 3 documents, 0 missing question texts, 0 duplicate ids. Paragraphs are long (mean ~890 chars; UPR texts mean ~151). Text repeats across documents (verified Phase 2.5): **0/225** paragraph texts, **186/294** question texts.
- Role, strictly auxiliary:
  1. embedding-model selection probe (similarity AUC + question retrieval, Phase 2),
  2. threshold-calibration evidence (what similarity scores "same topic" actually look like, Phase 4),
  3. training pairs for the optional cross-encoder reranker (Phase 6) and/or embedder fine-tuning (Phase 9),
  4. an independent text ↔ description retrieval benchmark (Phase 3).
- Guardrails: never merged into UPR train artifacts, never reported as a UPR metric; auxiliary runs get the `omct-` run_id prefix (Phase 3.5 grammar; pre-2026-10-20 rows used `aux-`, see `docs/experiments.md`); domain differs (AI/OMCT reporting vs UPR recommendations) so results are always a separate line.

## 3. Target architecture

    Text -> Preprocess -> Embed -> [Cluster routing: top-n clusters]
         -> Retrieve: (a) top-k similar labeled examples (Example Index)
                      (b) top-m candidate labels       (Label Index)
         -> Score fusion (kNN votes + prototype sim + zero-shot sim - negative memory)
         -> [Optional rerank: cross-encoder (LLM variant skipped by decision)]
         -> Suggestions (labels + scores + evidence)
         -> User feedback -> Feedback Store -> online index/prototype updates
                                            -> periodic fine-tuning

Core components:
| Component | Responsibility |
|---|---|
| Embedding backends | One interface for Hugging Face (sentence-transformers) and Ollama (`/api/embed`). Results are cached on disk. |
| Example Index | Vectors of labeled texts plus their label sets. Starts as numpy and moves to FAISS/hnswlib if needed. |
| Label Index | One vector per label: a blend of name/description embedding and the mean of its examples (prototype). |
| Cluster tree | Hierarchical clusters over examples and/or labels, used for routing and discovery. |
| Scorer | Fuses signals into per-label scores and applies thresholds. |
| Reranker (optional) | Cross-encoder over (text, label description) pairs; LLM variant skipped by decision. |
| Feedback Store | SQLite log of feedback events. It is the source of truth for online updates. |
| Active learner | Picks items for human review. |
| Evaluation harness | Offline metrics plus a simulated-user feedback loop. |

## 4. Phases

Each phase lists **tasks**, **deliverables** and **exit criteria**. Do them in order, because later phases depend on the evaluation harness from Phase 3.

### Phase 0: Project setup
- [x] Create the repo skeleton (see AGENTS.md "Repository layout").
- [x] Set up `pyproject.toml`, ruff, pytest and a config system (YAML + pydantic or dataclasses).
- [x] Add `.gitignore` entries for `data/`, `cache/`, `results/` artifacts and model files.
- [x] Write a smoke test that loads the sample JSONL.

**Exit:** `pytest` passes and the CLI works (`uv run cluster-rag --help`, equivalently `python -m cluster_rag.cli --help`).

### Phase 1: Data exploration and splits
- [x] Loader with validation (pydantic `Record` model) that reports and drops malformed rows. *(built in Phase 0, exercised on the full dataset in Phase 1: 3000 valid, 0 dropped)*
- [x] Stats: number of records, number of unique labels, labels per record, label frequency distribution (head / torso / tail buckets), label co-occurrence, languages, text lengths. *(run `base` -> `results/base/stats.json`)*
- [x] Prefix analysis: most common leading 1-3 words ("Continue to", "Ensure that", ...).
- [x] Multi-label **iterative stratified** split into train / val / test (e.g. `iterative-stratification` or `scikit-multilearn`). Fix and save the split ids.
- [x] Create a "label-holdout" split: some labels removed entirely from train, to test zero-shot and new-label discovery.

**Deliverables:** `notebooks/01_eda.ipynb` or a script report, plus `data/splits/*.json`.
**Exit:** The splits are reproducible and the label statistics are written to `docs/experiments.md`.

### Phase 1.5: Auxiliary pairwise data integration
- [x] Loader: `PairRecord` pydantic model with strict validation (drop-and-report malformed entries, same pattern as `Record`) reading `data/external/omct_pairs.json`, driven by a `paths.data_external` config entry. *(module `src/cluster_rag/pairs.py`)*
- [x] Pairwise stats (CLI, e.g. `uv run cluster-rag stats --pairs`): n pairs, positive/negative balance, distinct questions, documents, duplicate ids, text lengths; written to `results/<run_id>/` and summarized in `docs/experiments.md`. *(run `aux-base` → `results/aux-base/pairs_stats.json`)*
- [x] Decide and implement: `data/external/` committed to the repo or gitignored? *(decided 2026-10-16: keep it gitignored/external, see section 7; only the Phase 1.5 stats run and its sha256 provenance reference it)*

**Deliverables:** loader + stats report for the pairwise dataset.
**Exit:** the pairs load and are summarized; nothing else depends on them yet.

### Phase 2: Embedding layer
- [x] `Embedder` protocol: `encode(texts, kind: "query"|"document"|"label") -> np.ndarray` (L2-normalized float32). *(module `src/cluster_rag/embed/base.py`; contract enforced by `ensure_embedding_contract` at every backend + cache boundary; prefix templates with `{content}` supported for prompt-wrapping models)*
- [x] `HFEmbedder` (sentence-transformers) and `OllamaEmbedder` (HTTP `POST /api/embed`, batched). *(`embed/hf.py` records the pinned/loaded revision, `embed/ollama.py` records the digest(es)+context from `/api/tags`+`/api/show`; connection errors say "is `ollama serve` running?", unknown model says `ollama pull <model>`. Full-dataset runs: `results/embed-e5-base-v2/` 3000 texts 4.9s GPU, rerun 3000 cache hits / 0 encoded; `results/embed-arctic2/` 3000 via /api/embed 16.3s. Integration tests marked and run locally.)*
- [x] Model-specific prefix handling (e.g. e5 `query:`/`passage:`, nomic `classification:`/`clustering:`, instruction-style prompts for bge/gte/qwen-embedding). Keep the prefixes in config, never hard-coded in the logic. *(all prefixes/templates live in `configs/experiments/*.yaml`; verified against model cards: e5, arctic-embed v2 (`query: ` queries only), embeddinggemma (`task: search result | query: {content}` / `title: none | text: {content}`), mxbai (`Represent this sentence for searching relevant passages: ` query-side); a HF integration test asserts dropping the e5 prefix changes vectors)*
- [x] Disk cache keyed by `(model_id, prefix, preprocessing_version, sha1(text))`. *(module `src/cluster_rag/embed/cache.py`, `CachedEmbedder` decorator over any `Embedder`; one atomic `.npy` per vector under `cache/v1/<model>/<prep-version>/<prefix>/`; hits make zero model calls — proven by unit test and checked end-to-end by the `embed` command metadata)*
- [x] Preprocessing variants: `raw`, `strip_leading_formulaic` (remove formulaic openers; implemented under exactly this name), `lowercase`. *(module `src/cluster_rag/preprocess.py`; opener list is config (`preprocess.strip_openers`) and hashed into the preprocessing version that forms the cache key — on the real UPR texts the default list modifies 57.97% of records, mean 10.7 chars)*
- [x] Candidate models to compare (pick 3-5): `intfloat/multilingual-e5-base|large`, `BAAI/bge-m3`, `sentence-transformers/all-MiniLM-L6-v2` (fast baseline), Ollama `embeddinggemma`, `mxbai-embed-large`, `bge-m3`, `snowflake-arctic-embed`. *(implemented as 6, one over budget, all already local on disk — zero downloads — because PLAN's list predates the environment check: all-MiniLM-L6-v2 and intfloat/e5-base-v2 (en-only data → e5-base-v2 preferred over the multilingual variant; e5-large deferred until Phase 9 justifies the cost), BAAI/bge-m3 via HF, snowflake-arctic-embed2 / embeddinggemma / mxbai-embed-large via Ollama; configs in `configs/experiments/probe_*.yaml`, revisions/digests recorded per run)*
- [x] Auxiliary probe on the pairwise dataset: per candidate embedder, cosine-similarity AUC separating label 1 vs 0, and R@1 / R@10 question retrieval. Independent evidence for model choice, next to the UPR metrics. *(CLI `probe-aux` → `results/aux-probe-<model>/probe_report.json`; per run both question transforms (keep vs strip_block) and truncation diagnostics vs the model context; metrics are pure functions in `eval/metrics.py` (hand-tested); findings in docs/experiments.md: strip_block wins on AUC 6/6; truncation negligible; modest absolute AUCs 0.42-0.58 used as relative evidence; top all-scope AUC e5-base-v2 0.579, best same-doc hit@10 arctic2 0.813-0.826)*

**Exit:** The full dataset embeds with at least 2 backends and cache hits skip re-encoding. *(met 2026-10-16: `results/embed-e5-base-v2/` (HF, GPU) and `results/embed-arctic2/` (Ollama /api/embed); both 3000 texts, rerun of the e5 job = 3000 cache hits / 0 encoded; cache-key test proves hits make zero model calls.)*

### Phase 2.5: Phase 2 touch-ups (review follow-ups)
- [x] Cache path readability: replace URL-quoted slugs with readable ones — keep `[A-Za-z0-9._-]`, every other character becomes `_`, empty collapses to `-`; the model directory gains a short `sha1(model_id)` prefix so ids that slugify identically stay distinct. Cache wiped and rebuilt in the Phase 2.5 runs. *(module `embed/cache.py`: `BAAI/bge-m3` → `<sha8>-BAAI_bge-m3`, prefix `query: ` → `query__`; key semantics `(model_id, prefix, preprocessing_version, sha1(text))` unchanged; layout + slugging-collapse tests in `test_embed_cache.py`)*
- [x] Question transform default: `strip_block` becomes the probe's default; `keep` only as an explicit comparison. *(schema default `probe.question_transforms: [strip_block]` + config-time validator for unknown transforms; the 6 `probe_*.yaml` configs rely on the default; `keep` returns by listing `[strip_block, keep]` — covered by tests incl. the CLI end-to-end in comparison mode. Phase 2 keep-vs-strip evidence stays in the original `aux-probe-*` artifacts and the log)*
- [x] Probe breakdown per (document, source_csv): n pairs, positives, pair-AUC, same-doc hit@1 / hit@10 reusing the pure metrics with group row masks; counts shown next to every number. *(module `aux_probe.group_breakdown`, wired into `run_probe` per transform variant; `subset_pair_metrics` gained the candidate-column restriction used by the same-doc scope; single-class groups report `pair_auc: null` + note; counts and n_rows_with_gold in the report and the CLI printout — one document has only 99 pairs)*
- [x] Comparison table `docs/experiments_table.md`: human-readable markdown tables, one per (phase, command), auto-appended by the CLI commands; delete `results/summaries/phase2_aux_probe.json` once the tables carry its content. *(`embed`/`probe-aux` (and therefore the sweep) append rows upserted by run key — run_id (+ transform/document per row kind); path is config `paths.experiment_table` (unset → skip); module `src/cluster_rag/experiment_table.py`; three Phase 2 tables: embed, probe-aux, probe-aux groups. The summary-JSON delete happens with the Phase 2.5 runs)*
- [x] Experiment sweep as Python: typed `SweepCell`s (label, command, config overrides) built with plain loops; `define_sweep(cells, base=...)` resolves every cell over the base config into a validated `AppConfig`; a new `cluster-rag sweep --file ...` command executes the resolved cells through the existing command functions; one run folder per cell with the frozen resolved config, exactly as single runs. *(`src/cluster_rag/sweep.py` — validation of command/label/uniqueness/forbidden `run_id`/`extends` overrides; `experiments/aux_probe_sweep.py` holds the 6-model probe grid (`labels probe2-*` → `results/aux-probe2-*`); first failing cell aborts; table rows come from the commands, sweep code only produces configs; the sweep resolves over `configs/base.yaml` only — `configs/experiments/probe_*.yaml` are not sweep inputs)*
- [x] Log correction (Phase 2 finding 5): paragraph texts do **not** repeat across documents — 0/225 in more than one document (they repeat only within documents, against different questions); 186/294 question texts are judged in >1 document (handled by one column per text + per-document doc sets + per-paragraph golds; the all-scope never counts them wrong). *(verified on the data; `summarize_pairs` now carries both counts, recorded in the `aux-base` stats rerun and the Phase 2.5 log section)*

**Exit:** Phase 2 artifacts regenerate under the final layout/configs with identical strip-block numbers; the sweep infrastructure exists for Phase 3+ grids. *(met 2026-10-19: cache rebuilt under the slug layout — warm passes 3000 hits / 0 encoded per model; six `aux-probe2-*` sweep runs reproduce the Phase 2 strip_block numbers digit-for-digit; tables in `docs/experiments_table.md`)*

### Phase 3: Evaluation harness (build before modeling)
- [x] Metrics: P@k, R@k (k = 1, 3, 5, 10), nDCG@k, micro-F1, macro-F1, sample-F1, label coverage. Report each per frequency bucket (head / torso / tail / unseen). *(pure functions in `eval/multilabel.py` + bucket map `eval/buckets.py` + assembler `eval/report.py`, 28 hand-computed unit tests; settled conventions: stable-descending ranking ties → earlier column; restricted-view buckets = columns masked, rows contribute iff ≥1 gold in view, k clipped to view size, overall = same code path unmasked; nDCG@k binary gains/log2 discount/IDCG from min(n_gold, k) ideal ones on the raw score ranking; one config-driven decision rule (top_k | score_threshold | top_k_min_score) feeds F1 + coverage, F1 = 0 on zero division; macro over evaluable (≥1 gold) labels, goldless ones counted; bucket basis = full-dataset counts with unseen = absent-from-train precedence)*
- [x] **Shortlist recall@m** (m = 20, 50, 100): is the gold label anywhere in the retrieved candidates? This is the ceiling for any reranker. *(merged into the metrics change set; `shortlist_recall` in `eval/multilabel.py` keeps the full-view global ranking — `cell_mask` restricts only which gold cells count, so per-bucket numbers ask whether the bucket's golds sit anywhere in the record's global top-m)*
- [x] Trivial baselines: most-frequent labels and random. *(predictor port `predictors/base.py` (Protocol: `score(texts) -> (n, n_labels)` matrix + `describe()` lineage; context carries the label space + train-only records) + `predictors/trivial.py` (most-frequent = train-count scores, never covers unseen; random = seeded uniform draws) + registry `build_predictor`; `eval:` config schema (predictor/decision/metrics/split) in `config.py`; unit tests incl. train-only-counts leakage check; first UPR runs: `experiments/baseline_eval_sweep.py` -> `ev-{mf,random}-topk{1,2,3}-val` + `ev-{mf,random}-topk2-test`, narrative + numbers in `docs/experiments.md` Phase 3 section)*
- [x] Pairwise-retrieval metrics (pure functions in `eval/metrics.py`): per paragraph, rank the distinct questions by embedding similarity and report recall@k against the manual labels; exercised by the Phase 2/3 auxiliary probe. *(built in Phase 2 as the probe's dependency — `roc_auc`, `pairwise_retrieval_metrics`, `subset_pair_metrics` (incl. per-row candidate columns), all hand-tested; the Phase 2.5 group breakdown adds group row masks)*
- [x] Results written as JSON to `results/<run_id>/metrics.json` together with the resolved config; plus a table row appended to `docs/experiments_table.md` (auto) and a narrative row in `docs/experiments.md`. *(`evaluate` CLI command = the exit criterion — one command evaluates any predictor on val or test: `cluster-rag evaluate --config ... --split val|test`; split ids loaded from `data/splits/` with a source sha256 consistency check; `load_split_ids`/`select_records_by_ids` in `data.py`; artifacts under "Phase 3 — evaluate" in the table, key run_id; `sweep` executes evaluate cells; end-to-end CLI tests incl. seeded rerun determinism)*

**Exit:** One command evaluates any predictor on val or test.

### Phase 3.5: naming + experiment UX (Python sweeps, per-dataset tables)

All decisions settled with the reviewer 2026-10-20 (do not re-litigate): sweeps are authored in **Python only** (no YAML authoring for experiments); one **sweep = one dataset = its own table section**; tables are per-dataset files — `experiments_table_upr.md` and `experiments_table_omct.md`, nothing else; every artifact is recreatable, so old-named results and the global table file may be deleted after a verified re-run migration. Work order, one checkbox per change set:

- [x] **A. Run-id conventions in code + docs.** Grammar `<dataset>-<command>-<detail>[-<split>]`, datasets `upr-` (UPR data) and `omct-` (auxiliary OMCT pairs). `stats --pairs` and `probe-aux` emit `omct-<run_id>` (was `aux-`); UPR commands (`stats`/`split`/`embed`/`evaluate`) only **warn** when the run_id lacks `upr-` (warning, never a hard error). Update `AGENTS.md`: new "Runs and where things live" section (grammar, both prefixes, `results/<run_id>/` contents, narrative vs table file roles, legacy-id note for pre-2026-10-20 runs) and rework the experiment-log paragraphs that still name `aux-` and the global table file. Sweep/docstring/help strings updated. *(done 2026-10-20: `cli.py` emits `omct-` in `stats --pairs`/`probe-aux` with a double-prefix warning, `_warn_upr_prefix` on the four UPR commands; AGENTS.md gained the "Runs and where things live" section + reworked layout/log paragraphs; `sweep.py`/`aux_probe.py`/`pairs.py` docstrings and CLI help updated; new warning tests in `test_cli.py`/`test_pairs.py` — 181 unit tests green)*
- [x] **B. Python-only typed sweep layer.** New typed cells beside the generic `SweepCell`: `EvalCell(label, predictor, params={}, decision=None, k=None, split="val", metrics=None)` (`k` = shorthand for a `top_k` decision; `decision` XOR `k`), `EmbedCell(...)`, `ProbeCell(...)`. Labels mandatory and filesystem-safe (explicit, no auto-names). `define_sweep(cells, *, dataset, label)`: `dataset` in {upr, omct}, validated against the commands allowed for that dataset (upr: stats/split/embed/evaluate; omct: probe-aux and stats --pairs); upr labels must start with `upr-`, omct labels must NOT carry a prefix (the command adds `omct-` at run time); `define_sweep` injects `table.sweep_label` into every resolved config so each run knows its table section. Rewrite `experiments/baseline_eval_sweep.py` in the typed style (the reference example); zero YAML/dict wiring in sweep files. Tests: cell validation, the decision/k XOR rule, label uniqueness, dataset/prefix checks, resolution over base. *(done 2026-10-20: `sweep.py` rewritten — `EvalCell`/`EmbedCell`/`ProbeCell` + generic `SweepCell`, commands `stats`/`stats-pairs`/`split`/`embed`/`evaluate`/`probe-aux`; settled with the reviewer: `stats --pairs` gets its own `stats-pairs` command name, the sweep `label` is free-form (heading only, filesystem-safe), and `define_sweep` resolves over the literal `configs/base.yaml` (no base parameter; tests write a fixture base under their tmp CWD); new `table.sweep_label` config section (`config.py`); `baseline_eval_sweep.py` (label `baselines`) + `aux_probe_sweep.py` → `omct_probe_sweep.py` (label `probe2`) rewritten typed; CLI dispatch + help updated; 13 sweep tests — suite 186 green)*
- [x] **C. Per-dataset comparison tables + visible defaults.** Exactly two table files: `docs/experiments_table_upr.md` and `docs/experiments_table_omct.md` (config `paths.experiment_table_upr` / `paths.experiment_table_omct`); the global `docs/experiments_table.md` is deleted. The command's dataset decides the file (command-derived, not name-derived: UPR commands → upr file, OMCT commands → omct file). The sweep decides the section heading (`## UPR — <sweep-label>` / `## OMCT — <sweep-label>[- groups]`); standalone runs land under `## <UPR|OMCT> — <command>` — there is no "singles" concept anywhere. Upsert by run_id unchanged. The evaluate columns are regrouped and renamed: run_id, date, split, predictor, **f1 decision**, micro-F1, macro-F1, sample-F1, coverage | P@5, R@10, nDCG@5, shortlist R@50, n_texts, seconds (decision view first, ranking view second). `configs/base.yaml`: commented `eval:` defaults block (decision `top_k, k=2`, ks/shortlist_ms grids, split) — the predictor name stays per-run by design; slim `configs/experiments/*.yaml` run ids to the new grammar and `eval_baseline_most_frequent.yaml` to run_id + predictor. *(done 2026-10-20: config keys `paths.experiment_table_upr/omct` (unset → skip), `experiment_table.section_heading`/`intro_for` (dataset tags UPR/OMCT; groups suffix " - groups"), CLI append sites dataset-mapped; evaluate table regrouped with the "f1 decision" column; base.yaml carries both table keys + the commented eval defaults; embed configs get `upr-embed-*` ids, eval config slimmed to predictor + run_id; global `docs/experiments_table.md` deleted — its rows regenerate per dataset in D)*
- [x] **D. Migration re-runs + verification + deletion.** All re-runs are warm-cache/CPU (nothing re-encoded, no re-splitting; `ollama serve` must be up for the Arctic digest metadata; the HF backend may load its model briefly for construction — fine): `upr-stats-base` (was `base`), `upr-embed-e5-base-v2` + `upr-embed-arctic2` (expect 3000 cache hits / 0 encoded each), `omct-stats-base` (was `aux-base`), the probe2 sweep → `omct-probe2-*` (labels unchanged — the command adds `omct-`), the baseline eval sweep → `upr-eval-{mf,random}-topk{1,2,3}-val` + `upr-eval-{mf,random}-topk2-test` (was `ev-*`). Verify each new metrics.json equals the old one field-for-field (run_id/total_seconds exempt) and record the verification; then delete ALL old-named folders (`base`, `embed-*`, `aux-*` including the 6 legacy `aux-probe-*` Phase 2 originals, `ev-*`, and the empty `results/summaries/`). Append ONE migration note to `docs/experiments.md`: the old→new mapping, "numbers unchanged, verified", and the legacy-id explanation — existing log rows stay exactly as written (append-only). *(done 2026-10-20: 4 migration sweeps — `upr_migration_sweep.py` (label `migration`: stats + 2 embeds, 3000 hits / 0 encoded each), `omct_stats_sweep.py`, `omct_probe_sweep.py`, `baseline_eval_sweep.py` — 18 new runs; script diff verified all artifact pairs field-for-field identical (exempt: run_id, `*seconds*`, probe `cache` hit/miss counters; embeds' `embeddings.npz` also elementwise-equal); old folders incl. `results/summaries/` deleted — `results/` now holds only the 18 `upr-*`/`omct-*` run folders; mapping + verification + legacy-id note appended as ONE migration note, old log rows untouched)*
- [x] **E. Close-out.** `ruff check` + `ruff format --check` over src/tests/experiments, full `pytest` (at least the current 179 unit tests green, plus the new ones), tick the Phase 3.5 checkboxes, short summary with the Phase 4 next step. *(done 2026-10-20: ruff check + format clean over src/tests/experiments, 189 unit tests + 4 integration tests green (suite grew 179→189); stale `!results/summaries/` gitignore negation removed with the folder; Phase 4 next step = embed the remaining kinds for the two candidate UPR models (`--kinds document,query`: query-side for the text being classified, document-side for labeled examples), then the kNN-voting predictor + ExampleIndex/LabelIndex checked against the harness and the `upr-eval-*` baselines*)

**Exit:** every experiment is defined in a Python sweep file; every comparison row lives in its dataset's file under its sweep's section; no artifact or doc row needs the old naming to be found. *(met 2026-10-20: `experiments/` holds 4 typed Python sweep files; tables are per-dataset, sections chosen by `table.sweep_label`; legacy ids survive only in old log rows + the migration mapping note)*

### Phase 3.6: omct command naming (review follow-up)

Follow-up to Phase 3.5, settled with the reviewer 2026-10-20: dataset-scoped commands carry the dataset name — no `aux`/`pairs` ambiguity in any command, flag or sweep name. The `<dataset>-<command>-<detail>[-<split>]` grammar and the `omct-` prefixing are unchanged.

- [x] **Rename to `omct` + re-run + verify.** `probe-aux` → `probe-omct`; `stats --pairs` → `stats --omct` (sweep command `stats-pairs` → `stats-omct`); module `aux_probe.py` → `omct_probe.py` (+ `tests/test_aux_probe.py` → `test_omct_probe.py`, table constants `PROBE_OMCT_*`, "omct question transform" wordings in `preprocess.py`/`eval/metrics.py`/`embed/hf.py`); omct stats artifact `pairs_stats.json` → `stats.json` (uniform with the one-artifact-per-command convention). Deliberately kept: `pairs.py` / `PairRecord` / `load_pairs` / `summarize_pairs` and pair-count report fields — "pair" names the record type (paragraph↔question pair), not the dataset; `paths.data_external` unchanged; "auxiliary" survives only as the English adjective (the OMCT data *is* auxiliary — PLAN §2). Warm re-run of the two omct-run-producing sweeps (7 runs; frozen configs + payload `command` fields refreshed to `probe-omct`; run ids and table headings unchanged — labels own them), field-for-field diff vs the pre-3.6 artifacts (exempt: `run_id`, `*seconds*`, probe cache counters, the renamed `command` field), verification noted in `docs/experiments.md` (append-only).

**Exit:** every experiment is defined in a Python sweep file; every comparison row lives in its dataset's file under its sweep's section; no artifact or doc row needs the old naming to be found.

### Phase 4 opening: embedding-pool widening (settled with the reviewer 2026-10-20)

**Executed 2026-10-20** — the pool widened from the recorded 6 to **14 single-vector text
embedders** (bi-encoders only), the typed embedding kinds landed, the val-driven cut chose
the semifinalists **{embeddinggemma, jina-v5, voyage-4-nano, arctic2}**, the eval ↔
embedding-artifact wiring landed, and the stale experiment YAMLs were deleted. The full
record — the reviewer's decisions, the 14 models with their card-verified prefixes and
revisions, and all five change sets — is in `docs/text_processing_plan_archive.md` §PLAN,
moved verbatim on 2026-10-29. The pool itself is live in `experiments/embedder_pool.py`.

### Phase 4: Retrieval baselines (no clustering yet)
- [x] **kNN voting:** top-k neighbors in the Example Index, with labels scored by similarity-weighted votes (optionally sim^p or softmax with temperature). *(needs per-model UPR embeddings of both kinds — `--kinds document,query`: query-side for the text being classified, document-side for labeled examples; asymmetric models' document matrices are no query substitute; done 2026-10-20: `predictors/knn.py` (port) over the pure core `retrieve/knn.py` — votes = Σ max(cos,0)^p over the top-k train examples, p configurable (softmax-with-temperature deliberately deferred until a grid needs it), ties → earlier train row, k clamped to n_train; consumes the Phase 4 wiring (train-only document vectors at construction); fake-embedder unit tests incl. hand-computed votes, tie and clipping cases; val evidence arrives with Phase 4 opening C)*
- [x] **Label prototypes:** mean of the example embeddings per label. *(done 2026-10-20: `predictors/prototype.py` over `retrieve/prototypes.py` — renormalized per-label mean of train document vectors (float64 accumulation → float32 output); zero-example labels carry zero prototypes and score 0, matching the trivial baselines' unseen-label behavior; max-sim aggregation over the same matrix joins with the checkbox-C sweep)*
- [x] **Per-label max-similarity:** label score = max cosine between the text
  and any single train example of that label (over the same sentence matrix
  the prototype mean uses — complements the mean for tail labels with few
  examples); same predictor interface, runs in the Phase 4 opening checkbox C
  val sweep (settle 2026-10-20 with the reviewer: the candidate family for
  the pool cut is {knn, prototype, max-sim}). *(done 2026-10-20: `predictors/maxsim.py` (registry name `maxsim`) over `retrieve/prototypes.py: max_sim_scores` — per-label max cosine on the same train document matrix the mean averages; labels with zero train examples score the cosine floor -1.0 so they can never enter a top-k decision (a 0-floor would ride above genuinely negative similarities); fake-embedder unit tests incl. hand maxima + contract rejections; the sweep cells themselves run under Phase 4 opening C — reviewer-approved grid {knn k in 1/3/5/10/20, prototype, max-sim} x 14 models in `experiments/pool_cut_sweep.py`)*
- [x] **Zero-shot label embeddings:** embed label names and descriptions, optionally with templates such as "This recommendation is about {label}." *(mechanism approved by the reviewer at the Phase-4 checkpoint and implemented 2026-10-20 [no catalog file exists — PLAN section 7 — so names/templates only; a `descriptions` variant waits for that file]* **command `labelembed`** *(new: `results/<run_id>/label_embeddings.npz` — one config-driven text per dataset label [sorted unique] under the model's card-verified `prefixes.label`, through the same preprocess+cache discipline; `LabelTextsConfig` variants `name`|`template{"label"}`; run-id grammar `upr-labelembed-<model>[-<variant>]`)* **wiring** *(schema change ok'd: `EvalCell.label_embedding_run` → `eval.label_embedding_run_id` [prefix-validated]; `predictor_requirements(name, params)` decides needs [zeroshot: always; fusion: only c>0]; loader hard-validates existence, command=sha provenance, **label space equality with the harness label space [order included]**, and model/dimension equality with the primary embedding run [no cross-model mixing]; `PredictorContext.label_vectors` train-legal for holdout labels per AGENTS.md)* **predictor `zeroshot`** *(registry: query↔label cosines, `REQUIRED_KINDS=("query",)`; unit + CLI e2e tests incl. tampered-label-artifact refusal)* **fusion c-term** *(c>0 activates the third component + the label artifact; c=0 keeps the pool-cut mix exactly [fusion sweep cells unaffected]).* **sweep** *(experiments/zeroshot_sweep.py: 4 semifinalists × {name, "This recommendation is about {label}.", "Topic: {label}."} → 12 `upr-labelembed-*` + 12 `upr-eval-zeroshot-*` cells; real-artifact smoke [throwaway /tmp e5-base-v2 chain] verified the CLI path incl. unseen-bucket coverage >0; 24-cell resolution checked; Ollama/HF runs execute on the reviewer's machine)*
- [x] **Fusion:** `score = a*knn + b*proto + c*zeroshot`. Tune the weights on val. *(tuned + verified 2026-10-20 — two grids, UPR val only, top_k k=2, both sweeps over the four semifinalists: `fusion_weight_sweep.py` [c=0, a∈{0.25..8} b=1 k∈{10,20}: proto-heavy ratios win, best a ≤ 0.5 per model, flat landscape] then `fusion_c_sweep.py` [a=0.5 b=1, c∈{0.1,0.25,0.5,1.0}, k∈{10,20}, zero-shot template `about`: every model's best c-cell beats its c=0 twin, but all gains are near-ties ≤0.46 pp — flat landscape, ~1–6 gold cells of 1312; shortlist R@50 stays ~0.73–0.74, the kNN 0.89–0.96 shortlist role is untouched]. **Champion: `upr-eval-fusionc-embeddinggemma-k10-c1` 0.5479** [c=0.25 exactly micro-tied; c=1 wins the tail-macro tiebreak 0.5181/0.5117 and dominates macro/shortlist] — +0.15 pp over its c=0 twin, +0.91 pp over the pool-cut family best; **proposed Phase-4 config: embeddinggemma, fusion a=0.5 b=1.0 c=1.0, knn k=10, decision top_k k=2**. Important negative: c>0 does NOT lift unseen coverage inside fusion (0.000 for every c ≤ 1 — fused seen-label scores dominate the top-2), so the unseen-label route remains the standalone zeroshot predictor [coverage 0.62–0.75]. Per-model bests: jina-v5 c0.5 k10 0.5426, voyage-4-nano c0.5 k20 0.5395 [3-way micro tie, tail-break], arctic2 c0.25 k10 0.5426. Evidence + final table in `docs/experiments.md` 2026-10-20 c-grid section and `## UPR — fusion_c` [40/40 artifacts verified])*
- [x] **Thresholding:** global, per-label (calibrated on val) and top-k with a minimum score. Compare them. *(global + top-k-min-score compared on UPR val + verified 2026-10-20, 30 runs: `experiments/threshold_sweep.py` [24 cells, champion scale] + `experiments/threshold_refine_sweep.py` [6 cells, runner-up's recalibrated scale] — on the champion's fused scores, rank-2 gold (q25 ≈ 1.64) separates from rank-2 non-gold (q75 ≈ 0.70), and **score_threshold t=1.75 wins micro 0.5988 vs top_k2's 0.5479** [+5.10 pp; top_k_min_score best 0.5852; top_k k∈{1,2,3} all ≤ 0.5479]; every threshold rule trades tail for micro at the micro optimum [st 1.75: tail 0.4489 vs topk2 0.5181] — the Pareto alternative tkms t=1.25 [0.5756/0.5334/0.5616] and the balanced st t=1.5 [0.5917/0.5278/0.5133] are logged for the reviewer; the standing convention [best micro, tail tiebreak] picked st=1.75. Calibration evidence, strictly aux-annotated: OMCT pair score distributions via the new `probe_report.json` `score_distributions` block [4 runs `omct-probe-thresholds-*`] — judged-pair AUC only 0.47–0.54, label-0 maxima 0.66–0.79 above label-1 p95, goldless paragraphs' best same-doc sims up to 0.66–0.79: NO absolute threshold is importable from OMCT; UPR val decides any operating point. **Per-label thresholds deferred** [would add a new decision rule + a fitting mechanism = schema change pending reviewer OK; on 660 val rows × 68 labels it would fit ~68 scalars from ~660 rows — overfit-prone; open question in section 7])*
- [ ] Optional lexical signal: BM25 over texts and label names, for hybrid retrieval. *(NOT done — deferred with a reviewer question: needs the `rank-bm25` dependency + a 4th fusion term [predictor + param] = explain-first per AGENTS §3; the Phase-4 evidence that fusion's shortlist R@50 ~0.74 ≪ kNN's 0.89–0.96 gives BM25 a lexical-breadth role to test, but Phase 5 clustering is next in order)*
- [x] Ablations: embedding model x preprocessing x k x fusion weights. *(executed staged 2026-10-20, per the checkpoint plan "embed all, ablate few": model pool cut from 14 to 4 on val [checkbox C]; k∈{1,3,5,10,20} grid; fusion weight grids [a-grid 48 cells + c-grid 32 cells]; preprocess ablation on the 4 semifinalists only at tuned weights [24 runs `## UPR — preprocess_ablation`: strip/lower vs raw — no clean winner: strip helps jina-v5/arctic2, hurts embeddinggemma; lowercase helps arctic2, hurts 3 of 4; all deltas 1–9 gold cells; the raw champion stays within noise of the best and wins under its recalibrated threshold rule]*

**Exit:** The best config is chosen on val, with test results reported once. Expect kNN + prototype fusion to clearly beat the trivial baselines.

### Phase 5: Clustering (discovery and routing)
- [ ] Discovery: UMAP + HDBSCAN (or BERTopic) on the embeddings. For each cluster, report size, label purity / entropy, top labels and c-TF-IDF keywords.
- [ ] Analyze the mixed clusters (overlapping labels?) and the outliers (missing labels?). On the label-holdout split, check whether held-out labels show up as coherent clusters.
- [ ] Routing: build a 2-level hierarchy (e.g. k-means over examples or over label prototypes). At query time, route to the **top-n clusters** and retrieve only inside them.
- [ ] Measure shortlist recall loss and speedup for routing vs flat retrieval, as a function of n.
- [ ] Cluster-level propagation: in pure, dense clusters, propagate the majority labels to unlabeled members with high confidence, and check the result on val.

**Exit:** A documented decision on whether routing is worth it at the current data size, plus a cluster report artifact.

### Phase 6: Reranking (optional, after Phase 4 is solid)

*Track input, not yet delivered:* the text-processing track's named deliverable is a **characterized pair-relevance benchmark on OMCT** (the reviewer-verified hard negatives, the unjudged-proxy control and the per-document strata) plus a combiner/feature scorer — i.e. exactly the negative-pair source this phase lacks today. See `TEXT_PROCESSING_PLAN.md` §8; nothing is adopted before its artifact lands in `results/`.

*Track input, delivered 2026-10-27 (the text-processing track's CS-15/CS-16,
`TEXT_PROCESSING_PLAN.md` §7.1.3 → `docs/text_processing_plan_archive.md` §7.1.3):* the **zero-shot** half of this phase is
measured, on the UPR val split, with three Hugging Face rerankers (a 278M
cross-encoder, a 1B yes/no head and a 0.6B **listwise** reranker — the last
one is not a multi-vector model, it reads a whole candidate list in one
context). The evidence, in one line each:

- **as a re-ordering of an existing candidate set** (the decision model's
  top-10, the union's top-10, the champion's top-10 as the control): every
  zero-shot reranker is **worse** than the ordering we already have (the best
  arm a paired **+0.0091 [−0.0122, +0.0294]** against the decision model's
  own ordering — a null), while the **label-blind counterpart** is
  catastrophic (−0.27 to −0.38): the signal is label-driven and the
  *ordering* is the weak part. The per-bucket view says the rerankers are a
  **rare-label specialist** (the tail +0.1766 [+0.1141, +0.2419]) and lose
  on the head and the torso.
- **as a pack feature** (the CS-13 precedent, one level up): the
  **listwise** model's score lifts the pack from the micro-F1 0.5965 to
  **0.6117** (a paired **+0.0152 [+0.0015, +0.0280]**, the label-blind
  counterpart at null) — the best UPR number the track has measured — but
  the pre-registered **+0.02 margin is not met**, so this phase's exit
  criterion ("enabled only if it improves val F1 / P@k by a meaningful
  margin") is **not** satisfied and the slot decision is the reviewer's.
- **the cost**: the candidate-restricted scorer form is 4-43 s per 660 val
  texts (against the decision model's ~13 min); the full-grid feature form
  is 266 s for the val grid **and** the fit side (the listwise model, one
  call per text) against the 1B head's 909 s (one pass per pair).
- **the next step is the fine-tuning**, not a zero-shot model: the track
  holds all three ingredients (the OMCT hard negatives, the UPR safe-negative
  hard negatives, the decision model's 99.87%-recall candidate set as the
  positive mining source) — see the track's CS-17.

- [ ] Cross-encoder reranker on (text, label name/description) pairs. Zero-shot first, then fine-tuned on the auxiliary pairwise data (Phase 1.5) plus hard negatives from rejected suggestions.
- ~~LLM reranker via Ollama~~ **Skipped by decision (2026-10): no LLMs in the pipeline.**
- [ ] Compare against fusion alone: quality gain vs latency and cost.

**Exit:** The reranker is enabled only if it improves val F1 / P@k by a meaningful margin.

### Phase 7: Feedback store and online learning
- [ ] SQLite schema `feedback(id, item_id, text, label, action, source, user, ts, model_version)` with `action` ∈ {accept, reject, add, remove}.
- [ ] Online update rules (no retraining):
  - accept/add: the item goes into the Example Index with its confirmed labels.
  - reject: the (text embedding, label) pair goes into the **negative memory**, and similar future texts get a penalty for that label.
  - Rocchio update of label prototypes (`a*proto + b*mean(accepted) - c*mean(rejected)`, then renormalize).
  - Corrected items become few-shot candidates for the optional reranker (LLM variant skipped by decision).
- [ ] Every update can be replayed from the event log (rebuild the indexes from scratch at any time).
- [ ] **Simulated user** for evaluation: an oracle that answers with the gold labels, optionally with configurable noise.
- [ ] Learning curve: metric vs number of feedback events, with and without each update rule.

**Exit:** The simulation shows metrics improving with feedback, and the indexes can be rebuilt from the log.

### Phase 8: Active learning
- [ ] Selection strategies: random (baseline), uncertainty (small margin between top-2 scores, or a score near the threshold), cluster representatives (medoids of large unlabeled clusters), boundary items, outliers, and a diversity-aware batch mix.
- [ ] Exploration: occasionally show lower-ranked labels (epsilon-greedy or Thompson sampling) to avoid feedback loops.
- [ ] Compare the strategies in the simulator with an equal labeling budget.

**Exit:** At least one strategy beats random on the learning curve.

### Phase 9: Embedding / model fine-tuning

*Track input, not yet delivered:* the text-processing track's "which unjudged pairs are safe negatives" rule (a similarity margin + a reviewer-confirmed spot-check) is the missing recipe for mining the 44k-cell unreviewed space; the track's combiner/feature scorer is the candidate feature set for the bi-encoder below. See `TEXT_PROCESSING_PLAN.md` §8.

- [ ] Build training data from the labels plus accumulated feedback: (text, positive label/text, hard negative). Hard negatives come from rejected suggestions and confusable labels. The auxiliary pairwise pairs (Phase 1.5) join as (paragraph, question) positive/negative examples.
- [ ] Leakage rule: fine-tune inputs may include the auxiliary pairwise data and UPR train only — never UPR val/test.
- [ ] Options: SetFit, sentence-transformers `MultipleNegativesRankingLoss`, or a bi-encoder over (text, label description) pairs.
- [ ] Retrain on a schedule (e.g. every N feedback events), re-embed, and compare with the previous model version.

**Exit:** The fine-tuned embedder improves shortlist recall and tail-label macro-F1 on test.

### Phase 10: Interface (optional)
- [ ] FastAPI endpoints: `/suggest`, `/feedback`, `/next_items`.
- [ ] Minimal Streamlit or Gradio UI: show a text, the suggested labels with evidence (nearest examples), and accept/reject/add buttons.

## 5. Experiment tracking
- Every run gets a `run_id`, a frozen config and metrics in `results/`, plus a narrative row/section in `docs/experiments.md` (append-only).
- Run-id grammar: `<dataset>-<command>-<detail>[-<split>]` with dataset prefixes `upr-` (UPR data) and `omct-` (auxiliary OMCT pairwise data). A sweep is authored for exactly one dataset (`define_sweep(cells, dataset=..., label=...)`), and the dataset decides which comparison table file a row lands in. *(reworked by Phase 3.5; the pre-3.5 `aux-` prefix and unprefixed/`ev-` UPR ids are legacy)*
- Comparison tables: **per dataset only** — `docs/experiments_table_upr.md` and `docs/experiments_table_omct.md`. Every sweep owns its own section (`## UPR — <sweep-label>`, injected into its cells by `define_sweep`); standalone runs land under `## <UPR|OMCT> — <command>`. Rows are auto-appended by the CLI commands and upserted by run key (the latest run per key wins). The human-readable leaderboard. *(reworked by Phase 3.5; the pre-3.5 single global `docs/experiments_table.md` is retired on migration)*
- The text-processing track (`TEXT_PROCESSING_PLAN.md`) keeps **its own pair of tables** — `docs/experiments_table_text_processing_omct.md` / `..._upr.md` — and **its own log** (`docs/experiments_text_processing.md`). Which file a command writes to is command-derived: every command maps to a `(dataset, track)` pair, so track rows never mix with the tables above. Its sweeps live in `experiments_text_processing/`; the sweep loader keeps `experiments/` importable so `embedder_pool.MODELS` stays shared.
- Grids: sweep files under `experiments/` are **Python-only** — typed cells (`EvalCell`/`EmbedCell`/`ProbeCell` with explicit labels) resolved by `define_sweep(cells, dataset=..., label=...)` over `configs/base.yaml`; `cluster-rag sweep --file <file>.py` executes the cells through the existing command functions (one run folder per cell with the frozen config), so sweep code only produces configs.
- Auxiliary-dataset runs get the `omct-` run_id prefix, live in their own table file, and are never reported as UPR metrics.
- Fix all random seeds. Record model ids and versions — HF models pinned by revision in the config (the backend also records the loaded snapshot), Ollama by digest.
- MLflow not planned for now: the comparison tables fulfill the leaderboard role (revisit only if run volume demands it).

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Embeddings group texts by formulaic openers rather than topic | Preprocessing variants and instruction prefixes; check in the Phase 5 cluster report |
| Long tail / unseen labels | Zero-shot label embeddings, label descriptions, label-holdout evaluation |
| One global threshold fits badly | Per-label calibration, top-k + minimum score |
| Feedback loop bias | Exploration and the "add label" action; log impressions as well as clicks |
| Noisy feedback | Weight by source, cap the effect of a single event, replayable log |
| Data leakage | Strict split discipline (see AGENTS.md) |

## 7. Open questions
- Is the full label taxonomy available with descriptions and hierarchy? *(Phase 1 finding: no catalog file exists; the 68 labels are known from annotations only — zero-shot label names/descriptions remain a controlled experiment)*
- How many languages are there, and what is the volume per language? *(Phase 1 finding: one language, `en`, 3000 records total — multilingual models are unnecessary for now)*
- Latency budget per suggestion? *(was an input to feasibility of the LLM reranker, which is skipped by decision — only matters again if that decision is revisited; cross-encoder latency still TBD in Phase 6)*
- Expected data size (decides between numpy, FAISS and hnswlib)? *(Phase 1 finding: 3000 records / 68 labels — plain numpy indexes are sufficient; revisit if more data arrives)*
- Auxiliary pairwise data: commit `data/external/omct_pairs.json` to the repo or keep it external? *(decided 2026-10-16: keep it gitignored/external — collaborator-owned file; the sha256 recorded in the `aux-` runs is its provenance)*
- Auxiliary pairwise data: is the "Answer choices" text inside `question_text` signal or noise for embeddings? *(answered by the Phase 2 probe, 2026-10-16: mostly noise — `strip_block` raises pair-AUC for all 6 embedders (e5 +0.087, mxbai +0.068, MiniLM +0.034, bge-m3 +0.019, arctic2 +0.017, embeddinggemma +0.013) and same-doc hit@1 for 4/6; occasional hit@10 losses (arctic2, embeddinggemma) mean the block sometimes carries signal too. Default question-side transform for aux runs: `strip_block`.)*
- Auxiliary pairwise data: annotation provenance — single annotator or merged/verified several times? (affects how much we trust it for threshold calibration) *(Phase 4 sharpening 2026-10-20: judged label-0 pairs reach cosines up to 0.66–0.79 — above every label-1 pair's p95 — and "should answer nothing" paragraphs reach best same-doc sims in the same range; with formulaic furniture this may be genuine near-miss text, but noisy/ambiguous negatives would look identical; worth asking the collaborator)*
- Per-label decision thresholds (calibrated on val): worth adding as a new decision rule + fitting mechanism? *(Phase 4 decision 2026-10-20: deferred — the global operating point on val is strong (+5.10 pp micro over top_k2); a per-label rule would fit ~68 calibration scalars from 660 val rows (overfit-prone), needs `decision` schema + a fitted-thresholds artifact that carries to test; revisit only if Phase 6 (reranker) or Phase 9 (fine-tune) shows label-miscalibration being the bottleneck)*
- Optional BM25 hybrid (Phase 4 lexical checkbox): run it? *(Phase 4 decision 2026-10-20: deferred past the Phase-4 exit — it needs the `rank-bm25` dependency and a fourth fusion term (predictor + params + config), i.e. explain-first per AGENTS.md §3; the logged evidence for its role: fusion's shortlist R@50 ≈ 0.74 vs kNN's 0.89–0.96, so lexical breadth is the candidate mechanism for the Phase-6 reranker shortlist — propose as a Phase 5/6 side experiment rather than reopening Phase 4)*
- Embedding-model pool: stay with the recorded 6 candidates, or widen — more models already in the local HF/Ollama caches (e.g. all-mpnet-base-v2, nomic-embed-text-v1.5) and/or other models worth testing — before Phase 4 grids are locked? *(decided 2026-10-20 at Phase 4 opening: widen to 14 single-vector text embedders — the recorded 6 plus embeddinggemma-2, nomic-embed-text-v2-moe, jina-embeddings-v5-text-small, voyage-4-nano, granite-embedding-english-r2, F2LLM-v2-80M, e5-large-v2, all-mpnet-base-v2; bi-encoders only, no generative/multi-vector models — the OMCT-probe and UPR-val evidence (checkboxes A–C) narrow the pool to 3–4 semifinalists before any grids; nomic-embed-text-v1.5 stays out for now: same family as v2-moe with an older recipe)*

## 8. Milestones
1. **M1:** Phases 0-3, the evaluation harness with baselines. (Phase 1.5 auxiliary data + probe included, the Phase 2.5 review touch-ups, and the Phase 3.5 naming/UX rework.)
2. **M2:** Phase 4, a strong retrieval baseline.
3. **M3:** Phases 5-6, the clustering report and reranker decision.
4. **M4:** Phases 7-8, feedback + active learning shown working in simulation.
5. **M5:** Phases 9-10, fine-tuning and a demo UI.
