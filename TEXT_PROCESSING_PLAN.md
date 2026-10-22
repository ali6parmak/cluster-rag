# TEXT_PROCESSING_PLAN.md — text processing / segmentation track

Satellite to `PLAN.md` (created 2026-10-22; PLAN.md stays master — a track result adopted downstream gets a note in the relevant PLAN phase, e.g. Phase 6 reranker slot).
Style follows PLAN.md: numbered stages, one checkbox per change set, exit criteria, open questions. Reviewer = project owner (solo — no external collaboration assumed anywhere in this track).

# TEXT_PROCESSING_PLAN.md: text processing, segmentation & feature-based scoring

## 0. Why this track exists (context)

- Track motivation (meeting 2026-10-21): the text processing layer — how text
  is retrieved/split/segmented/processed — is the baseline everything else sits
  on, and it is the thinnest, least-explored layer of the current pipeline
  (one preprocessing variant + opener stripping).
- Idea sources: BOTH a reference experiment log exp01-exp19 (OMCT
  paragraph<->question matching: chunking families, aggregation, learned
  feature models, semantic boundaries, cross-encoder, LLM rerank) — LOCAL,
  untracked, not going to GitHub — AND our own invented experiments. Scope
  rule for the track: nothing is out of bounds (within the hygiene rules
  below); expected starting point: the reference lineage, re-derived
  evidence-first on OUR data with OUR harness discipline. We are not bound to
  it — where it can be beaten or replaced, we try.
- First focus: **OMCT** (~890-char paragraphs where segmentation can matter;
  binary relevance labels; a live suspicion that data/labels/aggregation —
  not representation — drives the pipeline's observed anomalies).

## 1. Governance (what carries over / what this track relaxes)

Carried over (unchanged):
- Harness-as-judge: every claim comes from a run artifact; append-only logs;
  one change set per checkbox; explain-first for schema/deps (AGENTS §3);
  seeds pinned; `uv run`; ruff+pytest gates; OMCT results NEVER reported as
  UPR metrics.
- **Data discipline: raw data is read-only and never fixed/edited** —
  `omct_pairs.json` is the input as-is; no labeling interventions, no manual
  curation, no data cleaning passes of any kind (reviewer decision
  2026-10-22). Methods must be robust to whatever the labels are; anomalies
  are characterized and documented, not "corrected".
Relaxed for this track only:
- **No-LLM rule suspended**: LLM calls (Ollama local server) are allowed as
  experiments — under hygiene: pinned model+digest, temperature 0 where
  available, prompt versions recorded, calls cached (no re-judging on re-runs),
  and evaluated by the SAME harness/number discipline. exp09-style probes are
  in scope; nothing moves to production without the same val protocol.
- Cost concerns are out of scope (reviewer has the hardware; re-embedding is
  fine). Simplicity of *change sets* is still kept (one checkbox each).

## 2. Naming & infrastructure contract (the §3 explain-first list — approved in review)

- `TEXT_PROCESSING_PLAN.md` — this document (repo root, satellite to PLAN.md;
  PLAN.md stays master — a track result adopted downstream gets a note in the
  relevant PLAN phase, e.g. Phase 6 reranker slot).
- `docs/experiments_text_processing.md` — ONE narrative log for the track,
  both datasets, append-only, one short row/section per run (date, run_id,
  what, numbers, verdict).
- NEW result tables: `docs/experiments_table_text_processing_omct.md` now,
  `docs/experiments_table_text_processing_upr.md` when the UPR stage starts.
  Wiring (schema change, approved here): `paths.experiment_table_text_processing_omct`
  / `_upr` config keys; producer commands map to these files when the run's
  `table.sweep_label`/command belongs to this track (mirrors the existing
  dataset-mapping in `experiment_table.py`; new constants follow).
- Sweeps: **`experiments_text_processing/`** (top-level, parallel to
  `experiments/` — reviewer decision: existing sweeps stay where they are).
  Infra change (approved here): the sweep loader adds the sweep file's own dir
  AND keeps `experiments/` importable so `embedder_pool.MODELS` stays shared
  (tiny loader tweak, docstring + test).
- Source: `src/cluster_rag/text_processors/`:
  - `base.py` — Protocols: `Chunker` (`segment(texts) -> list[list[str]]`,
    deterministic & pure; `version` = config hash that plugs the chunk
    *inputs* into the existing embedding-cache key (model, prefix,
    chunking-version, hash(text))), `FeatureExtractor` (pair -> float row,
    pure), `Combiner` (fit/score protocol; split handling stays with the
    caller).
  - `chunkers/` — one file per method, one class per file: `word_windows.py`,
    `sentence_windows.py`, `atoms.py` (sentence/clause atoms incl.
    comma/semicolon splits), `semantic.py` (boundary detection à la exp18;
    injects an `Embedder` port — NO model client inside), `question_parts.py`
    (question stem vs. answer-option decomposition, exp12),
    `full_text.py` (the no-op baseline chunker that keeps everything
    comparable).
  - `features/pair_features.py` — feature pack v0 (pure numpy: global/best
    chunk cosines, top-K chunk sims + spread, soft lexical overlap + char
    n-grams, question-part coverage, length, document/question-type one-hots,
    label-free by construction).
  - `combining/` — `logistic.py` (sklearn LR baseline), `lightgbm.py` (new dep,
    added when stage 3 starts), both behind the Combiner port.
  - Model clients (embedder, Ollama LLM) stay at the edges — wired in CLI
    code, never inside pure cores.
- Run-id grammar: unchanged `<dataset>-<command>-<detail>`; commands added per
  stage (below); stage-0 analysis reuses existing `omct-probe2-*` artifacts.

## 3. Stages (OMCT-first)

### 3.0 Methods inventory (what "try methods" concretely means)

**Reference-lineage methods** (from the untracked reference log; one line each, mapped to stages):

| ref | method | track stage |
|---|---|---|
| exp01 | full-text vs full-text cosine + threshold (baseline re-anchor) | 0 |
| exp02 | overlapping sentence windows on paragraphs (2 sent, step 1) | 2 |
| exp03 | fixed word windows (default 20 words, stride 10) | 2 |
| exp04 | (model swap with same chunking — we own the model axis separately) | n/a |
| exp05 | symmetric windowing (both sides chunked) | 2 |
| exp10 | clean both-sides word-chunk SBERT baseline | 2 |
| exp12 | question stem vs. answer-options decomposition (typed parts) | 1 |
| exp18(i) | paragraph atoms: sentence/clause units incl. comma/semicolon splits | 1 |
| exp18(ii) | semantic chunks: cut where adjacent-atom similarity drops | 2 |
| exp06 | local+global fusion: α·(best window) + (1−α)·(best full) | 2 |
| exp07 | top-K chunk interactions, decaying average − spread penalty | 2 |
| exp08 | peak shortcut: top-K threshold OR single-best above a higher threshold | 2 (+harness later) |
| exp11 | LR on simple features (cosine, Jaccard, bidirectional coverage) | 3 |
| exp13 | soft lexical match (token-embedding sim + char n-grams) | 3 |
| exp15/16 | top-K alignment-cosine features + df + alignment-kind one-hots | 3 |
| exp17 | ranked cross-chunk match profiles (sentences + 5-grams) as features | 3 |
| exp14 | attribution dictionary (explain what aligns — diagnostic only) | 3/optional |
| exp19 | cross-encoder reranker (our PLAN Phase 6 slot; not this track's core) | deferred |
| exp09 | embedding shortlist + LLM rerank | 4 |

**Our own inventions** (OMCT-focused, harness-judged; the reviewer will co-invent
more with the agent during the track):
- per-question-type / per-document calibrated thresholds + the
  "should-answer-nothing" negative channel — clever use of the pair similarity
  structure on the OMCT decision layer;
- aggregation hybrids of our own design (position/coverage-weighted top-K,
  spread/peak penalties beyond exp07/08);
- new methods proposed mid-track (by the reviewer or via AI-assisted
  ideation) — logged and judged like everything else.

Deliberately OUT of this track's menu (reviewer trim, 2026-10-22): UPR-side
retrieval/decision machinery (union shortlists, two-channel UPR decision), the
BM25-fusion channel (a UPR fusion idea; lexical FEATURES on OMCT stay in stage
3 via exp13), multi-vector/late-interaction scoring (standing PLAN Phase 6
topic). Those live in the main-plan discussions, not here.

Stage checkboxes reference this inventory; sweep cells instantiate concrete
(parameterized) variants of these methods — the inventory is the menu, the
sweeps are the orders.

### Stage 0 — Diagnosis (no new methods; tick when failure modes are classified)
- [ ] **0.1 Where does similarity fail?** Per-document and per-question-group
  AUC/error localization using the existing `omct-probe2-*` artifacts (14
  models, warm): error concentration by document, by question length/type,
  positives-vs-negatives separation per group; the label-0-above-label-1-p95
  anomaly quantified per document. Deliverable: a written failure-mode
  classification (representation vs. aggregation vs. labels) into
  `experiments_text_processing.md`, and the stage-1 entry point it implies.
  (The anomaly is treated as a fixed property of the data: documented,
  designed-around, never "fixed".)
### Stage 1 — Structural decomposition
- [ ] **1.1 Question-parts + atoms.** `question_parts.py` (stem vs options as
  separate embedding variants — exp12) and `atoms.py` (sentence/clause
  splitting); embed both sides' parts (cheap, cached); aggregate by max /
  top-K / mean; compare against `full_text.py` baseline on AUC + recall@k.
- Exit: measured contribution of structure alone (no learned model yet).
### Stage 2 — Segmentation ladder (the headline comparison)
- [ ] **2.1 Fixed windows** (`word_windows`: N in {15, 25, 40}, stride ~N/2, per
  card/context sanity) vs [ ] **2.2 sentence windows** (2 sentences, step 1)
  vs [ ] **2.3 semantic boundaries** (`semantic.py`: cut where adjacent-atom
  similarity drops below threshold; threshold swept on fit-side documents
  only) — each aggregated by max / top-K-with-decay (exp07) / local+global α
  (exp06, α swept); reported on all pairs (descriptive) with per-document
  breakdowns. Both reference-lineage methods and our own variants are in
  scope here.
- Exit: a ranked, artifact-backed statement of which segmentation+aggregation
  wins where — or "no chunking beats full text" (an honest result too).
### Stage 3 — Learned pair scorer (the "get crazy" slot)
- [ ] **3.1 Feature pack v0 + LR baseline**, [ ] **3.2 LightGBM combiner** (dep
  added here), [ ] **3.3 chunk-profile features** (best chunk matches,
  spread/peak features — exp17/15/16 lineage) folded in.
- Discipline: document-level holdout — fit on two of the three source
  documents, evaluate on the held-out third (all three rotations when cheap);
  negatives = unlisted pairs (subsampled); calibration curves + thresholds
  reported per document. OMCT line stays auxiliary; it informs method choice,
  never UPR metrics.
### Stage 4 — LLM experiments (rule suspension lives here)
- [ ] **4.1 exp09-style rerank** on embedding shortlists (pinned model, temp 0,
  cached responses, prompt version in run artifacts);
- [ ] **4.2 question/side enrichment** (LLM-written question paraphrases or
  topic descriptions embedded as extra variants) — same evaluation protocol;
  wins only count if they beat stage 1-3 numbers on the same split.
### Stage 5 — UPR port (deferred until OMCT results exist)
- [ ] port surviving ideas: combiner/feature scorer through the main UPR
  harness (train-split fit, val calibration, ONE test shot per adopted config);
  chunking on UPR gets its explicit null-check documented.

## 4. Cross-cutting rules (binding here)

- Every number: run artifact first, then log row; never hand-written tables.
- OMCT findings annotate method choice; they are never a UPR claim.
- Learned models: fit data never touches the holdout document (stage 3) or the
  UPR val/test splits (stage 5); thresholds calibrated only on fit-side.
- LLM hygiene (stage 4): model+digest, temperature, prompt version in every
  artifact; cached calls; a rerun must be reproducible modulo model drift.
- Tests per processor (tiny fixtures + FakeEmbedder); ruff+pytest before any
  tick; commits stay one change set each.

## 5. Open questions
- The label anomaly (label-0 pairs above label-1 p95): assumed to be a fixed
  property of the data (provenance unknown, verification unavailable and not
  pursued — reviewer decision). Stage 0 documents its size and location;
  methods are chosen for robustness to it, not correction of it.
- Question `question_text_sources`/document idiosyncrasies: characterized in
  stage 0, not assumed.

## 6. Milestones
- T1: stages 0-1 (diagnosis + structure) → what matters on OMCT decided.
- T2: stage 2 (segmentation ladder) → the chunking verdict.
- T3: stage 3 (learned scorer) → the reranker-slot evidence for PLAN Phase 6.
- T4: stage 4-5 as reviewer-approved extensions.

## 7. First execution change set (agreed)
- Skeleton: TEXT_PROCESSING_PLAN.md (this content), `docs/experiments_text_processing.md`
  (header only), `experiments_text_processing/` + loader tweak + test,
  `src/cluster_rag/text_processors/{base,chunkers/{full_text,atoms}.py}` with
  unit tests, config keys for the new table + `paths` wiring.
- Then Stage 0 immediately (artifact analysis, no new compute).
