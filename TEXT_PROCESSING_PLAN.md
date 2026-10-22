# TEXT_PROCESSING_PLAN.md — text processing / segmentation track

Satellite to `PLAN.md` (created 2026-10-22, revised 2026-10-23; PLAN.md stays master — a track result adopted downstream gets a note in the relevant PLAN phase, e.g. Phase 6 reranker slot).
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
- First focus: **OMCT** (paragraphs of mean ~478 chars / ~71 words, where
  segmentation can matter; binary relevance labels; a live suspicion that
  data/labels/aggregation — not representation — drives the pipeline's observed
  anomalies).

### 0.1 Measured baseline of the OMCT data (2026-10-23, ad-hoc diagnostics)

**Caveat first, because it binds (AGENTS.md rule 4):** these numbers were
produced by an ad-hoc script against the warm embedding cache. They are **not
harness runs** — no `results/` artifact, no `docs/experiments.md` row — so they
are *indicative*, not reportable. Change set **CS-1** (`analyze-omct`) must
re-derive every one of them through the harness before any of them enters a
log row or a table, and Stage 0's written classification is based on the
harness version.

Corpus shape (from the data file itself):

| fact | value |
|---|---|
| judged pairs | 1520 (524 positive / 996 negative) |
| distinct paragraphs / questions | 225 / 294 (one text per question number; no collisions after `strip_block`) |
| documents | 3: AI (99 pairs, 79 pos = 80%), CIDH (576, 234 = 41%), NU (845, 211 = 25%) |
| same-document pair grid | 44,467 cells, 1520 judged = **3.4% covered** |
| paragraph length | mean 71 words / 478 chars; median 63 words / 427 chars; max 344 words |
| paragraph sentences | 190/225 multi-sentence, 35 single-sentence |
| paragraph boilerplate | "The Committee" in 33/225, "The Committee recalls" in **0/225** |
| questions | all start with "Question:"; 281/294 carry an "Answer choices:" block (2-16 options, median 5); 261/294 include a "no data / n/a"-style option |
| answer options | overwhelmingly scales ("Very often", "Yes", "Generally, yes") — a **question-type** signal, not topical content |
| question sharing across documents | 186/294 question numbers judged in >1 document; **101 are positive in one document and negative in another** |
| paragraph sharing across documents | 0/225 (paragraph holdout is clean; question holdout is **not**) |
| top-1 same-document candidate per paragraph (225 rows) | given positive 55, given negative 17, **unjudged 153** |

Score structure (arctic2, `strip_block`, warm cache):

| group | n | mean | p50 | p95 |
|---|---|---|---|---|
| positives (label 1) | 524 | 0.371 | 0.374 | 0.531 |
| negatives (label 0, given) | 996 | 0.379 | 0.389 | 0.511 |
| control: unjudged same-document pairs | 4014 | 0.289 | 0.287 | 0.443 |

Rank percentile inside the paragraph's own same-document candidate list
(1.0 = top): positives median **0.097**, negatives median **0.207**, unjudged
control median **0.565** (what random should give).

Chunking sensitivity (word windows n=20, stride=10):

| scoring scheme | pair AUC | negatives' median rank percentile |
|---|---|---|
| full text vs full text | 0.4680 | 0.207 |
| paragraph chunked | 0.4708 | 0.219 |
| question chunked | 0.4507 | 0.208 |
| both sides chunked | 0.4354 | 0.202 |

Derived reading (the four things Stage 0 must formalize):

1. **The negatives are reviewer-verified model false positives.** The dataset's
   own `description` says the negatives come from "exp18 FP relabel
   (reject_model only)" — i.e. the chunking model's highest-scoring non-links
   that a human then rejected. So OMCT is a **hard-negative** benchmark and
   its absolute AUC (0.45-0.60 across the 14-model pool) measures how hard the
   reviewed FPs are, **not** embedding quality. Never read "AUC 0.47" as "the
   model is bad".
2. **The "label-0 above label-1 p95" anomaly is tail overlap, not a
   contradiction.** The two distributions are nearly identical (means 0.379 vs
   0.371); the negatives sit at the ~21st percentile of their paragraph's
   candidate list (top fifth, not the top), against a random control at the
   56th. Documented, designed-around, never "fixed" (raw data is read-only).
3. **The judged set covers 3.4% of the grid; unjudged pairs are *unreviewed*,
   not negatives.** The positives are only as complete as the review that
   surfaced them. So "unlisted pair" = *presumed* negative with an unknown
   positive rate, and it must never be pooled with the verified hard
   negatives (see Stage 3).
4. **The segmentation effect is small (~±0.03 AUC) and the benchmark is not
   rigged against it.** The negatives' position in the ranking is stable
   across full-text and chunked scoring, so Stage 2's comparison is fair — but
   ±0.03 sits at the resolution limit of an unpaired AUC comparison
   (SE ≈ 0.02 at n=1520). **Stage 2 must use paired statistics.**

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
  are characterized and documented, not "corrected". This also holds for the
  relaxed LLM rule: an LLM may score pairs, it may **not** relabel, curate
  or "clean" the data.
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
  / `_upr` config keys. **The mapping is command-derived, never a heuristic**
  (mirrors the existing command→dataset mapping in `experiment_table.py`):
  the track's command family (e.g. `chunk-omct`) maps to
  dataset=`omct` + track=`text_processing` → the `_omct` file; the UPR
  counterpart maps to the `_upr` file. Run ids keep the `omct-`/`upr-`
  dataset prefixes, so a track row is still visibly auxiliary.
- **AGENTS.md and PLAN.md are updated in change set 0** (they currently say
  "exactly two table files" and list neither the new log, the new tables nor
  the new package): AGENTS.md gains the track log + the two track table files
  + `text_processors/` in the repository layout and the pure-core list; PLAN.md
  gains a pointer line to this satellite plan.
- Sweeps: **`experiments_text_processing/`** (top-level, parallel to
  `experiments/` — reviewer decision: existing sweeps stay where they are).
  Infra change (approved here): the sweep loader adds the sweep file's own dir
  AND keeps `experiments/` importable so `embedder_pool.MODELS` stays shared
  (tiny loader tweak, docstring + test).
- Source: `src/cluster_rag/text_processors/`:
  - `base.py` — Protocols: `Chunker` (`segment(texts) -> list[list[str]]`,
    deterministic & pure; `version` = config hash that plugs the chunk
    *inputs* into the existing embedding-cache key), `FeatureExtractor`
    (pair -> float row, pure), `Combiner` (fit/score protocol; split handling
    stays with the caller).
  - `chunkers/` — one file per method, one class per file: `word_windows.py`,
    `sentence_windows.py`, `atoms.py` (sentence/clause atoms incl.
    comma/semicolon splits), `semantic.py` (boundary detection à la exp18;
    injects an `Embedder` port — NO model client inside), `question_parts.py`
    (question stem vs. answer-option decomposition, exp12),
    `full_text.py` (the no-op baseline chunker that keeps everything
    comparable).
  - `features/pair_features.py` — feature pack v0 (pure numpy: global/best
    chunk cosines, top-K chunk sims + spread, soft lexical overlap + char
    n-grams, question-part coverage, question-type one-hots, length;
    label-free by construction; **no document-identity feature** — see
    Stage 3's leakage rules).
  - `combining/` — `logistic.py` (sklearn LR baseline), `histgb.py`
    (sklearn `HistGradientBoostingClassifier`), both behind the Combiner
    port. **`lightgbm` is deferred**: scikit-learn is already a core dep and
    covers this problem; a new dependency needs evidence that it beats
    HistGB (AGENTS §3 explain-first).
  - Model clients (embedder, Ollama LLM) stay at the edges — wired in CLI
    code, never inside pure cores. `text_processors` may import
    `cluster_rag.embed.base` (a `typing.Protocol` + numpy only) and nothing
    heavier; no `torch`/`sentence-transformers`/`httpx` import anywhere in
    this package.
- **Embedding cache: no change to the cache key layout.** The chunking version
  goes into the *existing* version slot —
  `CachedEmbedder(backend, root, preprocessing_version="<chunking-version>")`
  — and the hashed text is the **chunk** itself, so the effective key is
  (model, prefix, chunking-version, sha1(chunk)) exactly as intended, while
  the warm cache for the 14 pool models stays valid. Adding a new key
  component would invalidate that cache for no gain.
- Run-id grammar: unchanged `<dataset>-<command>-<detail>`; commands added per
  stage (below).

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
- **question-type features** (is it a frequency question? a yes/no? does it
  offer a "no data" category? how many options?) — the answer-option set is a
  *type* signal, not content (see Stage 1);
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

- [ ] **0.1 The pair-score artifact (`analyze-omct`).** A new CLI command that
  writes ONE artifact per run with: per-pair scores (paragraph x question) for
  the selected models, the **control set** (sampled unjudged same-document
  pairs, recorded with their sampling seed and the exclusion of all judged
  pairs), per-document strata with base rates, the negative-enrichment
  measurement (rank percentile of verified negatives vs positives vs control),
  and the score distributions per label. Warm cache, no new compute beyond
  cosine; the artifact is the *only* thing Stage 0's claims may cite.
  (The existing `omct-probe2-*` artifacts hold no per-pair scores, so this
  cannot be derived from them — hence the command.)
- [ ] **0.2 The written failure-mode classification.** Into
  `docs/experiments_text_processing.md`: representation vs. aggregation vs.
  label-incompleteness vs. benchmark-construction, with the §0.1 numbers
  re-derived through the harness; the "what this data can and cannot answer"
  section; and the stage-1 entry point it implies. The label anomaly is
  recorded as a **derived, quantified property of the hard-negative
  construction** (§0.1 items 1-2) — documented, designed-around, never
  "fixed".
- Exit: every claim in §0.1 re-derived from a `results/<run_id>/` artifact and
  logged; the two negative subsets (verified / unjudged-proxy) are defined and
  implemented for Stage 3; the control set exists as a reusable artifact.

### Stage 1 — Structural decomposition

- [ ] **1.1 Question parts, reframed.** The answer options are scales
  ("Very often", "Yes", "Generally, yes"), so the decomposition is:
  (a) **stem-only vs. full question** (does the option block help or hurt the
  topical signal?), and (b) **question-type features** (frequency? yes/no?
  "no data" option present? option count?) as label-free features — instead of
  exp02's "one embedding variant per answer option", which mostly injects
  scale words. `atoms.py` (sentence/clause splitting) covers the paragraph
  side. Embed both sides' parts (cheap, cached); aggregate by max / top-K /
  mean; compare against the `full_text.py` baseline on AUC + recall@k.
- [ ] **1.2 Report on both negative subsets** (verified hard negatives and the
  unjudged-proxy control), pooled **and** per-document (the pooled number is
  dominated by the NU document: 845/1520 pairs, 25% positive; the AI document
  is 99 pairs at 80% positive — a per-document macro average is reported next
  to every pooled number).
- Exit: measured contribution of structure alone (no learned model yet), with
  the per-document breakdown and both negative subsets.

### Stage 2 — Segmentation ladder (the headline comparison)

- [ ] **2.1 Fixed windows** (`word_windows`: N in {15, 25, 40}, stride ~N/2) vs
  [ ] **2.2 sentence windows** (2 sentences, step 1) vs [ ] **2.3 semantic
  boundaries** (`semantic.py`: cut where adjacent-atom similarity drops below
  threshold; threshold swept on fit-side documents only) — each aggregated by
  max / top-K-with-decay (exp07) / local+global α (exp06, α swept); reported
  on all pairs (descriptive) with per-document breakdowns. Both
  reference-lineage methods and our own variants are in scope here.
- [ ] **2.4 Model axis: `embeddinggemma` (Ollama, 768d, template prefixes) and
  `voyage-4-nano` (HF, 2048d, revision `433ece5e…`, retrieval prompts)** —
  the whole ladder runs on both, not just the winner. Both are UPR
  semifinalists and both sit mid-pack on OMCT pair-AUC (0.537 / 0.500 vs the
  pool's best 0.599), so the ladder is measured where the signal is *not*
  strongest: the control set and the paired tests carry the weight.
  Chunk side/kind assignment: a paragraph chunk is encoded with the
  `document` prefix, a question chunk with the `query` prefix (embeddinggemma's
  are `{content}` templates — the existing `apply_prefix` handles them).
- [ ] **2.5 Paired statistics.** Every ladder comparison is paired (same
  pairs, same model, different segmentation), so report the **paired bootstrap
  CI of the AUC difference** (or DeLong) — never two independent AUCs. The
  measured effect size (~±0.03 AUC) sits at the unpaired resolution limit
  (SE ≈ 0.02 at n=1520), so an unpaired comparison cannot settle it. Also
  report the max-over-k inflation: a chunked method takes a max over more
  candidates, so its score scale is not comparable to full text's — the
  control set makes that visible.
- Exit: a ranked, artifact-backed statement of which segmentation+aggregation
  wins where — with paired CIs, per-document breakdown, and both negative
  subsets — or "no chunking beats full text" (an honest result too).

### Stage 3 — Learned pair scorer (the "get crazy" slot)

- [ ] **3.1 Feature pack v0 + LR baseline**, [ ] **3.2 gradient-boosted
  combiner** (sklearn `HistGradientBoostingClassifier`; `lightgbm` only if it
  beats HistGB and the dependency is approved), [ ] **3.3 chunk-profile
  features** (best chunk matches, spread/peak features — exp17/15/16 lineage)
  folded in.
- **Discipline.**
  - *Split*: document-level holdout — fit on two of the three source
    documents, evaluate on the held-out third (all three rotations when
    cheap). Thresholds/calibration fitted on the fit side only.
  - *Two negative subsets, reported separately, never pooled*:
    **A — verified hard negatives** (the 996 reviewer-confirmed ones; the
    evaluation set) and **B — unjudged-proxy control** (sampled from the
    same-document grid, excluding every judged pair; *presumed* negative,
    unreviewed, positive rate unknown). B is a diagnostic control and a
    training source — it is never presented as ground truth.
  - *Leakage rules* (both follow from the measured data structure):
    (i) **no corpus-level question statistics** — 186/294 question numbers are
    judged in more than one document and 101 are positive in one document and
    negative in another, so a question-level prior leaks across the holdout;
    question-level statistics, if any, are computed on the fit side only.
    (ii) **no document-identity feature** (and beware length as a proxy) —
    the per-document base rates differ wildly (80% / 41% / 25%) and paragraph
    length tracks the document (AI median 364 chars vs NU 863), so an
    identity or length feature can memorize the document instead of the topic.
  - *Sanity floors*: a label-permutation baseline and a random-vector
    baseline, so a "win" can be checked against chance.
  - *Reporting*: calibration curves + thresholds per document; pooled and
    per-document macro; both negative subsets separately.
- OMCT line stays auxiliary; it informs method choice, never UPR metrics.

### Stage 4 — LLM experiments (rule suspension lives here)

- [ ] **4.1 exp09-style rerank** on embedding shortlists (pinned model, temp 0,
  cached responses, prompt version in run artifacts). Local models available
  on this machine: `gemma4:e2b` (5.1B), `llama3.2:3b`, `gpt-oss:latest`
  (20.9B), `qwen2.5-coder:14b` (`laya:latest` is only 421M — too small to
  trust). The model is **named per experiment**, never "a local LLM". LLM
  cache key: `(model_id, digest, prompt_version, temperature, max_tokens,
  sha1(input))` → the raw response.
- [ ] **4.2 question/side enrichment** (LLM-written question paraphrases or
  topic descriptions embedded as extra variants) — same evaluation protocol;
  wins only count if they beat stage 1-3 numbers on the same split.

### Stage 5 — UPR port (deferred until OMCT results exist)

- [ ] port surviving ideas: combiner/feature scorer through the main UPR
  harness (train-split fit, val calibration, ONE test shot per adopted config);
  chunking on UPR gets its explicit null-check documented. **Exit criterion:**
  the UPR-side chunking axis ends in a *documented null* (UPR texts average
  151 chars — one recommendation each — so chunking them is near-vacuous by
  construction); the stage's real deliverable is the combiner/feature
  transfer to PLAN's Phase 6 reranker slot.

## 4. Cross-cutting rules (binding here)

- Every number: run artifact first, then log row; never hand-written tables.
  The §0.1 diagnostics are the exception *until* CS-1 re-derives them — after
  that, only harness numbers count.
- OMCT findings annotate method choice; they are never a UPR claim.
- **Method comparisons are paired and reported with a CI.** No verdict rests
  on two independent AUCs.
- **No pooled number ever mixes the two negative subsets** (verified hard
  negatives / unjudged-proxy control).
- Learned models: fit data never touches the holdout document (stage 3) or the
  UPR val/test splits (stage 5); thresholds calibrated only on fit-side.
- LLM hygiene (stage 4): model+digest, temperature, prompt version in every
  artifact; cached calls; a rerun must be reproducible modulo model drift.
  An LLM may score, never relabel.
- Tests per processor (tiny fixtures + FakeEmbedder); ruff+pytest before any
  tick; commits stay one change set each.

## 5. Open questions

- The label anomaly (label-0 pairs above label-1 p95) is **no longer an open
  question**: §0.1 derives it from the hard-negative construction (the
  negatives are exp18's model FPs; the two score distributions overlap almost
  completely). Stage 0 (CS-1) re-derives the numbers through the harness and
  records them; the anomaly is documented and designed-around, never corrected.
- The *positive* set's completeness is still open: positives come from the
  merged `Questions` column after reviewing exp18's FPs **and** FNs, so
  unlisted pairs may hide unreviewed true positives. Stage 0 characterizes the
  size of that gap (the 153 unjudged top-1 candidates per paragraph are the
  leading symptom); no data is changed.
- Question `question_text_sources` / document idiosyncrasies: characterized in
  stage 0, not assumed.
- `lightgbm`: a new dependency is only proposed if sklearn's HistGB is beaten
  (AGENTS §3 explain-first).

## 6. Milestones

- T1: stages 0-1 (diagnosis + structure) → what matters on OMCT decided.
- T2: stage 2 (segmentation ladder) → the chunking verdict.
- T3: stage 3 (learned scorer) → the reranker-slot evidence for PLAN Phase 6.
- T4: stage 4-5 as reviewer-approved extensions.

## 7. First execution change sets

Numbered, one per commit, in order (the plan's style: one checkbox per change
set):

- [x] **CS-0 — Skeleton.** `TEXT_PROCESSING_PLAN.md` (this content);
  `docs/experiments_text_processing.md` (header only);
  `experiments_text_processing/` + the sweep-loader tweak (+ docstring + test)
  so `embedder_pool.MODELS` stays shared; `src/cluster_rag/text_processors/`
  with `base.py` and `chunkers/{full_text,atoms}.py` + unit tests; the new
  `paths.experiment_table_text_processing_omct` / `_upr` config keys and the
  command-derived table mapping; **AGENTS.md + PLAN.md updates** (new log, new
  tables, new package, satellite-plan pointer).
  *(done 2026-10-23: the package `src/cluster_rag/text_processors/` with the
  `Chunker`/`FeatureExtractor`/`Combiner` ports, the `full_text` and `atoms`
  chunkers and a 17-test unit suite (incl. a guard that the package imports no
  model client); the chunking version composes into the *existing* cache
  version slot and the identity chunker keeps the warm cache valid
  (`cache_version("raw-v1") == "raw-v1"`); `experiments_text_processing/` with
  the loader tweak (the file's dir **and** `experiments/` on `sys.path`, tested
  by importing the real `embedder_pool` from a track-dir sweep file) and an
  explicit sweep dispatch (an unimplemented command now aborts instead of
  falling through to `probe-omct`); the command → (dataset, track) →
  table-file mapping in `experiment_table.py` with the two new config keys and
  a CLI helper that all four append sites now use; the track log + its OMCT
  table file; AGENTS.md/PLAN.md updated (layout, log/table rules, pure-core
  rule, satellite pointer, Phase 6/9 track-input notes). No run, no metric.)*
- [ ] **CS-1 — `analyze-omct`.** The Stage-0 artifact command: per-pair scores for
  the selected models, the control set (seeded sampling, judged pairs
  excluded), per-document strata with base rates, the negative-enrichment
  measurement, per-label score distributions. Warm cache, no re-encoding.
- [ ] **CS-2 — Stage 0 write-up.** The failure-mode classification and the
  "what this data can answer" section in `docs/experiments_text_processing.md`,
  based on the CS-1 artifact (this is where §0.1's diagnostics become
  reportable numbers).
- Then Stage 1.

## 8. The durable deliverable (named, so it is not lost)

The headline is **not** "which chunker won". It is **a characterized
pair-relevance benchmark on OMCT**: the verified hard-negative set, the
unjudged-proxy control set, per-document strata with base rates, and a
"what this data can and cannot answer" section — an artifact that PLAN Phase 6
(cross-encoder training pairs) and Phase 9 (fine-tuning pairs) can consume
directly. Both of those phases need *negative pairs* and today have 996
verified hard ones plus a 44k-cell unreviewed space with no recipe for mining
it safely; a measured "which unjudged pairs are safe negatives" rule (e.g.
similarity margin + a reviewer-confirmed spot-check) is the most valuable
thing this track hands back to the main plan.
