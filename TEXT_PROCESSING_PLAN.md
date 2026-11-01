# TEXT_PROCESSING_PLAN.md — text processing / segmentation track

Satellite to `PLAN.md` (created 2026-10-22, revised 2026-10-23, split 2026-10-29; PLAN.md stays master — a track result adopted downstream gets a note in the relevant PLAN phase, e.g. Phase 6 reranker slot).
Style follows PLAN.md: numbered stages, one checkbox per change set, exit criteria, open questions. Reviewer = project owner (solo — no external collaboration assumed anywhere in this track).

**This file is the LIVE plan: the open work.** The executed pre-registrations
and the questions they answered moved to `docs/text_processing_plan_archive.md` on
2026-10-29 — **verbatim**, and that archive is append-only like the logs. Every moved
block leaves a tombstone line here naming the archive and the track log's section, so
*moved* is always distinguishable from *lost*.

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

**Moved 2026-10-29** (superseded by CS-1, which re-derived every number through the
harness) → `docs/text_processing_plan_archive.md` §0.1, and the track log's §"Runs".

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
    `units.py` (the Stage-3.3 chunk profile's two granularities: the
    sentences and the word pieces inside them — the pieces tile a sentence),
    `full_text.py` (the no-op baseline chunker that keeps everything
    comparable).
  - `features/pair_features.py` — the feature pack (the global and best chunk
    cosines, the top-K chunk sims + spread, the soft lexical overlap + char
    n-grams, the question-part coverage, the question-type one-hots; **no
    length feature** — length is a document proxy, see Stage 3's leakage
    rules; label-free by construction; **no document-identity feature**; and
    the Stage-3.3 **chunk profile** in its own group: the ranked cross-chunk
    match profile (exp17) and the alignment kinds (exp16)). It
    is a pure core (numpy + the standard library), but the chunk-level
    features need the chunk embeddings, so it takes the **`Embedder` port**
    (as the semantic chunker does) and embeds the pair's own chunks — the
    package still imports no model client, the CLI wires the cached backend
    in.
  - `combining/` — `logistic.py` (sklearn LR baseline), `histgb.py`
    (sklearn `HistGradientBoostingClassifier`), both behind the Combiner
    port. **`lightgbm` is deferred**: scikit-learn is already a core dep and
    covers this problem; a new dependency needs evidence that it beats
    HistGB (AGENTS §3 explain-first). The combiner is the one place the
    track's package imports a library beyond numpy and the standard library
    (scikit-learn: no model download, no network, no I/O); the guard test
    still forbids torch / sentence-transformers / httpx / sqlite3.
  - Model clients (embedder, Ollama LLM) stay at the edges — wired in CLI
    code, never inside pure cores. `text_processors` may import
    `cluster_rag.embed.base` (a `typing.Protocol` + numpy only) and nothing
    heavier; no `torch`/`sentence-transformers`/`httpx` import anywhere in
    this package (scikit-learn in `combining/` excepted, as above).
  - The **random-vector floor's** embedder lives one level up, in
    `src/cluster_rag/embed/random.py` (`RandomEmbedder`, the same `Embedder`
    port): a seeded random unit vector of the model's dimension per
    `(kind, text)`, drawn from `sha256(seed | kind | text)`. It is a *model
    client* in the sense that it replaces one, so it belongs with the
    backends, not in the track's pure core — and it is **never** wrapped in
    the disk cache, so the random vectors cannot pollute the model's cache.
  - The **decision models** (Ollama's new model type: a *state* (the text to
    judge) plus up to 64 typed *questions* — `noul` (true/false), `choice`
    (pick from a list), `score` (a rubric) — answered with **probabilities**
    in one forward pass; no decoder, no generation) are a **model client**
    like the embedder: they live at the edge, in
    `src/cluster_rag/decision/`, and never inside a pure core. The port is a
    `Decider` (a state + a question spec -> the answers with their
    probabilities), the client is the `/v1/systemone` caller (the endpoint is
    **not** in the Ollama CLI or its Python/JS libraries, so it is a raw HTTP
    call), and the **cache** is keyed by (model id, digest, prompt version,
    the state, the question spec) — the track's LLM hygiene, unchanged: a
    pinned model **and digest**, a recorded prompt version, cached calls, no
    re-judging on a re-run, and no temperature (there is no sampling). The
    pure side (the question specs, the arm's construction, the measurement)
    lives in `text_processors/` like every other pure piece.
  - The decision models' **pure side** is the sub-package
    `text_processors/decision/`: `questions.py` (the typed question specs and
    the prompt's rendering) and `probe.py` (the cells, the **call plan**, the
    synthetic competence control and the measurement). It imports numpy and
    the standard library only — no httpx, no model client.
  - **The prompt's two halves.** The config carries the **task** instruction
    ("is the paragraph about the same topic as the question?") and the
    **state's shape** decides where the *question* goes: the `pair` shape puts
    it in the state (next to the paragraph), the `paragraph` shape puts it in
    the instructions (in front of the task). A task that repeats the question
    is rejected at config time: it would put the question into the call
    **twice**, which biases the grid's comparison and breaks the smaller
    model's budget (measured: laya cannot take the longest paragraph and
    question twice at all).
  - **The input budget** is a per-model config value (`max_input_tokens`):
    the server refuses a prompt longer than the model's context, and the
    measured limits are **512 tokens for laya** and **~2,500 for tev1**. The
    call planner groups the cells into calls that fit the budget and
    **truncates the state's paragraph** when a single cell does not fit — the
    question and the instructions are never cut, and every truncated cell is
    flagged in the artifact (a shortened input is never silent, the same
    discipline as the embedding probe's "paragraphs over the context").
- **Embedding cache: no change to the cache key layout.** The chunking version
  goes into the *existing* version slot —
  `CachedEmbedder(backend, root, preprocessing_version="<chunking-version>")`
  — and the hashed text is the **chunk** itself, so the effective key is
  (model, prefix, chunking-version, sha1(chunk)) exactly as intended, while
  the warm cache for the 14 pool models stays valid. Adding a new key
  component would invalidate that cache for no gain.
- **The decision models' cache is a separate cache** (its own root under
  `cache/`), keyed as above: it must never mix with the embedding cache (a
  decision answer is not a vector).
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


The **executed** stages' checkboxes are in `docs/text_processing_plan_archive.md` §3;
the open ones are below.

### Stage 1 — Structural decomposition

**Executed 2026-10-25…2026-10-26** (CS-3 the paragraph-side atoms, CS-5 the
question-type features; boxes 1.1a, 1.1b, 1.2 all ticked) → the full pre-registration is
in `docs/text_processing_plan_archive.md` §3 Stage 1; the verdict is in the track log's
§"Stage 1.1a — the paragraph side's atoms: the verdict" (the atoms are a **scale shift**,
not discrimination).


### Stage 2 — Segmentation ladder (the headline comparison)

**Executed 2026-10-26** (CS-4 the ladder, CS-7d the chunk header, CS-7e the late
chunking; boxes 2.1, 2.4, 2.5, 2.6, 2.7 all ticked) → the full pre-registration is in
`docs/text_processing_plan_archive.md` §3 Stage 2; the verdicts are the track log's
§"Stage 2 — the verdict" (chunk the paragraph side, never the question side),
§"Stage 2.6" and §"Stage 2.7".


### Stage 3 — Learned pair scorer (the "get crazy" slot)

**Executed 2026-10-26** (CS-5 the pack, CS-6a the ablation, CS-6b the booster, CS-6c
the chunk profile, CS-6d the random-vector floor, CS-6e the reporting) → the full
pre-registration is in `docs/text_processing_plan_archive.md` §3 Stage 3; the verdicts
are the track log's §"Stage 3.1" … §"Stage 3 — the reporting" (the **features** are
the lever, not the learner).


### Stage 3.5 — The decision models (a probe, then a decider)

**Executed 2026-10-26** (CS-7a the probe, CS-7b the grid arm + the rule, CS-7c the
shortlist rerank; boxes 3.5.1, 3.5.2, 3.5.3 all ticked) → the full pre-registration is in
`docs/text_processing_plan_archive.md` §3 Stage 3.5; the verdicts are the track log's
§"Stage 3.5", §"Stage 3.5.2" and §"Stage 3.5.3" (the decision model's within-document
AUC 0.7701 against the cosine's 0.5020–0.5632).

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
  *(the port's **methods** are done — CS-8a…CS-9f: the UPR command and the
  guard, the segmentation ladder, the text-side extras, the feature pack with
  the logistic combiner and the three floors, the chunk profile / the
  ablation / the boosted combiner, the reporting blocks, and the decision
  probe. Two of the stage's premises need the reviewer's attention, and the
  checkbox stays open because the **one test shot for the adopted config** is
  still owed:*

  * ***The chunking axis is a null *pooled*, not everywhere*** (CS-8b): the
    two chunkers that do not split UPR's texts give exactly zero in every
    bucket, but the atoms and the word windows gain +0.11 to +0.13 of
    micro-F1 in the 300+ character bucket (25 texts). The exit criterion's
    "near-vacuous by construction" holds for the *pooled* number and not for
    the long-text tail.
  * ***The stage's deliverable is bigger than the combiner*** (CS-9): the
    decision model (tev1) reaches a shortlist recall@50 of **0.9978** on the
    UPR sample against the champion's 0.7322 — it is a near-perfect
    *candidate generator*, and it finds the eight unseen labels (the unseen
    bucket's micro-F1 0.4706 against the champion's 0.0000), which the
    champion's geometry cannot do at all. PLAN's Phase 6 reranker slot has
    its evidence; **the slot decision is the reviewer's**.)*

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

The **answered** questions moved to `docs/text_processing_plan_archive.md` §5 on
2026-10-29 — a closed question is history, not work, and it keeps its answer verbatim
there. The ones still open:

- The *positive* set's completeness is still open, and it will stay **open
  and unmeasured**: positives come from the merged `Questions` column after
  reviewing exp18's FPs **and** FNs, so unlisted pairs may hide unreviewed
  true positives. Measured directly from the data file (to be re-derived in
  CS-1's artifact): **70 of 225 paragraphs carry no positive record at all**
  (NU 29/75, CIDH 38/112, AI 3/38) — the positive set is demonstrably not
  exhaustive. What we do **not** measure is the rate at which the ~43k
  unjudged cells hide positives: that needs human labelling, which the
  reviewer ruled out on 2026-10-23 (no time). Consequence, stated plainly:
  the track's safe-negative rule (§8) ships as the **margin half only** — a
  similarity-margin heuristic, without the reviewer-confirmed spot-check the
  original formulation proposed — and CS-2's "what this data can and cannot
  answer" section must say so. No data is changed.
- **The "same-report direction" mechanism (CS-2, 2026-10-24): a hypothesis,
  not a measurement.** The Stage-0 artifact holds scores, rank positions and
  distributions, not the embedding space, so the plan's candidate mechanism
  for the measured floor (ordinary same-document cells averaging
  0.2194–0.2436, with 5.6–8.6% of them above the median verified positive)
  cannot be tested by it. Settling it needs a new measurement in
  `analyze-omct` (e.g. the share of the similarity direction shared by all
  paragraphs of one document) — a separate change set, with the reviewer's
  approval, before the mechanism is called more than a hypothesis.
- **The swept semantic threshold is not identifiable on three documents
  (CS-4, 2026-10-26).** The document-level cross-fit chose a different
  threshold per document *and* per rule, and the fit side's AUC did not
  predict the holdout's (arctic2's `top_k_mean_2` with NU held out: fit AUC
  0.5886 at threshold 0.4, holdout AUC 0.4565 — below the full-text
  baseline's 0.4773 for the same document, while the grid's best holdout
  value was 0.5663 at another threshold). **Open:** would a finer holdout
  (paragraph-level inside one document) make the threshold identifiable, or
  is a threshold that must be fitted on a *corpus* statistic (which the
  track's leakage rules forbid) the only kind that works?
- **The chunk profile is not the lever either (CS-6c, 2026-10-26).** The
  features that say *where* the paragraph and the question match carry no
  label-driven signal beyond what the v0 pack's best chunk cosine already
  carries: the block alone is indistinguishable from the full-text cosine
  (all three paired intervals include zero), in the full pack it buys a small
  gain that is established on one of the three models only (+0.0172 on
  voyage-4-nano), and it *raises the chance floor* above chance (0.514–0.534
  against the v0 pack's 0.494–0.505) — a label-free signal. Its lower ranks
  are **length proxies** (the length correlation grows to +0.43…+0.56 with
  the rank's depth), which the plan's leakage rule (ii) forbids as a feature.
  **Open:** the feature ceiling on this benchmark is the **corpus**, not the
  pack: 294 questions, a 1,520-pair reviewed set and a bag-of-words overlap
  over them. Is any *other* family of features (not a re-measurement of the
  similarity, not a corpus statistic) likely to beat the lexical block here,
  or is the track's feature work done at 0.65?
- **exp16's `log(1+df)` was dropped, not measured (CS-6c, 2026-10-26).** The
  reviewer's decision: it is a statistic of the **labeled corpus**, so it
  breaks the pack's pair-local contract; a fit-side df would need a
  per-rotation feature matrix (the features are currently computed once for
  the grid); and with ~150 fit-side paragraphs the counts would be 0/1/2, so
  the feature would be noise. **Open:** if a *rarity* signal is wanted, the
  pair-local way to get it is not df but the **alignment's specificity**
  (does the best alignment match a *distinctive* phrase or a generic one?) —
  which would need a corpus statistic of some kind, so the tension between
  "rarity" and "pair-local" is the real open question, not the feature's
  name.
- **The alignment-kind taxonomy is a choice, and the chosen one is at chance
  (CS-6c, 2026-10-26).** The reviewer chose "which granularity produced the
  best alignment × whether the two aligned units share a content word"
  (2026-10-26); measured, the kinds are at chance (0.469–0.531) and nearly
  constant on the ordinary cells (the best alignment is a word-piece match
  99% of the time). **Open:** would a taxonomy that says *which question part*
  the alignment lands in (the stem vs an answer option) say something? It
  would need the question's options inside the profile's question side, so it
  is a different cut of the same machinery, not a new dependency.
- **The random-vector floor separates the arm's two roles (CS-6d asked;
  CS-6d answered, 2026-10-26).** With the embedding sides replaced by noise,
  the v0 pack and the full pack keep their within-document AUC (the floor is
  indistinguishable from the real arm on five of the six arms, and lands on
  the Stage-3.1b embedding-free arm's 0.6589), but the positives' hit@1
  collapses from 0.065–0.092 to 0.027–0.042 (the cosine's: 0.105–0.136). So
  the **lexical block is the ranking signal** and the **embedding geometry is
  the retrieval signal** — the learned arm and the cosine are complementary,
  not competing, and the deliverable's shape follows: retrieve with the
  cosine, rank with the learned arm, and measure the margin against the
  **control-role floor** (0.5410–0.5434 for a pack with noise in the
  embedding slots, 0.5569–0.5788 for the real one). The chunk profile's floor
  sits at or below chance (0.4733 / 0.4830 / 0.4706), so that block is a
  re-measurement of the embedding geometry and nothing else. **Open:** the
  floor is **one draw** (seed 42) — a different seed would move the numbers
  slightly; and it says nothing about a *different* feature family (the
  corpus-as-ceiling question stays open).
- **The threshold's criterion matters, and the principled one fails
  (CS-6e, 2026-10-26).** The per-document threshold chosen on the fit side
  transfers on 6 of 6 arms with the **`floor_gap`** criterion (the verified
  positives against the *unjudged control cells*) and not with the
  **`youden`** criterion (the verified positives against the *reviewer-verified
  hard negatives*): the hard negatives sit above the positives, so their
  threshold lands far too high (arctic2's holdout recall 0.02–0.06). **Open:**
  the NU document (the lowest base rate) does not take the fit side's
  threshold at all (the holdout's recall 0.02–0.29 at a precision of
  0.87–1.00) — the same non-transferability the Stage-2 semantic threshold
  showed. Is a *per-paragraph* or a *rank-based* operating point (no
  absolute cut at all) the shape that transfers? The track's rules forbid
  absolute cuts, so a threshold is already the compromise.
- **The decision models' input budgets are small and differ by an order of
  magnitude (CS-7a, 2026-10-26).** The server refuses a prompt longer than
  the model's context, and the measured limits are **512 tokens for laya**
  and **~2,500 for tev1** - the corpus's longest paragraph (2,255 characters)
  and longest question (1,560) do not fit laya's budget, so the call planner
  truncates the **state's paragraph** (never the question, never the
  instructions) and flags every truncated cell (laya: 8.0-8.1% of the cells;
  tev1: none). **Open:** does the truncation bias laya's arm? The truncated
  cells are the *longest* paragraphs, so laya's arm is measured on a
  systematically shorter view of the corpus's most content-rich paragraphs -
  and that is not separable from the model difference in this change set. A
  paired comparison (the same cell with and without the truncation) would
  settle it; the track's own rule (never a silent absolute cut) says a
  truncated input is a *different* input.
- **The decision models are not deterministic (CS-7a, 2026-10-26).** The same
  call with the same prompt returns a slightly different probability on a
  re-run (the within-document AUC moves in the third decimal), so the
  **cache** is what makes a re-run exactly reproducible, and two arms are
  never a *paired* comparison at the level of the model's own noise.
  **Open:** how large is that noise relative to the grid's differences? The
  probe's arms are 0.006-0.22 apart and the observed drift is ~0.001-0.003,
  but the drift is measured once (a fresh-cache re-run) and on one model.
- **A calibrated probability does NOT give the rule a fixed operating point
  (CS-7b, 2026-10-26).** The plan's premise was that a calibrated decision
  model would let the safe-negative rule take a **fixed 0.5 cut**. Measured:
  the fixed cut clears 97.9% of the unjudged grid but discards **60.5%** of
  the verified positives — the model is calibrated (the ECE 0.089-0.137) but
  its probabilities are *compressed* toward the low end (the positives' mean
  probability is 0.41), so it answers "no" for most true pairs. The
  **fit-side threshold** is the form that ships (the reach 83.1% at the harm
  0.1851). **Open:** is the compression a property of the model (it is
  trained to be cautious) or of the *prompt* (the "same topic" question is
  harder than "is this a match")? A prompt that asks the model for the
  *match* probability rather than the *same topic* probability would move the
  whole scale up — and the fixed cut would only become meaningful if the
  compression is the prompt's, not the model's.
- **The harm on the verified positives is finally label-driven (CS-7b,
  2026-10-26).** The **label-blind counterpart** — the same decision model
  with every cell's question replaced by another question of the same
  document — discards **41.5%** of the verified positives at the same reach
  where the real arm discards **18.5%** (and 80.4% against 60.5% at the
  fixed cut). CS-6e could not certify the harm because the *control-role
  floor* reproduced it; here the label-blind scorer is 2.2x worse on the same
  cells. So the deliverable's cost claim is a *quality* claim. **Open:** is
  the counterpart the right one? The alternatives are a permuted-paragraph
  control and no counterpart at all; the mismatched control was measured
  rather than chosen by taste, and it is the only one of the three that keeps
  the model's shape, its calibration and its cost while losing only the
  pairing.
- **The late chunking's gain is on the reviewed set, and the reviewed set is
  rank-selected (CS-7e, 2026-10-26).** The late arm's within-document AUC
  is 0.5944 against the early arm's 0.5084 — the largest segmentation
  effect the track has measured — but the reviewed set's positives were
  chosen by exp18's model, so a geometry change can pick up the
  **construction** and not the labels. The track's mitigation is the
  control side: the ordinary unjudged cells' score level and rank position
  do not move (the control's mean shift +0.0011, the rank-p50 -0.0065, the
  cells' hit@1 0.0037), and the **reviewed hard negatives' hit@1 falls**
  (0.0241 -> 0.0171) while the positives' rises — a signature the early
  chunking does not have (it lifts everything by +0.0413). **Open:** is
  there a *label-free* way to tell a representation gain from a
  construction gain on this benchmark? The decision models were that way for
  the "are the labels learnable" question (CS-7a); for the "does this
  geometry carry a topic signal" question the analogous test would be a
  **label-blind counterpart** of the late arm (the same geometry with the
  paragraph's topic destroyed — e.g. the chunk's tokens shuffled across
  the document), which would separate "the context helps" from "the
  reviewed set's structure helps". Not measured; it needs the reviewer's
  go-ahead (a new arm in the harness).
- **The UPR control-role floor's construction had to be read (CS-8d,
  2026-10-26).** The plan's wording ("the gold labels permuted, the non-gold
  labels keeping their presumed-negative role") is unambiguous on OMCT, where
  the judged pairs carry both classes; on UPR the gold cells are all positive,
  so permuting them among themselves is a no-op. The construction used is the
  structural analogue: the gold cells' positives are **relocated into the
  non-gold space** (the original gold cells become negatives, the class counts
  and the weights are preserved). It measures what it should — the training
  design's prior alone — and it lands at chance, which is the expected
  reading for a grid that is not rank-selected. **Open (for the reviewer):**
  is the relocation the intended reading, or should the floor instead keep
  the gold cells' positive labels and permute only the *label identities*
  (the mismatched-pair construction, which would teach the inverse signal)?
- **The decision model's ordering is the weak point, and the shortlist's
  recall is the ceiling (CS-9b / CS-12, 2026-10-27).** The decision model is
  a near-perfect **candidate generator** on UPR (the shortlist R@50 0.9987
  against the champion's 0.7423) and a **weak ranker**: its own top-1
  (0.3483) is below the champion's (0.3666), the champion's top-10 holds
  only 74.6% of the gold cells (so even a perfect reranker of that shortlist
  would reach 74.6%), and an **explicit** 20-way choice (CS-12's listwise
  stage) is *worse* than the implicit order of the 68 probabilities. **Open
  for PLAN's Phase-6 slot:** a reranker worth having has to be fed the
  *decision model's* candidates (or the union's — CS-14's recall@10 0.9120),
  and it has to be a **trained** ranker, not a zero-shot decision model. The
  slot decision is the reviewer's.
- **The UPR pack + the decision probability beats the champion at the
  decision view (CS-13, 2026-10-27) — at a stated cost.** The augmented
  pack's micro-F1 0.5965 against the champion's 0.5479, and the unseen
  bucket 0.5697 against 0.0000, with the fit side never seeing the held-out
  labels (0 positives of 3,087) and the label-blind counterpart's arm
  reaching only 0.4977. **Open:** the cost is the decision model in the loop
  (2 calls per text; 1,320 calls for the val split, ~3,352 for the fit side)
  — is that a price the pipeline should pay, and is the *feature* the right
  shape (the model's probability as a column) or should the pipeline ask the
  model directly? And the one **test shot** for an adopted config is still
  owed.
- **The union's decision view has a scale artefact (CS-14, 2026-10-27).** The
  per-row z-max is a *candidate generator's* number (the recall@10 0.9120
  against the decision model's 0.8845 and the champion's 0.7126, with the
  blind control at 0.6947) and not a scorer's: the union's pooled micro-F1
  (0.5843) is the track's best UPR number, but its unseen bucket (0.2540) is
  half the decision model's (0.5267) because the champion's z-scores are
  more spread out and the max is biased toward its scale. **Open:** a union of
  two *calibrated* scores (not two z-scored ones) would keep the recall and
  fix the decision view — is it worth measuring, or is the union's number
  enough as it stands?
- **The catalog's per-label structure is not measured (CS-11, 2026-10-27).**
  The description channel's pooled per-feature AUC (0.7067 against the
  name's 0.7566) says the name is the stronger channel *pooled*, and the
  pack's decision view says the two are complementary — but the scratch's
  claim ("the description beats the name where the name is weak — the generic
  labels") is a **per-label** claim, and the diagnostics are pooled per
  feature. **Open:** a per-label, per-feature breakdown (the description
  coverage against the name coverage, per label) would settle it.
- **The reranker's ordering against the pack's: answered (CS-15/CS-16,
  2026-10-27).** The two forms of the same score answer two different
  questions, and the answers differ. As an **ordering** of the decision
  model's candidates, every zero-shot reranker is *worse* than the
  decision model's own ordering (the best arm: a paired +0.0091 with the
  interval including zero — a null), while the *same* score as a **pack
  column** is the best UPR number the track has measured (the listwise
  model's full-grid feature: 0.6117 against the control's 0.5965, a paired
  +0.0152 [+0.0015, +0.0280], the label-blind counterpart at null). So the
  reranker's value is **not** in the order it produces but in the *signal*
  it adds to a learner that already has the decision model's probability —
  which is the CS-13 precedent one level up (the decision probability: +0.0023
  as a scorer, +0.1147 as a feature). **Open (still):** the "fusion rule"
  this suggests — a *fitted* weight between the pack's order and the
  reranker's order — is **not measured**, and it would be a new thing to fit
  (with the fit side's own discipline), so it needs its own pre-registration
  and is not in CS-15/CS-16.
- **The fine-tuning's training target (CS-17, 2026-10-27).** The track holds
  the three ingredients (the OMCT hard negatives, the UPR safe-negative hard
  negatives, the decision model's 99.87%-recall candidate set as the positive
  mining source), but the **target** is undecided: a pairwise cross-encoder
  (the two cross-encoders' own objective) against a listwise one (the
  listwise model's shape, and the CS-12 lesson that an explicit choice is
  *worse* than an implicit order). **Open:** which target, and does the
  mined positive set (the candidate set) teach the ordering or only the
  presence?


## 6. Milestones

- T1: stages 0-1 (diagnosis + structure) → what matters on OMCT decided.
- T2: stage 2 (segmentation ladder) → the chunking verdict. *(reached
  2026-10-26, CS-4: chunk the paragraph, keep the peak, never chunk the
  question — and the pooled AUC is the wrong headline on this benchmark.)*
- T3: stage 3 (learned scorer) → the reranker-slot evidence for PLAN Phase 6.
- T3.5: stage 3.5 (the decision models) → is the reviewed set learnable at
  all, and can a calibrated independent decider carry the deliverable?
  *(proposed 2026-10-26; the reviewer's call on tev1 + laya.)*
- T4: stage 4-5 as reviewer-approved extensions.

## 7. First execution change sets

The executed change sets moved to `docs/text_processing_plan_archive.md` §7 on
2026-10-29; their pre-registrations and their verdicts stay there **verbatim**. One line
per change set, so a reader can always tell *moved* from *lost*:

- **CS-0 … CS-7e** (the skeleton, the OMCT command, Stages 0–3.5) → executed
  2026-10-23…2026-10-26 → the archive's §7 and the track log's §Stage * sections.
- **CS-8a … CS-9b** (the UPR port) → executed 2026-10-26…2026-10-27 → the archive's §7.1
  and the track log's §CS-8a … §CS-14.
- **CS-10** (the LLM stage on UPR) → **not executed** — it needs the reviewer's explicit
  OK, which was never given → the archive's §7.1.
- **§7.1.1** (the reviewer's decisions of 2026-10-27) → executed → the archive's §7.1.1.
- **CS-11 … CS-14** (the new change sets P1–P4) and **CS-15, CS-16** (the zero-shot
  reranker round) → executed 2026-10-27 → the archive's §7.1.2 and the track log's §CS-11 …
  §CS-16.
- **§7.1.3** (the reviewer's decisions of 2026-10-27, the reranker round) → executed → the
  archive's §7.1.3.
- **CS-18** (the hit map: which words carry the signal) → executed 2026-10-29 → the
  archive's §7.1.4 and the track log's §CS-18.
- **CS-19** (the hit map's two cheap modes: the text-side word map and the
  surrogate attribution) → executed 2026-10-29 → the archive's §7.1.5 and the
  track log's §CS-19.
- **CS-20** (the hit map's terminal viewer: the interactive map) → executed
  2026-10-29 → the archive's §7.1.6 and the track log's §CS-20.
- **CS-23** (the corpus map: the train split's own word↔label association) →
  executed 2026-11-01 → the archive's §7.1 (the corpus map) and the track log's
  §CS-23.
- **CS-24** (the surrogate's full coefficient vector) → executed 2026-11-01 → the
  archive's §7.1 (the surrogate's full vector) and the track log's §CS-24.
- **CS-25** (a clearer screen, and the colour) → executed 2026-11-01 → the
  archive's §7.1 (the clearer screen) and the track log's §CS-25.
- **CS-27** (the gold labels are always visible, and the in-text row keeps its
  shade) → executed 2026-11-01 → the archive's §7.1 (the gold labels are always
  visible) and the track log's §CS-27.
- **CS-28** (the viewer shows every gold label) → executed 2026-11-01 → the
  archive's §7.1 (the viewer shows every gold label) and the track log's
  §CS-28.

The change sets still open:

- [ ] **CS-17 — the *fine-tuned* cross-encoder (PLAN Phase 6's second step;
  acknowledged, NOT scheduled here).** The zero-shot measurement of CS-15 /
  CS-16 is the cheap first step; the destination is a **trained** reranker,
  and the track now holds all three of its ingredients: the characterized OMCT
  benchmark (the 996 verified hard negatives), the UPR hard negatives from the
  safe-negative rule (CS-9b), and the decision model's 99.87%-recall candidate
  set as the positive mining source. What is missing is a **reviewer's OK**
  (a training run is a new artifact class: a fine-tuning script, a training
  split of its own, and the label-holdout discipline applied to the training
  data) and a decision on the training target (pairwise vs. listwise). The
  checkbox stays open and unticked: the round that measures it is not this
  one.

### 7.1 The hit map's next round (CS-20 … CS-22)

**The stop rule (the reviewer's, 2026-10-29 — it binds every session).** One
change set at a time, and a **stop between them**: when a change set is done
(the pre-registration written, the code and its tests in, the run's numbers in
an artifact, the log section written, the box ticked with the numbers, the
pre-registration moved to the archive, the commit made and the gate green), the
work **stops and reports** — what was built, the numbers with the falsification
conditions read out, what the reviewer has to decide, what the next change set
would be. **The next change set starts only on the reviewer's word.** The same
applies inside a change set: if it turns out to need a decision, it stops and
asks.

**The order:** CS-20 (the viewer) → CS-21 (the OMCT map) → CS-22 (mode E at
corpus scale). The viewer comes first because it is the cheapest thing that
teaches the most, and because it tells us what mode E should even measure.

- [ ] **CS-21 — the hit map for the **OMCT** data (the auxiliary pair-relevance
  data).** The same map on the other dataset: which words carry the signal there.

  **The mechanism, and why it is not just CS-20 with another dataset.** The
  OMCT pack already has **both** lexical directions (`lex_question_coverage` and
  `lex_paragraph_coverage`), so the two cheap modes are the same code with the
  **question** playing the label's role: per pair, the map is "which words of the
  paragraph matched which words of the question" — the **word pairs** view. The
  causal arm is affordable because the OMCT decision arms use the **paragraph
  shape**, so a call is per *paragraph*, not per pair: the reviewed set is
  **1,520 pairs over only 225 paragraphs** (3 documents; the same-document grid
  is 44,467 cells; the unjudged control is 4,000 cells), so the whole clause
  ablation is 225 paragraphs × ~3 clauses × 2 calls × 2 (the label-blind
  counterpart) ≈ **2,700 calls ≈ 25 minutes**.

  **The discipline.** The two negative subsets are **never pooled**: the fit side
  is the reviewed set (A) and the unjudged control (B) stays a control. Every
  OMCT number is an OMCT number (never a UPR metric), the run ids carry the
  `omct-` prefix, and the log section names the dataset.

  **What it is not.** The **surrogate** (mode C) does not transfer: its unit is
  the "topic" and its fit needs many cells per topic, while on OMCT the topics
  are *questions*, each appearing in only a handful of paragraphs — a
  per-question fit is hopeless and a global fit (the paragraph's words → the
  model's probability, with a per-question intercept) measures the model's
  prior, not the pair. A **word-pair** surrogate (the features = the
  paragraph-word × question-word matches) is the workable version, and it would
  be its own change set.

- [ ] **CS-22 — mode E at corpus scale (the word-level causal ablation).** The
  leave-one-**word**-out ablation: remove one word from a text, ask again, read
  the drop — the same instrument as the clause-level ablation, one level finer.

  **The mechanism.** Over a **pre-registered** subsample (the plan's candidate:
  150 texts × ~20 words × 2 calls × 2 for the label-blind counterpart ≈
  **12,000 calls ≈ 2 hours**), with the paired reads and the blind counterpart
  exactly as CS-18 did them, and the resampling unit the text.

  **The precondition.** The viewer's interactive what-if should show *what* to
  measure (the topical clause's words? the boilerplate's? the topic's own name
  words?) **before** the pre-registration is written — a pre-registration
  written before that exploration would be a guess. The open questions for that
  pre-registration: the removal unit (the word, or the word **pair** — the
  plan's own phrase "words or word pairs"), the subsample's size and how it is
  drawn, and the falsification conditions (the analogue of CS-18's "the
  label-linked unit's drop is larger" and "the boilerplate's drop is not
  smaller").

  **What it is not.** It does not ship in the same change set as the viewer: the
  viewer's what-if is the *interactive* mode E (one text at a time), and this
  is the *corpus-level* one.

### 7.2 The viewer's round two (CS-23 … CS-26)

**What prompted it (2026-11-01).** The review session after CS-20 asked what the
viewer's numbers *are*, and three things came out of it:

1. **the corpus's own word↔label association was never built** — the map
   answers a *neighbouring* question (which words carry the *model's* signal),
   never the plain statistics of the corpus ("check all the words for all the
   texts in the train and find a correlation between the labels and words");
2. **the surrogate's column was almost empty** — the artifact stores 15
   positive + 15 negative words per topic out of 81,804 coefficients, so
   nearly every word of a text reads "—";
3. **the screen was hard to read** — two instruments shared one table with one
   footnote, the mode letters (A/B/C/D/E) and the feature jargon made the
   blocks unreadable, and there was not enough colour to see the structure.

**The order:** CS-23 (the corpus map) → CS-24 (the surrogate's full vector) →
CS-25 (the clearer screen and the colour) → CS-26 (the clause ablation in the
viewer). One change set at a time, with a **stop and a report between them**
(the stop rule of §7.1 — the next change set starts only on the reviewer's
word). CS-21 and CS-22 stay untouched and unscheduled.

- **CS-23** (the corpus map: the train split's own word↔label association) →
  executed 2026-11-01 → the archive's §7.1 (the corpus map) and the track log's
  §CS-23.
- **CS-24** (the surrogate's full coefficient vector) → executed 2026-11-01 → the
  archive's §7.1 (the surrogate's full vector) and the track log's §CS-24.
- **CS-25** (a clearer screen, and the colour) → executed 2026-11-01 → the
  archive's §7.1 (the clearer screen) and the track log's §CS-25.
- [ ] **CS-26 — the clause ablation in the viewer (mode D).** The clause-level
  ablation (CS-18) is implemented and measured but invisible: the viewer shows
  the interactive word-level version instead.

  **The mechanism.** The viewer reads `results/upr-hitmap-clause/hitmap.json`
  (a third artifact; **0 model calls**) and prints the text's clauses, which
  one is the **label-linked** one, and each clause's **drop** — the gold cells'
  probability when that clause is deleted — with the label-blind counterpart
  beside it. For the 312 single-clause val texts: "not ablatable", never
  silently skipped.

  **The cost.** 0 model calls (the artifact exists).

  **The exit criteria.** The viewer's numbers match the artifact's per-text
  rows; the "not ablatable" case is covered by a test; the gate green.

### 7.3 The viewer's round three (CS-27 … CS-28)

**What prompted it (2026-11-01).** The review session after CS-25 asked what
the viewer's numbers **are** (how the contributions, the surrogate's
coefficients, the corpus map's log-odds and the side-by-side's ranks are
computed), and three things came out of it:

1. **a gold label was visible only if it ranked in the top five** — the
   side-by-side table did list the gold labels, but the two **ranking** blocks
   showed only the top-k, so a gold label below it was missing from the
   ranking, and its topic-side rank was not shown at all;
2. **the corpus map's row for a word that is in the text lost its ramp shade**
   — it was printed bold, so the log-odds' size was unreadable on that row;
3. **two blocks answered their question for the focus topic only** — the topic
   side of the feature ("THE TOPIC'S OWN WORDS": how much of the topic's name is
   in the text?) and the corpus map ("THE CORPUS'S OWN WORDS FOR THIS TOPIC":
   which words go with this label in the training corpus?) did it for the
   focus topic alone, so the truth for the other gold labels of the text was
   readable only as a number in the ranking, never as the words that carry it.

- **CS-27** (the gold labels are always visible, and the in-text row keeps its shade) →
  executed 2026-11-01 → the archive's §7.1 (the gold labels are always visible) and the
  track log's §CS-27.
- **CS-28** (the viewer shows every gold label) → executed 2026-11-01 → the
  archive's §7.1 (the viewer shows every gold label) and the track log's
  §CS-28.

### 7.4 The lexical methods round (CS-29 … CS-35)

**What prompted it (2026-11-01).** A literature sweep on *lexical* label inference
— label→word lexicon mining, label embeddings built from corpus-mined words, the
state of the art on "infer the label from the text only", and the measurement of
word-level attribution — plus the reviewer's four ideas from the same session:
**try the corpus map as a classifier**; **compose the corpus map with the learned
pack**; **mine a label→word lexicon from a corpus**; and **use part-of-speech
tagging to see what kind of words affect the label in which way**.

**What the sweep established, in one paragraph.** The corpus map (CS-23) *already
is* the label→word lexicon — 18,707 (label, word) pairs over 3,049 distinct
content words, with the counts behind every one — so there is nothing to mine
again; what the literature adds is a **better estimator** for it and two **new
uses** of it (as a label-side channel, and as the only lexical route to a label
that has no train examples). And the reference point for the whole round is
external: the **Danish Institute for Human Rights** runs a purely *lexical*
classifier (their "TC8") on UN human-rights recommendations in this very genre,
with a per-category dictionary of weighted terms, and publishes per-category F1
(0.87–0.99 on most categories; the Danish Institute's `Performance_HuRi_SDG.pdf`).
Those are *their* categories on *their* corpus with human-annotated ground truth,
so they are **not** comparable to our numbers — but they establish that a
lexical system on this genre is a serious thing to beat.

**The order:** CS-29 (the lexical floor) → CS-30 (the map as a classifier) →
CS-31 (the map's estimator) → CS-32 (the map in the pack) → CS-33 (the word-class
readout) → CS-34 (real POS) → CS-35 (the lexicon as a label-side channel).

**The reviewer's decisions of 2026-11-01 (they bind the change sets below).**

1. **The batch is authorised.** The reviewer: *"let's put the things you think are
   useful in the PLAN and after the implementations I'll try to understand what's
   going on."* So the change sets run in order, each one complete and committed on
   its own (its own pre-registration, artifact, log section, ticked box, archive
   move and green gate), and the report comes at the end rather than between each.
   **The stop rule's other half still holds: a change set that turns out to need
   a decision stops and asks.** Nothing about the *order* changes without the
   reviewer's word.
2. **CS-34 and CS-35 are gated** on the two approvals named below (a new
   dependency and a model download; a schema change). They are pre-registered and
   costed so that the approval is one word each.
3. **Every number this round produces is a UPR *val* number** and never a UPR
   *test* number. The test split stays untouched.
4. **The corpus map's existing block is not renamed or reshaped**; the new measures
   are added *beside* the current ones (CS-31), so every number CS-23 reported
   stays reproducible from its artifact.

**The directory.** The new lexical machinery goes to
**`src/cluster_rag/text_processors/lexical/`** — a new sibling of `chunkers/`,
`features/`, `combining/`, `decision/` and `rerank/`. It is **not** a
repo-root directory: everything importable lives under `src/cluster_rag/`, and the
track's **pure-core guard test** only scans `text_processors/`, so a root-level
directory would be unguarded and the "numpy only, scikit-learn only in the
combiner" rule unenforceable. Two things are deliberately **not** moved:
`text_processors/features/corpus_map.py` stays where it is (four modules import it;
a move is a refactor over more than three modules, which AGENTS.md §3 says to
explain first), and the POS tagger — a *model client*, since it downloads a model
— goes to `src/cluster_rag/pos/` next to `embed/`, `decision/` and `rerank/`, with
a `base.py` port and an `nltk_tagger.py` backend; the track's package imports the
port only.

- **CS-29** (the lexical floor: TF-IDF + one-vs-rest) → executed 2026-11-01 →
  the archive's §7.4 (the lexical floor) and the track log's §CS-29. The
  pooled micro-F1 is **0.3845** against the champion's 0.5479 (neither
  pre-registered falsification fires); the tail collapses (0.0301 against
  0.5287) and the shortlist is on par with the champion's (R@50 0.7205).
- [ ] **CS-30 — the corpus map as a classifier.** A new predictor kind
  (`corpus_map`): every (text, label) cell is scored from the corpus map's own
  weights, and is measured by the same harness as CS-29. The scorer lives in
  `text_processors/lexical/corpus_map_score.py`; the adapter in
  `src/cluster_rag/predictors/corpus_map.py`.

  **The scoring function**, with its alternatives pre-registered (all four are
  cheap, so all four are measured, and the choice is made on the **fit side**,
  never on val): `sum_logodds` (the sum of the map's log-odds over the text's
  words, a word missing from the label's table contributing 0), `sum_positive`
  (the same over the words whose log-odds is positive — the "evidence **for**"
  score), `max_logodds` (the single best word — the "is the label's own word
  here" score) and `support_count` (how many of the label's table words the text
  contains, unweighted — the measure-free control).

  **The discipline.** The map is built from the **train** split only; the
  evaluated split is scored with it, so no val text and no val label enters the
  map. The operating point is `top_k` k=2, so nothing is fitted on the evaluated
  split at all.

  **The falsification, registered in advance.** The **unseen bucket is 0.0000 by
  construction** — the map's table for the eight held-out labels is empty (CS-23
  measures this), so the map cannot see a label it has no examples of. The
  interesting read is the **seen-bucket** number and the map's **own measure
  ablation**: does the log-odds weighting beat `support_count`? If a *count* of
  the label's table words is as good as the log-odds weighting, then the map's
  careful measure work (CS-23) buys nothing predictive. And if the pooled
  micro-F1 comes **within 0.02 of the champion's 0.5479**, the
  retrieval-plus-decision pipeline is buying little over a corpus statistic — a
  finding to report, not to bury.

  **The cost.** Half a day to a day. **No new dependency.**

  **The pre-registration (2026-11-01, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **The predictor.** A new registry entry `corpus_map`, behind the same
    `Predictor` port as every other predictor: the `evaluate` command builds
    it, scores the split and measures it with the **existing** metric code and
    the **existing** decision rules — the same instrument that measured CS-29
    and the champion.
  - **Where the code lives.** The **scorer** is pure and lives in the track's
    package, `src/cluster_rag/text_processors/lexical/corpus_map_score.py`
    (numpy + the standard library only); the **adapter** lives in
    `src/cluster_rag/predictors/corpus_map.py`, next to CS-29's
    (`predictors/` is where a predictor goes; the plan's §2 keeps scikit-learn
    out of the track's package, and this scorer needs none).
  - **The map.** Built by the **existing** CS-23 function
    (`text_processors/features/corpus_map.py: build_corpus_map`) from the
    **train** split only, with the track's own tokenizer
    (`content_tokens` — the same unit the map's tables are keyed by). No val
    text, no val label and no test text enters the map.
  - **The four scoring functions** (the plan's names, the literal definitions):
    - `sum_logodds` — the sum of the map's log-odds over the text's **distinct**
      content words; a word missing from the label's table contributes 0;
    - `sum_positive` — the same sum, over the words whose log-odds is
      **positive** (the "evidence **for**" score);
    - `max_logodds` — the **single best** word: the maximum log-odds over the
      text's words that are in the label's table, and **0.0** when the text has
      no word in the table ("no evidence");
    - `support_count` — how many of the label's table words the text contains,
      unweighted (the **measure-free control**).
    The text's words are **distinct**: the map's unit is the word's presence,
    and `support_count`'s own wording ("how many of the label's table words the
    text contains") says so.
  - **The empty tables.** A label with **no train example** has an empty table
    (CS-23 measures it: 8 of the 68 labels), so there is no evidence at all
    about it: those labels are ranked **last in every row** and can never
    enter a top-k. That is what makes the unseen bucket **0.0000 by
    construction**. *Amended before the first call (2026-11-01):* the floor
    was first written as `-inf`, but the harness **rejects non-finite
    scores** (`eval/multilabel.py`: "scores must be finite (NaN/inf would
    corrupt every ranking"), so the floor is **finite and per row**: the
    smallest score among the labels that *do* have a table, minus one. The
    floor touches the empty tables alone — the seen labels' scores and their
    order are untouched — and it is strictly below every seen label's score
    whatever the arm's scale, which the two signed arms need (they are
    unbounded below). *One reading note, registered with it:* the unseen
    bucket's **decision** view (the F1 and the coverage) and the **global**
    shortlist are the honest numbers (0.0000); the bucket's **ranking**
    metrics *inside* the restricted view are an artefact of the tie (the
    harness ranks equal scores by the column order, so the eight tied columns
    are ordered alphabetically) and are not read as evidence. The same
    artefact is in CS-29's run.
  - **The choice is made on the fit side, and the fit side is cross-fit.** The
    plan says the choice between the four arms is made on the fit side, never
    on val. The **fit-side number is the out-of-fold train micro-F1 at the same
    operating point** (`top_k` k=2), with **5 folds** over the train split
    (seeded, deterministic): each fold's texts are scored by a map built from
    the **other** folds. The reason is the map's own construction: the map is
    built from the train split, so scoring the train split with it is
    optimistically biased — and most of all for the arms that use the log-odds
    the map itself computed (a hapax word's log-odds is huge, and it
    contributes its full size to a sum and only 1 to a count). A selection made
    on that number would prejudge the very ablation this change set exists to
    run ("does the log-odds weighting beat `support_count`?"). This is the
    track's **own** guard, registered in CS-32 ("the fit side's map must be
    cross-fit") and in CS-9b (the fit side's threshold), applied one change set
    earlier. The **in-fold** number (the train split scored by the full map) is
    reported **beside** the out-of-fold one, so the bias is a measurement and
    not an assertion.
  - **The four runs.** One per arm, all on UPR **val** at `top_k` k=2:
    `upr-eval-corpus-map-sum-logodds-val`,
    `upr-eval-corpus-map-sum-positive-val`,
    `upr-eval-corpus-map-max-logodds-val` and
    `upr-eval-corpus-map-support-count-val`. The sweep file is
    `experiments/corpus_map_sweep.py` (the sweep label `corpus_map`, so the
    table's section is `## UPR — corpus_map`); `evaluate` is a main-pipeline
    command, so the rows land in the **main** UPR table
    (`docs/experiments_table_upr.md`) and the narrative in the track log
    (`docs/experiments_text_processing.md`, the section
    `## CS-30 — the corpus map as a classifier`).
  - **What is reported.** All four arms' val numbers (the **measure ablation**)
    and all four fit-side numbers (the **selection**); the headline — "the
    corpus map as a classifier" — is the arm the fit side selects, and the
    falsification is read on it. The arm with the best *val* number is reported
    too, clearly marked as **not** the selection (choosing it would be choosing
    on val).
  - **The falsification, registered in advance.** (i) The **unseen bucket is
    0.0000 by construction** — the empty tables score the floor. (ii) The
    **measure ablation**: if a plain **count** of the label's table words
    (`support_count`) is as good as the log-odds weighting, then the map's
    careful measure work (CS-23) buys nothing predictive — a null, reported as
    a null. (iii) If the pooled micro-F1 comes **within 0.02 of the champion's
    0.5479** (that is, at or above **0.5279**), the retrieval-plus-decision
    pipeline is buying little over a corpus statistic — a finding to report,
    not to bury. (iv) The **seen-bucket** read: the pooled number mixes the
    eight labels the map cannot see into its denominator, so the three seen
    buckets (head, torso, tail) are pooled **by their own counts** and read
    beside it.
  - **The exit criteria.** The four artifacts hold the numbers; the four table
    rows are auto-appended; the log section reads the numbers out with the
    falsification conditions; the gate green (ruff over `src`/`tests`/
    `experiments`, the full pytest suite, the two verify tools, and both final
    sweeps reproducing their artifacts field for field).

- [ ] **CS-31 — the corpus map's estimator, and the lexicon as a first-class
  artifact.** Two parts, one change set, because the second is measured with the
  first.

  1. **The estimator.** Beside the current log-odds, the map gains the
     **informative-prior log-odds** (Monroe, Colaresi & Quinn 2008: the word's
     global frequency as the pseudo-count) and its **z-score** (δ over the
     standard error); a **significance** per pair, so "this word goes with this
     label" can be stated with a p-value instead of only a size (Fisher's exact
     or the G² statistic on the same 2×2 table — the same table, so nearly
     free); and the **conditional smoothing** variant (0.5 only where a cell is
     actually zero), with the share of pairs whose *sign* flips under it reported.
     The reason to change it: our fixed 0.5 on all four cells is the variant the
     statistics literature criticises (Jackson et al., *JRSS-M* 2021: adding 0.5
     to *every* cell of *every* table performs **worse** than adding it only to
     the tables that actually have a zero cell; Agresti recommends a much smaller
     constant for sparse data).
  2. **The lexicon artifact.** The map's words, per label, with the new measures
     and the counts — reusable by CS-32, CS-33, CS-35 and the viewer, so it is
     written once. It goes into `text_processors/lexical/lexicon.py` (the
     estimator, pure) and `text_processors/features/corpus_map.py` (the artifact
     block, so the existing CS-23 shape is preserved and the viewer keeps
     reading one block).

  **The reference implementations** are used as *test oracles*, not as runtime
  dependencies (the track's package stays numpy-only): `logodds` on PyPI (MIT, no
  dependencies, and it takes the informative prior directly), `tidylo::bind_log_odds()`
  in R (the same method with an empirical-Bayes prior) and
  `quanteda.textstats::textstat_keyness()` (the chi² / Fisher / G² / PMI family,
  with p-values).

  **The falsification, registered in advance.** The head word changes for a
  measurable number of labels: CS-23 measured the **lift**'s head as a hapax for
  **55 of the 60** labelled tables and the log-odds's head as a hapax for only
  **4**, so the z-score's head should be a hapax for **few** labels — the prior
  is supposed to demote the single-document words. **If the z-scored head word is
  unchanged for ≥ 55 of the 60 labels, the prior does nothing here and the metric
  upgrade is a null**; report it as a null. And CS-30's classifier is re-measured
  with the z-scored weights: if the pooled micro-F1 does not move, the better
  measure is a better *description*, not a better *model*.

  **The cost.** About a day. **No new dependency.**

- [ ] **CS-32 — the corpus map in the learned pack.** The map's per-cell score
  becomes a feature of the UPR pack, so the learned combiner can use it beside the
  embedding and lexical features: a new group in `UPR_FEATURE_GROUP_PREFIXES`
  (e.g. `cmap_`), gated like the catalog groups (the group is only selectable when
  the map is available, and a run without it records the gap in its artifact).

  **The guard, and it is not optional.** The pack is fitted on **train**; the map
  is built from **train**. On the fit side the map has already seen the cells it
  scores, so the feature is optimistically biased exactly where the combiner
  learns its weight. The fit side's map must be **cross-fit** — K folds over
  train, each fold's cells scored against a map built from the other folds — the
  same shape as the fit-side threshold's cross-fit (CS-9b). The artifact records
  the folds.

  **The control, which is the point of the change set.** Three floors, as the
  track already does: the **control** (the same pack with the group removed); the
  **random-vector floor** (unchanged); and the **label-blind control** (the same
  pack, but the map is built with the labels **permuted**). If the label-blind
  map buys the same Δ as the real one, the "gain" is the map's construction, not
  the corpus's word↔label association — the exact failure mode the track's own
  control-role floor was built to catch.

  **The measurement and its falsification.** The paired Δ micro-F1 against the
  control, with its CI, over the texts (the resampling unit), plus the per-bucket
  breakdown. The unseen bucket is expected flat: the map is empty for the eight
  held-out labels, so this feature cannot help there. The Δ's CI must exclude
  zero **and** the label-blind control's Δ must be materially smaller. If the Δ is
  inside the noise, the corpus's word statistics add nothing that the
  label-name overlap features (`lex_label_coverage`, `lex_text_coverage`) do not
  already carry.

  **The cost.** About a day and a half (the pack's group machinery, the
  cross-fit, the controls, the report block).

- [ ] **CS-33 — the word-class readout: what kind of words affect the label, in
  which way.** The reviewer's part-of-speech question, answered first **without a
  new dependency**. The map's own structure gives every (label, word) pair three
  word classes for free: **generic** (the word sits in many labels' tables —
  CS-23: "rights" in 60, "include" in 59), **distinctive** (one or two tables) and
  **name word** (the word is in the label's own name). Crossed with the **sign**
  (the map's log-odds: up = with the label, down = against), that is the
  reviewer's exact question in the map's own vocabulary: *which kind of word
  pushes the label up, and which kind pushes it down* — per label, and for the
  **model's** own habits too (the surrogate's per-topic coefficients, split the
  same way).

  **And the payoff, not only the readout.** If the distinctive words are
  systematically one class and the generic ones another, then a **class-restricted
  map** should beat the full map as a classifier — measured by re-running CS-30's
  arms with the restricted map. That is a number, not a story.

  **The falsification, registered in advance, and it decides CS-34.** If the word
  class does **not** separate the label-carrying words from the register words —
  i.e. the generic and the distinctive words have the same class profile — then
  the readout is a null, no class-restricted map is worth building, and **CS-34
  (the POS dependency) is cancelled**.

  **Where.** `text_processors/lexical/word_class.py` (the pure classification) and
  a new block in the map's artifact; the viewer gets one more block (the viewer's
  own pattern: a block names the question it answers).

  **The cost.** About a day.

- [ ] **CS-34 — real part-of-speech tags.** *Gated on the reviewer's OK: a new
  dependency and a model download (AGENTS.md §3).* Tag the corpus with a real POS
  tagger and redo CS-33's readout with the grammatical class, plus the
  class-restricted map's number.

  **The dependency, stated plainly.** `nltk` (MIT) plus its
  `averaged_perceptron_tagger_eng` model; the install and the tagger's run over
  our 3,000-text corpus (0.64 s) were checked, and 4,089 distinct words come back
  of which **16.3 %** take more than one tag across the corpus — so the readout
  must report the per-word tag *distribution*, not a single tag, or it will lie.
  The tagger is a **model client** (it downloads a model), so it lives at the
  edge: `src/cluster_rag/pos/` with a `base.py` port and an `nltk_tagger.py`
  backend; the track's package imports the port only. The model download needs a
  cache and a recorded digest, like the embedder's and the decision model's. One
  line in AGENTS.md's pure-core rule (the `cluster_rag.pos.base` port).

  **The falsification.** The same as CS-33's, now with the grammatical class: if
  the class does not separate the label-carrying words from the register words,
  the whole POS line is a null and the dependency is not worth keeping.

  **The cost.** About a day, plus the dependency.

- [ ] **CS-35 — the label→word lexicon as a label-side channel.** *Gated on the
  reviewer's OK: a schema change (AGENTS.md §3).* The lexicon's biggest use: a
  **label-side representation built from the corpus-mined words** rather than from
  the name and the description — the PIFA idea (X-Transformer, KDD 2020: the
  label's embedding is the TF-IDF centroid of its positive instances) — and the
  **label-expansion route for the eight labels with no examples** (the label's
  *name* matched against the corpus, with no labels involved).

  **Why it is gated.** It touches the label-embedding path and the label
  catalog's schema, and it has a **label-holdout** consequence to settle before
  any code: a label's mined words come from its **train** examples, so the channel
  is available for the sixty labelled labels and **empty for the eight held-out
  ones**, exactly like the catalog's channels are today. The expansion route is
  the only lexical path to the eight — and it must not use val text.

  **The falsification.** The mined-words channel must beat the name+description
  channel (CS-11's Δ was **+0.0258 [+0.0091, +0.0433]** micro-F1) on the seen
  buckets, with the same paired-CI discipline. If it does not, the lexicon's
  value stays confined to the readout (CS-31) and the classifier (CS-30) and it
  should **not** become a catalog channel.

  **The cost.** About two days, plus the schema decision.

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

**What the track has measured about that rule (CS-6e, 2026-10-26).** The
rule ships as the **margin half only** (no human labelling in this track):
a cell of the same-document grid is a *provisionally* safe negative when it
sits at least the chosen margin below **its own paragraph's best candidate**
— a margin rule, never an absolute cut. The best-measured shape is the
**two-part** rule (the cosine's margin *and* the learned arm's margin, i.e.
CS-6d's "retrieve with the cosine, rank with the learned arm"): it reaches
**50–60% of the 42,947-cell unjudged grid** at a measured cost of **11–19% of
the 524 verified positives**. What the track **cannot** claim is a precision:
the unjudged cells' true positive rate is unmeasured, and the only
ground-truth proxy — the harm on the verified positives — is **reproduced by
the control-role floor** (a model fitted with the judged labels permuted),
because the reviewed set is rank-selected and its positives sit near the top
of their paragraph's candidate list whatever model scores them. So the
recipe that leaves the track is: *a stated reach, a stated cost, and no
precision claim.* PLAN Phase 6 and Phase 9 may use it to **provisionally**
label unjudged cells as negatives — and must treat the labels as
provisional, exactly as the track treats the control cells.

**What would change that (Stage 3.5, proposed 2026-10-26).** Two things
would turn the deliverable from a *reach* claim into a *quality* claim, and
both are measured by the decision-model stage: (1) a decision model that is
**calibrated on this corpus** takes a **fixed** operating point, so the
rule's threshold stops being a fitted, non-transferable parameter; (2) a
decision model that is **independent of exp18's ranking** (which is how the
reviewed set was selected) makes the harm on the verified positives a
*label-driven* number instead of a measurement of the benchmark's
construction — which is exactly what the control-role floor comparison
refused to certify in CS-6e. If neither holds, the deliverable's claim
stands as it is, and the track says so explicitly.

**What the probe already settled (CS-7a, 2026-10-26).** Both preconditions
**hold** for `tev1` (a Qwen3.5-4B fine-tune by Together AI):

1. **It is calibrated on this corpus.** Its expected calibration error on the
   reviewed set is 0.0891-0.1365 pooled and 0.2009-0.2443 per-document macro
   (the Brier 0.1881-0.2064) - better calibrated than the track's own learned
   arm (0.175-0.189 pooled / 0.327-0.349 macro), so a **fixed** cut is
   meaningful.
2. **It is independent of exp18's ranking.** It is frozen and zero-shot: it
   never saw the corpus, so its scores cannot be a measurement of the
   reviewed set's construction. On the reviewed set it reaches a
   within-document AUC of **0.7767** (the best arm) where every embedder sits
   at 0.5020-0.5632, and its **synthetic competence control** passes
   decisively (the AUC 0.944-0.9996), so the reviewed set's labels *are*
   learnable.

So the deliverable's *quality* claim is no longer blocked by the scorer, and
the plan's Stage-0.2 reading ("the benchmark's construction dominates its
signal") is too strong: the construction makes the **embedding geometry**
useless as a ranker on this set, not the labels themselves. What is **not**
settled is the rule itself: the harm of a tev1-driven rule on the verified
positives, measured against the control-role floor, is **CS-7b's** question,
and the unjudged cells' true positive rate stays unmeasured (no human
labelling in this track).

**What the rule now measures (CS-7b, 2026-10-26).** With `tev1` behind it, the
rule's best form — the **fit-side threshold** on the decision model's
probability — reaches **83.1% of the 42,947-cell unjudged grid** at a harm
of **18.5% of the 524 verified positives** (the old best form, the learned
arm's two-part rule, reached 50–60% at 13.6–19.3%). And the harm is
**label-driven**: the label-blind counterpart (the same model, the wrong
pairings) discards 41.5% of the positives at the same reach. So the
deliverable now ships with a **reach claim** (83% of the unjudged space), a
**cost claim** (18.5% of the known positives) **and** the evidence that the
cost is a quality signal and not an artefact of the reviewed set's
construction. The *precision* on the unjudged cells is still unmeasured (no
human labelling), and the rule still ships as the **margin half only** — the
fixed 0.5 cut is measured and rejected (the reach 97.9% at the harm 60.5%).

**What the embedding side contributes to it (CS-7e, 2026-10-26).** The
deliverable's *retrieval* half is the cosine (CS-6d's shape: "retrieve with
the cosine, rank with the learned arm"). The late chunking makes that
retrieval half measurably better on the same model: the positives' hit@1
rises from the cosine's 0.1050 to **0.1260** and the within-document AUC
from 0.5238 to **0.5944**, with the control's score level unmoved (+0.0011)
and the reviewed hard negatives' hit@1 *falling* — i.e. a better first
guess for a paragraph at no cost in ordinary cells, and with 5.3x fewer
encoder calls (the document is embedded once, not every chunk). It does
**not** replace the decision model as the deliverable's scorer (0.5944
against 0.7701 within-document, 0.1260 against the rerank's 0.1565
hit@1), and the arm is **HF-only** (the token-level capability), so the
shape it suggests for the UPR port is "the late-chunked cosine retrieves,
the decision model decides" — a Stage-5 question, not a Stage-2 one.
