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

- **CS-30** (the corpus map as a classifier) → executed 2026-11-01 → the
  archive's §7.4 (the corpus map as a classifier) and the track log's §CS-30.
  The corpus statistic reaches **micro-F1 0.4924** against the champion's
  0.5479 (90% of it, with no embedding and no model call); the map's
  **log-odds weighting buys +0.2226** over a plain word count (0.4924 against
  0.2698), and the **cross-fit fit side was the right instrument** — the
  naive in-fold number would have put the count only 0.0227 behind (the
  out-of-fold puts it 0.2780 behind).
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

  **The pre-registration (2026-11-01, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **Where the code lives.** The **estimator** is pure and lives in
    `text_processors/lexical/lexicon.py` (the standard library only — no
    numpy needed, no model client); the **artifact block** stays in
    `text_processors/features/corpus_map.py`, so the CS-23 shape is preserved
    and the viewer keeps reading one block. The **scorer** (CS-30's) gains the
    z-scored arms.
  - **The estimator's five new measures**, all from the **same 2×2 table**
    (`a` word+label, `b` word+no label, `c` no word+label, `d` neither), the
    label's count `n_label`, the corpus's count `N` and the word's **document
    frequency** `df`:
    1. **`log_odds_informative`** — the **informative-prior log-odds** in the
       form of the reference implementation the block above names
       (`logodds` on PyPI, MIT, no dependencies — Monroe, Colaresi & Quinn
       2008's Dirichlet prior), **adapted to the document level**: the word's
       global document frequency is the pseudo-count, `α_w = df + alpha`
       (`alpha` = 0.01, the reference's floor), and `α₀ = Σ_w α_w` over the
       map's vocabulary (the corpus's own (text, word) incidences). Then
       `δ_label = log((a+α_w)/(n_label+α₀-a-α_w))`,
       `δ_other = log((b+α_w)/(n_other+α₀-b-α_w))` and the score is
       `δ_label − δ_other`.
    2. **`standard_error`** — `σ = sqrt(1/(a+α_w) + 1/(b+α_w))`, the
       reference's own variance of the difference.
    3. **`z_score`** — the plan's **δ over the standard error**: the
       informative log-odds divided by its σ (the reference's z-score).
    4. **`g2` and `p_value`** — the **significance** on the same table: the
       likelihood-ratio **G²** statistic (1 degree of freedom) and its
       p-value, `p = erfc(sqrt(G²/2))` — the chi-square 1-df survival
       function, which the standard library computes exactly (the track's
       package may not import scipy, so the p-value is computed here rather
       than taken from a library).
    5. **`log_odds_conditional`** — the **conditional smoothing**: the 0.5
       added **only to the cells that are actually zero**, the variant the
       statistics literature criticises our fixed 0.5 for (Jackson et al.,
       *JRSS-M* 2021; Agresti). The **share of pairs whose sign flips**
       under it is reported.
  - **The current log-odds is NOT touched.** CS-23's numbers stay exactly as
    they are; the new measures are added **beside** them (the reviewer's
    decision of 2026-11-01), so every number CS-23 reported stays reproducible
    from its artifact.
  - **The reference implementations are oracles, not dependencies.** The
    formulas follow the published method (the `logodds` package's own
    "Method" section, Monroe et al. 2008; `tidylo::bind_log_odds()`; the
    `quanteda.textstats::textstat_keyness()` family for the significance);
    **no package is installed** (this change set's cost line says *no new
    dependency*), so the implementation is pinned by **hand-computed unit
    tests** instead.
  - **The lexicon artifact.** The map's words, per label, with the new
    measures and the counts behind them — **written once**, in the map's own
    block, so CS-32, CS-33, CS-35 and the viewer all read the same numbers.
    The block gains the new measures per pair and a **summary**: the head
    word per label under each measure (with its counts), the head-word
    agreement between the measures, and the conditional smoothing's sign-flip
    share.
  - **The classifier re-measurement.** The map's weights are re-measured with
    the **z-score** instead of the log-odds, through two new arms of the same
    scorer: `sum_zscore` (the sum of the z-scores over the text's words) and
    `sum_zscore_positive` (the same sum over the words whose z-score is
    positive — the analogue of the **selected** arm, `sum_positive`). Two
    more runs, `upr-eval-corpus-map-sum-zscore-val` and
    `upr-eval-corpus-map-sum-zscore-positive-val`, on UPR **val** at the same
    `top_k` k=2.
  - **The comparison is paired.** The two arms are scored on the **same
    texts** with the same map and the same operating point, so the
    difference is paired (the track's rule: no verdict rests on two
    independent numbers). A small tool (`tools/lexical_compare.py`) rebuilds
    both arms' score matrices from the runs' frozen configs, **checks them
    against the runs' own artifacts** (the recomputed micro-F1 must equal the
    artifact's), and reports the paired bootstrap CI of the difference
    (`eval/multilabel.py: paired_metric_bootstrap`, the resampling unit the
    text).
  - **The falsification, registered in advance.**
    1. **The head-word test.** The z-scored head word must be **different**
       from the log-odds's head word for **at least 5 of the 60** labelled
       tables: **if it is unchanged for ≥ 55 of the 60, the prior does nothing
       here and the metric upgrade is a null** — reported as a null. (Read
       only on the **train** split, before the run: with the formulas above
       the head changes for **13 of the 60**, so the registered null does not
       fire — but the plan's *expectation* that the z-score demotes the
       single-document words is **not** met: the z-scored head is a hapax for
       the **same 4** labels as the log-odds's head. The reason is the
       formulas' own: the informative prior shrinks the **common** words (the
       register — "rights" is in all 60 tables), while the **standard error**
       is what shrinks the rare ones. Both readings are reported.)
    2. **The model test.** The z-scored weights must **move** the classifier:
       if the pooled micro-F1 does not move, the better measure is a better
       **description**, not a better **model** — a null, reported as a null.
       The reading is on the **val** re-measurement (the two new runs) and on
       the **fit side** (the out-of-fold numbers, which the selection sees).
  - **The exit criteria.** The map's artifact carries the lexicon; the six
    map runs (the four CS-30 arms and the two z-scored ones) hold their
    numbers; the table rows are auto-appended; the log section reads the
    numbers out with the falsification conditions; the paired comparison's
    tool checks its own rebuild against the artifacts; the gate green (ruff
    over `src`/`tests`/`experiments`, the full pytest suite, the two verify
    tools, and both final sweeps reproducing their artifacts field for field).

- **CS-31** (the corpus map's estimator, and the lexicon as an artifact) →
  executed 2026-11-01 → the archive's §7.4 (the corpus map's estimator) and
  the track log's §CS-31. The **z-scored weights move the classifier**:
  0.4924 → **0.5243** (a paired **-0.0319 [-0.0526, -0.0120]**, the interval
  excluding zero) — the better measure is a better **model**, not only a better
  description; the head-word test's null does not fire (13 of the 60 heads
  change) but the plan's expected demotion of the single-document words does
  not happen (the prior shrinks the **common** words, the standard error the
  rare ones).
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

  **The pre-registration (2026-11-01, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **The feature.** A new **feature group** `corpus_map` in the UPR pack's
    `UPR_FEATURE_GROUP_PREFIXES` (the prefix `cmap_`), with **one** column:
    the map's **per-cell score** under the z-scored arm
    (`sum_zscore_positive` — the arm CS-31's fit side selected). The pack
    does **not** compute it: the map is built at the edge (from the train
    split) and its score matrix is passed in, exactly like the decision
    probability (CS-13) and the reranker's score (CS-16) — so the column is
    **cell-local** by construction (no document identity, no length).
  - **The gating.** The group is **gated like the catalog groups**: it can
    only be selected when the map is available, and a run without it records
    the gap in its artifact. The map's availability is the config's
    `upr_analysis.learned.corpus_map_feature.enabled` (off by default, so
    every earlier run's artifact keeps its exact content) — and the group's
    selection is refused without it, the same contract the catalog groups
    have.
  - **The guard (not optional).** The pack is fitted on **train** and the map
    is built from **train**, so on the fit side the map has already seen the
    cells it scores — the feature would be optimistically biased exactly
    where the combiner learns its weight. The **fit side's map is
    cross-fit**: **5 folds** over the train split, each fold's cells scored
    against a map built from the **other** folds (CS-31's
    `cross_fit_scores`, the same shape as the fit-side threshold's cross-fit
    in CS-9b). The **evaluated** split is scored with the **full** train map.
    The artifact records the folds and the cross-fit's seed.
  - **The three floors, and the control is the point.**
    1. the **control**: the same pack with the group **removed** — fitted and
       scored **in the same run**, so the Δ is paired by construction;
    2. the **random-vector floor**: unchanged (the pack's own);
    3. the **label-blind control**: the same pack, but the map is built with
       the **labels permuted** among the train records (the per-label counts
       are preserved, only the assignment changes — the track's own
       permutation discipline). If the label-blind map buys the same Δ as the
       real one, the "gain" is the map's *construction*, not the corpus's
       word↔label association — the failure mode the track's control-role
       floor was built to catch.
  - **The run.** One: `upr-analyze-cmap`, the `analyze-upr` command with the
    learned arm's groups `embedding + lexical + chunk_profile + corpus_map`
    (the CS-13 pack's groups plus the new one), the combiner the CS-13 pack's
    (the gradient-boosted one), `top_k` k=2 on the **val** split. The sweep
    file `experiments_text_processing/upr_learned_cmap_sweep.py` (the sweep
    label `upr_learned_cmap`, so the table's section is
    `## UPR — upr_learned_cmap`); the artifact
    `results/upr-analyze-cmap/analysis.json`. The label-blind counterpart
    rides in the **same run** (the CS-13 precedent).
  - **The measurement and its falsification.** The **paired** Δ micro-F1
    (the real arm minus the control) with its CI over the texts (the
    resampling unit), plus the per-bucket breakdown, plus the label-blind
    arm's Δ beside it. The **unseen bucket is expected flat**: the map's
    table for the eight held-out labels is empty, so the feature cannot help
    there. The falsification: the Δ's CI must **exclude zero** *and* the
    label-blind control's Δ must be **materially smaller**. If the Δ is
    inside the noise, the corpus's word statistics add **nothing** that the
    label-name overlap features (`lex_label_coverage`, `lex_text_coverage`)
    do not already carry — a null, reported as a null. "Materially smaller"
    is read as: the label-blind Δ's interval includes zero (the blind map
    buys nothing measurable) while the real Δ's excludes it.
  - **The exit criteria.** The artifact holds the Δs with their intervals, the
    folds and the three floors; the table row is auto-appended; the log
    section reads the numbers out with the falsification conditions; the gate
    green (ruff over `src`/`tests`/`experiments`, the full pytest suite, the
    two verify tools, and both final sweeps reproducing their artifacts field
    for field).

- **CS-32** (the corpus map in the learned pack) → executed 2026-11-01 → the
  archive's §7.4 (the corpus map in the learned pack) and the track log's
  §CS-32. The map's feature is worth **+0.0684 [+0.0441, +0.0915]** of
  pooled micro-F1 over the same pack without it (the pack reaches **0.5502**,
  past the champion's 0.5479) and the **label-blind** map buys **−0.0509** —
  the gain is label-driven; but it is a **seen-labels** feature: it costs the
  unseen bucket **−0.4867** (the empty tables mark the eight held-out labels
  as impossible).
- **CS-33** (the word-class readout: the classes, the readout and the
  class-restricted map) → executed 2026-11-01 → the archive's §7.4 (the
  word-class readout) and the track log's §CS-33. The classes separate widely
  (the registered null does not fire, so **CS-34 is not cancelled**), but the
  **class-restricted map is much worse** as a classifier than the full map
  (0.3815 and 0.0798 against 0.5243).

- **CS-34** (real part-of-speech tags: the tagger at the edge, the
  grammatical readout and the POS-restricted map) → executed 2026-11-01 → the
  archive's §7.4 (the real POS tags) and the track log's §CS-34. **Both halves
  of the registered falsification fail** — the nouns do not separate from the
  verbs (+0.0188 against a margin of 1.0) and the noun-only map is materially
  worse than the full map (0.4840 against 0.5243, a paired **+0.0403
  [+0.0258, +0.0545]**) — so the **POS line is a null** and the `nltk`
  dependency is not worth keeping.

- **CS-35** (the label→word lexicon as a label-side channel: the mined-words
  query and the mined-words column, plus the expansion route for the eight
  labels with no train example) → executed 2026-11-02 → the archive's §7.4 (the
  label→word lexicon as a label-side channel) and the track log's §CS-35. **The
  pre-registered bar is not met** — the mined words are the **best cell-level
  signal in the pack** (the AUC **0.8109**, above the whole-text cosine's
  0.7818 and the description's 0.7067) and the channel buys **+0.0493 /
  +0.0507** of seen-bucket micro-F1 over the control, but the head-to-head
  against the name+description channel does not separate them (the query half's
  seen view **+0.0217 [−0.0014, +0.0445]**; the feature half's **+0.0272
  [+0.0044, +0.0502]**, with only one of the three seen buckets clearing) — so
  the lexicon **does not become a catalog channel**; the **expansion route** is
  a null for the eight.

### 7.5 The map's quality round (CS-36 … CS-41)

**What prompted it (2026-11-02).** The reviewer's seven ideas, after CS-35: the
map's unit could be a **phrase** ("do phrases beat their own words?"); the map
could have a **pair interaction** ("which pairs say more than their parts?"); the
**label space** could have a lexical geometry of its own ("which labels are
lexical neighbours or near-duplicates; which have no vocabulary of their own");
the map could have a **stability** ("which word-to-label links survive
resampling"); the **retrieval** could be lexical ("how much of retrieval is
words alone"); the map could say **"unknown"** instead of "impossible", per
label and per text; and the map could **explain itself** ("why A and not B for
one text").

**What the round so far established, in one paragraph.** The corpus map (CS-23)
is the round's centrepiece, and we now know three things about it that were
never designed for. Its **bulk is fragile**: **66.7 %** of the 18,707
(label, word) pairs rest on a **single train text** and **13.6 %** are hapaxes —
**CS-36 measured that fragility** (not one of the 12,483 single-text links
survives a resampling) **and found that removing it makes the classifier
worse**, so the fragility is real but it is not what the map's classifier is
made of — while its **heads are clean** (not one of CS-35's 600 mined words is
a hapax: the idf factor already demotes them), so the noise sits in the
**tail** the sum arms add up. Its **estimator is asymptotic** (CS-31's
z-score), never empirical — but the empirical counterpart now exists
(CS-36's resampling). And its **biggest defect is not its quality but its
abstention**: the empty tables make
the eight held-out labels *impossible*, and with the map's column in the pack
the unseen bucket goes **0.4867 → 0.0000** (CS-32) — **and CS-37 measured that
this is not an encoding problem**: re-encoding the channel's silence as "no
evidence" instead of "impossible" changes nothing (the eight stay at 0.0000),
because on the fit side the silence is a *perfect* proxy for the label's
absence from the training data. And the label space's own geometry is **not a
lexical discovery**: the corpus **rediscovers the taxonomy** (CS-38: the labels
that share a catalog parent are closer in vocabulary, **+0.0413 [+0.0372,
+0.0451]**, and there are no lexical twins at all) — so what the map can add
is the **words**, and where it has nothing to add is the labels whose own
lexical identity is thin. And the map's **unit** stays the word: the phrase
table (CS-39) is **too rare to be a table** (82.8 % of its pairs rest on one
train text) and the union it buys is noise, not signal — the register shows up
as a *fragment* ("on its", "with an"), not as a topic. And the map **can
explain itself**: CS-40's contrastive read of two tables against each other
answers "why A and not B" — and the answer is **not** the name-overlap
features' (the vocabulary share **0.6100** against the registered bar 0.50:
for **80 %** of its rows the screen does not print both of the pair's
log-odds), and the sharp case is the common one (**62.5 %** of the measured
pairs have a word of this text that pulls the two labels in **opposite**
directions). And the **words retrieve almost as well as the embedding**: CS-41's
lexical kNN (the TF-IDF neighbours over the train split, voted by the
pipeline's own kNN machinery) reaches a shortlist R@50 of **0.9437** against
the embedding kNN's **0.9576** — while the champion's own fusion sits at
**0.7423**, barely above the chance level of a top-50 shortlist over 68
labels. So the round's last question ("how much of the retrieval is words
alone?") has the answer: **almost all of the breadth** — and the fusion's
mixing **dilutes** the candidate set. So the round's question is
one question: **how good is the corpus map, and where does it not know?**

**The order:** CS-36 (the stability map) → CS-37 (the selective map) → CS-38
(the label-pair map) → CS-39 (the phrase map) → CS-40 (the contrastive map) →
CS-41 (the lexical kNN).

**The reviewer's decisions of 2026-11-02 (they bind the change sets below).**

1. **The batch is authorised, one change set at a time.** Each change set is
   pre-registered (the mechanism, the shape of the number, the falsification and
   the cost) **before its first call**, and the round runs in the order above; a
   change set that turns out to need a decision stops and asks.
2. **Every number this round produces is a UPR *val* number** and never a UPR
   *test* number; the test split stays untouched.
3. **The corpus map's existing block is not renamed or reshaped.** The phrase
   table (CS-39) is a **separate** builder, and every number CS-23 and CS-31
   reported stays reproducible from its own artifact.
4. **The multiple-comparisons discipline.** This round adds arms to a val split
   that has now carried about sixty of them, so a claim that the map "got
   better" must carry the **fit side's out-of-fold** number (CS-30's
   instrument) beside the val one, and the **label-blind control** (the track's
   own yardstick) must still buy nothing.
5. **A new unit inherits the hapax problem.** A phrase is rarer than a word, so
   the estimator's work (CS-31) has to be redone per unit — or the new map will
   look worse for a reason that has nothing to do with phrases.

- **CS-36** (the stability map: which word-to-label links survive the
  resampling) → executed 2026-11-02 → the archive's §7.5 (the stability map)
  and the track log's §CS-36. **The readout is the deliverable and the filter
  is a null**: the map's bulk *is* sampling noise, now measured (the survival
  curve has a **cliff at 63 %** — 1 − e⁻¹, exactly where a single-text link
  belongs — and **not one** of the map's 12,483 single-text links survives
  τ = 0.75; only **27.6 %** of the map does), but the **model test fires**: the
  filtered map's fit side's out-of-fold micro-F1 is **lower** (0.6544 at
  τ = 0.75, 0.6473 at τ = 0.90, against the full map's **0.6613**) and the val
  number agrees (a paired **+0.0053 [−0.0015, +0.0128]**) — a better
  description is not a better model, so **the filter does not ship**. The
  label-blind control buys nothing (the blind map stays at chance and the
  filter makes it worse), and the map's **head words** keep their place among
  the head in only **50 %** of the resamples.

- **CS-37** (the selective map: "unknown" instead of "impossible") →
  executed 2026-11-02 → the archive's §7.5 (the selective map) and the track
  log's §CS-37. **The abstention is a null, and the reading is the finding**:
  a label with no table carrying **NaN** ("no evidence") instead of the floor
  ("impossible") leaves the eight held-out labels at **0.0000** — the paired
  unseen-view difference is exactly **+0.0000** — so the "impossible" verdict
  was **never in the encoding**: a map with no word↔label information at all
  collapses them the same way (the blind arm's unseen 0.0067), and it is the
  **channel's silence on the fit side** (the eight are the labels the map is
  silent about *and* the labels that are never positive in train) that costs
  them. The seen buckets are untouched (a paired **+0.0032 [−0.0037,
  +0.0100]**) and the label-blind control buys nothing (−0.0053 [−0.0189,
  +0.0091] on the seen view), so the map's column stays a **seen-labels**
  channel. The **per-cell** abstention (not registered) is the variant that
  could still work: a channel that is silent for *many* cells is no longer a
  perfect proxy for the label's absence.

- **CS-38** (the label-pair map: the lexical neighbours, the twins, the
  labels with no vocabulary of their own) → executed 2026-11-02 → the archive's
  §7.5 (the label-pair map) and the track log's §CS-38. **The relation is a
  null and the class is a finding**: the corpus's label-pair relation **is**
  the taxonomy's (the registered disagreement test fires on both legs — the
  sibling pairs' mean vocabulary Jaccard is **+0.0413 [+0.0372, +0.0451]**
  above the other pairs' and **11 of the corpus's 20 closest pairs** are
  siblings against a bar of 4; their co-occurrence lift is four times the
  others'), so **the corpus knows nothing the taxonomy did not already say**
  and no action is licensed; but the **"no vocabulary of their own" class**
  exists with **13** members (the eight held-out labels plus five whose own
  words lie almost entirely inside one neighbour's table — *Sexual and
  reproductive rights* 94 % inside *Women's rights*, *Freedom of assembly* 91 %
  inside *Detention*), and **there are no lexical twins** at all (not one of
  the 2,278 pairs reaches the registered near-duplicate threshold; the closest
  is 0.489).


- **CS-39** (the phrase map: do phrases beat their own words?) → executed
  2026-11-02 → the archive's §7.5 (the phrase map) and the track log's §CS-39.
  **The unit change is a null**: the **formulaic test fails** (the phrase
  table's generic share is **15.5 %**, far *below* the word table's **43.6 %** —
  the phrases are too rare to sit in many tables, so the class measure reads
  rarity as topicality) and the **label-blind control fails** (the blind union
  gains **more** on the fit side, **+0.0240**, where the real union gains
  **+0.0050**: the phrases' contribution is noise, not signal); the **model
  test** passes only mechanically (the union's fit side's out-of-fold **0.6663**
  against the word map's **0.6613**, the val interval including zero). The
  register is real but it is a **fragment**: the top interactions are "on its",
  "with an", "and will" — the corpus's formulas, which say nothing about the
  topic. And the phrases are **too rare to be a table** (82.8 % of the phrase
  pairs rest on a single train text, against the word table's 66.7 %).

- **CS-40** (the contrastive map: "why A and not B") → executed 2026-11-03 →
  the archive's §7.5 (the contrastive map) and the track log's §CS-40. **The
  block stays**: the discriminating words are **not** the ones the topic-side
  and the text-side maps already show (the vocabulary share **0.6100** against
  the registered bar 0.50) and for **80 %** of its rows the screen does not
  print **both** of the pair's log-odds — the reader could not have formed the
  difference; and the sharp case is the common one (**62.5 %** of the measured
  pairs have a word of this text that pulls the two labels in **opposite**
  directions). A **tool** change: no run, no table row, no artifact
  re-written.

- **CS-41** (the lexical kNN: how much of the retrieval is words alone?) →
  executed 2026-11-03 → the archive's §7.5 (the lexical kNN) and the track
  log's §CS-41. **The lexical route IS a viable candidate generator**: the
  words alone retrieve nearly as well as the embedding (the shortlist R@50
  **0.9437** against the embedding kNN's **0.9576** at the same k, and above
  the embedding family's floor 0.8918) — so **PLAN.md's BM25 checkbox is
  LIVE**; and the **champion's own fusion is the worse candidate generator**
  (0.7423, barely above the 0.7353 chance level of a top-50 shortlist over 68
  labels) — the fusion **dilutes** the candidate set, while the embedding's
  advantage is in the **ordering** (micro-F1 +0.0783), not in the **breadth**.
  The measured caveat: the shortlist's floor is high (the label-blind arm's
  0.8111), so the term's own payoff must be read on the fusion's shortlist with
  the term, or on the **union** of the two candidate sets.

### 7.6 The fit-side dictionary round (CS-42 … CS-45)

**What prompted it (2026-11-03).** The corpus map (CS-23/CS-31) is the train
split's own dictionary: 18,707 (label, word) links over 1,676 texts. As one
column in the learned pack it buys **+0.0684 [+0.0441, +0.0915]** of micro-F1
(the pack goes 0.4818 → 0.5502, past the champion's 0.5479), but it costs the
eight held-out labels everything: their bucket falls **0.4469 → 0.0000**,
because their tables are empty. CS-37 proved the fix is **not** in the encoding
(NaN changed nothing; a label-blind dictionary collapses the eight identically):
on the fit side, "the dictionary is silent" and "the label is never positive"
are the same fact. The four change sets below attack that coincidence from four
sides. The bonus (§7.7) is independent: it is a candidate-generator measurement
on CS-41's / PLAN.md's BM25 thread.

**The order:** CS-42 (the fit-side mask) → CS-43 (the dictionary + the catalog
description in one pack) → CS-44 (the borrowed words) → CS-45 (the distinctive
lexicon) → the bonus CS-46.

**The reviewer's decisions of 2026-11-03 (they bind the change sets below).**

1. **One change set at a time**, pre-registered before its first call: the
   mechanism, the shape of the number, the falsification and the cost. A change
   set that turns out to need a decision stops and asks. The next starts only on
   the reviewer's word.
2. **Every number is a UPR *val* number**; the test split stays untouched.
3. **A new knob is off by default**, so every earlier artifact keeps its exact
   content and reproduces field for field.
4. **The map's existing block is not renamed or reshaped**; a new instrument is
   a separate option or a separate builder.
5. **The masked labels are chosen by a fixed seed from the *train* map's label
   list**, never by a val statistic; the chosen names are recorded in the
   artifact.
6. **The multiple-comparisons discipline.** A claim that the map "got better"
   carries the paired CI and the label-blind control (the track's own yardstick)
   beside it.

- **CS-42** (the fit-side mask: the dictionary's floor stops meaning
  "impossible") → executed 2026-11-03 → the archive's §7.6 (the fit-side mask)
  and the track log's §CS-42. **The mask is a lever, but not the whole lever**:
  breaking the "floor ⇒ never positive" coincidence on the train side lifts the
  eight held-out labels from **0.0000** to **0.2947** (five of the eight carry
  real scores again), and it costs the seen buckets nothing — they **gain**
  (the seen view **+0.1478 [+0.1236, +0.1710]** against the control; the pooled
  number **0.5874**, the track's second-best pack number, above the champion's
  0.5479) — but the registered bar was **0.45**, so the recovery condition
  **fires**: the silence proxy is **part** of the cause, not all of it. The
  three that stay down are the two catch-alls (*General*, *Other*) and the broad
  *ESC rights - general* — the labels whose meaning is not in the texts' words.
  The label-blind control buys nothing (the seen view −0.0032 [−0.0181,
  +0.0119]; the unseen bucket 0.0066). The knob is off by default; CS-32's and
  CS-37's artifacts keep their exact content.

- **CS-43** (the dictionary + the catalog description: the description cannot
  outvote the floor) → executed 2026-11-03 → the archive's §7.6 (the description
  in the dictionary's pack) and the track log's §CS-43. **The dictionary
  overrules the description**: with the description in the pack the eight
  held-out labels still collapse — **0.0066** (the feature half) and **0.0000**
  (the query half) against the same runs' controls' **0.4824** / **0.4690**,
  paired −0.4758 [−0.5297, −0.4174] and −0.4690 [−0.5228, −0.4112] — while the
  **pooled** numbers pass (**0.5608** / **0.5532**, both at or above 0.55; +0.0532
  [+0.0304, +0.0751] and +0.0448 [+0.0190, +0.0688] over the controls). So the
  two channels are complementary on the **seen** labels and not on the eight: a
  non-silent neighbour does not change what the silent channel teaches the
  learner. The registered branch fires: cell 2 is exactly 0.0000, cell 1 is
  0.7 % of its control (0.0066 — recorded, not smoothed). The label-blind
  controls buy nothing on the seen view (+0.0036 [−0.0093, +0.0169] /
  −0.0131 [−0.0258, −0.0002]), and both cells' controls reproduce CS-11's arms
  field for field (0.5076/0.4824 and 0.5084/0.4690 — the guard). **The route for
  the eight stays CS-42's fit-side instrument**, which itself recovered only
  two-thirds; the description's own value is the seen-label floor (its pack is
  the track's third-best pooled number, behind CS-13's 0.5965 and CS-42's
  0.5874).

- **CS-44** (the borrowed words: the partner's vocabulary is not the thin
  labels') → executed 2026-11-03 → the archive's §7.6 (the borrowed words) and
  the track log's §CS-44. **The lending is not licensed**: it works mechanically
  — the thin labels' predicted counts rise **69 → 118** and *CP rights -
  general* genuinely gains (**0.0000 → 0.2308**, its partner being the broad
  *International instruments*) — but on the registered cells view the five
  **fall** (**−0.0707 [−0.1356, −0.0035]**, and only **2 of 5** individual signs
  are positive), the partner set's interval's lower bound crosses the −0.02
  floor (**−0.0270**; its point estimate is −0.0012, so "no material change"),
  and the pair becomes confusable exactly as registered (the co-predictions rise
  from 0–5 to 6–21; *Detention* pays −0.1026). The reading: the thin labels'
  evidence is **shared** evidence — the corpus's words for sexual and
  reproductive rights *are* its words for women's rights — so re-weighting the
  same words cannot create the distinction the words do not carry. The lending
  relation reproduces CS-38's five pairs and shares; the real/control/blind/
  random arms reproduce CS-32's numbers exactly.

- **CS-45** (the distinctive lexicon: the register was the signal) → executed
  2026-11-03 → the archive's §7.6 (the distinctive lexicon) and the track log's
  §CS-45. **The bar is not met again — and the reading is that the register was
  the signal**: keeping only the distinctive/shared words (60 mined, 4 expanded,
  4 fallback — *ESC rights - general*, *Gender-based violence*, *General*,
  *Other* have no non-generic word) **halved** the channel's gain over the
  control (the seen view **+0.0272 [+0.0074, +0.0474]** and **+0.0339
  [+0.0139, +0.0548]** against CS-35's +0.0507 and +0.0493) and took the column's
  cell-level AUC down to the description's own level (**0.7073** against CS-35's
  0.8109 and the description's 0.7067); the head-to-head against the
  name+description channel does not clear the seen view on either half
  (+0.0037 [−0.0186, +0.0252] and +0.0064 [−0.0155, +0.0272]), the feature half
  clears **0 of 3** seen buckets and the query half only the torso (+0.0312
  [+0.0000, +0.0644], with the head falling −0.0227). So CS-35's redundancy
  reading is confirmed: the mined channel's advantage over the description was
  the register — the part the name and the description already carry. The filter
  does help the **torso and the tail** (the rare labels' signal) and costs the
  **unseen** (the feature −0.0503; the query −0.0973 [−0.1450, −0.0529], the
  filter having emptied four expansions). The CS-35 sweep re-ran and all seven
  artifacts reproduce identically apart from the wall clock.

### 7.7 The candidate-set union (CS-46)

**Why it is separate.** It is not a dictionary change: it is CS-41's retrieval
thread, and the evidence PLAN.md's deferred BM25 checkbox is waiting for.

- **CS-46** (the kNN union: the two retrievers see the same labels) → executed
  2026-11-03 → the archive's §7.7 (the kNN union) and the track log's §CS-46.
  **The shortlist gap is not a breadth gap**: the per-row z-max union of the
  lexical kNN (CS-41) and the embedding kNN reaches **recall@10 0.6943** against
  the decision model's union's **0.9120** (the breadth test fails by 0.22), and
  its **recall@50 0.9548** is *below* the embedding arm's own 0.9576 — the union
  **is** the embedding arm (**+0.001** at k = 10, **−0.003** at k = 50): the two
  candidate sets are nearly the same set, and the lexical half adds nothing the
  embedding did not already see. The label-blind floor is **0.2313 / 0.8029**
  (the frequencies alone), so the label-driven breadth is +0.46 at k = 10 and
  +0.15 at k = 50. And the fusion's shortlist gap is an **ordering** problem at
  depth 50: its recall@10 (**0.7126**) is the best of all five candidate
  generators while its recall@50 (0.7423) is the worst. **The BM25 checkbox
  stays closed** — not because the lexical route is weak (CS-41 measured it
  strong) but because it is **already covered**.

**The round is executed** (CS-42 … CS-46): every change set has its archive block
and its log section. The next step is the reviewer's: the fit-side mask (CS-42)
is the one that moved the eight, the description (CS-43) and the lending (CS-44)
did not, the distinctive lexicon (CS-45) confirmed CS-35's redundancy reading,
and the kNN union (CS-46) closes the *breadth* argument on `PLAN.md`'s deferred
BM25 checkbox — its *scoring* question (a lexical term in the fusion, a fourth
fusion parameter) remains the reviewer's.

### 7.8 The dictionary's own ranking (CS-47)

- **CS-47** (the dictionary's own ranking: the classifier's answer) → executed
  2026-11-04 → the archive's §7.8 (the dictionary's own ranking) and the track
  log's §CS-47. A **tool** change: the viewer's block, the pure helper
  (`text_processors/lexical/contrastive.py`'s `dictionary_ranking`) and the
  tests; **no run**, **no table row**, **no artifact re-written**, **no new
  dependency**. The block's scores are `map_score` over the artifact's own
  rows, the empty tables are omitted (**8** on the UPR map), and the contrast
  block's B carries the same ranking's rank.

### 7.9 The dictionary's chain ported to OMCT (CS-48 … CS-57)

**What prompted it (2026-11-05).** The UPR dictionary track ran from CS-18 to
CS-47: a word↔label map built from the training corpus (CS-23), used as a
classifier (CS-30) and improved with an informative-prior/z-score estimator
(CS-31), then read out by word class (CS-33), POS (CS-34), per-label lexicon
(CS-35/45), stability (CS-36), abstention (CS-37), label pairs (CS-38), phrases
(CS-39), a contrast view (CS-40) and a ranking view (CS-47). This round ports
**the whole dictionary-side chain** to OMCT — **including the change sets that
were NULLS on UPR**.

**The transfer test, and the two datasets.** A method that failed on UPR may
work on OMCT, and a null that reproduces is just as informative. **The UPR
verdict is the prior, never a reason to skip.** Every number in this round is an
**OMCT** number (run ids keep `omct-`, the log names the dataset); the two
negative subsets (A = the reviewer-verified hard negatives, B = the unjudged
control) are **never pooled**; nothing here is a UPR metric and no UPR artifact
is touched. **OMCT's own prior** (CS-21, `§7.1` of this plan): the surrogate
does not transfer; the **word-pair** view is the workable object; the clause
ablation is ~2,700 calls and is **out of scope** for this round.

**The order, and the stop rule.** CS-48 (the map) → CS-49 (the map as a pair
scorer) → CS-50 (the estimator) → CS-51 (word classes) → CS-52 (POS) → CS-53
(phrases) → CS-54 (stability) → CS-55 (per-label lexicon + register) → CS-56
(abstention) → CS-57 (question pairs). Foundation first, cheap readouts next,
odd ones last. **One change set at a time, with a stop and a report between
them** (the §7.1 stop rule, which binds every session): a change set is done
when its pre-registration is written, its code and tests are in, its run's
numbers are in an artifact, its log section is written, its box is ticked with
the numbers, its pre-registration is moved to the archive and the commit is
made — then the work **stops** and reports. **The nine below CS-48 are the
plan, not the queue: each gets its own pre-registration before its run, and
none starts without the reviewer's word.**

**The measurement discipline (it binds every change set of the round).**

- **Within-document AUC is the headline** (a pair is ranked against the other
  pairs of the same document); the **pooled AUC goes beside it, never alone**
  (the pooled number moves with the documents' score levels, and 58.7% of the
  reviewed set's positive-negative pairs are cross-document).
- **The reviewed set is rank-selected**: the Stage-0 cosine's **pooled** pair
  AUC is **0.4680 / 0.5004 / 0.5371** on the three models (its
  **within-document** AUC **0.5020–0.5632**), so absolute AUCs look low **by
  construction**. The instruments are the **relative** comparison (against the
  plain full-text cosine of the same run) and the **control role** (does the
  method only re-scale the ordinary unjudged cells?). A dictionary scorer that
  **beats the cosine is the finding**; one that matches it is a null.
- The map is built on the **reviewed pairs only** (subset A), with the track's
  **cross-fit**: folds over the pairs, each fold's map from the other folds
  (the same `default_rng(seed).permutation` + contiguous-chunk scheme as
  CS-30/CS-31). **The unjudged control (B) never enters the fit and is never
  ground truth.**
- An **empty table** (a question with no evidence in a fold) carries the row's
  floor and can never be ranked first; **the number of pairs that carries the
  floor is reported in every change set**.
- **Per document (NU / CIDH / AI) as well as pooled**; the AI stratum is small
  (99 pairs) and is read with that in mind.
- **No thresholds in the map**: every (question, word) pair that occurs is
  stored with its counts; the noise is displayed (the counts sit next to the
  number), never removed by dropping data.

**The round's UPR priors, in one table** (the per-change-set pre-registration
states its own row in full; this table is the plan's shape). Reading the table:
the UPR verdict is what the same change set concluded on the **UPR val** data,
the reason is **why** it concluded so, and the flip is what "it works on OMCT"
would look like — the question the change set answers.

| # | the change set (its UPR twin) | the UPR verdict (the prior) | the reason it was so | the flip on OMCT (what "it works" would be) |
|---|---|---|---|---|
| CS-48 | **the map** (CS-23) | **not a null**: a description, and the substrate of CS-30/31 | the unit was **thick** — 1,676 texts × 68 labels, each (label, word) pair with a df spanning the corpus | the **per-question** table is a table at all: the shape and the availability read out as pre-registered |
| CS-49 | **the map as a pair scorer** (CS-30) | **not a null**: the z-scored map reached micro-F1 **0.5243**, 90% of the champion, with no embedding | the map's measure work was worth **+0.2226** over a plain word count, and the cross-fit was the right instrument | the map's within-document AUC **beats the plain cosine's** on the same 1,520 pairs (the cosine is 0.50–0.56 there); matching it is a null |
| CS-50 | **the estimator** (CS-31) | **a gain**: 0.4924 → **0.5243**, the paired interval excluded zero | the informative prior shrinks the common words and the standard error the rare ones — a **model** gain, not only a description | the z-scored weights beat the log-odds weights **out-of-fold** on the pair unit |
| CS-51 | **word classes** (CS-33) | a **readout** win, a **payoff** loss (0.3815 / 0.0798 against the full map's 0.5243) | the strongest words are **rare**, the register is **everywhere**; a classifier needs coverage | the restriction **helps** (or hurts differently) when the "register" is the questionnaire's own vocabulary |
| CS-52 | **POS** (CS-34) | a **null**: both halves of the registered test failed | the nouns and the verbs were nearly identical (+2.0044 / +1.9856) and the noun-only map was materially worse | the grammatical axis separates (it did not on UPR; the test is cheap and the null may reproduce) |
| CS-53 | **phrases** (CS-39) | a **null**: the phrase table is too rare and the blind union gained **more** | 82.8% hapax; the register is a **fragment** ("on its", "with an") | bigrams beat their own words on short paragraphs (or the rarity null reproduces) |
| CS-54 | **stability** (CS-36) | the **filter is a null**, the **readout** is the deliverable (27.6% survives τ = 0.75; not one single-text link survives) | the map's classifier works on **coverage**, not on the strength of its best words | the survival curve and the per-question stability say how much of the pair unit is sampling noise |
| CS-55 | **per-label lexicon + register** (CS-35/45) | the **bar is not met** (the channel does not become a catalog channel) — and the register **was** the signal | the mined channel's advantage over the description was the register; the filter halved the gain | the **question's own words** as the label side beat the baseline — but the "name" analog must be said before measuring |
| CS-56 | **abstention** (CS-37) | a **null**: the "impossible" verdict was never in the encoding | the fit-side **silence** was the perfect proxy for the label's absence | here the silence is per-question and large (85 empty tables, 331 pairs): does the "impossible" lesson appear? |
| CS-57 | **question pairs** (CS-38) | a **null for the relation** (the corpus rediscovers the taxonomy), a **finding** for the vocabulary-less class | there is no catalog on OMCT, so the taxonomy's claim cannot be the test | the question-to-question vocabulary/co-occurrence relation, **adapted or dropped with the reason said** |

**What is not in this round.** The **pack-side** change sets (CS-32, CS-42,
CS-43, CS-44) need the OMCT pack and are not queued; the **viewer-side** ones
(CS-40, CS-47) have no viewer in this session and are not queued; CS-21/CS-22
stay untouched; the **clause ablation** (~2,700 LLM calls) is out of scope.

---

- **CS-48** (the question map on OMCT) → executed 2026-11-05 → the archive's
  §7.9 (the question map) and the track log's §CS-48. The map of the reviewed
  pairs' own word↔question association (the pair is the row, the paragraph the
  text, the question the label): **209 tables / 85 silent**, **27,979**
  (question, word) pairs, the lift-vs-log-odds separation **not** transferring
  (47.4 % of the contrastable tables against UPR's 91.7 %), and the cross-fit
  availability **65.9 %** (61.1 % with evidence). **The next change set is
  CS-49** (the map as a pair scorer): it is pre-registered before its run and
  starts only on the reviewer's word.

- **CS-49** (the map as a pair scorer) → executed 2026-11-05 → the archive's
  §7.9 (the map as a pair scorer) and the track log's §CS-49. The pre-registered
  primary arm (`sum_positive`) **matches the cosine** (0.5419 against 0.5238,
  +0.0181 [−0.0267, +0.0630]) — a null — and its all-pairs edge is the
  **floor's** (the non-floor reading is 0.4439); the spread shows the
  **measure-free count** as the only arm that beats the cosine (0.6360,
  +0.1122 [+0.0672, +0.1584], non-floor 0.6645). **The reading re-scopes
  CS-50**: the log-odds' magnitude is a per-question scale on the pair unit, so
  a within-question estimator cannot make the weights comparable across
  questions — the reviewer decides whether the estimator is still worth a
  change set, or whether a **per-question-normalized weight / the count**
  becomes CS-50's pre-registration instead.

- **CS-50** (the estimator) → executed 2026-11-05 → the archive's §7.9 (the
  estimator) and the track log's §CS-50. **The registered null was refuted**:
  the primary `sum_zscore_positive` reaches **0.7020** within-document against
  the cosine's 0.5238 (**+0.1782 [+0.1303, +0.2272]**, P 1.000), its
  **non-floor** reading **0.8158**, the blind gap **+0.2340** and **all three**
  documents above the cosine — the estimator **transfers**, and the reason (the
  prior's per-word normalization makes the weights comparable across questions,
  which the raw log-odds never were) corrects CS-49's reading. The head-word
  prediction was refuted too (84.7 % of the tables change their z head).
  **The next change set is CS-51** (word classes): it now has to say which
  increment the class restriction adds **on top of** the soft prior.

- **CS-51** (the word classes) → executed 2026-11-05 → the archive's §7.9 (the
  word classes) and the track log's §CS-51. **The hard filter is a null**: on
  top of CS-50's soft prior the "no generic" z arm is **−0.0123 [−0.0221,
  −0.0024]** against the full z arm (the registered ±0.02 band holds at the
  point estimate, but the interval excludes zero — a small, detectable
  **cost**), and the "distinctive only" arm loses **−0.0758 [−0.1052,
  −0.0472]** with its floor jumping to **71.6 %**. The **class readout** is the
  finding: the z-score separates the classes (distinctive **+0.3782** / generic
  **+0.0234**) while the **raw log-odds reverse CS-33's order** (distinctive
  **−0.2317** / generic **+0.9081**) — the α₀ denominators cancel the
  question's positive rate, which is what the raw log-odds' sign was reading.
  **The next change set is CS-52** (POS): with the class filter a null, the
  grammatical filter is unlikely to be a lever — the pre-registration must say
  so and test it.

- **CS-52** (the grammatical axis) → executed 2026-11-05 → the archive's §7.9
  (the grammatical axis) and the track log's §CS-52. **A null that reproduces**:
  the noun-only z arm is **−0.0041 [−0.0097, +0.0016]** against the full z arm
  (the interval includes zero — a free-ish filter, not a lever) and the
  verb-only arm **−0.0723 [−0.0878, −0.0571]**; the classes' **z means are
  within 0.016** of each other (noun +0.0885, verb +0.1032, the noun − verb
  difference **−0.0148** against the 0.5 bar, of the wrong sign). The
  grammatical axis is a null on both datasets, and CS-50's prior does not
  interact with it. **The next change set is CS-53** (phrases): the unit change
  is the next lever candidate — and the one whose UPR verdict was a null for
  rarity (82.8 % hapax).

- **CS-53** (the phrases) → executed 2026-11-05 → the archive's §7.9 (the
  phrases) and the track log's §CS-53. **A null that reproduces for the other
  reason**: the union is **−0.0043 [−0.0074, −0.0012]** against the word z arm
  (0.6976 against 0.7020) — inside the registered ±0.02 band at the point
  estimate, the interval excluding zero (a small, detectable **cost**) — and
  the phrase arm alone reads **0.6840**, below the word arm. The phrase table
  is **not** rare on the pair unit (14.8 % hapax against CS-39's 82.8 %; the
  median question's table holds 120 rows against the word table's 65), so the
  UPR null's **premise** does not transfer: the phrases are **redundant** with
  their own words, not rare — the pre-registered correction, made after
  measuring the prior, held. The CS-39 **readout's direction** does reproduce
  (the generic-class row share 14.3 % against the word table's 30.3 %).
  **The next change set is CS-54** (stability): its reference on OMCT is now
  the **z-scored** map (the arm that works), not the raw log-odds.

- **CS-54** (the stability) → executed 2026-11-06 → the archive's §7.9 (the
  stability) and the track log's §CS-54. **The UPR null becomes a cost**: the
  τ = 0.75 filtered z arm is **0.6399** against the full arm's 0.7020
  (**−0.0621 [−0.0859, −0.0395]**) and τ = 0.90 is −0.1094, with the floor
  rising to **61.97 %** / 74.9 %. The readout is the deliverable: the presence
  cliff **63.3 %** (the 1 − e⁻¹ prediction), the **sign** cliff **54.1 %** (so
  **not one** single-pair link clears 0.75), the head retention **65.0 %**, and
  the median question's table keeping none of its links. The registered
  survival magnitude (10–15 %) **missed** at 6.79 %, above its 5 % falsifier.
  CS-36's verdict reproduces in kind (the filter is not a lever) and not in
  size (a ~0.007 null there, a −0.062 cost here). **The next change set is
  CS-55** (the per-label lexicon + register): the bar and the register analog
  must be stated before measuring.

- **CS-55** (the label side's own words) → executed 2026-11-06 → the archive's
  §7.9 (the label side's own words) and the track log's §CS-55. **Two
  registered branches fired.** The channel test's below-cosine branch: the raw
  count is **0.4602** against the cosine's 0.5238 (**−0.0636 [−0.1008,
  −0.0269]**), the idf and topic-only arms are at the cosine (nulls), the map
  bar holds (**−0.2417 [−0.2887, −0.1949]**) — and the register test's tower
  case fired: **topic-only beats the full channel by +0.0821 [+0.0448,
  +0.1185]**, so CS-45's "the register was the signal" **does not transfer**
  (the label-side register is the questionnaire's boilerplate, identical across
  questions; the blind gap is **−0.0078**, so the raw count reads the
  questionnaire's shape, not the relation). **The chain stops here: the next
  change set would be CS-56** (abstention), and it starts only on the
  reviewer's word.

- **CS-56** (abstention) → executed 2026-11-06 → the archive's §7.9
  (abstention) and the track log's §CS-56. **The registered falsifier fired:
  the abstention is a lever.** The neutral fill (the row's median) is
  **0.7616** against the floor's 0.7020 — **+0.0596 [+0.0291, +0.0927]**,
  above the +0.05 bar and in all three documents; the max fill loses
  (−0.0914). The silence **is** informative (77.80 % of the 518 floor pairs
  negative, 85 silent questions) and is still the wrong encoding: the floor
  buries the 115 floored positives (22.2 %) below every seen pair, and the
  pair-weighted AUC counts that error heavier than the negatives' gain. CS-37's
  null does **not** transfer — the "impossible" should not ship on the pair
  unit. **The next change set is CS-57** (question pairs): the round's last,
  a description with its registered predictions.

- **CS-57** (question pairs) → executed 2026-11-06 → the archive's §7.9
  (question pairs) and the track log's §CS-57. **The questionnaire's own
  relation is real and weak, and the adaptation is kept**: the co-occurring
  pairs' mean vocabulary Jaccard is **0.0958** against **0.0892**
  (**+0.0066 [+0.0027, +0.0109]**), **15 of the 20 closest pairs co-occur**
  (75.0 % against the 34.14 % base rate), and **57 questions (19.4 %)** have
  no word of their own — all three registered tests hold. There is no catalog
  to be the expected source, so a positive relation is the finding, unlike
  CS-38's taxonomy null; the premium is six times smaller because the
  questions' vocabularies are homogeneous. **The round CS-48 … CS-57 is
  complete**; its transferable findings are CS-50's z arm and CS-56's neutral
  fill (the floor should not ship on the pair unit).





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
