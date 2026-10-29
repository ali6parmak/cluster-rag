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
