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
*(CS-1 re-derived them on 2026-10-23: every number about the **judged** set
reproduces digit-for-digit — the score distributions, the pair AUC 0.4680, the
rank percentiles 0.097 / 0.207 and the top-1 provenance 55 / 17 / 153. The
**control** row below does *not* reproduce: that ad-hoc script's control
construction was never recorded, and the control settled on 2026-10-23 is
deliberately rank-unbiased (a uniform draw over the unjudged cells), so its
cells score lower (mean 0.2436 vs 0.289 for arctic2) and sit at rank p50
0.5021 (≈ random) instead of 0.565. The harness numbers live in
`results/omct-analyze-*/analysis.json` and in
`docs/experiments_text_processing.md`.)*

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
| top-1 same-document candidate per paragraph (225 rows) | given positive 55, given negative 17, **unjudged 153** — ⚠ *model-dependent row*: this is arctic2's ranking intersected with the file's coverage, **not a property of the dataset**; it says "our model's top choice for 153 paragraphs has no record", never "153 paragraphs have no match". The dataset-only counterpart is in §5 (70/225 paragraphs carry no positive record) |

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

- [x] **0.1 The pair-score artifact (`analyze-omct`).** A new CLI command that
  writes ONE artifact per run with: per-pair scores (paragraph x question) for
  the selected models, the **control set** (sampled unjudged same-document
  pairs, recorded with their sampling seed and the exclusion of all judged
  pairs), per-document strata with base rates, the negative-enrichment
  measurement (rank percentile of verified negatives vs positives vs control),
  and the score distributions per label. Warm cache, no new compute beyond
  cosine; the artifact is the *only* thing Stage 0's claims may cite.
  (The existing `omct-probe2-*` artifacts hold no per-pair scores, so this
  cannot be derived from them — hence the command.)
  **Models: `embeddinggemma` + `voyage-4-nano` + `arctic2`** (settled
  2026-10-23: the two Stage-2 axis models plus arctic2, whose full-text
  `strip_block` scores produced §0.1's indicative numbers, so the artifact
  re-derives them on the same footing; all three are warm in the cache).
  **Control set — settled design (2026-10-23):**
  - *Why it exists:* the reviewed set is **rank-biased** — it is the pairs one
    model (exp18) put on top, human-reviewed (§0.1 item 1) — so it is not a
    sample of the grid and carries no "typical same-document pair". The
    control is the **only rank-unbiased sample of the grid**: the one part
    nobody's ranking chose. That is what makes it the reference for "what
    does an ordinary same-document pair look like, and what does a method's
    score inflation look like?" (§0.1 item 3).
  - *What it is:* a seeded sample of the **unjudged** same-document cells
    (the pairs with no record in `omct_pairs.json`), **not** a set of
    negatives. Terms: subset **A** = the 996 reviewer-verified hard
    negatives (ground truth, hard by construction); subset **C** = the 524
    verified positives; subset **B** = the control sample — "unjudged" or,
    where a label-shaped number is unavoidable, "presumed-negative", always
    flagged as such.
  - *How it is built (pure, deterministic):* build the same-document grid →
    drop every judged cell → group the rest by document → draw a **seeded,
    proportional** sample per document (`numpy.random.default_rng(seed)`,
    seed from the config). Same data file + same seed → the same cells.
    `omct_pairs.json` is never modified; the control is *derived*.
  - *Size:* **n = 4,000** (config `analysis.control_size`), proportional →
    NU ~1,618 / CIDH ~2,210 / AI ~171. The AI stratum is the binding
    constraint (SE on its mean ≈ 0.008); 8,000 is a one-line change if that
    stratum proves too thin. The artifact records the seed, the sampling
    rule, the per-document quotas and the sampled cell ids, so any later
    stage re-uses exactly the same cells.
  - *Binding rules:* **never pooled with subset A**; every reported number
    names its subset; label-free wherever possible (distributions, rank
    positions, the scale check need no label at all); **every method number
    comes with its control-side counterpart** — no AUC on the reviewed set is
    quoted alone, and no absolute score is compared across methods without
    the control-side shift beside it; comparisons stay **paired**.
  - *Reporting rule that follows:* for each method, the change in AUC on the
    reviewed set (discrimination) sits next to the change in score level and
    rank position on the control (scale / mechanical gain). A method counts
    as a win only if the discrimination gain survives the scale shift — the
    cheap operational version: subtract the control's scale shift from the
    method's reviewed-set scores and recompute the AUC; if the advantage
    survives, it is a real change in ranking. (A uniform additive shift
    cannot change an AUC, so an AUC gain on the reviewed set is never a pure
    rescaling — but the *score level* is, and that is what the control
    measures.)
  - *Caveat on the correction:* the inflation is **not uniform**. A longer
    paragraph yields more chunks and is inflated more, so the control's scale
    shift is partly a *length* effect and partly a *document* effect (NU
    paragraphs are much longer than AI's, §5). A single global shift is
    therefore only an approximation: report the **per-document** shift and
    correct per document, and measure the correlation between paragraph
    length and per-pair shift rather than assuming it away.
  *(done 2026-10-23, CS-1: the `analyze-omct` command, the `analysis:` config
  section, the pure core `text_processors/control.py` (the grid, the quota
  allocation, the seeded draw, the rank percentile), the driver
  `omct_analysis.py`, the `AnalyzeOmctCell` sweep cell, the track's table
  columns and the track's first sweep (`experiments_text_processing/
  omct_analysis_sweep.py`, three cells = three models = three runs:
  `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}`; 0 re-encodes — every
  vector a cache hit). Every §0.1 number about the **judged** set reproduces
  digit-for-digit (the score distributions, the pair AUC 0.4680, the rank
  percentiles 0.097 / 0.207, the top-1 provenance 55 / 17 / 153), and the
  artifact's pair AUCs reproduce the Phase-2 probe's AUCs for the same three
  models; the §0.1 **control** row does *not* reproduce (that ad-hoc script's
  control construction was never recorded, and the settled control is
  deliberately rank-unbiased) — see the track log. The rank percentile uses
  the numbers' convention (0 = top), not §0.1's parenthetical.)*
- [x] **0.2 The written failure-mode classification.** Into
  `docs/experiments_text_processing.md`: representation vs. aggregation vs.
  label-incompleteness vs. benchmark-construction, with the §0.1 numbers
  re-derived through the harness; the "what this data can and cannot answer"
  section; and the stage-1 entry point it implies. The label anomaly is
  recorded as a **derived, quantified property of the hard-negative
  construction** (§0.1 items 1-2) — documented, designed-around, never
  "fixed".
  *(done 2026-10-24, CS-2: the classification is the track log's "Stage 0.2 —
  the failure-mode classification" section, built only on the three
  `omct-analyze-*` artifacts plus numbers **derived** from their own
  per-cell records (each derivation written out next to the number).
  Verdict: **benchmark construction supported and dominant** — the reviewed
  set is rank-selected (the control's top-fifth share 0.19–0.20 against the
  verified negatives' 0.43–0.50) and all three models sit within ~2.4
  unpaired SE of chance on it (AUC 0.4680 / 0.5004 / 0.5371); **label
  incompleteness supported on the positive side** (70/225 paragraphs carry
  no positive record; the unjudged cells' positive rate stays unmeasured —
  no human labelling, so the safe-negative rule ships as the margin half
  only); **representation plausible but not separable** (a measured floor —
  5.6–8.6% of the control's ordinary cells outscore the median true match —
  with an untested mechanism, n = 3); **aggregation unmeasured** (full text
  only, no harness number). The write-up also states the limits (no "153
  paragraphs have no match", no cross-model score comparison, no UPR claim)
  and the stage-1 entry point: the paragraph side first, measured against the
  control rather than against the AUC. One §0.1 claim is still *not*
  harness-re-derived — the ad-hoc chunking-sensitivity table — so it stays
  indicative and the aggregation axis is left to Stage 1/2; the Stage-0 exit
  criteria are met for everything CS-1 measured.*
- Exit: every claim in §0.1 re-derived from a `results/<run_id>/` artifact and
  logged; the two negative subsets (verified / unjudged-proxy) are defined and
  implemented for Stage 3; the control set exists as a reusable artifact.

### Stage 1 — Structural decomposition

- [x] **1.1a The paragraph side's structure, measured against the control.**
  The paragraph is split into sentence/clause atoms (`atoms.py`), every atom
  is embedded, and the atom-level scores are folded into one pair score by an
  aggregation rule (`max`, `top_k_mean:k`, `mean` — the new pure module
  `text_processors/aggregation.py`). The question side stays **unchunked**;
  the free `keep` vs `strip_block` question-transform comparison runs as one
  more arm. Every arm is compared with the **full-text baseline arm in the
  same run** (so the pairing is by construction) by the **paired bootstrap of
  the AUC difference** (2,000 resamples, seed recorded in the artifact) and
  by the **control-side counterpart** (the control's score level and rank
  position on the same 4,000 cells), plus the **de-shifted AUC** (the
  control's per-document mean shift subtracted from the method's reviewed-set
  scores). All three Stage-0 models, so the arms sit next to the Stage-0
  artifacts.
  *(done 2026-10-25, CS-3: the `analyze-omct` command now scores the
  baseline arm and the structural/question arms in one run (the `analysis:`
  config section gained the chunker spec, the aggregation list, the
  question-transform list and the paired-bootstrap params; the new
  `ANALYZE_OMCT_ARMS_COLUMNS` table shape lands in its own sweep section, so
  the Stage-0 rows are untouched); three runs
  `omct-analyze-{arctic2,voyage-4-nano,embeddinggemma}-atoms` (1,742 atom
  vectors + 294 `keep`-question vectors per model, 0 for the baseline arm —
  the warm cache stayed valid). The baseline arm reproduces the Stage-0
  artifacts field by field (same control cell ids, same per-cell scores, same
  AUCs and control means). **Verdict: the falsification test does not
  fire** — the floor gap (the verified positives' mean minus the control's
  mean) narrows in 11 of the 12 atom arms (count corrected 2026-10-26, CS-4:
  four aggregation rules × three models = twelve arms, not nine; see the track
  log's correction entry), none of the twelve has a 95% interval
  that excludes zero on the positive side, and the five that do exclude zero
  are all negative. The paragraph side's internal structure is **not** the
  lever that lowers the floor: the atoms mostly re-scale the scores. The
  question-side transform is model-dependent (the Phase-2 probe's "strip
  6/6" does not reproduce on the paired footing for voyage-4-nano). The
  numbers are in the track log's Stage-1.1a section.)*
- [x] **1.1b The question-type features → Stage 3's feature pack.** The
  answer options are scales ("Very often", "Yes", "Generally, yes"), so the
  decomposition is: (a) **stem-only vs. full question** (does the option
  block help or hurt the topical signal?) — *measured at Stage 1.1a: the
  effect is model-dependent, see the log* — and (b) **question-type
  features** (frequency? yes/no? "no data" option present? option count?) as
  label-free features, instead of exp02's "one embedding variant per answer
  option", which mostly injects scale words. **Moved to Stage 3**: they are
  feature-pack material (a combiner's inputs), not a scoring method, and
  Stage 3's leakage rules (no corpus-level question statistics) are where
  they belong.
  *(done 2026-10-26, CS-5: the question-type features are part of the v0
  feature pack — the options present, a "no data" option, a yes/no question,
  a frequency question and the option count — together with the question's
  decomposition into its topical **stem** and its typed **options**: the
  stem carries the lexical-overlap features, the options carry the coverage
  and the type one-hots. The pack sees the question's own text only, so no
  question-level *statistic* enters the features; the combiner's weight on a
  type feature is a fit-side statistic and is recorded in the artifact.)*
- [x] **1.2 Report on both negative subsets** (verified hard negatives and the
  unjudged-proxy control), pooled **and** per-document (the pooled number is
  dominated by the NU document: 845/1520 pairs, 25% positive; the AI document
  is 99 pairs at 80% positive — a per-document macro average is reported next
  to every pooled number).
  *(the CS-3 artifact already reports both subsets, pooled and per-document,
  with the per-document macro beside the pooled number — ticked at the
  reviewer's word, one checkbox per change set; the CS-4 runs keep the same
  reporting and add the within-document / cross-document split of the pooled
  AUC)*
- Exit: measured contribution of structure alone (no learned model yet), with
  the per-document breakdown and both negative subsets.

### Stage 2 — Segmentation ladder (the headline comparison)

- [x] **2.1 Fixed windows** (`word_windows`: N in {15, 25, 40}, stride ~N/2) vs
  [x] **2.2 sentence windows** (2 sentences, step 1) vs [x] **2.3 semantic
  boundaries** (`semantic.py`: cut where adjacent-atom similarity drops below
  threshold; threshold swept on fit-side documents only) — each aggregated by
  max / top-K-with-decay (exp07) / local+global α (exp06, α swept); reported
  on all pairs (descriptive) with per-document breakdowns. Both
  reference-lineage methods and our own variants are in scope here.
  *(done 2026-10-26, CS-4: the whole ladder ran — 7 variants × 3 models = 21
  runs `omct-analyze-<model>-<variant>` (the reviewer chose the whole ladder
  over a reduced grid). New pure chunkers `word_windows.py` (exp03),
  `sentence_windows.py` (exp02) and `semantic.py` (exp18(ii), the Embedder
  port injected; the CLI wires the cached backend in); new aggregation rules
  `decayed_top_k:k:decay:spread` (exp07), `local_global:alpha:<local rule>`
  (exp06, needs the full-text score as a second input) and `peak:k:margin`
  (exp08, a margin instead of the lineage's absolute bars — the track forbids
  absolute cuts). The semantic threshold is swept on the **fit side only**
  through the document-level cross-fit: every document's threshold comes
  from the rotation that holds it out. The numbers are in the track log's
  Stage-2 sections; the guard holds (all 21 runs reproduce the Stage-0
  baseline arm field by field).)*
- [x] **2.4 Model axis: `embeddinggemma` (Ollama, 768d, template prefixes) and
  `voyage-4-nano` (HF, 2048d, revision `433ece5e…`, retrieval prompts)** —
  the whole ladder runs on both, not just the winner. Both are UPR
  semifinalists and both sit mid-pack on OMCT pair-AUC (0.537 / 0.500 vs the
  pool's best 0.599), so the ladder is measured where the signal is *not*
  strongest: the control set and the paired tests carry the weight.
  Chunk side/kind assignment: a paragraph chunk is encoded with the
  `document` prefix, a question chunk with the `query` prefix (embeddinggemma's
  are `{content}` templates — the existing `apply_prefix` handles them).
  *(done 2026-10-26, CS-4: the ladder ran on **all three** Stage-0 models
  (`embeddinggemma`, `voyage-4-nano`, `arctic2` — the reviewer's call, so the
  n = 3 argument of Stage 0/1 keeps holding and the new arms sit next to the
  Stage-0 and Stage-1 artifacts). The symmetric variants chunk the question
  side too: the question's chunks are encoded with the `query` prefix, the
  paragraph's with `document`.)*
- [x] **2.5 Paired statistics.** Every ladder comparison is paired (same
  pairs, same model, different segmentation), so report the **paired bootstrap
  CI of the AUC difference** (or DeLong) — never two independent AUCs. The
  measured effect size (~±0.03 AUC) sits at the unpaired resolution limit
  (SE ≈ 0.02 at n=1520), so an unpaired comparison cannot settle it. Also
  report the max-over-k inflation: a chunked method takes a max over more
  candidates, so its score scale is not comparable to full text's — the
  control set makes that visible.
  *(done 2026-10-26, CS-4: paired throughout (every comparison is inside one
  run, against the full-text baseline arm on the same cells; the paired
  bootstrap, 2,000 resamples, seed 42). **The pooled AUC turned out to be the
  wrong headline**, so the harness now also carries the **within-document /
  cross-document split** of the pooled AUC (`grouped_auc_decomposition`:
  58.7% of the positive-negative pairs are cross-document, so a method that
  changes a document's score level moves the pooled number without changing
  the ranking inside a document) and a **paired bootstrap of the
  within-document AUC difference** (stratified by document and class). The
  max-over-k inflation is reported as the control's mean shift beside the
  chunks-per-text: it grows with the chunk count (+0.054 at 9.4 chunks per
  paragraph against +0.008 at 1.8).)*
- Exit: a ranked, artifact-backed statement of which segmentation+aggregation
  wins where — with paired CIs, per-document breakdown, and both negative
  subsets — or "no chunking beats full text" (an honest result too).
  *(met 2026-10-26, CS-4: the ranked statement is in the track log's
  "Stage 2 — the verdict" section. It is **not** "no chunking beats full
  text": chunking the *paragraph* side with a peak-like aggregation improves
  the **within-document** AUC by +0.02–0.04 on two of the three models
  (24/42 and 28/42 arms with a significant positive interval), while chunking
  the *question* side hurts on all three; the effect does not transfer across
  embedders (voyage-4-nano gains nothing), and the largest **pooled** gain in
  the ladder is a cross-document level effect, not a ranking gain.)*

### Stage 3 — Learned pair scorer (the "get crazy" slot)

- [x] **3.1 Feature pack v0 + LR baseline**, [x] **3.1b the feature-group
  ablation** (the v0 pack's four blocks — the embedding, the lexical, the
  question-part and the question-type block — measured one at a time: which
  block carries the gain, and which block costs the hit@1), [x] **3.2
  gradient-boosted combiner** (sklearn `HistGradientBoostingClassifier`;
  `lightgbm` only if it beats HistGB and the dependency is approved), [x]
  **3.3 chunk-profile features** (best chunk matches, spread/peak features —
  exp17/15/16 lineage) folded in.
  *(3.1 done 2026-10-26, CS-5: the v0 feature pack (15 features — the
  global and best chunk cosines, the top-K chunk similarities and their
  spread, the soft lexical overlap and the char n-grams, the question-part
  coverage and the question-type one-hots; **no length feature**), the
  sklearn logistic-regression combiner behind the ``Combiner`` port, and the
  **document-level holdout** harness (three rotations, one out-of-fold score
  matrix, the label-permutation floors). Three runs
  `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3`. **Verdict:**
  the learned arm beats the full-text cosine on the within-document AUC on
  every model and every held-out document (+0.1023 / +0.1485 / +0.1452, all
  95% intervals excluding zero); the chance floor lands at chance, but a
  **control-role floor** (no label information, the control cells still
  treated as negatives) reaches 0.557–0.579 — *above* the plain cosine —
  so ~85% of the arm's gain is label-driven and the rest is the training
  design's own signal. Caveat: the arm's **hit@1 is worse** than the
  cosine's (the learned pair ranker is not a better retriever). The numbers
  are in the track log's Stage-3.1 section.)*
  *(3.1b done 2026-10-26, CS-6a: the **feature groups** are a config knob
  (`analysis.stage3.features.groups`) and the pack keeps only the selected
  groups' columns; 24 runs (8 group sets × 3 models,
  `omct-analyze-<model>-stage3-<groupset>`), the same harness, the same three
  rotations, the same two floors. **Verdict: the gain is the lexical block's,
  not the embedding geometry's and not the question-type prior's.** Removing
  the question-type block costs 0.001–0.005 of the within-document AUC
  (0.6655 → 0.6632, 0.6723 → 0.6670, 0.6472 → 0.6462); removing the lexical
  block costs 0.047–0.113; removing the embedding block costs *nothing*
  (0.6589 on all three models — and the three arms that use no embedding at
  all give identical numbers on all three models, which is itself the check
  that the runs share the data and the seeds). The **hit@1 loss is the lexical
  block's**: dropping it restores the cosine's hit@1 (0.1279 vs the cosine's
  0.1355 on embeddinggemma; 0.1050 = the cosine's 0.1050 on voyage-4-nano),
  while dropping the question-type block moves it by ≤0.006. The
  question-type-only arm is a question-level prior (its features never look
  at the paragraph): it reaches only 0.5713 and it is the worst retriever of
  all (hit@1 0.0076 against the cosine's 0.1355) — the hypothesis is right
  about the *mechanism* and wrong about the *size*. The numbers are in the
  track log's Stage-3.1b section.)*
  *(3.2 done 2026-10-26, CS-6b: the **gradient-boosted combiner** (sklearn's
  `HistGradientBoostingClassifier`) behind the same ``Combiner`` port, with
  the feature-group selection as the knob (so the ablation's arms compose
  with the learner): 9 runs (3 models × {`all`, `lex-only`, `embed-only`},
  `omct-analyze-<model>-stage3-histgb-<groupset>`), the same harness, the same
  three rotations, the same two floors; the hyperparameters are scikit-learn's
  defaults, explicitly pinned, with **no** grid (a fit-side selection was
  already measured to be unidentifiable here — the Stage-2 semantic
  threshold). The artifact also gained the **importances capability**: a tree
  ensemble has no coefficients, so the trees' split usage (per feature: the
  splits, the gain, the depth) is recorded. **Verdict: the learner is not the
  lever.** The booster is *below* the logistic regression on 8 of the 9 arms
  (−0.022 to −0.049 of the within-document AUC; the ninth is a tie to four
  decimals) and ahead in only 5 of the 27 per-rotation paired comparisons, so
  the bar the reviewer set (beat a 3-feature logistic at 0.6461) is not met;
  its hit@1 is worse on 8 of the 9 arms. The two learners' *label-driven*
  parts (the excess over their own control-role floor) are a wash — the
  logistic's advantage comes mostly from a **higher control-role floor**,
  i.e. it is the better learner of the training design's own label-free
  signal. The booster's gain is also diffuse (the top feature takes 16–17%
  of the trees' gain) and it leans harder on the embedding block (43–52% of
  the gain) — the block the ablation showed carries no label-driven signal.
  So `lightgbm` stays deferred and no new dependency is proposed. The
  numbers are in the track log's Stage-3.2 section.)*
  *(3.3 done 2026-10-26, CS-6c: the **chunk profile** as the pack's fifth
  group (`chunk_profile`, 10 features), the four existing groups untouched
  (so the CS-6a arms still describe the same feature sets). The block holds
  exp17's **ranked cross-chunk match profile** (both sides cut into sentence
  units and 5-word pieces, every piece×piece cosine sorted, the top three
  ranks of each granularity kept) and exp16's **alignment kinds** (which
  granularity produced the best alignment, and whether the two aligned units
  share a content word) — the taxonomy the reviewer chose on 2026-10-26.
  exp16's `log(1+df)` was **dropped** (a corpus-level statistic; the pack is
  pair-local by contract). 9 runs (3 models × {`base` = the v0 pack (the
  control), `full` = v0 + the profile, `profile` = the block alone},
  `omct-analyze-<model>-stage3-cp-<groupset>`), the same harness, the same
  three rotations, the same two floors. The artifact also gained the
  **per-feature diagnostics** (every feature's own within-document AUC, its
  control-side counterpart and its length correlation), the **length check**
  (the arm's length ↔ score correlation beside the cosine's, plus the length
  ↔ rank-shift correlation — the §5 open item about the shift) and the arms'
  **per-pair out-of-fold scores** (which make a paired per-pair comparison
  between two arms of different runs possible: the new tool
  `tools/compare_stage3_arms.py`). **Verdict: the chunk profile is not the
  lever either.** The block alone is indistinguishable from the full-text
  cosine (all three intervals include zero); in the full pack it buys a
  small gain that is established on one of the three models only (+0.0172 on
  voyage-4-nano, the interval excluding zero; +0.0066 and −0.0049 elsewhere,
  both including zero; 7 of the 9 per-rotation paired comparisons favour
  it). Its best feature re-measures the v0 pack's best chunk cosine at a
  finer granularity (0.5642 against 0.6049 on embeddinggemma) and its lower
  ranks are **length proxies** (the length correlation grows to +0.43…+0.56
  with the rank's depth), and it *raises the chance floor* above chance
  (0.514–0.534 against the v0 pack's 0.494–0.505) — a label-free signal. The
  full arm is also a slightly worse retriever (the positives' hit@1 0.0802 /
  0.0763 / 0.0649 against the v0 pack's 0.0897 / 0.0821 / 0.0859). The
  numbers are in the track log's Stage-3.3 section.)*
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
- Question `question_text_sources` / document idiosyncrasies: characterized in
  stage 0, not assumed.
- `lightgbm`: a new dependency is only proposed if sklearn's HistGB is beaten
  (AGENTS §3 explain-first).
- **The "same-report direction" mechanism (CS-2, 2026-10-24): a hypothesis,
  not a measurement.** The Stage-0 artifact holds scores, rank positions and
  distributions, not the embedding space, so the plan's candidate mechanism
  for the measured floor (ordinary same-document cells averaging
  0.2194–0.2436, with 5.6–8.6% of them above the median verified positive)
  cannot be tested by it. Settling it needs a new measurement in
  `analyze-omct` (e.g. the share of the similarity direction shared by all
  paragraphs of one document) — a separate change set, with the reviewer's
  approval, before the mechanism is called more than a hypothesis.
- **The aggregation axis now has harness numbers for the whole ladder
  (CS-4, 2026-10-26).** Stage 2 ran the complete ladder (7 variants × 3
  models, 126 chunked arms, paired throughout): the fixed word windows, the
  sentence windows, the semantic boundaries and the two symmetric variants,
  under `max`, `top_k_mean`, exp07's decayed average, exp06's local+global
  blend and exp08's peak shortcut. The paragraph-side **windows** with a
  peak-like aggregation improve the **within-document** AUC by +0.02–0.04 on
  two of the three models; the question side's chunking hurts on all three;
  the semantic boundaries do not transfer (the fit-side threshold choice does
  not predict the holdout's). §0.1's ad-hoc table is still **not**
  re-derived — it measured a different thing (unpaired, no control, and it
  pooled the documents' levels), so the ad-hoc numbers stay indicative.
- **The pooled AUC on the reviewed set is structurally sensitive to the
  documents' score levels (CS-4, 2026-10-26).** 58.7% of the positive-negative
  pairs of the pooled AUC are **cross-document** (41.3% are within a
  document), so a method that changes a document's score *level* moves the
  pooled number without changing the ranking inside a document. The
  measurement is now in the artifacts (`pair_auc_decomposition`) and the
  ladder's verdict rests on the within-document statistic. **Open:** should
  the track's headline statistic be the within-document AUC from now on (the
  reviewer's call)? The same decomposition is available for the Stage-0
  artifacts, so the earlier entries could be re-read on that footing — but
  only if the reviewer wants the log's earlier numbers re-stated.
- **The paragraph-side window gain does not transfer across embedders
  (CS-4, 2026-10-26).** Two of the three models gain +0.02–0.04
  within-document AUC; voyage-4-nano gains nothing (3 of 42 arms positive
  and significant against 12 negative). **Open:** is the gain a property of
  the *chunking* or of the *embedding geometry* of those two models? The
  Stage-0 "same-report direction" mechanism (still a hypothesis) would
  explain it: a model whose document direction dominates has less to gain
  from separating the paragraph's parts. Settling it needs the embedding-
  space measurement the plan already lists as an open change set.
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
- **The question-side transform's effect is model-dependent (CS-3,
  2026-10-25).** The Phase-2 probe found that stripping the "Answer
  choices:" block raises the pair AUC for 6 of 6 embedders. On the Stage-1.1a
  footing (the same 1,520 pairs, a paired bootstrap, the control side beside
  it) the sign flips for voyage-4-nano (+0.0170 [−0.0056, +0.0392] for
  `keep` against `strip_block`) while embeddinggemma (−0.0133) and arctic2
  (−0.0167) prefer the stripped question. **Open:** is the block's value
  model-specific (a prefix / instruction effect) or is the Phase-2 probe's
  6/6 an artefact of the unpaired comparison? It changes what Stage 2 and
  Stage 3 assume about the question side, so it must not be settled by one
  more model.
- **The length↔shift correlation the Stage-0 caveat asks for is now
  measurable (CS-5, 2026-10-26).** The command records the paragraph lengths
  (per document and per paragraph) and the **length ↔ score correlation** of
  the control cells, so the caveat's question can be answered: for the
  baseline arm the Pearson correlation between a control cell's paragraph
  length and its score is +0.309 (NU), +0.259 (AI) and +0.061 (CIDH) on
  embeddinggemma — the ordinary cells' score already travels with the
  length in two documents and hardly at all in the third, so the inflation
  is document-dependent. The *shift* (a method's per-cell change against the
  baseline) still needs a second arm and is a later change set. Note: the
  figures this question was written with ("AI median 364 chars vs NU 863")
  were wrong — the artifact says AI 331 / NU 375 / CIDH 514 (see the track
  log's correction entry).
- **The training design's own signal is stronger than the cosine
  (CS-5, 2026-10-26).** A model that knows **nothing** about the labels but
  is told "the ordinary unjudged same-document cells are negative" reaches a
  within-document AUC of 0.557–0.579 on the reviewed set — *above* the
  full-text cosine's 0.502–0.563. The reviewed set is rank-selected, so
  "control-like" is itself a predictor of the label. **Open:** what does
  this mean for the durable deliverable? A "which unjudged pairs are safe
  negatives" rule built on such a model is partly self-fulfilling (the
  model learns the construction of the reviewed set, not the topic), so the
  rule's margin has to be measured against the **control-role floor**, not
  against the cosine. The margin half of the rule is unchanged; the
  measurement around it is not.
  *(CS-6a adds the decomposition: the label-driven part of the arm's gain is
  the **lexical** block's — the embedding-only arm is *at* its own
  control-role floor (0.6006 against 0.5940 on embeddinggemma; 0.5126
  against 0.5259 on arctic2, i.e. below it), while the lexical-only arm is
  well above its (0.6461 against 0.5632). So the embedding geometry adds
  no label-driven signal beyond the training design's own prior, and the
  floor is the right yardstick for the deliverable.)*
- **The learned pair ranker is not a better retriever — and the loss is the
  lexical block's (CS-5 asked; CS-6a answered, 2026-10-26).** The Stage-3.1
  arm beats the cosine on the within-document AUC but loses on the hit@k (the
  positives' hit@1 0.090 vs 0.136 on embeddinggemma). A bag-of-words
  overlap ranks a *pair* better than an embedding cosine does, but it is a
  worse first guess for a paragraph. The feature-group ablation settles
  *which* block pays for it: dropping the **lexical** block restores the
  cosine's hit@1 (0.1279 against the cosine's 0.1355 on embeddinggemma;
  0.1050 = the cosine's 0.1050 on voyage-4-nano), while dropping the
  **question-type** block moves it by ≤0.006. The question-type-only arm
  confirms the *mechanism* (a question-level prior reorders a paragraph's
  candidates without looking at the paragraph: its hit@1 is 0.0076, the
  worst of all the arms) but not the *size* (the question-type block is worth
  0.001–0.005 of the within-document AUC). **Open:** the lexical signal is a
  bag-of-words overlap over a 294-question corpus — does it transfer to a
  corpus with a different vocabulary, and is a *retrieval* score (the
  cosine's geometry, with the learned features only re-ranking a shortlist)
  the better shape for the deliverable?
- **The learner is not the lever; the features are (CS-6b, 2026-10-26).** A
  gradient-boosted ensemble (sklearn's `HistGradientBoostingClassifier`, the
  defaults, no grid) is *below* the logistic regression on 8 of the 9 arms
  (−0.022 to −0.049 of the within-document AUC) and ahead in 5 of the 27
  per-rotation paired comparisons, and it is the worse retriever on 8 of the
  9 arms. The two learners' label-driven parts — the excess over their own
  control-role floor — are a wash; the logistic's edge comes mostly from a
  **higher control-role floor** (it is the better learner of the training
  design's own label-free signal). The booster's trees also lean harder on
  the embedding block (43–52% of their gain), the block the ablation showed
  carries no label-driven signal. **Open:** is the *learner's* shape relevant
  at all here, or is the ceiling set by the features and the corpus (a
  294-question vocabulary)? A different question is now open for the
  deliverable: the track's rule is that a safe-negative rule is measured
  against the control-role floor, and the floor itself depends on the
  learner — so which learner should the shipped rule use? (The logistic
  regression, on this evidence: it is the better retriever and its floor is
  the higher one.)
- **The two learners' arms live in two runs, and the artifacts carry the
  per-rotation AUCs, not the per-pair scores (CS-6b asked; CS-6c answered,
  2026-10-26).** So the paired evidence between the two arms is the per-rotation
  differences (the same held-out document, the same pairs) plus each arm's own
  paired interval against the common reference (the cosine). **Answered:** the
  Stage-3 block now carries each arm's **per-pair out-of-fold scores**
  (`out_of_fold_scores`), and the repo tool
  `tools/compare_stage3_arms.py` reads them: a paired per-pair bootstrap of
  the within-document AUC difference between two arms of different runs
  (stratified by document and class, 2,000 resamples, seed 42). The CS-6c runs
  use it for the chunk profile's arms. **Still open:** the *older* Stage-3
  runs (the CS-5/CS-6a/CS-6b ones) predate the block, so a paired CI between
  two of *those* arms would need them re-run — only worth it if a later change
  set needs it.
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

## 6. Milestones

- T1: stages 0-1 (diagnosis + structure) → what matters on OMCT decided.
- T2: stage 2 (segmentation ladder) → the chunking verdict. *(reached
  2026-10-26, CS-4: chunk the paragraph, keep the peak, never chunk the
  question — and the pooled AUC is the wrong headline on this benchmark.)*
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
- [x] **CS-1 — `analyze-omct`.** The Stage-0 artifact command: per-pair scores for
  the selected models, the control set (seeded sampling, judged pairs
  excluded), per-document strata with base rates, the negative-enrichment
  measurement, per-label score distributions. Warm cache, no re-encoding.
  *(done 2026-10-23: the command, the `analysis:` config section, the pure
  core `text_processors/control.py`, the driver `omct_analysis.py`, the
  `AnalyzeOmctCell` sweep cell, the track's table columns, the track's first
  sweep and three runs — `omct-analyze-embeddinggemma`,
  `omct-analyze-voyage-4-nano`, `omct-analyze-arctic2` — with the numbers in
  the track log. The control is a seeded (42), proportional draw of 4,000
  unjudged same-document cells (NU 1,619 / CIDH 2,210 / AI 171), identical
  across the three runs; the artifact records the seed, the rule, the quotas,
  the excluded judged cells and the drawn cell ids.)*
- [x] **CS-2 — Stage 0 write-up.** The failure-mode classification and the
  "what this data can answer" section in `docs/experiments_text_processing.md`,
  based on the CS-1 artifact (this is where §0.1's diagnostics become
  reportable numbers).
  *(done 2026-10-24: the write-up is the track log's "Stage 0.2 — the
  failure-mode classification" section. Every number comes from the three
  `omct-analyze-*` artifacts or is derived from their per-cell records with
  the derivation stated; the two §0.1 facts CS-1 did not re-derive (the
  ad-hoc chunking table, the corpus shape) are quoted only where they are
  flagged as indicative, never as results; no number is computed from the
  raw pair file. Plan checkboxes 0.2 and CS-2 ticked; Stage 1 is next, and
  its entry point (paragraph side first, control-side measurement) is
  recorded in the write-up and as an open question in §5.)*
- [x] **CS-3 — Stage 1.1a: the paragraph side's atoms.** The arms in the
  `analyze-omct` command (the baseline arm plus the structural and
  question-transform arms, one run, paired by construction), the paired
  bootstrap of the AUC difference, the control-side counterpart, the
  de-shifted AUC, the hit@k report, the `analysis:` config schema (the
  chunker spec, the aggregation list, the question-transform list, the
  bootstrap params), the new pure module `text_processors/aggregation.py`,
  the `ANALYZE_OMCT_ARMS_COLUMNS` table shape and the track's Stage-1 sweep
  (`experiments_text_processing/stage1_atoms_sweep.py`).
  *(done 2026-10-25: three runs
  `omct-analyze-{arctic2,voyage-4-nano,embeddinggemma}-atoms`; the baseline
  arm reproduces the Stage-0 artifacts field by field (the same control cell
  ids, the same per-cell scores, the same AUCs and control means);
  **verdict: the falsification test does not fire** — the floor gap narrows
  in 11 of the 12 atom arms (count corrected 2026-10-26, CS-4 — four
  aggregation rules × three models, not nine arms), no arm's 95% interval
  excludes zero on the
  positive side, and the five that do exclude zero are all negative, so the
  paragraph side's structure mostly re-scales the scores instead of lowering
  the floor. The question-side transform is model-dependent. Plan checkbox
  1.1a ticked; 1.1b moved to Stage 3; two new open questions in §5; the
  numbers are in the track log's Stage-1.1a section.)*
- [x] **CS-4 — Stage 2: the segmentation ladder.** The whole ladder (the
  reviewer's call), on all three Stage-0 models: the new pure chunkers
  (`word_windows.py` exp03, `sentence_windows.py` exp02, `semantic.py`
  exp18(ii) with the injected `Embedder` port), the new aggregation rules
  (`decayed_top_k` exp07, `local_global` exp06, `peak` exp08), the swept
  semantic threshold with the **document-level cross-fit** (the threshold of a
  document is chosen on the other two documents — the plan's fit/holdout
  discipline), the symmetric variants (both sides chunked), the
  within-document / cross-document split of the pooled AUC with a paired
  bootstrap of the within-document difference, the `analysis:` config
  additions (`question_chunker`, `semantic_thresholds`, `semantic_selection`),
  the arms table shape's growth, the track's Stage-2 sweep
  (`experiments_text_processing/stage2_ladder_sweep.py`) and the repo tool
  `tools/verify_analysis_baseline.py` (the guard).
  *(done 2026-10-26: 21 runs `omct-analyze-<model>-<variant>` (7 variants × 3
  models, 126 chunked arms); the guard holds — all 27 runs (Stage 0, Stage 1,
  Stage 2) reproduce the Stage-0 baseline arm field by field, and the Stage-1
  arms are unchanged under the refactored driver. **Verdict:** chunking the
  *paragraph* side with a peak-like aggregation improves the within-document
  AUC by +0.02–0.04 on two of the three models; chunking the *question* side
  hurts on all three; the effect does not transfer across embedders; and the
  largest *pooled* gain in the ladder is a cross-document level effect, not a
  ranking gain. The Stage-2 checkboxes and 1.2 are ticked; the Stage-1
  atom-arm count is corrected in the plan and in an appended log entry.)*
- [x] **CS-5 — Stage 3.1: the feature pack + the holdout harness + the LR
  baseline.** The v0 feature pack (`text_processors/features/pair_features.py`,
  15 features, the ``Embedder`` **port** for the chunk-level ones — the same
  precedent the semantic chunker set), the sklearn logistic combiner
  (`text_processors/combining/logistic.py`, behind the ``Combiner`` port,
  which gained an optional ``sample_weight``), the document-level holdout
  harness in the driver (three rotations, one out-of-fold score matrix, the
  label-permutation floors), the `analysis.stage3` config section, the
  Stage-3 table shape, the paragraph-length capability, and the track's
  Stage-3 sweep (`experiments_text_processing/stage3_features_sweep.py`).
  *(done 2026-10-26: three runs `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3`;
  the guard holds (all 30 runs reproduce the Stage-0 baseline arm field by
  field). **Verdict:** the learned arm beats the full-text cosine on the
  within-document AUC on every model and every held-out document
  (+0.1023 / +0.1485 / +0.1452, the 95% intervals excluding zero); the
  chance floor lands at chance (0.494–0.505) as the plan requires, but the
  **control-role floor** (no label information, the control cells still
  treated as negatives) reaches 0.557–0.579 — above the plain cosine — so
  ~85% of the gain is label-driven and the rest is the training design's own
  signal. Caveat: the arm's hit@1 is *worse* than the cosine's. The numbers
  are in the track log's Stage-3.1 section; 367 unit tests green.)*
- [x] **CS-6a — Stage 3.1b: the feature-group ablation.** The **feature
  groups** as the ablation's unit (the v0 pack's four blocks: the embedding,
  the lexical, the question-part and the question-type block), the
  `analysis.stage3.features.groups` config knob (validated at config time:
  an unknown, an empty or a repeated group fails before any run), the pack's
  group selection (the pack computes every feature; the combiner sees only
  the selected groups' columns), the question-overlap diagnostic of the
  rotation (how many of the holdout's questions the fit side also judged),
  the Stage-3 table shape's growth (the groups, the hit@1, the
  within-document floors) and the track's ablation sweep
  (`experiments_text_processing/stage3_ablation_sweep.py`).
  *(done 2026-10-26: 24 runs `omct-analyze-<model>-stage3-<groupset>` (8
  group sets × 3 models); the `all` set reproduces the Stage-3.1 artifacts
  field by field; the guard holds (all 54 runs reproduce the Stage-0 baseline
  arm field by field). **Verdict: the gain is the lexical block's.** Removing
  the question-type block costs 0.001–0.005 of the within-document AUC;
  removing the lexical block costs 0.047–0.113; removing the embedding block
  costs nothing (0.6589 on all three models — the three embedding-free arms
  give identical numbers on all three models, which is the check that the
  runs share the data, the control cells and the seeds). The **hit@1 loss is
  the lexical block's**, not the question-type block's: dropping the lexical
  block restores the cosine's hit@1 (0.1279 vs 0.1355 on embeddinggemma,
  0.1050 = the cosine's on voyage-4-nano) while dropping the question-type
  block moves it by ≤0.006. The question-type-only arm (a question-level
  prior that never looks at the paragraph) reaches only 0.5713 and is the
  worst retriever of all (hit@1 0.0076 vs the cosine's 0.1355). The numbers
  are in the track log's Stage-3.1b section; 376 unit tests green.)*
- [x] **CS-6b — Stage 3.2: the gradient-boosted combiner.** sklearn's
  `HistGradientBoostingClassifier` behind the same `Combiner` port (the
  `combining/` registry's second entry); `lightgbm` stays deferred until
  HistGB is beaten (a new dependency needs the AGENTS.md §3 approval).
  *(done 2026-10-26: the new combiner
  `text_processors/combining/histgb.py` (`HistGbCombinerConfig` +
  `HistGbCombiner`) and the registry entry — the existing
  `analysis.stage3.combiner` knob validates the name and the parameters at
  config time, so **no config-schema change**; the hyperparameters are
  scikit-learn's defaults, explicitly pinned, with early stopping off (no
  internal validation split carved out of the fit side) and no
  standardization (the booster bins the features itself). The **importances
  capability**: a tree ensemble has no coefficients, so the artifact now
  carries the trees' split usage per rotation (the splits, the gain, the
  depth, the shares) — a *usage* diagnostic, not an accuracy-based importance.
  The sweep `experiments_text_processing/stage3_histgb_sweep.py`: 9 runs
  (3 models × {`all`, `lex-only`, `embed-only`}),
  `omct-analyze-<model>-stage3-histgb-<groupset>`. The guard holds (all 63
  runs reproduce the Stage-0 baseline arm field by field).
  **Verdict: the learner is not the lever** — the booster is below the
  logistic regression on 8 of the 9 arms and ahead in 5 of the 27 per-rotation
  paired comparisons, and it is the worse retriever on 8 of the 9 arms; the
  two learners' label-driven parts (the excess over their own control-role
  floor) are a wash, and the logistic's edge comes from a higher
  control-role floor. `lightgbm` stays deferred; no new dependency is
  proposed. The numbers are in the track log's Stage-3.2 section; 384 unit
  tests green.)*
- [x] **CS-6c — Stage 3.3: the chunk-profile features.** exp17's ranked
  cross-chunk match profile and exp16's alignment kinds folded into the pack
  as a fifth group (`chunk_profile`); exp16's corpus-level df dropped (the
  reviewer's decision).
  *(done 2026-10-26: the new pure module
  `text_processors/chunkers/units.py` (the profile's two granularities: the
  sentences and the 5-word pieces inside them) and the pack's fifth group
  (10 features: the six profile ranks and the four alignment kinds, with
  `analysis.stage3.features.profile_top_k` / `profile_ngram_size` as the new
  config knobs — validated at config time; the four existing groups keep
  their exact meaning, so the CS-6a arms still describe the same feature
  sets). Four new embedding sides (the units of each side, each with its own
  cache-version slot), four new pure-core feature functions, and the
  **per-feature diagnostics** + the **length check** + the arms' **per-pair
  out-of-fold scores** in the Stage-3 artifact (the last one with the new
  repo tool `tools/compare_stage3_arms.py`, which answers the CS-6b open
  question about a paired per-pair CI between two arms of different runs).
  The sweep `experiments_text_processing/stage3_chunk_profile_sweep.py`: 9
  runs (3 models × {`base` = the v0 pack (the control), `full` = v0 + the
  profile, `profile` = the block alone}),
  `omct-analyze-<model>-stage3-cp-<groupset>`. The guard holds (all 72 runs
  reproduce the Stage-0 baseline arm field by field).
  **Verdict: the chunk profile is not the lever either** — the block alone
  is indistinguishable from the full-text cosine (all three paired intervals
  include zero), in the full pack it buys a small gain that is established on
  one of the three models only (+0.0172 on voyage-4-nano, the interval
  excluding zero; +0.0066 and −0.0049 elsewhere, both including zero; 7 of
  the 9 per-rotation paired comparisons favour it); its best feature
  re-measures the v0 pack's best chunk cosine at a finer granularity
  (0.5642 against 0.6049 on embeddinggemma), its lower ranks are **length
  proxies** (the length correlation grows to +0.43…+0.56 with the rank's
  depth), it *raises the chance floor* above chance (0.514–0.534 against the
  v0 pack's 0.494–0.505 — a label-free signal), and the full arm is a
  slightly worse retriever (the positives' hit@1 0.0802 / 0.0763 / 0.0649
  against the v0 pack's 0.0897 / 0.0821 / 0.0859). The numbers are in the
  track log's Stage-3.3 section; 398 unit tests green.)*
- [ ] **CS-6d — the Stage-3 random-vector floor.** The second sanity floor
  of the plan's Stage 3 (the label-permutation one is done): the arm is
  fitted on random vectors instead of the model's embeddings, so a "win"
  can be checked against what the *training design* alone buys.
- [ ] **CS-6e — the Stage-3 reporting.** The calibration curves and the
  thresholds per document, and the safe-negative margin measured against the
  **control-role floor** (the reviewer's decision 2026-10-26: the track's
  durable deliverable is measured against that floor, not against the plain
  cosine; the rule still ships as the margin half only).

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
