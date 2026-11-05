# Archive — the executed text-processing track plan (moved 2026-10-29)

**The split note (2026-10-29) — the reviewer's decision.** The track's plan
(`TEXT_PROCESSING_PLAN.md`) had grown to 3,247 lines because it holds ~30 change sets,
each pre-registered and then answered. It is split into a **live** part — the open work,
the rules, the deliverable — and **this archive**: the executed pre-registrations and the
questions they answered. The live plan stays a **forward** document.

**The moved text is byte-for-byte.** Nothing here is rewritten, compressed or summarised:
this archive *is* the pre-registration record, and a summary would be a second source of
truth that drifts. **This file is append-only**, like the logs: what is added later is added
at the end, and what is here is never edited.

**1. What moved, and where it went.**

| the live plan's | moved to | why |
|---|---|---|
| §0.1 the ad-hoc OMCT diagnostics | "The moved §0.1" below | superseded by CS-1, which re-derived every number through the harness |
| §3 Stages 0, 1, 2, 3, 3.5 (the full pre-registrations) | "The moved §3" below | executed — the live plan keeps a short stub per stage, pointing here and at the track log |
| §5 the **answered** open questions | "The moved §5" below | a closed question is history, not work |
| §7 CS-0 … CS-16 and the §7.1.1 / §7.1.2 / §7.1.3 pre-registrations | "The moved §7" below | executed — except **CS-10**, which was never executed (it needs the reviewer's explicit OK) |
| `PLAN.md`'s Phase-4 opening (the embedding-pool widening sweep) | "The moved PLAN.md Phase 4 opening" below | executed on 2026-10-20 |

**2. What stayed live, and why.** §0 (the track's context), §1 (the governance), §2 (the
naming and infrastructure contract), §3.0 (the methods inventory — the menu the stage
checkboxes order from), §4 (the cross-cutting rules), §6 (the milestones), §8 (the durable
deliverable), and the open parts of §3 (Stages 4 and 5), §5 (the questions still open) and
§7 (CS-17). **The live plan keeps the checkboxes**: the repo's rule "tick the checkbox in
PLAN.md" has to keep working, so every still-open task stays in the live plan with its
box.

**3. Three checkboxes were ticked during the move.** Their own notes recorded the work as
done while the boxes stayed unticked — a slip, not a decision: `PLAN.md`'s Phase-4-opening
**A** and **B** (both note *"done 2026-10-20"*) and the track plan's **Stage 2.7** (its
note says *"done 2026-10-26, CS-7e"*). They are ticked **here**, in the archive, and the
ticking is the only edit the move made to any moved text.

**4. Cross-references that moved with the text.** The track log's §"Correction — the
Stage-1 atom-arm count" cites "three notes in `TEXT_PROCESSING_PLAN.md`" — those three
notes are in this archive's "The moved §3", under Stage 1. The track log's §"Correction
— the plan's paragraph-length figures" cites "`TEXT_PROCESSING_PLAN.md` section 5" —
that text is in this archive's "The moved §5". `PLAN.md`'s Phase 6 note cites
"`TEXT_PROCESSING_PLAN.md` §7.1.3" — that section is in this archive's "The moved §7",
under §7.1.3. The log itself is append-only and was **not** edited.

**5. The convention for later.** When a change set is executed, its pre-registration moves
here and the live plan keeps one tombstone line for it, naming this archive and the track
log's section. The live plan's §5 keeps only the questions that are still open; a question
moves here when a change set answers it.

---

## The moved §0.1 — the OMCT baseline diagnostics

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


## The moved §3 — the executed stages

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
- [x] **2.6 The chunk header: the paragraph's own first sentence, prepended to
  every chunk.** The classic "contextual chunk header" idea, in the one form
  that is legal on this data: the header is the paragraph's **own first
  sentence** (pair-local, label-free, no document identity, and it keeps one
  embedding per text — so the embedding cache stays at ~500 vectors per model
  instead of the ~175,000 a question-stem header would need). The other
  headings are ruled out by the track's leakage rules: the **document title**
  is document identity (rule ii) and the **question number** is a corpus
  statistic (rule i — 186/294 questions are judged in more than one document,
  101 with conflicting labels).
  *Pre-registered hypothesis:* the chunk geometry carries no label-driven
  **ranking** signal (CS-6d's random-vector floor), so the expected
  within-document AUC gain is ≈ 0; the plausible payoff is the **retrieval**
  side (the positives' hit@1, today 0.105–0.136 for the cosine). The arm runs
  in the existing harness on **all three** Stage-0 models, paired against the
  no-header arm of the same run.
  *(done 2026-10-26, CS-7d: the new chunker `header` (the text's own first
  sentence, in two modes — `others`, the header on every chunk but the first,
  and `all`, the literal reading), the config knob
  `analysis.header_chunker` (a second chunked side beside the plain one, so
  the header is measured against the no-header arm **in the same run**), the
  repo tool `tools/compare_arms_in_run.py` (the paired comparison of two arms
  of one run), and 6 runs
  `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-header-{others,all}`.
  **Verdict: the header does not ship.** The ranking side is not neutral but
  **negative**: the within-document AUC falls by **-0.0315** on embeddinggemma
  (the 95% interval excludes zero) and is a wash on the other two models. The
  retrieval payoff — the hypothesis's target — is at most **+0.0095** of
  hit@1 (voyage-4-nano, the `all` mode) and a wash elsewhere, i.e. far below
  the decision model's own 0.1317 and the shortlist rerank's 0.1565. The two
  modes differ by at most 0.0015 of AUC. The numbers are in the track log's
  Stage-2.6 section.)*
- [x] **2.7 Late chunking: the document embedded once, the chunks pooled out
  of it.** Instead of embedding each chunk separately, embed the whole
  document and pool its **token-level** embeddings per chunk, so every chunk
  carries the document's context. Two costs to state up front: (a) it needs
  **token-level** embeddings, which the `Embedder` port does not expose and
  which Ollama's `/api/embed` **cannot** return → the arm is **HF-only**
  (embeddinggemma and arctic2 drop out), so the comparison is
  **within-model** (late-chunked voyage-4-nano against early-chunked
  voyage-4-nano, same cells, same seeds) and not against the other two
  models' numbers; (b) it needs the chunker's **character spans** (today the
  `Chunker` port returns strings) so the tokens can be pooled per chunk, plus
  a graceful "not available for this backend" path.
  *Pre-registered hypothesis:* the same one as 2.6 (the retrieval side, not
  the ranking side) — and with the decision models' retrieval as the ceiling
  if Stage 3.5 pays off.
  *(done 2026-10-26, CS-7e: the **optional token capability** of the
  `Embedder` port (`TokenEmbedding` / `TokenEmbedder` / `token_capability`
  in `embed/base.py`, the HF implementation in `embed/hf.py`, the cache
  forwarding it **without** caching it — a late-chunked vector is a
  function of the whole document, so the plan's cache-key layout is
  unchanged), the **character spans** of a chunker (the optional
  `SpanChunker` capability, the generic fallback in `BaseChunker` and the
  **exact** word-window spans), the **pooling** as a new pure module
  (`text_processors/late_chunking.py`: the model's own formula
  `Normalize(mean(tokens))` and the per-token-normalized variant), the
  "not available for this backend" path (Ollama's `/api/embed` cannot
  return token-level embeddings, so the arm is HF-only and the CLI says so
  before the run), the config knobs (`analysis.late_chunker`,
  `analysis.late_question_chunker`, `analysis.late_pooling`) and the
  **within-model** arm: 2 runs `omct-analyze-voyage-4-nano-late-{words25,
  symwords25}`, each carrying the run's **early-chunked arm** beside the
  late one (the pairing is by construction) and reproducing the published
  early reference exactly (the cross-run check). **Verdict: the late
  chunking is the ladder's one winner, and the pre-registered hypothesis is
  falsified on the ranking side.** The within-document AUC rises by
  **+0.0859** [+0.0650, +0.1074] against the early-chunked arm of the same
  run (the interval excludes zero; the gain is in all three documents), and
  the positives' hit@1 rises from 0.0916 to **0.1260** — the best
  embedding-side arm of the track on this model, with the control's mean
  shift at +0.0011 (against the early arm's +0.0413), so the gain is a
  ranking gain and not a scale shift, and the reviewed hard negatives' hit@1
  *falls* (0.0241 -> 0.0171) while the ordinary control cells stay put. The
  model's own pooling formula beats the per-token-normalized variant
  (+0.0260 of AUC). It is still **not** the deliverable's scorer: the
  decision arm is at 0.7701 and the shortlist rerank at 0.1565 (the decision
  model's own hit@1, 0.1317, is already above the late chunking's 0.1260).
  The numbers are in the track log's Stage-2.7 section; the arm is
  **HF-only**, so the other two Stage-0 models are unmeasured.)*
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
  *(2.6 and 2.7 are the ladder's two open rungs: the chunk header and the
  late-chunked geometry. Both are *retrieval*-side experiments by the
  pre-registered hypothesis above, and both run **after** Stage 3.5: if the
  decision models work, the embedding axis's remaining job is cheap retrieval
  at scale, and these two rungs are then measured against the decision
  model's hit@1 as the ceiling.)*


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
    baseline, so a "win" can be checked against chance. *(Both are measured:
    the label-permutation floor in CS-5, the random-vector floor in CS-6d —
    the latter replaces the feature pack's embedding sides with seeded random
    vectors, so it says what the training design alone buys.)*
  - *Reporting*: calibration curves + thresholds per document; pooled and
    per-document macro; both negative subsets separately. *(done 2026-10-26,
    CS-6e: the three blocks land in the Stage-3 artifact
    (`stage3.reporting`) and in the track's table columns; the reporting is
    opt-in — `analysis.stage3.reporting.enabled` — so the earlier runs'
    artifacts keep their exact content.)*
- OMCT line stays auxiliary; it informs method choice, never UPR metrics.


### Stage 3.5 — The decision models (a probe, then a decider)

*Why this stage exists (2026-10-26, the reviewer's proposal).* The track's
deliverable is a **decision** — "is this unjudged cell safe to call a
negative?" — and the track built a **ranker** and had to invent a threshold
to turn it into one. That threshold is the track's weakest link: it does not
transfer to the document with the lowest base rate (CS-6e), and the harm it
produces on the verified positives is **reproduced by a label-blind model**
(the control-role floor), so it cannot validate the rule. A **decision
model** — Ollama's new model type: a *state* (the text to judge) plus typed
*questions* (`noul` / `choice` / `score`), answered with **probabilities** in
one forward pass, no decoder and no generation — attacks both problems at
once:

- it is **calibrated by construction** (laya: trained with RLCD against
  strictly proper scoring rules, in-task ECE 0.060; tev1's card says
  calibration is **untested**, so tev1's probabilities must be calibrated by
  us), so the rule gets a **fixed** operating point and the threshold's
  transfer problem disappears;
- it is **independent** of the reviewed set's rank-selection (the pairs were
  chosen by exp18's model, so every embedder's score on that set may be
  measuring the selection, not the labels), so it is the one probe that can
  say whether the **labels** are learnable at all.

The two models are **named per experiment** (never "a local model"), pinned
by id **and digest**, and every call is cached with its prompt version
(§2's hygiene). Both are already local: `tev1:latest` (Together AI, a
Qwen3.5-4B fine-tune, digest `7954dce2…`) and `laya:latest` (Convai
Innovations, a ModernBERT-large fine-tune, digest `1f196b37…`).

- [x] **3.5.1 The probe: is the reviewed set learnable at all?** The decision
  models score the **1,520 judged pairs** (a `noul` question: "is the
  paragraph about the same topic as the question?"), and the artifact carries
  the same measurement blocks the other arms use (the pair AUC pooled and
  per document, the **within-document** AUC, the rank percentiles, the
  control side, the **calibration** curve / ECE / Brier score) so the numbers
  sit next to the embedding arms' in the same table. Two design elements are
  **not** optional:
  - a **declared prompt grid** (the state's shape: the pair in one state vs.
    the paragraph as the state with the question in the instructions — the
    latter allows 64 questions per call and is ~3× cheaper; and the question
    type: `noul` vs. a two-option `choice`, because the docs warn that `noul`
    can get "stuck on no"), each variant logged with its prompt version;
  - a **synthetic competence control** (constructed positive pairs — a
    paragraph and a question sharing a distinctive phrase — and unrelated
    negatives), because a chance-level result on the reviewed set is
    otherwise ambiguous: *the data is at the ceiling* or *the model is the
    wrong shape*. The control separates the two.
  *The decision tree this probe produces:* **well above chance** → the labels
  are topically separable, the track's "the benchmark is construction-
  dominated" conclusion is wrong or overstated, and the deliverable gets a
  real quality claim; **at chance with the control passing** → the reviewed
  set is at the ceiling, the deliverable's precision is *provably*
  unmeasurable on this data, and Stage 4 (the LLM experiments) should be
  skipped; **at chance with the control failing** → the wrong tool (the
  MS MARCO cross-encoders in the HF cache are the next try).
  *(done 2026-10-26, CS-7a: the `decide-omct` command, the `Decider` port,
  the `/v1/systemone` client, the decision cache, the call planner and the
  pure side (`text_processors/decision/`); 8 runs
  `omct-decide-{tev1,laya}-{pair,paragraph}-{noul,choice}` (2 models x the
  declared prompt grid), the scope `judged` (the 1,520 reviewed pairs plus the
  same seeded 4,000-cell control the Stage-0 runs draw) with the synthetic
  competence control beside every arm. **Verdict: the first branch, for
  tev1.** The within-document AUC is **0.7767** (tev1-pair-choice), 0.7604
  (tev1-pair-noul), 0.7708 (tev1-paragraph-choice), 0.7244
  (tev1-paragraph-noul) — against the cosine's 0.5020–0.5632 and the track's
  own learned arm's 0.6472–0.6723; laya's best arm is 0.5992, above the
  cosine but below the learned arm. The **competence control passes
  decisively** (the AUC 0.944–0.9996 on every arm), so the reviewed set is
  *not* at the ceiling. tev1's probability is also better calibrated than
  the track's learned arm (the ECE 0.0891–0.1365 pooled against the arm's
  0.175–0.189), which is what a **fixed** cut needs. The `choice` type beats
  the `noul` for tev1 in both shapes (and the `noul` beats the `choice` for
  laya); the `paragraph` shape is 16× cheaper (339 calls against 5,520) for
  0.006 of AUC. The numbers are in the track log's Stage-3.5 section; the
  next step is 3.5.2 (CS-7b) with **tev1 + the two-option `choice`**.)*
- [x] **3.5.2 The decision arm on the grid + the rule's decider.** The decision
  model scores the **whole same-document grid** (44,467 cells), so it becomes
  an *arm* of `analyze-omct` (the hit@k, the rank percentiles, the control
  side, the de-shifted AUC — all the existing blocks, unchanged) and the
  **safe-negative rule's** ranking signal becomes the decision model's
  probability with a **fixed** cut (0.5, and the fit-side-threshold
  comparison beside it). The deliverable's headline numbers then carry a
  *label-driven* scorer behind them, and the control-role floor comparison
  (the permuted-label model) says how much of the rule's quality is the
  training design's prior. The unjudged cells' true positive rate stays
  **unmeasured** — the rule still ships as the margin half only.
  *(the design proposal, recorded before the code — CS-7b, 2026-10-26: the
  **arm** goes into `analyze-omct` (the plan's wording), so the decision
  model's grid scores, the embedding arms' scores and the rule live in **one
  run** and the pairing is by construction. The arm reuses the existing
  blocks unchanged: the pair AUC, the within/cross decomposition, the score
  distributions, the rank percentiles, the hit@k, the top-1 provenance and
  the comparison against the full-text cosine (the paired bootstrap, the
  control shift, the de-shifted AUC) — with the arm's **level** numbers
  flagged as a different scale (a probability vs. a cosine), the CS-6e
  precedent. The **rule** takes three forms with the decision model's
  probability as the ranking signal: (i) the **fixed 0.5 cut** (the absolute
  rule the calibrated model makes meaningful), (ii) the **fit-side
  threshold** (the same probability cut, but fitted on the fit side and
  applied to the holdout — the old way, beside it), and (iii) the **margin
  rule** (the cell at least X below its own paragraph's best candidate, X
  fitted on the fit side — the existing rule's shape with the decision model's
  probability). The **control-role floor's counterpart** is proposed as the
  **mismatched-pair control**: the same model, the same prompts, but every
  cell's question replaced by another question of the same document (a
  deterministic shift), so the arm is label-blind while keeping the model's
  shape, its calibration and its cost. The alternative readings (a permuted-
  paragraph control, or no counterpart at all) are recorded as the open
  question in section 5; the mismatched control is *measured* either way, so
  the choice is settled by the numbers and not by taste.)*
  *(done 2026-10-26, CS-7b: the decision model scores the **whole
  same-document grid** inside `analyze-omct` (44,467 cells, 1,805 calls in the
  `paragraph` shape), so it becomes an arm with the existing blocks unchanged
  and the paired comparison against the full-text cosine; the rule's three
  forms (the fixed 0.5 cut, the fit-side threshold, the margin rule) with the
  **label-blind counterpart** beside them; 3 runs
  `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage35-grid`.
  **Verdict: the arm is the best scorer on the reviewed set** (the
  within-document AUC 0.7701 against the cosine's 0.5020-0.5632 and the
  track's own learned arm's 0.6472-0.6723; the paired delta +0.207 to +0.268
  with the intervals excluding zero, and the de-shifted AUC 0.759-0.764), and
  **the rule reaches 83.1% of the unjudged grid at the harm 0.1851** — the
  same cost the old best form paid at a 50-60% reach. **The harm is
  label-driven**: the label-blind counterpart discards 41.5% of the verified
  positives at the same reach where the real arm discards 18.5%, so the
  cost claim is finally a quality claim. **And the plan's premise is
  half-wrong**: the fixed 0.5 cut is *not* the good operating point (the
  reach 97.9% at the harm 60.5%) — the model is calibrated but its
  probabilities are compressed, so a *fitted* threshold is still needed. The
  numbers are in the track log's Stage-3.5.2 section.)*
- [x] **3.5.3 The shortlist rerank (PLAN Phase 6's shape).** The cosine's
  top-K candidates per paragraph are re-decided by the decision model; the
  metric is the positives' **hit@1 / hit@K** (today 0.105–0.136 for the
  cosine, 0.065–0.092 for the learned arm — a lot of headroom) and the
  artifact is the Phase-6 evidence PLAN's cross-encoder slot needs. The
  decision models are *not* trained on OMCT, so nothing leaks (leakage rule
  (i) is about *fitted* question-level priors; a frozen model's question
  text carries no OMCT label). **If anyone fine-tunes a decision model on
  OMCT, rule (i) bites hard** (101 questions are positive in one document and
  negative in another) — that boundary is written into the plan.
  *(done 2026-10-26, CS-7c: the rerank's block on the CS-7b run's grid (the
  cosine's top-K per paragraph, reordered by the decision model's
  probability; the positives' hit@1 with the shortlist's own recall as the
  ceiling, against the cosine's own hit@1 and the decision model's own);
  3 runs `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage35-rerank`
  (**no new model calls** — the numbers come from the grid CS-7b already
  scored). **Verdict: the rerank beats both of its references** — the
  positives' hit@1 **0.1565** (arctic2, K=10) against the cosine's 0.1050
  and the decision model's own 0.1317, i.e. the hybrid is above "rerank
  everything" on all three embedders, and the hard negatives' hit@1 *falls*
  with it (0.0171 -> 0.0100). The price is the shortlist's cells: 2,250 at
  K=10 against the grid's 44,467 (~90 decision calls against 1,805, 20x
  cheaper). The winning K differs by the cosine (10 / 20 / 5-10) and is read
  off the reviewed set, so it is an upper bound. The numbers are in the
  track log's Stage-3.5.3 section.)*
- Exit: a measured answer to "does an independent, calibrated scorer beat
  the reviewed set's floor?", with the decision model's numbers sitting next
  to the embedding arms' in the same artifact and the same table, and the
  deliverable's rule re-measured with a label-driven scorer behind it — or an
  explicit, artifact-backed statement that the reviewed set is at the ceiling
  and the deliverable's precision is unmeasurable on this data.


## The moved §5 — the answered open questions

- The label anomaly (label-0 pairs above label-1 p95) is **no longer an open
  question**: §0.1 derives it from the hard-negative construction (the
  negatives are exp18's model FPs; the two score distributions overlap almost
  completely). Stage 0 (CS-1) re-derives the numbers through the harness and
  records them; the anomaly is documented and designed-around, never corrected.
- Question `question_text_sources` / document idiosyncrasies: characterized in
  stage 0, not assumed.
- `lightgbm`: a new dependency is only proposed if sklearn's HistGB is beaten
  (AGENTS §3 explain-first).
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
- **The deliverable's reach is measured, but its quality signal is blind
  (CS-6e, 2026-10-26).** The **two-part margin rule** (the cosine's margin
  *and* the arm's margin below their paragraph's best candidate) reaches
  50–60% of the 42,947-cell unjudged grid at a harm of 11–19% of the 524
  verified positives — the best of the three variants. But at the **same
  reach** the **control-role floor** (the model fitted with the judged pairs'
  labels permuted) discards just as few of the verified positives
  (0.129–0.179 against the real arm's 0.109–0.193; the real arm is better on
  0–3 of the three documents per arm). The reason is the reviewed set's
  **rank-selected construction**: its positives sit near the top of their
  paragraph's candidate list whatever model produces the scores, so a margin
  rule protects them even when the model knows nothing about the labels.
  **Open:** the harm on the verified positives — the only ground-truth number
  the rule can be judged on — therefore measures the benchmark's
  construction, not the rule's topical quality. Is there **any** measurable
  proxy for the unjudged cells' positive rate that is *not* the reviewed
  set's rank structure (a leave-one-paragraph-out construction? a
  perturbation test? a synthetic null with the same coverage?), or does the
  deliverable have to ship with a reach claim and no quality claim at all?
  The reviewer ruled out human labelling (2026-10-23), so the spot-check
  half stays out.
- **The arm's probability is not a probability (CS-6e, 2026-10-26).** The
  expected calibration error is 0.175–0.189 pooled and 0.327–0.349 per
  document (the Brier score 0.259–0.265), and the full-text cosine put on a
  probability scale by a fit-side calibration map is **worse** (ECE
  0.210–0.223, the Brier 0.278). The miscalibration is worst on the AI
  document (the per-document ECE 0.52–0.62, an 80%-positive stratum), and
  the **pooled** ECE flatters every arm (0.18 against a 0.33 macro) because
  the pooled diagram mixes three calibrations. **Open:** does the harm
  change if the rule works on the **rank** (the margin) rather than on the
  probability? It already does — the margin rule never reads the
  probability's absolute level — so the calibration's role in the shipped
  rule is nil, and the calibration block is a *diagnostic* of the arm, not
  part of the deliverable.
- **Is the reviewed set at the ceiling, or were the embedders the
  limitation? (Stage 3.5, 2026-10-26.)** Every embedder sits at chance on
  the reviewed set (AUC 0.468 / 0.500 / 0.537) and the reviewed set was
  **rank-selected** (exp18's model chose which pairs got reviewed), so those
  numbers may be measuring the selection rather than the labels. The
  decision models are the first **independent** rankers on this benchmark, so
  their number is the one that can settle it. **Open:** if they also sit at
  chance (with the synthetic competence control passing), the reviewed set is
  at the ceiling and the deliverable's precision is *provably* unmeasurable
  here — is the honest close-out then "this benchmark cannot validate a
  rule", and should the track's remaining effort move to the UPR side
  (Stage 5) instead of to more OMCT methods?
  *(CS-7a answered it: **they do not sit at chance.** tev1's within-document
  AUC is 0.7767 (the best arm) against the cosine's 0.5020-0.5632 and the
  track's own learned arm's 0.6472-0.6723, and the synthetic competence
  control passes decisively (the AUC 0.944-0.9996 on every arm). So the
  reviewed set's labels **are** learnable and the Stage-0.2 classification's
  "benchmark construction dominates" is **too strong**: the construction
  makes the *embedding* geometry useless as a ranker here, not the labels.
  What remains open is which learner to ship and CS-7b's rule.)*
- **The decision models' calibration on OMCT is unmeasured (Stage 3.5).**
  laya is trained for calibrated probabilities (RLCD, strictly proper scoring
  rules, in-task ECE 0.060) and tev1's card says calibration is **untested**;
  a first smoke test (one verified positive vs. one verified hard negative)
  gave tev1 0.455 / 0.040 and laya 0.262 / 0.155 — the right direction, but
  a one-pair anecdote. **Open:** which of the two is the better *decider* on
  the grid, and does either keep its calibration on this corpus (the ECE and
  the reliability diagram are already the harness's reporting)? And if a
  decision model's probability is calibrated enough to take a **fixed** 0.5
  cut, does the rule's harm on the verified positives stop being reproduced
  by the control-role floor?
  *(CS-7a measured the calibration on the reviewed set: tev1's ECE
  **0.0891-0.1365 pooled** and 0.2009-0.2443 per-document macro, the Brier
  0.1881-0.2064 - **better calibrated than the track's own learned arm**
  (0.175-0.189 pooled / 0.327-0.349 macro), so tev1's probability can take a
  fixed cut; laya's arms are the opposite (0.1188-0.2423 pooled). Whether
  the *harm* stops being reproduced by the floor is CS-7b's question.)*
- **Late chunking is not portable to the Ollama backends (Stage 2.7).** It
  needs token-level embeddings, which `/api/embed` cannot return, so the arm
  is HF-only and its comparison is **within-model**. **Open:** is the
  document-level context worth a within-model-only result, or should the
  chunk header (Stage 2.6, which keeps all three models) be the only new
  chunking rung?
  *(CS-7e answered it: **yes** — the late chunking gains +0.0859 of
  within-document AUC and +0.0344 of hit@1 against the early-chunked arm of
  the same run, with the interval excluding zero, while the chunk header
  (CS-7d) lost 0.0315 of AUC and gained at most 0.0095 of hit@1. A
  within-model-only result is worth having: the arm's mechanism (the
  document's context in every chunk) is model-agnostic, and the two
  Ollama-backed Stage-0 models are unmeasured only because their *backend*
  cannot return token-level embeddings, not because the mechanism is
  model-specific. The stage is now closed.)*
- **The UPR chance floor is a distribution, not a number (CS-8d, 2026-10-26).**
  The label-permutation floor (the fit side's whole label vector permuted) has
  a **micro-F1 between 0.045 and 0.378** over its five draws on UPR, against
  the ~0.029 a random top-2 would give — the same construction landed on a
  tight 0.494-0.505 AUC on OMCT. The mechanism: the UPR pack's feature space
  has a dominant **topical** axis (the gold cells are its extremes), so a
  linear model fitted on a *random* 2.7% target still finds that axis, with a
  random sign; the label-blind model's val AUC moves between 0.44 and 0.66
  across draws, and the top-2 decision amplifies the spread. **Open:** the
  floor is therefore a weak yardstick on UPR — is the right reading the
  floor's *best* draw (the most conservative comparison), its mean over many
  more draws, or a scale-free statistic (the nDCG) instead of the micro-F1?
  The same mechanism means the floor is *not* evidence of a leak; the
  control-role floor (the construction that does not leave the true gold cells
  unlabelled) is the tight one and sits at chance.
- **The UPR pack's lexical block finds the unseen labels (CS-8d, 2026-10-26).**
  The soft lexical overlap discovers five of the eight label-holdout labels
  (the unseen bucket's micro-F1 0.4758, the coverage 0.875, the shortlist
  R@50 0.7211 — against the champion's 0.000 / 0.000 / 0.034) and not the
  three generic ones ("General", "Other", "ESC rights - general"): for
  "General" the gold cells even score *below* the background. **Open:** does
  the mechanism (a label's *name* against the text) survive the CS-8e's
  ablation — i.e. is it the lexical block, and would a label **description**
  (the catalog CS-8d lacks) do better? This is the strongest UPR argument for
  a label-side catalog, and it bears on PLAN's zero-shot route to the
  held-out labels.
  *(CS-8d's question is **answered** by CS-11 and CS-13: the catalog exists
  now, and the decision probability — which reads the text against the
  label's **name** — transfers to the held-out labels even better (the
  unseen bucket 0.5697 against the pack's 0.4469, CS-13).)*
- **The reranker's ordering against the pack's (CS-15/CS-16, 2026-10-27).**
  The pre-registration fixes the comparison (the same candidate set, two
  orderings; the pack + the decision probability as the incumbent), but the
  **shape** of the winner is not decided in advance: a reranker is a
  *reordering* of a candidate set (its candidate set is someone else's), while
  the pack is a *learned scorer* over the whole grid (its own candidate set is
  implicit in the combiner's weights). **Open:** if the reranker wins the
  mechanism claim but not the pipeline claim, is the right pipeline shape
  "generate with the decision model, rank with the reranker" (the reranker
  alone) or "generate with the decision model, rank with the pack, and let
  the reranker break the ties" (the pack and the reranker fused)? The second
  is a *fusion* rule with a fitted weight, so it is a **new thing to fit** and
  it needs its own pre-registration; it is not in CS-15/CS-16.
- **The held-out labels and the reranker (CS-15/CS-16, 2026-10-27).** The
  champion cannot suggest the eight held-out labels at all (its unseen
  bucket is 0.0000) because its geometry never saw them; the decision model
  can (0.5267) and the pack + the decision probability can (0.5697). A
  reranker reading the label's **description** has a third way in. **Open:**
  the reranker's unseen bucket is bounded by the *candidate set's* unseen
  coverage (a reranker cannot enlarge a set), so the question is whether the
  decision model's candidate set already contains the held-out labels often
  enough — and whether the reranker's ordering puts them first when it does.
  The runs measure the set's unseen coverage and the reranked unseen bucket
  side by side; the *ceiling* of the unseen bucket is stated, not assumed.
- **The held-out labels and the reranker: answered (CS-15/CS-16,
  2026-10-27).** The bound is the candidate set's unseen coverage, and it is
  real: the champion's own top-10 contains **no** held-out label at all (the
  arms' unseen bucket is 0.0000 — nothing to re-order), the decision
  model's set holds 84.6% of the held-out gold cells and the union's 92.0%.
  Inside the decision model's set, the rerankers' re-ordering moves the
  unseen bucket's *point estimate* up (the listwise model +0.0239, the 278M
  cross-encoder's label-first arm +0.0222) but not significantly, and the
  **feature** form is where the held-out labels gain: the listwise model's
  full-grid column's unseen nDCG@5 gain is significant (+0.0205 [+0.0016,
  +0.0390]) and its candidate-restricted column reaches the unseen micro-F1
  **0.5957** — the best unseen number the track has measured (the control's
  0.5697, the champion's 0.0000). The weight transfers to the labels the
  fit side never saw, exactly as the CS-13 hypothesis predicted.

## The moved §7 — the executed change sets

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
- [x] **CS-6d — the Stage-3 random-vector floor.** The second sanity floor
  of the plan's Stage 3 (the label-permutation one is done): the arm is
  fitted on random vectors instead of the model's embeddings, so a "win"
  can be checked against what the *training design* alone buys.
  *(done 2026-10-26: the new `src/cluster_rag/embed/random.py`
  (`RandomEmbedder` — a seeded random unit vector of the model's dimension
  per `(kind, text)`, drawn from `sha256(seed | kind | text)`, never
  written to the embedding cache), the config knob
  `analysis.stage3.embeddings` (`"model"` — the default, the Stage-3.1
  behaviour — or `"random"`, validated at config time), and the wiring in
  `cli.py` (the feature pack's **embedding sides** are replaced; the run's
  baseline arm keeps the model's embeddings, so the guard still holds). The
  sweep `experiments_text_processing/stage3_random_floor_sweep.py`: 9 runs
  (3 models x {`base`, `full`, `profile`}, the Stage-3.3 group sets, so
  every CS-6c arm gets its floor),
  `omct-analyze-<model>-stage3-rand-<groupset>`. The guard holds (all 81
  runs reproduce the Stage-0 baseline arm field by field).
  **Verdict: the embedding geometry carries no label-driven signal, and it
  is the retrieval signal besides.** The random floor of the v0 pack and of
  the full pack is *indistinguishable* from the real arm on five of the six
  arms (every interval includes zero; on arctic2 the floor is even higher),
  and it lands on the Stage-3.1b embedding-free arm's 0.6589 — the v0 pack
  *is* the embedding-free pack. The chunk profile's floor sits at or below
  chance (0.4733 / 0.4830 / 0.4706), so that block's whole signal is the
  embedding geometry (the same conclusion the Stage-3.3 paired comparison
  reached from the other side). The **hit@1** is where the geometry shows:
  the floor's 0.027–0.042 for the v0 and full packs (0.006–0.015 for the
  profile block alone) against the real arm's 0.065–0.092 and the cosine's
  0.105–0.136 — the lexical block is the *ranking* signal, the embedding
  geometry is the *retrieval* signal. The numbers are in the track log's
  random-vector floor section; 413 unit tests green.)*
- [x] **CS-6e — the Stage-3 reporting.** The calibration curves and the
  thresholds per document, and the safe-negative margin measured against the
  **control-role floor** (the reviewer's decision 2026-10-26: the track's
  durable deliverable is measured against that floor, not against the plain
  cosine; the rule still ships as the margin half only).
  *(done 2026-10-26: the plan's Stage-3 reporting discipline, measured and
  shipped as three new artifact blocks (`stage3.reporting`): the
  **calibration** (the reliability diagram of the learned arm's probability
  and of the full-text cosine put on a probability scale by a **fit-side**
  calibration map, pooled and per document, with the bin counts, the
  expected calibration error and the Brier score), the **thresholds** (each
  document's operating point, chosen on the **fit side** and applied to the
  held-out one, with the pooled out-of-fold cut beside it and flagged as
  *not* like-for-like) and the **safe-negative margin** (a margin rule, never
  an absolute cut: a cell is a safe negative when it sits at least the
  chosen margin below its own paragraph's best candidate; the reach over the
  **whole 42,947-cell unjudged grid**, the harm on the 524 verified
  positives, the **control-role floor's** counterpart and the explicit
  statement that the unjudged cells' true positive rate is unmeasured). The
  fit side's own rows are scored by an **inner** document-level cross-fit, so
  no model that saw the holdout produced a number the holdout is judged on.
  The new pure module `text_processors/calibration.py` (numpy only: the
  reliability diagram, the ECE, the Brier score, the operating point, the
  fit-side threshold and margin choices, the margin and the rule's counts);
  the config knob `analysis.stage3.reporting` (the default `false`, so every
  earlier run's artifact keeps its exact content — the same precedent as
  CS-6d's `embeddings` knob); the sweep
  `experiments_text_processing/stage3_reporting_sweep.py`: 12 runs (3 models
  x 2 group sets x 2 fit-side criteria),
  `omct-analyze-<model>-stage3-report-<groupset>-<criterion>`. The guard
  holds (all 93 runs reproduce the Stage-0 baseline arm field by field).
  **Verdict: the reach is measured, the quality signal is blind.** The
  **two-part rule** (the cosine's margin *and* the arm's margin — CS-6d's
  "retrieve with the cosine, rank with the learned arm") is the best of the
  three variants: it reaches 50–60% of the unjudged grid at a harm of
  11–19% of the verified positives, where the geometry alone reaches 78–83%
  at 31–34% and the arm's margin alone 59–65% at 26–40%. But at the **same
  reach** the **control-role floor** (the model fitted with the judged
  labels permuted) discards just as few of the verified positives (0.129–0.179
  against the real arm's 0.109–0.193; the real arm is better on 0–3 of the
  three documents per arm), so the only ground-truth number the rule can be
  judged on measures the reviewed set's **rank-selected construction**, not
  the rule's topical quality. The calibration is bad everywhere (the arm's
  ECE 0.175–0.189 pooled / 0.327–0.349 per-document macro, the cosine's
  calibration map worse: 0.210–0.223; the pooled ECE flatters both). The
  threshold's **`floor_gap` criterion** (positives vs. the ordinary cells)
  transfers on 6 of 6 arms, the `youden` criterion (positives vs. the
  reviewer-verified hard negatives) does not — the hard negatives sit above
  the positives, so their threshold lands far too high (arctic2's holdout
  recall 0.02–0.06). The deliverable therefore ships with a **reach** claim
  and a **cost** claim and **no precision claim**; the numbers are in the
  track log's reporting section; 433 unit tests green.)*
- [x] **CS-7a — the decision-model client + the probe (Stage 3.5.1).** The
  `Decider` port, the `/v1/systemone` client, the decision cache and the
  probe command; the synthetic competence control; the declared prompt grid.
  **Explain-first (schema):** a new config section
  `analysis.decision` with `models` (each: `name`, `model_id`, `digest`
  (pinned; the run fails on a mismatch), `state` (`pair` | `paragraph`),
  `question_type` (`noul` | `choice`), `instructions` (the **task**
  instruction — the question's placement is the shape's business, so a
  task that repeats the question is rejected), `options` / `criteria` /
  `positive_option` (the `choice` type only) and `max_input_tokens` (the
  model's input budget)), `scope` (`judged` | `grid`),
  `competence_control: bool`, `competence_pairs`, `prompt_version`, `host`,
  `timeout`, `max_questions_per_call` and `bins`; a new command
  `decide-omct` writing `results/<run_id>/decisions.json` (the per-pair
  probabilities, the competence control, the measurement blocks, the
  provenance: the model id, the digest, the prompt version, the cache's
  hit/miss counters). Runs: `omct-decide-<model>-<variant>` (2 models x the
  prompt grid). The track's hygiene: no relabelling (a decision model only
  *scores*), the model and digest pinned, the prompt version recorded, the
  calls cached (no re-judging on a re-run), and the **leakage boundary
  written down** (a frozen, zero-shot model carries no OMCT label; a
  *fine-tuned* one would break leakage rule (i)).
  *(done 2026-10-26: the new edge package `src/cluster_rag/decision/`
  (`base.py` the port and the answer shapes, `systemone.py` the raw
  `/v1/systemone` client with the digest check, `cache.py` the decision
  models' own disk cache under `cache/decisions/`, keyed by the model id,
  the digest, the prompt version, the state and the question specs); the new
  pure package `src/cluster_rag/text_processors/decision/` (`questions.py`
  the typed question specs and the prompt's rendering, `probe.py` the cells,
  the call plan, the competence control and the measurement); the driver
  `omct_decision.py`; the command `decide-omct` with the `decisions.json`
  artifact; the config section `analysis.decision` (validated at config
  time); the sweep cell `DecideOmctCell`; the shared model specs
  `experiments/decision_pool.py` (the pinned digests, the task instruction
  and the measured input budgets); the sweep
  `experiments_text_processing/stage35_decision_sweep.py`: 8 runs
  `omct-decide-{tev1,laya}-{pair,paragraph}-{noul,choice}`. **Verdict: the
  reviewed set IS learnable** — the within-document AUC 0.7767 (the best
  arm, tev1-pair-choice) against the cosine's 0.5020–0.5632 and the track's
  own learned arm's 0.6472–0.6723, with the **competence control passing
  decisively** (the AUC 0.944–0.9996 on every arm) and tev1's probability
  better calibrated than the learned arm's (the ECE 0.0891–0.1365 pooled
  against 0.175–0.189). The `choice` type beats the `noul` for tev1 (and the
  `noul` beats the `choice` for laya); the `paragraph` shape is 16× cheaper
  (339 calls against 5,520) for 0.006 of AUC. laya's 512-token context truncates 8.0–8.1% of the
  cells' paragraphs (tev1: none), which is recorded per cell. The guard
  holds (all 93 `omct-analyze-*` runs reproduce the Stage-0 baseline arm
  field by field). The numbers are in the track log's Stage-3.5 section; 469
  unit tests green.)*
- [x] **CS-7b — the decision arm on the grid + the rule's decider (Stage
  3.5.2).** The decision model's scores over the whole same-document grid as
  an **arm** of `analyze-omct` (the hit@k, the rank percentiles, the control
  side, the de-shifted AUC — the existing blocks, unchanged) and the
  safe-negative rule re-measured with the decision model's probability as the
  ranking signal and a **fixed** cut (0.5) beside the fit-side-threshold
  comparison; the control-role floor's counterpart beside both.
  *(done 2026-10-26: the decision arm inside `analyze-omct` (the
  `score_decision_arm` block + the `compare_arms` comparison against the
  full-text cosine, the pairing by construction), the rule's three forms
  (the fixed 0.5 cut, the fit-side threshold, the margin rule) with the
  **label-blind counterpart** (the same model, every cell's question replaced
  by another question of the same document) beside them, the config knob
  `analysis.decision.mismatched_control`, and the sweep
  `experiments_text_processing/stage35_grid_sweep.py`: 3 runs
  `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage35-grid` (the
  decision arm: `tev1`, the `paragraph` shape, the two-option `choice`; the
  whole 44,467-cell grid plus its label-blind counterpart, 1,805 calls each).
  **Verdict: the decision arm is the best scorer on the reviewed set** (the
  within-document AUC 0.7701 vs. the cosine's 0.5020-0.5632 and the track's
  learned arm's 0.6472-0.6723; the paired delta +0.207 to +0.268, the
  intervals excluding zero, the de-shifted AUC 0.759-0.764), and **the rule
  reaches 83.1% of the unjudged grid at the harm 0.1851** — the same cost
  the old best form paid at a 50-60% reach. **The harm is label-driven**:
  the label-blind counterpart discards 41.5% of the verified positives at
  the same reach where the real arm discards 18.5%, so the deliverable's
  cost claim is finally a *quality* claim and not only a reach claim. **And
  the plan's premise is half-wrong**: the fixed 0.5 cut is *not* the good
  operating point (the reach 97.9% at the harm 60.5%) — the model is
  calibrated but its probabilities are compressed, so a *fitted* threshold
  is still needed. The numbers are in the track log's Stage-3.5.2 section.)*
- [x] **CS-7c — the shortlist rerank (Stage 3.5.3).** The cosine's top-K
  candidates re-decided by the decision model; the positives' hit@1 / hit@K
  against the cosine's and the learned arm's; the artifact is PLAN's Phase-6
  evidence.
  *(done 2026-10-26: the rerank's block (`shortlist_rerank_block`) on the
  CS-7b run's grid — the cosine's top-K per paragraph, reordered by the
  decision model's probability, with the positives' hit@1, the per-document
  macro, the shortlist's own recall as the ceiling, the hard negatives' and
  the control cells' hit@1, and the two references (the cosine's own hit@1
  and the decision model's own) — the config knob
  `analysis.decision.rerank_ks`; the sweep
  `experiments_text_processing/stage35_rerank_sweep.py`: 3 runs
  `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage35-rerank`, no
  new model calls. **Verdict: the rerank beats both of its references** (the
  positives' hit@1 0.1565 on arctic2 at K=10 against the cosine's 0.1050
  and the decision model's own 0.1317) and the hard negatives' hit@1 falls
  with it (0.0171 -> 0.0100) — PLAN's Phase-6 slot now has its evidence:
  the hybrid is the best retriever on the reviewed set, at ~90 decision calls
  instead of the grid's 1,805. The numbers are in the track log's
  Stage-3.5.3 section.)*
- [x] **CS-7d — the chunk header (Stage 2.6).** The paragraph's own first
  sentence prepended to every chunk (the pack's option, so the embedding
  cache stays at one vector per text); the arm in the existing harness on all
  three Stage-0 models, paired against the no-header arm; the
  pre-registered hypothesis (the **retrieval** side, not the ranking side).
  *(done 2026-10-26: the new chunker `header` (the text's own first sentence
  prefixed to the inner chunker's chunks; two modes — `others` (every chunk
  but the first) and `all` (every chunk)), the config knob
  `analysis.header_chunker` (a second chunked side beside the plain one, so
  the header arm and the no-header arm share the run and the pairing), the
  tool `tools/compare_arms_in_run.py` (the paired arm-vs-arm comparison of
  one run), and 6 runs
  `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-header-{others,all}`.
  **Verdict: the header does not ship** — the ranking side is *negative* on
  embeddinggemma (the within-document AUC -0.0315, the interval excluding
  zero) and a wash elsewhere, and the retrieval payoff is at most +0.0095 of
  hit@1 (voyage-4-nano, `all`), far below the decision model's 0.1317 and the
  rerank's 0.1565. The numbers are in the track log's Stage-2.6 section.)*
- [x] **CS-7e — late chunking (Stage 2.7).** The token-level embedding port
  (a new optional capability of the `Embedder` port, implemented by the HF
  backend only), the chunker's **character spans** (a new method on the
  `Chunker` port), the pooling of the parent's token embeddings per chunk, the
  graceful "not available for this backend" path, and the **within-model**
  arm (late-chunked voyage-4-nano against early-chunked voyage-4-nano, same
  cells, same seeds).
  *(done 2026-10-26: the token capability (`TokenEmbedding` /
  `TokenEmbedder` / `token_capability` in `embed/base.py`, the HF
  implementation, the cache forwarding it without caching it), the spans
  (the optional `SpanChunker` capability, the generic fallback and the exact
  word-window spans), the pooling (the new pure module
  `text_processors/late_chunking.py`, two rules), the "not available for
  this backend" path, the config knobs and the within-model arm: 2 runs
  `omct-analyze-voyage-4-nano-late-{words25,symwords25}`, each with the
  run's early-chunked arm beside the late one. **Verdict: the late chunking
  is the ladder's one winner** — the within-document AUC +0.0859
  [+0.0650, +0.1074] against the early-chunked arm and the positives'
  hit@1 0.0916 -> 0.1260, with the control's mean shift at +0.0011 (so the
  gain is a ranking gain, not the early chunking's scale shift) — and the
  pre-registered hypothesis (the ranking gain would be ≈ 0) is **falsified**.
  It is still below the decision arm (0.7701) and the shortlist rerank
  (0.1565), so the deliverable's scorer does not change; it is the best
  embedding-side arm the track has. The numbers are in the track log's
  Stage-2.7 section.)*


### 7.1 The UPR port — the change sets (CS-8a … CS-10)

*Pre-registered 2026-10-26, before the first UPR run.* The intent: run
**every** method the track measured on OMCT on the UPR data — **including the
ones that failed there** — so that "it does not work here" is a *measurement*
and not an assumption. One change set per commit, one checkbox per change set;
every number comes from a `results/<run_id>/` artifact; the claims discipline
holds (a UPR number is a UPR metric and always names its split; an OMCT number
is method-choice evidence only; `test` is touched once, for the final adopted
config).

The **pre-registered expectations** (written down before the runs):

1. *The chunking axis is a null on the ranking side.* UPR texts average 151
   characters; the track's smallest word window was 40 words (~250
   characters), so a text is usually **one** chunk and the fold is the
   identity. Expect ≈ 0 difference in every headline number, and a measured
   null (with an artifact) is the deliverable, not a failure.
2. *A possible small win on the retrieval side for the long-text tail.* The
   pooled UPR number would hide exactly that effect, so the texts are bucketed
   by length and every length bucket is reported separately.
3. *The lexical block's transfer is the open question.* On OMCT the learned
   pack's gain was the **lexical** block's, not the embedding geometry's. The
   UPR vocabulary is a different corpus (68 human-rights labels against 294
   questions), so the question "does the lexical signal transfer?" is re-asked
   here, and the answer is the track's most interesting UPR result either way.
4. *The chunk header and the symmetric (label-side) variants are expected
   nulls* — the labels are 1–3 words, so chunking them is near-vacuous (the
   degeneracy is recorded in the artifact).

- [x] **CS-8a — the UPR command, the plumbing, the UPR-side background.** The
  track's UPR command (`analyze-upr`, the UPR analogue of `analyze-omct`; the
  placeholder `chunk-upr` in `experiment_table.py` is renamed), the sweep cell
  and the sweep's allowed command set, the `upr_analysis:` config section, the
  track's UPR table file (`paths.experiment_table_text_processing_upr`), the
  UPR driver (`src/cluster_rag/upr_analysis.py`) with the UPR-side
  diagnostics: the gold (text, label) matrix per split, the **non-gold labels
  as the background** (fully enumerable on UPR — no control sampling, no seed
  to pin, and the artifact says so), each label's **rank percentile** inside
  its own text's 68-label list (0 = best) and the **top-fifth share**, the
  gold label's **hit@1 / hit@10**, the score distributions per bucket (head /
  torso / tail / unseen) and per label, the **level and rank shift of the
  non-gold labels** beside every headline number, the **text-length and
  chunk-count distributions**, and the **top-1 provenance** analogue. The
  whole-text reference arm must reproduce the main pipeline's `evaluate`
  numbers for the same predictor and split, field by field (the guard, the
  UPR analogue of `tools/verify_analysis_baseline.py`: a new repo tool). The
  leak-free boundaries: the indexes and the fit side come from **train** only,
  val is for tuning, the label-holdout split's 8 held-out labels are excluded
  everywhere in the train artifacts, `test` is touched once, for the final
  adopted config.
  *Artifacts:* `results/<run_id>/analysis.json` (the resolved config beside
  it), the auto-appended rows in `docs/experiments_table_text_processing_upr.md`,
  the log entry in `docs/experiments_text_processing.md`.
  *Exit:* the command runs, the reference arm reproduces the `evaluate` run
  field by field, the guard tool is green, the table file exists with its
  first rows, and the log carries the run.
  *(done 2026-10-26: the command `analyze-upr` (the placeholder `chunk-upr`
  renamed; the never-implemented `chunk-omct` entry removed, so every
  registered command exists), the shared evaluation wiring
  (`cli._prepare_evaluation` — the `evaluate` command and the track's
  reference arm now run the **same** code path, which is what makes the
  field-by-field guard possible; the champion's `evaluate` artifact is
  byte-identical after the refactor), the `upr_analysis:` config section
  (the length buckets, the guard's reference run, the bootstrap settings),
  the pure diagnostics module `text_processors/upr_diagnostics.py`, the
  driver `upr_analysis.py`, the sweep cell `AnalyzeUprCell` (a UPR sweep
  command, so the labels carry the `upr-` prefix), the sweep
  `experiments_text_processing/upr_analysis_sweep.py`, the repo tool
  `tools/verify_upr_baseline.py` and the track's **first UPR table**
  (`docs/experiments_table_text_processing_upr.md`). One run
  `upr-analyze-champion` (the whole-text reference arm of the standing
  champion on **val**). **Verdict:** the guard holds (the reference arm
  reproduces `upr-eval-fusionc-embeddinggemma-k10-c1` field by field, inside
  the command and through the tool), the background needs no sampling
  (43,568 non-gold cells against 1,312 gold ones), the background's rank p50
  is exactly 0.5000 (the random expectation — a check of the rank machinery)
  against the gold side's 0.0147, and two UPR facts fall out that the
  pipeline's own harness does not show: the **unseen** labels are not
  discoverable by the champion's geometry (coverage 0.000, shortlist R@50
  0.034; the per-label distributions show the gold and the background cells of
  an unseen label score the same, e.g. "General" 0.0946 vs 0.0881), and the
  **long-text tail** (300+ characters, 25 val texts) is the champion's
  weakest bucket (micro-F1 0.4727, nDCG@5 0.6104) — the starting point of
  the CS-8b hypothesis. The numbers are in the track log's CS-8a section; 511
  unit tests green.)*
- [x] **CS-8b — the segmentation ladder on UPR.** The one-sided family (the
  **text** chunked with `atoms`, `word_windows` N = 15/25/40,
  `sentence_windows`, `semantic`) folded with `max`, `top_k_mean:2`,
  `top_k_mean:3`, `mean`, `decayed_top_k:3:0.5:0.25`, `local_global:0.5|max`,
  `local_global:0.75|max`, `peak:3:0.05`; the **symmetric** family (the
  **label** side chunked too — run anyway, as the null-check, with the
  degeneracy recorded). The semantic chunker's threshold is selected on the
  **fit side only** (the train split's own label-scoring metric; the grid, the
  chosen threshold and the per-fold numbers go into the artifact). Two
  families, because UPR has both: the **pair-scoring** family (the text's
  chunks against the zero-shot label vectors) and the **retrieval** family
  (each chunk is a query against the kNN / prototype index, per-label scores
  per chunk, folded the same way; the fusion's weights re-fitted on val for
  the chunked components). Every number per bucket (head / torso / tail /
  unseen) and per **length bucket**, against the standing champion
  (embeddinggemma; fusion a=0.5 / b=1.0 / c=1.0, knn k=10, `top_k k=2` →
  val micro-F1 0.5479; 0.5988 at the calibrated `score_threshold t=1.75`) and
  against the plain `zeroshot` and `knn` predictors.
  *Exit:* the ladder's verdict on UPR, with the length buckets, the
  background-side shift beside every headline number, and the null recorded
  as a measured null.
  *(done 2026-10-26: the chunked-arm machinery in the driver
  (`upr_analysis.py`: the chunked side, the fold of the chunk-level scores
  through the **pipeline's own predictor**, the symmetric family, the label
  side's fold), the config's arm families and weight grid, the **paired
  bootstrap of the F1 / nDCG difference** (the one new function CS-8f
  planned — `eval/multilabel.paired_metric_bootstrap`, pulled forward
  because the ladder's verdict needs an interval; the resampling unit is the
  text, pooled **and per length bucket**), the sweep
  `experiments_text_processing/upr_ladder_sweep.py` and one run
  `upr-analyze-ladder` (153 arms in 13 families on val). **Verdict:** the
  pre-registered expectation is **half confirmed and half refined** — the
  pooled ladder is a near-null (the best un-refitted arm +0.0137 of micro-F1)
  but the gain is **entirely in the long texts**: the 300+ character bucket
  (+0.1091 micro-F1 [+0.0194, +0.2069] for the atoms, +0.1273 [+0.0377,
  +0.2264] for the word windows) and the 200-299 bucket (+0.0393 [+0.0039,
  +0.0757]), while the short buckets do not move at all. The two chunkers
  that do not split UPR's texts (the sentence windows, the semantic
  boundaries: 1.00 chunks per text) give **exactly zero** in every bucket and
  every rule — the ladder's null, measured. The **sentence boundaries beat
  the word windows** pooled (the opposite of the OMCT verdict), and the scale
  shift is there while the ranking gain survives it (the background's mean
  +0.0505, its rank position unchanged). Two further findings: the
  **pair-scoring family is much worse at the decision view** (-0.0965 to
  -0.1345, the intervals excluding zero) **and much better at the
  shortlist** (R@50 0.8658-0.8701 against the champion's 0.7423 — the
  champion's kNN and prototype terms push 12.7 points of gold cells out of
  the top-50, which is a Phase-6-relevant fact about the champion itself),
  and the **symmetric family is degenerate** (0 of the 68 labels produce more
  than one chunk; the arms are identical to the one-sided pair-scoring arms).
  The semantic threshold is selected on the fit side and the grid is flat
  (0.4606-0.4609 over the six thresholds; all three folds choose the same
  one). The numbers are in the track log's CS-8b section.)*
- [x] **CS-8c — the text-side extras and the label-side transform.** The
  **chunk header** (the text's own first sentence, prepended to every chunk,
  in both modes `others` and `all`; the other headings stay ruled out), the
  **late chunking** (the text embedded once, its token-level embeddings pooled
  per chunk, with the model's own `Normalize(mean(tokens))` formula and the
  per-token-normalized variant; **HF-only**, so the comparison is
  within-model against the early-chunked arm of the same run, and the CLI says
  so *before* the run), and the **label-side transform** (the label text as
  the bare name vs the config template — the analogue of OMCT's `keep` vs
  `strip_block`; paired, on the same cells, with the non-gold labels' level
  and rank shift beside it).
  *Exit:* every arm measured and logged, including the ones expected to be
  nulls.
  *(done 2026-10-26: two runs. `upr-analyze-extras` (the champion's
  geometry): the **chunk header** in both modes (the inner chunker: the
  atoms) and the **label-side transform** (the bare name vs the config
  template, each family pointing at its own prior label run, so the two
  variants sit in one run). `upr-analyze-late`: the **late chunking** on
  `voyage-4-nano` — the arm is **HF-only**, so the comparison is
  **within-model** (the late arm against the early-chunked arm of the same
  run), with the token path's consistency check in the artifact (the cosine
  between the text's pooled token vector and the model's own embedding:
  0.9468). **Verdicts:** the **header does not ship on UPR either** (a null
  to negative: -0.0030 to -0.0091, the `mean` rule significant; the only
  positive arm is the local+global blend, a wash); the **template wins
  decisively** over the bare label name (the name variant -0.1527 to -0.1748
  of micro-F1 against the champion, the template -0.1102 to -0.1345 — so the
  label's sentence is a better query than the label's bare noun, where OMCT
  found the effect model-dependent); the **model's own pooling formula wins
  again** (the late `mean` beats the late `norm` by +0.0114 of micro-F1 and
  +0.0183 of nDCG@5, as on OMCT), the late chunking gains on the **ranking**
  side (the nDCG@5 +0.0075, the shortlist R@50 0.8374 against the early
  arm's 0.8158, and +0.06 of shortlist recall in the long-text tail) but not
  on the decision side, and none of its intervals excludes zero. The numbers
  are in the track log's CS-8c section.)*
- [x] **CS-8d — the feature pack (UPR variant) + the logistic combiner + the
  floors.** The UPR pack: the global cosine, the best-chunk cosine, the mean
  of the 2 and of the K=3 best chunk cosines, their spread and the peak
  margin, the soft lexical overlap in both directions (text ↔ label name) and
  the char-n-gram cosine, the label-description coverage, the label-type
  features and the chunk-profile block. **No length feature as such** — the
  length is *measurable* on UPR, so it is recorded and reported as a
  diagnostic (the CS-6c precedent), never fed to the model (length tracks the
  label's frequency and the record's nature: a document proxy); the label
  frequency is a **corpus statistic** (fit side only). The **question-part and
  question-type blocks have no clean UPR analogue** (a UPR label has no answer
  options, no "no data" option and no yes/no structure) and their label-side
  counterparts (the label's name vs description vs parent) need a **label
  catalog**, which the repo does not have (`label_catalog: null`): the pack
  therefore runs **without the label-type block** and the gap is stated in
  the log. The **combiner**: the logistic regression behind the `Combiner`
  port, fitted on train (the train gold pairs plus the train non-gold labels
  as down-weighted presumed negatives, weight 0.3 as on OMCT), evaluated on
  val. The **floors**, all three, on the same statistic as the headline: the
  label-permutation **chance floor**, the **control-role floor** (the gold
  labels permuted, the non-gold labels keeping their presumed-negative role)
  and the **random-vector floor** (the embedding sides replaced with seeded
  noise). On UPR the control-role floor is more natural than on OMCT: the
  non-gold labels are enumerable, so "the ordinary pairs are negative" is a
  complete statement rather than a sampled one.
  *Exit:* the pack's verdict on UPR, the combiner's numbers on val, the three
  floors beside them.
  *(done 2026-10-26: the new pure module
  `text_processors/features/upr_features.py` (the UPR pack: the global cosine,
  the best-chunk cosine, the means of the 2 and of the K=3 best chunk cosines,
  their spread, the peak margin, the soft lexical overlap in both directions
  against the label's **name**, and the char-3-gram cosine; the output is the
  grid's cells, a cube of `(n_texts, n_labels, n_features)`), the new config
  section `upr_analysis.learned` (the groups, the chunker, the combiner spec,
  the control weight, the three floors and their resamples; **off** by
  default, so the earlier runs' artifacts keep their exact content), the new
  driver `upr_learning.py` (the fit on **train**, the evaluation on **val**,
  the three floors), the edge module `label_catalog.py` (the optional
  name / description / parent catalog behind the `paths.label_catalog` hook)
  and the sweep `experiments_text_processing/upr_learned_sweep.py`: one run
  `upr-analyze-learned`. The guard holds (all five UPR runs reproduce their
  `evaluate` run field by field). **Verdict:** the pack is a **worse
  classifier** than the champion (-0.0729 of micro-F1, the paired interval
  [-0.1010, -0.0454]) and a **much better candidate generator** (the shortlist
  R@50 0.8692 against the champion's 0.7423) — the same shape as the CS-8b
  pair-scoring family, and the Phase-6-relevant fact. The **lexical** block
  transfers and carries the ranking (the random-vector floor keeps 0.3837 of
  the arm's 0.4749, and its shortlist recall is unchanged), so OMCT's finding
  holds on a different vocabulary — but the lexical signal alone does not beat
  the champion at the decision view. The **control-role floor is at chance**
  (0.027-0.043), so UPR's grid carries no construction signal (unlike OMCT's
  rank-selected reviewed set) and the arm's gain over its floor is entirely
  label-driven; the **chance floor is a distribution, not a number** (0.045-0.378
  over five draws), because the pack's feature space has a dominant topical
  axis that even a label-blind linear model finds (with a random sign). And
  the pack **discovers five of the eight unseen labels** (the unseen bucket's
  micro-F1 0.4758, the coverage 0.875, the shortlist R@50 0.7211, against the
  champion's 0.000 / 0.000 / 0.034) — the specific ones ("Gender-based
  violence", "Freedom of association…", "Asylum-seekers - refugees", "Special
  procedures"), not the generic ones ("General", "Other"). **The stated gap:**
  the label-description coverage and the label-type features need a label
  catalog (`paths.label_catalog: null`), so the run packs the two
  catalog-free groups only. The numbers are in the track log's CS-8d
  section; 540 unit tests green.)*
- [x] **CS-8e — the two "not the lever" checks and the ablation, re-run on
  UPR.** The **gradient-boosted combiner** (`HistGradientBoostingClassifier`,
  the pinned defaults, no grid) behind the same port, with the feature-group
  selection as the knob; the **chunk-profile block** as the pack's fifth group
  (the same three arms: base / full / profile-alone); the **feature-group
  ablation** (the groups are a config knob) with the lexical block's transfer
  as the headline question; and the **per-feature diagnostics** (every
  feature's own ranking quality, its background-side counterpart and its
  length correlation).
  *Exit:* the two OMCT verdicts ("the learner is not the lever", "the chunk
  profile is not the lever") re-measured on UPR, and the ablation's answer.
  *(done 2026-10-26: the chunk profile as the pack's fifth group (the text cut
  into its sentence units and its word pieces, the label side one unit — the
  CS-8b verdict — with the profile's ranks and the four alignment kinds), the
  per-feature diagnostics (every feature's own AUC, its background-side
  counterpart and its length correlation, the CS-6c precedent on the UPR
  grid), and ten runs: `upr-analyze-learned-cp-{base,full,profile}`,
  `upr-analyze-learned-abl-{embed,lex,noembed,nolex}` and
  `upr-analyze-learned-histgb-{full,lex-only,embed-only}` (the booster is
  sklearn's `HistGradientBoostingClassifier` with the pinned defaults, no
  grid — the registry entry already existed). The guard holds for all fifteen
  UPR runs, and the `cp-base` run reproduces the CS-8d run's numbers exactly.
  **Verdicts:** the **chunk profile is not the lever** (the OMCT verdict
  transfers): it costs -0.0045 of micro-F1 in the full pack, its random-vector
  floor (0.1421 against the real arm's 0.4726) says its signal is the
  embedding geometry re-measured, its lower ranks are **length proxies** (the
  length correlation grows from +0.259 to +0.605 with the rank's depth) and
  its second sentence is at chance (UPR's texts have one or two sentences);
  the profile's one new signal is **lexical** (the alignment kind
  `cprof_align_lexical` reaches 0.7177), which the lexical block already
  carries. The **ablation's answer**: the lexical block transfers but it is
  **not** the dominant block on this vocabulary (the opposite of OMCT) — the
  embedding block alone reaches 0.4460 against the lexical block's 0.3837,
  and the lexical block adds +0.0289 on top of the embedding block. And the
  **learner does matter a little on UPR** (the opposite of OMCT's "the learner
  is not the lever"): the booster is above the logistic on the two packs that
  carry the lexical block (+0.0114 and +0.0206 of micro-F1) and below on the
  embedding-only arm, and its shortlist recall (0.8909-0.8991) is the best
  the track has measured on UPR — but it still does not beat the champion at
  the decision view. The numbers are in the track log's CS-8e section; 545
  unit tests green.)*
- [x] **CS-8f — the reporting blocks.** The **calibration** block (ECE, Brier,
  the reliability diagram, pooled and per bucket) and the **decision-rule
  comparison** (`top_k`, `score_threshold`, `top_k_min_score`) on the best
  arm. The **paired-F1 bootstrap** (a paired bootstrap over the val texts of
  the F1 / nDCG difference between two scorers on the same texts) is **one
  new pure function** — the existing bootstraps in `eval/metrics.py` are
  AUC-specific, and the track's "a gain needs a paired interval that excludes
  zero" discipline is worth keeping on an F1 difference. *Proposal, not an
  assumption:* the reviewer decides whether the function is added.
  *Exit:* the reporting blocks in the artifact and in the table, and the
  paired intervals where they were approved.
  *(done 2026-10-26: the config section `upr_analysis.learned.reporting` (the
  bins, the threshold's quantile grid; **off** by default, so the earlier
  runs' artifacts keep their exact content), the reporting block in the driver
  (the **calibration** — the reliability diagram, the expected calibration
  error and the Brier score, pooled and per label bucket and per
  text-length bucket — and the **decision-rule comparison** over `top_k`,
  `score_threshold` and `top_k_min_score`, the threshold rules' cut chosen on
  the **fit side** by the Youden separation, so the evaluated split never
  picks the operating point), and two runs
  `upr-analyze-learned-report-{histgb,logistic}` on the CS-8e grid's best arm
  (the boosted combiner on the full pack) and the logistic on the same pack.
  The guard holds for all seventeen UPR runs. **Verdicts:** the learned arm's
  probability is a **probability** — the pooled ECE **0.0323** (the booster)
  and 0.0436 (the logistic), against the OMCT learned arm's 0.175-0.189 — and
  the booster is better calibrated than the logistic, the second way in which
  the learner matters on UPR. The calibration is best on the **unseen** labels
  (the ECE 0.012) and worst on the **long texts** (0.0517/0.0705), the
  bucket the CS-8b ladder found the champion's weakest. The **best operating
  point is a budget rule**: `top_k_min_score` (the top-2 with the fit side's
  cut as a floor) beats the plain `top_k` on both learners (0.4945 against
  0.4818 for the booster, 0.4800 against 0.4704 for the logistic), while the
  **pure threshold rule is much worse** (0.2465/0.1773) because the
  probabilities are compressed toward the 2.9% base rate — the same
  compression the CS-7b decision arm showed on OMCT. The numbers are in the
  track log's CS-8f section; 547 unit tests green.)*
- [x] **CS-9 — the decision models on UPR (a measurement, not a pipeline
  component).** `tev1` and `laya` through the existing `Decider` port and the
  `/v1/systemone` client, with the existing disk cache and the track's LLM
  hygiene (model id + digest, prompt version, cached calls, no temperature).
  The **probe**: the UPR-native question type ("is this text about
  <label>?") per label, with the `noul` and `choice` types and both prompt
  shapes in the declared prompt grid, each variant logged with its prompt
  version; the `paragraph` shape (state = the text, up to 64 typed questions
  per call) is the cheap one (68 labels → 2 calls per text). The **synthetic
  competence control** beside every arm (without it a chance-level result is
  ambiguous between "the data is at the ceiling" and "the model is the wrong
  shape"). The **grid arm** (the decision model as an arm of the UPR
  evaluation, over the whole (text × label) grid) and the **safe-negative
  rule** in its three forms (the fixed 0.5 cut, the fit-side threshold, the
  margin rule) with the **label-blind counterpart** beside them (the same
  model, the same prompts, every text's label list replaced by a deterministic
  shuffle) — the yardstick that turns the rule's cost into a *quality* claim.
  The **shortlist rerank** (the fusion's top-K labels per text, re-decided by
  the decision model, with the shortlist's own recall as the ceiling, against
  the cosine's own hit@1 and the decision model's own). The **probability
  compression** is measured, not assumed. The deliverable's honesty rule
  carries over: the non-gold labels' true positive rate is **unmeasured**, so
  any rule this produces ships as a **reach claim + a cost claim, with no
  precision claim**. *Open for the reviewer:* the Phase-6 slot decision (a
  decision model returns probabilities with no decoder — arguably a zero-shot
  cross-encoder rather than an LLM reranker) stays the reviewer's call; the
  decision models enter as **measurement instruments** only.
  *(done 2026-10-26 — **with one scoped deviation, the reviewer's call**: the
  new command `decide-upr` (the UPR counterpart of `decide-omct`), the pure
  core `text_processors/decision/upr_probe.py` (the UPR question text, the
  (text, label) cells, the label-blind counterpart, the measurement: the
  harness's own metrics, the calibration and the probability compression), the
  driver `upr_decision.py`, the config section `upr_analysis.decision` (the
  models, the scope, the sample size, the seed, the competence control, the
  mismatched control) and the sweep
  `experiments_text_processing/upr_decision_sweep.py`: five runs
  `upr-decide-{tev1,laya}-{paragraph,pair}-{noul,choice}`. The **deviation**:
  the plan's wording asks for the **grid** arm (the whole (text x label)
  grid), the safe-negative rule and the shortlist rerank; the grid's cost is
  now measured — the `paragraph` shape is ~2 calls per text (~1,320 calls per
  model for the val grid) and the `pair` shape is **one call per cell** (the
  pair-shape arm made 1,360 calls for 20 texts = 68 calls per text, so the
  full grid in that shape is ~44,880 calls per arm per model) — so this change
  set measures a **seeded sample** (150 of the 660 val texts for the paragraph
  shape, 20 for the pair shape) and leaves the full grid, the rule and the
  rerank to a follow-up (**CS-9b**) with that cost in front of it. The
  artifact carries the pipeline's own scoring of the **same cells**, so every
  probe number has its reference beside it on the same texts.
  **Verdicts:** **tev1 beats the champion on UPR** — the micro-F1 0.5482
  against the champion's 0.5415 on the same 150 texts, the nDCG@5 0.7055
  against 0.6565, and a near-perfect **candidate generator** (the shortlist
  R@50 **0.9978** against the champion's 0.7322: the decision model puts the
  gold label in the text's top-50 of 68 labels almost always, where the
  champion's kNN and prototype terms lose 27 points of gold cells — the most
  Phase-6-relevant number the track has measured on UPR). The **label-blind
  counterpart is at chance** (0.0365), so the quality is label-driven, and the
  **competence control passes** (the AUC 0.9816-0.9896). **laya is much
  weaker** (0.2226/0.1960) but its shortlist is nearly as good (0.9956): its
  loss is a *top-1* loss, not a ranking loss. The prompt grid flips on UPR:
  the **`noul` type beats the `choice` type** for tev1 (0.5482 against
  0.5349; the opposite of OMCT), while the `choice` type is better calibrated
  (the ECE 0.1553 against 0.1971). And the decision model **finds the eight
  unseen labels** (the unseen bucket's micro-F1 0.4706, the coverage 1.000,
  the nDCG@5 0.8737, against the champion's 0.0000/0.000/0.6046 and the
  label-blind counterpart's 0.1090) — the champion's geometry cannot suggest
  a label it has never seen, and a decision model can. The **probability
  compression is measured**: tev1 is **not** compressed (the gold cells' mean
  0.7578, 87% of them above 0.5), so a fixed 0.5 cut is a meaningful
  operating point on UPR (the opposite of the CS-7b arm on OMCT), while laya
  is the compressed one (its `noul` arm's gold mean 0.4852). The numbers are
  in the track log's CS-9 section; 557 unit tests green.)*
- [x] **CS-9b — the full-val pointwise grid, the safe-negative rule and the
  shortlist rerank (the follow-up CS-9's own deviation promised).** The
  `upr_analysis.decision.scope: grid` scope scores the **whole val grid**
  (660 texts x 68 labels = 44,880 cells) instead of the CS-9 sample's 150
  texts, on the **tev1** arms (the `paragraph` shape, the `noul` and the
  `choice` types) — the shape the pipeline would use and the shape whose cost
  is known (2 calls per text in the `paragraph` shape against the `pair`
  shape's 68). Three new blocks in the `decisions.json` artifact:
  - the **safe-negative rule** in its three forms — the **fixed 0.5 cut**, the
    **fit-side threshold** and the **margin rule** (a cell at least X below
    its own text's best candidate, X fitted on the fit side) — with the
    **label-blind counterpart** beside each form. The rule's **fit side is
    the train split** (its grid's decision probabilities, computed here and
    cached), so no threshold is ever fitted on the evaluated split. The
    UPR-side deliverable keeps the OMCT honesty rule: a **reach** claim (how
    many of the 44,880 cells the rule calls safe negatives) and a **cost**
    claim (how many of the 1,312 gold cells it discards), **never a precision
    claim** (the background cells' true positive rate is unmeasured).
  - the **shortlist rerank**: the champion's top-K labels per text, re-ordered
    by the decision model's own probability. The full grid is already scored,
    so the rerank costs **zero** additional calls; the shortlist's own recall
    is the ceiling, and the champion's own hit@1 and the decision model's own
    sit beside it. This is the Phase-6 shape; it is **not** P4's union (a
    hard shortlist against a soft score combination) — the two are different
    measurements and must never be conflated.
  - the **train-side** decision probabilities (the fit side for the rule's
    threshold and, later, for CS-13's feature), computed once and cached.
  *Pre-registered expectation:* the **fit-side threshold** is the best of the
  three forms (the OMCT verdict, CS-7b), and the **label-blind counterpart's
  harm is materially larger** at the same reach (the label-driven signature
  that turned the OMCT rule's cost into a quality claim). If the blind
  counterpart's harm is *not* larger at the same reach, the rule's cost is a
  construction artefact and the deliverable ships as a reach claim only.
  *Cost:* 11,984 new decision calls (5,992 per arm: the val grid's tev1 arm
  1,320, its label-blind counterpart 1,320, and the train grid's fit side
  3,352 — the counterpart takes the **same** operating point, so it needs no
  fit side of its own, the CS-7b precedent). The measured cache throughput is
  ~100 calls/min, so the wall time is about two hours. The train grid's calls
  are **reused** by CS-13 (so CS-13's marginal cost is zero).
  *Leakage:* the thresholds and the margin are fitted on the **train** grid
  only; the decision model is frozen and zero-shot (it carries no UPR
  label); the label-blind counterpart is the yardstick that says how much of
  the cost is the model's own label-blind signal; val is the evaluated split
  and `test` is not touched.
  *Not in this change set:* `laya` (a second decision model is closed as a
  lever — tev1 beats it by 0.33 — and its full grid costs 4 calls per text)
  and the `pair` shape (its full grid is 44,880 calls per arm for a shape
  whose cost is already measured and which no proposal uses). Both stay at
  their CS-9 samples, which remain their record. This exclusion is the
  reviewer's call and is one line to reverse.
  *Exit:* the full-val pointwise arm (the reference for CS-12), the rule's
  three forms with their label-blind counterparts, the shortlist rerank and
  the train-side probabilities, all in the artifact; the table carries the
  rows; the log carries the numbers.
  *(done 2026-10-27: the new pure module `text_processors/decision/upr_rule.py`
  (the rule's three forms, the margins, the label-blind counterpart **at the
  same operating point** — the CS-7b OMCT precedent) and the driver stage
  (`run_upr_rule_stage`), the config section `upr_analysis.decision.rule` (off
  by default), the sweep `experiments_text_processing/upr_decision_grid_sweep.py`
  and 2 runs `upr-decide-tev1-paragraph-{noul,choice}-grid` (the whole val
  grid; 0 truncated cells). **Verdicts:** the decision model **beats the
  champion** on the full val grid (the micro-F1 0.5502 against 0.5479, the
  nDCG@5 0.7247 against 0.6725, the shortlist R@50 0.9987 against 0.7423)
  and it finds the eight held-out labels (the unseen bucket 0.5267 against
  0.0000); the label-blind counterpart is at chance (0.0517), and the CS-9
  sample's findings hold at full-val scale (the `noul` type beats the
  `choice` type; the `choice` type is better calibrated; the probabilities
  are not compressed). The **rule's cost is a quality claim**: the
  label-blind counterpart discards **75-85%** of the gold cells where the real
  arm discards 13-20% at a similar reach, so the UPR deliverable's cost claim
  is label-driven (the background's true positive rate stays unmeasured, so
  there is still **no precision claim**). The best form on UPR is the
  **margin rule** (the coverage 0.9143 at the harm 0.1502) and **not** the
  fit-side threshold (0.9328 at 0.2012) — the OMCT verdict does not
  transfer, so that pre-registered clause is **not met**. The **shortlist
  rerank does not ship**: the reranked hit@1 (0.3521 at K = 5) is below the
  champion's own (0.3666) at every K, so that pre-registered falsification
  fires — but the same table says where the ceiling is: the **shortlist's own
  recall** (0.7462 at K = 10) is the binding constraint, while the decision
  model alone reaches 0.9987, so a Phase-6 reranker has to be fed the
  decision model's candidates and has to be **trained**. The numbers are in the
  track log's CS-9b section.)*
- [ ] **CS-10 — the LLM stage on UPR (needs the reviewer's explicit OK).** The
  **exp09-style rerank** on the embedding shortlist with the local models
  (gemma4:e2b, llama3.2:3b, gpt-oss — the decision models are far too small
  for a generative role) and the **side enrichment**, which on UPR becomes
  **label-side enrichment** (LLM-written label descriptions or paraphrases
  embedded as extra label vectors). Two things to raise with the reviewer
  first: `PLAN.md` Phase 6 crossed out "LLM reranker via Ollama" — *Skipped by
  decision (2026-10): no LLMs in the pipeline* — so this stage is
  measurement-only, exactly like CS-9; and the label enrichment would touch
  the **train-time index build**, a different kind of intervention from
  scoring. If the reviewer prefers, the label-description catalog that
  CS-8d needs can be produced here instead (a human-readable `labels.jsonl`
  with name / description / parent — the config hook the plan already
  provides).


### 7.1.1 The reviewer's decisions of 2026-10-27 (they bind the change sets below)

Recorded here because **the plan is where the reviewer's decisions live**
(AGENTS.md rule 5: if something is ambiguous, ask — and the answer is written
down, not remembered).

1. **P1 (the listwise stage) runs on the FULL val split**, not the 150-text
   sample: the decision model is cheap enough (1,320 calls for the whole val
   grid in the `paragraph` shape) that the sample buys little, and the
   *ordering* is what the sample measures worst. A 3-text smoke test comes
   first, because the 20-option `choice` has never been exercised against
   the server.
2. **The previous `upr_decision` probe also gets a FULL run** — that is
   **CS-9b** above: the full val grid for the pointwise arms, the
   safe-negative rule in its three forms with the label-blind counterpart
   beside each, and the shortlist rerank. The 150-text sample was CS-9's
   deliberate scope deviation with the cost in front of it; the cost is now
   known and small.
3. **P1-P4 are recorded as new numbered change sets** (one checkbox each,
   the plan's style), each with the mechanism, the pre-registered
   falsification, the cost and the leakage check written out: **P3 -> CS-11**,
   **P1 -> CS-12**, **P2 -> CS-13**, **P4 -> CS-14**.
4. **The catalog is LLM-written**, subject to the freeze-before-the-runs rule
   in CS-11: the writer sees the label name and the public taxonomy only,
   never the records, and the catalog is frozen and committed before the runs
   that use it.

**The execution order** (the numbering skips CS-10, which is blocked on the
reviewer's explicit OK and is a separate branch):

    CS-9b -> CS-11 (P3, the catalog) -> CS-12 (P1, the listwise) ->
    CS-13 (P2, the decision probability) -> CS-14 (P4, the union)

CS-11 comes **before** CS-12 because the listwise question's `criteria` field
carries the label's description (P3's third use), so the catalog must exist
and be frozen before the listwise run.

**Settled and closed on 2026-10-27 — do not reopen:** more chunking families
(153 arms already; the pooled null is measured and the long-text tail is 25
texts); text length as a feature (forbidden by the plan's leakage rules and
confounded); fine-tuning a decision model (forbidden here — the OMCT
questions are shared across documents with conflicting labels); a second or
bigger decision model (tev1 beats laya by 0.33); reading the decision
answer's `confidence` field as a probability (the port's own docstring says
it is not one); the R@k curve and its tie-break (the shortlist metric and
the ranking metric are the same number, the artifacts already carry
recall@1/@3/@5/@10 and shortlist_recall@20/@50/@100, and the champion, the
decision model and the union have zero ties — the kNN arm's tie-break is a
main-plan Phase 4 matter).


### 7.1.2 The new change sets (P1-P4)

- [x] **CS-11 — the label catalog (name / description / parent) (P3).** The
  repo has no label catalog, so the UPR pack's two catalog-gated feature
  groups (`label_part`, `label_type`) cannot be selected and the label's
  *description* cannot be used anywhere. This change set writes the catalog
  and measures its three proposed uses:
  - **the description as its own channel** — the label's description as the
    label-side **query** (the third point of the label-text comparison: the
    bare name, the config template and the description), measured in the
    learned pack beside the template (the control) and the name;
  - **the parent as a sibling-max backoff** — a label with **no fit-side
    support** (zero gold cells in train) inherits the score of a taxonomy
    sibling: `score'(text, L) = max(score(text, L), lambda * max over the
    siblings S of score(text, S))`, with `lambda` a pre-registered grid
    (0.5, 1.0) and **no fitting on val**;
  - **the description as the wording inside CS-12's comparison question**
    (the `criteria` field) — the reason the catalog is written *before*
    CS-12, not after it.
  Plus the pack's two **catalog-gated feature groups** (`label_part`: the
  label-description coverage; `label_type`: the label-type one-hots), which
  the catalog unlocks for free (the code exists; only the input was
  missing).
  **The catalog itself** (the reviewer's decision, 2026-10-27): it is
  **LLM-written** — the writer sees the **68 label names and the public
  taxonomy only**, never a UPR record and never a label-frequency statistic
  — and it is **frozen and committed before the runs that use it** (a catalog
  written after looking at val is a leak). It lives in
  `catalogs/upr_labels.jsonl` (one JSON object per label: `name`,
  `description`, `parent`) with its provenance in
  `catalogs/upr_labels.meta.json` (the writer, the date, the sources, the
  freeze's commit). The repo's existing hook (`paths.label_catalog`) reads
  it; the raw data is untouched.
  *Pre-registered falsification:*
  - the **description channel**: the description arm's **unseen** bucket (the
    generic labels — "General", "Other", "ESC rights - general" — whose
    names are weak) must rise with the paired interval excluding zero, and
    its pooled micro-F1 must not fall by more than 0.01. The scratch's
    finding (a description beats the name where the name is weak and loses
    where the name is already distinctive) predicts exactly this trade: the
    description is a **specialist**, not a replacement for the name;
  - the **sibling backoff**: it must raise the unseen labels that *have* a
    supported sibling, and the **control** — the same rule applied to the
    fit-side-chosen **seen** labels — must show the backoff **not helping**
    them. If the backoff helps the seen labels too, the honest claim changes
    to "another correlated signal" and the log says so;
  - a label with **no supported sibling** (a catch-all, if the catalog gives
    it none) is a **measured null by construction** — the catalog decides,
    and the result is reported, not engineered around.
  *Cost:* **zero model calls** for the experiments (the catalog's LLM writing
  is a one-off outside the harness) and 68 new label embeddings (one new
  label-embedding run for the description query). The arms are the learned
  pack's arms; the pack and the combiner are cheap.
  *Leakage:* the catalog is **label-side knowledge** (the same kind of thing
  the zero-shot label vectors are allowed to use), written from the label
  names and the public taxonomy only and frozen before the runs; which labels
  get the backoff (zero fit-side support) and which labels form the control
  are decided on the **fit side**; the fit side is train only; val is the
  evaluated split; `test` is not touched.
  *Runs:* the input artifacts `catalogs/upr_labels.jsonl` (+ its meta) and
  `upr-labelembed-embeddinggemma-desc`; the guard's reference run for the
  description label side `upr-eval-fusionc-embeddinggemma-k10-c1-desc` (the
  guard compares the run's whole-text reference arm against an `evaluate` run
  with the **same** label vectors, so a new label side needs its own
  reference); the arms
  `upr-analyze-learned-cat-desc` (the label query = the description),
  `upr-analyze-learned-cat-feat` (the catalog-gated feature groups on),
  `upr-analyze-learned-cat-backoff` (the sibling backoff) and
  `upr-analyze-learned-cat-desc-backoff` (the description query + the
  backoff), against the control `upr-analyze-learned-cat-control` (the
  CS-8f grid's best arm, **re-run in this sweep** so the cross-run paired
  comparison has both sides' per-text blocks — the re-run reproduces the
  CS-8f artifact's metrics exactly, so it doubles as the reproducibility
  check). The **bare name** is not an arm: the name is the pack's *lexical*
  block's text (it is in every pack), so the single-variable change is the
  **query** (the template against the description) — and a name-only query
  would need a new main-pipeline `evaluate` run to guard its reference arm.
  *Exit:* the catalog is committed and frozen; the arms are measured with the
  per-bucket view, the three floors and the paired intervals against the
  control; the log carries the numbers and a verdict on each of the three
  uses.
  *(done 2026-10-27: the catalog `catalogs/upr_labels.jsonl` (+ its meta: the
  writer, the sources, the sha256 and the freeze) — **LLM-written from the 68
  label names and the public taxonomy only**, frozen and committed before the
  runs, 12 parents, the label space covered exactly; the new pure module
  `text_processors/features/upr_backoff.py` (the sibling-max backoff, the
  unsupported labels, the control's labels and the sibling sets), the config
  section `upr_analysis.learned.backoff` (off by default), the **description
  label-text variant** (`label_texts.variant: description` — the config
  already reserved it) with the labelembed command's catalog rendering, the
  artifact's **per-text** summary (the decision view's (tp, fp, fn) counts and
  the ranking view's per-text nDCG, overall and per label bucket) and the new
  repo tool `tools/compare_upr_arms.py` (the **cross-run** paired comparison:
  two arms in two different runs, the text the resampling unit); 7 runs (the
  two input artifacts, the guard's reference and the four arms + the
  control). The guard holds for every `analyze-upr` run, and the control
  reproduces the CS-8f artifact's metrics exactly. **Verdicts:** the
  **description channel is a real, label-driven win** (the description as the
  label's query +0.0266 of micro-F1 [+0.0084, +0.0435] against the control;
  the catalog-gated groups +0.0258 [+0.0091, +0.0433]; the pooled intervals
  exclude zero in both), but the **unseen** bucket's rise (+0.0221 and
  +0.0355) has an interval that **includes** zero, so the pre-registered
  unseen-bucket condition is **not met** — the honest claim is "the
  description adds a signal the name does not have, everywhere", not "it
  rescues the held-out labels"; the description is a **weaker channel than
  the name pooled** (the per-feature AUC 0.7067 against the name's 0.7566)
  and the pack gains because the two are complementary. The **sibling-max
  backoff does not ship**: the mechanism acts (20-81% of the unsupported
  labels' cells change, the mean lift +0.011 to +0.083; the two catch-alls
  have no supported sibling — a null by construction) but the decision view
  **falls** (the unseen bucket −0.0356 [−0.0579, −0.0117] at λ = 0.5 and
  −0.1080 [−0.1505, −0.0626] at λ = 1), because a **max** over the siblings
  transfers a relevance *peak*, not the relevance — the pre-registered
  falsification fires. The **parent one-hots** are nearly empty on their own
  (the AUC 0.4705-0.5433). The numbers are in the track log's CS-11
  section; 582 unit tests green.)*
- [x] **CS-12 — the listwise second decision stage (P1).** A **second round**
  on top of the pointwise arm: round 1 is the existing 68 pointwise `noul`
  questions per text (unchanged, and **free** — the decision cache already
  holds them from CS-9b), and round 2 is **one** call per text asking "which
  of these 20 topics is the text about?", the 20 options being the round-1
  top-20 and each option's `criteria` its **catalog description** (CS-11).
  The hypothesis: the decision model is good at **finding** the candidates
  and weak at **ordering** them, and the ordering is where the pipeline's
  `top_k k=2` lives.
  The arm's final ranking is a **rank fusion** (parameter-free, so nothing is
  fitted on val): the round-2 order over the 20 candidates, then the
  remaining 48 in the round-1 order. The arm's score matrix is the negative
  final rank, so every rank-based metric (the decision view's top-2, the
  nDCG, the recall@k, the shortlist recall) is well defined; the
  **calibration** and the **compression** blocks are explicitly **not
  applicable** (a within-text distribution over 20 options is not a per-cell
  probability) and are flagged as such in the artifact.
  *Pre-registered falsification:* the listwise arm's micro-F1 (the top-2
  view) must beat the **measured** full-val pointwise arm's (CS-9b) with a
  **paired** bootstrap interval excluding zero (the text is the resampling
  unit, 2,000 resamples, seed 42 — `eval.multilabel.paired_metric_bootstrap`),
  and the listwise arm's **unseen** bucket micro-F1 must not fall below the
  pointwise arm's. The scratch predicted 0.5489 overall and 0.4706 unseen;
  those are **indicative** (a scratch computation, not a harness result, on
  a different population than the run that measures the comparison), so the
  operative criterion is the **relative** one and the absolute numbers are
  reported beside it. If the listwise arm's top-2 equals the pointwise arm's
  top-2 on every text (to four decimals), the comparative mechanism buys
  nothing and the arm is recorded as a null.
  The **controls:** the listwise arm with the **name-only** criteria (no
  description) isolates the *listwise mechanism* from the *descriptions*; the
  **label-blind counterpart** (the wrong pairing, the same pipeline) must
  stay at chance, which is what makes the gain label-driven.
  *Feasibility first:* a 20-option `choice` has never been exercised against
  the server (every track `choice` arm is two-option), so the shape is
  **smoke-tested on 3 texts** before the run; if the server refuses it or
  degrades on it, the shape is abandoned and the plan records that (a
  feasibility check, not a metric: no run, no number, no row). The call
  planner must also **count the options and the criteria** in the estimated
  input size (today it counts only the instructions and the state), or a
  20-option call could exceed tev1's 2,500-token budget unnoticed.
  *Cost:* 2,640 new decision calls (the listwise round 2 for the real arm
  660 and for the label-blind counterpart 660, twice over for the two
  criteria variants); round 1 is free (the cache).
  *Leakage:* the round-2 options are the round-1 model's **own output** (no
  label is involved); the criteria are **label-side catalog text**, frozen
  before the run; the model is frozen and zero-shot; the fusion is
  parameter-free (nothing fitted on val); the train split is not used.
  *Runs:* `upr-decide-tev1-paragraph-noul-listwise` (the pointwise arm + the
  listwise stage with the catalog's criteria) and
  `upr-decide-tev1-paragraph-noul-listwise-nodesc` (the name-only control).
  *Exit:* both arms are measured on the full val grid with the paired
  interval against CS-9b's pointwise arm, the per-bucket view and the
  label-blind counterpart; the log carries the numbers and the verdict.
  *(done 2026-10-27: the new pure module
  `text_processors/decision/upr_listwise.py` (the second round's question
  spec, the call plan that counts the options **and** the criteria in the
  input budget, the candidates and the rank fusion), the driver's listwise
  stage, the config section `upr_analysis.decision.listwise` (off by
  default), the measurement's `probability_scores` flag (the calibration and
  the compression are reported as **not applicable** for a rank-based arm),
  the sweep `experiments_text_processing/upr_listwise_sweep.py` and 2 runs
  `upr-decide-tev1-paragraph-noul-listwise{,-nodesc}` (the whole val grid;
  660 calls per arm; 0 truncated cells). The 20-option shape was
  **smoke-tested on 3 texts first**, as the plan required: the server accepts
  it. **Verdict: the listwise stage does not ship** — the pre-registered
  falsification fires on both clauses: the listwise arm's micro-F1 does not
  beat the pointwise arm's (0.5205 against 0.5502, the paired difference
  −0.0296 [−0.0535, −0.0038]) and its unseen bucket falls by 0.0609
  [−0.1106, −0.0078]. The mechanism is not inert — the **first** pick
  improves (the hit@1 0.3613 against 0.3483) while the second degrades —
  and the **descriptions make it worse** than the name-only control
  (−0.0296 against −0.0175). The order of the reviewer's hypothesis,
  restated by the measurement: the model's **explicit** ordering (a 20-way
  choice) is **worse** than its **implicit** one (the 68 probabilities'
  order), so the decision model is an excellent candidate generator (the
  shortlist R@50 0.9987 against the champion's 0.7423) and should not be
  asked to order. The label-blind counterpart is at chance (0.0433). The
  numbers are in the track log's CS-12 section.)*
- [x] **CS-13 — the decision probability as a feature in the UPR pack (P2).**
  One more column in the pack's `(n_texts, n_labels, n_features)` cube: the
  decision model's probability for the (text, label) cell. The pack stays a
  pure core (the matrix is passed in; the model is called at the edge), and
  the feature is **cell-local**: no corpus statistic, no document identity, no
  length. The combiner is **sklearn's HistGB** (the plan's CS-6b defers
  `lightgbm` until HistGB is beaten, and HistGB is already a core dep and
  beats the logistic on UPR); the logistic arm is measured beside it.
  The hypothesis: the pack's lexical block finds five of the eight unseen
  labels because the **name-vs-text** mechanism is identical for a seen and
  an unseen label, so its fitted weight transfers; the decision probability
  has the same property (it is computed from the label's **name**, identically
  for a held-out label). The fit side has **zero** positives for the eight
  unseen labels (0 of 3,087 train gold cells), so the combiner must
  **transfer** the weight — that is the hypothesis, and it is what to
  measure.
  *Pre-registered falsification:* the augmented pack's **unseen** bucket
  micro-F1 must be **not below** the control's (0.4758 — a harness result
  from CS-8d, so directly comparable) and its shortlist R@50 not below the
  control's (0.8692); the **paired** interval of the difference (augmented
  minus control) is reported for every metric, and a *win* is claimed only
  where the interval excludes zero; the augmented pack's own **control-role
  floor** must stay at chance (the CS-8d level, 0.027-0.043), which is what
  says the gain is label-driven; and the arm with the **label-blind**
  decision feature (the same pack, the wrong pairing) must not beat the real
  arm.
  *Cost:* 3,352 new decision calls — the **label-blind** counterpart's fit
  side (the train grid, 3,352 calls). Everything else is a cache hit: the
  train and val probabilities of the real arm are CS-9b's (the rule's fit
  side needs them) and the blind val side is CS-9b's too.
  *Leakage:* the feature is cell-local and label-blind in the fit sense (the
  model is frozen and carries no UPR label); the fit side is **train** only;
  the eight unseen labels' absence from the fit side is the *hypothesis*, not
  a leak (nothing about them is used to fit); val is the evaluated split;
  `test` is not touched.
  *Runs:* `upr-analyze-learned-dec-histgb` and
  `upr-analyze-learned-dec-logistic` (the pack + the decision feature),
  `upr-analyze-learned-dec-blind-histgb` and
  `upr-analyze-learned-dec-blind-logistic` (the same, with the label-blind
  decision feature), against the control `upr-analyze-learned-cat-control`
  (the same pack and combiner without the feature, re-run in the CS-11 sweep
  so the cross-run paired comparison has both sides' per-text blocks; it
  reproduces the CS-8f artifact's metrics exactly).
  *Exit:* the four arms are measured with the per-bucket view, the three
  floors, the calibration and the paired intervals against the controls; the
  log carries the numbers and the verdict (including the "the weight
  transferred / it did not" answer).
  *(done 2026-10-27: the pack's new **decision** feature group (the decision
  probability as one more column of the cube — the model is called at the
  edge and the matrix is passed in, so the feature is cell-local by
  construction), the config section `upr_analysis.learned.decision_feature`
  (off by default; the model pinned by id and digest), the driver's scoring
  of the fit side and the evaluated side, the sweep
  `experiments_text_processing/upr_learned_decision_sweep.py` and 4 runs
  (`upr-analyze-learned-dec-{histgb,logistic}` and their two label-blind
  counterparts). **Verdict: the hypothesis is confirmed, and it is the
  track's largest UPR result.** The augmented pack's pooled micro-F1 is
  **0.5965** (the boosted combiner) against the control's 0.4818 — the
  paired difference **+0.1147** [+0.0964, +0.1349] — and its **unseen**
  bucket **0.5697** against 0.4469 (+0.1228 [+0.0763, +0.1718]): the fit side
  has **zero** positives for the eight held-out labels, so the weight
  **transfers** (the decision probability is computed from the label's name,
  identically for a held-out label). Every pre-registered clause holds: the
  unseen bucket is not below the control's, the paired intervals exclude
  zero, the augmented pack's own **control-role floor** stays at chance
  (0.0120), and the **label-blind** counterpart's arm reaches only 0.4977
  (+0.0160 over the control) — so **93% of the feature's contribution is
  the model's label reading**. The **random-vector floor** keeps 0.5798: the
  decision probability carries the ranking and the embedding geometry
  contributes almost nothing (0.5965 against 0.5798). The augmented arm's
  micro-F1 (0.5965) is above the champion's (0.5479) — the first UPR number
  the track has measured that beats the main pipeline's whole-text scorer at
  the decision view, and it does so while **finding the eight held-out
  labels** (the unseen bucket 0.5697 against the champion's 0.0000). The
  price is stated: the decision model in the loop (2 calls per text). The
  numbers are in the track log's CS-13 section.)*
- [x] **CS-14 — the union candidate generator as a number (P4).** The
  **measurement** of the union — the per-row **z-max** of the champion's and
  the decision model's scores (the row = the text, the z-score over the 68
  labels) — and nothing else: the union is **not built** and enters no
  pipeline. The 2026-10-22 trim put "union shortlists" out of the track's
  menu (they are main-plan Phase 4/6 machinery), so this change set produces
  a number for PLAN, not a component.
  *Pre-registered falsification:* the union's shortlist recall@10 (and the
  whole recall curve) is reported on the full val grid, beside the
  champion's and the decision model's own, with the **control**: the same
  z-max with the decision model's **label-blind** scores. If the blind union
  reaches the same recall, the union's gain is a construction artefact and
  the number is reported as such.
  *Cost:* **zero new calls** (both score matrices are already computed) and
  no new machinery beyond the block.
  *Leakage:* the z-score is a **per-row** normalization (no statistic across
  texts, no fit, no threshold); nothing is fitted on val; the blind control
  is the yardstick; `test` is not touched.
  *Runs:* `upr-decide-tev1-paragraph-noul-grid-union` (the union block on;
  the pointwise arm is a cache hit).
  *Exit:* the union's recall curve is in the artifact and the table beside
  the two references, the blind control beside it, and the log states
  plainly that the union is **not** built and that the slot decision is
  PLAN's (and the reviewer's).
  *(done 2026-10-27: the new pure module
  `text_processors/decision/upr_union.py` (the per-row z-score, the z-max
  and the block), the config flag `upr_analysis.decision.union`, the
  driver's union stage, the sweep
  `experiments_text_processing/upr_union_sweep.py` and 1 run
  `upr-decide-tev1-paragraph-noul-grid-union` (the whole val grid; **zero**
  new model calls — the decision calls are CS-9b's cache hits). **Verdict:**
  the union's **recall@10 = 0.9120** against the decision model's 0.8845 and
  the champion's 0.7126 — and the **control** (the same z-max with the
  decision model's label-blind scores) reaches only 0.6947, *below* the
  champion's own, so the gain is label-driven. The union is a **measurement,
  not a component** (the 2026-10-22 trim put union shortlists out of the
  track's menu): the number is for PLAN. The union's *decision* view has a
  scale artefact, and it is stated plainly — the union's pooled micro-F1
  (0.5843) is the best number the track has measured on UPR, but its unseen
  bucket (0.2540) is half the decision model's (0.5267), because the
  champion's per-text z-scores are more spread out and the z-max is biased
  toward its scale. The numbers are in the track log's CS-14 section.)*

- [x] **CS-15 — the zero-shot rerankers on UPR, as a *scorer* (the candidate
  set fixed, the ordering changed).** The three HF rerankers, measured as
  **reorderings of a candidate set** (never as a new candidate generator):
  the client (a new edge package `src/cluster_rag/rerank/`: the port, the HF
  client, its own disk cache), the pure core
  (`text_processors/rerank/upr_rerank.py`: the candidate sets, the label-blind
  counterpart, the rank fusion, the ceiling, the measurement), the new CLI
  command `rerank-upr`, the config section `upr_analysis.rerank`, the sweep
  `experiments_text_processing/upr_rerank_sweep.py` and the feasibility tool
  `tools/smoke_upr_rerank.py`. The mechanism, the falsification, the cost and
  the leakage check are in **§7.1.3** (the reviewer's decisions of
  2026-10-27), which this change set executes. The runs: 9 (the three models
  x the three candidate sets), each answering one question.
  *Exit:* every arm's measurement (the decision view, the ranking view, the
  per-bucket view, the gold side), the candidate set's **ceiling** beside the
  incumbent's own numbers, the **label-blind counterpart** beside every arm,
  the **paired** differences over the texts, the **cost** block (the pairs per
  text, the pairs per second, the wall time per 660 texts, the peak GPU
  memory) and the log carries the numbers and the verdict.
  *(done 2026-10-27: the new edge package `src/cluster_rag/rerank/` (the
  port, the Hugging Face client with its three families — the cross-encoder,
  the yes/no head and the **listwise** model — and the rerankers' own disk
  cache), the new pure core `text_processors/rerank/upr_rerank.py` (the
  candidate sets, their **ceilings**, the label side, the label-blind
  counterpart, the rank fusion and the measurement), the new CLI command
  `rerank-upr` (with the pipeline's own scoring as the incumbent and the
  track's **guard** beside it), the config section `upr_analysis.rerank`, the
  sweep `experiments_text_processing/upr_rerank_sweep.py` and the feasibility
  tool `tools/smoke_upr_rerank.py`; 9 runs (`upr-rerank-<model>-<set>`, the
  three models x the three candidate sets). **Verdict: a zero-shot reranker
  does NOT beat the ordering we already have.** Every arm's paired
  difference against the incumbent of the **same** candidate set is negative
  except one, and that one is a null: the best arm (the 1B yes/no head, the
  label's **name**, the label-first order) reaches the micro-F1 0.5593
  against the decision model's 0.5502, a paired **+0.0091** [−0.0122, +0.0294]
  (the nDCG@5 −0.0084 [−0.0266, +0.0093]) — the interval includes zero, so
  the pre-registered ship criterion is **not met** and the null criterion
  fires. The **label-blind counterpart** is catastrophic beside every arm
  (−0.29 to −0.39), so the arms do read the label: the signal is there and
  the *ordering* is the weak part. The **per-bucket** view is the finding:
  the rerankers are a **rare-label specialist** (the 1B model's name arm
  gains **+0.1766** [+0.1141, +0.2419] on the tail, the listwise model's
  name arm **+0.1488** [+0.0889, +0.2086], the held-out bucket's point
  estimates are positive for the description arms) while the head and the
  torso fall — the reviewer's expectation, measured. The **cost** clears the
  pre-registered bar with a wide margin: the candidate-restricted scorer
  form is 10 pairs per text, and the measured throughput projects **4-5 s**
  (the 278M cross-encoder), **33-59 s** (the 1B head) and **22-24 s** (the
  listwise model) per 660 val texts, against the decision model's ~13
  minutes; the peak VRAM is 1.2-5.9 GB. The numbers are in the track log's
  CS-15 section.)*
- [x] **CS-16 — the reranker's score as a **pack feature** (the CS-13
  precedent).** The reranker's score per (text, label) cell becomes one more
  column of the UPR pack's cube, in **two shapes**: the **full grid** (68
  pairs per text, no fill value needed) and the **candidate-restricted**
  feature (the decision model's top-10, 10 pairs per text, with the
  **pre-registered fill value**: the row's minimum scored value). The pack
  stays a pure core (the matrix is scored at the edge and passed in), the
  combiner is the boosted one (CS-13's), the fit side is **train** only, and
  the **label-blind counterpart** runs beside every arm. The mechanism, the
  falsification, the cost and the leakage check are in §7.1.3. The runs:
  `upr-analyze-learned-rerank-<model>-{full,restricted}` (+ their two
  label-blind counterparts) against the re-run control.
  *Exit:* the augmented pack's numbers against the control's with the paired
  intervals, the per-bucket view (the unseen bucket is the one the champion
  cannot reach at all), the three floors, the cost block and the log's
  verdict on the "scorer vs. feature" question.
  *(done 2026-10-27: the pack's new **rerank** feature group (one more column
  of the cube, scored at the edge and passed in, so it is cell-local by
  construction), the config section `upr_analysis.learned.rerank_feature` (the
  two shapes — the full grid and the candidate-restricted one with the
  pre-registered row-minimum fill — the label side, the order and the
  label-blind counterpart), the plumbing of the two matrices through the
  learning driver, the sweep
  `experiments_text_processing/upr_rerank_feature_sweep.py` and 13 runs (the
  control + the two shapes x the three models x the label-blind counterpart);
  the guard is green for every run. **Verdict: the feature form is the one
  that pays, and only the *listwise* model's feature pays.** The listwise
  model's full-grid column lifts the pack from the control's micro-F1 0.5965
  to **0.6117** — the best UPR number the track has measured — with the
  paired difference **+0.0152 [+0.0015, +0.0280]** (the nDCG@5 +0.0194
  [+0.0080, +0.0314]), the unseen bucket not below the control's (0.5766
  against 0.5697, and its nDCG@5 gain +0.0205 [+0.0016, +0.0390] is
  significant), the arm's own control-role floor at chance (0.0122) and the
  **label-blind counterpart at null** (-0.0076, the interval including zero)
  — the yardstick confirms the gain is the model's label reading. **But the
  pre-registered +0.02 margin is NOT met** (+0.0152), so the ship criterion
  is not satisfied and the slot decision stays the reviewer's. The two
  cross-encoders' features are **nulls** in both shapes (every interval
  includes zero). The **cost** is where the listwise model
  wins: its full-grid feature costs 266 s (measured, cold) for the val grid
  **and** the 1,676-text fit side — one call per text — against the 1B head's
  909 s (one pass per pair) and the 278M cross-encoder's 99-141 s (the
  projection from its measured rate); the first clears the cost bar, the
  second misses it. The numbers are in the track log's CS-16 section.)*

### 7.1.3 The reviewer's decisions of 2026-10-27 (the reranker round: CS-15, CS-16)

*Recorded **before** any run, so the pre-registration is auditable. The
round's question: **can a ranking model beat the ordering we already have,
and what does it cost?***

**0. What the three models actually are (the round's premise, corrected by
the model cards on 2026-10-27).** The round was proposed with
`jina-reranker-v3` described as a "ColBERT-style late-interaction,
multi-vector" model. It is not. Its card says: *"a novel last but not late
interaction architecture. Unlike ColBERT's separate encoding with multi-vector
matching, this model performs causal self-attention between query and
documents within the same context window, extracting contextual embeddings
from the last token of each document"* — a **listwise** reranker on
Qwen3-0.6B that takes **one query and up to 64 documents per forward pass**.
Three consequences, and they change the design:

| model | shape | one forward pass per | the card's score |
|---|---|---|---|
| `jinaai/jina-reranker-v2-base-multilingual` (278M) | cross-encoder | one **(query, document) pair** | a sigmoid on the classification logit |
| `nvidia/llama-nemotron-rerank-1b-v2` (1B) | cross-encoder | one **(query, document) pair** | the raw relevance logit |
| `jinaai/jina-reranker-v3` (0.6B) | **listwise** | one **(query, up to 64 documents)** | the relevance score of each document |

So: (i) **none of the three is a multi-vector scorer**, so the 2026-10-22
trim's "multi-vector/late-interaction scoring" clause does not bind this
round at all — the trim's *other* clause (the methods inventory's "exp19:
cross-encoder reranker — our PLAN Phase 6 slot; not this track's core →
deferred") is the one this round acts on; (ii) a reranker's **cost is two
shapes**, not one: the per-pair cross-encoder and the per-call listwise
model, and the listwise model is the **cheap one per text** (one call for a
whole candidate set) and the **expensive one per pair**; (iii) the three are
not comparable with each other, only **with themselves on the same candidate
set**.

**1. Where the work lives (the reviewer's question (a)).** **Both places, with
a strict division of labour.** PLAN.md's Phase 6 already asks for exactly
this ("Cross-encoder reranker on (text, label name/description) pairs.
Zero-shot first, then fine-tuned on the auxiliary pairwise data" plus
"Compare against fusion alone: quality gain vs latency and cost", exiting on
"enabled only if it improves val F1 / P@k by a meaningful margin"), so **the
slot is PLAN's**. But the *measurement* can only be done in the track: the
track owns the UPR harness, the candidate sets, the paired-comparison tool,
the floors, the per-bucket view and the cost accounting. So: **CS-15/CS-16
are track change sets that produce PLAN's Phase 6 evidence**, and the track's
log ends with a note for PLAN (as the track's §2 requires: "a track result
adopted downstream gets a note in the relevant PLAN phase"). The **code** is
not in the track's pure core either: the model client is a new **edge**
package `src/cluster_rag/rerank/` — the package AGENTS.md's target layout
already reserves for the main plan's Phase 6 (`rerank/cross_encoder.py,
llm.py`) — while the track's pure core gets only the pure parts (the
candidate sets, the fusion, the measurement). The 2026-10-22 trim is
**not** reopened: this round adds no multi-vector scoring, no union
shortlist, no new decision channel, and no pipeline component.

**2. The candidate set fixes the ceiling (the reviewer's instruction, kept
in the mechanism).** A reranker can **reorder** a set, never enlarge it. So
the candidate set is the experiment's independent variable, and the
**decision model's** candidates (or the union's) are the ones fed to the
rerankers; the champion's shortlist is measured only as the **control** that
shows the ceiling's dependence on the set. The three sets and their ceilings
(the numbers are the artifacts' own; the two aggregations are named because
they differ):

| the candidate set fed to the reranker | the row-mean recall@10 (the ranking view) | the cell-level hit@10 (the gold cells' share) | the ceiling a **perfect** reordering could reach (micro-F1) | the incumbent ordering of the same set |
|---|---|---|---|---|
| the champion's top-10 | 0.7126 | *measured in the run* | ~0.71 | the champion (0.5479) |
| the **decision model's** top-10 | 0.8845 | 0.8788 | ~0.88 | the decision model (0.5502) |
| the **union's** top-10 | 0.9120 | 0.9276 | ~0.91 | the union (0.5843) |

The ceiling's formula is the reviewer's, unchanged: the perfect reordering
puts the gold label first, so `TP = (the gold cells inside the set)`, the
top-2 rule predicts 1,320 cells, so `FP = 1,320 − TP` and `FN = 1,312 − TP`.
**Which aggregation is used is pre-registered: the cell-level share** (the
artifact's `gold.hit@k`), because the decision view's micro-F1 is a
cell-level statistic and the ceiling arithmetic must be on the same footing;
the row-mean recall@10 is reported beside it, never mixed with it. (The two
differ by up to 1.6 points: the decision model's 0.8845 row-mean against its
0.8788 cell-level, the union's 0.9120 against its 0.9276.)

**3. "The same candidate set, two orderings" (the reviewer's instruction, and
the CS-9b lesson).** Every comparison holds the candidate set **fixed** and
changes **only** the ordering. The reordering is parameter-free: the
candidates are ordered by the reranker's score, and the cells outside the set
keep the incumbent's own order below them (the CS-12 listwise arm's rank
fusion, reused). The incumbent is therefore **the ordering that produced the
set** — the decision model's own probabilities, the union's own z-max, the
champion's own scores — and the comparison is **paired over the texts** (the
per-text (tp, fp, fn) counts and the per-text nDCG, 2,000 resamples, seed 42).
The head-to-head numbers the arms are read against are the track's best:
the **pack + the decision probability** (micro-F1 **0.5965**, hit@1 0.3758)
and the **union's** ranking view (nDCG@5 **0.7693**) — never the champion
alone. The CS-9b mistake (a *weaker* ranker re-ordering the champion's
shortlist, blamed for the result) is avoided by construction: the *same set*,
the *same metric*, the *same texts* on both sides.

**4. The work order: the **scorer** first, then the **feature** (the
reviewer's question (b), and the CS-13 precedent).** The CS-13 measurement is
the reason: the decision probability as a **stand-alone scorer** was worth
+0.0023 of micro-F1 over the champion, and the *same* probability as a **pack
feature** was worth **+0.1147**. So CS-15 measures the reranker as a scorer
(a reordering of a candidate set, 10 pairs per text) and CS-16 measures it as
a feature (a column of the pack's cube, 68 pairs per text in the full-grid
shape). The **fill value** of the cheap, candidate-restricted feature shape is
**pre-registered here**: the **row's minimum scored value** (so the
candidates always outrank the cells outside the set, and no separation is
manufactured by the fill). A constant fill (0.0) is **not** used: it makes
the feature's absolute level arbitrary and lets the combiner's fitted weight
manufacture the separation the reranker did not produce.

**5. The label side and the order (pre-registered).** The label's **document**
is the **catalog's description** (`catalogs/upr_labels.jsonl`, frozen before
the runs, label-side knowledge exactly like the zero-shot label vectors); the
**name-only** variant is the cheap ablation. The **order** of the pair is
pre-registered as **the text first, the label second** (the pipeline's
retrieval framing: the text is the user's query, the labels are the documents
to retrieve) — and it is the only framing the listwise model's API admits
(one query, many documents). The **reverse** order (the label as the query,
the text as the passage: the orientation the two cross-encoders' training
distribution suggests) is **measured**, not assumed: it is the order-ablation
arm of the decision-set run, where it is possible. Both orders are in the
artifact, so the choice is a measurement and not a belief.

**6. The cost bar (PLAN's Phase 6 exit criterion: "quality gain vs latency
and cost").** The bar is stated **in advance**: the reranker's **wall time
for the 660 val texts must not exceed the decision model's** — the bar is
1,320 calls at the measured ~100 calls/min, i.e. **~13 minutes for the val
split** (the CS-9b cost note) — and the **peak GPU memory** must be reported
(16 GB available). A quality gain that costs more than the model it would sit
beside is not "enabled": the run reports the pairs per text, the pairs per
second, the wall time per 660 texts and the peak memory for every arm, and
the **ship** criterion below adds the quality margin. The indicative
throughputs measured on this machine before the runs (a scratch benchmark, not
a harness number): the 278M cross-encoder ~1,485 pairs/s and 0.6 GB of peak
VRAM, the 1B cross-encoder ~467 pairs/s and 2.5 GB, the listwise model
0.265 s per 10-document call and 0.539 s per 64-document call and 1.3 GB.

**7. The pre-registered falsification (the ship / null criteria), in
advance.** For the **scorer** form (CS-15), against the **decision model's
top-10** as the primary candidate set and the **union's** top-10 as the
second:

- **Ship (the mechanism claim).** The reranker's reordering of the set beats
  **the incumbent ordering of the same set** on the **decision view**
  (micro-F1) with a **paired** 95% interval **excluding zero** in the
  positive direction, **and** on the **ranking view** (nDCG@5) with a paired
  interval excluding zero, **and** the **label-blind counterpart** (the same
  reranker, the shifted pairing — the yardstick) shows **no** such gain.
- **Null.** The paired interval includes zero on either view, or the
  label-blind counterpart shows a comparable gain (a construction artefact).
- **The pipeline claim** (a *different*, stronger bar, and it is the
  reviewer's slot decision, not the track's): the reranked arm's micro-F1
  must beat the **pack + the decision probability** (0.5965) with a paired
  interval excluding zero, and by at least **+0.02** ("a meaningful margin",
  PLAN's Phase 6 wording, with the number fixed in advance).
- For the **feature** form (CS-16), against the **control** (the same pack
  and combiner **without** the reranker's column): the augmented pack's
  micro-F1 must beat the control's with a paired interval excluding zero,
  the **unseen bucket must not fall below** the control's, the augmented
  pack's own **control-role floor** must stay at chance, and the
  **label-blind** arm must not beat the real one. The ship criterion adds the
  same +0.02 margin and the cost bar.

**8. The dependencies this round needs (AGENTS.md §3's explain-first).** Two,
both recorded here: (i) **`einops`** — the jina-reranker-v2 card's own
instruction ("install transformers and einops"); its custom modelling code
imports it, and without it the model does not load. It is a pure-Python
library (no model download, no network, no I/O) and it goes into the core
dependencies with a comment, like `httpx` and `peft`. (ii) A **one-function
compatibility shim** for `transformers` 5.x: the jina-reranker-v2 custom
modelling code (pinned at revision `9cfeff2d`) imports
`create_position_ids_from_input_ids` from
`transformers.models.xlm_roberta.modeling_xlm_roberta`, which transformers 5
removed. The client installs the transformers-4.x implementation of that one
helper before the dynamic module is loaded. **A transformers downgrade is
explicitly rejected**: the repo's embedding artifacts were produced with
transformers 5.19, so a downgrade would change the embeddings and break the
UPR guard (`tools/verify_upr_baseline.py`) — the guard is the track's
contract. The shim is contained in the edge client, is documented, and is
covered by a test.

**8b. The dtype is a measurement decision, not a default (2026-10-27).**
The two cross-encoders run in **float32**, not in their checkpoints' own
bfloat16, because in bfloat16 a **score depends on the batch's
composition** (the padding and the kernels' accumulation order): the
measured wobble is up to **0.007** on the 278M model's [0, 1] scale and up
to **0.3 logits** on the 1B model's scale, which is enough to **flip a
near-tie** — i.e. to let the GPU's rounding decide a measurement. In
float32 the wobble is 1e-6 (the 278M model) and 7e-5 (the 1B model) and the
ranking is identical across the batch sizes. The price is the throughput
(898 pairs/s instead of 1,485 for the 278M model, 94 instead of 467 for the
1B one), which the cost bar below still clears. The **listwise** model stays
in its own dtype: it reads one request per forward pass (no batch padding),
so its score is already batch-independent.

**9. The prize one step further out (acknowledged, not scheduled).** The
**fine-tuned** cross-encoder is PLAN Phase 6's second step, and the track now
holds all three of its ingredients: the characterized OMCT benchmark (the 996
verified hard negatives), the UPR hard negatives from the safe-negative rule
(CS-9b), and the decision model's 99.87%-recall candidate set as the positive
mining source. The zero-shot measurement of CS-15/CS-16 is the cheap first
step, **not** the destination: CS-17 names the fine-tuning change set and
leaves it open, because it needs the reviewer's OK on the training target and
on a training split of its own.


## The moved PLAN.md Phase 4 opening — the embedding-pool widening sweep

### Phase 4 opening: embedding-pool widening (settled with the reviewer 2026-10-20)

Decided, not re-litigated: the candidate pool widens from the recorded 6 to **14
single-vector text embedders** (bi-encoders only — the no-LLM ruling also excludes
generative and multi-vector/late-interaction models from the pool; multi-vector
rerankers remain a Phase 6 topic). New entries with card-verified prefixes, pinned
HF revisions / Ollama digests at run time: `embeddinggemma-2` (Ollama; v1 task
templates), `nomic-embed-text-v2-moe` (Ollama; `search_query: ` / `search_document: `
required, ctx 512), `jinaai/jina-embeddings-v5-text-small` (cc-by-nc-4.0 accepted for
research; ST config prompts `Query: ` / `Document: ` + `default_task`),
`voyageai/voyage-4-nano` (Apache-2.0; retrieval prompts, 2048d default MRL),
`ibm-granite/granite-embedding-english-r2` (no prefixes, 768d, ctx 8192),
`codefuse-ai/F2LLM-v2-80M` (Instruct query prompt, bare documents, 320d MRL),
`intfloat/e5-large-v2` (e5 prefixes, **1024d** — the earlier 768d note was a slip),
`sentence-transformers/all-mpnet-base-v2` (no prefixes, ctx 128 — truncates long OMCT
paragraphs; UPR texts fit). Work order, one checkbox per change set:

- [x] **A. OMCT probe sweep widened to the 14-model pool.** Extend
  `experiments/omct_probe_sweep.py` with the 8 new `ProbeCell`s (label
  `probe2-<detail>`, card-verified `PrefixesConfig`, revisions pinned / digests
  recorded at run time). Wiring accommodations for the new models (small, in
  `embed/` + `config.py`): `embed.model_kwargs` passthrough for jina-v5's
  required `default_task`; Ollama dimension = trust a set `embed.dimension`
  over a disagreeing `/api/show embedding_length` (embeddinggemma-2 reports its
  pre-projection 512, serves 768d) with the encode-time contract check still
  hard-failing; `embed.context_tokens` explicit context override for honest
  truncation diagnostics (embeddinggemma-2's `/api/show` reports the 256K
  loader context, text window is 8K). Run the per-model exactly-once prefix
  sanity checks, then `cluster-rag sweep --file experiments/omct_probe_sweep.py`
  → `omct-probe2-*` rows (6 warm, 8 fresh) and the log note.
  *(done 2026-10-20, code committed `b5b1e96`, runs executed + verified
  2026-10-20: 14/14 cells → `omct-probe2-*`; the 6 re-run cells digit-for-digit
  identical vs the 3.6-era table (embeddinggemma digest drift `3a2d26…`→`82f0…`
  noted — cache served identical vectors); 204/204 artifact checks; log notes
  in `docs/experiments.md`; peft added for jina-v5's own loading mechanism)*
- [x] **B. Typed embedding kinds + full UPR kind sweep.** `EmbedCell` gains a
  validated `kinds` field (default stays `("document",)`; the resolved config
  carries it so the frozen config.yaml tells the whole run). New UPR sweep
  embedding kinds `document,query` for all 14 pool models
  (`upr-embed-<detail>`; the two existing embed runs refresh their meta.json
  kinds lists, document side warm). Verify + log note.
  *(done 2026-10-20, code committed `167f44e`, sweep executed + verified
  2026-10-20: 14 runs × `document,query`, matrices (3000, d) float32
  unit-norm, dims as listed in the log; e5-base-v2/arctic2 refresh in place
  (document warm), no-prefix models' shared-key cache reuse verified correct;
  204/204 artifact checks; log note in `docs/experiments.md`)*
- [x] **C. Val-driven cut of the pool to 3–4 semifinalists.** kNN, prototype
  and per-label max-similarity evidence on UPR val across the pool BEFORE any
  fusion/preprocess weight grids ("embed all, ablate few" — no 14³ factorial).
  *(done + verified 2026-10-20: 98/98 sweep cells `upr-eval-*` (knn k∈{1,3,5,
  10,20} × prototype × max-sim × 14 models) on val, decision top_k k=2; all
  three candidate families measured — reviewer's family note satisfied: the
  cut ranks by best micro-F1 across a model's 7 cells, tie-broken by best tail
  macro-F1. **Semifinalists {embeddinggemma, jina-v5, voyage-4-nano, arctic2}**;
  slot 4 is a measured near-tie (arctic2 0.5372 wins the tail tiebreak vs
  e5-large-v2 0.5372/0.5625; embeddinggemma2 0.5365 best-balanced tail) with a
  one-tuple swap protocol in the fusion sweep. Findings: k rises to the grid
  edge everywhere; prototype tail macro beats best-knn tail macro 14/14 while
  micro-F1 ties it; **max-sim below both families on all buckets (the
  "complements the mean for tail labels" prior did not hold on UPR val)**;
  kNN shortlist R@50 0.89–0.96 vs proto ~0.73 flat; unseen coverage 0 as
  designed. Evidence + decision table in `docs/experiments.md` Phase 4 pool-cut
  section; all rows in the auto table `## UPR — pool_cut`; OMCT corroboration
  noted as a separate line)*
- [x] **D. Eval ↔ embedding-artifact wiring.** Proposal first (schema change:
  explain and wait for the reviewer per AGENTS.md §3), then implement — how an
  `evaluate` config points at a prior `embed` run's artifact.
  *(settled + done 2026-10-20: reviewer picked the run-level key among
  {run-level key, predictor params-bag, raw path} — `EvalCell.embedding_run`
  lands as `eval.embedding_run_id` (validated `upr-embed-` reference at config
  time; existence checked at run time). Sub-decisions with the reviewer:
  embedding-free predictors warn-and-ignore a set run id (never silently);
  the UPR evaluate table gained an auto-appended `embedding` column (detail of
  the run id; full lineage in metrics.json). `evaluate` resolves and
  hard-validates `results/<run_id>/{embeddings.npz, meta.json}` (command=embed,
  source-sha vs current data, required kinds, float32 unit-norm, id alignment);
  only train document vectors reach the predictor's construction context, only
  the eval split's query vectors reach score(). Lineage (model + digest +
  prefixes + preprocess) recorded in the eval run's metrics.json
  `embedding_run` block. Module `embed_artifact.py`; 39 new unit tests incl. a
  fake-embedder CLI e2e; real-artifact smoke via a throwaway /tmp copy of
  `upr-embed-e5-base-v2` (nothing in results/ or the log))*
- [x] **E. Repo config cleanup (housekeeping).** The 9 authored experiment
  YAMLs in `configs/experiments/` are deleted: every experiment since Phase
  2.5 is defined in Python sweep files, nothing (code/tests/sweeps) reads the
  stale copies, and every past run's settings stay reproducible from its
  frozen `results/<run_id>/config.yaml`. `configs/base.yaml` remains (the
  `define_sweep` base + the stats/split standalone entry). *(done 2026-10-20:
  9 files deleted — 6 `probe_*`, 2 `embed_*`, `eval_baseline_most_frequent`;
  docstring/help references in `config.py`/`cli.py` and the stale migration
  comment reworded; AGENTS.md layout + "Running things" examples moved to
  sweeps; 191 unit + 12 integration tests green, ruff clean)*

---

## The hit map (CS-18) — the pre-registration, the amendment and the verdict

*Added on 2026-10-29, when the round was executed: the live plan's §7.1
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-18.*

### 7.1 The hit map (CS-18)

- [x] **CS-18 — the hit map: which words carry the signal.** A colleague asked for
  it: a map of the words (or word pairs) that actually carry the ranking signal, so
  the method's story can be read off a picture instead of a table of AUCs.

  **The mechanism — three versions, and the plan is to measure the first two and say
  plainly that the third is not where the value is.**

  - **Version A (exact, ~free) — the lexical feature's own per-word decomposition.**
    `lex_label_coverage` is already the *average over the label's words* of how much
    of each word appears in the text, so the per-word contribution of one of a
    label's words to a cell's feature is **exact**, not an approximation. It maps
    the **label side**: which of a label's words the pack actually keys on.
  - **Version B (the interesting one, cheap) — the decision model's sentence-level
    ablation.** The decision model reads the paragraph as its state and asks about
    all 68 labels in one batched call, so the cost is **per text, not per cell**:
    removing one sentence and re-asking costs the same ~2 calls per text. It maps
    the **text side**, and it is the only one of the three that *perturbs* the
    input — so it is the only one that says anything causal.
  - **Version C (expensive, and not where the story is) — the embedding model's own
    attribution.** Everything measured so far says the embedding geometry contributes
    almost nothing to the ranking, so it is **out of scope**; if it is ever measured
    it is a separate change set with its own cost.

  **The three questions the map must answer.**

  1. Does it **confirm** the statistical finding that the **lexical block** is the
     ranking signal?
  2. Does it **explain** why the *specific* held-out labels are found and the
     *generic* ones are not?
  3. **Does the decision model key on the right words or on boilerplate?** — the one
     worth knowing before someone else finds it.

  *Pre-registered falsification:*

  - **Version B (question 3) is falsified** if the sentence ablation's effect is
    **not** concentrated on the sentences that carry the topical content — e.g. if
    removing a formulaic opener ("Continue to ensure that …") moves the
    probabilities as much as removing the topical sentence does. The measure is the
    per-sentence drop in the decision probability for the cell's gold labels, read
    against the **label-blind counterpart** (the same ablation with the wrong
    pairing). If the counterpart's pattern is the same, the attribution is not
    label-driven and the map says nothing about the labels.
  - **Version A (question 2) is falsified** if the held-out labels' per-word
    contributions are carried by their **generic** words ("rights", "human",
    "recommendation") rather than by their own distinctive ones — i.e. if the map
    says the name transfers because of the word every label has, not because of the
    label's own topic word. The measure is the per-word contribution split between
    the held-out labels and the generic catch-alls ("General", "Other").
  - **A null is a legitimate outcome** and is reported as one: the map may show the
    signal is *distributed* and that no single word carries it.
  - The per-word numbers must **not** be read out of the **boosted-tree ensemble**:
    a tree ensemble has no per-word attribution, and the pack's importances are per
    *feature*, not per word. The map is of the **feature's own definition**, and it
    must be labelled as such.

  *Cost:*

  - **Version A: ~free.** A pure computation over inputs the pack already has. It
    needs one small, pure addition to the lexical feature's own module (expose the
    per-word decomposition as a pure function) and **no model call**.
  - **Version B: 2 decision calls per (text, sentence-removed) variant**, so the
    round's total is **2 × Σ_text (the text's sentence count)** — the same ~2 calls
    per text the CS-13 feature cost, times the number of sentences. **Measure the
    sentence count before scheduling**: UPR texts average ~152 characters, so most
    have one or two sentences, and a single-sentence text has **no** ablation — it
    is reported as "not ablatable", never silently dropped. The **label-blind
    counterpart** runs beside it, so the round's cost is **4 × Σ_text s_text**. The
    **fit side is not needed**: the map is a measurement, not a fitted model.
  - **Version C:** out of scope.

  *Leakage:*

  - The map is a **measurement, not a feature**: nothing is fitted, nothing enters
    a model, and the pack and the combiner are untouched.
  - The decision model is frozen (id **and** digest pinned) and carries no UPR
    label; it is the same `tev1` the CS-13 feature used, so the map and the
    feature are read against the same model.
  - The **label-blind counterpart** runs beside every arm — the track's own rule,
    and the only thing that makes the attribution label-driven rather than the
    model's own prior.
  - The **text set** must be **pre-registered** (or the whole val split used): a
    set chosen after looking at the results is cherry-picking.
  - The **fit side is not touched at all** (nothing is fitted). **`test` is not
    touched**: val only, as everywhere in this track.
  - The two negative subsets are never pooled, and the map is over the judged
    pairs.

  *Caveats to carry into the write-up:*

  - **Attribution is not causation unless you perturb** — so for the decision model
    the answer comes from **Version B** (the ablation), not from Version A (a
    decomposition of a feature's own definition).
  - **A pack hit map is a map of the *word-overlap* feature, not of the embedding
    model**, and must be labelled as such wherever it is shown.
  - **Do not read per-word attributions out of the boosted-tree ensemble.**
  - **The sampled text set must be pre-registered**, or it is cherry-picking.

  *Runs:* the sweep file, the run ids and the log section are fixed in the round's
  own pre-registration, before the first call.

  **The amendment of 2026-10-29 — written before the first call, after the
  sentence count was measured.** The pre-registration asks for the sentence count
  to be measured *before* scheduling. Measured on the UPR **val** split (660
  texts, the track's own `split_sentences`):

  | sentences | 1 | 2 | 3 |
  |---|---|---|---|
  | texts | 653 | 6 | 1 |

  mean 1.012, median 1, Σ_text s_text = 668. **653 of the 660 texts (98.94%) have
  no sentence-level ablation** — they are reported as "not ablatable", never
  dropped. Of the seven texts with two or three "sentences", **four are false
  splits**: the splitter cuts at "decree No. 170" ("No." is not in its
  abbreviation list), so only **three val texts (0.45%) are genuinely
  multi-sentence** (eight gold cells between them). A sentence-level map on this
  corpus therefore rests on three texts: it cannot answer question 3.

  **What the amendment does.** It adds a **second ablation unit** beside the
  plan's, and fixes the runs. The plan's sentence-level arm still runs, exactly
  as pre-registered, and its near-null is reported as a near-null.

  - **The clause-level arm.** The track's own atom chunker
    (`AtomChunker(split_clauses=True)` — the CS-2/CS-3 atoms, a deterministic
    rule-based splitter already in the repo, no new dependency) cuts a UPR text
    at its clause separators. Measured on val: the clause counts are 1 (312
    texts), 2 (190), 3 (84), 4 (47), 5 (14), 6 (9), 7 (2), 8 (1), 9 (1) — mean
    1.95, median 2, Σ = 1,287. **348 texts (52.7%) are ablatable and hold 799
    gold cells**; the other 312 are "not ablatable" and reported as such. 205
    texts (31.1%) open with a formulaic opener clause ("Continue to ensure that
    …") — the pre-registration's own falsification example.

  **The fixed runs** (the "*Runs:*" clause, discharged):

  | what | run id | artifact | new calls |
  |---|---|---|---|
  | the sentence-level arm (the plan's unit) + Version A | `upr-hitmap-sentence` | `results/upr-hitmap-sentence/hitmap.json` | 60 |
  | the clause-level arm (the amendment) + Version A | `upr-hitmap-clause` | `results/upr-hitmap-clause/hitmap.json` | 3,900 |

  The sweep file is `experiments_text_processing/upr_hitmap_sweep.py` (sweep
  label `upr_hitmap`, so the comparison table's section is `## UPR — upr_hitmap`);
  the log section is **`## CS-18 — the hit map: which words carry the signal
  (2026-10-29)`** in `docs/experiments_text_processing.md`.

  **The text set is the whole val split** (660 texts), pre-registered — not a
  sample chosen after looking at the results.

  **The cost, corrected.** A text with *k* units yields *k* variants (one per
  removable unit: removing a unit leaves the other *k − 1*), and a text with
  **one** unit yields **none** — removing it would leave nothing to ask about,
  so it is not ablatable and is reported as such. The round's new calls are
  therefore 4 × Σ over the **ablatable** texts of their unit count (2 calls per
  variant: the real pairing and the label-blind counterpart): **60** for the
  sentence arm (15 variants over the 7 ablatable texts) and **3,900** for the
  clause arm (975 variants over the 348 ablatable texts). The plan's own
  formula (2 × Σ_text s_text) is an upper bound: it counts the not-ablatable
  texts' meaningless variants too. The full-text baselines are **CS-9b's cache
  hits** (the same tev1 id and digest, the same prompt, the same cell layout —
  checked: 1,320 of 1,320 planned calls are already cached), so the round pays
  for the ablations only.

  **The falsification readouts, made concrete** (the pre-registration states the
  conditions; this fixes the numbers):

  - *Version B, question 3.* For every gold cell the **label-linked unit** is
    the unit whose content tokens overlap the gold label's name the most (ties:
    the earliest). The measure: the mean drop in the decision probability of the
    gold cells when the **label-linked** unit is removed, against the mean drop
    when a **non-linked** unit is removed, with the **label-blind counterpart's**
    two drops beside them. The second measure: the mean drop of the **formulaic**
    units (a unit that starts with one of the config's `preprocess.strip_openers`)
    against the **topical** units'. The third: the mean drop over the **gold**
    cells against the mean drop over the **background** cells. **Falsified** if
    the label-linked drop is not larger than the non-linked drop, or if the
    formulaic drop is not smaller than the topical drop, or if the label-blind
    counterpart's pattern is the same as the real arm's (the attribution would
    then not be label-driven).
  - *Version A, question 2.* A label's word is **distinctive** when no other
    label in the 68-label space contains it, and **generic** otherwise. The
    measure: the held-out labels' per-word contributions split between the
    generic and the distinctive words, with the generic catch-alls ("General",
    "Other", "ESC rights - general" — all three are themselves held out) beside
    them. **Falsified** if the generic words carry the held-out labels' coverage.
  - *Both.* The per-word contributions must sum to `lex_label_coverage` exactly
    (the artifact records the largest deviation). A null is a legitimate outcome
    and is reported as one.

  *Exit:* the map is in the track log with its numbers beside it, and it answers —
  or explicitly fails to answer — each of the three questions, with the
  falsification conditions read out one by one.

  *(pre-registered 2026-10-29; **executed 2026-10-29** — the two runs are
  `upr-hitmap-sentence` and `upr-hitmap-clause`, the log section is
  `## CS-18 — the hit map: which words carry the signal (2026-10-29)` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (UPR val, 660 texts, 68 labels, 1,312
  gold cells; the text set = the whole val split, pre-registered).**

  - **Version A (the lexical map, exact and free).** The per-word contributions
    sum to `lex_label_coverage` with the largest deviation **0.0** over all
    **44,880 cells**. The signal is **distributed**: the top word ("rights",
    18 labels) carries **0.1939** of the grid's total contribution, the top five
    **0.3962**, the top twenty **0.6359** (over 106 words). The generic words
    carry **0.5096** of the grid's total.
  - **Question 2 (the held-out labels) — the falsification does *not* fire.** The
    eight held-out labels' coverage is **0.3105 generic / 0.6895 distinctive**:
    the name transfers through the label's **own** topic word ("special
    procedures", "refugees", "religion"), not through the word every label has.
    The generic catch-alls are **outside the feature's reach**: "Other" has no
    content word at all ("other" is a stopword of the coverage), so its feature
    is 0.0 in every cell **by definition**; "General" is carried by its single
    generic word (the mean gold-cell coverage **0.1316**).
  - **Question 3 (the decision model) — a yes and a no.** The **label-linked**
    clause (the one carrying the gold label's own name) is the strongest
    carrier: its drop **0.2687** against **0.0918** for any other clause,
    **+0.1769 [+0.1501, +0.2025]** over 287 texts; the **label-blind
    counterpart** is flat (**−0.0179 [−0.0354, +0.0005]**), so the attribution is
    label-driven; the fall is concentrated on the **gold** cells (**0.1511**
    against the background's **−0.0160**). **But** a **formulaic** clause's drop
    is *larger* than a topical clause's (**0.1782** vs **0.1293**,
    **+0.0489 [+0.0139, +0.0849]**) — the pre-registered falsification **fires on
    that clause**: the model's confidence leans on the recommendation's opening
    clause at least as much as on a topical one. The instrument's own limit: the
    clause splitter cannot separate the frame from the content in this corpus
    ("Continue to ensure that X" is one clause), so the map says the model leans
    on the **opening**, not how much of that is boilerplate.
  - **Question 1 (the lexical block as the ranking signal) — not confirmed, and
    the map says why not.** On UPR the lexical block was never the ranking
    signal (CS-8e: the embedding block alone 0.4460 against the lexical block
    alone 0.3837; CS-13's random-vector floor says the same from the other
    side). The ranking signal on UPR is the decision model's own label reading —
    and Version B is what says what *it* keys on.
  - **The cost.** The clause arm planned **3,892 calls** (1,946 per arm, all of
    them new) and took **2,186 s**; the sentence arm's 60 calls took 34.7 s; the
    intact-text baselines were **CS-9b's cache hits** (2,640 hits, 0 misses).
    The round reproduces: a cold re-run of the whole sweep (the round's own
    ablation calls deleted from the cache) returned **every number identical**
    to the sixth decimal. The **sentence** arm rests on **three genuinely
    multi-sentence texts** (653 of the 660 val texts are a single sentence) —
    it is reported as the near-null it is, and the clause arm is the one that
    answers the round's questions.

---

## The hit map's two cheap modes (CS-19) — the pre-registration and the verdict

*Added on 2026-10-29, when the round was executed: the live plan's §7.1
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-19.*

### 7.1 The hit map's cheap modes (CS-19)

- [x] **CS-19 — the hit map's two cheap modes: the text-side word map and the
  surrogate attribution.** CS-18 (the archive's §7.1.4) measured the map's
  first two versions: the **topic-side** decomposition of the word-overlap
  feature (mode A) and the decision model's clause-level ablation (mode D, the
  causal one). A colleague then asked for the **text side** of the word map —
  "split the text into words, check which one contributes more for each label" —
  which is the pack's *second* lexical feature (``lex_text_coverage``)
  decomposed the same exact way. And the question the map was built for is
  about the **model**, not about a feature, so the map needs a way to read the
  model's own words cheaply.

  **The mechanism — two modes, both of which cost no new model call.**

  - **Mode B — the text-side word map.** The pack has *two* lexical features
    and they are not symmetric: ``lex_label_coverage`` is "the topic's words,
    how much of each is in the text" (CS-18's mode A) and
    ``lex_text_coverage`` is "the text's words, how much of each is in the
    topic's name". Both are averages, so both decompose exactly. Mode B is the
    second: per document, which word of *this* recommendation did the matching,
    and over the grid, the corpus's own long tail of matching words.
  - **Mode C — the surrogate attribution.** A **ridge surrogate** of the
    decision model's own probability: the text's content words in, the model's
    confidence out, one coefficient vector per topic. Its coefficients are a
    per-topic word attribution ("these words push the model's confidence on
    this topic up"). The surrogate is fitted on the **train** split (the CS-9b
    run's fit-side probabilities, every call a cache hit) and read on val.

  **The falsification (pre-registered, read out one by one).**

  - *Mode C's validity.* A surrogate's attribution is worth nothing if the
    surrogate does not track the model: the **out-of-sample quality** (the
    per-topic correlation and R² against the model's own val probabilities) is
    reported first, and a surrogate that does not track the model is reported
    as **not usable** (a null, not a map). The **topic-faithfulness** check is
    the yardstick: the real attribution's top words must be the topic's *own
    name* words, and the **label-blind counterpart** (the same fit on the
    shifted pairing) must follow the *shifted* topic instead — if the blind
    counterpart's top words are the same as the real one's, the attribution is
    about the model's prior and says nothing about the topics.
  - *Mode C against the causal map.* The cheap attribution and the expensive
    ablation must agree about **where** the signal is: the share of a topic's
    top words that fall **inside** the clause the CS-18 ablation identified as
    the carrier, against the share that fall in the text but in another clause.
    A low inside-share means the two methods do not agree, and that is
    reported as such.
  - *Mode B's exactness.* The per-word contributions must sum to
    ``lex_text_coverage`` exactly (the artifact records the largest deviation,
    as CS-18's mode A did for its own feature).

  **The cost.** **Zero new model calls.** The val grid and the train-side fit
  grid are the CS-9b run's cache hits (1,320 + 3,352 planned calls, all cached —
  checked before the run), and mode B is pure arithmetic. The ablation is
  recomputed only for mode C's cross-check (also cache hits).

  **Leakage.** The surrogate is fitted on the **train** split alone: no val text
  and no val label enters the fit, and the vocabulary is the train corpus's
  words (a word that appears only in val never becomes a feature). Val is used
  only to *measure* the surrogate and for the cross-check. Nothing is fitted on
  the evaluated split.

  **The runs** (the sweep file, the run ids and the log section are fixed here,
  before the first call):

  | what | run id | artifact | the log section |
  |---|---|---|---|
  | mode B — the text-side word map | `upr-hitmap-text` | `results/upr-hitmap-text/hitmap.json` | `## CS-19 — the hit map's cheap modes` in `docs/experiments_text_processing.md` |
  | mode C — the surrogate attribution | `upr-hitmap-surrogate` | `results/upr-hitmap-surrogate/hitmap.json` | the same section |

  The sweep file is `experiments_text_processing/upr_hitmap_words_sweep.py`
  (the sweep label `upr_hitmap_words`, so the comparison table's section is
  `## UPR — upr_hitmap_words`). The text set is the **whole val split**
  (pre-registered, as in CS-18). `test` is not touched.

  **What stays open.** The **word-level causal ablation** (mode E: delete one
  word and re-ask) is *not* in this change set: it is the expensive instrument
  (~1k–26k calls), and it needs its own pre-registration when it runs.

  *(pre-registered 2026-10-29; **executed 2026-10-29** — the two runs are
  `upr-hitmap-text` and `upr-hitmap-surrogate`, the log section is
  `## CS-19 — the hit map's two cheap modes` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (UPR val, 660 texts, 68 labels, 1,312
  gold cells; the text set = the whole val split, pre-registered; **zero new
  model calls** — every call is a CS-9b cache hit).**

  - **Mode B — the text-side word map.** The decomposition is **exact** (the
    largest deviation **0.0** over all 44,880 cells). The matching words are
    the **UPR register**, not the topics: "rights" **0.2260** (it matches all
    68 topics), "human" 0.0597, "right" 0.0347, "national" 0.0344,
    "international" 0.0255. The signal is **distributed**: the top word 0.2260,
    the top five 0.3804, the top twenty 0.5173, over **1,985** words.
  - **Mode C — the surrogate attribution.** The surrogate **tracks the model**
    out of sample (the mean correlation **0.6933**, the R² 0.4466, the per-topic
    range 0.2920–0.8783), so its coefficients are readable. The
    **topic-faithfulness** yardstick holds: the real attribution's top words are
    the asked topic's own name words **0.0786** of the time, while the
    **label-blind counterpart**'s are the **shifted** topic's **0.0806** — and
    only **0.0090** for the topic it was asked about (its "General" reads *hiv*,
    its "Other" reads *poverty*: the pairing is followed). The two
    **worst-fitted** topics are the catch-alls ("Other" 0.2920, "General"
    0.3681): the model's confidence on them is the least word-driven of any
    topic.
  - **The cross-check against the causal map.** When the surrogate's top words
    appear in a text, they land inside the clause the CS-18 ablation
    identified as the carrier **0.8338** of the time, against a **0.4839**
    chance level: the cheap attribution and the expensive ablation **agree**.
  - **The word-level causal ablation (mode E) is not in this change set** — it
    is the expensive instrument (~1k–26k calls) and it needs its own
    pre-registration.

## The hit map's terminal viewer (CS-20) — the pre-registration and the verdict

*Added on 2026-10-29, when the round was executed: the live plan's §7.1 block
moved here **verbatim** (the archive's convention: an executed change set's
pre-registration moves here and the live plan keeps one tombstone line). The
verdict and the numbers are the track log's §CS-20.*

### 7.1 The hit map's terminal viewer (CS-20)

- [x] **CS-20 — the hit map's *terminal viewer* (the interactive map).** The map
  exists as four modes (CS-18's topic-side map + causal ablation, CS-19's
  text-side map + surrogate attribution) and all four live in artifacts. What is
  missing is the **per-document** view: one command, one text on screen, and
  (a) the text's words with **what each contributes**, (b) **who the words say
  it is** — the top-k the cheap modes predict, (c) **what the model says** — its
  own top-k, live, (d) **what is true** — the gold labels, so (b), (c) and (d)
  sit side by side, and (e) behind a flag, **what happens if a specific word is
  removed** — the causal one (the text minus the word → the model is asked
  again → the new probability, the new top-k, the delta).

  **The mechanism.** A repo tool (`tools/hitmap_view.py`, next to the
  verification scripts) with **no new dependency**: the colours are raw ANSI
  escape codes with a plain-text fallback. Three modes and their costs: the
  **offline** mode (the default) reads only the artifacts — the text-side map,
  the surrogate's per-topic coefficients, the gold matrix — and costs **0 model
  calls**; the **live** mode re-scores the text through the cached calls (the
  machinery in `commands/upr.py`) to show the model's own top-k, which is in no
  artifact — cache hits, free for the val texts; the **what-if** mode spends
  **2 real calls per word** (~1 s) and must say that the decision model is
  **not deterministic** (the CS-7a precedent).

  **The discipline.** The viewer is a **tool, not a number source**: anything it
  shows that becomes a claim lands in an artifact first. The seeds are explicit
  and the text set is the val split. The **attribution/causation distinction is
  visible in the output** (the lexical and surrogate columns are attributions,
  the what-if column is the causal one) — the lesson of CS-18 and CS-19 is
  exactly that difference, and a viewer that blurs it would teach the wrong
  thing. The track's pure-core boundary is untouched (the tool lives at the
  edge).

  **What it is not.** No new measurement, so no run id, no artifact and no
  table row; the track log gets a short section (what the tool shows and how to
  use it, with an example output quoted from the tool's own run). The run-level
  details are fixed before the first call, as they were for CS-18 and CS-19.

  **The pre-registration (written before the first call, 2026-10-29).**

  - **The runs: none.** The viewer is a **tool**, not a measurement: it has no
    run id, writes no artifact and appends no table row. It *reads* the CS-19
    runs' artifacts — `results/upr-hitmap-text/hitmap.json` (the text-side map)
    and `results/upr-hitmap-surrogate/hitmap.json` (the surrogate's per-topic
    coefficients) — plus the data, the split and the CS-9b run's decision
    cache. The log section is **`## CS-20 — the hit map's terminal viewer`** in
    `docs/experiments_text_processing.md`.
  - **The texts: the val split** (the track's split; `test` is not touched). A
    text is picked by its **record id** (`--text <id>`) or by a **seeded draw**
    (`--sample --seed N` — the seed is explicit and printed). The text is never
    chosen after looking at the results.
  - **The three modes and their costs**, fixed here:

    | mode | what it reads | the cost |
    |---|---|---|
    | **offline** (the default) | the two artifacts + the data + the split | **0 model calls** |
    | **live** (`--live`) | the same, plus the decision model's own grid for that text | **2 calls per text** (68 labels, 64 questions per call), and for the val texts both are **cache hits** of the CS-9b run — the tool prints the hits and the misses, so "free" is a measurement, not a claim |
    | **what-if** (`--what-if word[,word]`) | the same, plus one re-scored grid per word | **2 real calls per word** (~1 s); the model is **not deterministic** (the CS-7a precedent) and the output says so |

  - **The cache discipline (what makes the live mode free).** The live mode
    asks the model **the same call the CS-9b run asked**: the same state, the
    same question specs and the same question names (the grid's own cell
    index), so the answer comes from the decision cache. A cache miss is
    reported (the hits/misses line), never hidden.
  - **The numbers the tool prints.** The offline numbers are the artifacts'
    own (the map's exact per-word decomposition and the surrogate's stored
    coefficients). The live and what-if numbers are the **tool's own readings**
    of the model (its probability and the delta), printed with the model's id
    and digest — they are **tool output, not claims**: anything that becomes a
    claim lands in an artifact first (the track's own rule).
  - **The exit criteria.**
    1. The tool runs on the **tiny fixture dataset** and on the **real val
       split**.
    2. The **offline mode's numbers are pinned by a test against the
       artifacts**: the viewer's recomputed top label and its per-cell values
       must equal the artifact's (`per_text_top_label`, `per_gold_cell`), and a
       **stale artifact fails loudly** — the tool reports the mismatch and the
       test exercises that path.
    3. The **live and what-if modes are exercised against a fake decider** (no
       server in the tests).
    4. The **attribution/causation distinction is visible in the output** (every
       column says which of the two it is) and the what-if mode prints the
       **not-deterministic** note.
    5. `pytest` and `ruff` are clean; the two verify tools and the two final
       sweeps still reproduce.
  - **The falsification.** A tool has no result to falsify; what it must not
    do is **blur the attribution/causation distinction** (the lesson of CS-18
    and CS-19) or **print a number it did not measure**. Both are checked by
    the output itself (the column labels and the provenance line).

  *(pre-registered 2026-10-29; **executed 2026-10-29** — the tool is
  `tools/hitmap_view.py`, its tests are `tests/test_hitmap_view.py`, the log
  section is `## CS-20 — the hit map's terminal viewer` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen.** The change set has **no run**: the
  viewer is a tool, so there is no artifact and no table row. What the tool
  shows on the UPR **val** split (660 texts, 68 labels, 1,312 gold cells) for
  the text `0000000000000047d5d150d7` (row 4), quoted from the tool's own run:

  - the **offline** numbers are the CS-19 artifacts' own: the text-side map's
    exact decomposition (the largest deviation **0.0** over all 44,880 cells)
    and the surrogate's stored coefficients. For this text the text-side
    ranking's top-1 is "National Human Rights Institution" (**0.1733**) — the
    UPR register, not the topic — while the topic-side map puts "Migrants" at
    **0.8750** and the text's three gold labels sit at the text-side ranks
    **9, 28 and 11**.
  - the **live** numbers are the **tool's own reading** of the model (never a
    claim): the model's top-3 for this text is exactly the three gold labels
    (**0.8926 / 0.8681 / 0.8076**), read from **2 cache hits** of the CS-9b
    run's grid (the tool prints the cache's counters: 2 hits, 0 misses).
  - the **what-if** numbers are the tool's own readings too, and they are
    **causal**: removing "workers" drops "Labour rights" by **0.5177** — the
    word that carries the label's own name — while removing "protection" moves
    the three probabilities by **-0.0110 / +0.0021 / +0.0028**, inside the
    model's own noise of ~0.001-0.003. The cost: **2 real calls per word** (the
    cache's misses; a variant that was already asked is a cache hit, and the
    tool prints the counters).
  - the **exit criteria**: the tool runs on the tiny fixture dataset and on the
    real val split; the offline numbers are pinned by a test against the
    artifacts and a **stale artifact fails loudly** (the tool exits 1); the live
    and what-if modes are exercised against a **fake decider** (no server in
    the tests); the attribution/causation distinction is in the output itself
    (the lexical and surrogate columns say ATTRIBUTION, the what-if block says
    CAUSAL) and the what-if prints the not-deterministic note.
  - the **gate**: `pytest` **678 passed**, `ruff` clean, the two verify tools
    OK (110 + 45 runs) and both final sweeps reproduce their artifacts field
    for field (only the wall-clock fields move).

---

## The corpus map (CS-23) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.2
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-23.*

### 7.1 The corpus map (CS-23)

- [x] **CS-23 — the corpus map: the train split's own word↔label
  association.** For every label and every word of the **training** corpus: how
  strongly does this word go with this label? The plain statistics of the data
  — the map's neighbour, and the one thing the map never measured.

  **The measures** (every one stored with the counts behind it, so any number
  can be recomputed and audited):

  - `n_with` — the train texts that contain the word **and** carry the label;
  - `df` — the train texts that contain the word at all;
  - `lift` = `P(label | word present) / P(label)` — how much likelier a text
    with this word is to carry the label; 1.0 = the word says nothing, > 1 =
    it pulls the label, < 1 = it pushes it away;
  - `log-odds` = `log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5))` over the 2×2
    table (`a` word+label, `b` word+no label, `c` no word+label, `d` neither),
    the 0.5-smoothed signed version: > 0 with the label, < 0 against it, 0
    nothing. It is preferred to lift because lift explodes on rare words (a
    word in 2 of 2 texts gives lift 57 and means nothing).

  **No thresholds.** The reviewer's instruction: **all** words for **all**
  labels. Every (label, word) pair that occurs is stored, with the four counts;
  the noise is handled at **display** time (the counts sit next to the number,
  and the viewer offers a `--min-df` filter, default off), never by dropping
  data.

  **The mechanism.**

  - a new **pure function** in the track's pure core: the train records + the
    label space → the per-label word tables and the per-word reverse tables
    (the same data read the other way);
  - a new **mode on the existing `hitmap-upr` command**: the optional bool
    `upr_analysis.hitmap.include_corpus_map` — a **config-schema change**
    (AGENTS §3, explain-first; the reviewer's decision of 2026-11-01: *add it
    to the existing command*, not a new one);
  - the **viewer**: two new blocks — *"the corpus's own words for this topic"*
    (the topic's word list by log-odds, with the counts, the words that are
    **in this text** marked) and *"this text's words and the labels they lean
    to"*.

  **The cost.** **0 model calls**, ~2 s of CPU; the artifact grows by a few MB.

  **Leakage.** The **train** split only: no val text and no val label enters
  the counts. The eight **held-out** labels have no train examples at all, so
  their tables are empty **by construction** — reported as empty, never
  fabricated. `test` is not touched.

  **What it is expected to say** (measured read-only on 2026-11-01, so the
  numbers are known before the run): "Detention" ← detention (lift 16.90),
  conditions (9.04), torture (8.03), treatment (5.63), disappearance (5.38);
  "Migrants" ← migrant (27.86), workers (20.75), families (20.47), members
  (16.18); "Women's rights" ← women (5.92), men (5.76), equality (5.52), girls
  (5.34), violence (4.21).

  **The exit criteria.** The counts/lift/log-odds pinned by a hand-computed
  unit test; a test that the train side never touches a val text; the viewer's
  numbers pinned against the artifact (the stale-artifact guard); the fixture
  and the real val split; the gate green.

  **The runs** (fixed before the first call): the run id `upr-hitmap-corpus`,
  the artifact `results/upr-hitmap-corpus/hitmap.json` (the block
  `corpus_map`), the sweep cell in a new sweep file
  `experiments_text_processing/upr_hitmap_corpus_sweep.py` (the sweep label
  `upr_hitmap_corpus`, so the table's section is `## UPR — upr_hitmap_corpus`),
  and the log section `## CS-23 — the corpus map` in
  `docs/experiments_text_processing.md`.

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the run is
  `upr-hitmap-corpus` (the artifact `results/upr-hitmap-corpus/hitmap.json`, the
  block `corpus_map`), the log section is `## CS-23 — the corpus map` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (UPR **train**: 1,676 texts, 68 labels;
  the corpus is the train split, so no val text and no val label enters the
  counts and `test` is not touched).**

  - **The cost.** **0 new model calls** (the intact texts' grid is the CS-9b
    run's cache hits: 2,640 hits, 0 misses, 0 stored) and **1.3 s** of wall
    time; the artifact is 6.4 MB.
  - **The map's shape.** **18,707** (label, word) pairs over **3,049** distinct
    words; **60 of 68** labels have a train example and **8** have an empty
    table **by construction** (the held-out ones). The per-label table's size:
    min 11, median 268, max 1,125. **No thresholds**: every pair that occurs
    is stored — 2,540 of them (13.6%) rest on a hapax word (df = 1) and 1,749
    (9.4%) have a negative log-odds.
  - **The pre-registered expectation is reproduced digit for digit.**
    "Detention" ← detention (lift **16.90**), conditions (9.04), torture
    (8.03), treatment (5.63), disappearance (5.38); "Migrants" ← migrant
    (**27.86**), workers (20.75), families (20.47), members (16.18);
    "Women's rights" ← women (**5.92**), men (5.76), equality (5.52), girls
    (5.34), violence (4.21).
  - **The two measures rank differently, and that is the point.** The lift's
    head word and the log-odds' head word **differ for 55 of the 60 labels**:
    the lift's head is a hapax almost every time (every word in a label's
    texts has the same lift), while the log-odds' head is a topical word (only
    4 of the 60 labels have a hapax there).
  - **The corpus's own topical vocabulary**: *hiv* (log-odds **12.034**, "HIV -
    Aids"), *trafficking* (10.636), *defenders* (10.263), *impunity* (9.540),
    *corruption* (9.434). **And the register is the other end of the same
    table**: *rights* is in **all 60** labels' tables (df 485), *human* in 58 —
    the UPR register, now measured on the train side with the corpus's own
    statistics.
  - **The label's own name word** is in its own table for **59 of the 60**
    labels (at rank 1 for **33**, in the top 10 for **45**); the exception is
    "SDGs", whose name is the abbreviation (the train texts say "sustainable
    development goals").
  - **The exit criteria.** The counts / lift / log-odds are pinned by a
    hand-computed unit test (and counts that do not form a 2×2 table are
    refused); the train side never touches a val text (the fixture run's
    artifact holds neither of the fixture's two val-only words); the viewer's
    numbers are pinned against the artifact and a **stale artifact fails
    loudly** (the tool exits 1); the fixture and the real val split both run.
  - **The gate.** 690 tests, ruff clean, the two verify tools OK and both final
    sweeps reproducing field for field.

---

## The surrogate's full vector (CS-24) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.2
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-24.*

### 7.1 The surrogate's full vector (CS-24)

- [x] **CS-24 — the surrogate's full coefficient vector.** The viewer's
  surrogate column must be complete: every word of the text gets its
  coefficient instead of "—".

  **The mechanism.** The surrogate artifact stores the **full** per-topic
  coefficient vector beside the existing top-15 lists (which stay, for the
  log): 68 topics × 1,203 features = **81,804** numbers, ~1.5 MB of JSON. The
  viewer then prints the coefficient of every word of the text; a word outside
  the model's vocabulary (the train corpus's words with df ≥ 3) reads **"not in
  the model's vocabulary"** — a different and more useful statement than "not
  stored".

  **The cost.** **0 model calls** (the surrogate's re-run is all cache hits);
  the artifact grows ~1.5 MB.

  **The discipline.** The log is append-only: CS-19's section is **not**
  edited; the re-run's section is appended, and the top-15 lists (the numbers
  CS-19 reported) are unchanged.

  **The exit criteria.** The stored vector matches the fitted one (a test); the
  artifact's numbers reproduce; the viewer shows a number for every word of the
  text; the gate green.

  **The runs.** The run id `upr-hitmap-surrogate` (re-run; the artifact's shape
  changes) and the log section `## CS-24 — the surrogate's full vector`.

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the CS-19 sweep was
  re-run (`upr-hitmap-surrogate` and, alongside it, `upr-hitmap-text`: the sweep
  owns the comparison table's section, so its rows must share one shape and the
  corpus map is on by default now), the log section is
  `## CS-24 — the surrogate's full vector` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (the UPR **val** split: 660 texts, 68
  labels, 1,312 gold cells; the surrogate is fitted on the **train** split).**

  - **The cost.** **0 new model calls** (6,532 cache hits, 0 misses, 0 stored)
    and 16.6 s for the two runs.
  - **The vector.** **68** topics × **1,203** features = **81,804** coefficients
    (the plan's number, exactly), **1.37 MB** of JSON. It is **dense**: 81,803
    of the 81,804 coefficients are non-zero (the ridge shrinks, it does not
    zero); 35,034 positive, 46,769 negative, 1 exactly zero; the range is
    **+0.7529 … −0.2467**.
  - **The top-15 lists are unchanged** — every number CS-19 reported is
    reproduced: the out-of-sample correlation **0.6933** (the per-topic range
    0.2920–0.8783), the R² **0.4466**, the asked topic's own-name share
    **0.0786** against the blind counterpart's **0.0090** (and **0.0806** for
    the shifted topic), and the surrogate's top words inside the carrier clause
    **0.8338** against the chance level **0.4839**.
  - **The viewer.** The surrogate column is complete: every word of the model's
    vocabulary shows its fitted coefficient, and a word outside the vocabulary
    reads **"not in the model's vocabulary"** (never zero). On the CS-20 example
    text the 21 content words split into 16 with a number and 5 outside the
    vocabulary; the topic's own words carry the positive coefficients
    (*migrant* +0.564, *migration* +0.229, *workers* +0.126, *families* +0.056)
    and the boilerplate is flat (*take* +0.030, *steps* −0.013).
  - **The exit criteria.** The stored vector matches the fitted one (a unit
    test, number for number; a mismatched shape is refused), the artifact's
    numbers reproduce, the viewer shows a number for every word of the text,
    and the gate is green.

---

## A clearer screen, and the colour (CS-25) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.2
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-25.*

### 7.1 A clearer screen, and the colour (CS-25)

- [x] **CS-25 — a clearer screen, and the colour.** Every block says **which
  question it answers**, in plain words; the mode letters leave the viewer's
  headings (they stay in the plan and the log, where the history lives); and
  the screen gets **colour**.

  **The headings** (the names replace the letters):

  | the block | the heading | the question it answers |
  |---|---|---|
  | the feature's own split (text side) | "THE FEATURE'S OWN SPLIT — a spelling question: how much of each word is in the topic's name?" | the feature's arithmetic |
  | the feature's own split (topic side) | "THE TOPIC'S OWN WORDS — a spelling question: how much of the topic's name is in the text?" | the feature's arithmetic |
  | the corpus map (CS-23) | "THE CORPUS'S OWN WORDS FOR THIS TOPIC — a statistical question: which words go with this label in the training corpus?" | the data's statistics |
  | the surrogate (CS-24) | "THE MODEL'S WORD HABITS — a learned question: does the model react to this word for this topic?" | the model, indirectly |
  | the clauses (CS-26) | "THE CLAUSES OF THIS TEXT — the causal one, at clause level" | the model, directly |
  | the what-if | "WHAT IF THIS WORD IS REMOVED — the causal one, at word level" | the model, directly |

  **The palette** (raw ANSI, the 256-colour ramps for the shades, still **no
  dependency**): **bold** = the header or the row that matters (a top-1, the
  label-linked clause); **cyan** = structure (label names, the topic's own name
  words); a **green ramp** = pushes the score up / is true (the brightness is
  the size); a **red ramp** = pushes the score down; **yellow** = caution (the
  model's noise, "not ablatable", the not-deterministic note); **dim** = nothing
  here (zeros, "—", the notes).

  **Where it lands.** The gold labels in bold green (and in every ranking they
  appear in); the contribution bars on a green ramp; the words that **are** the
  topic's name in cyan; the surrogate and corpus columns on a green/red ramp;
  the corpus words that are **in this text** in bold with the counts in dim;
  each clause's drop on a red ramp with the label-linked clause in bold; the
  what-if deltas on a ramp, with **anything inside the model's noise band
  (±0.003) in dim and marked "noise"** — the rule that separates "the model
  used this word" from "the model twitched"; the header's dataset/split in
  cyan.

  **The rule.** Colour never carries information the text does not also carry,
  so `--no-colour` loses nothing. A `--colour` flag forces it on when piping
  (e.g. into `less -R`); the default is "on when stdout is a terminal".

  **The cost.** Presentation only; 0 calls.

  **The exit criteria.** The plain output still carries every number and every
  marker; the coloured output is checked by a test (the right codes on the
  right cells); the gate green.

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the change set is a
  **tool** change: the viewer gets new headings, blocks and colour, it writes no
  artifact and appends no table row, and it costs **0 model calls**. The log
  section is `## CS-25 — a clearer screen, and the colour` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (the tool's own output on the UPR **val**
  split; every number on it is the CS-19/CS-23 artifacts' or the tool's own
  reading of the model).**

  - **The headings.** Every block names **the question it answers** and the
    mode letters have left the headings (they stay in the plan and the log):
    "THE FEATURE'S OWN SPLIT — a spelling question: …", "THE TOPIC'S OWN WORDS
    — a spelling question: …", "THE CORPUS'S OWN WORDS FOR THIS TOPIC — a
    statistical question: …", "THE MODEL'S WORD HABITS — a learned question:
    …", "WHAT IF THIS WORD IS REMOVED — the causal one, at word level" (and
    "THE CLAUSES OF THIS TEXT — the causal one, at clause level", which is
    CS-26's block).
  - **The two instruments that shared one table** now have their own blocks
    (the feature's own decomposition; the surrogate's fitted coefficients), and
    the **topic side** of the feature — the topic's own name words and how much
    of each the text has — is on the screen at all.
  - **The attribution/causation distinction** moved from the mode letters into
    each block's own line ("an attribution: the feature's own arithmetic …,
    never the model's reading"; "an attribution of the model's own
    probability …, never a causal claim"; "a description of the data …, never
    the model's reading"; "the causal one, at word level").
  - **The palette.** Bold, cyan, a five-shade green ramp, a five-shade red
    ramp, yellow and dim — raw ANSI, the 256-colour ramps for the shades, no
    dependency. The brightness is the size. The gold labels are bold green, the
    contribution bars and the positive numbers are on the green ramp, the
    negative ones on the red ramp, the topic's own name words and the header's
    dataset and split are cyan, the corpus words that are **in this text** are
    bold with their counts in dim, and a what-if delta inside the model's noise
    band (**±0.003**) is dim and marked **"noise"**.
  - **The rule, enforced by a test.** The coloured screen, with its escape
    codes stripped, is **byte-identical** to the plain screen — so
    `--no-colour` loses nothing. A `--colour` flag forces the colours on when
    the output is piped; the default is "on when stdout is a terminal".
  - **The exit criteria.** The plain output carries every number and every
    marker (the strip-and-compare test), the coloured output is checked cell by
    cell (the header's split in cyan, a gold label in bold green, the topic's
    name words in cyan, the two ramps on the positive and the negative numbers,
    the notes dim), and the gate is green (700 tests, ruff clean, the two
    verify tools OK, both final sweeps reproducing field for field).

---

## The gold labels are always visible (CS-27) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.3
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-27.*

### 7.1 The gold labels are always visible (CS-27)

- [x] **CS-27 — the gold labels are always visible, and the in-text row keeps
  its shade.** Two presentation fixes in the viewer; the numbers do not move.

  **The mechanism.**

  - the two ranking blocks show the top-k **and**, below them, the **gold
    labels the top-k did not show** — each with its true rank and the ★
    marker, so the truth is readable whatever its rank is;
  - the corpus map's row for a word that is **in this text** keeps the ramp's
    shade and takes the bold on top of it (the counts stay dim and the "★ in
    this text" marker stays), so the log-odds' size stays readable on that row.

  **The cost.** Presentation only; **0 model calls**. **No run, no artifact and
  no table row** — the viewer is a tool (as in CS-20 and CS-25).

  **The exit criteria.** The plain output still carries every number and every
  marker (CS-25's strip-and-compare test); a gold label below the top-k
  appears in the ranking with its rank; the in-text row's shade is the ramp's;
  the gate green.

  **The runs.** None (a tool change). The log section is `## CS-27 — the gold
  labels are always visible` in `docs/experiments_text_processing.md`.

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the change set is a
  **tool** change: the viewer changes, it writes no artifact and appends no
  table row, and it costs **0 model calls**. The log section is
  `## CS-27 — the gold labels are always visible` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (the tool's own `--sample --seed 7`
  draw on the UPR **val** split; every number on it is the artifacts' or the
  tool's own reading of the model).**

  - **The ranking blocks** show the top-k **and**, below them, the gold labels
    the top-k did not show, each with its true rank and the ★ marker. On the
    seed-7 draw the text-side ranking ends with `17 ESC rights - general
    0.0804 ★` and `27 Labour rights 0.0714 ★`, and the topic-side ranking with
    `11 Labour rights 0.5000 ★` and `27 ESC rights - general 0.3810 ★`.
  - **The corpus map's in-text row** keeps the ramp's shade and takes the bold
    on top of it (the counts stay dim, the "★ in this text" marker stays), so
    the log-odds' size stays readable on that row.
  - **No number moved**: the rankings are the same numbers in the same order,
    and the two added rows are rows that were already in the artifact's
    ranking (the viewer recomputes the whole ranking, it just printed only its
    head).
  - **The exit criteria.** The plain output still carries every number and
    every marker (CS-25's strip-and-compare test); a gold label below the
    top-k appears in the ranking with its rank (a unit test of the ranking
    helper and a screen test on the fixture); the in-text row's shade is the
    ramp's (a test on the coloured output); and the gate is green (703 tests,
    ruff clean, the two verify tools OK, both final sweeps reproducing field
    for field).

---

## The viewer shows every gold label (CS-28) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.3
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-28.*

### 7.1 The viewer shows every gold label (CS-28)

- [x] **CS-28 — the viewer shows every gold label.** The two blocks that ask
  a question **of a label** ask it of every gold label of the text, not only
  of the focus topic.

  **The mechanism.** In both blocks the focus topic comes first (named), and
  the **other gold labels** follow, each with its own words, their numbers and
  the label's own value. A label with no train example keeps saying so (its
  corpus table is empty **by construction**) rather than being skipped.

  **The cost.** Presentation only; **0 model calls**. **No run, no artifact and
  no table row** — the viewer is a tool (as in CS-20, CS-25 and CS-27).

  **The exit criteria.** The plain output still carries every number and every
  marker (CS-25's strip-and-compare test); every gold label of the text
  appears in both blocks with its words and its value; the gate green.

  **The runs.** None (a tool change). The log section is `## CS-28 — the viewer
  shows every gold label` in `docs/experiments_text_processing.md`.

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the change set is a
  **tool** change: the viewer changes, it writes no artifact and appends no
  table row, and it costs **0 model calls**. The log section is
  `## CS-28 — the viewer shows every gold label` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (the tool's own `--sample --seed 7`
  draw on the UPR **val** split; every number on it is the artifacts' or the
  tool's own reading of the model).**

  - **The topic side.** The focus topic first (named), then the other gold
    labels of the text, each with its own name words, their coverage in this
    text and the label's own value:
    - the focus topic "Right to education" — the feature's value **0.9000**
      (*right* 0.8000, *education* 1.0000);
    - "ESC rights - general" — **0.3810** (*esc* 0.0000, *rights* 1.0000,
      *general* 0.1429);
    - "Labour rights" — **0.5000** (*labour* 0.0000, *rights* 1.0000);
    - "Minority rights" — **0.8750** (*minority* **0.7500**, *rights* 1.0000).
  - **The corpus map.** The same shape: the focus topic's table first, then
    the other gold labels' tables (the train corpus's own counts):
    - "Right to education": *dropout* 6/6 (the log-odds 5.461), *education*
      79/120 (5.317, the word is in this text), *higher* 3/3 (4.808),
      *illiteracy* 3/3 (4.808), *students* 3/3 (4.808);
    - "ESC rights - general": **no train example** — the table is empty **by
      construction** (it is one of the eight held-out labels);
    - "Labour rights": *workers* 35/39 (5.454), *189* 4/4 (5.131), *labour*
      31/36 (5.061), *icrmw* 2/2 (4.520), *trade* 2/2 (4.520);
    - "Minority rights": *minorities* 18/19 (**6.225**, the word is in this
      text), *minority* 17/18 (6.145), *roma* 10/11 (5.474), *albinism* 2/2
      (4.983), *dalits* 2/2 (4.983).
  - **What the blocks now say that they did not say before.** The text carries
    "**minorities**", and "Minority rights"'s own word is "minority", so the
    topic side's coverage is **0.7500** and not 1.0 — the word's last three
    characters differ; and the corpus map says the same thing from the other
    side: the corpus's word for that label is "**minorities**" (18 of the 19
    train texts that have it carry the label, the log-odds **6.225**), with
    "minority" right behind (6.145). "Labour rights" scores 0.5000 because the
    text has no "labour" at all — while the corpus's own words for that label
    are "workers" (35 of 39, the log-odds 5.454), "labour" (31 of 36) and
    "icrmw". So the two blocks are two readings of the same near-miss: the
    topic side says how much of the label's *name* the text has, the corpus map
    says which *words* the label actually travels with.
  - **No number moved**: the values on the added rows are the feature's own
    values (the same `lex_label_coverage` the ranking shows) and the corpus
    map's own counts (the same rows the artifact holds); the change adds rows.
  - **The exit criteria.** The plain output still carries every number and
    every marker (CS-25's strip-and-compare test); every gold label of the
    text appears in **both** blocks with its words and its value (two tests on
    the fixture, including the label without a train example); and the gate is
    green (705 tests, ruff clean, the two verify tools OK, both final sweeps
    reproducing field for field).

---

## The lexical floor (CS-29) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.4
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-29.*

### 7.4 The lexical methods round (CS-29 … CS-35)

- [x] **CS-29 — the lexical floor: TF-IDF + one-vs-rest.** A new predictor kind
  (`lexical_tfidf`): TF-IDF (word 1–2 grams, sublinear tf) over the **train**
  split, one-vs-rest logistic regression per label, scored on the evaluated split.
  It uses the same `Predictor` port as every other predictor, so the existing
  `evaluate` command measures it with the existing metric code and the existing
  decision rules.

  **Why it is first.** It is the measurement that says whether the pipeline's
  numbers are good, and every later change set is read against it. The reviewer
  did not propose it; the sweep did.

  **The cost.** Half a day. **No new dependency** — scikit-learn is already a
  core dependency. One precedent to record: `predictors/` has never imported
  scikit-learn (it is numpy-only today); scikit-learn is not a model client (no
  download, no network, no I/O) and is not on AGENTS.md's forbidden list for the
  pure cores, so this is allowed — but it is a first, and the pre-registration
  says so.

  **The run.** `upr-eval-lexical-tfidf-val`, the operating point `top_k` k=2 (the
  champion's own). `evaluate` is a main-pipeline command, so the row lands in the
  **main** UPR table (`docs/experiments_table_upr.md`, the `baselines` section)
  and the narrative in the track log.

  **The falsification, registered in advance.** The **unseen bucket is 0.0000 by
  construction** — the eight held-out labels have no train examples, so a
  supervised model cannot predict them. That is a structural fact, not a result
  (and the champion's unseen bucket is 0.0000 too, so the two *are* comparable
  there; the decision-model run's 0.5697 is the one that is not). If the pooled
  micro-F1 is **≥ 0.5479** (the champion's), the champion's advantage is not real
  and PLAN's Phase 6/9 priorities change — report it loudly. If it is **≤ 0.30**,
  the lexical route is closed as a standalone and only the map-as-feature route
  (CS-32) remains. The band we expect is **0.35–0.50**, and the reason is
  external rather than hopeful: on *ECtHR* (human-rights legal text, multi-label)
  LexGLUE's leaderboard puts `TFIDF+SVM` at **64.7 micro-F1**, and Clavié &
  Alphonsus (2021) measure the gap between a plain linear SVM and a fine-tuned
  legal transformer at only **15.1 %** error reduction on that task (against
  85.1 % on general-domain text).

  **The exit criteria.** The run's artifact holds the number; the table row is
  auto-appended; the log section reads the numbers out with the falsification
  conditions; the gate green.

  **The pre-registration (2026-11-01, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **The predictor.** A new registry entry `lexical_tfidf`, behind the same
    `Predictor` port as every other predictor: the `evaluate` command builds
    it, scores the split and measures it with the **existing** metric code and
    the **existing** decision rules. Nothing about the measurement changes.
  - **Where the code lives.** `src/cluster_rag/predictors/lexical_tfidf.py` —
    the adapter **and** the scikit-learn fit, because the vectorizer and the
    logistic regression are scikit-learn's own and the track's pure package
    may not import scikit-learn outside its combiners
    (`TEXT_PROCESSING_PLAN.md` §2). It is the same "adapter in `predictors/`"
    shape as CS-30's corpus-map predictor.
  - **The fit is train-only.** The TF-IDF **vectorizer is fitted on the train
    split** (its vocabulary and its idf), and so is every logistic regression
    (the train texts' matrix against the train labels' indicator). The
    evaluated split is only ever **transformed**. No val text and no val label
    enters the fit; `test` is not touched.
  - **The vectorizer.** Word analyzer, **1–2 grams** (`ngram_range=(1, 2)`),
    **sublinear tf** (`1 + log(tf)`), scikit-learn's default token pattern
    (2+ word characters) and its default **lowercasing**, **no stop-word
    removal** (`stop_words=None` — no list is a config value here, and the idf
    weighting already neutralises the register: "rights" is in all 60 of the
    corpus map's tables), `min_df=1` (every word the train split has),
    L2-normalized rows.
  - **The classifier.** One **one-vs-rest logistic regression per label**:
    `LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
    class_weight=None)` — scikit-learn's defaults, untuned. `max_iter` is
    raised from the default 100 so the fit converges (the run records whether
    a convergence warning was raised).
  - **The score.** The positive-class **probability**
    (`predict_proba()[:, 1]`) — a bounded [0, 1] scale.
  - **The eight held-out labels.** A label with no train example cannot be
    fitted, so it scores **0.0** for every text; and since a fitted label's
    probability is strictly above 0, the eight always rank **last** and never
    enter a top-2. That is what makes the unseen bucket **0.0000 by
    construction** — comparable with the champion's own 0.0000, and with the
    decision-model run's 0.5697 the one number that is not comparable.
  - **The parameters** (all in `eval.predictor.params`, all recorded in the
    run's `predictor` block): `ngram_max` (2), `min_df` (1),
    `sublinear_tf` (true), `lowercase` (true), `stop_words` (null), `C` (1.0),
    `max_iter` (1000), `class_weight` (null). **The run uses the defaults**;
    the knobs exist so a later change set can vary one thing at a time
    without editing code.
  - **The precedent, recorded.** `predictors/` has never imported
    scikit-learn (it was numpy-only until now). Scikit-learn is not a model
    client — no download, no network, no I/O — and it is **not** on AGENTS.md's
    forbidden list for the pure cores, so this is allowed; but it is a first,
    and it is recorded here.
  - **The run.** `upr-eval-lexical-tfidf-val`, the **val** split, the
    operating point **`top_k` k=2** (the champion's own). `evaluate` is a
    main-pipeline command, so the row lands in the **main** UPR table
    (`docs/experiments_table_upr.md`, the `baselines` section) and the
    narrative in the track log (`docs/experiments_text_processing.md`, the
    section `## CS-29 — the lexical floor`).
  - **The sweep.** `experiments/lexical_tfidf_sweep.py` — one cell, the sweep
    label `baselines` (the section the block above names).
  - **The exit criteria.** The run's artifact holds the number; the table row
    is auto-appended; the log section reads the numbers out with the
    falsification conditions; the gate green (ruff over `src`/`tests`/
    `experiments`, the full pytest suite, the two verify tools, and both final
    sweeps reproducing their artifacts field for field).

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the run is
  `upr-eval-lexical-tfidf-val` (the artifact
  `results/upr-eval-lexical-tfidf-val/metrics.json`), the log section is
  `## CS-29 — the lexical floor: TF-IDF + one-vs-rest` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, 1,676 train
  texts, 68 label columns, the champion's own `top_k` k=2 operating point; the
  run is the **existing** `evaluate` command, so the metric code and the
  decision rules are the ones the champion was measured with).**

  - **The headline.** The pooled **micro-F1 0.3845** against the champion's
    **0.5479** — a gap of **16.3 points**. The macro-F1 **0.2772**, the
    sample-F1 **0.3676**, the coverage **0.588** (40 of the 68 labels are
    suggested at least once, against the champion's 59), the P@5 **0.2161**,
    the R@10 **0.6085**, the nDCG@5 **0.5053** and the shortlist R@50
    **0.7205** (the champion's 0.7423).
  - **Neither pre-registered falsification fires.** The "≥ 0.5479" condition
    (the champion's advantage is not real) does not hold — the number is
    0.1634 below the champion. The "≤ 0.30" condition (the lexical route is
    closed as a standalone) does not hold either — the number is 0.0845 above
    that line. The number sits inside the **expected band 0.35–0.50**.
  - **The unseen bucket is 0.0000 by construction, as registered.** The eight
    held-out labels have no train example, so they are not fittable; they
    score 0.0 and always rank last. The measured unseen micro-F1 is
    **0.0000** and its coverage **0.000** — comparable with the champion's
    own 0.0000, and *not* comparable with the decision-model run's 0.5697.
  - **The per-bucket reading: the head and the torso, and nothing else.**
    The head **0.4695** (the champion's 0.6953, the coverage 1.000), the
    torso **0.4061** (0.5121, the coverage 0.846), the **tail 0.0301**
    (0.5287, the coverage **0.111** — two of the eighteen tail labels are ever
    suggested). A tail label's own words are rare words (CS-23's corpus map),
    so a one-vs-rest regression fitted on 1,676 texts has almost nothing to
    learn from.
  - **The lexical route is a candidate generator on par with the champion and
    a ranker behind it.** The shortlist R@50 is 0.7205 against the champion's
    0.7423 (and the trivial baseline's 0.6764), while the nDCG@5 is 0.5053
    against 0.6725 and the P@5 0.2161 against 0.2770 — the same shape as the
    learned pack's CS-8d finding, reached by a different route.
  - **The fit's own facts.** **17,217** TF-IDF features (the train
    vocabulary, unigrams + bigrams), **60 of 68** labels fitted, **8** with no
    train example, **0** degenerate, the train positives per label min **1** /
    max **289**, and **0** convergence warnings (all 60 fits converged inside
    `max_iter=1000`). Scikit-learn 1.9.1.
  - **The cost.** **67.3 s** of CPU (61.6 s on the re-run; the artifact is
    identical apart from the wall clock). **No model calls**, no embedding
    run, no new dependency.
  - **The precedent, recorded.** `predictors/` imports scikit-learn for the
    first time. It is not a model client (no download, no network, no I/O),
    it is a core dependency, and it is not on AGENTS.md's forbidden list for
    the pure cores — so it is allowed, but it is a first.
  - **One thing this change set also did, and says so plainly.** The row
    landed in the main UPR table's `baselines` section, whose eight older rows
    predate the `embedding` column; the table writer re-synced the section's
    **header**, which left those rows one cell short, so the **baselines
    sweep was re-run** and every row in the section was rewritten in the
    current shape. Every number in those eight rows is **identical
    field-for-field**; only the `date` and the wall-clock `seconds` move.
  - **The exit criteria.** The artifact holds the number; the table row is
    auto-appended; the log section reads the numbers out with the
    falsification conditions; the gate green (716 tests, ruff clean over
    `src`/`tests`/`experiments`, the two verify tools OK, both final sweeps
    reproducing their artifacts field for field).

---

## The corpus map as a classifier (CS-30) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.4
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-30.*

### 7.4 The lexical methods round (CS-29 … CS-35)

- [x] **CS-30 — the corpus map as a classifier.** A new predictor kind
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

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the four runs are
  `upr-eval-corpus-map-{sum-logodds,sum-positive,max-logodds,support-count}-val`
  (the artifacts `results/upr-eval-corpus-map-*-val/metrics.json`), the log
  section is `## CS-30 — the corpus map as a classifier` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, the champion's
  own `top_k` k=2 operating point, the map built from the **train** split only
  — 1,676 texts, 3,049 words, 18,707 (label, word) pairs; **no model calls, no
  embedding, 0.8 s per run**).**

  - **The headline.** The **corpus map as a classifier** is the arm the fit
    side selects, **`sum_positive`** (the "evidence for" score); its pooled
    **micro-F1 is 0.4924** against the champion's **0.5479** — a gap of **5.6
    points**. The best arm *on val* is the same arm, so the discipline cost
    nothing here.
  - **The four conditions.** (i) The **unseen bucket is 0.0000 by
    construction** — it holds (the empty tables are ranked last in every row);
    the bucket's *ranking* metrics inside the restricted view are a tie
    artefact and are not read. (ii) The **measure ablation does not fire**:
    the log-odds arms reach **0.4848–0.4924** against the measure-free
    control's **0.2698**, so the map's careful measure work buys **+0.2226
    micro-F1** — the opposite of the registered null. (iii) The "within 0.02
    of the champion" condition **does not fire, but it nearly does**: the
    selected arm is **0.0555** below the champion, and the seen-bucket reading
    **0.0626** below it (0.5555 against 0.6181) — a **pure corpus statistic
    reaches 90% of the champion's micro-F1**. (iv) The seen-bucket read: the
    map 0.5555, the champion 0.6181, CS-29's lexical floor 0.4338, the trivial
    floor 0.1595 — the same order as the pooled reading.
  - **The cross-fit was the right instrument, and that is now a measurement.**
    The **in-fold** fit side puts the measure-free control only **0.0227**
    behind the best arm (0.7504 against 0.7731) — it would have said the
    measure work buys almost nothing; the **out-of-fold** fit side puts it
    **0.2780 behind** (0.3361 against 0.6141), and the **val** ablation lands
    between them (**0.2226**). The mechanism is CS-23's own: a word that
    occurs in exactly one train text has a huge log-odds (2,540 of the map's
    18,707 pairs rest on such a hapax), so in-fold the *sum* collects a big
    inflated contribution from a text's own hapax word that the *count* (which
    sees the same word as 1) does not.
  - **The per-bucket reading.** head **0.6611**, torso **0.4745**, **tail
    0.4386** (the coverage **1.000** — every tail label is suggested at least
    once), unseen 0.0000. The tail is where the two lexical instruments
    separate most sharply: the map reaches 0.4386 there where CS-29's TF-IDF
    floor reaches **0.0301** (the coverage 0.111 against the map's 1.000).
  - **The cost.** **No model calls, no embedding, no new dependency**; 0.8 s
    per run (the map build plus the 5-fold cross-fit).
  - **The exit criteria.** The four artifacts hold the numbers; the four table
    rows are auto-appended (the main UPR table's new `corpus_map` section); the
    log section reads the numbers out with the falsification conditions; the
    gate green (ruff over `src`/`tests`/`experiments`, the full pytest suite,
    the two verify tools, both final sweeps reproducing their artifacts field
    for field).

---

## The corpus map's estimator (CS-31) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.4
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-31.*

### 7.4 The lexical methods round (CS-29 … CS-35)

- [x] **CS-31 — the corpus map's estimator, and the lexicon as a first-class
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

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the six runs are
  `upr-eval-corpus-map-{sum-logodds,sum-positive,max-logodds,support-count,sum-zscore,sum-zscore-positive}-val`
  (the artifacts `results/upr-eval-corpus-map-*-val/metrics.json`), the lexicon
  artifact is `results/upr-hitmap-corpus/hitmap.json` (re-written with the new
  measures), the paired comparison is `tools/lexical_compare.py`, the log
  section is `## CS-31 — the corpus map's estimator, and the lexicon as an
  artifact` in `docs/experiments_text_processing.md`, and the numbers are
  below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, the champion's
  own `top_k` k=2; the map is built from the **train** split only; no model
  calls, no embedding, 1.3 s per run).**

  - **The falsification, read out.** (i) The **head-word test: the registered
    null does NOT fire** — the z-scored head word differs from the log-odds's
    for **13 of the 60** labelled tables (the null needed ≥ 55 unchanged) —
    **but the plan's expectation does NOT hold either**: the z-scored head is a
    hapax for the **same 4 labels** as the log-odds's head. The reason is the
    formulas' own: the **informative prior shrinks the COMMON words** (the
    register) and **the standard error is what shrinks the rare ones**. (ii)
    The **model test: the better measure is a better MODEL, not only a better
    description** — the pooled micro-F1 **moves** from the log-odds arm's
    **0.4924** to the z-scored arm's **0.5243** (**+0.0319**), and the
    **paired** bootstrap over the texts gives **-0.0319 [-0.0526, -0.0120]**
    (the log-odds arm minus the z-scored arm) — the interval **excludes zero**
    and the z-scored arm wins on every resample.
  - **The selection moved.** With the z-scored arms in the pool, the fit side's
    out-of-fold maximum is `sum_zscore_positive` (**0.6613** against the
    log-odds arms' 0.6026–0.6141), so **CS-30's headline (0.4924) is
    superseded**: the corpus map as a classifier is now **0.5243**, and its gap
    to the champion's 0.5479 is **0.0236**.
  - **The shrinkage shows itself where it should.** The z-scored arm's
    **in-fold** number (0.7399) is *below* the log-odds arms' (0.7656–0.7731)
    and its **out-of-fold** number (0.6613) is *above* them — the textbook
    signature of a prior that shrinks the noisy weights. (The opposite of
    CS-30's `support_count` case, where the in-fold number was the
    optimistic one.)
  - **The plain `sum_zscore` arm collapses to 0.1011**, and that is a reading
    rather than a defect: the z-score is signed, so the words that go
    *against* a label drag the sum down; the "evidence for" restriction is what
    makes the z-scored weights usable.
  - **The tail.** The z-scored arm's tail is **0.5435** — above the
    champion's **0.5287** and well above the log-odds arm's 0.4386 and
    CS-29's lexical floor's 0.0301.
  - **The conditional smoothing** flips the sign of **798 of the 18,707 pairs
    (4.27%)** — the share the plan asked for.
  - **The lexicon artifact** is the map's own block, written once
    (`results/upr-hitmap-corpus/hitmap.json`, 10.7 MB): every pair's new
    measures beside CS-23's (whose numbers are **unchanged**) plus a summary
    block (the head word per label under each measure, the head-word
    agreement, the sign-flip share) — so CS-32, CS-33, CS-35 and the viewer
    all read the same numbers.
  - **The exit criteria.** The map's artifact carries the lexicon; the six
    runs hold their numbers; the table rows are auto-appended; the log section
    reads the numbers out with the falsification conditions; the paired
    comparison's tool **checks its own rebuild against the runs' artifacts**
    before it reports anything; the gate green.

---

## The corpus map in the learned pack (CS-32) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.4
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-32.*

### 7.4 The lexical methods round (CS-29 … CS-35)

- [x] **CS-32 — the corpus map in the learned pack.** The map's per-cell score
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

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the run is
  `upr-analyze-cmap` (the artifact `results/upr-analyze-cmap/analysis.json`),
  the sweep is `experiments_text_processing/upr_learned_cmap_sweep.py` (the
  label `upr_learned_cmap`), the log section is `## CS-32 — the corpus map in
  the learned pack` in `docs/experiments_text_processing.md`, and the numbers
  are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2,
  the gradient-boosted combiner; the map is built from the **train** split only
  and the fit side's map is **cross-fit** over 5 folds; no model calls; 93 s).**

  - **The headline.** The pack with the map's feature reaches a pooled
    **micro-F1 0.5502** — past the **champion's 0.5479** — against the same
    pack **without** the group (the control) at **0.4818**. The paired
    difference (the resampling unit the text) is **+0.0684 [+0.0441, +0.0915]**,
    and the real arm wins on **every** resample.
  - **The label-blind counterpart.** The blind map's own gain over the same
    control is **−0.0509 [−0.0678, −0.0340]**: the blind map buys nothing,
    it costs five points — so the gain is **label-driven**, not the map's
    construction. (The registered gloss "the blind Δ's interval includes zero"
    is **not** met — the interval is significantly **negative**; the
    condition's substance holds and its literal wording is recorded as not
    met in the log.)
  - **The unseen bucket is NOT flat.** The map's empty tables give the eight
    held-out labels the floor, so the column says "impossible" for them and
    the control's unseen **0.4867** collapses to **0.0000** (the paired
    difference **−0.4867 [−0.5446, −0.4256]**). The seen buckets give back
    more: head **+0.1675 [+0.1343, +0.2045]**, torso **+0.0901 [+0.0397,
    +0.1408]**, tail **+0.0876 [+0.0146, +0.1638]**.
  - **The falsification does not fire**: the corpus's word statistics add
    something the label-name overlap features do not already carry — but what
    they add is a **seen-labels** feature.
  - **The round's most useful number is in the random-vector floor**: the
    floor (the embedding sides replaced by noise) reaches **0.5357**, only
    **0.0145** below the real arm, so on this pack the **embedding geometry
    contributes 0.0145** and the **map's feature 0.0684** — a corpus statistic
    worth roughly five times the embedding side.
  - **The exit criteria.** The artifact holds the Δs with their intervals, the
    folds and the three floors; the table row is auto-appended; the log
    section reads the numbers out with the falsification conditions; the gate
    green.

## The word-class readout (CS-33) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.4
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-33.*

### 7.4 The lexical methods round (CS-29 … CS-35)

- [x] **CS-33 — the word-class readout: what kind of words affect the label, in
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

  **The pre-registration (2026-11-01, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **The classes** (from the map's own structure, no new dependency):
    - **distinctive** — the word sits in **1 or 2** of the labelled tables
      (the block's own wording);
    - **generic** — the word sits in **at least a quarter** of the labelled
      tables (≥ 15 of the 60; a share, not a magic number, and it puts the
      register where the block's examples put it: "rights" in 60, "include"
      in 59);
    - **shared** — the middle (3–14 tables);
    - and the **name word** as a *flag*: the word is in the label's own name
      (it overlaps the classes above — "rights" is the name word of "Rights of
      the Child" *and* generic).
  - **The readout.** Per class: the number of (label, word) pairs, the share
    with a **positive** log-odds (the word goes *with* the label), the share
    with a **negative** one (it goes *against* it) and the mean log-odds — the
    reviewer's question in the map's own vocabulary: *which kind of word
    pushes the label up, and which pushes it down*. The same readout for the
    **surrogate's** per-topic coefficients (the model's own habits: the
    coefficients split the same way).
  - **The payoff, and the decision it drives.** If the classes separate, a
    **class-restricted map** is built and measured as a classifier: the map
    with the **generic** class dropped (the register — the readout says it is
    the weak and often negative class) and, at the other extreme, the
    **distinctive-only** map. Both are re-run through the **existing** scorer
    and the **existing** `evaluate` command, on the same texts at the same
    operating point, so the comparison against the full map is paired
    (`tools/lexical_compare.py`).
  - **The falsification, registered in advance, and it decides CS-34.** The
    readout is a **null** (and **CS-34, the POS dependency, is cancelled**) if
    the generic and the distinctive classes have the **same** profile: the
    test is that the distinctive class's mean log-odds exceeds the generic
    class's by **at least 1.0** (one natural-log unit — a factor of e in the
    odds) **and** the generic class's share of positive pairs is lower. If
    either half fails, the readout is a null, no class-restricted map is
    worth building, and CS-34 is cancelled.
    *(Read only on the **train** split, before the run: the classes separate
    widely — the distinctive words (1–2 tables) are **100%** positive with a
    mean log-odds of **+4.47**, the generic ones (≥ 15 tables) **70.8%**
    positive with a mean of **+0.43** and **29.2%** of their pairs pushing
    the label *down*. So the registered null does not fire, and the
    class-restricted map is worth building.)*
  - **The runs.** Two more corpus-map runs, the same arm as CS-31's selection
    (`sum_zscore_positive`) with a **restricted map**:
    `upr-eval-corpus-map-sum-zscore-positive-nogeneric-val` and
    `upr-eval-corpus-map-sum-zscore-positive-distinctive-val` (UPR **val**,
    `top_k` k=2). The sweep file
    `experiments_text_processing/corpus_map_classes_sweep.py` (the sweep label
    `corpus_map_classes`); the rows land in the **main** UPR table.
  - **The exit criteria.** The map's artifact carries the class per pair and
    the readout block; the surrogate's block carries the model's own readout;
    the two runs hold their numbers; the table rows are auto-appended; the
    viewer gains the block (the viewer's own pattern: a block names the
    question it answers); the log section reads the numbers out with the
    falsification conditions; the gate green.

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the two runs are
  `upr-eval-corpus-map-sum-zscore-positive-nogeneric-val` and
  `upr-eval-corpus-map-sum-zscore-positive-distinctive-val` (UPR **val**, 660
  texts, `top_k` k=2, the arm `sum_zscore_positive`), the sweep is
  `experiments_text_processing/corpus_map_classes_sweep.py` (the label
  `corpus_map_classes`), the map's and the surrogate's artifacts are
  `results/upr-hitmap-corpus/hitmap.json` and
  `results/upr-hitmap-surrogate/hitmap.json`, the log section is `## CS-33 —
  the word-class readout` in `docs/experiments_text_processing.md`, and the
  numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2,
  the map built from the **train** split only; no model calls; 0.8 s and 0.6 s
  per run).**

  - **The readout** (the map's own block, `corpus_map.word_classes`; the map's
    log-odds on the train split's 18,707 pairs):
    - **distinctive** (the word sits in 1–2 of the 60 labelled tables): 1,836
      pairs, **100.0 %** positive, mean log-odds **+4.2676**;
    - **shared** (3–14 tables): 8,711 pairs, 97.8 % positive, mean **+2.6410**;
    - **generic** (≥ 15 tables): 8,160 pairs, **81.0 %** positive (19.0 %
      negative), mean **+0.8439**;
    - **129** of the pairs are **name words** (89 generic, 39 shared, 1
      distinctive).
  - **The registered null does NOT fire**: the distinctive class's mean exceeds
    the generic class's by **+3.4237** (the margin needs 1.0) **and** the
    generic class's share of positive pairs is lower (81.0 % against 100.0 %).
    The readout is **not a null** and — by the pre-registration's own rule —
    **CS-34 is not cancelled** by it.
  - **The payoff is a loss, and it is the more useful half.** The
    class-restricted maps are **much worse** as classifiers than the full map:
    the map **without the generic class** reaches a pooled **micro-F1 0.3815**
    (the fit side's out-of-fold 0.5097) and the **distinctive-only** map
    **0.0798** (out-of-fold 0.1109), against the full map's **0.5243**
    (out-of-fold 0.6613). The **paired** bootstrap over the texts gives
    **+0.1429 [+0.1200, +0.1655]** and **+0.4445 [+0.4178, +0.4713]** against
    the full map — the full map wins on **every** resample in both comparisons,
    and the **fit side agrees**, so it is not a val artefact.
  - **The reason**: the distinctive words are the **strongest** words but they
    are **rare**; the register is **weak** but it is **everywhere**. A
    class-restricted map keeps the signal and throws away the coverage.
  - **The surrogate's own readout** (the model's habits, the same split): the
    **name words** are 82.9 % positive with a mean coefficient of **+0.20762**,
    an order of magnitude above every class mean (the distinctive class
    45.2 % positive at −0.00066, the generic class 42.5 % at −0.00233). The
    model's habit is "the label's own name words"; the corpus's habit is "the
    register" — and the two readouts disagree about what carries a label.
  - **The exit criteria.** The map's artifact carries the class per pair and
    the readout block; the surrogate's block carries the model's own readout;
    the two runs hold their numbers; the table rows are auto-appended (the main
    UPR table's `corpus_map_classes` section); the viewer gains the block; the
    log section reads the numbers out with the falsification conditions; the
    gate green.

## The real POS tags (CS-34) — the pre-registration and the verdict

*Added on 2026-11-01, when the round was executed: the live plan's §7.4
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-34.*

### 7.4 The lexical methods round (CS-29 … CS-35)

- [x] **CS-34 — real part-of-speech tags.** *Gated on the reviewer's OK: a new
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

  **The pre-registration (2026-11-01, written before the first call — the
  reviewer's OK on the dependency and the download was given on 2026-11-01;
  this fixes the *shape* of the number).** The block above fixes the
  mechanism; this fixes the number.

  - **The tagger, at the edge.** A new package `src/cluster_rag/pos/`:
    `base.py` — the **port** (a `Tagger` protocol: the token stream in, the
    `(token, tag)` pairs out, plus the provenance) — and `nltk_tagger.py` — the
    **backend** (`nltk`'s `averaged_perceptron_tagger_eng`), with its own disk
    cache (`cache.py`, the decision cache's shape). The track's package
    imports the **port** only: one line in AGENTS.md's pure-core rule (the
    `cluster_rag.pos.base` port). The dependency is **`nltk`** (MIT), added to
    `pyproject.toml`; a new `pos:` config section (the model id and the
    backend, **off by default** so every earlier run's artifact keeps its exact
    content).
  - **The cache and the digest.** The tagger is a model client, so its answers
    are cached on disk like the embedder's and the decision model's: the key
    is `(model id, digest, the text's sha1)`, the cache lives under the run's
    own cache root (`cache/pos/`), and the **model's digest** (the sha256 of
    the tagger's own model files) is recorded in the artifact's provenance — so
    a re-run knows which build of the tagger it read.
  - **The corpus that is tagged, and the tokens that are tagged.** The **train**
    split (the map's corpus): no val text and no test text enters the tag
    distribution. The tagger's input is the **track's own tokenization**
    (`[a-z0-9]+` on the lower-cased text, the map's unit) — so the tagger's
    tokens and the map's words line up **by position**, exactly: no second
    tokenizer, no `punkt` download, and no join to be guessed. The token
    stream keeps the stopwords and the one-character tokens (the tagger needs
    the sentence's shape); the **map's words** are the content tokens of that
    same stream, so every map word gets its tags from the positions where it
    occurs.
  - **The classes.** The **Penn tagset** grouped into five coarse classes:
    **noun** (NN, NNS, NNP, NNPS), **verb** (VB, VBD, VBG, VBN, VBP, VBZ),
    **adjective** (JJ, JJR, JJS), **adverb** (RB, RBR, RBS, WRB) and **other**
    (the rest). A **word's** class is the class of its **modal tag** — the tag
    it takes most often across the train corpus — and the words that take
    **several** tags are reported as such: the per-word tag *distribution* is
    stored and the **share of the map's words with more than one tag** is
    reported (the plan's own check says 16.3 % over the 3,000-text corpus;
    this is the train split's own number, and it is the one the artifact
    carries). A word with no tag at all is counted apart, never dropped
    silently.
  - **The readout.** Per class: the number of (label, word) pairs, the share
    with a **positive** log-odds, the share with a **negative** one and the
    mean log-odds — CS-33's readout, crossed with the grammatical class.
    Reported **twice**: by the **modal tag** (the partition — every pair
    counted once) and by **every tag the word takes** (the non-partition, so a
    word that is a noun here and a verb there is visible in both). And the
    same readout for the **surrogate's** per-topic coefficients (the model's
    own habits).
  - **The payoff.** The **POS-restricted map** as a classifier: the map with
    the **nouns only** (the topic words) and, at the other extreme, the map
    with the **verbs only** (the predicate words) — the two ends of the
    grammatical axis, fixed in advance. Both through the **existing** scorer
    and the **existing** `evaluate` command, on the same texts at the same
    operating point (`top_k` k=2), so the comparison against the full map is
    paired (`tools/lexical_compare.py`).
  - **The runs.** Two classifier runs:
    `upr-eval-corpus-map-sum-zscore-positive-noun-val` and
    `upr-eval-corpus-map-sum-zscore-positive-verb-val` (UPR **val**, `top_k`
    k=2, the arm CS-31's fit side selected), the sweep
    `experiments_text_processing/corpus_map_pos_sweep.py` (the sweep label
    `corpus_map_pos`); the rows land in the **main** UPR table. Plus the two
    artifacts re-written once: the map's own block (the POS readout) and the
    surrogate's own block (the model's POS readout).
  - **The falsification, registered in advance.** The POS line is a **null**
    (and the dependency is not worth keeping) if **both** of these fail:
    1. **the readout's separation** (the plan's "the same as CS-33's, now with
       the grammatical class"): the **noun** class's mean log-odds must exceed
       the **verb** class's by **at least 1.0** (the same natural-log unit) and
       the noun class's share of positive pairs must be higher. (The nouns are
       the label-carrying words; the verbs are this genre's register.)
    2. **the payoff**: the **noun-only** map must not be materially worse than
       the full map — the paired Δ's interval must **include zero** (the
       track's own reading of "materially smaller", as in CS-32).
    If the separation holds and the payoff does not, the line is a good
    **description** and a bad **model** (CS-33's shape) — reported as such,
    not as a null.
  - **The exit criteria.** The map's artifact carries the POS block (the
    tagger's provenance, the tag distribution, the readout per class and per
    tag, the separation test); the surrogate's block carries the model's POS
    readout; the two runs hold their numbers; the table rows are auto-appended;
    the viewer gains the POS lines; the log section reads the numbers out with
    the falsification conditions; the gate green (ruff over `src`/`tests`/
    `experiments`, the full pytest suite, the two verify tools, and both final
    sweeps reproducing their artifacts field for field).

  *(pre-registered 2026-11-01; **executed 2026-11-01** — the reviewer's OK on
  the dependency and the model download was given on 2026-11-01; the two runs
  are `upr-eval-corpus-map-sum-zscore-positive-{noun,verb}-val` (UPR **val**,
  660 texts, `top_k` k=2, the arm `sum_zscore_positive`), the sweep is
  `experiments_text_processing/corpus_map_pos_sweep.py` (the label
  `corpus_map_pos`), the map's and the surrogate's artifacts are
  `results/upr-hitmap-corpus/hitmap.json` and
  `results/upr-hitmap-surrogate/hitmap.json`, the log section is `## CS-34 —
  real part-of-speech tags` in `docs/experiments_text_processing.md`, and the
  numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2;
  the map built from the **train** split only and tagged with `nltk`'s
  `averaged_perceptron_tagger_eng` (the digest recorded in the artifact, the
  train corpus tagged with 1,676 cache hits and 0 misses; no model calls).**

  - **The readout** (the map's own block, `corpus_map.pos_classes`; the map's
    log-odds over the 18,707 pairs): **noun** 9,752 pairs, 90.1 % positive, a
    mean log-odds of **+2.0044**; **verb** 4,057, 91.4 %, **+1.9856**;
    adjective 3,605, 90.3 %, +2.0403; adverb 554, 94.6 %, +2.2221; other 739,
    93.0 %, +2.0820. The map's 3,049 words by class: 1,555 nouns, 660 verbs,
    600 adjectives, 108 adverbs, 126 other; **412 of the 3,049 (13.5 %)** take
    more than one tag (the per-word distribution is stored).
  - **The readout's separation FAILS**: the noun class's mean exceeds the verb
    class's by **+0.0188** (the margin needs 1.0) and the verb class's share of
    positive pairs (**91.4 %**) is *higher* than the noun class's (**90.1 %**).
    The grammatical axis does not separate the topic words from the predicate
    words.
  - **The payoff FAILS too**: the **noun-only** map reaches a pooled **micro-F1
    0.4840** (the fit side's out-of-fold 0.6175) against the full map's
    **0.5243** (0.6613) — a paired **+0.0403 [+0.0258, +0.0545]**, the interval
    **excluding zero**, so the restriction costs something measurable; the
    **verb-only** map reaches **0.1261** (**+0.3982 [+0.3735, +0.4244]**,
    out-of-fold 0.1550).
  - **So the POS line is a NULL** (both halves of the registered falsification
    fail) and **the dependency is not worth keeping**. The one thing it bought
    is the **per-tag** readout, where the closed-class words (the register in
    the grammatical sense: FW +0.67, RBR +0.49, MD +0.83, IN +1.30) are the
    weakest and the content words the strongest (NNP +3.65, JJS +3.29, CD
    +2.73). That is a **reading**, not a measurement — a closed-class map was
    not a pre-registered arm, so it is the reviewer's next question.
  - **The surrogate's own readout** agrees: the noun and the verb classes are
    within 0.2 points of each other (42.1 % against 44.4 % positive) and every
    class mean is within a thousandth of zero, while the **name words** are
    82.9 % positive at a mean coefficient of **+0.20762** — CS-33's finding
    stands.
  - **The exit criteria.** The map's artifact carries the POS block (the
    tagger's provenance, the tag distribution, the readout per class and per
    tag, the separation test); the surrogate's block carries the model's POS
    readout; the two runs hold their numbers; the table rows are auto-appended
    (the main UPR table's `corpus_map_pos` section); the viewer gains the POS
    lines; the log section reads the numbers out with the falsification
    conditions; the gate green.

## The label lexicon as a label-side channel (CS-35) — the pre-registration and the verdict

*Added on 2026-11-02, when the round was executed: the live plan's §7.4
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-35.*

### 7.4 The lexical methods round (CS-29 … CS-35)

- [x] **CS-35 — the label→word lexicon as a label-side channel.** *Gated on the
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

  **The pre-registration (2026-11-02, written before the first call — the
  reviewer's OK on the schema change was given on 2026-11-02; this fixes the
  *shape* of the number).** The block above fixes the mechanism; this fixes the
  number.

  *One reading settled before the first call.* The block says the mined-words
  channel is "empty for the eight held-out ones — exactly like the catalog's
  channels are today". The **catalog's** channels are *not* empty for the eight
  (the catalog is static label-side knowledge: it carries all 68 entries); what
  is empty for the eight is the **corpus map's** table (CS-23 measures it: 8 of
  the 68 have none). So the channel's label-holdout consequence is: **the
  mined-words channel is a seen-labels channel** — it exists for the sixty
  labels with train examples and is empty for the eight — and the **expansion
  route** is the only lexical path to the eight. The falsification's scope (the
  seen buckets) follows from that.

  - **The lexicon, and where it is computed.** The **corpus map** (CS-23/CS-31)
    *is* the label→word lexicon; there is nothing to mine again. The new object
    is the **label-side representation**: per label, the words that characterise
    it. It is computed **at the edge** (in the command, from the **train** split
    only — no val text, no val label, no test text), exactly like the corpus
    map's own feature (CS-32) and for the same reason: **the fit side's words
    must be cross-fit**, and a frozen file cannot provide a *fold's* words (see
    the guard below). The words live in a new pure module,
    `text_processors/lexical/label_lexicon.py` (the standard library only), and
    they ride into the pack on the **label side** (`UprLabel.mined_words`),
    beside the description and the parent.

  - **The mined words: which words, which weights.** For a label `L` with `n_L`
    train instances and a word `w` in its table: `tf(L,w) = n_with(L,w) / n_L`
    (the share of the label's instances that contain the word) and
    `idf(w) = log((1 + N) / (1 + df(w)))` (`N` = the train corpus's size, 1676;
    `df` = the word's document frequency in the train corpus). The **PIFA
    weight** is `tf(L,w) × idf(w)` — the coordinate of the label's **TF-IDF
    centroid** over its positive instances (the centroid's coordinate for `w` is
    exactly the mean of the instances' TF-IDF values, i.e. `tf × idf`; this is
    the X-Transformer PIFA construction, computed from the map's own counts).
    The label's **mined words** are the **top 10** words by that weight (ties by
    the word, so the order is deterministic). *K = 10, fixed, no grid:* the
    catalog's own description carries 6–11 content tokens (median 9), so ten
    words puts the mined-words label side at **the same length** as the
    name+description channel's label side — the comparison changes the label
    side's *content*, not its length (CS-11's one-variable discipline). Each
    mined word is stored with its weight, its `n_with`, its `df`, its
    `log_odds` and its `z_score`, so every number is auditable.

  - **The expansion route (the eight labels with no train example).** For a
    label with an **empty** table: take the label's **name**'s content tokens
    (the track's own tokenizer — `[a-z0-9]+`, the stopwords dropped) and match
    them against the **train corpus's vocabulary** (the map's own word list). A
    token that is in the vocabulary contributes **exactly that word**; a token
    that is not contributes the corpus words that **share a prefix of at least 4
    characters** with it (a crude stemmer: `seekers` → `seek`, `belief` →
    `beliefs`). The expansion's words are ordered by **rising document
    frequency** (the rarest first — the most specific word leads), ties
    alphabetically. **No labels are involved**: the route reads the corpus's
    *words* and their document frequencies, never a label association and never
    a gold label. **No val text is involved**: the vocabulary is the train
    corpus's. A label whose expansion is empty (its name's tokens are all
    outside the corpus — `Other`, whose only word is a stopword) falls back to
    its **name**, and the artifact records the fallback.
    *Read on the train split before the run (not a measurement — the same kind
    of reading CS-31 and CS-33 made):* **seven of the eight** held-out labels
    get an expansion (`Asylum-seekers - refugees` → `seek`; `ESC rights -
    general` → `general`, `rights`; `Freedom of association and peaceful
    assembly` → all four of its words; `Freedom of religion and belief` →
    `beliefs`, `freedom`; `Gender-based violence` → all three; `General` →
    `general`; `Special procedures` → both), and **one** (`Other`) falls back
    to its name. The corpus never says `asylum`, `refugees`, `religion` or
    `esc` — the route's reach is exactly what the corpus says.

  - **The label side (the query half — the PIFA idea).** A new `label_texts`
    variant **`mined_words`**: the label's text is its mined words (the
    expansion words for the eight, the name for the fallback), joined by a
    space, and rendered through the **same template** as the champion's
    (`This recommendation is about {label}.`) — against the name+description
    channel, the one variable that changes is the *content* of the label side,
    not the template and not the length. The `labelembed` command builds the
    map and the lexicon from the **train** split (it loads the split's fixed
    ids, with the same source-sha256 consistency check the `evaluate` command
    makes), renders the 68 label texts and writes them into the artifact as the
    other variants do — plus the **lexicon block** (the per-label words, the
    weights, the routes) and the provenance (the split's source sha256, the
    map's shape, the knobs). **No new dependency.**

  - **The pack's channel (the feature half).** A new **gated feature group**
    `label_mined` (the prefix `labelmined_`) with **one** column: the **soft
    coverage** of the label's mined words by the text — the *same statistic* as
    CS-11's `labelpart_description_coverage` (the label-description's soft
    coverage), over the mined words instead of the description. The words ride
    on the label side, so the group's gate is the catalog-gated machinery's
    own: the group is refused when the run has no lexicon, and a run without one
    records the gap in its artifact. The **label catalog's schema** gains the
    optional `mined_words` field (the loader parses it); the committed catalog
    stays **frozen** — it carries none, so its sha256 does not move — because
    the mined words are **corpus-derived**, and a frozen file cannot provide
    the fit side's fold words (the guard below).

  - **The guard (not optional): the fit side's cross-fit.** The pack is fitted
    on **train** and the mined words come from **train**, so on the fit side the
    label side has already seen the texts it scores: the channel would be
    optimistically biased exactly where the combiner learns its weight (CS-32's
    rule, applied to the label side). The fit side is therefore **cross-fit**:
    **5 folds** over the train records (seeded, deterministic — the same fold
    assignment as CS-32's `cross_fit_scores`), and each fold's cells are
    extracted with the label side built from the **other** folds' map (so the
    fold's label vectors *and* the fold's mined words both come from the other
    folds). The **evaluated** split is scored with the **full** train map. The
    artifact records the folds and the seed. The runs whose label side is
    **not** train-derived (the control, the description query, the catalog-gated
    groups) are **not** cross-fitted, so their artifacts must reproduce CS-11's
    exactly — the reproducibility check.

  - **The runs.** Seven, all on UPR **val** at the pack's operating point
    (`top_k` k=2; the gradient-boosted combiner on the full pack `embedding +
    lexical + chunk_profile`; the CS-8f/CS-11 configuration), in the sweep
    `experiments_text_processing/upr_learned_lexicon_sweep.py` (the sweep label
    `upr_learned_lexicon`, so the track's UPR table's section is
    `## UPR — upr_learned_lexicon`):

    | run | what |
    |---|---|
    | `upr-labelembed-embeddinggemma-mined` | the label artifact: the 68 mined-words label texts (the mined words; the expansion for the eight) |
    | `upr-eval-fusionc-embeddinggemma-k10-c1-mined` | the guard's reference for the mined label side (the champion with the mined label vectors) |
    | `upr-analyze-learned-lexicon-control` | the **control**: the champion's label side, the plain pack (CS-11's `cat-control`, re-run) |
    | `upr-analyze-learned-lexicon-desc` | the name+description channel's **query** half (the description label side; CS-11's `cat-desc`, re-run) |
    | `upr-analyze-learned-lexicon-catfeat` | the name+description channel's **feature** half (the catalog-gated groups; CS-11's `cat-feat`, re-run) |
    | `upr-analyze-learned-lexicon-mined` | the mined-words channel's **query** half (the mined label side) |
    | `upr-analyze-learned-lexicon-minedfeat` | the mined-words channel's **feature** half (the `label_mined` group) |

    The `labelembed` and `eval` runs are main-pipeline commands (their rows land
    in the **main** UPR table); the five `analyze-upr` runs land in the
    **track's** UPR table. The three re-runs of CS-11's arms double as the
    **reproducibility check** (they must reproduce `upr-analyze-learned-cat-
    {control,desc,feat}`'s numbers).

  - **What is reported.** Every arm's val numbers (the headline micro-F1, the
    macro-F1, the nDCG@5, the shortlist R@50, the per-bucket micro-F1), each
    arm's **paired Δ against the control** (the text is the resampling unit;
    2,000 resamples; seed 42 — CS-11's discipline, through
    `tools/compare_upr_arms.py`), the **difference of the differences** (the
    mined arm minus the name+description arm; the same tool, the same texts),
    the lexicon's block (the routes, the weights, the counts, the expansion's
    reach) and the cross-fit's provenance (the folds, the seed). The tool gains
    one **seen-bucket view** (the head's, the torso's and the tail's per-text
    counts summed — the buckets partition the label space, so the union is
    exact) for the test's headline scope.

  - **The falsification, registered in advance.** The **seen-bucket view** (the
    head, the torso and the tail pooled — the labels the channel exists for) is
    the test's scope; the pooled view (which mixes in the eight) is reported
    beside it, and the **unseen** bucket beside that.
    1. the **query half**: `upr-analyze-learned-lexicon-mined` −
       `upr-analyze-learned-lexicon-desc`;
    2. the **feature half**: `upr-analyze-learned-lexicon-minedfeat` −
       `upr-analyze-learned-lexicon-catfeat`.

    The channel **wins** if the paired difference is **positive with an interval
    that excludes zero** on the seen-bucket view **and** on **at least two of
    the three seen buckets** (head, torso, tail). The channel **loses** if
    neither half wins. A mixed result (one half wins, the other does not) is
    reported as **mixed**. If it does not win, the lexicon's value stays
    confined to the readout (CS-31) and the classifier (CS-30) and it **should
    not become a catalog channel** — a null, reported as a null.
    3. the **expansion route's own bar** (the eight, separately): the expansion
       route must raise the **unseen bucket's** micro-F1 against the control
       with an interval that **excludes zero**; else the route is a null for the
       eight, and the eight stay a lexical blind spot (CS-32's finding — an
       empty table marks the held-out labels as impossible — is the risk the
       route has to beat).

  - **The exit criteria.** The label artifact carries the mined-words label
    texts, the lexicon block and the provenance; the two analyze runs of the
    mined channel carry the cross-fitted fit side (the folds and the seed) and
    the lexicon block; the seven table rows are auto-appended; the log section
    reads the numbers out with the falsification conditions; the three CS-11
    re-runs reproduce CS-11's artifacts; the gate green (ruff over
    `src`/`tests`/`experiments`/`tools`, the full pytest suite, the two verify
    tools, and both final sweeps reproducing their artifacts field for field).

  *(pre-registered 2026-11-02; **executed 2026-11-02** — the reviewer's OK on
  the schema change was given on 2026-11-02; the seven runs are
  `upr-labelembed-embeddinggemma-mined`,
  `upr-eval-fusionc-embeddinggemma-k10-c1-mined` and the five
  `upr-analyze-learned-lexicon-{control,desc,catfeat,mined,minedfeat}` (UPR
  **val**, 660 texts, `top_k` k=2, the pack `embedding + lexical +
  chunk_profile`, the gradient-boosted combiner), the sweep is
  `experiments_text_processing/upr_learned_lexicon_sweep.py` (the label
  `upr_learned_lexicon`), the log section is `## CS-35 — the label→word
  lexicon as a label-side channel` in `docs/experiments_text_processing.md`,
  and the numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, 1,312 gold
  cells, `top_k` k=2; the lexicon built from the **train** split only, the fit
  side's label side **cross-fit** over 5 folds).**

  - **The arms** (the learned arm; the control is the same pack with the
    champion's label side):

    | arm | micro-F1 | macro-F1 | nDCG@5 | shortlist R@50 | head | torso | tail | unseen |
    |---|---|---|---|---|---|---|---|---|
    | the **control** | 0.4818 | 0.4652 | 0.6137 | 0.8909 | 0.5505 | 0.4122 | 0.4531 | 0.4469 |
    | the name+description channel's **query** | 0.5084 | 0.4872 | 0.6557 | 0.8864 | 0.5949 | 0.4101 | 0.5121 | 0.4690 |
    | the name+description channel's **features** | 0.5076 | 0.5077 | 0.6643 | 0.8955 | 0.5492 | 0.4559 | 0.5047 | 0.4824 |
    | the mined-words channel's **query** | **0.5152** | 0.5174 | 0.6605 | 0.8838 | 0.6165 | 0.4444 | 0.5031 | 0.3902 |
    | the mined-words channel's **features** | **0.5243** | 0.5265 | **0.6742** | 0.8909 | 0.6234 | 0.4522 | 0.4815 | 0.4591 |

    The three re-runs of CS-11's arms reproduce CS-11's artifacts **field for
    field** (the whole `analysis.json`, bar the `run_id` and the wall-clock
    fields), and each mined run's guard holds.
  - **The lexicon**: the routes are **60 mined / 7 expanded / 1 fallback**
    (`Other`); the mined words are the top 10 by the PIFA weight (the TF-IDF
    centroid's coordinate) — *Access to justice* → `justice access fair trial
    victims ensure system perpetrators right barriers`. The expansion reaches
    seven of the eight held-out labels, and the registered four-character
    prefix rule is looser than a stemmer (`refuse`, `seeking`,
    `relinquishing` are not inflections of the name's words).
  - **Against the control** (the paired bootstrap over the texts; 2,000
    resamples; seed 42): the mined **query** buys **+0.0334 [+0.0145, +0.0531]**
    pooled and **+0.0493 [+0.0284, +0.0710]** on the seen view, and costs the
    unseen bucket **−0.0567 [−0.1011, −0.0109]**; the mined **features** buy
    **+0.0426 [+0.0248, +0.0602]** pooled and **+0.0507 [+0.0312, +0.0708]** on
    the seen view, and the unseen bucket is flat (**+0.0122 [−0.0268,
    +0.0530]**).
  - **The head-to-head** (the mined arm minus the name+description arm): the
    query half's seen view **+0.0217 [−0.0014, +0.0445]** (the interval
    includes zero) and the feature half's **+0.0272 [+0.0044, +0.0502]** — but
    only **one of the three** seen buckets clears in each half (the torso in
    the query half, the head in the feature half), where the rule needs two.
  - **The registered falsification FIRES**: neither half wins, so the mined
    words **do not become a catalog channel** — the lexicon's value stays
    confined to the readout (CS-31) and the classifier (CS-30). The expansion
    route's own bar fires too (the unseen bucket must rise with an interval
    excluding zero: **−0.0567 [−0.1011, −0.0109]** in the query half and
    **+0.0122 [−0.0268, +0.0530]** in the feature half), so **the route is a
    null for the eight**.
  - **The reading**: the mined words are the **best cell-level signal in the
    pack** (the AUC **0.8109**, above the whole-text cosine's 0.7818, the
    label name's coverage's 0.7566 and the description's coverage's 0.7067) —
    the pack simply cannot spend the extra signal, because it already carries
    the name's overlap and the cosines.
  - **The exit criteria.** The label artifact carries the mined-words label
    texts, the lexicon block and the provenance; the two mined analyze runs
    carry the cross-fitted fit side (the folds and the seed) and the lexicon
    block; the seven table rows are auto-appended; the log section reads the
    numbers out with the falsification conditions; the three CS-11 re-runs
    reproduce CS-11's artifacts; the gate green.

---

## The stability map (CS-36) — the pre-registration and the verdict

*Added on 2026-11-02, when the round was executed: the live plan's §7.5
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-36.*

### 7.5 The map's quality round (CS-36 … CS-41)

- [x] **CS-36 — the stability map: which links survive resampling.** The map's
  uncertainty, measured.

  **The mechanism.** Resample the **train** split's texts (B bootstrap
  resamples, seeded), rebuild the corpus map on each, and read every
  (label, word) pair's **stability**: the share of resamples in which the pair is
  present in the label's table and keeps its sign (and, separately, the share in
  which it keeps its place among the label's head words). Then two things: the
  **readout** — the stability of every pair, the share of the map that survives
  at a stated threshold, and the per-label stability — and the **payoff** — the
  map restricted to the stable pairs, through the **existing** scorer, as a
  classifier against the full map.

  **Why.** The map's bulk is fragile (two thirds of its pairs rest on one train
  text) and we have never measured how much of it is sampling noise: CS-31's
  z-score is an *asymptotic* repair, this is the empirical one. And the target
  is known: the **heads are already clean** (CS-35's 600 mined words contain no
  hapax), so what a stability filter can remove is the **tail** the sum arms
  add up — and CS-33 already said the tail is where the trouble is (the
  distinctive-only map is a much worse classifier: 0.0798).

  **The falsification, registered in advance.** (i) The **noise test**: if the
  share of links that survive at the stated threshold is high, the map is not
  noise-dominated and the filter has nothing to remove — the readout is then
  the deliverable and the filter is a null, reported as a null. (ii) The
  **model test**: the filtered map must move the **fit side's** classifier — the
  out-of-fold train micro-F1 (CS-30's instrument) must be **higher** than the
  full map's; if it does not move, the better description is not a better model
  (CS-31's shape). The val number is read beside the fit side's, never as the
  selection.

  **The cost.** About a day. **No new dependency.**

  **The pre-registration (2026-11-02, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **Where the code lives.** The **stability estimator** is pure and lives in
    `text_processors/lexical/stability.py` (numpy and the standard library only
    — no model client, no I/O). The **filter** rides into the **existing**
    scorer (`text_processors/lexical/corpus_map_score.py`: `score_cells` and
    `cross_fit_scores`) as a new argument, and into the **existing**
    `corpus_map` predictor as new parameters (`stability_min`,
    `stability_resamples`, `label_blind`). The **readout** is a new sub-block
    of the map's **own** artifact, written by the same `hitmap-upr` command that
    wrote CS-23's block — the block is **not** renamed or reshaped.

  - **The resampling.** The **train** split's texts (1,676) are resampled
    **B = 500** times, **with replacement** (the ordinary bootstrap: each
    resample draws 1,676 texts uniformly with replacement from the 1,676),
    seeded with the **run's seed (42)**. On each resample the corpus map is
    rebuilt **from scratch** — the same builder, the same tokenizer, the same
    label space. No val text, no val label, no test text enters any resample.
    The resample is a *corpus*, so a text drawn twice counts as two documents
    (the document frequencies see it twice). The resamples are **streamed**: one
    map is alive at a time.

  - **The statistic.** For every **(label, word) pair of the full map** (the
    18,707 pairs), three numbers, each a **share of the 500 resamples**:
    - **`present_share`** — the share of resamples in which the pair is **in
      the label's table** at all (the word occurs in at least one resampled
      text that carries the label);
    - **`sign_stability`** — the share of resamples in which the pair is
      present **and keeps its sign**: the sign of its log-odds is the sign it
      has in the full map. **This is the statistic the filter uses** — a link
      whose sign flips from sample to sample is not a link, it is a coin;
    - **`head_stability`** — the share of resamples in which the pair is
      present **and keeps its place among the label's head words**: it is
      still among the label's **top 10** words by log-odds (**K = 10**, the
      head size — CS-35's top-10 convention, so the number is the same
      convention as the mined lexicon's). This one is **reported, not filtered
      on**: it is the "does the map's own ordering survive" question, and the
      plan asks for it separately.

  - **The filter.** A pair **survives** at a threshold **τ** iff its
    **`sign_stability` ≥ τ**; the dropped pairs are removed from the label's
    table entirely (as if they had never been in it). **A label whose every
    pair is dropped has, by construction, no evidence left**: it is then
    treated exactly like a label with no train example — it carries the row's
    floor and can never be predicted. (The filter's one extra rule, fixed here,
    in advance.)

  - **The thresholds (two, pre-registered).** **τ = 0.75** and **τ = 0.90**.
    The reading: a link that rests on a **single** train text is present in
    about **63 %** of the resamples (1 − e⁻¹) and one that rests on **two** in
    about **86 %**, so τ = 0.75 is the level at which "the link rests on at
    least two train texts" and τ = 0.90 is roughly "at least three". **The
    choice between them is never made on val**: the arm with the higher **fit
    side's out-of-fold** micro-F1 is the selected one (CS-30's instrument), and
    the val numbers of both are reported beside it.

  - **The readout** (the map's own artifact,
    `results/upr-hitmap-corpus/hitmap.json`, a new `stability` sub-block
    inside the existing `corpus_map` block):
    - the knobs (B, the seed, K) and the statistic's own definitions;
    - **every** pair's three shares, beside its `n_with`, its `df` and its
      log-odds, so any threshold can be read off the table;
    - the **survival curve**: the share of the map's pairs that survives at
      every threshold from 0.00 to 1.00 in steps of 0.05 (the readout itself is
      threshold-free; the two registered thresholds are read off it);
    - the survival share **by support** — the pairs resting on 1, 2, 3, … train
      texts — the reading of the fragility the plan states;
    - the **per-label** stability: each label's table size before and after the
      filter at each threshold, its mean `sign_stability`, and the labels that
      lose their whole table.

  - **The payoff.** The **existing** scorer (the arm `sum_zscore_positive` —
    CS-31's fit-side selection, the arm every map arm uses) and the
    **existing** `evaluate` command, on UPR **val** at the same `top_k` k=2
    operating point: the map restricted to the stable pairs, against the full
    map. The filter rides into the **fit side's cross-fit**: each fold's
    filter is estimated on **that fold's own map** (the fold's training texts
    resampled), so the fit side reads the same kind of map the evaluated split
    is scored with.

  - **The runs.** Four `evaluate` runs — all UPR **val**, `top_k` k=2, the arm
    `sum_zscore_positive` — in the sweep
    `experiments_text_processing/corpus_map_stability_sweep.py` (the sweep
    label `corpus_map_stability`, so the rows land in the **main** UPR table's
    `## UPR — corpus_map_stability` section), plus the **readout**: the map's
    own artifact is re-written once (the `corpus_map.stability` block).

    | run | what |
    |---|---|
    | `upr-eval-corpus-map-sum-zscore-positive-stable75-val` | τ = 0.75, the real map |
    | `upr-eval-corpus-map-sum-zscore-positive-stable90-val` | τ = 0.90, the real map |
    | `upr-eval-corpus-map-sum-zscore-positive-stable75-blind-val` | τ = 0.75, the **label-blind** counterpart |
    | `upr-eval-corpus-map-sum-zscore-positive-stable90-blind-val` | τ = 0.90, the label-blind counterpart |

    The label-blind counterpart is the **same map built with the labels
    permuted among the train records** (the per-label counts are preserved,
    only the assignment changes — the track's own label-permutation control,
    as in CS-32).

  - **What is measured.** Each run's **val** micro-F1 (pooled and per bucket)
    and the **paired** difference against the full map (`tools/lexical_compare.py`:
    the same texts, the same operating point, the bootstrap over the texts as
    the resampling unit), and the **fit side's** in-fold and out-of-fold
    micro-F1 (CS-30's instrument). The full map's numbers are the ones already
    in the artifact (`upr-eval-corpus-map-sum-zscore-positive-val`: pooled
    **0.5243**, the fit side's out-of-fold **0.6613**).

  - **The falsification, registered in advance.**
    1. **The noise test.** At the primary threshold (τ = 0.75), the share of
       the map's pairs that survives must be **below 0.90**. If **90 % or more**
       of the map survives, the map is **not** noise-dominated, the filter has
       almost nothing to remove, and the filter is a **null** — the readout is
       then the deliverable, and it is reported as a null.
    2. **The model test.** The **selected** arm's **fit side's out-of-fold**
       micro-F1 must be **strictly higher** than the full map's **0.6613**. If
       it does not move, the better description is not a better model (CS-31's
       shape) and the filter is a **null**, reported as a null. If the fit side
       moves but the **val** paired interval **includes zero**, the reading is
       *not established on val* and the filter does not become the map's
       default. The val number is read **beside** the fit side's, never as the
       selection.
    3. **The control test.** The **label-blind** counterpart must buy
       **nothing**: its paired Δ (filtered − full) must not be positive with an
       interval that excludes zero, and its pooled micro-F1 must stay at the
       chance level. If the blind arm's Δ is positive with an interval that
       excludes zero, the filter's gain is **not label-driven** and the claim
       is withdrawn.

  - **The exit criteria.** The map's artifact carries the stability block
    (every pair's three shares, the survival curve, the per-label stability);
    the four runs hold their numbers and their fit-side blocks; the table rows
    are auto-appended; the log section reads the numbers out with the
    falsification conditions; the paired comparison's tool checks its own
    rebuild against the runs' artifacts (the filter included); the gate green
    (ruff over `src`/`tests`/`experiments`/`tools`, the full pytest suite, the
    two verify tools, and both final sweeps reproducing their artifacts field
    for field).

  - **The cost, in compute.** The readout is 500 map builds (about 40 s); each
    filtered run also estimates the filter on each of the 5 cross-fit folds
    (about 3 min per run), so the four runs are about 15 min of compute. **No
    new dependency.**

  *(pre-registered 2026-11-02; **executed 2026-11-02** — the five runs are
  `upr-eval-corpus-map-sum-zscore-positive-{stable75,stable90,stable75-blind,stable90-blind,blind}-val`
  (UPR **val**, 660 texts, `top_k` k=2, the arm `sum_zscore_positive`), the
  map's own artifact is `results/upr-hitmap-corpus/hitmap.json` (re-written
  with the new `corpus_map.stability` block; every number CS-23 and CS-31
  reported is reproduced unchanged), the sweep is
  `experiments_text_processing/corpus_map_stability_sweep.py` (the label
  `corpus_map_stability`), the paired comparison is `tools/lexical_compare.py`
  (it checks its own rebuild against the runs' artifacts, the filter
  included), the log section is `## CS-36 — the stability map: which links
  survive resampling` in `docs/experiments_text_processing.md`, and the
  numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2,
  the arm `sum_zscore_positive`; the map is built from the **train** split
  only; no model call, no embedding; 500 bootstrap resamples, seed 42).**

  - **The readout** (the map's own block, `corpus_map.stability`; the map's
    18,707 pairs):
    - the **survival curve** has a **cliff** between τ = 0.60 (0.8530 of the
      map) and τ = 0.65 (**0.4168**) — exactly where the arithmetic puts it: a
      link that rests on a **single** train text is present in about
      **1 − e⁻¹ = 63 %** of the resamples;
    - **not one** of the **12,483** single-text links (66.7 % of the map)
      survives τ = 0.75, and only 13 of the 2,798 two-text links survive
      τ = 0.90: at τ = 0.75 **5,168** pairs survive (**0.2763** of the map),
      at τ = 0.90 **2,349** (**0.1256**);
    - the pairs' mean presence share rises with their support exactly as the
      bootstrap predicts (0.6313, 0.8645, 0.9498, 0.9820, 0.9934, 0.9993 for
      1, 2, 3, 4, 5 and 6+ train texts);
    - the **600 head words** (the 60 labelled tables' own top 10) are present in
      **81.1 %** of the resamples and keep their **sign** whenever they are
      present — but they keep their **place among the head** in only **50.2 %**
      of the resamples: the map's own ordering is half a resampling artefact;
    - the per-label stability: the most stable tables are *Death penalty*
      (mean sign 0.731), *National action plan* (0.711) and *Human rights
      defenders* (0.711); the least stable is *Freedom of assembly* (0.622, and
      all **11** of its pairs rest on a single train text) — it loses its table
      at τ ≥ 0.65 and joins the eight held-out labels in the floor.
  - **The falsification, read out.**
    1. **The noise test does NOT fire**: at τ = 0.75 the surviving share is
       **0.2763**, far below the registered 0.90 — the map's bulk *is* noise, in
       the link sense.
    2. **The model test FIRES**: the selected arm's fit side's **out-of-fold**
       micro-F1 must be **higher** than the full map's **0.6613**, and it is
       **lower** — **0.6544** (τ = 0.75) and **0.6473** (τ = 0.90). The
       filtered map is **not** a better model: the filter is a **null**,
       reported as a null. The val number agrees: the paired bootstrap over the
       texts gives **+0.0053 [−0.0015, +0.0128]** (the full map minus the
       τ = 0.75 map) and **+0.0122 [+0.0045, +0.0198]** against τ = 0.90.
    3. **The control test passes**: the label-blind counterpart buys nothing —
       its pooled micro-F1 is at the chance level (**0.0372** without the
       filter, **0.0281** and **0.0160** with it) and the filter's paired
       difference on the blind map is **negative** with an interval that
       excludes zero (**−0.0091 [−0.0187, −0.0008]** and **−0.0213 [−0.0323,
       −0.0100]**).
  - **The arms** (UPR **val**, 660 texts, `top_k` k=2):

    | the map | micro-F1 | macro-F1 | the fit side's **out-of-fold** | head | torso | tail | unseen | shortlist R@50 |
    |---|---|---|---|---|---|---|---|---|
    | the **full** map (CS-31) | **0.5243** | 0.4662 | **0.6613** | 0.6886 | 0.4180 | 0.5435 | 0.0000 | 0.7157 |
    | the filtered map, τ = 0.75 | 0.5190 | 0.4846 | 0.6544 | 0.6948 | 0.3907 | **0.5645** | 0.0000 | 0.7067 |
    | the filtered map, τ = 0.90 | 0.5122 | 0.4856 | 0.6473 | **0.6960** | 0.3714 | **0.5654** | 0.0000 | 0.7062 |
    | the **blind** map (the control) | 0.0372 | 0.0187 | 0.0512 | 0.0847 | 0.0074 | 0.0167 | 0.0000 | 0.5895 |
    | the blind map, τ = 0.75 | 0.0281 | 0.0094 | 0.0398 | 0.0627 | 0.0149 | 0.0000 | 0.0000 | 0.5771 |
    | the blind map, τ = 0.90 | 0.0160 | 0.0027 | 0.0264 | 0.0177 | 0.0197 | 0.0000 | 0.0000 | 0.5687 |

  - **The reading.** The filter trades the **torso** for the **tail** (the
    torso loses 0.0273 and 0.0466 of micro-F1, the tail gains 0.0210 and
    0.0219), and the fit side says the same thing as val — so it is not a val
    artefact. The reason is CS-33's reason, measured again: the map's
    classifier works on **coverage**, not on the strength of its best words.
    The fragile tail — two thirds of the map — contributes *many weak links*
    whose sum is worth something even when each link is a coin; deleting them
    deletes the coverage. **The map's fragility is real and now measured, but
    it is not what the classifier is made of.**
  - **The exit criteria.** The map's artifact carries the stability block
    (every pair's three shares, the survival curve, the per-label stability);
    the five runs hold their numbers and their fit-side blocks; the table rows
    are auto-appended; the log section reads the numbers out with the
    falsification conditions; the paired comparison's tool checks its own
    rebuild against the runs' artifacts (the filter included); the gate green.

---

## The selective map (CS-37) — the pre-registration and the verdict

*Added on 2026-11-02, when the round was executed: the live plan's §7.5
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-37.*

### 7.5 The map's quality round (CS-36 … CS-41)

- [x] **CS-37 — the selective map: "unknown" instead of "impossible".** The
  map's abstention, made honest.

  **The mechanism.** The map's per-cell score has two states today: evidence
  (a number) and no evidence (the **floor** — the row's smallest seen score
  minus one, which ranks the label last in every row). The floor is what makes
  CS-32's pack lose the eight held-out labels outright (the unseen bucket
  0.4867 → **0.0000**): the combiner learns to trust the channel, and the
  channel says "impossible" where it should say "I don't know". The change: a
  label with **no table at all** (and, in the deeper variant, a cell where the
  text says nothing about the label) **abstains** — the cell carries "no
  evidence" instead of "impossible", so the combiner can learn that the channel
  is silent there rather than that the label is out of the question. The
  encodings are alternatives to be pre-registered: a **flag column**
  (combiner-agnostic) or **NaN** (the gradient booster takes NaN natively; the
  logistic combiner does not).

  **The falsification, registered in advance.** (i) The **abstention test**:
  with the abstention, the map's arm must **not** lose the unseen bucket — the
  eight must keep the plain pack's number (about 0.45) instead of collapsing to
  0.0000. If they still collapse, the combiner's *trust* in the channel (not
  the floor) is what costs them, and the abstention is a null. (ii) The **no
  free lunch** test: the seen buckets and the pooled number must not fall — the
  abstention must not cost what it buys. (iii) The **label-blind control** must
  still buy nothing (the track's yardstick).

  **The cost.** About a day. **No new dependency.**

  **The pre-registration (2026-11-02, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **The abstention.** A label with **no table at all** — the eight held-out
    labels — **abstains**: every one of its cells carries "no evidence"
    instead of the floor. The abstention is the **pack's** channel, not the
    classifier's: the harness rejects non-finite scores, so the `corpus_map`
    predictor's floor stays exactly as it is (CS-30's numbers are untouched).

  - **The encoding: NaN.** The abstaining cells carry **NaN** in the pack's
    feature cube — the value is *absent*, not a number that could be misread
    as evidence. The gradient-boosted combiner takes NaN natively
    (`HistGradientBoostingClassifier`); the logistic combiner does not, so the
    abstention arm is registered for the **gradient-boosted** pack alone (the
    pack's arm in every run of this round).

  - **What is NOT registered.** (i) The **flag column** (the
    combiner-agnostic alternative): with the floor left in place the booster
    can still split on it — on the fit side the floor is perfectly predictive
    of "this label is never positive" — so a flag would need a second decision
    (what the score column carries where the channel abstains), and one change
    set fixes one thing. (ii) The **deeper variant** (a cell where the *text*
    says nothing about the label): the map's `sum_zscore_positive` arm's 0.0
    means both "no word of the text is in the label's table" and "only
    negative evidence", so the variant needs its own definition of "says
    nothing" per arm. Both are the reviewer's to open, not this change set's.

  - **Where the code lives.** The abstention is a **scorer** option
    (`text_processors/lexical/corpus_map_score.py`: `score_cells` and
    `cross_fit_scores` gain an `abstain` flag — the empty columns' cells
    become NaN instead of the floor); the command computes both score matrices
    (the floor version and the abstaining one, for the real map and for the
    blind one); the pack fits two more arms (`upr_learning.py`). The config's
    knob is `upr_analysis.learned.corpus_map_feature.abstain` (**off** by
    default, so every earlier run's artifact keeps its exact content).

  - **The guard (unchanged, and not optional).** The fit side's map is
    **cross-fit** over 5 folds (CS-32's rule), and the abstention rides into
    it: a label with no example in a fold's training texts abstains **in that
    fold** (the honest reading — the fold's map really does not know it).

  - **The run.** One: `upr-analyze-cmap-abstain` — the `analyze-upr` command
    with the learned arm's groups `embedding + lexical + chunk_profile +
    corpus_map` (CS-32's configuration), the gradient-boosted combiner, `top_k`
    k=2 on the **val** split. The sweep file
    `experiments_text_processing/upr_learned_cmap_abstain_sweep.py` (the sweep
    label `upr_learned_cmap_abstain`, so the table's section is
    `## UPR — upr_learned_cmap_abstain`); the artifact
    `results/upr-analyze-cmap-abstain/analysis.json`. **Five arms** in the one
    artifact:
    1. `learned_histgb` — the map's column with the floor (CS-32's arm);
    2. `learned_histgb_control` — the pack without the group;
    3. `learned_histgb_label_blind` — the blind map with the floor;
    4. `learned_histgb_abstain` — the map's column with the abstention (the
       new arm);
    5. `learned_histgb_abstain_label_blind` — the blind map with the
       abstention (the new control).

    The pack's own floors (the chance floor, the control-role floor, the
    random-vector floor) ride along unchanged.

  - **The measurement.** The **paired** Δ micro-F1 (the resampling unit the
    text; 2,000 resamples; seed 42) through `tools/compare_upr_arms.py`,
    pooled and per bucket, plus the tool's **seen view** (the head, the torso
    and the tail pooled):
    - the abstention arm − the map's arm (the floor): **the abstention's own
      effect**;
    - the abstention arm − the control: the abstention arm's gain over the
      plain pack;
    - the blind abstention arm − the control: the label-blind control.

  - **The falsification, registered in advance.**
    1. **The abstention test.** The paired difference on the **unseen view**
       (the abstention arm − the map's arm with the floor) must be **positive
       with an interval that excludes zero** — the abstention must move the
       eight — **and** the abstention arm's unseen micro-F1 must be **at least
       0.40** (the plain pack's own number is 0.4867 — CS-32's control — and
       the collapse was 0.0000; the bar sits a little under the plain pack's
       number because the abstention may cost a little of the eight too). If
       the eight still collapse, the combiner's *trust* in the channel (not
       the floor) is what costs them, and the abstention is a null.
    2. **The no free lunch test.** The paired difference on the **seen view**
       (the abstention arm − the map's arm) must have an interval that
       **includes zero** — the abstention must not cost the seen buckets — and
       the **pooled** difference must be **not negative**: the abstention must
       not cost what it buys.
    3. **The label-blind control.** On the **seen view**, the blind abstention
       arm's Δ against the control must be **materially smaller** than the real
       abstention arm's (CS-32's reading: the blind Δ's interval includes zero
       while the real Δ's excludes it). The **unseen** recovery is **not** a
       label-driven gain and is not read as one: the blind map's eight labels
       have empty tables too, so the blind arm gets the same recovery — that is
       the abstention's doing, and the blind arm is what shows it.

  - **The exit criteria.** The artifact holds the five arms with their
    per-bucket numbers and the abstention's provenance (the eight abstaining
    labels, the cross-fit's folds); the table row is auto-appended; the log
    section reads the numbers out with the falsification conditions; **CS-32's
    own run reproduces its artifact field for field** (the abstention is off by
    default, so nothing of CS-32's moves); the gate green (ruff over
    `src`/`tests`/`experiments`/`tools`, the full pytest suite, the two verify
    tools, and both final sweeps reproducing their artifacts field for field).
  *(pre-registered 2026-11-02; **executed 2026-11-02** — the run is
  `upr-analyze-cmap-abstain` (the artifact
  `results/upr-analyze-cmap-abstain/analysis.json`), the sweep is
  `experiments_text_processing/upr_learned_cmap_abstain_sweep.py` (the label
  `upr_learned_cmap_abstain`), the log section is `## CS-37 — the selective map:
  "unknown" instead of "impossible"` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2,
  the gradient-boosted combiner, the pack `embedding + lexical + chunk_profile
  + corpus_map`; the map is built from the **train** split only and the fit
  side's map is **cross-fit** over 5 folds; no model calls; 153 s).**

  - **The arms**:

    | the arm | micro-F1 | macro-F1 | head | torso | tail | unseen | nDCG@5 | shortlist R@50 |
    |---|---|---|---|---|---|---|---|---|
    | the champion (the whole retrieval + decision pipeline) | 0.5479 | 0.5006 | 0.6953 | 0.5121 | 0.5287 | 0.0000 | 0.6725 | 0.7423 |
    | the map's arm (the floor) — CS-32's arm | 0.5502 | 0.5081 | 0.6931 | 0.5088 | 0.5566 | 0.0000 | 0.6694 | 0.7979 |
    | **the abstention arm** | **0.5532** | **0.5173** | 0.6900 | 0.5161 | **0.5761** | **0.0000** | **0.6734** | 0.7972 |
    | the **control** (the pack without the group) | 0.4818 | 0.4652 | 0.5505 | 0.4122 | 0.4531 | **0.4469** | 0.6137 | 0.8909 |
    | the blind map (the floor) | 0.4309 | 0.4277 | 0.5395 | 0.4099 | 0.4665 | 0.0066 | 0.5493 | 0.7741 |
    | the blind map, abstaining | 0.4293 | 0.4354 | 0.5280 | 0.4157 | 0.4832 | 0.0067 | 0.5508 | 0.7808 |
    | the pack's random-vector floor | 0.5357 | 0.4787 | 0.6758 | 0.4885 | 0.5223 | 0.0000 | 0.6506 | 0.7729 |

  - **The falsification, read out.**
    1. **The abstention test FAILS**: the paired difference on the unseen view
       (the abstention arm − the map's arm with the floor) is **+0.0000
       [+0.0000, +0.0000]** — exactly zero in every one of the 2,000
       resamples — and the abstention arm's unseen micro-F1 is **0.0000**, far
       under the registered bar of 0.40. The eight do not come back.
    2. **The no-free-lunch test passes**: the seen view's paired difference is
       **+0.0032 [−0.0037, +0.0100]** (the interval includes zero) and the
       pooled difference is **+0.0030 [−0.0030, +0.0091]** — not negative; and
       no seen bucket's own interval separates (the head −0.0030 [−0.0138,
       +0.0070], the torso +0.0074 [−0.0119, +0.0278], the tail +0.0194
       [−0.0148, +0.0538]).
    3. **The label-blind control passes**: on the seen view the blind
       abstention arm's Δ against the control is **−0.0053 [−0.0189, +0.0091]**
       (the interval includes zero) against the real arm's **+0.1351 [+0.1108,
       +0.1603]**.
  - **The reading: the "impossible" verdict was never in the encoding.**
    Re-encoding the silence as NaN changes nothing for the eight (the decision
    view is exactly 0.0000 apart; the unseen nDCG@5 is 0.6544 against the map's
    arm's 0.6380, a paired +0.0163 [−0.0089, +0.0418] — the eight are not
    ranked terribly, they are simply never in the top 2); the **blind** map —
    which carries no word↔label information at all — collapses them the same
    way (the unseen 0.0067, a paired −0.4402 [−0.4926, −0.3826] against the
    control); and the pack **without** the column keeps them (the control's
    unseen 0.4469). So it is the **presence** of the column on the fit side
    that costs them: on train, the eight held-out labels are exactly the labels
    the map is silent about **and** exactly the labels that are never
    positive, so a learner that uses the column learns "the channel is silent
    ⇒ the label is out of the question" — NaN says "no evidence" faithfully,
    but the fit side answers the question the learner asks. **On a
    label-holdout split, any channel that is silent exactly for the held-out
    labels will be read as "impossible", whatever the encoding.** The map's
    column stays a **seen-labels** channel (CS-32's finding).
  - **The exit criteria.** The artifact holds the five arms with their
    per-bucket numbers and the abstention's provenance (the eight abstaining
    labels, the encoding, the cross-fit's folds); the table row is
    auto-appended; the log section reads the numbers out with the falsification
    conditions; **CS-32's own run reproduces its artifact field for field**;
    the gate green.

---

## The label-pair map (CS-38) — the pre-registration and the verdict

*Added on 2026-11-02, when the round was executed: the live plan's §7.5
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-38.*

### 7.5 The map's quality round (CS-36 … CS-41)

- [x] **CS-38 — the label-pair map: the lexical neighbours, the twins, the
  labels with no vocabulary of their own.** The label space's own geometry.

  **The mechanism.** Read the map's reverse tables (word → labels) as a
  **label-pair** relation: for every pair of labels, how much vocabulary they
  share (the overlap of their word tables) and how often they co-occur in the
  same train text. Then the **readout**: which labels are lexical
  near-duplicates, which pairs are neighbours, and which labels have **no
  vocabulary of their own** (an empty table, or a table that is nearly a subset
  of a neighbour's) — against the catalog's 12 parents, which CS-11 measured to
  carry almost nothing per cell (the AUC 0.4705–0.5433). The comparison is the
  point: the parents are a *type* relation and the lexicon a *vocabulary* one,
  so the comparison says whether **the corpus knows the taxonomy better than
  the taxonomy's writer did**.

  **Why.** It is the structural version of what CS-35 brute-forced (the
  expansion route), and it has a hypothesis worth stating: the eight held-out
  labels include three `- general` ones, and they may be exactly the lexical
  twins with no vocabulary of their own — in which case the label-holdout is a
  *lexical* fact and the expansion route's failure is explained rather than
  merely measured.

  **The falsification, registered in advance.** The relation must be a
  **finding**, not a confirmation: (i) the corpus-derived label-pair relation
  must **not** be the catalog's 12-parent relation — if the two agree on the
  pairs, the corpus knows nothing the taxonomy did not already say; and (ii)
  the "no vocabulary of their own" class must **exist and be read**, and the
  eight held-out labels must be **in** it. If either fails, the readout is
  reported as a null (the relation is the taxonomy's, or the vocabulary-less
  class is empty or elsewhere) and no action is licensed.

  **The cost.** About a day. **No new dependency.**

  **The pre-registration (2026-11-02, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **Where the code lives.** The **relation** is pure and lives in a new
    module `text_processors/lexical/label_pairs.py` (the standard library only
    — no model client, no I/O); the **readout** is a new sub-block of the
    map's **own** artifact (the existing block is neither renamed nor
    reshaped), written by the same `hitmap-upr` command that wrote CS-23's
    block. The **catalog** is read at the edge (the committed
    `catalogs/upr_labels.jsonl` ✓).

  - **The relation** — for every pair of the 68 labels, two halves:
    - the **vocabulary half**: the shared words' count, the **Jaccard**
      (`|T₁ ∩ T₂| / |T₁ ∪ T₂|`) and the **containment** (the share of the
      *smaller* table that lies inside the larger one);
    - the **co-occurrence half**: the number of **train** texts carrying both
      labels, the **lift** over independence (the observed count over the
      expected one) and the Jaccard of the two labels' text sets.

  - **The readout** (the map's own artifact, `corpus_map.label_pairs`):
    - the **near-duplicates**: the pairs whose vocabulary **Jaccard ≥ 0.50**
      (fixed in advance), ranked by the Jaccard;
    - the **neighbours**: the **20** pairs with the strongest **co-occurrence
      lift** (fixed in advance);
    - the **"no vocabulary of their own"** class: the labels with an **empty
      table** (the eight held-out ones, by construction) ∪ the labels whose
      table lies **≥ 0.80 inside a single partner's table** (the containment
      threshold, fixed in advance — the plan's "nearly a subset of a
      neighbour's");
    - the **per-label** summary: the table size, the best partner (by the
      Jaccard and by the lift), the containment, the number of near-duplicate
      partners and the number of co-occurring partners.

  - **The catalog's comparison — the *sibling* relation.** The catalog's 12
    parents are **types, not labels** (none of the 68 entries is a parent), so
    the measurable form of the taxonomy's own claim about the label space is
    the **sibling** relation: the pairs of labels that **share a parent**
    (**227** pairs of the 2,278). The readout reports the siblings' and the
    other pairs' mean Jaccard, containment and lift, the **ranking
    agreement** (how many of the corpus's 20 strongest pairs by the Jaccard are
    sibling pairs — the chance rate is 227/2,278 ≈ 10 %), and the **separation
    with a bootstrap over the train texts** (the resampling unit the text;
    2,000 resamples; seed 42; the map is rebuilt on each resample, so the
    interval is over the corpus itself).

  - **One reading settled before the first call.** The block above says the
    eight held-out labels "include three `- general` ones". The map's own
    measurement (CS-23) says **two**: `ESC rights - general` and `General` —
    `CP rights - general` **has** a table. The hypothesis is untouched (the
    held-out labels may be the ones with no vocabulary of their own); only the
    count in the block's premise is corrected here.

  - **The falsification, registered in advance.**
    1. **The disagreement test** (the relation must be a finding, not a
       confirmation): (a) the bootstrap CI of the sibling pairs' mean vocabulary
       Jaccard **minus** the other pairs' mean must **include zero** (the
       corpus does not rank the taxonomy's own pairs higher than chance), and
       (b) the corpus's 20 strongest pairs by the Jaccard must contain **at
       most 4** sibling pairs (the chance expectation is about 2). If either
       fails, the corpus's relation **is** the taxonomy's: the readout is a
       **null**, no action is licensed, and it is reported as a null.
    2. **The vocabulary-less class test**: the class must **exist** and be
       read, and the eight held-out labels must be **in** it (they are, by
       their empty tables) — and the class's **extra** members (the labels
       whose table lies ≥ 0.80 inside one partner's) must be **read** (the
       readout lists them with their partner and their share). If the class is
       empty of extra members and the eight, the readout is a **null** (the
       vocabulary-less class is empty or elsewhere) and no action is licensed.

  - **The run.** The map's own artifact, re-written once:
    `results/upr-hitmap-corpus/hitmap.json` (the new `corpus_map.label_pairs`
    block) — the sweep
    `experiments_text_processing/upr_hitmap_corpus_sweep.py` (the sweep label
    `upr_hitmap_corpus`), so the row lands in the **track's** UPR table under
    `## UPR — upr_hitmap_corpus`. No model call, no embedding: the relation is
    a description of the corpus.

  - **The exit criteria.** The map's artifact carries the label-pair block
    (every pair's two halves, the near-duplicates, the neighbours, the
    vocabulary-less class, the per-label summary and the catalog's comparison
    with its bootstrap); every number CS-23, CS-31, CS-33, CS-34 and CS-36
    reported is **reproduced unchanged** (the block grew, nothing else moved);
    the table row is auto-appended; the log section reads the numbers out with
    the falsification conditions; the gate green (ruff over
    `src`/`tests`/`experiments`/`tools`, the full pytest suite, the two verify
    tools, and both final sweeps reproducing their artifacts field for field).

  - **The cost, in compute.** The relation is 2,278 pairs over the train
    split's 1,676 texts; the bootstrap (2,000 map rebuilds) is about four
    minutes. **No new dependency.**
  *(pre-registered 2026-11-02; **executed 2026-11-02** — the run is the map's
  own artifact, re-written once (`results/upr-hitmap-corpus/hitmap.json`, the
  new `corpus_map.label_pairs` block), the sweep is
  `experiments_text_processing/upr_hitmap_corpus_sweep.py` (the label
  `upr_hitmap_corpus`), the log section is `## CS-38 — the label-pair map: the
  lexical neighbours, the twins, the labels with no vocabulary of their own` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (the train split: 1,676 texts, 68
  labels, 60 with a table; 2,278 label pairs; no model call, no embedding;
  266 s).**

  - **The readout** (the map's own block, `corpus_map.label_pairs`):
    - the **sibling** pairs (the catalog's own claim, read as the pairs of
      labels that share a parent — the 12 parents are types, not labels):
      **227** of the 2,278, a chance rate of **9.96 %**;
    - the mean vocabulary Jaccard / containment / lift over all the pairs:
      0.1162 / 0.2906 / 0.5276;
    - the **near-duplicates** (the vocabulary Jaccard ≥ 0.50, fixed in
      advance): **none** — the closest pair is *Rights of the Child* /
      *Women's rights* at **0.489** (694 shared words; the bigger table keeps
      431 of its own);
    - the **neighbours** (the 20 strongest co-occurrence lifts): *Freedom of
      opinion and expression / Freedom of the press* (lift **18.6**, 18 shared
      texts), *Civil society / Human rights defenders* (16.5), *Racial
      discrimination / Violent extremism and human rights* (15.8), *Indigenous
      peoples / Right to land* (15.7), *Enforced disappearances / Extrajudicial
      executions* (15.7), *Right to health / Sexual and reproductive rights*
      (14.6) — **10 of the 20 are sibling pairs**;
    - the **"no vocabulary of their own"** class: **13** labels — the **eight
      held-out** ones (their tables are empty) and **five** more whose own
      words lie almost entirely inside one neighbour's table:
      *Sexual and reproductive rights* (0.935 inside *Women's rights*),
      *Freedom of assembly* (0.909 inside *Detention*), *Statelessness and the
      right to nationality* (0.864 inside *Rights of the Child*), *CP rights -
      general* (0.843 inside *International instruments*) and *Trafficking*
      (0.808 inside *Rights of the Child*).
  - **The falsification, read out.**
    1. **The disagreement test FAILS on both legs.** (a) The **separation**
       (the sibling pairs' mean vocabulary Jaccard minus the other pairs') is
       **+0.0413 [+0.0372, +0.0451]** (the bootstrap over the train texts;
       2,000 resamples; seed 42): the interval **excludes zero**, where the
       registered test wanted it to include it. (b) The corpus's **20 closest
       pairs** contain **11** sibling pairs, where the registered bar was at
       most 4 (the chance expectation is about 2). The co-occurrence agrees as
       loudly: the siblings' mean lift is **1.62** against the other pairs'
       **0.41**. So **the corpus knows nothing the taxonomy did not already
       say** — the readout is a **null** for the relation's claim and no action
       is licensed.
    2. **The vocabulary-less class test PASSES.** The class exists (13
       members), it is read, and the eight held-out labels are in it — the
       label-holdout is a **lexical** fact. The plan's premise about "three
       `- general`" labels is settled by the map's own measurement: **two** of
       the eight are (`ESC rights - general`, `General`; `CP rights - general`
       has a table) — but the third one *is* in the vocabulary-less class, with
       84 % of its words inside *International instruments*.
  - **The reading.** There are **no lexical twins** in the corpus's
    vocabulary: not one of the 2,278 pairs reaches the registered threshold,
    and the biggest table shares 62 % of its own words with one partner and
    keeps the rest. The map's vocabulary is broad and flat, not a set of
    duplicates. And the label space's geometry is a **rediscovery** of the
    taxonomy's: the families the catalog's writer drew are the families the
    corpus's words draw. Where the corpus says something the taxonomy does not
    is the vocabulary-less class — the labels whose own lexical identity is
    thin, the ones a lexical channel would have the least to say about.
  - **The exit criteria.** The map's artifact carries the label-pair block
    (every pair's two halves, the closest pairs, the near-duplicates, the
    neighbours, the vocabulary-less class, the per-label summary and the
    catalog's comparison with its bootstrap); every number CS-23, CS-31,
    CS-33, CS-34 and CS-36 reported is reproduced unchanged (the block grew,
    nothing else moved); the table row is auto-appended; the log section reads
    the numbers out with the falsification conditions; the gate green.

---

## The phrase map (CS-39) — the pre-registration and the verdict

*Added on 2026-11-02, when the round was executed: the live plan's §7.5
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-39.*

### 7.5 The map's quality round (CS-36 … CS-41)

- [x] **CS-39 — the phrase map: do phrases beat their own words?** The map's
  unit, one step finer.

  **The mechanism.** Build a **phrase table** — the (label, bigram) pairs of the
  train corpus, over the track's own content tokens, so the phrases are the
  corpus's own units (the topical ones: "human rights", "death penalty", "early
  marriage"; and the genre's formulas: "continue to", "ensure that"). Then three
  things: the **readout** — the class split over the phrase table (CS-33's own
  machinery applied to phrases), i.e. **how much of the phrase table is
  formulaic and how much is topical**; the **pair interaction** — for each
  pair, how much more the phrase says than its two words say
  (`logodds(pair) − [logodds(w₁) + logodds(w₂)]` in the label's table), which
  names the phrases that carry more than their parts; and the **payoff** — the
  **union** of the word map and the phrase map (the same scorer over both
  tables) against the word map alone, paired, with the label-blind control.

  **Why.** The map's unit is a single word and UPR's genre is formulaic, so the
  phrase table is where the genre's structure would show. The prior from the
  round: the topical phrases are **rare**, so a phrase map *alone* inherits the
  hapax problem (CS-33's distinctive-only map: 0.0798) — hence the **union** is
  the arm, not the phrase map alone.

  **The falsification, registered in advance.** (i) The **formulaic test**: the
  phrase table's generic share must be materially above the word table's (the
  phrases are mostly the register) — if the phrase table's classes look like the
  word table's, the unit change buys nothing. (ii) The **model test**: the
  union's **fit side's** out-of-fold train micro-F1 must be **higher** than the
  word map's — if it does not move, the phrases say nothing their words do not
  say, and the unit change is a null.

  **The cost.** About a day and a half. **No new dependency.** The existing
  map's block is not renamed or reshaped: the phrase table is a **separate**
  builder.

  **The pre-registration (2026-11-02, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **One reading settled before the first call.** The block says the phrases
    are the bigrams "over the track's own content tokens" and then names
    "continue to" and "ensure that" as the genre's formulas — but the track's
    stopword list holds **"to"** and **"that"**, so those two are **not**
    bigrams over the content tokens. The block's *purpose* ("where the genre's
    structure would show") needs the formulas, so the phrase unit is fixed
    here as: **the consecutive bigrams of the track's own token stream**
    (`tokenize` — the lower-cased alphanumeric tokens, the stopwords **kept**),
    with the track's own tokenization ✓ (the same unit the map's words come
    from).

  - **The phrase table** (a **separate builder** ✓ — the word map's block is
    neither renamed nor reshaped): the (label, phrase) pairs of the train
    corpus, with the **same measures** as the word map (the log-odds, the
    z-score, the counts ✓ — the estimator's 2×2 table over the phrase's
    presence in a text ✓).

  - **The readout** (the map's own artifact, a **new** `phrase_map` block
    beside the corpus map's ✓):
    - the phrase table's **shape**: the distinct phrases, the (label, phrase)
      pairs, and the **hapax share** (the pairs resting on a single train
      text ✓ — the round's fragility, one unit finer ✓);
    - the **class split** over the phrase table (CS-33's own machinery ✓): the
      pairs per class, the generic share and the mean log-odds — beside the
      word table's own numbers ✓;
    - the **pair interaction**: for every (label, phrase) pair, the phrase's
      log-odds **minus** the sum of its two words' log-odds in the same label's
      table ✓ (how much more the phrase says than its parts ✓) — its mean, its
      distribution and the **20 pairs that carry the most** ✓.

  - **The payoff — the union.** The **same scorer** over **both** tables: a
    cell's score is the word map's `sum_zscore_positive` **plus** the phrase
    map's own `sum_zscore_positive` (the phrase arm's units are the text's
    bigrams ✓). The union is the arm, **not** the phrase map alone (the
    topical phrases are rare ✓ — the round's prior ✓). The fit side is
    **cross-fit** over 5 folds ✓ (CS-32/CS-30's rule ✓): each fold's **phrase**
    map is built from the fold's training texts ✓, so the fit side reads the
    same kind of map the evaluated split is scored with.

  - **The runs.** Two `evaluate` runs on UPR **val** at the champion's own
    `top_k` k=2 (the `corpus_map` predictor, the new arm):
    `upr-eval-corpus-map-sum-zscore-positive-phrase-val` (the union) and
    `upr-eval-corpus-map-sum-zscore-positive-phrase-blind-val` (the union on
    the **label-blind** maps — the per-label counts preserved, only the
    assignment changed ✓). The sweep file
    `experiments_text_processing/corpus_map_phrase_sweep.py` (the sweep label
    `corpus_map_phrase`); the rows land in the **main** UPR table. The
    **readout**: the map's own artifact, re-written once with the new
    `phrase_map` block.

  - **The falsification, registered in advance.**
    1. **The formulaic test.** The phrase table's **generic share** (the
       generic class's share of the phrase table's pairs ✓) must be **at least
       10 points above the word table's own** (**43.6 %** ✓ — CS-33's
       readout ✓): the phrases are mostly the register. If it is not, the
       phrase table's classes look like the word table's, the unit change buys
       nothing, and the readout is reported as a **null**.
    2. **The model test.** The union's **fit side's out-of-fold** train
       micro-F1 must be **strictly higher** than the word map's (**0.6613** ✓
       — the same instrument CS-30/CS-31/CS-36 read). If it does not move, the
       phrases say nothing their words do not say and the unit change is a
       **null**, reported as a null. The val number is read **beside** the fit
       side's, never as the selection.
    3. **The label-blind control.** The blind union's gain over the word map's
       own blind arm must be materially smaller than the real union's ✓ — the
       round's discipline ✓ (CS-32's reading ✓).

  - **The exit criteria.** The map's artifact carries the `phrase_map` block
    (the shape, the class split, the pair interaction); the two runs hold their
    numbers and their fit-side blocks; the table rows are auto-appended; the
    log section reads the numbers out with the falsification conditions; the
    paired comparison's tool checks its own rebuild against the runs' artifacts
    (the union included ✓); every number CS-23, CS-31, CS-33, CS-34, CS-36
    and CS-38 reported is **reproduced unchanged**; the gate green (ruff over
    `src`/`tests`/`experiments`/`tools`, the full pytest suite, the two verify
    tools, and both final sweeps reproducing their artifacts field for field).

  - **The cost, in compute.** The phrase table is 1,676 texts' bigrams; the
    union's fit side rebuilds the phrase map 5 times. **No new dependency.**
  *(pre-registered 2026-11-02; **executed 2026-11-02* — the two runs are
  `upr-eval-corpus-map-sum-zscore-positive-phrase{-blind}-val` (UPR **val**,
  660 texts, `top_k` k=2, the arm `sum_zscore_positive_phrase`), the sweep is
  `experiments_text_processing/corpus_map_phrase_sweep.py` (the label
  `corpus_map_phrase`), the map's own artifact is re-written once with the new
  `phrase_map` block, the log section is `## CS-39 — the phrase map: do phrases
  beat their own words?` in `docs/experiments_text_processing.md`, and the
  numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2;
  the train split's 1,676 texts and 68 labels; no model call, no embedding;
  the two runs 3.1 s and 3.6 s, the map's artifact 256 s).**

  - **The phrase table's shape**: 14,167 phrases and 45,587 (label, phrase)
    pairs, of which **82.8 %** rest on a single train text (the word table's:
    3,049 words, 18,707 pairs, 66.7 %).
  - **The class split** (CS-33's machinery over phrases): the distinctive
    class 12,193 pairs (26.7 %), the shared class 26,331 (57.8 %) and the
    **generic** class 7,063 (**15.5 %**) — against the word table's 9.8 % /
    46.6 % / **43.6 %**.
  - **The pair interaction** (the phrase's log-odds minus its two words'): the
    mean **+0.6679**, the median **+0.7077**, 68.7 % of the pairs positive;
    the top interactions are the genre's fragments — *Freedom of assembly* ←
    **"on its"** and **"with an"**, *Right to life* ← **"and will"**, **"or
    when"**, **"under a"**, *Human rights defenders* ← **"out their"**,
    *Transitional justice* ← **"measures of"**.
  - **The arms**:

    | the arm | micro-F1 | macro-F1 | the fit side's **out-of-fold** | head | torso | tail | unseen |
    |---|---|---|---|---|---|---|---|
    | the **word map** (CS-31's arm) | 0.5243 | 0.4662 | 0.6613 | 0.6886 | 0.4180 | 0.5435 | 0.0000 |
    | the **union** | **0.5342** | 0.4663 | **0.6663** | 0.6644 | **0.5056** | 0.4943 | 0.0000 |
    | the blind word map (the control) | 0.0372 | 0.0187 | 0.0512 | — | — | — | — |
    | the blind union (the control) | 0.0456 | 0.0186 | 0.0752 | — | — | — | — |

    The paired difference (the text is the resampling unit; 2,000 resamples;
    seed 42): the union minus the word map **+0.0099 [−0.0008, +0.0214]**; on
    the blind maps **+0.0084 [−0.0023, +0.0202]** (both intervals include
    zero ✓).

  - **The falsification, read out.**
    1. **The formulaic test FAILS**: the phrase table's generic share is
       **15.5 %**, far below the word table's **43.6 %** (the registered test
       wanted it at least ten points above). The reason is the table's own
       shape — **82.8 %** of the phrase pairs rest on one train text, and a rare
       unit cannot sit in many tables, so the class measure (a unit's
       *breadth*) reads the phrases' rarity as topicality.
    2. **The model test PASSES mechanically**: the union's fit side's out-of-fold
       micro-F1 is **0.6663** against the word map's **0.6613** — strictly
       higher ✓ — and the val number agrees in sign but not in significance
       (the paired interval includes zero ✓).
    3. **The label-blind control FAILS**: on the fit side's out-of-fold the
       blind union gains **+0.0240** (0.0512 → 0.0752) where the real union
       gains **+0.0050** (0.6613 → 0.6663) — the phrase table's contribution
       is **not label-driven** ✓ (a map with no word↔label information at all
       gains more from the phrases than the real one does).
  - **The reading.** The register is real but it is a **fragment**: the phrases
    with the largest interaction are the ones that straddle a stopword ("on
    its", "with an", "and will"), the corpus's formulas — which say nothing
    about the topic and a lot about the genre. The class split cannot see them
    (a fragment is rare, so it reads distinctive). And the phrases are **too
    rare to be a table** (82.8 % hapax): the union's gain is not established on
    val and is *larger* on the blind maps, so what the phrases add is noise.
    What the union does move is the torso (0.4180 → 0.5056) and what it loses
    is the head (0.6886 → 0.6644) and the tail (0.5435 → 0.4943) — the trade
    a noisier channel makes.
  - **The exit criteria.** The map's artifact carries the `phrase_map` block
    (the shape, the class split, the pair interaction); the two runs hold their
    numbers and their fit-side blocks; the table rows are auto-appended; the
    log section reads the numbers out with the falsification conditions; the
    paired comparison's tool checks its own rebuild against the runs' artifacts
    (the union included ✓); every number CS-23, CS-31, CS-33, CS-34, CS-36 and
    CS-38 reported is reproduced unchanged; the gate green.

---

## The contrastive map (CS-40) — the pre-registration and the verdict

*Added on 2026-11-03, when the round was executed: the live plan's §7.5
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-40.*

### 7.5 The map's quality round (CS-36 … CS-41)

- [x] **CS-40 — the contrastive map: "why A and not B".** The map's explanation
  of itself, in the viewer.

  **The mechanism.** A **viewer's** block: for one text and two candidate
  labels, the words that discriminate them — the words in A's table with a
  positive log-odds and in B's with a negative one (or the largest log-odds
  difference between the two tables), with the numbers beside them. The map's
  artifact already holds every (label, word) pair, so this is a **read**, not a
  new corpus pass; the block names the question it answers (CS-25's rule).

  **The falsification, registered in advance.** A tool change must move **no
  number** (the existing artifacts and every run stay exactly as they are), and
  the block must answer a question the viewer's other blocks cannot: if the
  discriminating words are the ones the topic-side and the text-side maps
  already show, the block adds nothing and is dropped.

  **The cost.** Half a day. **No new dependency.**

  **The pre-registration (2026-11-03, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **Where the code lives.** The **contrast** is pure and lives in a new
    module `text_processors/lexical/contrastive.py` (the standard library only
    — no model client, no I/O): it reads **two tables** in the artifact's own
    row shape and answers the pair's question. The **block** is in the
    **viewer** (`tools/hitmap_view.py`), which reads the map's artifact and the
    data. The **measurement** is a new tool `tools/contrastive_map_check.py`.
    **No new run and no table row**: this is a **tool** change (the CS-25
    shape), and **no existing artifact is re-written** — the map's own
    artifact keeps its exact bytes.

  - **The contrast — the exact definition.** For a pair of labels (A, B) and a
    word w:
    - `lo_A(w)` — A's log-odds for w: the **stored** value when the pair
      (A, w) is in A's table, and **0** when it is not — the map's own
      **silence convention** (a word the table does not hold contributes
      nothing to that label, exactly as the map's scorer counts it);
    - `lo_B(w)` — the same for B;
    - `contrast(w) = lo_A(w) − lo_B(w)`: above 0 the word pulls A up relative
      to B, below 0 it pulls B up relative to A, 0 it says nothing about the
      pair;
    - the **discriminating words** of the pair are the words of the **union**
      of the two tables, ranked by `|contrast|`.
    - **The flags** each row carries: `in_a` / `in_b` (which table holds the
      word), `in_text` (the word is one of *this* text's content words), and
      **`sign_disagreement`** — the word is in **both** tables and its two
      log-odds have **opposite signs** (the block's first reading: the words
      in A's table with a positive log-odds and in B's with a negative one).

  - **One reading settled before the first call.** The block offers two
    readings — the two-sided set ("in A's table with a positive log-odds and
    in B's with a negative one") and the difference ("the largest log-odds
    difference between the two tables"). The **difference is the measure** (it
    is defined for **every** word of the union; the two-sided set is a subset
    of it), and the two-sided set rides along as the `sign_disagreement` flag.
    The difference is read with the map's **silence convention** (a missing
    pair contributes 0), so a word only one table holds contributes that
    table's own log-odds against the other side's 0 — and the row says which
    table holds it, so the reader never has to guess.

  - **The block (the shape on the screen).** Two halves, both read from the
    artifact's own rows (no rebuild):
    1. **the words of THIS text that separate them** — the text's content
       words that are in either table, with `lo_A`, `lo_B` and the difference
       (the actionable half);
    2. **the corpus's own words that separate them** — the **top 10 in each
       direction** of `|contrast|` over the union of the two tables (the map's
       own head size — CS-36's `HEAD_K = 10`), with the counts beside the
       numbers.

    The block names the question it answers (CS-25's rule): *"WHY A AND NOT B
    — a contrastive question: which words separate these two labels in the
    corpus?"*, and it says what it is: *a description of the data (the train
    split's own counts) read as a difference between two tables — never the
    model's reading*.

  - **The pair.** The block's **A** is the viewer's own **focus** (the gold
    label whose own name words the text carries the most — the viewer's
    existing rule, unchanged). Its **B** is the **map's own runner-up among
    the labels the text does not carry**: the non-gold label with the largest
    `sum_zscore_positive` score (the map's own arm, CS-31's — read from the
    artifact's tables, the z-scores included), among the labels whose table is
    not empty; ties → the label space's own order. When no non-gold label has
    a positive score the pair is **empty** and the block says so. A
    `--contrast <label>` flag overrides B (any label of the space).

  - **The measurement.** A new tool, `tools/contrastive_map_check.py`, over the
    UPR **val** split's 660 texts. It reads the map's artifact and the split
    (no model call, no embedding) and writes **its own** artifact
    `results/upr-hitmap-contrastive/contrastive.json` (a **tool's** readout,
    not a CLI run's: it writes nothing else and touches no other artifact).
    Two kinds of pairs, both fixed in advance:
    - the **alternative** pair: A = the focus, B = the map's runner-up among
      the labels the text does not carry (as above);
    - the **sibling** pair: A = the focus, B = the text's **other** gold label
      with the largest map score (the case where the screen already shows both
      tables — the harder case for the novelty measure).

    For each pair, the block's rows (the two halves, deduplicated by word) and
    four numbers:
    1. **the vocabulary share** — the share of the rows that are **neither a
       name word of A or B nor one of the text's content words**: the corpus's
       own vocabulary, which the **topic-side** and the **text-side** maps
       (the name-overlap features) structurally **cannot** show. **Bar: the
       mean over the measured pairs ≥ 0.50.**
    2. **the relevance share** — the share of the pairs with at least one
       **text word** among the rows: the block must be about *this* text, not
       about the corpus in general. **Bar: ≥ 0.50.**
    3. **the novelty share** — the share of the rows for which the screen (as
       it is now) does **not** print **both** of the pair's log-odds: the
       reader could not have formed the difference. "The screen prints the
       log-odds of (w, L)" means: L is the focus or one of the text's gold
       labels and w is among the **top-k** rows of L's displayed table
       (k = the viewer's `--top-k`, default 5) or w is one of the text's words
       in L's table; **or** w is one of the text's words and L is among the
       **top-3** labels of w's reverse table. **Bar: the mean over the
       measured pairs ≥ 0.50, and the share of the pairs with at least one
       such row ≥ 0.50.**
    4. **the sign-disagreement share** (reported, **no bar**): the share of the
       pairs with at least one **text word** that is in **both** tables with
       **opposite signs** — the plan's first reading, on the text's own words.

  - **The falsification, registered in advance.**
    1. **The no-moved-number test.** A tool change must move **no number**:
       every existing artifact keeps its exact bytes (the map's own artifact
       included — its sha256 is taken before and after the change set) and
       every run's numbers reproduce. The change set writes **no artifact**
       beyond the measurement tool's own readout, appends **no table row**, and
       adds **no dependency**.
    2. **The novelty test.** The block must answer a question the viewer's
       other blocks cannot. If the **vocabulary share** (bar 1) or the
       **relevance share** (bar 2) or the **novelty share** (bar 3) falls
       below its registered bar, the discriminating words are the ones the
       topic-side and the text-side maps already show, the block adds nothing
       and is **dropped** (the module and the tool stay; the block comes off
       the screen).
    3. The **label-blind control does not apply here**: this change set
       measures **no model** and **no classifier** — the falsification is the
       screen's own coverage, not a number a model produced. (Stated so the
       round's discipline is not read as a gap.)

  - **What is NOT registered.** (i) The **phrase-level** contrast: the phrase
    table is not in the artifact (CS-39 stored its shape, its classes and its
    interaction, not its tables), so a phrase contrast would need a **rebuild**
    — a corpus pass, which the block's mechanism excludes — and CS-39's
    verdict is that the phrases are noise anyway. (ii) The **model's** own
    top-2 as the pair: the block is the **map's** explanation, not the model's
    ranking (the model's ranking is the viewer's own `--live` block). (iii)
    The **pair's stability**: CS-36's resampling is not run over the contrast
    (not registered). (iv) The catalog's **siblings** (CS-38's relation) as
    the pair: the pair is the text's own, not the taxonomy's.

  - **The exit criteria.** The viewer's block is on the screen with the
    numbers beside the words, it names the question it answers, and its numbers
    are the artifact's own (the viewer's own guard checks them against the
    artifact); the measurement tool writes its readout and prints the verdict
    against the registered bars; every existing artifact's sha256 is unchanged
    and every run's numbers reproduce; the log section reads the numbers out
    with the falsification conditions; the gate green (ruff over
    `src`/`tests`/`experiments`/`tools` and the track's sweep folder, the full
    pytest suite, the two verify tools, and both final sweeps reproducing their
    artifacts field for field).

  - **The cost, in compute.** The contrast is a read of the artifact (the
    viewer's own corpus-map rebuild — the stale-artifact guard — already
    dominates the run); the measurement over the 660 val texts is a table
    lookup per text. **No new dependency.**
  *(pre-registered 2026-11-03; **executed 2026-11-03** — the change set is a
  **tool** change: the viewer's contrastive block, the pure module behind it
  (`text_processors/lexical/contrastive.py`) and the measurement tool
  (`tools/contrastive_map_check.py`); **no run**, **no table row**, **no
  artifact re-written**, **no new dependency**. The measurement's own readout
  is `results/upr-hitmap-contrastive/contrastive.json` (a **tool's** artifact,
  not a CLI run's). The log section is `## CS-40 — the contrastive map: "why A
  and not B"` in `docs/experiments_text_processing.md`, and the numbers are
  below.)*


  **The round's numbers, in one screen (the UPR **val** split: 660 texts, the
  train split's 1,676 texts and 68 labels; no model call, no embedding; the
  measurement 3.1 s, the viewer 1.6 s).**

  - **The measurement** (the tool's own artifact,
    `results/upr-hitmap-contrastive/contrastive.json`; the map's own artifact
    is **read**, not re-written — its sha256 `ba2d7037…` is the same before and
    after ✓):

    | the measure | the alternative pairs | the sibling pairs | the pooled | the bar |
    |---|---|---|---|---|
    | the pairs | 534 | 298 | 832 | — |
    | the **vocabulary share** | **0.6191** | 0.5939 | **0.6100** | ≥ 0.50 ✓ |
    | the **relevance share** | 1.0000 | 1.0000 | 1.0000 | ≥ 0.50 ✓ |
    | the **novelty share** | **0.8305** | 0.7563 | **0.8039** | ≥ 0.50 ✓ |
    | the pairs with at least one new row | 1.0000 | 1.0000 | 1.0000 | ≥ 0.50 ✓ |
    | the **sign-disagreement share** (no bar) | 0.5955 | 0.6779 | **0.6250** | — |

    (The mean block per pair: 29.9 rows.)
  - **The falsification, read out.**
    1. **The no-moved-number test HOLDS**: a **tool** change — no run, no table
       row, no new dependency and **no artifact re-written**. The map's own
       artifact is byte-identical (its sha256 before and after ✓); the only
       files under `results/` that changed are the **nine** the gate's own
       final-sweep re-runs rewrote, and those differ only in their wall-clock
       fields ✓ (their content is IDENTICAL field for field ✓).
    2. **The novelty test HOLDS** — all three registered bars are met, on both
       kinds of pairs and pooled ✓: the discriminating words are **not** the
       ones the topic-side and the text-side maps already show (the vocabulary
       share **0.6100** against the bar 0.50 ✓), the block is about **this**
       text (every measured pair has a text word among its rows ✓), and for
       **80 %** of the rows the screen does not print **both** of the pair's
       log-odds ✓ — the reader could not have formed the difference. **The
       block stays.**
    3. The **label-blind control does not apply here** ✓ (as registered ✓): no
       model and no classifier are measured — the falsification is the screen's
       own coverage ✓.
  - **The reading.** The block answers a question the screen could not: the
    screen shows each label's table **separately** and only for the text's
    **gold** labels ✓, while the contrast reads two tables **against each
    other** ✓, for any pair ✓ — and for the **tempting alternative** (the map's
    own runner-up among the labels the text does *not* carry ✓) the screen
    showed **nothing at all** about the second label before this block ✓. The
    sharp case is the common one: **62.5 %** of the measured pairs have a word
    of **this** text that is in **both** tables with **opposite** signs ✓ — the
    plan's first reading ✓. An example (the viewer's own seed-7 draw): the
    focus is *Right to education* and the map's runner-up is *Right to
    development*; **education** is **+5.317** in the first table and **−0.097**
    in the second (the difference **+5.415**, the signs disagree ✓), **socio**
    is **0.000** against **+5.437** (the difference **−5.437**), and the
    corpus's own strongest discriminators are *dropout* (+5.461, the first
    table only) and *gdp* / *road* / *vision* (−5.437, the second table only).
    The **caveat the reader can see**: the corpus half's strongest rows are the
    map's **fragile tail** ✓ (CS-36's finding ✓) — the rare words with the
    largest log-odds — and their counts sit beside the numbers ✓ (`n_with
    A/B`).
  - **The exit criteria.** The viewer's block is on the screen with the
    numbers beside the words ✓, it names the question it answers ✓ (CS-25's
    rule ✓), and its numbers are the artifact's own ✓ (the viewer's own guard
    checks them against the artifact ✓, the contrast label's table included
    ✓); the measurement tool writes its readout ✓ and prints the verdict
    against the registered bars ✓; the map's own artifact is byte-identical ✓
    and every run's numbers reproduce ✓; the log section reads the numbers out
    with the falsification conditions ✓; the gate green (867 tests ✓, ruff
    clean over `src`/`tests`/`experiments`/`tools` and the track's sweep folder
    ✓, the two verify tools OK ✓ (110 + 52 runs ✓), and both final sweeps
    reproducing their artifacts field for field ✓ (9 artifacts IDENTICAL ✓)).

---

## The lexical kNN (CS-41) — the pre-registration and the verdict

*Added on 2026-11-03, when the round was executed: the live plan's §7.5
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone
line). The verdict and the numbers are the track log's §CS-41.*

### 7.5 The map's quality round (CS-36 … CS-41)

- [x] **CS-41 — the lexical kNN: how much of the retrieval is words alone?**
  The retrieval side's lexical counterpart.

  **The mechanism.** A **predictor** kind (the main pipeline's, like CS-29's
  lexical floor): the TF-IDF nearest neighbours of the text over the **train**
  split (the same tokenizer and the same TF-IDF as CS-29's floor), their labels
  voted with the **existing** kNN machinery (`retrieve/knn.py` — the same
  votes, the same powers, the same top-k). The measurement is the same harness
  at the same `top_k` k=2 operating point, and above all the **shortlist R@50**
  — the candidate generator's ceiling, where the embedding's advantage is
  largest (the champion's fusion 0.7423 against the embedding kNN's 0.89–0.96).

  **Why.** It is the evidence PLAN.md's deferred BM25 checkbox is waiting for:
  "is a lexical retrieval channel worth adding to the fusion?" — and the answer
  needs the lexical retrieval's own **ceiling**, not the lexical classifier's
  **floor** (CS-29's 0.3845). The decomposition it delivers is the useful one:
  "the embedding's retrieval buys X of micro-F1 over the lexical retrieval".

  **The falsification, registered in advance.** (i) The **retrieval test**: if
  the lexical kNN's shortlist R@50 is materially **below** the embedding kNN's,
  the words alone retrieve less, the fusion's shortlist gap (0.7423 against
  0.89–0.96) is not a *lexical* breadth gap, and the BM25 checkbox stays
  closed. If it is **at or above** the embedding kNN's, the lexical route *is* a
  viable candidate generator and the BM25 question is live — that is the reading
  that would change the main plan. (ii) The **decomposition**: the lexical kNN's
  own micro-F1 in the same harness, beside the embedding kNN's and the
  champion's, so the reader sees what the embedding's retrieval buys.

  **The cost.** About a day. **No new dependency** (the TF-IDF machinery is
  CS-29's). It is a **main-pipeline** change: its rows land in the main UPR
  table, the narrative in the track log (CS-29's shape).

  **The pre-registration (2026-11-03, written before the first call).** The
  block above fixes the mechanism; this fixes the *shape* of the number.

  - **Where the code lives.** A new **predictor** in the main pipeline's
    package: `src/cluster_rag/predictors/lexical_knn.py` (the registry
    `src/cluster_rag/predictors/__init__.py` gains the name). The **TF-IDF** is
    CS-29's floor's own (`src/cluster_rag/predictors/lexical_tfidf.py` — the
    same vectorizer, the same knobs, fitted on the **train** split only ✓); the
    **votes** are the main pipeline's own (`src/cluster_rag/retrieve/knn.py` —
    `knn_vote_scores`: the same votes ✓, the same clipping ✓, the same power
    ✓, the same top-k ✓ and the same tie-break ✓). The predictor is
    **embedding-free** (`REQUIRES_EMBEDDINGS = False` ✓), so the `evaluate`
    command needs no embedding run ✓ — the same wiring as CS-29's floor ✓.

  - **One reading settled before the first call.** The block says "the TF-IDF
    nearest neighbours of the text over the **train** split (the same tokenizer
    and the same TF-IDF as CS-29's floor)". The TF-IDF's rows are
    **l2-normalised** ✓ (CS-29's vectorizer's own `norm="l2"` ✓), so the dot
    product the kNN machinery computes **is** the cosine ✓ — the same geometry
    the embedding kNN uses ✓, and the "nearest" neighbour is the nearest in the
    TF-IDF cosine ✓. The one thing the machinery needs and the TF-IDF does not
    give is a **float32 dense** matrix (the kNN core is dense ✓), so the sparse
    matrix is materialised dense ✓ — the numbers are unchanged by that ✓ (the
    zeros stay zeros ✓). A text with **no** known feature (a row of zeros ✓)
    has no neighbour with a positive similarity ✓, so its votes are 0 ✓ and it
    is ranked by the harness's own tie-break ✓ — reported, never dropped ✓.

  - **The arms.** The same harness (`evaluate` ✓), the same `top_k` k=2
    operating point ✓ (the champion's own ✓), the **val** split ✓:
    - the **grid**: k ∈ {1, 3, 5, 10, 20} ✓ — the **pool-cut family's own
      grid** ✓ (the embedding kNN was measured on exactly it ✓) — at the
      machinery's default power (1.0 ✓, the embedding kNN's own ✓);
    - the **primary**: k = 10 ✓ — the **champion's own** kNN k ✓ (the fusion's
      a-term uses k=10 ✓), so the like-for-like comparison is with
      `upr-eval-knn-embeddinggemma-k10` ✓;
    - the **label-blind control** at k = 10 ✓ (the track's own yardstick ✓: the
      train records' label sets permuted among the records ✓, the retrieval
      untouched ✓) — the **empirical floor** of the shortlist ✓.

    The **power** is **not** swept ✓ (the machinery's default ✓): one thing at
    a time ✓.

  - **The runs.** Six `evaluate` runs on UPR **val** ✓, in the sweep file
    `experiments/lexical_knn_sweep.py` (the sweep label `lexical_knn` ✓) — a
    **main-pipeline** change ✓, so the rows land in the **main** UPR table
    (`docs/experiments_table_upr.md` ✓) and the narrative in the **track log** ✓
    (CS-29's shape ✓): `upr-eval-lexical-knn-k{1,3,5,10,20}-val` and
    `upr-eval-lexical-knn-k10-blind-val` ✓.

  - **What is measured.** Each run's own numbers from its artifact ✓: the
    **shortlist R@50** above all ✓ (the candidate generator's ceiling ✓), the
    **shortlist R@20** and **R@100** ✓, the **micro-F1** ✓ (the decomposition
    ✓), the macro- and sample-F1 ✓, the nDCG@5 ✓, the coverage ✓ and the
    per-bucket view ✓ (the head, the torso, the tail, the unseen ✓).

  - **The numbers it is read against** (all from the artifacts ✓):
    - the **champion's fusion** (`upr-eval-fusionc-embeddinggemma-k10-c1`):
      micro-F1 **0.5479**, shortlist R@50 **0.7423** ✓;
    - the **embedding kNN at the same k** (`upr-eval-knn-embeddinggemma-k10`):
      micro-F1 **0.5388**, shortlist R@50 **0.9576** ✓ — and the pool-cut
      family's range over its 70 runs: shortlist R@50 **0.8918 … 0.9605** ✓;
    - **CS-29's lexical floor** (`upr-eval-lexical-tfidf-val`): micro-F1
      **0.3845**, shortlist R@50 **0.7205** ✓ — the **classifier**'s floor ✓,
      not the **retrieval**'s ceiling ✓;
    - the **chance level of the shortlist**: with 68 labels, a ranking that
      knows nothing still puts a gold label in the top 50 with probability
      **50/68 = 0.7353** ✓ — so the fusion's 0.7423 is **barely above chance** ✓
      and the blind control measures the floor empirically ✓.

  - **The falsification, registered in advance.**
    1. **The retrieval test.** The lexical kNN's shortlist R@50 must be **at
       least 0.8918** — the **lowest** of the embedding kNN family's 70 runs ✓
       (the family's own floor ✓) — for the lexical route to count as "at or
       above the embedding kNN's" ✓. If it is **materially below** that ✓, the
       words alone retrieve less ✓, the fusion's shortlist gap (0.7423 against
       0.89–0.96 ✓) is **not** a lexical breadth gap ✓ and PLAN.md's BM25
       checkbox **stays closed** ✓. If it is at or above ✓, the lexical route
       **is** a viable candidate generator ✓ and the BM25 question is **live**
       ✓ — the reading that would change the main plan ✓. The like-for-like
       number (the champion's own embedding kNN at k=10, **0.9576** ✓) is
       reported beside it ✓, and so is the lexical kNN's **best** k on the
       grid ✓.
    2. **The decomposition.** The lexical kNN's own micro-F1 at the primary k,
       beside the embedding kNN's (**0.5388** ✓) and the champion's (**0.5479**
       ✓) — so the reader sees what the embedding's retrieval buys ✓.
    3. **The label-blind control.** The blind arm's shortlist R@50 and its
       micro-F1 are the **empirical floor** ✓: the lexical route's numbers are
       read against them ✓ (and against the naive chance level 0.7353 ✓). The
       control buys nothing ✓ — the track's yardstick ✓.

  - **What is NOT registered.** (i) The **power** sweep (the machinery's
    default only ✓). (ii) A **BM25** channel (the dependency and the fourth
    fusion term are the reviewer's ✓ — this change set measures the *route* ✓,
    not the term ✓). (iii) The **fit side**: the `evaluate` command has no
    cross-fit ✓ and this change set measures a **ceiling** ✓, not a claim that
    anything improved ✓ (the corpus map's fit-side instrument belongs to the
    `analyze-upr` command ✓, and CS-29's floor has no fit side either ✓).
    (iv) The **phrase** unit (CS-39's verdict: the phrases are noise ✓).

  - **The exit criteria.** The six runs hold their numbers ✓; the rows are
    auto-appended in the **main** UPR table ✓; the log section reads the
    numbers out with the falsification conditions ✓; every number CS-29,
    CS-30, CS-31, CS-32, CS-36, CS-37, CS-38, CS-39 and CS-40 reported is
    **reproduced unchanged** ✓ — in particular **CS-29's floor** ✓ (the TF-IDF
    knobs are shared now ✓, so its artifact must reproduce field for field ✓);
    the gate green (ruff over `src`/`tests`/`experiments`/`tools` and the track's
    sweep folder ✓, the full pytest suite ✓, the two verify tools ✓, and both
    final sweeps reproducing their artifacts field for field ✓).

  - **The cost, in compute.** The TF-IDF is fitted on the train split's 1,676
    texts ✓ (17,217 features ✓ — CS-29's own count ✓); the kNN's similarities
    are one dense matmul ✓ (660 × 1,676 × 17,217 ✓). **No new dependency** ✓.
  *(pre-registered 2026-11-03; **executed 2026-11-03** — the change set is a
  **main-pipeline** one: a new predictor `lexical_knn`
  (`src/cluster_rag/predictors/lexical_knn.py`, the registry gains the name),
  the shared TF-IDF knobs (`predictors/lexical_tfidf.py` — CS-29's floor's
  own, so **CS-29's artifact reproduces field for field** ✓), the sweep
  `experiments/lexical_knn_sweep.py` (the label `lexical_knn`) and **six**
  `evaluate` runs — the rows land in the **main** UPR table
  (`docs/experiments_table_upr.md`, the section `## UPR — lexical_knn`) and the
  narrative in the track log. The log section is `## CS-41 — the lexical kNN:
  how much of the retrieval is words alone?` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*


  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2;
  the train split's 1,676 texts and 68 labels; no model call, no embedding; the
  six runs 0.9 s in total).**

  - **The arms** (the micro-F1 / the shortlist R@50 / the nDCG@5):

    | the arm | micro-F1 | macro-F1 | shortlist R@20 | shortlist R@50 | nDCG@5 |
    |---|---|---|---|---|---|
    | the lexical kNN, k=1 | 0.3146 | 0.2928 | 0.5235 | 0.8463 | 0.3900 |
    | the lexical kNN, k=3 | 0.3913 | 0.3271 | 0.6486 | 0.9125 | 0.5069 |
    | the lexical kNN, k=5 | 0.4263 | 0.3509 | 0.6770 | 0.9293 | 0.5416 |
    | **the lexical kNN, k=10 (the primary)** | **0.4605** | 0.4035 | 0.6967 | **0.9437** | 0.5803 |
    | the lexical kNN, k=20 | **0.4749** | **0.4392** | 0.6996 | 0.9415 | **0.5987** |
    | the **label-blind** counterpart (k=10) | 0.0707 | 0.0244 | 0.3854 | 0.8111 | 0.1104 |

    (The per-bucket view of the primary arm: the head 0.5795, the torso 0.4150,
    the tail 0.4414, the unseen 0.0000 ✓.)
  - **The numbers it is read against**: the champion's fusion (micro-F1
    **0.5479**, shortlist R@50 **0.7423** ✓), the embedding kNN at the same k
    (micro-F1 **0.5388**, shortlist R@50 **0.9576** ✓) and the pool-cut
    family's range over its 70 runs (shortlist R@50 **0.8918 … 0.9605** ✓),
    CS-29's lexical floor (micro-F1 **0.3845**, shortlist R@50 **0.7205** ✓),
    and the **chance level of the shortlist** — with 68 labels, a ranking that
    knows nothing still puts a gold label in the top 50 with probability
    **50/68 = 0.7353** ✓.
  - **The falsification, read out.**
    1. **The retrieval test HOLDS**: the lexical kNN's shortlist R@50 at the
       primary k is **0.9437** — above the registered bar **0.8918** ✓ (the
       embedding kNN family's own floor ✓) and **98.5 %** of the champion's own
       embedding kNN at the same k (0.9576 ✓). The words alone retrieve nearly
       as well as the embedding does ✓. **The BM25 question is LIVE** ✓ — the
       reading that would change the main plan ✓.
    2. **The decomposition**: the lexical kNN's own micro-F1 at the primary k
       is **0.4605**, against the embedding kNN's **0.5388** (+**0.0783** ✓)
       and the champion's **0.5479** (+**0.0874** ✓) — but on the **shortlist**
       the gap is only **0.0139** ✓: the embedding's advantage is in the
       **ordering**, not in the **breadth** ✓.
    3. **The label-blind control**: the blind arm reaches a micro-F1 of
       **0.0707** ✓ (the association is worth **+0.3898** ✓) and a shortlist
       R@50 of **0.8111** ✓. That floor is **high** ✓ — with 68 labels and a
       top-50 shortlist, the **label frequencies alone** put 81 % of the gold
       cells inside the shortlist ✓ — so the absolute shortlist numbers are
       less discriminating than they look ✓ (the label-driven part of the
       lexical route's shortlist is **+0.1326** ✓).
  - **The reading.** The lexical route is a real candidate generator ✓: its
    shortlist R@50 (**0.9437**) is above the embedding kNN family's floor
    (**0.8918**) ✓ and close to the champion's own embedding kNN (**0.9576**)
    ✓, while its **classifier** (CS-29's floor, 0.3845 ✓) is far behind the
    champion's ✓ — the lexical route's value is **breadth** ✓, exactly the role
    PLAN Phase 4's deferred BM25 checkbox was waiting for ✓. And the
    **champion is a worse candidate generator than either route alone**: the
    fusion's shortlist R@50 is **0.7423** — **barely above the 0.7353 chance
    level** ✓ — against the lexical route's 0.9437 ✓ and the embedding kNN's
    0.9576 ✓; the fusion **dilutes** the candidate set ✓ (CS-8b's finding, with
    a lexical witness ✓). **What the numbers do not settle**: the shortlist
    metric's floor is high ✓ (the frequencies alone: 0.8111 ✓), so the
    decision about a lexical term in the fusion belongs on a number that
    accounts for that floor — the fusion's own shortlist with the term, or the
    **union** of the two candidate sets ✓ (the track's own instrument, CS-14
    ✓) — and not on the shortlist alone ✓; the embedding kNN's **own** blind
    counterpart is not measured ✓ (not registered ✓).
  - **The exit criteria.** The six runs hold their numbers ✓; the rows are
    auto-appended in the **main** UPR table ✓; the log section reads the
    numbers out with the falsification conditions ✓; **CS-29's own floor
    reproduces its artifact field for field** ✓ (the TF-IDF knobs are shared
    now ✓ — re-run on 2026-11-03 ✓) and every other number CS-29, CS-30,
    CS-31, CS-32, CS-36, CS-37, CS-38, CS-39 and CS-40 reported is reproduced
    unchanged ✓; the gate green (ruff over `src`/`tests`/`experiments`/`tools`
    and the track's sweep folder ✓, the full pytest suite ✓, the two verify
    tools ✓, and both final sweeps reproducing their artifacts field for field
    ✓).

---

## The fit-side mask (CS-42) — the pre-registration and the verdict

*Added on 2026-11-03, when the change set was executed: the live plan's §7.6
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone line).
The verdict and the numbers are the track log's §CS-42.*

### 7.6 The fit-side dictionary round (CS-42 … CS-45)

- [x] **CS-42 — the fit-side mask: teach the pack that the floor is not
  "impossible".** The dictionary's tables are emptied for a fixed, seed-chosen
  set of **seen** labels during the pack's fit pass only; their gold rows are
  kept, and the evaluation is unchanged.

  **The mechanism.** With the `corpus_map` group in the pack, the learned
  combiner learns "the map's floor ⇒ this label is out of the question", because
  on train every floored cell belongs to a label that is never positive. The
  mask breaks that coincidence: during the fit pass, the tables of **8 randomly
  chosen seen labels** are emptied (`keep[label] = ∅`), so their fit cells carry
  the floor exactly like the eight held-out labels' cells, while their gold rows
  stay positive. The combiner then sees "positive with the channel at the floor"
  and cannot use the floor as a proxy for impossibility. The evaluation side is
  untouched: the val split is scored with the full train map, exactly as CS-32
  scores it. The mask is a *fit-side instrument* — it also gives the track a
  "pretend unseen" control it does not have today.

  **The selection rule, fixed before the call.** Candidates are the **train**
  labels whose map table is non-empty (60 of the 68); sorted by name;
  `np.random.default_rng(fit_mask.seed).choice(len(candidates), fit_mask.n_labels,
  replace=False)` with `n_labels = 8`, `seed = 42`. The eight held-out labels are
  ineligible by construction (no table), and no val number enters the draw. The
  chosen names and their train counts are recorded in the artifact. Uniform,
  unstratified: the draw may land on any frequency bucket, and that composition
  is recorded, not corrected.

  **Where the code lives.** The scorer
  (`text_processors/lexical/corpus_map_score.py`): `cross_fit_scores` gains
  `mask_labels`, which builds each fold's `keep` from **that fold's** map and
  empties the masked labels' entries; `score_cells` is unchanged (it already
  implements `keep`). The command
  (`commands/upr.py::_upr_corpus_map_feature`) selects the labels, computes the
  masked fit matrix and records the provenance; the driver
  (`upr_learning.py::score_upr_learned_arms`) fits one more arm with the masked
  fit column and the **unchanged** eval column. The config key is
  `upr_analysis.learned.corpus_map_feature.fit_mask` (`enabled`, `n_labels`,
  `seed`; **off** by default).

  **The guard.** The fit side's map is cross-fit over 5 folds (CS-32's rule), and
  the mask rides into it: each fold's mask empties the masked labels' tables in
  that fold's own map. The eval matrix is the real arm's own eval matrix,
  byte-identical by construction (a test asserts it).

  **The run.** One: `upr-analyze-cmap-fitmask` — the `analyze-upr` command with
  CS-32's configuration (groups `embedding + lexical + chunk_profile +
  corpus_map`, `histgb`, atoms chunker, `chunk_profile`'s 3 units, `top_k` k=2,
  val split) plus the knob. Sweep
  `experiments_text_processing/upr_learned_cmap_fitmask_sweep.py`, label
  `upr_learned_cmap_fitmask`; artifact
  `results/upr-analyze-cmap-fitmask/analysis.json`. **Four arms in the one
  artifact**: `learned_histgb` (the real map — CS-32's arm),
  `learned_histgb_control` (the pack without the group),
  `learned_histgb_label_blind` (the blind map) and `learned_histgb_fit_mask`
  (**the new arm**). The pack's three floors ride along unchanged.

  **The shape of the number.** The headline is one number from the artifact: the
  fit-mask arm's **unseen-bucket micro-F1** (`per_bucket.unseen.f1.micro_f1`).
  It is read beside the same run's real arm (0.0000), control (0.4469) and blind
  arm (0.0066). The paired differences are text-resampled (2,000 resamples, seed
  42, `tools/compare_upr_arms.py`'s own procedure): fit-mask − real on the unseen
  bucket (the mask's own effect), fit-mask − control on the unseen bucket, on the
  **seen view** (head+torso+tail pooled) and pooled, and blind − control on the
  seen view (CS-32's yardstick). Secondary, reported without a bar: the eight
  masked labels' own val numbers (the price of the mask), the per-label lines for
  the unseen bucket, and the shortlist R@50.

  **The falsification, registered in advance.**
  1. **The recovery test.** The fit-mask arm's unseen micro-F1 must be **at least
     0.45**, with the dictionary column still in the pack. The bar is the
     control's own number (0.4469) rounded up: the mask must give the eight back
     to at least the level the pack has *without* the column, not merely soften
     the collapse. If the eight still collapse, the silence proxy is not the
     binding cause and P1 is a null.
  2. **The no-free-lunch test.** The seen buckets must keep their gain: the
     paired **seen-view** difference (fit-mask − control) must exclude zero, its
     point estimate must stay at or above **+0.10** (the unmasked arm's
     seen-view number is **+0.1319 [+0.1087, +0.1554]** — three quarters of it),
     and the fit-mask arm's **pooled** micro-F1 must stay at or above **0.5502**
     (CS-32's pooled number). Without the seen-view floor, "the combiner simply
     stops trusting the column" would pass for the wrong reason.
  3. **The label-blind control.** The blind arm's Δ against the control on the
     seen view must still buy nothing (CS-32: −0.0509 [−0.0678, −0.0340]), and
     its unseen bucket must stay collapsed.

  **The cost.** One run, about 2–3 minutes (cache hits, no model calls); the
  change set about half a day including tests and docs; **no new dependency**.

  **What is NOT registered.** The per-cell abstention (CS-37's deferred
  variant); mask sizes other than 8 and seeds other than 42; choosing the masked
  labels by any frequency or val statistic; masking the eval side; the
  `corpus_map` predictor's path; the mask combined with the description channel
  (CS-43's question); a second draw of the mask; the fit-mask arm's label-blind
  counterpart (offered in review as an extra control, not taken).

  **The exit criteria.** The artifact holds the four arms, the per-bucket
  numbers and the mask's provenance (the eight names, their train counts, the
  rule, the seed); the table row is auto-appended; the log section §CS-42 reads
  the numbers out against the three conditions; the checkbox is ticked; the block
  is moved verbatim to the archive; the gate is green (ruff, pytest,
  `tools/verify_upr_baseline.py upr-analyze-cmap-fitmask`); and
  `upr-analyze-cmap` reproduces its artifact field for field (the knob is off by
  default); one commit.
  *(pre-registered 2026-11-03; **executed 2026-11-03** — the run is
  `upr-analyze-cmap-fitmask` (the artifact
  `results/upr-analyze-cmap-fitmask/analysis.json`), the sweep is
  `experiments_text_processing/upr_learned_cmap_fitmask_sweep.py` (the label
  `upr_learned_cmap_fitmask`), the log section is `## CS-42 — the fit-side mask:
  the dictionary's floor stops meaning "impossible"` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2,
  the gradient-boosted combiner, the pack `embedding + lexical + chunk_profile
  + corpus_map`; the map is built from the **train** split only and the fit
  side's map is **cross-fit** over 5 folds; the mask empties the tables of 8
  seed-chosen **seen** labels on the fit side only; no model calls; 123 s).**

  - **The mask's provenance**: 8 labels drawn from the train map's 60 with a
    table (`default_rng(42).choice(60, 8, replace=False)`, candidates sorted by
    name) — *Death penalty* (66 train texts), *Racial discrimination* (59),
    *Corruption* (30), *Sexual and reproductive rights* (29), *UPR process* (28),
    *Youth rights* (23), *Right to housing* (20), *Internally displaced persons*
    (19); two sit in the torso bucket and six in the tail.

  - **The arms**:

    | the arm | micro-F1 | macro-F1 | coverage | head | torso | tail | unseen | nDCG@5 | shortlist R@50 |
    |---|---|---|---|---|---|---|---|---|---|
    | the champion (the whole retrieval + decision pipeline) | 0.5479 | 0.5006 | 0.868 | 0.6953 | 0.5121 | 0.5287 | 0.0000 | 0.6725 | 0.7423 |
    | the real arm (the map's column with the floor) — CS-32's arm | 0.5502 | 0.5081 | 0.897 | 0.6931 | 0.5088 | 0.5566 | 0.0000 | 0.6694 | 0.7979 |
    | **the fit-mask arm** | **0.5874** | **0.5577** | 1.000 | **0.7094** | **0.5150** | **0.5931** | **0.2947** | **0.7342** | **0.8922** |
    | the **control** (the pack without the group) | 0.4818 | 0.4652 | 1.000 | 0.5505 | 0.4122 | 0.4531 | 0.4469 | 0.6137 | 0.8909 |
    | the blind map | 0.4309 | 0.4277 | 0.926 | 0.5395 | 0.4099 | 0.4665 | 0.0066 | 0.5493 | 0.7741 |
    | the pack's random-vector floor | 0.5357 | 0.4787 | 0.882 | 0.6758 | 0.4885 | 0.5223 | 0.0000 | 0.6506 | 0.7729 |

  - **The falsification, read out.**
    1. **The recovery test FAILS**: the fit-mask arm's unseen micro-F1 is
       **0.2947**, far under the registered bar of 0.45. The mask does move the
       eight — the paired difference (fit-mask − real) on the unseen bucket is
       **+0.2947 [+0.2333, +0.3550]** — but it does not give them back.
    2. **The no-free-lunch test PASSES**: the paired **seen-view** difference
       (fit-mask − control) is **+0.1478 [+0.1236, +0.1710]** (the registered
       floor was +0.10) and the pooled micro-F1 is **0.5874** (the registered
       floor was 0.5502); against the real arm the seen buckets still gain
       **+0.0158 [+0.0052, +0.0268]** and the pooled number **+0.0372
       [+0.0257, +0.0490]**.
    3. **The label-blind control PASSES**: on the seen view the blind arm's Δ
       against the control is **−0.0032 [−0.0181, +0.0119]** (it buys nothing)
       and its unseen bucket stays at **0.0066** (a paired −0.4403 [−0.4935,
       −0.3835] against the control).
  - **The reading: the mask is a lever, but not the whole lever.** Breaking the
    fit-side coincidence lifts the eight from **0.0000** to **0.2947**, and five
    of the eight carry real scores again (the per-label score distributions:
    *Gender-based violence* 0.402, *Freedom of association* 0.384, *Special
    procedures* 0.373, *Freedom of religion and belief* 0.353, *Asylum-seekers*
    0.316, against backgrounds of 0.005–0.030), while *ESC rights - general*
    (0.023), *General* (0.001) and *Other* (0.009) stay down — the labels whose
    meaning is not in the texts' words. The residual gap is measured: the
    fit-mask arm trails the control by **0.1522 [0.1008, 0.2066]** on the unseen
    bucket, so "the floor is not impossible" (learned) and "the floor is not the
    only evidence" (not learned) are different facts. The mask costs the seen
    buckets nothing — it gains them (+0.1478 against the control; +0.0158
    against the real arm) — and the pooled number **0.5874** is the track's
    second-best pack number (CS-13's decision-feature pack 0.5965 is first) and
    **above the champion's 0.5479**; the shortlist R@50 comes back to **0.8922**
    (the real arm's 0.7979). It is the lift the bar did not ask for.
  - **The exit criteria.** The artifact holds the four arms with their
    per-bucket numbers and the mask's provenance ✓; the table row is
    auto-appended ✓; the log section reads the numbers out with the three
    conditions ✓; **CS-32's own run reproduces its artifact field for field** ✓
    (the knob is off by default ✓ — re-run on 2026-11-03 ✓); the gate green ✓
    (ruff over `src`/`tests`/`experiments`/`tools` and the track's sweep folder,
    the full pytest suite, the two verify tools).


---

## The description in the dictionary's pack (CS-43) — the pre-registration and the verdict

*Added on 2026-11-03, when the change set was executed: the live plan's §7.6
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone line).
The verdict and the numbers are the track log's §CS-43.*

### 7.6 The fit-side dictionary round (CS-42 … CS-45)

- [x] **CS-43 — the dictionary + the catalog description in one pack.** Add
  CS-11's catalog channels to the same pack that carries the dictionary column.

  **The mechanism.** The dictionary's column is **silent for the eight held-out
  labels** (their tables are empty), and CS-32/CS-37 measured what that costs:
  the combiner learns "floor ⇒ impossible" and the unseen bucket collapses to
  0.0000. The catalog's description is the one channel that is **non-silent for
  all 68 labels**: alone it lifts the unseen bucket to **0.4690** (query half,
  `upr-analyze-learned-cat-desc`) and **0.4824** (feature half,
  `upr-analyze-learned-cat-feat`). Nobody has run the two channels in **one
  pack**, so the question is whether the description's presence stops the
  dictionary's floor from being read as a veto — or whether the dictionary
  overrules it.

  **Why.** It is the direct test of CS-42's reading: the mask broke the fit-side
  coincidence and recovered only two-thirds of the eight; this change set asks
  whether a **non-silent channel** does better, without touching the fit side at
  all (the description is a label-side feature, present for every label, and it
  is not built from the corpus).

  **The two cells, fixed before the call.** CS-11's pack exactly, plus the
  dictionary's group:

  - **cell 1 (the primary — the description's feature half)**:
    run `upr-analyze-cmap-catalog`; groups `embedding + lexical +
    chunk_profile + corpus_map + label_part + label_type`; the label-side
    embedding is the champion's (`upr-labelembed-embeddinggemma-about`); the
    whole-text reference run is `upr-eval-fusionc-embeddinggemma-k10-c1`. This
    is CS-11's `upr-analyze-learned-cat-feat` pack (its `CATALOG_PACK`) plus the
    `corpus_map` group;
  - **cell 2 (the registered robustness arm — the description's query half)**:
    run `upr-analyze-cmap-catalog-desc`; groups `embedding + lexical +
    chunk_profile + corpus_map`; the label-side embedding is the description's
    (`upr-labelembed-embeddinggemma-desc`, cached); the whole-text reference run
    is `upr-eval-fusionc-embeddinggemma-k10-c1-desc` (CS-11's own guard for the
    description label side). This is CS-11's `upr-analyze-learned-cat-desc` pack
    (its `FULL_PACK`) plus the `corpus_map` group.

  Both cells carry the dictionary's feature with **CS-32's own settings**:
  `enabled: true`, `scoring: sum_zscore_positive`, `folds: 5`,
  `label_blind: true`, `abstain: false`, and the **fit-side mask off** (one
  thing at a time — CS-42's instrument is not combined here). The pack's other
  knobs are CS-11's (the atoms chunker, `top_k` 3, `histgb`, `control_weight`
  0.3, the same floors, the same reporting, seed 42).

  **The run.** Two: `upr-analyze-cmap-catalog` and
  `upr-analyze-cmap-catalog-desc`, the `analyze-upr` command on the **val**
  split. Sweep `experiments_text_processing/upr_learned_cmap_catalog_sweep.py`,
  label `upr_learned_cmap_catalog` (one table section,
  `## UPR — upr_learned_cmap_catalog`, two runs); artifacts
  `results/upr-analyze-cmap-catalog/analysis.json` and
  `results/upr-analyze-cmap-catalog-desc/analysis.json`. **Four arms per
  artifact**, generated by the pack as usual: `learned_histgb` (the pack with
  the dictionary's column), `learned_histgb_control` (the **same catalog pack
  without the group** — the description's own level, so the pair is by
  construction), `learned_histgb_label_blind` (the blind map) and
  `learned_histgb_random` (the pack's random-vector floor). The pack's chance
  and control-role floors ride along unchanged.

  **The guard: the control arm reproduces CS-11.** Each cell's control is
  CS-11's pack exactly (the same groups, the same label embeddings, the same
  knobs), so it must reproduce the matching CS-11 arm:
  `upr-analyze-learned-cat-feat`'s `learned_histgb` (**0.5076** pooled /
  **0.4824** unseen) for cell 1, and `upr-analyze-learned-cat-desc`'s
  (**0.5084** / **0.4690**) for cell 2. A mismatch means the two runs' numbers
  cannot be compared and the change set stops there. The whole-text reference
  arm's own guard (against the matching `evaluate` run) rides along unchanged.

  **The shape of the number.** The headline is the real arm's **unseen-bucket
  micro-F1** (`per_bucket.unseen.f1.micro_f1`) beside its **pooled** micro-F1
  (`overall.f1.micro_f1`), each read against the same run's control. Reported
  beside them: the per-bucket view, the paired differences (real − control on
  the unseen bucket, the seen view and pooled; blind − control on the seen view;
  2,000 text resamples, seed 42, `tools/compare_upr_arms.py`'s procedure), the
  shortlist R@50 and the label-blind arm. The bars are on the point estimates;
  the intervals are the reading, not the verdict.

  **The falsification, registered in advance.**
  1. **The recovery test.** The real arm's unseen micro-F1 must be **at least
     0.45** — the round's bar, and the point where "the dictionary costs the
     eight a little" becomes "the dictionary still overrules the description"
     (the control's own level is **0.4824** in cell 1 and **0.4690** in cell 2,
     so the bar allows the column a cost of ~0.03 against the description
     pack). If the unseen stays at **0.00**, the dictionary overrules the
     description and CS-42's fit-side route is the only one left.
  2. **The pooled test.** The real arm's **pooled** micro-F1 must stay **at or
     above 0.55** (CS-32's map pack reached 0.5502, so the bar says the
     description must not cost what the column buys). Both legs are required;
     the log reports each one separately, so a pass on one and a fail on the
     other is read as such and not smoothed over.
  3. **The label-blind control.** The blind arm's Δ against the control on the
     seen view must still buy nothing (the round's multiple-comparisons
     discipline: the gain must be label-driven).

  The **verdict is cell 1's**; cell 2 is the registered robustness arm, read the
  same way and reported beside it. If the two disagree, the log says so and no
  claim rests on the cell that passes alone.

  **The cost.** Two runs, about 2–3 minutes each (cache hits: the description's
  label embeddings and all text embeddings are already cached; **no model
  calls**); about half a day including the docs; **no new dependency**. No code
  change is expected: the groups, the pack, the config and the arm machinery
  already exist (CS-11 and CS-32 built them), so this change set is a sweep, two
  runs and their record.

  **What is NOT registered.** CS-42's fit-side mask in the same pack (one thing
  at a time); the **decision feature** (CS-13) and the **reranker's** (CS-16);
  the catalog's **parent backoff** (CS-11's third channel); the **mined
  lexicon** (CS-35/CS-45); a **tuned** anything (the knobs are CS-11's); the
  query half **and** the feature half in one arm (each cell carries one half);
  and **test**.

  **The exit criteria.** The two artifacts hold the four arms with their
  per-bucket numbers; each cell's control reproduces its CS-11 arm field for
  field; the table rows are auto-appended; the log section §CS-43 reads the two
  cells out against the three conditions; the checkbox is ticked; the block is
  moved verbatim to the archive; the gate is green (ruff, pytest, and
  `tools/verify_upr_baseline.py` on both runs); one commit.

  *(pre-registered 2026-11-03; **executed 2026-11-03** — the runs are
  `upr-analyze-cmap-catalog` (the artifact
  `results/upr-analyze-cmap-catalog/analysis.json`) and
  `upr-analyze-cmap-catalog-desc` (`results/upr-analyze-cmap-catalog-desc/analysis.json`),
  the sweep is `experiments_text_processing/upr_learned_cmap_catalog_sweep.py`
  (the label `upr_learned_cmap_catalog`), the log section is `## CS-43 — the
  dictionary and the catalog description: the description cannot outvote the
  floor` in `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2, the
  gradient-boosted combiner; cell 1 = the description's feature half, cell 2 =
  its query half; the dictionary's column is CS-32's, CS-42's mask is off; no
  model calls; 112.6 s + 144.4 s).**

  - **The guard held exactly**: each cell's **control** (the same catalog pack
    without the column) reproduces CS-11's arm field for field — cell 1's
    **0.5076** pooled / **0.4824** unseen (CS-11's `upr-analyze-learned-cat-feat`)
    and cell 2's **0.5084** / **0.4690** (`-cat-desc`) — and the whole-text
    reference arms reproduce their own `evaluate` runs field by field.

  - **The arms, cell 1 (the feature half)**:

    | the arm | micro-F1 | macro-F1 | coverage | head | torso | tail | unseen | nDCG@5 | shortlist R@50 |
    |---|---|---|---|---|---|---|---|---|---|
    | the champion (the whole retrieval + decision pipeline) | 0.5479 | 0.5006 | 0.868 | 0.6953 | 0.5121 | 0.5287 | 0.0000 | 0.6725 | 0.7423 |
    | **the real arm (the description + the column)** | **0.5608** | **0.5268** | 0.897 | **0.7029** | **0.5226** | **0.5800** | **0.0066** | 0.6790 | 0.8095 |
    | the **control** (the same pack without the column) | 0.5076 | 0.5077 | 0.971 | 0.5492 | 0.4559 | 0.5047 | **0.4824** | 0.6643 | 0.8955 |
    | the blind map | 0.4597 | 0.4686 | 0.912 | 0.5682 | 0.4418 | 0.4956 | 0.0264 | 0.5834 | 0.7865 |
    | the pack's random-vector floor | 0.5395 | 0.4932 | 0.912 | 0.6710 | 0.5108 | 0.5256 | 0.0066 | 0.6619 | 0.7860 |

  - **The arms, cell 2 (the query half)**:

    | the arm | micro-F1 | macro-F1 | coverage | head | torso | tail | unseen | nDCG@5 | shortlist R@50 |
    |---|---|---|---|---|---|---|---|---|---|
    | the champion with the description label vectors | 0.5456 | 0.4959 | 0.868 | 0.6920 | 0.5040 | 0.5344 | 0.0000 | 0.6702 | 0.7352 |
    | **the real arm (the description + the column)** | **0.5532** | **0.5133** | 0.882 | 0.6851 | **0.5152** | **0.5831** | **0.0000** | 0.6717 | 0.8042 |
    | the **control** (the same pack without the column) | 0.5084 | 0.4872 | 0.985 | 0.5949 | 0.4101 | 0.5121 | **0.4690** | 0.6557 | 0.8864 |
    | the blind map | 0.4483 | 0.4315 | 0.912 | 0.5797 | 0.3917 | 0.5017 | 0.0263 | 0.5748 | 0.7726 |
    | the pack's random-vector floor | 0.5312 | 0.4809 | 0.897 | 0.6681 | 0.4854 | 0.5245 | 0.0000 | 0.6468 | 0.7836 |

  - **The falsification, read out.**
    1. **The recovery test FAILS in both cells**: the real arm's unseen micro-F1
       is **0.0066** (cell 1) and **0.0000** (cell 2), far under the registered
       0.45, while the same runs' controls carry the eight at **0.4824** /
       **0.4690**; the paired differences (real − control, unseen) are
       **−0.4758 [−0.5297, −0.4174]** and **−0.4690 [−0.5228, −0.4112]** — the
       column costs the eight the whole of the description's gain.
    2. **The pooled test PASSES**: **0.5608** and **0.5532** (>= 0.55); the
       column's gain over the description pack is **+0.0532 [+0.0304, +0.0751]**
       and **+0.0448 [+0.0190, +0.0688]** pooled, **+0.1199 [+0.0977, +0.1420]**
       and **+0.1076 [+0.0834, +0.1310]** on the seen view.
    3. **The label-blind control PASSES**: the blind arm's Δ against the control
       on the seen view is **+0.0036 [−0.0093, +0.0169]** (cell 1) and
       **−0.0131 [−0.0258, −0.0002]** (cell 2) — it buys nothing.
  - **The reading: the description cannot outvote the floor.** The two channels
    are complementary on the **seen** labels — the description raises the pack's
    floor (its control is 0.5076/0.5084 against CS-32's 0.4818) and the column
    adds its seen-labels gain on top, so the combined pack is the track's
    third-best pooled number (0.5608, behind CS-13's 0.5965 and CS-42's 0.5874;
    CS-32's dictionary-only pack is 0.5502, the champion 0.5479) — and not on the
    eight: the learner's floor-rule is inferred from the fit side, where the
    dictionary's silence is still a perfect proxy for "never positive" whether or
    not another channel is non-silent. A non-silent neighbour does not change
    what the silent one teaches. **The registered branch fires**: the
    pre-registration said "if the unseen stays at 0.00"; cell 2 is exactly
    **0.0000** and cell 1 is **0.0066** (0.7 % of its control — a collapse, but
    not literally zero, recorded as such). The blind arm's collapse is marginally
    less total with a non-silent channel (0.0264 / 0.0263 against CS-32's
    0.0066), and still a collapse. So the route for the eight stays **CS-42's
    fit-side instrument** (which itself recovered only two-thirds), and the
    description's value is the seen-label floor.
  - **The exit criteria.** The two artifacts hold the four arms with their
    per-bucket numbers ✓; each cell's control reproduces its CS-11 arm field for
    field ✓; the table rows are auto-appended ✓; the log section reads the two
    cells out against the three conditions ✓; the gate green ✓ (ruff, pytest,
    `tools/verify_upr_baseline.py` on both runs).


---

## The borrowed words (CS-44) — the pre-registration and the verdict

*Added on 2026-11-03, when the change set was executed: the live plan's §7.6
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone line).
The verdict and the numbers are the track log's §CS-44.*

### 7.6 The fit-side dictionary round (CS-42 … CS-45)

- [x] **CS-44 — lend the neighbour's words to the five thin labels.** For the
  five labels whose table is >= 80 % inside a partner's, the partner's words are
  added to the thin label's table with a **discount**.

  **The mechanism.** Five labels have (almost) no vocabulary of their own: their
  dictionary tables lie inside a partner's (CS-38's
  `corpus_map.label_pairs.vocabulary_less.subset_of_a_neighbour` — *Sexual and
  reproductive rights* **0.935** inside *Women's rights*; *Freedom of assembly*
  **0.909** inside *Detention*; *Statelessness* **0.864** inside *Rights of the
  Child*; *CP rights - general* **0.843** inside *International instruments*;
  *Trafficking* **0.808** inside *Rights of the Child*). So the map's column is
  evidence-poor for them exactly where it is evidence-rich for the partner: a
  text about sexual and reproductive rights is scored high on *Women's rights*
  and, for *Sexual and reproductive rights* itself, only on the words the two
  share. The change: the thin label's column also reads the partner's
  **evidence-for** words that its own table does not carry, at a **discount** —
  the words are lent, not copied, and the partner's own column is untouched.

  **The rule, fixed before the call.** The pairs: the labels with a non-empty
  table whose **best containment** is at or above **0.80** (CS-38's registered
  threshold), the partner being the first maximal containment in the label
  order — the five above. The lent entries: the partner's (label, word) pairs
  whose **z-score is positive** and whose word is **not in the lent label's own
  table**, each contributed at **discount 0.5** × the partner's weight (the
  arm's measure is the pack's own `sum_zscore_positive`). The lending relation
  is computed **once from the full train map** and recorded; it is applied to
  **every fold's own weights** on the fit side (the cross-fit: the pairs are a
  train-side fact, the weights are the fold's). A label whose own table is
  emptied on the fit side (the stability filter, CS-42's mask) does not borrow —
  the borrow composes with `keep`.

  **Where the code lives.** `text_processors/lexical/label_pairs.py` gains
  `containment_partners` (the map-only containment, the same computation CS-38's
  readout uses); `text_processors/lexical/corpus_map_score.py` gains
  `borrowed_tables` and the `borrow` option on `score_cells` / `cross_fit_scores`;
  the command (`commands/upr.py::_upr_corpus_map_feature`) computes the pairs,
  the two score matrices and the new arm's provenance; `upr_learning.py` fits the
  arm; the reporting (`eval/report.py::build_label_report` and
  `upr_analysis.py::arm_block`) gains **label sets** — a named label group's own
  section (`metrics.per_label_set`) — **off unless declared**, so every earlier
  artifact keeps its exact content. The config key is
  `upr_analysis.learned.corpus_map_feature.borrow` (`enabled`, `discount`,
  `threshold`; **off** by default).

  **The run.** One: `upr-analyze-cmap-borrow` — the `analyze-upr` command with
  CS-32's configuration exactly, plus the borrow knob. Sweep
  `experiments_text_processing/upr_learned_cmap_borrow_sweep.py`, label
  `upr_learned_cmap_borrow`; artifact
  `results/upr-analyze-cmap-borrow/analysis.json`. **Five arms**: the real arm
  (the plain column — CS-32's), the control (the pack without the group), the
  label-blind counterpart, the **borrow arm** (the lent column on both sides) and
  the random-vector floor; the chance and control-role floors ride along. The
  label sets declared in the run: the **five thin labels** and **their partners**
  (four distinct labels — *Rights of the Child* serves two pairs) as two pooled
  sets, plus the **ten labels individually**.

  **The shape of the number.** The **thin set's pooled micro-F1** in the borrow
  arm vs the plain arm — a paired Δ by text (2,000 resamples, seed 42) — and the
  same for the **partner set** and for the ten labels. The power is stated up
  front: the five thin labels carry **63** val gold cells (7 / 15 / 8 / 13 / 20)
  and the partners **274** (101 / 20 / 85 / 68), so the per-label intervals are
  wide and the **pooled sets are the instrument**. Reported beside the deltas:
  each arm's per-label-set sections, the pooled overall, the per-bucket view, the
  shortlist R@50 and the per-pair **swap readout** (how many val texts each arm
  predicts the thin label, the partner, both or neither).

  **The falsification, registered in advance.**
  1. **The lending test.** The thin set's paired Δ (borrow − plain) must be
     **positive with an interval that excludes zero**, and **at least 4 of the 5**
     individual thin labels' point estimates must be positive (the sign-
     consistency leg: one label may not carry the pooled gain alone). A positive
     point estimate whose interval includes zero is **unresolved (underpowered)**
     — with 63 gold cells that is a real possibility — and it does **not** license
     the change.
  2. **The partner test.** The partners' set's paired Δ must **not fall
     materially**: its point estimate **and** its interval's lower bound at or
     above **−0.02**. The partner's own column is untouched by the lending, so a
     fall can only come from the top-2 competition — exactly the registered risk
     ("the pair becomes confusable and the top-2 swaps them").
  3. **The swap readout (descriptive, no bar).** Per pair, the counts of val
     texts where each arm predicts the thin label, the partner, both or neither.

  **The guard.** The run's real arm, control, blind arm and random floor
  reproduce CS-32's numbers exactly (the knob is off in them); the lending
  relation reproduces CS-38's `subset_of_a_neighbour` — the same five labels,
  partners and shares (0.935 / 0.909 / 0.864 / 0.843 / 0.808); the reference arm
  reproduces its `evaluate` run field by field.

  **The cost.** One run, about 2–3 minutes (cache hits, **no model calls**); the
  change set about a day including tests and docs; **no new dependency**.

  **What is NOT registered.** The **map classifier** as the instrument (the pack
  is — it is where the dictionary's column lives and where the plan's cost line
  puts the cross-fit guard); a **discount or threshold sweep** (0.5 and 0.80, one
  value each); a **separate "borrowed" column** (the lending is folded into the
  lent label's own column); the combination with CS-42's mask or CS-43's
  description (one thing at a time); the **mined lexicon** (CS-45); **test**.

  **The exit criteria.** The artifact holds the five arms, the label sets, the
  lending provenance (the five pairs and their shares, the discount and the
  threshold, the per-fold application) and the paired blocks; the lending
  relation reproduces CS-38's five; the table row is auto-appended; the log
  section §CS-44 reads the numbers out against the three conditions; the
  checkbox is ticked; the block is moved verbatim to the archive; the gate is
  green (ruff, pytest, `tools/verify_upr_baseline.py` on the run); one commit.

  *(pre-registered 2026-11-03; **executed 2026-11-03** — the run is
  `upr-analyze-cmap-borrow` (the artifact
  `results/upr-analyze-cmap-borrow/analysis.json`), the sweep is
  `experiments_text_processing/upr_learned_cmap_borrow_sweep.py` (the label
  `upr_learned_cmap_borrow`), the log section is `## CS-44 — the borrowed words:
  the partner's vocabulary is not the thin labels'` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2, the
  gradient-boosted pack `embedding + lexical + chunk_profile + corpus_map`; the
  map is built from the **train** split only and the fit side's map is
  **cross-fit** over 5 folds; the lending is the five pairs of CS-38's class at
  the 0.80 containment, the partner's evidence-for words at the 0.5 discount;
  no model calls; 106.3 s).**

  - **The lending, recorded and reproduced**: the five pairs are computed from
    the full train map and match CS-38's `subset_of_a_neighbour` exactly —
    *Sexual and reproductive rights* 0.935 inside *Women's rights*; *Freedom of
    assembly* 0.909 inside *Detention*; *Statelessness* 0.864 inside *Rights of
    the Child*; *CP rights - general* 0.843 inside *International instruments*;
    *Trafficking* 0.808 inside *Rights of the Child*. The guard: the real,
    control, label-blind and random-vector arms reproduce CS-32's numbers
    exactly (0.5502 / 0.4818 / 0.4309 / 0.5357), and the reference arm
    reproduces its `evaluate` run field by field.

  - **The arms**:

    | the arm | micro-F1 | macro-F1 | coverage | head | torso | tail | unseen | nDCG@5 | shortlist R@50 |
    |---|---|---|---|---|---|---|---|---|---|
    | the champion (the whole retrieval + decision pipeline) | 0.5479 | 0.5006 | 0.868 | 0.6953 | 0.5121 | 0.5287 | 0.0000 | 0.6725 | 0.7423 |
    | the real arm (the plain column) — CS-32's arm | 0.5502 | 0.5081 | 0.897 | 0.6931 | 0.5088 | 0.5566 | 0.0000 | 0.6694 | 0.7979 |
    | **the borrow arm (the lent column)** | **0.5418** | 0.5060 | 0.882 | 0.6896 | 0.5014 | 0.5321 | 0.0000 | 0.6659 | 0.7953 |
    | the **control** (the pack without the group) | 0.4818 | 0.4652 | 1.000 | 0.5505 | 0.4122 | 0.4531 | 0.4469 | 0.6137 | 0.8909 |
    | the blind map | 0.4309 | 0.4277 | 0.926 | 0.5395 | 0.4099 | 0.4665 | 0.0066 | 0.5493 | 0.7741 |
    | the pack's random-vector floor | 0.5357 | 0.4787 | 0.882 | 0.6758 | 0.4885 | 0.5223 | 0.0000 | 0.6506 | 0.7729 |

  - **The pair's own cells** (the registered view is the standard F1 over the
    group's columns on **every** text; the secondary view restricts the texts to
    those carrying one of the group's labels and hides the lending's false
    positives):

    | the group | borrow F1 | plain F1 | Δ [95 % CI] — cells view (registered) | Δ — on the gold texts (secondary) |
    |---|---|---|---|---|
    | **the five thin labels** | **0.3536** | **0.4242** | **−0.0707 [−0.1356, −0.0035]** | +0.0580 [−0.0072, +0.1273] |
    | their partners (4 labels) | 0.7081 | 0.7094 | −0.0012 [−0.0270, +0.0241] | −0.0093 [−0.0322, +0.0123] |
    | *CP rights - general* | 0.2308 | 0.0000 | **+0.2308 [+0.0000, +0.4349]** | +0.3750 [+0.0000, +0.6316] |
    | *Trafficking* | 0.9048 | 0.8636 | **+0.0411 [+0.0000, +0.1095]** | +0.0000 [0, 0] |
    | *Freedom of assembly* | 0.2326 | 0.2500 | −0.0174 [−0.1818, +0.1622] | +0.1667 [+0.0000, +0.4011] |
    | *Sexual and reproductive rights* | 0.0556 | 0.2105 | −0.1550 [−0.3572, +0.0000] | −0.1944 [−0.6000, +0.0000] |
    | *Statelessness and the right to nationality* | 0.2353 | 0.4000 | −0.1647 [−0.3066, −0.0138] | +0.0000 [0, 0] |
    | the partners individually | — | — | *Detention* −0.1026 [−0.2223, −0.0116]; *International instruments* +0.0019 [−0.0460, +0.0455]; *Rights of the Child* −0.0166 [−0.0668, +0.0323]; *Women's rights* +0.0286 [−0.0098, +0.0712] | — |

  - **The falsification, read out.**
    1. **The lending test FAILS**: on the registered cells view the thin set
       falls (0.4242 → 0.3536, a paired **−0.0707 [−0.1356, −0.0035]**, the
       interval excluding zero) and the sign leg fails (only **2 of 5** signs
       positive: *CP rights - general* +0.2308, *Trafficking* +0.0411; three
       negative). The secondary view's opposite sign (+0.0580 [−0.0072,
       +0.1273], *unresolved*) is the lending's false positives: the five's
       predicted counts rise **69 → 118** (*Sexual and reproductive rights*
       12 → 29, *Freedom of assembly* 9 → 28, *Statelessness* 12 → 26).
    2. **The partner test fails on its letter**: the partner set's cells-view Δ
       is −0.0012 [−0.0270, +0.0241] — the point estimate is inside the −0.02
       bar, the interval's lower bound is not. *Detention* falls −0.1026
       [−0.2223, −0.0116]; the others are flat or slightly up.
    3. **The swap readout** (descriptive): the pair becomes confusable — the
       co-predictions rise from 0–5 to 6–21 (*Sexual and reproductive rights* +
       *Women's rights* 5 → 21; *Statelessness* + *Rights of the Child* 5 → 19;
       *Freedom of assembly* + *Detention* 0 → 9; *CP rights - general* +
       *International instruments* 0 → 6), while the partners' prediction counts
       fall slightly (*Women's rights* 136 → 130, *International instruments*
       94 → 88, *Rights of the Child* 117 → 116).
  - **The reading: the words are not the thin labels' missing ingredient.** The
    lending works mechanically and *one* label genuinely gains (*CP rights -
    general*, whose partner is the broad *International instruments*: 0.0000 →
    0.2308), but the partner's vocabulary is about the partner's topic: lending
    it makes the two look alike, the thin labels are predicted on 118 texts
    instead of 69, and the false positives cost more than the recall gains
    (*Sexual and reproductive rights* 0.2105 → 0.0556, *Statelessness* 0.4000 →
    0.2353, *Detention* −0.1026). What separates the pairs is **semantic (a
    sub-topic), not lexical** — the corpus's words for sexual and reproductive
    rights *are* its words for women's rights — so CS-38's "no vocabulary of
    their own" class is real and this change set shows why: the thin labels'
    evidence is *shared* evidence, and sharing it more does not make it theirs.
    The one exception is the pair whose partner is a *broad instrument* label
    rather than a near-synonym.
  - **The exit criteria.** The artifact holds the five arms, the label sets,
    the lending's provenance (the five pairs and their shares, the discount and
    the threshold) and both paired views ✓; the lending relation reproduces
    CS-38's five ✓; the table row is auto-appended ✓; the log section reads the
    numbers out against the three conditions ✓; the gate green ✓ (ruff, pytest,
    `tools/verify_upr_baseline.py` on the run).


---

## The distinctive lexicon (CS-45) — the pre-registration and the verdict

*Added on 2026-11-03, when the change set was executed: the live plan's §7.6
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone line).
The verdict and the numbers are the track log's §CS-45.*

### 7.6 The fit-side dictionary round (CS-42 … CS-45)

- [x] **CS-45 — retry the mined lexicon with a distinctive filter.** CS-35's
  label lexicon, but only the **distinctive/shared** words (CS-33's classes)
  enter the label text.

  **The mechanism.** CS-35's mined words are the label's top-10 PIFA words —
  and they carry the corpus's **register** ("ensure", "right", "system") beside
  its topic ("corruption", "asylum"). CS-33 measured what the classes are: a
  **distinctive** word sits in one or two of the labelled tables, a **shared**
  one in 3–14, a **generic** one in at least a quarter of them. The change: the
  label-side representation is built **from the non-generic words only** — the
  class is the map's own (a property of the word, not of a label), and the
  filter applies to the candidate pool **before** the top-K cut, so a label
  keeps up to K of its non-generic words (fewer when it has fewer) and the
  registers is dropped rather than the topic.

  **Why.** The mined coverage is the **best cell-level signal in the pack**
  (AUC **0.8109**, against the description's 0.7067 and the cosine's 0.7818),
  and the channel failed on **head-to-head redundancy** with the name and the
  description, not on signal (CS-35: the seen view cleared on one half, one
  bucket of three). The register words are exactly what the name overlap and the
  description already carry; the distinctive words are what no other label-side
  channel has.

  **The rule, fixed before the call.** The classes kept: ``distinctive`` and
  ``shared`` (CS-33's own definitions, the map's own thresholds); ``generic`` is
  dropped. The filter applies to **both** routes: a mined label's pool loses its
  generic words; an expansion label's matched words are filtered the same way,
  and a label left with **no** word falls back to its **name** (the route's own
  rule). K stays **10** (CS-35's), the expansion prefix **4**, the folds **5** —
  nothing else is tuned.

  **Where the code lives.** `text_processors/lexical/label_lexicon.py`:
  `build_label_lexicon` gains ``word_classes`` (the filter, applied to the
  candidate pool); the `LabelLexicon` records it. `config.py`:
  `LabelTextsConfig` (the `mined_words` variant's label text) and
  `UprLabelLexiconConfig` gain ``word_classes`` (``None`` = every class, the
  default, so every earlier artifact keeps its exact content). The command
  (`commands/upr.py::_upr_label_lexicon` and its per-fold cross-fit) passes it
  through; `commands/embed.py::_label_lexicon_from_train` does the same for the
  label artifact.

  **The runs.** Four, in the sweep
  `experiments_text_processing/upr_learned_lexicon_distinctive_sweep.py` (the
  section `## UPR — upr_learned_lexicon_distinctive` in the track's table):

  - `upr-labelembed-embeddinggemma-mined-distinctive` — the label artifact: the
    68 label texts from the **filtered** lexicon;
  - `upr-eval-fusionc-embeddinggemma-k10-c1-mined-distinctive` — the guard's
    reference for the filtered label side (the champion with those vectors);
  - **cell 1 (the primary — the feature half)**:
    `upr-analyze-learned-lexicon-distinctive-feat` — the pack's
    ``label_mined`` group, the champion's label side, the lexicon filtered
    (guard: `upr-eval-fusionc-embeddinggemma-k10-c1`);
  - **cell 2 (the query half)**: `upr-analyze-learned-lexicon-distinctive` —
    the **filtered mined label side** (the query half), groups
    `embedding + lexical + chunk_profile`, the lexicon filtered (guard: the
    filtered label side's reference run).

  **The comparison, fixed before the call.** The name+description channel's
  numbers are CS-35's own artifacts — `upr-analyze-learned-lexicon-catfeat`
  (the feature half) and `upr-analyze-learned-lexicon-desc` (the query half) —
  read by the same instrument CS-35 used
  (`tools/compare_upr_arms.py`, the per-text counts of the two runs, 2,000
  resamples, seed 42; the "seen" view is the head, the torso and the tail
  pooled). The **guard**: `upr-analyze-learned-lexicon-catfeat` is re-run and
  must reproduce its artifact **field for field** (the wall clock aside), so the
  head-to-head is against a contemporaneous artifact. The new channels' gains
  over the **control** (`upr-analyze-learned-lexicon-control`) are reported
  beside them (CS-35's own readout), and the filtered column's **AUC** over the
  val grid is read from the artifact's feature diagnostics.

  **The shape of the number.** The headline is the **head-to-head on the seen
  view** (the filtered channel's arm minus the name+description arm, paired by
  text), read per half; the three seen buckets get their own paired intervals.
  The pooled and unseen numbers and the shortlist R@50 are reported beside it;
  the unseen bucket is **not** a bar (the mined route has no words for the eight
  and the filtered route less).

  **The falsification, registered in advance.** The CS-35 bar, unchanged: the
  head-to-head must be **positive with an interval excluding zero on the seen
  view** and on **at least two of the three** seen buckets. The **verdict is
  cell 1's** (the feature half — the half whose bar CS-35 came closest to);
  cell 2 is read the same way and reported beside it. If neither clears it, the
  channel stays out of the catalog (CS-35's verdict holds); if one clears it,
  the channel is a candidate for the catalog and the reviewer's decision.

  **The guard.** The pack is fitted on train and the words come from train, so
  the fit side's label side is **cross-fit** over 5 folds (CS-35's rule), and
  the filter rides into it: each fold's lexicon is built from **that fold's**
  map, filtered by that map's own classes. The runs whose label side is not
  train-derived are not cross-fitted (the feature half included: its label side
  is the champion's).

  **The cost.** Four runs, about 2–3 minutes each for the analyze ones (cache
  hits: the filtered label vectors are the only new embeddings, 68 texts), plus
  the catfeat re-run; **no model calls**, no new dependency; about half a day
  including tests and docs.

  **What is NOT registered.** K other than 10; a **class-combination sweep**
  (only distinctive+shared; the distinctive-only and the generic-only variants
  are unmeasured); the expansion route as a separate arm (it is filtered, not
  re-designed); the combination with the mask/description/borrow channels (one
  thing at a time); the **map classifier** as the instrument (the pack is, as in
  CS-35); **test**.

  **The exit criteria.** The four artifacts hold their arms; the filtered
  lexicon's provenance (the classes kept, the per-label words, the labels that
  fell back to their name) is in the label artifact and the lexicon block; the
  catfeat re-run reproduces its artifact field for field; the table rows are
  auto-appended; the log section §CS-45 reads the head-to-head out against the
  bar; the checkbox is ticked; the block is moved verbatim to the archive; the
  gate is green (ruff, pytest, `tools/verify_upr_baseline.py` on the analyze
  runs); one commit.

  *(pre-registered 2026-11-03; **executed 2026-11-03** — the runs are
  `upr-analyze-learned-lexicon-distinctive-feat` and
  `upr-analyze-learned-lexicon-distinctive` (the artifacts
  `results/upr-analyze-learned-lexicon-distinctive{,-feat}/analysis.json`), the
  label artifact is `upr-labelembed-embeddinggemma-mined-distinctive` with its
  guard `upr-eval-fusionc-embeddinggemma-k10-c1-mined-distinctive`, the sweep is
  `experiments_text_processing/upr_learned_lexicon_distinctive_sweep.py` (the
  label `upr_learned_lexicon_distinctive`), the log section is `## CS-45 — the
  distinctive lexicon: the register was the signal` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The numbers, in one screen (UPR **val**: 660 texts, `top_k` k=2, the
  gradient-boosted pack; the filtered lexicon keeps the distinctive/shared words
  only, K = 10, the expansion prefix 4, the folds 5; no model calls; 50.4 s +
  90.1 s for the analyze runs).**

  - **The filtered lexicon**: 60 labels mined, **4 expanded**, **4 fallback**
    (*ESC rights - general*, *Gender-based violence*, *General*, *Other* — their
    names' words are all generic). The mined words are topical now: *Access to
    justice* → "fair trial barriers governance reparation impartial aftermath
    backlogs choice choose".
  - **The guard**: the CS-35 sweep was re-run in full and all **seven**
    artifacts reproduce **identically** apart from the wall clock.

  - **The arms**:

    | the arm | micro-F1 | macro-F1 | head | torso | tail | unseen | shortlist R@50 |
    |---|---|---|---|---|---|---|---|
    | **cell 1 — the distinctive feature half** | **0.5023** | 0.5074 | 0.5571 | 0.4578 | 0.5104 | 0.4322 | 0.8557 |
    | **cell 2 — the distinctive query half** | **0.4985** | 0.4989 | 0.5722 | 0.4413 | 0.5461 | 0.3718 | 0.8842 |
    | the name+description features (CS-35's `catfeat`) | 0.5076 | 0.5077 | 0.5492 | 0.4559 | 0.5047 | 0.4824 | 0.8955 |
    | the name+description query (CS-35's `desc`) | 0.5084 | 0.4872 | 0.5949 | 0.4101 | 0.5121 | 0.4690 | 0.8864 |
    | the control (the plain pack) | 0.4818 | 0.4652 | 0.5505 | 0.4122 | 0.4531 | 0.4469 | 0.8909 |

  - **The head-to-head** (the filtered arm minus the name+description arm,
    paired by text, 2,000 resamples, seed 42):

    | the scope | the **feature** half | the **query** half |
    |---|---|---|
    | the **seen** view | +0.0037 [−0.0186, +0.0252] | +0.0064 [−0.0155, +0.0272] |
    | head | +0.0079 [−0.0223, +0.0365] | −0.0227 [−0.0514, +0.0074] |
    | torso | +0.0019 [−0.0348, +0.0387] | **+0.0312 [+0.0000, +0.0644]** |
    | tail | +0.0057 [−0.0504, +0.0608] | +0.0340 [−0.0353, +0.1031] |
    | unseen | −0.0503 [−0.1061, +0.0070] | −0.0973 [−0.1450, −0.0529] |
    | pooled | −0.0053 [−0.0250, +0.0146] | −0.0099 [−0.0296, +0.0091] |

  - **The falsification, read out.**
    1. **The feature half (the primary) does not win**: the seen view's interval
       includes zero (+0.0037) and **no** seen bucket clears (head +0.0079,
       torso +0.0019, tail +0.0057 — all include zero). **0 of 3.**
    2. **The query half does not win either**: the seen view includes zero
       (+0.0064); only the **torso** clears (+0.0312 [+0.0000, +0.0644]) and the
       **head falls** (−0.0227 [−0.0514, +0.0074]). **1 of 3.**
  - **The reading: the register was the signal.** The filter **halved** the
    channel's gain over the control (the seen view **+0.0272 [+0.0074,
    +0.0474]** and **+0.0339 [+0.0139, +0.0548]** against CS-35's +0.0507 and
    +0.0493) and took the column's cell-level AUC down to the description's own
    level (**0.7073**, train 0.7871 — against CS-35's 0.8109, train 0.8657, and
    the description's 0.7067). The register words were doing the cell-level
    separation: they are frequent enough to appear in many cells and enriched on
    the gold ones, while the distinctive words are topical but rare and cannot
    separate 44,880 cells. CS-33's readout (the distinctive words 100 % positive,
    +4.27; the generic 81 %, +0.84) is about the **direction** of a word's
    association, not about how well its **presence** covers the cells. Where the
    filter does help is the **torso and the tail** — the rare labels' signal —
    but the head refuses to clear and its loss kills the seen view. The
    **unseen** bucket falls in both halves (the feature −0.0503; the query
    −0.0973 [−0.1450, −0.0529]): the filter emptied four expansions, whose texts
    are now their bare names. So CS-35's redundancy reading is confirmed and
    sharpened: the mined channel's advantage over the description **was** the
    register — exactly the part the name and the description already carry.
  - **The exit criteria.** The four artifacts hold their arms ✓; the filtered
    lexicon's provenance (the classes, the per-label words, the fallbacks) is in
    the label artifact and the lexicon block ✓; the catfeat re-run reproduces
    its artifact field for field ✓ (the full CS-35 sweep, seven artifacts
    identical apart from the wall clock); the table rows are auto-appended ✓;
    the log section reads the head-to-head out against the bar ✓; the gate green
    ✓ (ruff, pytest, `tools/verify_upr_baseline.py` on the analyze runs).


---

## The kNN union (CS-46) — the pre-registration and the verdict

*Added on 2026-11-03, when the change set was executed: the live plan's §7.7
block moved here **verbatim** (the archive's convention: an executed change
set's pre-registration moves here and the live plan keeps one tombstone line).
The verdict and the numbers are the track log's §CS-46.*

### 7.7 The candidate-set union (CS-46)

**Why it is separate.** It is not a dictionary change: it is CS-41's retrieval
thread, and the evidence PLAN.md's deferred BM25 checkbox is waiting for.

- [x] **CS-46 — the union of the lexical and embedding kNN candidate sets.**
  Fuse the TF-IDF kNN and the embedding kNN **rankings** and measure the
  candidate generator's recall@10 and recall@50 (no model calls).

  **The mechanism.** Two retrievers reach nearly the same **breadth** (CS-41:
  the lexical kNN's shortlist R@50 **0.9437**, the embedding kNN's **0.9576**)
  and differ in their **failures**: the words miss what only the embedding
  geometry can see, the embedding misses what only the words can see. The
  **union** is the track's own fusion rule — the per-row **z-max** of the two
  score matrices (CS-14's rule, `text_processors/decision/upr_union.py`) — a
  **measurement, not a component**: it says how much breadth the two
  retrievers have **together**, and it is the number PLAN.md's deferred BM25
  decision is waiting for.

  **Why.** CS-41 left the question open: the fusion's own shortlist (0.7423,
  barely above the 0.7353 chance level) is the *worse* candidate generator,
  and the lexical route's shortlist floor is high (the label-blind arm 0.8111),
  so the payoff must be read on the **union** of the two candidate sets. The
  decision model's union already reaches **0.9120** at k = 10 (CS-14) — the
  bar this measurement is read against.

  **The rule, fixed before the call.** Both halves at **k = 10** (the
  champion's own kNN k — the like-for-like operating point CS-41 used), the
  machinery's default power (1.0), the TF-IDF knobs CS-29's/CS-41's (word 1-2
  grams, min_df 1, sublinear tf, no stop-word list); the union is the per-row
  z-max of the two score matrices, nothing else. The label-blind counterpart
  (the track's own yardstick) permutes the train records' label sets **once**
  and hands the same permuted labels to both halves, so the fusion's floor is
  measured on the same control as the arms'.

  **Where the code lives.** A new predictor
  `src/cluster_rag/predictors/knn_union.py` (the registry
  `src/cluster_rag/predictors/__init__.py` gains the name): it constructs the
  existing lexical kNN predictor (CS-41's) and the existing embedding kNN
  predictor (Phase 4's) **unmodified**, scores both, and returns the z-max of
  the two matrices. It is embedding-dependent (`REQUIRES_EMBEDDINGS = True`,
  `REQUIRED_KINDS = ("document", "query")`) because one half is, and its
  `label_blind` builds the permuted train context both halves read.

  **The runs.** Two `evaluate` runs on UPR **val** in
  `experiments/knn_union_sweep.py` (the sweep label `knn_union`):
  `upr-eval-knn-union-val` and `upr-eval-knn-union-blind-val`. `evaluate` is a
  **main-pipeline** command, so the rows land in the **main** UPR table
  (`docs/experiments_table_upr.md`, `## UPR — knn_union`) and the narrative in
  the **track log** (CS-29's/CS-41's shape). Artifacts
  `results/upr-eval-knn-union-val/metrics.json` and
  `results/upr-eval-knn-union-blind-val/metrics.json`.

  **The shape of the number.** The headline is the union's **recall@10**
  (the share of gold cells whose label is in the row's top 10) beside its
  **recall@50**; the two arms' own R@10 / R@50 (the lexical kNN 0.6326 /
  0.9437, the embedding kNN 0.6933 / 0.9576, both from their artifacts) and
  the **decision model's union** (0.9120 at k = 10, CS-14) sit beside it. The
  blind union's R@10 / R@50 and its micro-F1 are the fusion's empirical floor.
  The micro-F1 of the union's decisions is reported (it is not a classifier —
  the union's value is the candidate set).

  **The falsification, registered in advance.** The union's **recall@10 must
  be at least 0.9120** (the decision model's union at the same k). At or above
  it, the two lexical retrievers together supply the breadth the decision
  model's union supplies, and **PLAN.md's deferred BM25 decision gets its
  evidence**; below it, the shortlist gap is **not** a breadth gap — the
  fusion is not buying back what the individual retrievers already see — and
  the BM25 checkbox stays closed. The recall@50 is read beside it: the union
  should not fall below the better arm's 0.9576 (a fusion that dilutes the
  breadth is a null regardless of its recall@10), and a union at 0.9576 or
  above means the two retrievers are almost mutually redundant at that depth.

  **The cost.** Two `evaluate` runs, about half a minute each (cache hits: the
  embeddings are already cached, the TF-IDF is fitted on the train split);
  **no model calls**, no new dependency; about half a day including tests and
  docs.

  **What is NOT registered.** A k sweep (k = 10, the champion's own); a
  different fusion rule (only the z-max); a BM25 channel, a lexical term in
  the fusion or a fourth fusion parameter (the dependency and the wiring are
  the reviewer's); the union as a **component** (it is a measurement — CS-14's
  shape); the decision model's own union re-measured (CS-14's artifact is the
  reference); **test**.

  **The exit criteria.** The two artifacts hold the union's metrics (R@10,
  R@50, the micro-F1, the blind floor); the rows are auto-appended to the
  **main** UPR table; the log section §CS-46 reads the union out against the
  two conditions; the checkbox is ticked; the block is moved verbatim to the
  archive; the gate is green (ruff, pytest, and the CLI's own evaluate guard);
  one commit.

  *(pre-registered 2026-11-03; **executed 2026-11-03** — the runs are
  `upr-eval-knn-union-val` and `upr-eval-knn-union-blind-val` (the artifacts
  `results/upr-eval-knn-union{,-blind}-val/metrics.json`), the sweep is
  `experiments/knn_union_sweep.py` (the label `knn_union`), the log section is
  `## CS-46 — the kNN union: the two retrievers see the same labels` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The numbers, in one screen (UPR **val**: 660 texts, both halves at k = 10,
  the per-row z-max of the two existing predictors; no model calls; 0.3 s per
  run).**

  - **The candidate generators**:

    | the candidate generator | recall@10 | recall@50 | micro-F1 |
    |---|---|---|---|
    | the champion's fusion (the whole pipeline) | **0.7126** | 0.7423 | 0.5479 |
    | the lexical kNN (CS-41) | 0.6326 | 0.9437 | 0.4605 |
    | the embedding kNN | 0.6933 | 0.9576 | 0.5388 |
    | **the union (the per-row z-max)** | **0.6943** | **0.9548** | 0.5266 |
    | the union's **label-blind floor** | 0.2313 | 0.8029 | 0.0798 |
    | the decision model's union (CS-14) | **0.9120** | — | — |

  - **The falsification, read out.**
    1. **The breadth test FAILS**: the union's recall@10 is **0.6943**, far
       under the registered **0.9120** (the decision model's union at the same
       k) — by 0.22.
    2. **The breadth readout fails too**: the union's recall@50 is **0.9548**,
       below the better arm's **0.9576** — the z-max does not add breadth, it
       dilutes it by 0.003.
    3. **The decomposition**: at k = 10 the union is +0.0010 above the embedding
       arm and at k = 50 it is 0.003 below it; **the union is the embedding
       arm** — the lexical half adds nothing the embedding did not already see.
  - **The reading.** The two retrievers are mutually redundant at the candidate
    level (their top-10 and top-50 sets nearly coincide), so a third lexical
    channel (BM25) would not widen the candidate set. The decision model's
    0.9120 is not lexical breadth — it is the decision model's own ranking. And
    the fusion's shortlist gap is an **ordering** problem at depth 50: its
    recall@10 (0.7126) is the best of all five candidate generators while its
    recall@50 (0.7423) is the worst. **The BM25 checkbox stays closed**: the
    lexical route is strong (CS-41) but already covered. The label-blind floor
    (0.2313 / 0.8029) shows the label-driven breadth is +0.46 at k = 10 and
    +0.15 at k = 50 — real, but the absolute shortlist numbers are flattered by
    the frequencies.
  - **The exit criteria.** The two artifacts hold the union's metrics (R@10,
    R@50, the micro-F1, the blind floor) ✓; the rows are auto-appended to the
    **main** UPR table ✓; the log section reads the union out against the two
    conditions ✓; the gate green ✓ (ruff, pytest, the CLI's own evaluate
    guard); one commit ✓.

---

## The dictionary's own ranking (CS-47) — the pre-registration and the verdict

*Added on 2026-11-04, when the change set was executed: the live plan's §7.8 block moved here **verbatim** (the archive's convention: an executed change set's pre-registration moves here and the live plan keeps one tombstone line). The verdict and the numbers are the track log's §CS-47.*

### 7.8 The dictionary's own ranking (CS-47)

**Why it is separate.** The viewer already reads the corpus map's evidence
(CS-23's topic words, CS-33's word classes, CS-34's grammatical axis, CS-40's
contrast), but it never shows the **map's classifier**: the arm that sums the
positive z-scores of a text's words in each label's table and takes the top 2
(CS-31's `sum_zscore_positive` — the classifier that reaches micro-F1 0.5243).
The contrast block's B is that classifier's best non-gold label, but the reader
cannot see the ranking behind it. This change set puts the ranking on the
screen.

- [x] **CS-47 — the dictionary's own ranking: the classifier's answer.**
  The corpus map's classifier, on the screen.

  **The mechanism.** A **viewer's** block between the corpus map and the
  contrastive block: for the text on screen, every label whose table is not
  empty scores `map_score(rows, text_words)` — the sum of the positive z-scores
  of the text's words in its table (CS-31's arm, read from the artifact's own
  `corpus_map.per_label` rows; **the map is not rebuilt** and **no second
  scoring rule is introduced**) — and the ranking is that score, largest first.
  Each row carries at most two contributing words with their z-scores, and a ★
  for the text's gold labels. Empty tables are **omitted** from the ranking
  (never printed at the floor), and the footnote counts them (8 on the UPR
  map); a label that scores nothing is not a pick and is left out too. The ties
  break by the label space's own order, exactly as `default_contrast_label`
  breaks them. `--dict-top N` (default 10, minimum 1) sets how many rows the
  block shows. The block **says what it is**: a description of the data, never
  the model's reading. The **contrast block's B** gains its rank in this
  ranking ("the dictionary's rank N for this text"), so the two blocks use one
  score and cannot disagree.

  **The falsification, registered in advance.** A tool change must move **no
  number**: the block writes no artifact, re-writes no artifact, adds no
  dependency and changes no existing number. The block's scores must equal the
  artifact's recomputation (`map_score` over `corpus_map.per_label`), and the
  existing **strip-and-compare** test (the coloured output stripped of its codes
  equals the plain output, byte for byte) must still hold. If the block's
  numbers are not the artifact's own, or if it prints an empty table at the
  floor, it is **dropped**.

  **The cost.** **0 model calls** (the offline mode: the artifact + the split);
  the block itself is a table lookup (~0 s — the existing offline run is
  ~1.2 s because its stale-artifact guard rebuilds the map); **no new
  dependency**.

  **The exit criteria.** The block is on the screen after the corpus map and
  before the contrastive block; on the seed-7 text it prints the registered
  order — *Right to education* **2.631** (education +2.631) ★, *Minority
  rights* **2.464** (minorities +2.464) ★, *Right to development* **1.076**
  (socio +1.076), *Extrajudicial executions* **0.622** (addition +0.622),
  *Disability rights* **0.619** (aspects +0.619), *Detention* **0.618**
  (addition +0.618) — and "8 labels have no table"; B carries the same
  ranking's rank; `--dict-top` works and rejects 0; the tests are added (a
  `map_score` recomputation of the block's scores, a fixture test for the top-1
  label and the empty-table footnote, a `--dict-top` test, the guard still
  covers the block's tables, and the strip-and-compare test holds); the plan
  checkbox is ticked; the log section is appended; the pre-registration is moved
  verbatim to the archive; the gate is green (pytest + ruff over
  `src`/`tests`/`tools`). No run, no table row, no commit.

  *(pre-registered 2026-11-04; executed 2026-11-04 — a **tool** change: the
  viewer's dictionary block, the pure helper behind it
  (`text_processors/lexical/contrastive.py`'s `dictionary_ranking`), and the
  tests; **no run**, **no table row**, **no artifact re-written**, **no new
  dependency**. The log section is `## CS-47 — the dictionary's own ranking:
  the classifier's answer` in `docs/experiments_text_processing.md`, and the
  numbers are below.)*

  **The numbers, in one screen (UPR val: 660 texts, 68 labels, the train
  split's 1,676 texts; no model call, no embedding; 1.2 s).**

  - **The seed-7 text** (`--sample --seed 7`, the record
    `5d8cf9f161cde0408225a428`, gold labels *ESC rights - general*, *Labour
    rights*, *Minority rights*, *Right to education*): the block's top rows are
    *Right to education* **2.631** (education +2.631) ★, *Minority rights*
    **2.464** (minorities +2.464) ★, *Right to development* **1.076** (socio
    +1.076), *Extrajudicial executions* **0.622** (addition +0.622),
    *Disability rights* **0.619** (aspects +0.619), *Detention* **0.618**
    (addition +0.618), *Rights of the Child* 0.603 (addition +0.603), *Right to
    health* 0.430 (medical +0.430) — every score is `map_score` over the
    artifact's rows; **8 labels have no table** (*Asylum-seekers - refugees*,
    *ESC rights - general*, *Freedom of association and peaceful assembly*,
    *Freedom of religion and belief*, *Gender-based violence*, *General*,
    *Other*, *Special procedures*), and the text's other gold label, *ESC
    rights - general*, is one of them (omitted, not floored).
  - **The contrast pair** on the same text: A *Right to education*, B *Right to
    development* — **"the dictionary's rank 3 for this text"** — the same score
    the block's third row shows. On `--text 0000000000000047d5d150d7` the pair
    is A *Migrants*, B *Right to land*, **rank 4**, and the block's top three
    are the text's three gold labels.
  - **The falsification, read out.**
    1. **The no-moved-number test HOLDS**: no run, no table row, no new
       dependency and **no artifact re-written**; no file under `results/`
       changed (the tree holds the same 1,108 files before and after, and the
       map's own artifact keeps its sha256 `ba2d7037…` ✓).
    2. **The scores equal the artifact's recomputation** ✓: the test
       (`test_the_dictionary_scores_are_the_artifacts_own`) re-reads
       `corpus_map.per_label` and recomputes every displayed row with
       `map_score`, and the fixture tests pin the top-1 label and the
       empty-table footnote ✓.
    3. **The strip-and-compare test HOLDS** ✓ (the coloured screen, stripped,
       is the plain screen, byte for byte) ✓; the CS-25 heading rule and the
       palette rules are kept ✓.
  - **The reading.** The screen now shows the **classifier's own answer**, not
    only its best non-gold label inside the contrast: the first two rows are its
    top-2, and the word beside each row says which text word gave the points —
    the map's own explanation of its pick. The empty tables stay off the
    ranking rather than at the floor, so a label with no train example is never
    presented as a score of 0; and because the contrast block's B is read from
    the same ranking, the two blocks cannot disagree.
  - **The exit criteria.** All met ✓: the block is on the screen in its
    registered place ✓, the seed-7 order and "8 labels have no table" are
    printed ✓, B carries its rank ✓, `--dict-top` works and rejects 0 ✓, the
    tests are added ✓ (58 in the two test files), the plan checkbox is ticked ✓,
    the log section is appended ✓, the pre-registration is moved to the
    archive ✓, the gate is green ✓ (**912 passed**, 13 deselected; `ruff check`
    and `ruff format --check` clean over 191 files in `src`/`tests`/`tools`).

---

## The question map on OMCT (CS-48) — the pre-registration and the verdict

*Added on 2026-11-05, when the round's first change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-48.*

### 7.9 The question map (CS-48)

- [x] **CS-48 — the question map: the reviewed pairs' own word↔question
  association.** For every question and every word of its reviewed paragraphs:
  how strongly does this word go with this question? The OMCT twin of CS-23 —
  the plain statistics of the pair unit, with the counts behind every number.

  **The UPR verdict this ports (the prior), and its reason.** CS-23 (executed
  2026-11-01) was **not a null**: it is a *description* (the map enters no
  measurement on its own), and it is the substrate CS-30/CS-31 then scored. The
  reason it worked: the unit was **thick** — 1,676 train texts × 68 labels gave
  every (label, word) pair a document frequency spanning the corpus, so the
  0.5-smoothed log-odds could put a **topical** word at the head of a table
  (56 of the 60 labelled tables; the lift's head was a hapax for 55) while the
  **register** ("rights" in all 60 tables) sat at the other end.

  **The mechanism, and the unit (pinned here because it is the whole question).**

  - **The row is the reviewed pair.** The "text" is the pair's **paragraph**;
    the "label" is the pair's **question**; the word is *present in the pair's
    paragraph*. For question `q` and word `w` the 2×2 table is over `q`'s
    reviewed pairs: `a` = positive pairs whose paragraph contains `w`,
    `b` = negative pairs whose paragraph contains `w`, `c` = positive pairs
    whose paragraph does not, `d` = negative pairs whose paragraph does not;
    `n_q = a + b + c + d` is the question's reviewed-pair count and
    `df_q(w) = a + b` the word's frequency **among `q`'s pairs**. The map's
    columns are the **294 questions**, its rows the words of the reviewed set.
  - **Why this unit, and why not the alternatives.** It is the literal analog
    of CS-23's "text contains the word × text carries the label" (the pair's
    paragraph is the only text a pair has; the question is the only label a
    pair has). (i) *Pooling all questions into one table* would lose the
    question column and could not score a pair — rejected. (ii) *Keying the
    label by the question number* adds nothing: the number is unique on this
    file (294 numbers, 294 texts), but the **text** is the semantic unit the
    scorer will read — the text is the key. (iii) *Deduplicating paragraphs*
    would drop each pair's own label — rejected. **CS-21's word-pair view**
    (paragraph word × question word) is a **different object** — it measures
    the match between the two sides, not the association of a word with the
    question's outcome — and it is **not** this change set.
  - **What the unit costs, stated before the measurement.** A question has a
    **median of 3 reviewed pairs** here (not 1,676 texts), so the table is a
    thin object by construction: read-only on 2026-11-05, **85 of 294
    questions** (28.9%) have **no positive pair at all** (their tables are
    empty by construction, 331 pairs = 21.8%), **95 questions** (249 pairs)
    are **all-positive** (no negative to contrast), and only **114 questions**
    (940 pairs = 61.8%) carry **both classes**. That is the transfer's central
    fact, and this change set measures it instead of assuming it.

  **The measures** (every one stored with the counts behind it, so any number
  can be recomputed and audited):

  - `n_with` — the question's reviewed pairs that are **positive** and whose
    paragraph contains the word;
  - `df` — the question's reviewed pairs whose paragraph contains the word at
    all (positives **and** negatives);
  - `lift` = `P(positive | word present) / P(positive)` within the question —
    1.0 says nothing, > 1 pulls the question, < 1 pushes it away;
  - `log_odds` = `log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5))` — the
    0.5-smoothed signed version (the same formula as CS-23; it is preferred to
    the lift because the lift explodes on rare words).

  **No thresholds.** Every (question, word) pair that occurs is stored; the
  noise is handled at **display** time (the counts sit next to the number).
  The **estimator** (CS-50) is the next change set: its informative prior needs
  a **global** document frequency (the word's df over the 1,520 reviewed pairs),
  so this change set stores the per-word aggregate as well and does **not** add
  the prior here.

  **The metric.** CS-48 is a **description**: no AUC, no arm, no comparison
  (CS-49's). Its metric is the map's **shape** plus four readouts, all
  pre-registered: (1) the **measure readout** — for every question with a
  table, the head word under the lift and under the log-odds, and how often the
  two disagree; (2) the **thickness readout** — how many of a table's words
  occur in **two or more** of its positives (a word shared by two positives is
  evidence that generalizes; a word in one positive is a memory of that
  paragraph); (3) the **register readout** — how many questions' tables each
  word sits in (the generic class's raw material, ahead of CS-51); (4) the
  **availability readout** — under the 5-fold cross-fit (seed 42, the track's
  scheme), for each pair: does the other folds' map hold a table for its
  question (`avail_table`), and does the pair's paragraph hold at least one
  word with a **positive** log-odds there (`avail_evidence`)? That last number
  is the map's **usability** number: a pair that carries the floor cannot be
  ranked above anything.

  **What the shape is expected to be** (measured **read-only** on 2026-11-05
  with the track's own tokenizer, so the numbers are known before the run — the
  CS-23 precedent). The run must reproduce them:

  | | |
  |---|---|
  | the reviewed pairs / the questions / the paragraphs | **1,520** (524 +, 996 −) / **294** / **225** |
  | the distinct words in some table | **2,082** |
  | the (question, word) pairs stored | **27,979** |
  | the questions with a table / empty tables | **209 / 85** |
  | the per-question table size | min **0**, median **65**, max **477** |
  | the questions with both classes (the contrastable set) | **114** (940 pairs) |
  | the tables with a word shared by ≥ 2 positives | **101 of 209** (48.3%); all **37** of the tables with ≥ 4 positives |
  | the words shared by ≥ 2 positives, per table | median **0** over the 209 tables (median **19** over the questions with ≥ 2 positives) |
  | the words in ≥ a quarter of the 294 tables (≥ 74) | **57** ("state" in 147, "committee" 136, "information" 132) |
  | the head under the lift vs under the log-odds | differ for **96 of 209** (45.9%), **54 of 114** (47.4%) of the contrastable ones |
  | the log-odds' head word | occurs **only in the positives** for **206 of 209** (98.6%), and is a hapax (df = 1) for **110 of 209** (52.6%) |
  | the cross-fit availability (5 folds, seed 42) | a table for **1,002 of 1,520** pairs (65.9%), evidence for **928** (61.1%) — positives **373 of 524** (71.2%), negatives **555 of 996** (55.7%) |
  | the same, per document (NU / CIDH / AI) | table 60.0 / 70.8 / 87.9 %, evidence 56.2 / 65.5 / 76.8 % |

  One example table (read-only): the question *"Have there been any of the
  following cases of serious attacks to HRDs in your country?"* has **19
  reviewed pairs, 16 positive, 477 table words**; its log-odds head is
  *threats* (n_with 10, df 10, lift **1.19**, log-odds **+2.425**), then
  *defenders* (9/9, 1.19, +2.182), *leaders* and *journalists* (7/7, 1.19,
  +1.710), *women* and *indigenous* (6/6, 1.19, +1.466). Note the lift is
  **the same for every word that occurs only in the positives** (1.19 =
  19/16): on OMCT the lift's degeneracy is not only "it explodes on rare
  words" — a word's lift does not depend on **which** word it is.

  **The falsification conditions, registered — read out one by one.** The run
  is complete when all five are read out; a failed **1, 3, 4 or 5** is a
  null/stop, while **2 is a prediction** (the UPR measure finding is expected
  **not** to transfer):

  1. **The reproduction guard.** The run's shape must equal the read-only table
     above (the questions, the words, the pairs, the empty tables, the
     contrastable set). A difference means the tokenizer, the pair key or the
     reviewed-set definition is not the pre-registered one → **stop and
     report**, do not proceed.
  2. **The measure test — and it is registered to FAIL (the null that may
     reproduce).** On UPR the lift and the log-odds picked **different** head
     words for 55 of 60 labels (92%). Here the registered prediction is
     **≤ 60%** for the contrastable questions (the read-only read: 47.4%), so
     "the two measures rank differently" does **not** transfer: with a handful
     of pairs a word that occurs only in the positives has the same lift
     whatever it is, and the log-odds' head is a positives-only word for 98.6%
     of the tables. If the run finds **> 80%**, the transfer is stronger than
     predicted and the reading changes. (This is a reading of the unit, not a
     defect of the map.)
  3. **The thickness test.** At least **40%** of the 209 tables with a positive
     must hold a word that occurs in **≥ 2** of its positives (the read-only
     read: 48.3%), and **all 37** of the tables with **≥ 4** positives must
     (read-only: 37 of 37). If it fails, the table is a memory of one paragraph
     everywhere and the direct port is not an estimator — **report it as a
     null**.
  4. **The availability test (the branch).** At least **50%** of the 1,520
     pairs must have a table for their question out-of-fold, and at least
     **45%** must receive a positive word (read-only: 65.9% / 61.1%, per
     document above). **If it fails**, the cross-fit starves the per-question
     tables and the scorer (CS-49) would be measured on a minority of pairs →
     **stop and report before CS-49**; the reviewer's options are a different
     fold unit (e.g. folds over paragraphs) or CS-21's word-pair view instead.
  5. **The register test.** At least **one** word must sit in ≥ a quarter of the
     tables (read-only: 57). If not, CS-51's class machinery has nothing to
     separate and that change set must be re-scoped.

  **The nulls this change set can produce, stated so they are not read as
  defects.** (i) The measures' separation (condition 2). (ii) The
  per-question table as a *description* that is **not** an estimator (condition
  3). (iii) The availability (condition 4). All three are legitimate results of
  the transfer test, and each is reported with its number.

  **The cost.** **0 model calls** (the run's cosine baseline is a cache hit —
  225 paragraphs + 294 questions). The map plus the five cross-fit builds is
  pure numpy/stdlib work (seconds); the artifact grows by the map's **27,979**
  rows (a few MB). No new dependency.

  **Leakage.** The map is built from the **reviewed pairs only** (the 1,520
  judged pairs); the **4,000 unjudged control cells never enter it** and are
  never a label. There is no val/test split on OMCT: the reviewed set is the
  only labelled set, which is exactly why the cross-fit is the scoring
  discipline (CS-49) and why the control role is the instrument. The map is
  **model-independent** (it never reads an embedding). The reviewed set is
  **rank-selected** (its negatives are exp18's top false positives), so the map
  describes the reviewed set and **not** the corpus; the control cannot correct
  that and is not a label. No UPR data and no UPR artifact is touched.

  **The runs** (fixed before the first call): the run id
  **`omct-analyze-arctic2-dictmap`** (the analyze command adds the `omct-`
  prefix; the baseline embedder is arctic2, a warm Stage-0 model — the map
  block itself is model-independent and appears unchanged in every later run);
  the artifact **`results/omct-analyze-arctic2-dictmap/analysis.json`** (the
  new top-level block `dictionary`); the sweep cell in a new sweep file
  `experiments_text_processing/omct_dictionary_sweep.py` (the sweep label
  `omct_dictionary`, so the table's section is `## OMCT — omct_dictionary`);
  the config schema change is `analysis.dictionary` (a new optional section:
  `include_map`, `folds` = 5, `seed` = None → the run's 42); and the log
  section **`## CS-48 — the question map`** in
  `docs/experiments_text_processing.md`.

  **The exit criteria.** The counts / lift / log-odds pinned by a
  **hand-computed 2×2 unit test** (and counts that do not form a 2×2 table are
  **refused**, never silently smoothed); a test that the map accepts the
  reviewed pairs and nothing else (the control cells cannot be handed in); a
  test of the empty-table case (a question with no positive has no table, and
  the availability readout marks its pairs); a test of the fold/availability
  determinism (the same seed reproduces the same folds); the fixture run and
  the real run; `ruff check` / `ruff format` clean over
  `src`/`tests`/`experiments_text_processing`; the full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-05** — the run is
  `omct-analyze-dictmap` (the artifact `results/omct-analyze-dictmap/analysis.json`,
  the block `dictionary`), the log section is `## CS-48 — the question map on
  OMCT` in `docs/experiments_text_processing.md`, and the numbers are
  below.)*

  **The round's numbers, in one screen (OMCT: the 1,520 reviewed pairs over 225
  paragraphs in 3 documents; the map is built on subset A only, and the
  unjudged control (B) never enters it).**

  - **The cost.** **0 model calls**, **0.5 s**, the artifact **5.4 MB**, **no
    new dependency**; the run carries **no embedder** (the map is a
    description, not a model).
  - **The shape.** **294** questions, **2,082** words, **27,979** (question,
    word) pairs; **209** questions with a table and **85** empty by
    construction (28.9 %); the table size min **0**, median **65** (119 over
    the 209 with a table), max **477**; **114** contrastable questions (940
    pairs), **95** all-positive (249 pairs) and **85** all-negative (331
    pairs).
  - **The measures readout — the null that reproduces.** The lift and the
    log-odds pick different heads for **96 of 209** tables (45.9 %) and **54
    of 114** contrastable ones (47.4 %), against UPR's **55 of 60** (91.7 %):
    the UPR separation does **not** transfer. The log-odds' head is a
    positives-only word for **206 of 209** (98.6 %) and a hapax for **110 of
    209** (52.6 %).
  - **The thickness.** **101 of 209** tables (48.3 %) hold a word shared by
    two positives; the median table holds **0** (19 among the 102 questions
    with ≥ 2 positives). **All 37** tables with ≥ 4 positives hold one;
    **none of the 107** single-positive tables does.
  - **The register.** **57** words sit in at least a quarter (74) of the
    tables — *state* 147, *committee* 136, *information* 132, *violence* 127,
    *human* 121, *rights* 117.
  - **The availability (5 folds over the pairs, seed 42).** A table for
    **1,002 of 1,520** pairs (65.9 %) and evidence for **928** (61.1 %);
    **518 pairs (34.1 %)** carry the floor. Positives **409/524** (78.1 %) /
    **373** (71.2 %); negatives **593/996** (59.5 %) / **555** (55.7 %). Per
    document (NU / CIDH / AI): **60.0 / 70.8 / 87.9 %** tables, **56.2 / 65.5
    / 76.8 %** evidence.
  - **The example.** The HRD question (19 pairs, 16 positive, 477 words):
    *threats* 10/10, lift **1.1875**, log-odds **+2.4255**; *defenders* 9/9
    +2.1823; *journalists* and *leaders* 7/7 +1.7095; *afro* and *agreement*
    6/6 +1.4663. The lift is identical for every positives-only word (19/16)
    and its head is the numeric token "025"; the pre-registration's 5th/6th
    rows were a tie broken differently by the read-only script — the counts,
    the lift and the log-odds are identical, only the order inside the tie
    differs.
  - **The falsification.** 1 holds (the shape reproduces exactly); 2 holds as
    predicted (47.4 % ≤ 60 %); 3 holds (48.3 % ≥ 40 %, 37 of 37); 4 holds
    (65.9 % ≥ 50 %, 61.1 % ≥ 45 %) — **the branch does not fire, so CS-49
    proceeds with the same fold scheme**; 5 holds (57 words).
  - **The gate.** **964** tests, ruff clean over 232 files, the run
    reproducible field for field; the baseline verify tool does not apply (a
    model-free run has no baseline arm).

---

## The map as a pair scorer on OMCT (CS-49) — the pre-registration and the verdict

*Added on 2026-11-05, when the round's second change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-49.*

### 7.9 The map as a pair scorer (CS-49)

- [x] **CS-49 — the map as a pair scorer: does the dictionary beat the plain
  cosine on the reviewed set?** Score every reviewed pair by summing the
  question's map weights over the paragraph's words, behind the same cross-fit
  as CS-48, and read the **within-document AUC** against the plain full-text
  cosine of the same run — the pooled AUC beside it, the **control role**
  beside that, and the **label-blind counterpart** as the label-driven check.

  **The UPR verdict this ports (the prior), and its reason.** CS-30 (executed
  2026-11-01) was **not a null**: the corpus map as a classifier reached
  micro-F1 **0.5243** (CS-31's z-scored arm; CS-30's own **0.4924**), 90 % of
  the champion's 0.5479, with **no embedding and no model call**. The reason it
  worked was the map's **measure work** — the log-odds weighting bought
  **+0.2226** of micro-F1 over a plain word count (0.2698 → 0.4924) — and the
  **cross-fit** was the instrument that showed it (the in-fold number would
  have said the measure work was worth 0.0227; the out-of-fold one said
  0.2780). On OMCT the same object is **thin** (CS-48: 209 tables / 85 silent,
  47.4 % of the contrastable tables where the lift and the log-odds disagree,
  101 of 209 tables with a word shared by two positives), so the honest prior
  is a *weak* scorer, not a copy of the UPR number.

  **The mechanism.** For a cell (paragraph `p`, question `q`) the score is a
  sum over the words of `p` that appear in `q`'s table:

  - `sum_positive` (the **primary** arm, pre-registered **a priori**): the sum
    of the **positive** log-odds of `p`'s words in `q`'s table — the "evidence
    for" reading, the arm the UPR fit side selected in CS-30. It is fixed here
    by the UPR prior, **not** selected on this run's numbers (the four-arm
    spread is the ablation beside it);
  - `sum_logodds`, `max_logodds`, `support_count`: CS-30's other three arms,
    reported as the spread, never as the headline.

  The map is built on the reviewed pairs with the **same 5-fold cross-fit and
  the same seed as CS-48** (folds over the pairs; each fold's map from the
  other folds), so the judged pairs are scored by a map that never saw them.
  A judged pair whose question has **no table** in its fold's map carries the
  row's floor (the row's smallest seen score minus one) and can never be
  ranked first — CS-48's availability (65.9 % of the pairs with a table,
  61.1 % with evidence, 518 pairs on the floor) is the cost, and every number
  below is reported over **all 1,520 pairs** (the track's convention), with a
  **secondary reading over the 986 pairs that have a table** beside it.

  **The reference.** The **plain full-text cosine of the same run** — the
  Stage-0 baseline arm, `voyage-4-nano` (the only Stage-0 model whose backend
  is HF, since no Ollama server is available here; the map's own numbers do
  not depend on the model). Its within-document AUC on these 1,520 pairs is
  **0.5238** (the pooled 0.5004; `results/omct-analyze-voyage-4-nano-stage35-grid`).
  The comparison is paired by construction: the same pairs, the same labels,
  the same documents, a paired bootstrap over the pairs.

  **The metric (the headline and its companions).**

  1. **Within-document AUC**, the primary arm against the cosine: the paired
     bootstrap's 95 % interval of the difference (the bootstrap resamples the
     **pairs**; the within-document AUC is the pair-count-weighted mean of the
     three documents' own AUCs).
  2. **The pooled AUC** beside it (the pooled number is never reported alone).
  3. **The control role**: the control cells' score level and rank-position
     shift (a method that only re-scales shows up here), the **floor gap** (the
     verified positives' mean minus the control's mean) and the **de-shifted
     AUC** (the control's per-document mean shift subtracted — the
     discrimination must survive it). The control cells never entered any map;
     they are scored by the **full reviewed-set map**, while the judged pairs
     are scored out-of-fold — a vintage difference (5/5 vs 4/5 of A) stated in
     the artifact, with the per-fold control-mean spread recorded so the reader
     can see its size.
  4. **The label-blind counterpart**: the same pipeline with the **questions
     permuted across the pairs** (one seeded permutation, the per-question
     counts preserved — the UPR "blind map" analog). The arm's within-document
     AUC minus the blind map's is the label-driven part.
  5. **Per document** (NU / CIDH / AI) as well as pooled; the AI stratum is
     small (99 pairs).
  6. **The floor**: the number and share of pairs on the floor (the primary
     arm's fold maps), beside the secondary reading on the pairs that have a
     table.

  **The falsification conditions, registered — read out one by one.**

  1. **The flip (the primary test).** "It works" = the primary arm's
     within-document AUC is **above the plain cosine's** with the paired 95 %
     interval **excluding zero**. If the interval includes zero → **a null that
     matches the cosine** (the dictionary does not beat the plain full-text
     geometry on this unit; reported as a null, not as a failure). If it is
     below with the interval excluding zero → the map is **worse**; reported
     with its number and its reason.
  2. **The label-driven test.** The primary arm's within-document AUC minus the
     **blind** map's must be ≥ **0.05**; below that the gain is not certifiably
     label-driven and no claim is made on it.
  3. **The control-role test.** The de-shifted AUC must be **above the cosine's
     within-document AUC** too — otherwise the raw difference is a scale shift,
     not a ranking gain.
  4. **The cross-fit test.** The arm's fold maps must not be starved beyond
     CS-48's readout: the floor share must be **within 3 points** of CS-48's
     34.1 % (a larger share means the arm's implementation reads a different
     map than the one CS-48 measured).
  5. **The per-document test.** The flip must not be one document's: the
     within-document AUC must be above the cosine's in **at least two of the
     three** documents (the per-document intervals are reported, not
     required to exclude zero — the AI stratum is 99 pairs).

  **The nulls this change set can produce.** (i) The primary arm matching the
  cosine (condition 1's middle branch) — the most likely outcome given CS-48's
  thickness. (ii) The blind gap below 0.05 (condition 2) — the scorer would
  then be a re-scaling of the question's vocabulary, not label information.
  (iii) The de-shifted AUC at the cosine (condition 3). All three are reported
  as results of the transfer test.

  **The cost.** **0 new model calls** (the cosine baseline is an embedding-cache
  hit on the Stage-0 voyage run's 225 paragraphs + 294 questions; at worst 519
  re-encodes on the HF backend), a few seconds of grid scoring (6 maps × 4
  scorings, a vectorized sum per paragraph), and the label-blind counterpart
  doubles the grid work. No new dependency; the artifact grows by the four
  arms' readouts and the comparison blocks (not by a matrix: only the arm
  blocks and the per-pair summary rows are stored).

  **Leakage.** The map is built on the reviewed pairs only, the same as CS-48;
  the control (B) never enters any fit; the blind arm's permutation is seeded
  and applied to the **questions only** (the paragraphs and labels stay in
  place); no UPR data and no UPR artifact is touched; the map's arms are
  scored by the out-of-fold map, so no pair's own label is in its table; the
  baseline cosine is the same run's own arm, not a stored artifact.

  **The runs.** One: **`omct-analyze-voyage-4-nano-dict`** (the analyze
  command adds the `omct-` prefix; the reference is voyage-4-nano's own
  full-text cosine), the artifact
  **`results/omct-analyze-voyage-4-nano-dict/analysis.json`** (the `dictionary`
  block of CS-48, reproduced, plus a `kind: dictionary` model entry with the
  four arms, their comparisons and a `dictionary_arm` readout block); the
  sweep cell in a **new** sweep file
  `experiments_text_processing/omct_dictionary_arm_sweep.py` (the sweep label
  `omct_dictionary_arm`, so the table's section is
  `## OMCT — omct_dictionary_arm` — the map's own section keeps its columns);
  the config schema change is `analysis.dictionary.scorings` (the four arms)
  and `analysis.dictionary.blind` (the counterpart, on); and the log section
  **`## CS-49 — the map as a pair scorer`** in
  `docs/experiments_text_processing.md`.

  **The exit criteria.** The scoring matrix pinned by a **hand-computed unit
  test** (a 2×2 cell grid whose sums, max and count are worked out by hand;
  the empty table's floor below every seen column); a test that the cross-fit
  reads the folds it is given (a pair's own fold's map never scores it); a
  test that the blind permutation preserves the per-question counts; the
  fixture run and the real run; `ruff check` / `ruff format` clean over
  `src`/`tests`/`experiments_text_processing`; the full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-05** — the run is
  `omct-analyze-voyage-dict` (the artifact
  `results/omct-analyze-voyage-dict/analysis.json`, the `kind: dictionary`
  model entry; the pre-registration named the run
  `omct-analyze-voyage-4-nano-dict`, the cell label is `analyze-voyage-dict`
  — the same single-model run, no number differs), the log section is
  `## CS-49 — the map as a pair scorer on OMCT` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (OMCT: the 1,520 reviewed pairs; the map
  is built on subset A only and the unjudged control (B) never enters it; the
  reference is voyage-4-nano's own plain full-text cosine).**

  - **The cost.** **0 model calls**, **10.8 s**, the artifact **9.0 MB** (the
    CS-48 map block is reproduced inside it), **no new dependency**.
  - **The pre-registered test (the primary arm `sum_positive`): a null that
    matches the cosine.** Within-document AUC **0.5419** against the cosine's
    **0.5238**, a paired **+0.0181 [−0.0267, +0.0630]** (P(Δ>0) 0.795); the
    pooled **0.5370** against **0.5004** (+0.0366); the de-shifted AUC
    **0.5429**; the control's rank p50 Δ **+0.0046** (not a re-scaling).
  - **The label-driven check holds**: the blind map's within-document AUC
    **0.4540**, the gap **+0.0879** ≥ 0.05 (per document +0.0685 / +0.0772 /
    +0.1204).
  - **The floor carries the edge**: **518 pairs (34.08 %)** — 403 of the 996
    negatives against 115 of the 524 positives (77.8 % negative) — and the
    **non-floor** reading (1,002 pairs) is **0.4439**, below chance. The
    positives' hit@1 is **0.0076** against the cosine's 0.1050.
  - **The per-document read**: above the cosine in **2 of 3** — NU +0.0194,
    CIDH +0.0206, AI **−0.2259** (the 99-pair stratum collapses: 0.3905 against
    0.6165).
  - **The spread (observations, measured in the artifact, not registered
    tests)**: `support_count` **0.6360** (+0.1122 [+0.0672, +0.1584], blind gap
    +0.2328, non-floor **0.6645**) — the only arm that beats the cosine;
    `sum_logodds` 0.5131 (−0.0107 [−0.0567, +0.0328], non-floor 0.4029);
    `max_logodds` 0.4967 (−0.0271 [−0.0705, +0.0168], non-floor 0.3389).
  - **The reading**: on UPR all 68 labels shared one corpus, so the log-odds
    lived on one scale and the weighted sum could rank labels (the measure work
    was worth +0.2226; the count was the floor at 0.2698). On OMCT each question
    has its own corpus (`n_q` 1–51, median 3), so the log-odds' magnitude is a
    **per-question scale** and the pooled ranking compares incommensurable
    magnitudes; the **count** is scale-free and is the arm that carries the
    signal. The UPR's measure lever **inverts**.
  - **The falsification, read out.** 1 (the flip) **fails** — a null that
    matches the cosine; 2 (label-driven) **holds** (+0.0879 ≥ 0.05); 3 (the
    de-shifted AUC above the cosine) **holds** (0.5429 > 0.5238); 4 (the floor
    within 3 points of CS-48's 34.1 %) **holds** (34.08 %); 5 (≥ 2 of 3
    documents) **holds**, with the AI stratum's collapse reported.
  - **The gate.** **970** tests, ruff clean over 236 files, two consecutive runs
    bit-identical (a set-iteration float-order bug was fixed: the token sum now
    runs over a sorted order); the baseline verify tool does not apply to a
    non-embedding arm.
  - **The deviation.** The run id is `omct-analyze-voyage-dict`; the
    pre-registration named `omct-analyze-voyage-4-nano-dict` (the cell's label
    is `analyze-voyage-dict`). The same single-model run; no number differs.

---

## The estimator on OMCT (CS-50) — the pre-registration and the verdict

*Added on 2026-11-05, when the round's third change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-50.*

### 7.9 The estimator (CS-50)

- [x] **CS-50 — the estimator: does the informative prior/z-score rescue the
  weights?** CS-31's measures on the pair unit (the informative-prior log-odds,
  the standard error, the z-score, G², the conditional smoothing), the two
  z-scored arms, and the decisive test CS-49's reading opened: **does the
  z-score make the weights rankable across questions, or is it a per-question
  rescaling that cannot?**

  **The decision taken here (the CS-49 finding re-scoped the change set, and
  the reviewer's "use what you think is best" covers it).** CS-49's reading is
  that the log-odds' magnitude is a **per-question scale**: each question's
  table has its own `n_q` (1–51, median 3), so a summed log-odds compares
  incommensurable magnitudes across questions, and the count — scale-free — is
  the only arm that beat the cosine (0.6360; the weighted arms' non-floor
  readings were 0.34–0.44, at or below chance). The estimator is therefore
  ported **as the pre-registered null test it now is**: the z-score is computed
  **within** a question's own table (the prior is the word's global df, the
  standard error is the same table's), so it can shrink a question's noisy
  weights but **cannot** make two questions' weights comparable. The
  pre-registration says so, and registers what would refute it.

  **The UPR verdict this ports (the prior), and its reason.** CS-31 (executed
  2026-11-01) was **a gain**: the z-scored arm moved the classifier from
  0.4924 to **0.5243** (a paired interval excluding zero), and the plan's
  head-word null did not fire (13 of 60 heads changed). The reason: the
  informative prior shrinks the **common** words (the register: "rights" is in
  all 60 tables) and the standard error the **rare** ones — a **model** gain,
  not only a description, because every label's table lived on the **same**
  corpus (1,676 texts) and the corpora were the same corpus.

  **The mechanism.** The six measures, the same formulas as CS-31, on the
  (question, word) table: `n_texts = n_q`, `n_label = n_pos(q)`,
  `df = df_q(w)` (the question's own frequency), with the **prior's** df the
  word's **global** df over the reviewed set (the map's `word_df`) and
  `α₀ = Σ_w (df_global(w) + 0.01)` — the map-level constant the reference
  implementation uses. The two arms are CS-31's: `sum_zscore_positive` (the
  "evidence for" score, **the pre-registered primary** — the UPR CS-31
  fit-side selection, ported as the prior) and `sum_zscore` (the signed sum,
  the control that collapsed to 0.1011 on UPR). The five CS-30 arms stay in
  the readout as the spread, CS-49's numbers reproduced exactly.

  **The falsification conditions, registered — read out one by one.**

  1. **The model test (the flip, and it is registered as a NULL).** The z-arm's
     within-document AUC and its **non-floor** reading (1,002 pairs) must be
     **above chance** for the estimator to transfer; the registered prediction
     is **≤ 0.50 on the non-floor reading** and **within CS-49's weighted
     spread (0.49–0.56)** on the within-document number, i.e. the z-score
     rescales within a question and does not fix the across-question scale.
     **If the non-floor reading is above 0.55 or the within-document AUC
     exceeds the count's 0.6360, the reading is refuted** and the estimator
     becomes the finding.
  2. **The head-word test.** The z-scored head must **differ** from the
     log-odds' for fewer than **50 %** of the tables (the prediction: the
     global-df prior swamps a 1–3-pair table, so the z-head is essentially the
     most frequent word and the table's order barely moves). UPR's number was
     13 of 60 (21.7 %). If the run finds **> 70 %**, the prior is not swamping
     the counts on this unit and the reading changes.
  3. **The count's persistence (CS-49's finding, re-read in this run).** The
     measure-free count must stay **the best arm** (its within-document AUC
     above every weighted arm's, as in CS-49); if a z-arm beats it, the
     estimator does what CS-49 said it could not.
  4. **The label-driven and control-role checks** (CS-49's 2 and 3, ported):
     the z-arm's blind gap ≥ 0.05 and its de-shifted AUC above the cosine's
     (0.5238) — reported, with the same interpretation.
  5. **The floor check** (CS-49's 4): the floor share within 3 points of
     34.1 %.

  **The cost.** **0 new model calls** (the voyage embeddings are the Stage-0
  cache hits); the six measures are computed inside the map build (27,979
  pairs, seconds), the two extra arms add two grid scorings (~2 s), and the
  spread comparisons are the same four plus two. No new dependency; the map
  block grows by the new measures (the CS-48 numbers stay in the same rows,
  beside them — CS-31's convention).

  **Leakage.** Unchanged from CS-48/CS-49: the map is built on the reviewed
  pairs only, the control (B) never enters any fit, the judged cells are
  scored out-of-fold, the blind arm permutes the questions only, and no UPR
  data is touched.

  **The runs.** One: **`omct-analyze-voyage-dict-z`** (the same reference
  cosine as CS-49, voyage-4-nano's own), the artifact
  **`results/omct-analyze-voyage-dict-z/analysis.json`** (the `dictionary`
  block with the six measures and the estimator readout, plus the
  `kind: dictionary` model entry with the six arms and their comparisons); the
  sweep cell in a **new** sweep file
  `experiments_text_processing/omct_dictionary_estimator_sweep.py` (the sweep
  label `omct_dictionary_estimator`, so the table's section is
  `## OMCT — omct_dictionary_estimator`); the config additions are
  `analysis.dictionary.scorings` (the two z arms, validated against the six
  known arms) and `analysis.dictionary.primary` (the pre-registered primary
  arm, so CS-49's config keeps its own default); and the log section
  **`## CS-50 — the estimator on OMCT`** in
  `docs/experiments_text_processing.md`.

  **The exit criteria.** The measures pinned by a **hand-computed unit test**
  (the informative log-odds with a separate prior df, σ, the z-score, G² and
  the conditional smoothing on a 4-cell table); a test that the prior's df and
  the table's df are read separately; a test that the z arm's scores use the
  z-score (not the log-odds); the fixture run and the real run; `ruff check` /
  `ruff format` clean over `src`/`tests`/`experiments_text_processing`; the
  full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-05** — the run is
  `omct-analyze-voyage-dict-z` (the artifact
  `results/omct-analyze-voyage-dict-z/analysis.json`, the `kind: dictionary`
  model entry), the log section is `## CS-50 — the estimator on OMCT: the UPR
  gain reproduces` in `docs/experiments_text_processing.md`, and the numbers
  are below.)*

  **The round's numbers, in one screen (OMCT: the same 1,520 reviewed pairs,
  4,000 control cells and 5-fold cross-fit as CS-49; the reference is
  voyage-4-nano's own full-text cosine).**

  - **The cost.** **0 model calls**, **14.5 s**, the artifact **15.1 MB** (the
    six measures on 27,979 rows), **no new dependency**.
  - **The registered null is REFUTED — the estimator transfers.** The primary
    `sum_zscore_positive`: within-document AUC **0.7020** against the cosine's
    **0.5238**, a paired **+0.1782 [+0.1303, +0.2272]** (P(Δ>0) 1.000); the
    **non-floor** reading (1,002 pairs) **0.8158**; the pooled **0.7485**
    (+0.2481); the de-shifted AUC **0.7334**; the per-document deltas
    **+0.1192 / +0.1816 / +0.2766** (all three above the cosine; the AI stratum
    that collapsed in CS-49 is now the best); the blind gap **+0.2340** (the
    blind map 0.4679); the floor **34.08 %**.
  - **The signed arm catches up**: `sum_zscore` **0.7023** (+0.1785
    [+0.1297, +0.2282], non-floor 0.8150, blind 0.5713, gap +0.1309) — **92.2 %**
    of the z-scores are positive (min −0.6378, median +0.0661, max +0.8102), so
    the signed and the restricted sums nearly coincide (on UPR the signed arm
    collapsed to 0.1011).
  - **The head-word test is refuted in the other direction**: the z head
    differs from the log-odds' for **177 of 209** tables (**84.7 %**), the
    informative head for **188** (90.0 %), the conditional for **1** (0.5 %);
    `α₀ = 97,460.14`. The pre-registration predicted **< 50 %** (the prior would
    swamp a small table); the "> 70 %" falsifier fired — the prior **reorders**
    the table, and the reordering is the gain.
  - **The reading that replaces CS-49's**: both δ denominators carry `α₀`, so δ
    reduces to a **per-word** ratio at the word's own frequency scale and σ is
    the same word's; every term is indexed by the **word**, not by the
    question's `n_q`, so the table's size drops out and the summed weights become
    **comparable across questions**. The common words get `α_w ≫ a, b`, hence
    `δ ≈ 0` and `z ≈ 0` (coverage, not direction); the rare topical words carry
    the z — a **soft** version of CS-33's class restriction, which the hard
    version failed on UPR by throwing the coverage away.
  - **The falsification, read out.** 1 **refuted** (the registered null: the
    non-floor 0.8158 > 0.55 and the within-document 0.7020 > the count's
    0.6360); 2 **refuted** (the head prediction: 84.7 % > 70 %); 3 **fails**
    (the z arms beat the count on the same pairs); 4 holds (blind gap +0.2340,
    de-shifted 0.7334 > 0.5238); 5 holds (34.08 %).
  - **What it does not fix**: the positives' hit@1 is **0.0744** against the
    cosine's 0.1050 — the pair ranking improves, the paragraph's best single
    candidate does not.
  - **The gate.** **974** tests, ruff clean over 237 files, two consecutive runs
    bit-identical.

---

## The word classes on OMCT (CS-51) — the pre-registration and the verdict

*Added on 2026-11-05, when the round's fourth change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-51.*

### 7.9 The word classes (CS-51)

- [x] **CS-51 — word classes: what does the hard restriction add on top of the
  soft prior?** CS-33's three classes on the pair unit (a word's class is how
  many questions' tables it sits in), the class readout crossed with the
  z-score, and the two class-restricted z arms measured **against the full
  z arm** — the bar CS-50 re-scoped: the prior already suppresses the register
  (the common words get `z ≈ 0`), so the pre-registered question is whether a
  **hard** filter adds anything to the **soft** one.

  **The UPR verdict this ports (the prior), and its reason.** CS-33 (executed
  2026-11-01) was a **readout win and a payoff loss**: the classes separate
  widely (the **distinctive** words 100 % positive at a mean log-odds of
  **+4.2676**, the **generic** 81 % at **+0.8439** — the registered null did
  not fire) — but the class-restricted maps were **much worse** as classifiers
  (0.3815 for "no generic" and **0.0798** for "distinctive only" against the
  full map's 0.5243). The reason: the strongest words are **rare** and the
  register is **everywhere**; the map's classifier works on **coverage**, so
  dropping the generic class keeps the signal and throws the coverage away.

  **The flip, re-scoped by CS-50.** On UPR the register was *inside* the sum
  (the plain log-odds), so dropping it hurt. On OMCT the z-score's per-word
  prior already sends the common words to `z ≈ 0` — a **soft** version of the
  same restriction — and the z arm is the chain's win (0.7020). So the
  pre-registered prediction is that the **hard** filter adds little: the
  "no generic" arm should land **within ±0.02** of the full z arm (the two are
  substitutes), and the **distinctive-only** arm should **lose** (coverage),
  as on UPR. "It works" (the restriction is a lever) = the "no generic" arm
  **beats** the full z arm by **≥ 0.05** on the same pairs with a paired
  interval excluding zero.

  **The mechanism.** A word's class is read off the map's own structure
  (CS-33's rule, unchanged): the number of the **209 labelled tables** it sits
  in — **distinctive** 1–2, **shared** 3–52, **generic** ≥ **53** (a quarter of
  209, the share the rule has always used). The readout is per class: the
  words, the (question, word) pairs, the share of **positive** log-odds, the
  mean log-odds **and the mean z-score** (CS-50's measure, the one that
  matters now). The two restricted arms are the **primary z-scored arm** with the
  map's classes filtered: **"no generic"** (distinctive + shared) and
  **"distinctive only"**. Each is scored behind the **same folds** as the full
  arm and compared with **the full arm on the same pairs** (the paired
  within-document difference — the increment test), with the cosine reference
  beside it.

  **The falsification conditions, registered.**

  1. **The increment test (the primary).** The "no generic" arm's paired
     within-document difference against the full z arm: the prediction is
     **within ±0.02** of zero (the filters are substitutes). **If it is ≥
     +0.05 with the interval excluding zero, the hard filter adds beyond the
     soft prior** and becomes a lever; if ≤ −0.05, the restriction costs the
     coverage the z arm needs (CS-33's failure reproduces on top of the prior).
  2. **The distinctive-only test.** Predicted to **lose** (the point estimate
     below the full arm's; no bar on the size — UPR's loss was −0.4445).
  3. **The class readout test.** The distinctive class's mean z-score must
     exceed the generic class's by **≥ 0.5** and the generic class's positive
     share must be lower — the z-measure version of CS-33's separation test
     (UPR's margin was 1.0 on the log-odds; the z-scores are on a smaller
     scale, so the bar is halved and stated in advance).
  4. **The count check** (CS-49's instrument, kept): the measure-free count's
     within-document AUC must stay **within 0.10 of the full z arm** — if a
     class filter somehow helps the count too, the reading is about the unit,
     not the classes.

  **The cost.** **0 new model calls**; the class index is computed inside the
  map build, and the two restricted arms double the fold-grid work (~12 s).

  **Leakage.** Unchanged: the map is built on the reviewed pairs only, the
  classes are a property of the **map's own tables** (no label information
  beyond the pairs), the control never enters a fit, and the restricted arms
  are scored by the same out-of-fold maps.

  **The runs.** One: **`omct-analyze-voyage-dict-classes`**, the artifact
  `results/omct-analyze-voyage-dict-classes/analysis.json` (the `dictionary`
  block with the class readout, the `kind: dictionary` model entry with the two
  restricted arms and their comparisons against the full arm); the sweep
  `experiments_text_processing/omct_dictionary_classes_sweep.py` (the label
  `omct_dictionary_classes`, the table's section
  `## OMCT — omct_dictionary_classes`); the config addition
  `analysis.dictionary.class_restrictions`; the log section
  **`## CS-51 — the word classes on OMCT`**.

  **The exit criteria.** A hand-computed unit test of the class rule on a toy
  map (a word in 1, 2, 3 and all of the tables); a test that a filtered word
  contributes nothing to the grid; a test that the restricted grid differs
  from the full one only through the filtered words; the fixture run and the
  real run; `ruff check` / `ruff format` clean over
  `src`/`tests`/`experiments_text_processing`; the full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-05** — the run is
  `omct-analyze-voyage-dict-classes` (the artifact
  `results/omct-analyze-voyage-dict-classes/analysis.json`, the
  `kind: dictionary` model entry with `class_arms`), the log section is
  `## CS-51 — the word classes on OMCT: the soft prior already does it` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (OMCT: the same 1,520 pairs, 4,000
  control cells and folds as CS-50; the full z arm is the reference).**

  - **The cost.** **0 model calls**, **18.7 s**, the artifact **15.1 MB**, **no
    new dependency**; CS-50's numbers reproduced exactly (the full arm 0.7020,
    the blind 0.4679, the floor 518).
  - **The increment test (the primary): a null.** The "no generic" z arm
    **0.6897** against the full arm's **0.7020** — a paired **−0.0123
    [−0.0221, −0.0024]** (P 0.005): the registered ±0.02 band holds at the
    point estimate, but the interval excludes zero, so the hard filter is a
    **small, detectable cost**; the ≥ +0.05 "lever" bar is far away.
  - **The distinctive-only arm loses as predicted**: **0.6262**, **−0.0758
    [−0.1052, −0.0472]**, with its **floor jumping from 34.1 % to 71.6 %**
    (1,088 pairs) and its non-floor reading **0.5157** — the hard filter
    empties most tables; the rest of its all-pairs number is the silence's
    prior.
  - **The class readout (the finding).** distinctive: 594 words, 918 pairs,
    **35.4 %** positive, mean log-odds **−0.2317**, mean z **+0.3782**; shared:
    1,380 / 18,598 / 64.7 % / +0.7971 / +0.1145; generic: 108 / 8,463 /
    **72.5 %** / **+0.9081** / **+0.0234**. The **raw log-odds reverse CS-33's
    class order** (the (d+0.5)/(c+0.5) term is dominated by the question's
    positive rate on a small table), while the **z-score's class order is the
    predicted one** (+0.3782 against +0.0234, a difference of **+0.3548**): the
    α₀ denominators cancel the positive-rate term.
  - **The separation test.** The registered bar (the z difference ≥ 0.5 **and**
    the generic positive share lower): **both legs fail** (0.3548 < 0.5; the
    generic share is higher, 72.5 % against 35.4 %) — the direction holds, the
    bar does not; the second leg is a CS-33 carry-over that does not transfer.
  - **The other checks.** The count stays within 0.10 of the full arm (0.6360
    against 0.7020) ✓. The "no generic" per-document numbers: 0.6003 / 0.7975 /
    0.8370.
  - **The gate.** **977** tests, ruff clean over 237 files, two consecutive runs
    bit-identical. No run-id deviation.

---

## The grammatical axis on OMCT (CS-52) — the pre-registration and the verdict

*Added on 2026-11-05, when the round's fifth change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-52.*

### 7.9 The grammatical axis (CS-52)

- [x] **CS-52 — POS: the grammatical axis (cheap, and predicted a null).** CS-34's
  five Penn classes on the pair unit (the noun/verb/adjective/adverb/other split
  of the map's words, the class of each word's modal tag), the POS readout
  crossed with the z-score, and the two POS-restricted z arms
  (`noun`, `verb`) measured against the full z arm.

  **The UPR verdict this ports (the prior), and its reason.** CS-34 (executed
  2026-11-01) was a **null on both halves**: the nouns and the verbs were
  nearly identical (+2.0044 against +1.9856, a difference of +0.0188 where the
  registered margin was 1.0) and the noun-only map was **materially worse**
  (0.4840 against the full map's 0.5243, a paired **+0.0403 [+0.0258,
  +0.0545]**); the verb-only map reached 0.1261. The reason: the map treats
  the grammatical classes almost identically — the register is in the *words*,
  not in their part of speech.

  **The flip, and the prediction.** CS-51 just showed the hard **class** filter
  is a small cost on top of the z-score's soft prior (the class readout's
  separation is the finding, the filter is the null). The grammatical axis was
  the weaker of the two on UPR, so the pre-registered prediction is a **null
  again**: the noun-only and verb-only z arms should land **within ±0.02** of
  the full z arm (the filters are not levers), and the POS readout should show
  the classes' **z-means close** (the noun–verb difference below the **0.5**
  bar). "It works" = the noun-only arm **beats** the full arm by **≥ 0.05**
  with a paired interval excluding zero **and** the noun–verb z difference
  clears 0.5.

  **The mechanism.** The tagger is the track's own (the edge's `NltkTagger`
  with the `averaged_perceptron_tagger_eng` model, its answers cached), tagging
  the reviewed set's **225 distinct paragraphs** with the track's tokenization
  (the words line up by position, as in CS-34); each word's class is the class
  of its **modal** tag. The readout is per class: the words, the pairs, the
  positive share, the mean log-odds and the **mean z-score**; the test is the
  **noun − verb z difference** (the registered bar 0.5, the z-measure version
  of CS-51's rule). The two restricted arms are the primary z arm with the
  map's POS classes filtered (`noun` keeps only nouns; `verb` only verbs),
  scored behind the same folds and compared with the **full** z arm on the same
  pairs.

  **The falsification conditions, registered.**

  1. **The increment test (the primary).** The noun-only arm's paired
     within-document difference against the full z arm: the prediction is
     **within ±0.02**; the ≥ +0.05 bar with the interval excluding zero would
     make the grammatical filter a lever (the unseen case).
  2. **The verb-only test.** Predicted to lose (the verbs are the register's
     carriers; the point estimate below the full arm's).
  3. **The POS readout test.** The noun class's mean z-score must exceed the
     verb class's by **≥ 0.5** for the grammatical axis to separate (the UPR
     null did not).
  4. **The count check** (kept from CS-49): the measure-free count within 0.10
     of the full z arm.

  **The cost.** **0 model calls** (the tagger is a local model, already
  downloaded for the UPR run; the 225 paragraphs are tagged once and cached);
  the two POS arms double the fold-grid work (~12 s).

  **Leakage.** Unchanged; the tagger reads the paragraph texts only (no
  labels), and its classes are a property of the words, so no label
  information enters the restriction.

  **The runs.** One: **`omct-analyze-voyage-dict-pos`**, the artifact
  `results/omct-analyze-voyage-dict-pos/analysis.json`; the sweep
  `experiments_text_processing/omct_dictionary_pos_sweep.py` (the label
  `omct_dictionary_pos`, the table's section `## OMCT — omct_dictionary_pos`);
  the config additions `analysis.dictionary.pos_restrictions` and the
  `pos:` section on; the log section **`## CS-52 — the grammatical axis on
  OMCT`**.

  **The exit criteria.** A hand-computed unit test of the POS class assignment
  on a toy map; a test that a POS-filtered word contributes nothing; the
  fixture run and the real run; `ruff check` / `ruff format` clean over
  `src`/`tests`/`experiments_text_processing`; the full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-05** — the run is
  `omct-analyze-voyage-dict-pos` (the artifact
  `results/omct-analyze-voyage-dict-pos/analysis.json`, the `kind: dictionary`
  model entry with `pos_arms`), the log section is
  `## CS-52 — the grammatical axis on OMCT: a null that reproduces` in
  `docs/experiments_text_processing.md`, and the numbers are below.)*

  **The round's numbers, in one screen (OMCT: the same 1,520 pairs, 4,000
  control cells and folds as CS-50; the full z arm is the reference).**

  - **The cost.** **0 model calls**, **18.6 s**, the artifact **15.1 MB**, **no
    new dependency**; the reviewed set's **225 paragraphs** tagged with
    `averaged_perceptron_tagger_eng` (cached; the 225 misses are the first
    pass).
  - **The classes do not separate (the UPR null reproduces, on the z-score).**
    noun 979 words / 14,387 pairs / 66.0 % positive / mean log-odds +0.7749 /
    **mean z +0.0885**; verb 483 / 5,731 / 66.9 % / +0.8307 / **+0.1032**;
    adjective +0.1045, adverb +0.1025, other +0.1004. The **noun − verb z
    difference is −0.0148** (the 0.5 bar not met; the sign is the opposite of
    the predicted direction) and every class's z mean lies within **0.016** of
    every other's.
  - **The arms.** noun-only **0.6979, −0.0041 [−0.0097, +0.0016]** (P 0.077) —
    the interval includes zero, the ±0.02 band holds: a free-ish filter, not a
    lever; verb-only **0.6297, −0.0723 [−0.0878, −0.0571]** — it loses, as
    predicted. Both keep the 518-pair floor (34.1 %): the POS filter empties no
    table.
  - **The falsification, read out.** 1 holds as a null; 2 holds; 3 fails (−0.0148
    against 0.5, wrong sign); 4 holds (the count 0.6360 within 0.10 of 0.7020).
  - **The reading.** The grammatical axis is a null on both datasets, and
    CS-50's per-word prior does not interact with it — the register is in the
    words, not in their part of speech.
  - **A correction appended to §CS-51**: its table row was rendered under the
    wrong columns (the section-choice flow fell through); no number changed, and
    the row was re-rendered by re-running CS-51 after the fix (in this commit).
  - **The gate.** **979** tests, ruff clean over 238 files, two consecutive runs
    bit-identical (bar the tagger's cache counters on the first pass). No
    run-id deviation.

## The phrases on OMCT (CS-53) — the pre-registration and the verdict

*Added on 2026-11-05, when the round's sixth change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-53.*

### 7.9 The phrases (CS-53)

- [x] **CS-53 — phrases: does the unit change buy anything on the pair unit?**
  The map rebuilt over the reviewed paragraphs' **consecutive bigrams** (the
  same builder, the same folds, the same estimator), the **phrase z arm**
  against the **word z arm** on the same pairs, and the registered test: the
  **union arm** = the word z arm + the phrase z arm, which must beat the word z
  arm by **≥ 0.05** (paired interval excluding zero) to count as "phrases
  work". This is the round's first **unit** change.

  **The UPR verdict this ports (the prior), and its reason.** CS-39 (executed
  2026-11-02) was a **null**: the phrase table's generic share was **15.5 %**
  against the word table's **43.6 %** (the phrases are too rare to sit in many
  labels' tables), **82.8 %** of the phrases were hapax in the train corpus,
  and the **blind union gained more than the real one** on the fit side
  (+0.0240 against +0.0050) — the phrases added noise, not signal. The reason:
  a 1,676-text corpus gives a bigram a df of 1 for four phrases in five, so
  the phrase unit is too **rare** to be a table. The UPR's one positive read:
  the register is real but it is a **fragment** ("on its", "with an", "and
  will").

  **The design decision taken here, and the one premise corrected.** The
  reviewer's instruction: the phrase table is the map over the reviewed
  paragraphs' consecutive bigrams (**the content-token stream plus the
  stopwords**, CS-39's settled wording — the track's `phrase_units`, not a new
  tokenizer), built with the same builder via its `units` parameter; the union
  is **the word z arm + the phrase z arm**, not a re-ranked or z-normalized
  blend. That is what this change set builds. The reviewer's premise for the
  predicted null — "each bigram's global df is tiny (mostly 1), so the
  z-score's shrinkage does almost nothing on phrases" — is **measured and
  refined**: the map's df is **pair-level** (a paragraph repeated over `k`
  pairs contributes `k`), so the phrase vocabulary's global df has median
  **5** (not 1), hapax **14.8 %**, and the prior's pseudo-count is
  `α_w = df + 0.01`; against the words' median **9** the phrase shrink is
  still **smaller** than the word shrink. So the phrase z is not "unshrunk":
  it is a **per-word statistic exactly like CS-50's** (the same estimator, a
  finer unit), and the predicted failure mode is **redundancy, not
  shrinkage** — a paragraph contains the bigram "human rights" only where it
  contains both words, so the phrase half re-reads the words' evidence at a
  second scale. A union of two highly correlated per-word sums moves the
  ranking only at the margin. The predicted null is therefore restated as:
  **the union lands within ±0.02 of the word z arm**; the rest of the
  reviewer's design (the bar, the readouts) is unchanged.

  **The mechanism.** Two maps, both built from the **reviewed pairs only**
  (subset A), both behind the same **5 folds over the pairs, seed 42** (the
  track's scheme, each fold's map from the other folds):

  - the **word map** — CS-48's map, unchanged (`content_tokens`);
  - the **phrase map** — the same builder with `units=phrase_units`: the
    consecutive bigrams of the track's **full token stream** (stopwords kept),
    so "human rights" and "on its" are single units of the table.

  The three scorings, all the primary **`sum_zscore_positive`** (CS-50's arm):
  the **word arm** (0.7020 in CS-50, reproduced inside this run), the
  **phrase arm** (the phrase table's own cells), and the **union**
  = `word_grid + phrase_grid` (the UPR CS-39 composition, cell for cell). The
  measure-free **`support_count`** arm is computed for **both** units and
  reported beside them (CS-49's instrument, kept as the reviewer asked). The
  **floor** is the row's smallest seen score minus one, applied per map; the
  word map's floor set and the phrase map's floor set **coincide on this data**
  (the same **518** pairs, read-only), so the union's floor is that set, and
  the secondary reading is over the same **1,002** pairs as CS-50's.

  **What is measured before the run, and must reproduce** (read-only,
  2026-11-05, the track's own tokenizer; the CS-48 precedent):

  | | the word table (CS-48) | **the phrase table** |
  |---|---|---|
  | the distinct units in some table | 2,082 | **7,360** |
  | the (question, unit) pairs stored | 27,979 | **58,769** |
  | the questions with a table / silent | 209 / 85 | **209 / 85** |
  | the per-question table size (min / median / max) | 0 / 65 / 477 | **0 / 120 / 1,107** |
  | the unit's global df (median / hapax share) | 9 / 9.4 % | **5 / 14.8 %** |
  | the stored rows with the question's own df = 1 | 75.4 % | **87.9 %** |
  | the stored rows with `n_with` = 1 | 89.1 % | **95.2 %** |
  | the rows' mean / median z (share positive) | +0.0956 / +0.0661 (92.2 %) | **+0.1371 / +0.0967 (96.4 %)** |
  | the cross-fit availability (table / evidence) | 65.9 % / 61.1 % | **65.9 % / 58.2 %** |
  | the pairs on the floor | 518 (34.1 %) | **518 (34.1 %)**, the same set |
  | the class split of the vocabulary (d / s / g) | 594 / 1,380 / 108 | **2,911 / 4,337 / 112** |
  | the **generic** class's share of the stored rows | 30.3 % | **14.3 %** |

  The last row is the CS-39 readout's **direction reproducing**: the phrase
  rows sit in fewer tables (14.3 % generic against the words' 30.3 %), exactly
  the UPR pattern (15.5 % against 43.6 %), even though the unit is **thicker**
  per question (median 120 against 65) and **not** hapax-rare at the pair
  level (14.8 % against CS-39's 82.8 %). The UPR null's *premise* does not
  reproduce; its *reading* of the unit ("rarity is read as topicality") may
  still.

  **The metric (the headline and its companions).**

  1. **Within-document AUC of the union against the word z arm** on the same
     1,520 pairs — the paired 2,000-resample bootstrap of the difference
     (the registered test; the pairs are the resampling unit).
  2. **The phrase arm alone** against the word arm and against the run's
     **plain full-text cosine** (voyage-4-nano's own, 0.5238) — the spread,
     reported so the unit's own reading is on the record.
  3. **The pooled AUC** of the union beside the within-document one (never
     alone), and **per document** (NU / CIDH / AI; the AI stratum is 99
     pairs).
  4. **The non-floor reading** of the union and of both halves (the 1,002
     pairs off the floor), and the **floor** (518 pairs, 34.1 %) beside it.
  5. **The count arms** (`support_count`) for both units, and the word arm's
     reproduction (it must read **0.7020**, the non-floor **0.8158**, the
     floor **518** — CS-50's numbers, same folds).
  6. **The phrase table's readout** (CS-39's shape, ported): the generic
     class's share of the rows, the hapax share, the table sizes — and the
     phrase map's own class/z readout (the `dictionary` block's `classes`,
     computed over phrases), which is where the "rarity read as topicality"
     claim would show.

  **The falsification conditions, registered — read out one by one.**

  1. **The flip (the primary test).** "Phrases work" = the union's
     within-document AUC is **at least 0.05 above the word z arm's** with the
     paired 95 % interval **excluding zero**. The registered prediction is a
     **null**: the union lands **within ±0.02** of the word arm (redundancy).
     If it is **≤ −0.05** with the interval excluding zero, the union is a
     **cost** (the phrases dilute the words' ranking) and is reported with its
     number and its reason.
  2. **The phrase arm alone.** Predicted **not** to beat the word arm (point
     estimate below 0.7020); if it does, the phrase unit is a lever on its
     own and the reading changes.
  3. **The reproduction guard.** The run's phrase-table shape must equal the
     read-only table above (7,360 / 58,769 / 209 / 85 / median 120), and the
     word arm it computes must read **0.7020** with the **518**-pair floor. A
     difference means the unit, the folds or the estimator is not the
     pre-registered one → **stop and report**, do not proceed to the reading.
  4. **The floor check.** The phrase table's floor set must be the **same 518
     pairs** as the word table's (read-only: it is). If not, the union's
     silence differs between its halves and the union's floor needs its own
     reading — **report** rather than substitute.
  5. **The count check** (CS-49's instrument). Each unit's count arm must land
     **within 0.10** of its own z arm (words: 0.6360 against 0.7020) — if the
     count collapses on phrases, the phrase z is a measure effect, not a unit
     effect, and the union's reading is about measure, not unit.
  6. **The class readout.** The phrase table's generic-class row share must be
     **below** the word table's (pre-read 14.3 % against 30.3 %): the CS-39
     direction. A reversal means the class measure is reading the unit's own
     prevalence on OMCT and the phrase table's "rarity" claim is a UPR fact
     only — reported either way.

  **The nulls this change set can produce, stated so they are not read as
  defects.** (i) The union within ±0.02 of the word arm — the registered null
  (the phrases are redundant with their own words). (ii) The phrase arm alone
  below the word arm — the unit is not a table on this data either. (iii) The
  union's non-floor reading falling below the word arm's while its all-pairs
  number rises — the floor doing the work again (CS-49's lesson). All three
  are legitimate results of the transfer test, each reported with its number.

  **The cost.** **0 model calls** (voyage-4-nano's cosine is the Stage-0
  cache hit); three cross-fit grids (the phrase map's two arms, the word
  map's two arms, and the union composed from them — no blind arm), each
  building six maps over the 1,520 pairs, ≈ **30–45 s**; the artifact grows by
  the phrase map's **58,769** rows (≈ **12 MB**), and no new dependency.

  **The label-blind counterpart is not re-run, and the reason is stated.** The
  registered test is an **increment between two arms built from the same
  labels on the same pairs** (the union against the word arm), so a difference
  between them cannot be a label-blind effect; the base arm's label-driven
  gap is already measured (**+0.2340**, CS-50). The blind arm would re-measure
  the word arm's label-drivenness, not the phrase increment.

  **Leakage.** The phrase map and the union's word half are both built from
  the **reviewed pairs only** (subset A) and both scored **out-of-fold**; the
  unjudged control (B) never enters any fit, is never a label, and its cells
  are scored by the full reviewed-set map only (the vintage difference
  recorded, as in CS-49). The union's two halves share the same folds, so no
  pair's own label reaches either half. No UPR data and no UPR artifact is
  touched.

  **The runs.** One: **`omct-analyze-voyage-dict-phrases`**, the artifact
  `results/omct-analyze-voyage-dict-phrases/analysis.json` (the `dictionary`
  block now describing the **phrase** table, with its unit name and a
  unit-profile readout; the `kind: dictionary` model entry with the phrase
  arm, the count arm and the **union** block); the sweep
  `experiments_text_processing/omct_dictionary_phrases_sweep.py` (the label
  `omct_dictionary_phrases`, the table's section
  `## OMCT — omct_dictionary_phrases`); the config additions
  `analysis.dictionary.units` (`words` | `phrases`) and
  `analysis.dictionary.union` (the union arm on, valid only with
  `units: phrases`); the log section **`## CS-53 — the phrases on OMCT`**.

  **The exit criteria.** A **hand-computed unit test** of the bigram unit (a
  sentence's consecutive bigrams, the stopwords kept: "the death penalty" →
  `the death`, `death penalty`); a test that the map keyed by bigrams stores
  bigrams and that a one-token paragraph has an **empty** phrase table; a
  **hand-computed union test** (a toy grid whose word and phrase halves are
  summed cell for cell, the floor preserved); a test that the union's floor
  set is the word map's and the phrase map's shared set; a test that the
  `units: phrases` config is required for `union: true` (and that a bad unit
  is refused at config time); the fixture run and the real run; `ruff check` /
  `ruff format` clean over `src`/`tests`/`experiments_text_processing`; the
  full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-05** — the run is
  `omct-analyze-voyage-dict-phrases` (the artifact
  `results/omct-analyze-voyage-dict-phrases/analysis.json`, the
  `kind: dictionary` model entry with the `union` block and the phrase
  table's `dictionary` block), the log section is
  `## CS-53 — the phrases on OMCT: the UPR null reproduces, for the other
  reason` in `docs/experiments_text_processing.md`, and the numbers are
  below.)*

  **The round's numbers, in one screen (OMCT: the same 1,520 reviewed pairs,
  4,000 control cells, 5 folds and seed as CS-50; the word z arm is
  reproduced inside the run; the reference is voyage-4-nano's own plain
  full-text cosine).**

  - **The cost.** **0 model calls**, **15.2 s**, the artifact **26.7 MB** (the
    phrase table's 58,769 rows), **no new dependency**.
  - **The registered test (the union against the word z arm): a null.** The
    union **0.6976** against the word arm's **0.7020** — a paired **−0.0043
    [−0.0074, −0.0012]** (P(Δ>0) 0.0045). The registered prediction was
    **within ±0.02**: the point estimate is inside the band, the interval
    excludes zero → a **small, detectable cost**, and the ≥ +0.05 bar is far
    away. The union's non-floor reading **0.8057** (the word arm's 0.8158) over
    the same 1,002 pairs; below the word arm in **all three** documents
    (−0.0055 / −0.0013 / −0.0024). Against the cosine: **+0.1738 [+0.1264,
    +0.2222]**, pooled **0.7421** (+0.2418), de-shifted **0.7272**.
  - **The phrase arm alone: it works, and it is redundant.** **0.6840**
    (+0.1602 [+0.1129, +0.2088] against the cosine, P 1.000; non-floor
    **0.7740**; per document 0.6031 / 0.7911 / 0.8171; de-shifted 0.7091),
    **below** the word arm — condition 2 holds.
  - **The reproduction guard holds, digit for digit.** 294 questions, **7,360**
    phrases, **58,769** (question, phrase) rows; **209** tables / **85** silent;
    table size min 0 / median **120** / max **1,107**; availability **65.92 %**
    / **58.16 %**; the floor **518** pairs (34.08 %); the word arm **0.7020**
    (non-floor 0.8158, count 0.6360); the phrase map's floor set is the word
    map's (`same_floor_set: true`).
  - **The UPR null's premise does not transfer.** CS-39's rarity (82.8 %
    hapax, the generic share 15.5 % against 43.6 %) becomes, on the pair unit:
    **14.8 %** hapax (global df median **5**; the df is pair-level), the median
    question's phrase table **120** rows against the word table's **65**, and
    the generic-class row share **14.3 %** against the word table's **30.3 %**.
    The **class readout's direction** reproduces (the phrase rows sit in fewer
    tables — rarity read as topicality) even though the rarity does not; the
    null reproduces for **redundancy** (a bigram occurs only where both its
    words do).
  - **The class readout on the phrase vocabulary.** distinctive 2,911 phrases /
    4,436 rows / **35.4 %** positive / mean log-odds **−0.2104** / mean z
    **+0.3900**; shared 4,337 / 45,917 / 67.7 % / +0.9307 / +0.1333; generic
    **112** / 8,416 / **75.2 %** / **+1.0106** / **+0.0242**. The z
    difference is **+0.3658** (CS-51's pattern, on a finer unit: the direction
    holds, the 0.5 bar — CS-51's, not this change set's — does not).
  - **The count arms.** Word **0.6360** within 0.10 of its z arm (0.7020) ✓;
    phrase **0.6217** within 0.10 of its own (0.6840) ✓.
  - **The falsification, read out.** 1 **holds as a null** (−0.0043, inside the
    ±0.02 band, the interval excluding zero: a small cost); 2 **holds** (the
    phrase arm below the word arm); 3 **holds** (the shape and CS-50's 0.7020 /
    518); 4 **holds** (the same 518 pairs in both halves); 5 **holds** (both
    counts); 6 **holds** (14.3 % < 30.3 %).
  - **The deviation.** None: the run id, the sweep label, the artifact and the
    log section are the pre-registered ones; no blind arm (as registered).
  - **The gate.** **986** tests, ruff clean over **240** files, two consecutive
    runs bit-identical (bar the clock field).

## The stability of the pair unit (CS-54) — the pre-registration and the verdict

*Added on 2026-11-06, when the round's seventh change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-54.*

### 7.9 The stability (CS-54)

- [x] **CS-54 — stability: how much of the pair unit's map is sampling noise?**
  CS-36's bootstrap of the map, ported to the reviewed pairs: the survival
  curve of the (question, unit) links, the per-question and head readouts, and
  the **filtered z arm** measured against the **full z arm** — the reference is
  CS-50's arm (0.7020, non-floor 0.8158), not the raw log-odds.

  **The UPR verdict this ports (the prior), and its reason.** CS-36 (executed
  2026-11-02) found the **filter a null and the readout the deliverable**: the
  survival curve had a **cliff at 63 %** (1 − e⁻¹, where a single-text link
  belongs), **not one** of the 12,483 single-text links survived τ = 0.75,
  only **27.6 %** of the map did — and the filtered map was a **worse**
  classifier (the fit side's out-of-fold 0.6544 against the full map's
  0.6613). The reason: the map's classifier works on **coverage**, not on the
  strength of its best words, so dropping the fragile tail drops coverage.

  **The mechanism, and the pair-unit prediction.** The **reviewed pairs** (the
  map's rows) are resampled with replacement, B = **500** times (seed 42), and
  the map is rebuilt from scratch on every resample with the same builder; for
  every link of the full map: `present_share` (in the question's table at
  all), `sign_stability` (present **and** keeping the full map's log-odds
  sign — the statistic the filter uses), `head_stability` (**present and still
  among the question's top 10** by log-odds, K = 10 — reported, not filtered
  on). The theory is CS-36's: a link resting on **k** reviewed pairs survives
  about **1 − e^(−k)** of the resamples — a single-pair link **≈ 63 %**, two
  pairs **≈ 86 %**, three **≈ 95 %**. On the pair unit the map is thinner than
  UPR's: CS-48 measured **89.1 %** of the word map's rows at `n_with` = 1 and
  **75.4 %** at df = 1, so the registered prediction is that only **≈ 10–15 %**
  of the map survives **τ = 0.75** (the links with ≥ 2 reviewed pairs are
  10.9 %), that the **filtered arm's floor rises sharply** (from 34.1 % —
  predicted **≥ 50 %**), and that the filter is therefore a **cost**, not a
  lever — CS-36's conclusion, at a thinner unit.

  **The filter, and its leakage discipline.** A link survives τ iff its
  `sign_stability ≥ τ`; the dropped links leave the question's table entirely
  (a word contributes nothing). **Each fold's filter is estimated on that
  fold's own training pairs** (B = **100** resamples, the same seed), exactly
  as CS-36 estimated it per fold, so a judged pair is scored by a map and a
  filter that both never saw it; the **unjudged control's** filter is the
  full reviewed-set estimate (the control enters no fit). Two thresholds,
  **τ = 0.75** (the registered one) and **τ = 0.90** (reported beside it);
  the control never enters a resample.

  **The metric.** The **readout** (the deliverable): the survival curve
  (τ from 0.00 to 1.00 in steps of 0.05), the survival share by support
  (links resting on 1, 2, 3 … pairs — the fragility reading), the per-question
  stability (each table's size and the share of its links surviving), the
  head stability (how many of the 209 heads keep their place in ≥ 50 % of the
  resamples), and the full per-link table. The **model test**: the τ = 0.75
  filtered z arm's within-document AUC against the **full z arm's on the same
  pairs** (the paired 2,000-resample bootstrap — the registered headline),
  its floor and non-floor readings, and the τ = 0.90 arm beside it.

  **The falsification conditions, registered — read out one by one.**

  1. **The model test (the primary).** "The filter is a lever" = the τ = 0.75
     arm **beats** the full z arm by **≥ +0.05** with the paired 95 % interval
     excluding zero. The registered prediction is the opposite: the filtered
     arm's point estimate is **below** the full arm's (the coverage cost). If
     it is ≥ +0.05 with the interval excluding zero, CS-36's conclusion is
     refuted on the pair unit and the reading changes.
  2. **The floor test.** The filtered arm's floor share must **rise** above
     the full arm's 34.1 % (predicted **≥ 50 %**): the filter empties the
     questions whose every link rests on one pair. If it stays within 5 points
     of 34.1 %, the map is not as single-pair-dominated as CS-48's readout
     says and the rationale (not the verdict) changes.
  3. **The survival test.** The share of the map surviving τ = 0.75 is
     predicted **10–15 %** (the `n_with ≥ 2` share, 10.9 %, is the upper
     reading; single-pair links survive ≈ 63 % and sit below the threshold).
     **If it is > 40 %**, the pair unit's map is far thicker than CS-48's row
     statistics suggest; **if < 5 %**, the filter keeps almost nothing — both
     are reported, with the number.
  4. **The cliff test.** The single-pair links' survival share must be within
     **±10 points of 63 %** (the 1 − e⁻¹ prediction; the bootstrap's own
     finite-corpus correction is small at n = 1,520). A large deviation means
     the resampling is not the ordinary bootstrap of the pairs.
  5. **The count check** (CS-49's instrument, kept): the filtered map's
     measure-free count arm, reported beside the filtered z arm.

  **The nulls this change set can produce.** (i) The filter a cost (the
  registered prediction; CS-36's verdict reproduces). (ii) The filter a null
  (within ±0.02, free). (iii) The readout alone: no arm conclusion, the
  survival numbers are the deliverable. All three are results of the transfer
  test, each reported with its number.

  **The cost.** **0 model calls**; **500** resamples for the readout (≈ 70 s)
  and **5 × 100** for the per-fold filters (≈ 65 s), plus two filtered
  cross-fit grids (≈ 20 s) — about **2.5 minutes**; the artifact grows by the
  27,979 per-link rows (≈ **3 MB**); no new dependency.

  **Leakage.** The bootstrap draws from the **reviewed pairs only** (subset A);
  each fold's filter is estimated on that fold's **training** pairs; the
  unjudged control (B) never enters a resample and is never a label; the
  filtered arm's judged cells are scored by their own fold's filtered map; no
  UPR data and no UPR artifact is touched.

  **The runs.** One: **`omct-analyze-voyage-dict-stability`**, the artifact
  `results/omct-analyze-voyage-dict-stability/analysis.json` (the `dictionary`
  block with the `stability` readout, the `kind: dictionary` model entry with
  the full arm reproduced and the two filtered arms); the sweep
  `experiments_text_processing/omct_dictionary_stability_sweep.py` (the label
  `omct_dictionary_stability`, the table's section
  `## OMCT — omct_dictionary_stability`); the config additions
  `analysis.dictionary.stability`, `stability_resamples` (500),
  `stability_fold_resamples` (100) and `stability_thresholds` ([0.75, 0.90]);
  the log section **`## CS-54 — the stability of the pair unit`**.

  **The exit criteria.** A **hand-computed** stability test on a corpus whose
  every paragraph is identical (every share must be exactly 1.0) and on a
  hand-authored report (the survival curve, the by-support shares, `survivors`
  at a threshold); a test that a filtered-out link contributes nothing to the
  grid and that a question with no surviving link carries the floor; a test
  that the fold's filter is the fold's own (a link surviving only in the
  full-set estimate is still dropped in a fold whose estimate lacks it); the
  fixture run and the real run; `ruff check` / `ruff format` clean over
  `src`/`tests`/`experiments_text_processing`; the full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-06** — the run is
  `omct-analyze-voyage-dict-stability` (the artifact
  `results/omct-analyze-voyage-dict-stability/analysis.json`, the
  `kind: dictionary` model entry with the two filtered arms and the
  `stability_readout`), the log section is
  `## CS-54 — the stability of the pair unit: the null becomes a cost` in
  `docs/experiments_text_processing.md`, and the numbers are below. The run
  finished after midnight, hence the artifact's 2026-11-06.)*

  **The round's numbers, in one screen (OMCT: the same 1,520 reviewed pairs,
  4,000 control cells and folds as CS-50; the full z arm is reproduced inside
  the run; 500 resamples for the readout, 100 per fold for the judged cells'
  filters).**

  - **The cost.** **0 model calls**, **136 s**, the artifact **60.7 MB** (the
    27,979 per-link rows; the pre-registration's "≈ 3 MB" estimate was wrong by
    the per-link table's size), **no new dependency**.
  - **The model test (the primary): the filter is a cost.** τ = 0.75:
    **0.6399** against the full arm's **0.7020** — **−0.0621 [−0.0859,
    −0.0395]** (P 0.000); non-floor **0.7659**; the floor **942 pairs
    (61.97 %)**. τ = 0.90: **0.5926**, **−0.1094 [−0.1384, −0.0828]**, floor
    **1,139 (74.9 %)**, non-floor 0.6647. Per document at 0.75: 0.5693 /
    0.7174 / 0.7562 (the full arm 0.6219 / 0.7981 / 0.8338).
  - **The readout.** The survival curve: 77.0 % at 0.50, 55.4 % at 0.60,
    **19.6 % at 0.65**, 7.4 % at 0.70, **6.79 % at 0.75**, **1.02 % at 0.90**,
    0.01 % at 1.00 — the cliff between 0.60 and 0.70. By support: the
    single-pair links (24,942, 89.1 %) are present **63.3 %** of the resamples
    (the 1 − e⁻¹ prediction) and keep their sign **54.1 %**, so **not one**
    clears 0.75; the two-pair links survive 61.6 %, the three-pair 67.4 %, the
    four-pair 64.1 %. The **heads**: 1,357 of 2,087 (65.0 %) keep their place
    in ≥ 50 % of the resamples. The **median question's table keeps none** of
    its links at τ = 0.75.
  - **The count arms (filtered).** τ = 0.75: **0.6195** against the filtered
    z arm's 0.6399 (a 0.020 gap); τ = 0.90: 0.5841 against 0.5926.
  - **The falsification, read out.** 1 **holds** (a cost, both thresholds, the
    intervals excluding zero); 2 **holds** (the floor 61.97 % ≥ 50 %);
    3 **misses** — **6.79 %** against the predicted 10–15 % (below the band,
    above the 5 % falsifier); 4 **holds on the presence statistic** (63.3 %
    against 63 % ± 10) and the **sign** statistic adds a cliff the registration
    did not state (54.1 %); 5 **holds** (the count within 0.10 of its z arm at
    both thresholds).
  - **The reading.** The UPR verdict reproduces in **kind** (the filter is not
    a lever; the readout is the deliverable) and not in **size** (a ~0.007 null
    there, a −0.062 cost here): 89.1 % of the pair unit's links rest on one
    reviewed pair, the filter drops all of them, and the coverage the z arm
    reads collapses with them.
  - **The gate.** **991** tests, ruff clean over **243** files, two consecutive
    runs bit-identical (bar the clock field).

## The label side's own words (CS-55) — the pre-registration and the verdict

*Added on 2026-11-06, when the round's eighth change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-55.*

### 7.9 The label side's own words (CS-55)

- [x] **CS-55 — the per-label lexicon, ported to the label side: the question's
  own words, and the register.** CS-35/45's channel on the pair unit: the
  **question text's own vocabulary** as a label-side scorer (no pair labels
  read), the **register** analog (the questionnaire's shared vocabulary), and
  the three arms — `all`, `topic`-only and `register`-only — against the
  cosine and against the map's z arm.

  **The UPR verdict this ports (the prior), and its reason.** CS-35 (executed
  2026-11-02) found the **mined lexicon the best cell-level signal in the
  pack** (the AUC 0.8109 against the description's 0.7067) but **the bar was
  not met**: it did not separate from the name+description channel
  (+0.0272 [+0.0044, +0.0502], only one of three seen buckets clearing), so it
  did **not** become a catalog channel. CS-45 then showed the **register was
  the signal**: dropping the distinctive filter halved the channel's gain
  (0.0272 against 0.0507) and took its AUC to the description's own level
  (0.7073). The reason: the mined words' advantage over the description lived
  in the corpus's own register, not in the labels' topic words.

  **The analog, stated before measuring (the reviewer's condition).** OMCT has
  **no label catalog**: the only text a label has is the **question's own
  text**. So on this unit the label's "name" and its "description" are the
  **same object** — the question — and the CS-11/CS-35 "name+description
  channel" analog is **the question's own vocabulary**. The **register**
  analog is the **questionnaire's** shared vocabulary: a label-side word that
  sits in ≥ a quarter (74) of the 294 questions' texts (CS-33's rule, read on
  the label side). The **topic** analog is the label-side word in 1–2
  questions' texts. This is a **label-side** register (the questions' habit),
  not CS-45's corpus-side one (the map tables' habit) — the difference is
  stated because it is where the port could fail.

  **The read-only shape** (measured 2026-11-05, no pair labels read; the run
  must reproduce it): 294 questions, **2,445** label-side words — **1,567**
  distinctive (1–2 questions), **868** shared, **10 generic** (≥ 74);
  the per-question vocabulary min **9** / median **32** / max **100** words;
  the generic words *question* (294 questions), *choices* (281), *answer*
  (281), *data* (265), *torture* (118), *applicable* (107), *state* (83),
  *following* (82); the reviewed pairs whose paragraph shares **no** label-side
  word: **3.6 %**; the mean label-side words found per pair's paragraph
  **4.44** (median 4).

  **The mechanism.** Three arms, all **label-side only** (the question texts;
  no pair label, no embedding):

  - `all` (**the primary**, fixed a priori — the "name" analog is the whole
    question text): the count of the question's content words present in the
    paragraph;
  - `idf`: the same, weighted by the word's label-side idf
    `log(294 / (the number of questions whose text holds it))` — the label
    side's own rarity prior (CS-31's shape, at the label side);
  - `topic`: the count over the question's **distinctive** words only (the
    register dropped — CS-45's filter, ported);
  - `register`: the count over the question's **generic** words only.

  Each arm scores the whole grid (225 × 294) and is read on the reviewed pairs
  as the run reads every arm: the **within-document AUC**, the paired
  difference against the run's cosine (0.5238) and — for the primary — against
  the map's z arm (0.7020, reproduced in the same run). The **label-blind**
  counterpart is the same channel with the **questions permuted** across the
  pairs (CS-49's seeded permutation), so its gap is the label-driven part.

  **The metric.**

  1. **The channel test (the primary).** The `all` arm's within-document AUC
     receives a signal only if it is **above the cosine's**, the paired 95 %
     interval **excluding zero**.
  2. **The map bar.** The `all` arm against the **map's z arm** (0.7020): the
     registered reading (CS-35's lesson) is that the label side **loses** to
     the corpus side; "it becomes a channel" (beats the map by ≥ +0.05 with
     the interval excluding zero) is the unseen case.
  3. **The register test (CS-45's port).** `topic` against `all`: the
     prediction is that dropping the register **loses** (the register was the
     signal); **if `topic` beats `all` by ≥ +0.05 with the interval excluding
     zero, CS-45's reading does not transfer** to the label side.
  4. **The label-driven check.** The primary arm's within-document AUC minus
     the blind channel's must be **≥ 0.05**.
  5. **The readout.** The vocabulary's class split, the top register words,
     the per-question vocabulary sizes, the zero-overlap share (3.6 %), and
     the arms' non-blind AUCs.

  **The falsification conditions, registered.**

  1. **The channel test.** Below the cosine with the interval excluding zero →
     the label side carries **no** signal on this unit (reported as a null);
     within the interval → a null (the question text is not a channel).
  2. **The map bar.** As above (the registered direction: the map wins).
  3. **The register test.** The ≥ +0.05 tower case **falsifies CS-45's
     transfer** on the label side, and the number is reported with it.
  4. **The label-driven check.** A blind gap below 0.05 → the channel's edge is
     the questionnaire's shape, not the question-to-paragraph relation.

  **The cost.** **0 model calls**, **0 pair labels read** (the channel is
  built from the question texts and the paragraph texts only); four grid
  scorings (225 × 294 each) plus the blind counterpart — seconds; no new
  dependency; the artifact grows by the readout and the arms' blocks.

  **Leakage.** The channel reads the question texts and the paragraph texts
  **only** — no pair label, no map, no embedding — so it cannot leak the
  reviewed set's labels; the blind counterpart permutes the questions; the
  control cells never enter any fit (there is none) and are never a label; no
  UPR data and no UPR artifact is touched.

  **The runs.** One: **`omct-analyze-voyage-dict-lexicon`**, the artifact
  `results/omct-analyze-voyage-dict-lexicon/analysis.json` (the `dictionary`
  block's `question_lexicon` readout and the `kind: dictionary` model entry's
  `question_lexicon` block); the sweep
  `experiments_text_processing/omct_dictionary_lexicon_sweep.py` (the label
  `omct_dictionary_lexicon`, the table's section
  `## OMCT — omct_dictionary_lexicon`); the config addition
  `analysis.dictionary.question_lexicon` (the on-switch); the log section
  **`## CS-55 — the label side's own words`**.

  **The exit criteria.** A **hand-computed** vocabulary test (a question's
  content words, its class from the number of questions' texts holding them);
  a test that a stopword and a one-letter token never enter the channel; a
  hand-computed score grid (the count, the idf and the register-only arm on a
  two-question toy); a test that the blind permutation keeps the per-question
  counts and that the blind channel differs from the real one; the fixture run
  and the real run; `ruff check` / `ruff format` clean over
  `src`/`tests`/`experiments_text_processing`; the full `pytest` suite green.

  *(pre-registered 2026-11-05; **executed 2026-11-06** — the run is
  `omct-analyze-voyage-dict-lexicon` (the artifact
  `results/omct-analyze-voyage-dict-lexicon/analysis.json`, the
  `dictionary.question_lexicon` readout and the model entry's
  `question_lexicon` block), the log section is
  `## CS-55 — the label side's own words: the register was the noise` in
  `docs/experiments_text_processing.md`, and the numbers are below. **Two of
  the registered branches fired** (the channel test's below-cosine branch and
  the register test's tower case); the chain stops after this change set for
  the reviewer's word.)*

  **The round's numbers, in one screen (OMCT: the same 1,520 reviewed pairs,
  4,000 control cells and folds as CS-50; the channel reads the question and
  paragraph texts only).**

  - **The cost.** **0 model calls**, **0 pair labels read**, **13.5 s**, the
    artifact **15.1 MB**, **no new dependency**.
  - **The channel test (the primary `all`): the below-cosine branch fires.**
    **0.4602** against the cosine's **0.5238** — **−0.0636 [−0.1008,
    −0.0269]**. The `idf` arm **0.5474** (+0.0236 [−0.0125, +0.0611]) and the
    `topic`-only arm **0.5423** (+0.0185 [−0.0215, +0.0575]) are at the cosine
    (nulls); the `register`-only arm **0.3754** (−0.1484) anti-predicts.
  - **The map bar: holds.** The `all` arm against the map's z arm
    **−0.2417 [−0.2887, −0.1949]**.
  - **The register test: the registered falsifier FIRES.** `topic` against
    `all` is **+0.0821 [+0.0448, +0.1185]** (the ≥ +0.05 tower case): dropping
    the questionnaire's register **helps**, so CS-45's "the register was the
    signal" does not transfer to the label side.
  - **The label-driven check: fails.** The blind channel **0.4680**, the gap
    **−0.0078** (the real channel is *below* the blind one).
  - **The shape.** 294 questions, **2,445** label-side words (1,567
    distinctive / 868 shared / 10 generic); the per-question vocabulary min 9 /
    median **31.5** / max 100; the zero-overlap share **3.6 %**; the mean
    overlap **4.44**. The register's top: *question* 294, *answer* and
    *choices* 281, *data* 265, *torture* 118, *applicable* 107, *state* 83.
  - **The falsification, read out.** 1 **fires** (below the cosine); 2
    **holds**; 3 **fires** (the register test's tower case); 4 **fails**.
  - **The reading.** The register's **side** is the reason: UPR's was
    corpus-side (everywhere in the corpus, carrying the mined channel's edge);
    OMCT's is label-side (the questionnaire's boilerplate, identical across
    questions), so it carries nothing question-specific and costs the raw count
    0.08. The label side's topic words reach the cosine and no further.
  - **The gate.** **996** tests, ruff clean over **246** files, two consecutive
    runs bit-identical (bar the clock field).

## Abstention on the pair unit (CS-56) — the pre-registration and the verdict

*Added on 2026-11-06, when the round's ninth change set was executed: the live
plan's §7.9 block moved here **verbatim** (the archive's convention: an
executed change set's pre-registration moves here and the live plan keeps one
tombstone line). The verdict and the numbers are the track log's §CS-56.*

### 7.9 Abstention (CS-56)

- [x] **CS-56 — abstention: is the floor's "impossible" the right encoding?**
  CS-37's abstention on the pair unit: the empty-table cells' **fill** — the
  floor's *impossible*, the row's median's *no evidence* (`neutral`), or the
  row's maximum's *possible* (`max`) — measured on the primary z arm against
  the floor on the same pairs.

  **The UPR verdict this ports (the prior), and its reason.** CS-37 (executed
  2026-11-02) was a **null**: the abstention ("no evidence" as NaN instead of
  the floor) left the eight held-out labels at **0.0000**, a paired
  **+0.0000** — the "impossible" verdict was **never in the encoding**; a map
  with no word↔label information at all collapsed them the same way. The
  reason: it was the **fit-side silence** (the eight were the labels the map
  was silent about *and* the labels that were never positive) that cost them,
  not the floor's arithmetic.

  **The OMCT mechanism, and the flip.** The silence is per **question** and
  large: **85 silent questions** (28.9 %), **331 pairs** with no positive in
  the question, and **518 floor pairs** (34.1 %) — **77.8 %** of them negative
  (CS-48/CS-49). So the floor's "impossible" is **right four times in five** on
  this unit: the silence **is** informative here, and the flip the UPR null
  opens is the opposite one — the floor should **win**. The test replaces the
  fill of the empty-table cells only:

  - `floor` (the current rule, the reference): the row's smallest seen score
    minus one;
  - `neutral`: the row's **median** of the seen cells ("no evidence");
  - `max`: the row's **maximum** of the seen cells ("possible").

  **The metric.** Each fill's within-document AUC on the primary
  `sum_zscore_positive` arm, and the paired difference against the floor
  (the pairs the resampling unit, 2,000 resamples); the floor's share and the
  **floor pairs' negative share** (the number that makes the floor work); the
  non-floor reading (the fill does not touch it); the silent questions' pair
  counts, per document.

  **The falsification conditions, registered.**

  1. **The abstention test (the primary).** `neutral` against the `floor`: the
     prediction is the **floor wins** (the point estimate above `neutral`).
     "The abstention is a lever" = `neutral` beats the floor by **≥ +0.05**
     with the paired 95 % interval excluding zero — then CS-37's null does not
     transfer and the floor's "impossible" is a cost.
  2. **The optimistic-fill test.** `max` is predicted **above the floor** (it
     lifts the empty cells over) — reported, with its number, but it is not a
     claim about abstention (it is the "everything is possible" encoding).
  3. **The informativeness test.** The floor pairs' negative share must be
     **> 50 %** (the pre-read: 77.8 %): the floor's prior is real on this unit.
     If it is ≤ 50 %, the floor is a coin and the reading changes.
  4. **The non-floor check.** The non-floor reading must be the same under all
     three fills (the fill only touches the empty cells).

  **The cost.** **0 model calls**; one cross-fit per fill (the same folds, the
  same map; three grids, ≈ **8 s**), **no new dependency**; the artifact grows
  by the fills' arm blocks and the readout.

  **Leakage.** Unchanged: the map is built on the reviewed pairs only, the
  control (B) never enters a fit, and the fills change only the score of cells
  whose fold map is silent — a score, never a label.

  **The runs.** One: **`omct-analyze-voyage-dict-abstain`**, the artifact
  `results/omct-analyze-voyage-dict-abstain/analysis.json` (the `dictionary`
  block's `abstention` readout and the model entry's `abstention_arms`); the
  sweep `experiments_text_processing/omct_dictionary_abstain_sweep.py` (the
  label `omct_dictionary_abstain`, the table's section
  `## OMCT — omct_dictionary_abstain`); the config addition
  `analysis.dictionary.abstention_fills`; the log section
  **`## CS-56 — abstention on the pair unit`**.

  **The exit criteria.** A **hand-computed** fill test on a toy grid (one
  empty column: the floor, the neutral median and the max on the same row);
  a test that the fill never touches a non-empty cell and that the floor flags
  are unchanged by the fill; a test that an all-empty row keeps its fill; the
  fixture run and the real run; `ruff check` / `ruff format` clean over
  `src`/`tests`/`experiments_text_processing`; the full `pytest` suite green.

  *(pre-registered 2026-11-06; **executed 2026-11-06** — the run is
  `omct-analyze-voyage-dict-abstain` (the artifact
  `results/omct-analyze-voyage-dict-abstain/analysis.json`, the
  `dictionary.abstention` readout and the model entry's `abstention_arms`),
  the log section is
  `## CS-56 — abstention on the pair unit: the floor's "impossible" is a
  cost` in `docs/experiments_text_processing.md`, and the numbers are below.
  **The registered falsifier fired** (the abstention is a lever); the chain
  continues to CS-57 on the reviewer's standing instruction, and the round
  ends there.)*

  **The round's numbers, in one screen (OMCT: the same 1,520 reviewed pairs,
  4,000 control cells and folds as CS-50; the fills change only the cells
  whose fold map is silent).**

  - **The cost.** **0 model calls**, **11.6 s**, **no new dependency**.
  - **The abstention test: the falsifier FIRES.** `neutral` **0.7616** against
    the floor's **0.7020** — **+0.0596 [+0.0291, +0.0927]** (P 1.000), above
    the ≥ +0.05 bar; per document 0.7120 / 0.8136 / 0.8435 against the floor's
    0.6219 / 0.7981 / 0.8338. The registered "the floor wins" is **refuted**.
  - **The optimistic fill: refuted too.** `max` **0.6105**, **−0.0914
    [−0.1438, −0.0374]** against the floor (the prediction said above; it is
    below — "possible" maximizes the inversions).
  - **The informativeness test: holds.** The 518 floor pairs are **77.80 %**
    negative (NU 338 pairs, 77.2 %; CIDH 12, 33.3 %; AI 168, 82.1 %); **85**
    silent questions; the floor share 34.08 %.
  - **The non-floor check: holds.** 0.8158 under all three fills.
  - **The reading.** The silence is informative as a **group** prior and the
    floor is the wrong **cell encoding**: it buries the 115 floored positives
    (22.2 % of the floor pairs) below every seen pair, and the pair-weighted
    AUC counts that error heavier than the 403 floored negatives' gain. The
    "no evidence" fill is the better encoding by +0.0596; the "impossible"
    should not ship on the pair unit.
  - **The gate.** **998** tests, ruff clean over **247** files, two consecutive
    runs bit-identical (bar the clock field).

