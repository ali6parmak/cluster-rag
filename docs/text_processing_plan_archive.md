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
  - **The cost.** The clause arm planned **3,892 calls** (3,669 of them new) and
    took **1,970 s**; the intact-text baselines were **CS-9b's cache hits**
    (1,320 of 1,320 planned calls already cached). The sentence arm cost 60
    calls and rests on **three genuinely multi-sentence texts** (653 of the 660
    val texts are a single sentence) — it is reported as the near-null it is.
