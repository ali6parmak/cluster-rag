# Text-processing track — experiment log

Append-only narrative log for the text-processing / segmentation track
(`TEXT_PROCESSING_PLAN.md`, revised 2026-10-23). One short section per run:
date, run_id, what was run, the numbers, the verdict.

The rules that bind this log (AGENTS.md plus the track plan):

- **Every number comes from a run artifact first** — `results/<run_id>/` — and
  is written here afterwards. No hand-written tables: the comparison tables
  are auto-appended by the CLI commands into
  `docs/experiments_table_text_processing_omct.md` (and
  `docs/experiments_table_text_processing_upr.md` once the UPR stage starts),
  upserted by run key.
- **Old entries are never edited.** A corrected run is appended as a new
  entry that says what changed.
- **OMCT is auxiliary.** The OMCT pairwise data (`omct-` run ids) is model
  selection, threshold evidence and reranker/fine-tune training material. Its
  findings annotate method choice; they are never a UPR metric. UPR numbers
  always name the split (val/test) they come from.
- **Comparisons are paired** and reported with a CI; no verdict rests on two
  independent AUCs.
- **The two negative subsets are never pooled**: A = the reviewer-verified
  hard negatives (the evaluation set), B = sampled *unjudged* same-document
  pairs (a control and a training source, never ground truth). B carries no
  label of its own: it is "unjudged", or "presumed-negative" where a
  label-shaped number is unavoidable — always flagged as such.
- **Raw data is read-only.** An LLM may score pairs; it may never relabel,
  curate or clean them.

## Change sets

Infrastructure change sets (no run artifact, hence no numbers):

- **CS-0 (2026-10-23) — skeleton.** The track's package
  (`src/cluster_rag/text_processors/`: the `Chunker` / `FeatureExtractor` /
  `Combiner` ports plus the `full_text` and `atoms` chunkers), the track's
  sweep directory `experiments_text_processing/` (the sweep loader keeps
  `experiments/` importable, so `embedder_pool.MODELS` stays shared), the
  track's log (this file) and its OMCT comparison table, and the
  command-derived command → (dataset, track) → table-file mapping with the
  new `paths.experiment_table_text_processing_omct` / `_upr` config keys.
  No run, no metric.

## Decisions (design, settled before the first run)

Recorded here so the numbers in CS-1's artifact have a stated design behind
them. The decisions themselves live in `TEXT_PROCESSING_PLAN.md` §3 Stage 0.

- **2026-10-23 — the control set (subset B).** Built from the **unjudged**
  cells (pairs with no record in `omct_pairs.json`): same-document grid →
  drop every judged cell → group the rest by document → seeded, proportional
  sample per document (seed from the config). Default **n = 4,000**
  (NU ~1,618 / CIDH ~2,210 / AI ~171; the AI stratum is the binding
  constraint). The dataset file is never modified — the control is derived
  and reproducible from the seed.
- **2026-10-23 — the reporting rule.** Every method number comes with its
  control-side counterpart: the change in AUC on the reviewed set
  (discrimination) sits next to the change in score level and rank position
  on the control (scale and mechanical gain). A method counts as a win only
  if the discrimination gain survives the scale shift.
- **2026-10-23 — models for the Stage-0 artifact:** `embeddinggemma`,
  `voyage-4-nano`, `arctic2` (warm cache; no re-encoding).
- **2026-10-23 — no human labelling.** The rate at which the unjudged cells
  hide true positives is **not measured**; the track's safe-negative rule
  therefore ships as the margin half only. Stated as a limitation in CS-2.

## Runs

### 2026-10-23 — `omct-analyze-arctic2` (Stage 0.1 artifact, arctic2)

- **What:** the Stage-0 pair-score artifact (TEXT_PROCESSING_PLAN.md section 3,
  Stage 0.1). Full-text cosine: the paragraph under the `document` kind, the
  question under `query` with the `strip_block` transform, both sides through
  the track's `full_text` chunker (one chunk per text, so the embedding cache
  stays warm: 225 + 294 cache hits, 0 encodes). Plus the **control set**
  (subset B): a seeded (seed 42), proportional sample of 4,000 *unjudged*
  same-document cells — quotas NU 1,619 / CIDH 2,210 / AI 171.
- **Numbers** (OMCT auxiliary data, never a UPR metric): 1,520 judged pairs
  (524 label-1 / 996 label-0) and 4,000 control cells; the same-document grid
  has 44,467 cells, 3.4% covered; 70 of 225 paragraphs carry no positive
  record.
  - score distributions (mean / p50 / p95): positives 0.3712 / 0.3740 /
    0.5307; verified negatives 0.3786 / 0.3892 / 0.5107; control 0.2436 /
    0.2386 / 0.3975.
  - pair AUC on the reviewed set: 0.4680 (NU 0.4773, CIDH 0.5408, AI 0.6209).
  - rank percentile (position from the top of the paragraph's own
    same-document candidate list; 0 = best, a random candidate sits at ~0.5):
    positives p50 0.0968, verified negatives 0.2066, control 0.5021.
  - top-1 same-document candidate per paragraph (model-dependent, **not** a
    dataset property): 55 given-positive / 17 given-negative / 153 unjudged.
- **Verdict:** the artifact stands up. Every section-0.1 indicative number
  about the *judged* set reproduces digit-for-digit (the score distributions,
  the pair AUC 0.4680, the rank percentiles 0.097 / 0.207 and the top-1
  provenance 55 / 17 / 153). The indicative **control** row does *not*
  reproduce (it was mean 0.289 / p50 0.287 / p95 0.443 with a rank p50 of
  0.565): that ad-hoc script's control construction was never recorded, and
  the control design settled on 2026-10-23 is deliberately *rank-unbiased* (a
  uniform draw over the unjudged cells), so its cells score lower and sit at
  rank p50 0.5021 (≈ random). The section-0.1 control row is superseded by
  this one.

### 2026-10-23 — `omct-analyze-embeddinggemma` (Stage 0.1 artifact, embeddinggemma)

- **What:** the same artifact with the second Stage-0 model (Ollama,
  `embeddinggemma:latest`, digest 82f094e4c0e1, 768d; the two Stage-2 axis
  models plus arctic2 are the plan's Stage-0 set). Same seed, so **exactly
  the same 4,000 control cells** as the arctic2 run.
- **Numbers:** score distributions — positives 0.3636 / 0.3626 / 0.5112;
  verified negatives 0.3465 / 0.3573 / 0.4907; control 0.2194 / 0.2130 /
  0.3682. Pair AUC (reviewed set) 0.5371 (NU 0.5231, CIDH 0.6263, AI 0.7633).
  Rank percentile p50: positives 0.0783, verified negatives 0.2488, control
  0.5042. Top-1 candidate per paragraph (model-dependent): 71 / 20 / 134.
- **Verdict:** the best discrimination of the three models on the reviewed
  set (AUC 0.5371) and the *highest* verified-negative rank position
  (0.2488) — its negatives sit lower in the candidate list than arctic2's
  (0.2066), so its margin over the positives is the widest of the three.
  Its control is also the lowest (mean 0.2194): the score level is not
  comparable across models, only the rank position and the AUC are.

### 2026-10-23 — `omct-analyze-voyage-4-nano` (Stage 0.1 artifact, voyage-4-nano)

- **What:** the same artifact with the third Stage-0 model (HF,
  `voyageai/voyage-4-nano`, revision 433ece5e27a1, 2048d). Same seed, same
  4,000 control cells.
- **Numbers:** score distributions — positives 0.3766 / 0.3741 / 0.5364;
  verified negatives 0.3721 / 0.3789 / 0.5256; control 0.2351 / 0.2297 /
  0.4050. Pair AUC (reviewed set) 0.5004 (NU 0.5027, CIDH 0.5573, AI
  0.6165). Rank percentile p50: positives 0.0980, verified negatives 0.2058,
  control 0.5023. Top-1 candidate per paragraph (model-dependent): 55 / 21 /
  149.
- **Verdict:** mid-pack on the reviewed set (AUC 0.5004, the pool's
  mid-pack value the plan expects) with a negative rank position very close
  to arctic2's (0.2058 vs 0.2066) — the same Stage-0 picture on a different
  model. Its control level (0.2351) sits between embeddinggemma's and
  arctic2's, again showing that the absolute score level is model-specific
  while the rank position and the AUC are the comparable numbers.

### Notes on the Stage-0 numbers (all three runs)

- **Conventions.** `rank_percentile` is the *position from the top* of the
  paragraph's own same-document candidate list: 0 = the best candidate, 1 =
  the worst ((n-1)/n for the bottom candidate of n), so a random candidate
  sits at ~0.5. The plan's section 0.1 writes "(1.0 = top)" in prose, but its
  own numbers (positives 0.097, negatives 0.207) correspond to 0 = top — the
  numbers win, and the artifact uses them. The cell ids follow the data
  file's own convention (`<code>-<paragraph number>-Q<question number>`, e.g.
  `NU-10-Q2`), so a judged cell's id equals its `pair_id`.
- **The two negative subsets are never pooled.** Subset A = the 996
  reviewer-verified hard negatives (ground truth, hard by construction);
  subset B = the 4,000 sampled *unjudged* cells (a control, never ground
  truth; its positive rate is unmeasured — the reviewer's 2026-10-23 decision
  rules out human labelling).
- **The control is reusable.** The seed, the sampling rule, the per-document
  quotas, the excluded judged cells and the drawn cell ids are in every
  artifact, so a later stage re-uses exactly these cells.
- **The score level is not comparable across models** (the three controls sit
  at 0.2194 / 0.2351 / 0.2436 mean for the same cells); the rank position
  and the AUC are.
- **Cross-check.** The artifact's pair AUC reproduces the Phase-2 probe's AUC
  for the same three models digit-for-digit (0.5371 / 0.5004 / 0.4680 against
  the probe's 0.5371 / 0.5004 / 0.4680) — even though the probe artifacts
  hold no per-pair scores, which is why the Stage-0 command had to exist.
- **The control is rank-unbiased, and that is measurable.** The share of a
  group that sits in the **top fifth** of its paragraph's candidate list:
  positives 0.6527 / 0.6679 / 0.6985, verified negatives 0.4930 / 0.4950 /
  0.4347, control 0.1923 / 0.2003 / 0.1930 (arctic2 / voyage-4-nano /
  embeddinggemma). A rank-unbiased sample must land at ≈0.20 — the control
  does (0.19–0.20), the reviewed set does not: the verified negatives sit
  around the middle of the list, the positives in its top third. That is the
  rank bias of the reviewed set, measured.

## Stage 0.2 — the failure-mode classification (CS-2, 2026-10-24)

**What this section is.** Stage 0 of the text-processing track
(`TEXT_PROCESSING_PLAN.md` §3) asks one question: *why does the OMCT
pair-relevance benchmark behave the way it does?* The plan names four
candidate failure modes, and this section says which of them the data
supports:

- **Representation** — the embedding cannot tell a topical match from an
  ordinary same-report pair.
- **Aggregation** — the way chunk-level scores are combined into one pair
  score loses matches.
- **Label incompleteness** — the labels miss true matches that exist in the
  grid.
- **Benchmark construction** — the way the reviewed set was assembled makes
  it a hard, rank-biased sample.

**Where the numbers come from.** Every number below is a field of one of
the three Stage-0 run artifacts — `results/omct-analyze-arctic2/analysis.json`,
`results/omct-analyze-voyage-4-nano/analysis.json`,
`results/omct-analyze-embeddinggemma/analysis.json` — or is explicitly
flagged as **derived** from those artifacts' own per-cell records, with the
derivation written out next to it. No number is computed from the raw pair
file. Two kinds of number are *not* used: anything from the plan's §0.1
ad-hoc table that CS-1 did not re-derive (the chunking-sensitivity table
and the corpus shape), which is quoted only where it is flagged as
indicative, and anything about UPR. This is the auxiliary OMCT pairwise
data: every finding here annotates method choice, and none of it is a UPR
metric.

**The three subsets — never pooled.**

| subset | what it is | n |
|---|---|---|
| **C — verified positives** | reviewed pairs a human marked label 1 | 524 |
| **A — verified hard negatives** | reviewed pairs a human rejected; the file's own `description` says they are the exp18 model's highest-scoring non-links, so they are hard *by construction* | 996 |
| **B — the control** | sampled **unjudged** same-document cells (seed 42, proportional: NU 1,619 / CIDH 2,210 / AI 171, identical in all three runs); it carries no label — "unjudged", or "presumed-negative" only where a label-shaped number is unavoidable | 4,000 |

Why A and B must never be pooled: an A cell is a pair a human looked at and
rejected; a B cell is a pair nobody looked at, and it may be a true match.

**The shape of the grid** (model-independent, from the artifacts'
`analysis` block): 225 paragraphs and 294 questions in three documents
(NU, CIDH, AI) form 44,467 same-document cells; 1,520 of them are judged
(3.42% covered) and 42,947 are unjudged. Per document (from the artifacts'
`analysis.strata`):

| document | judged pairs | positive share | paragraphs | paragraphs with no positive record | grid cells | coverage | control drawn (B) |
|---|---|---|---|---|---|---|---|
| NU | 845 | 211 (25.0%) | 75 | 29 | 18,225 | 4.64% | 1,619 |
| CIDH | 576 | 234 (40.6%) | 112 | 38 | 24,304 | 2.37% | 2,210 |
| AI | 99 | 79 (79.8%) | 38 | 3 | 1,938 | 5.11% | 171 |
| **pooled** | **1,520** | **524 (34.5%)** | **225** | **70** | **44,467** | **3.42%** | **4,000** |

The pooled positive share (34.5%) is not the corpus's positive rate: the
per-document macro is 48.5%, because NU holds 845 of the 1,520 reviewed
pairs. Every pooled number in this track needs its per-document macro beside
it.

### 1. Benchmark construction — supported, and the dominant effect

**What the mode means.** The reviewed set is not a sample of the grid. It
is the set of pairs one model (exp18) put on top, plus a human's verdict on
each of them. If the review concentrated on the top of each paragraph's
candidate list, then the reviewed positives *and* the reviewed negatives are
both top-of-list pairs, and the reviewed set cannot be read as "the corpus".

**What the data says.** The control is the only rank-unbiased sample of the
grid: a uniform draw over the unjudged cells, so it must land in the top
fifth of a paragraph's candidate list about 20% of the time. It does. The
reviewed set does not.

| share in the top fifth of its paragraph's candidate list | arctic2 | voyage-4-nano | embeddinggemma |
|---|---|---|---|
| verified positives (C) | 0.6527 | 0.6679 | 0.6985 |
| verified hard negatives (A) | 0.4930 | 0.4950 | 0.4347 |
| control, unjudged (B) | 0.1923 | 0.2003 | 0.1930 |

The review concentrated on the top of the list: about half of the verified
negatives sit in the top fifth, against 19–20% for an ordinary cell. Their
median position is at that fifth's boundary — the rank percentile (position
from the top of the paragraph's own candidate list) has p50 0.2066 / 0.2058 /
0.2488 for the verified negatives, so the median verified negative outscores
about 75–79% of its own paragraph's candidate list, against about 50% for an
ordinary cell (control p50 0.5021 / 0.5023 / 0.5042). The verified
positives sit higher still (p50 0.0968 / 0.0980 / 0.0783, i.e. they outscore
about 90% of their list).

**Consequence: the reviewed AUCs measure the set, not the models.**

| model | pair AUC (reviewed set, C vs A) | 95% interval (unpaired) | distance from chance | positives' mean (C) | verified negatives' mean (A) | control mean (B) | control rank p50 (B) |
|---|---|---|---|---|---|---|---|
| arctic2 | 0.4680 | [0.4377, 0.4982] | −2.08 SE | 0.3712 | 0.3786 | 0.2436 | 0.5021 |
| voyage-4-nano | 0.5004 | [0.4698, 0.5309] | +0.03 SE | 0.3766 | 0.3721 | 0.2351 | 0.5023 |
| embeddinggemma | 0.5371 | [0.5065, 0.5678] | +2.37 SE | 0.3636 | 0.3465 | 0.2194 | 0.5042 |

*Derived:* the intervals are the unpaired Hanley–McNeil standard error at
n = 524 positives / 996 negatives (SE 0.0154 / 0.0156 / 0.0157), computed
from the artifact's own AUC and class sizes. This is **not** a method
comparison — there is no second method — it calibrates the reviewed set
against chance, and it is the unpaired error because the artifact holds no
paired statistic. One model is *below* chance, one at chance, one 2.4 SE
above: the set does not rank these three models by topical quality. Never
read "AUC 0.47" as "the model is bad" — the reviewed negatives are the
exp18 model's false positives, so the reviewed AUC measures how hard the
reviewed set is and how much a model agrees with the ranking that produced
it.

The mean columns are the *score levels*, which are model-specific and not
comparable across models (the same 4,000 control cells score 0.2436 /
0.2351 / 0.2194 under the three models); the AUC and the rank position are
the comparable numbers.

**The whole pool is in the same band.** The 14-model probe sweep
(`omct-probe2-*` artifacts — the same 1,520 judged pairs and the same
measurement; the three Stage-0 models' AUCs reproduce digit-for-digit)
spans 0.4492 (nomic-v2-moe) to 0.5986 (f2llm-80m). No model in the pool
separates the reviewed positives from the reviewed hard negatives by more
than about 0.10 AUC. The set is hard.

**The "label-0 above label-1 p95" anomaly, quantified — and designed
around, never fixed.** The anomaly is real, and it is a *derived property of
the hard-negative construction*, not a bug: the negatives are the
top-scoring non-links of another model, so their upper tail overlaps the
positives' upper tail. *Derived* from the artifacts' per-pair records
(`judged_pairs`): the verified negatives scoring above the positives' 95th
percentile are 29 of 996 (2.9%) for arctic2 (their scores run 0.5312 to
0.6943), 40 of 996 (4.0%) for voyage-4-nano and 24 of 996 (2.4%) for
embeddinggemma. In the other direction, the verified positives scoring above
the negatives' 95th percentile are 46 of 524 (8.8%) / 30 of 524 (5.7%) /
45 of 524 (8.6%). So the overlap is a tail overlap of a few percent in both
directions, on top of two nearly identical distributions (arctic2: positives
0.3712 mean / 0.3740 median against negatives 0.3786 / 0.3892 — the rejected
pairs score *higher* on average than the accepted ones).

What the anomaly forbids: an absolute score threshold on this data. The
positives' p95 (0.5307 for arctic2) sits below 29 verified negatives'
scores, so "score above 0.53 ⇒ a match" is wrong on the *rejected* pairs
alone, and unmeasured on the unjudged ones. Every operating point in this
track is therefore a **rank**-based or **margin**-based rule, never an
absolute cut. The raw data is read-only: no pair is relabelled, and the
anomaly is designed around, not fixed.

**What it leaves open.** Whether the three models differ in topical quality
at all. On this reviewed set, the answer is "the set does not say".

### 2. Label incompleteness — supported on the positive side, unmeasured on the unjudged side

**What the mode means.** The positive set may be missing true matches, in
which case the labels understate how good a matcher is.

**What the data says.** The positive set is demonstrably not exhaustive: 70
of the 225 paragraphs carry no positive record at all (NU 29/75, CIDH
38/112, AI 3/38), and only 3.42% of the grid is judged (2.37%–5.11% per
document). A paragraph with no positive record is not a paragraph with no
match.

**What the data does not say.** How many of the 42,947 unjudged cells are
true positives. Measuring that needs human labelling, which the reviewer
ruled out on 2026-10-23, so the rate is unmeasured and stays unmeasured in
this track.

**Consequence, stated plainly.** The track's safe-negative rule (the plan's
§8 deliverable — "which unjudged pairs are safe negatives") ships as the
**margin half only**: a similarity-margin heuristic, without the
reviewer-confirmed spot-check the original formulation proposed. No spot-
check sheet: no human labelling happens in this track. The margin half is
therefore a heuristic whose false-negative rate is unknown, and it must not
be presented as a measured rate.

### 3. Representation — plausible, partially measured, not separable from §1

**What the mode means.** The embedding puts ordinary same-report pairs too
close to true matches, so the topical signal is compressed into a narrow
band.

**What the data says.** The control measures the *floor* — the level a
model gives to an ordinary same-document pair — and the floor is high. For
arctic2 the control's mean is 0.2436 against the positives' 0.3712, and
the control's 95th percentile (0.3975) sits **above** the positives' median
(0.3740). The same holds for the other two models (control p95 0.4050
against positives p50 0.3741; control p95 0.3682 against positives p50
0.3626). *Derived* from the artifacts' `control_cells` records: the
ordinary unjudged cells that outscore the median verified positive are 308
of 4,000 (7.7%) for arctic2, 344 of 4,000 (8.6%) for voyage-4-nano and 224
of 4,000 (5.6%) for embeddinggemma — roughly one ordinary same-document
cell in twelve to eighteen.

The same property shows up in three places at once across the three models:
the AUC (0.4680 / 0.5004 / 0.5371), the rank gap between the verified
positives and the verified negatives (rank p50 0.0968 against 0.2066 → gap
0.1098; 0.0980 against 0.2058 → 0.1077; 0.0783 against 0.2488 → 0.1705;
*derived* as the difference of two artifact fields), and the floor itself
(control mean 0.2436 / 0.2351 / 0.2194). The AUC ordering and the floor
ordering agree exactly — the model that separates a true match from a
near-miss better also keeps the floor lower — and the rank gap follows,
bar a 0.002 inversion between arctic2 (0.1098) and voyage-4-nano (0.1077).
That is one model property seen three times, not three effects.

**Why this stays a hypothesis.** The plan's candidate mechanism is a
dominant "same-report" direction in the embedding space — paragraphs from
one report are similar to each other whatever their topic — that lifts the
floor and squeezes the topical signal. The Stage-0 artifact cannot test
it: it holds scores, rank positions and distributions, not the embedding
space. Two further caveats: n = 3 models, and the reviewed set was built
from a *different* model's false positives, so part of what the AUC
measures is agreement with exp18's ranking rather than topical truth.
Settling the mechanism needs a new measurement (e.g. the share of the
similarity direction shared by all paragraphs of a document), i.e. a change
to the `analyze-omct` command and a re-run — a separate change set, not
this one.

**A methodological note that belongs here.** The AUC is a ranking
statistic: it is invariant under any monotone rescaling of the scores. The
score level and the AUC are therefore two symptoms of one model property,
never a cause-and-effect pair. This is why the control's score level is
quoted next to the AUC, never after it.

### 4. Aggregation — unmeasured

**What the mode means.** How chunk-level scores are combined into one pair
score (max, top-K with decay, local+global) changes the pair score; a bad
rule loses matches that are spread over several chunks.

**What the data says.** Nothing yet. The three Stage-0 runs score full
text against full text — the artifact records `chunker: full_text` with one
chunk per text (225 paragraph chunks, 294 question chunks, 0 re-encodes) —
so the harness holds no chunked scores and no aggregation number at all.

The only aggregation evidence in the repo is the plan's §0.1 ad-hoc table
(full text 0.4680 / paragraph chunked 0.4708 / question chunked 0.4507 /
both sides chunked 0.4354). It came from an ad-hoc script, CS-1 did not
re-derive it, and under the track's rule that only harness numbers are
reportable it is not a result. It is quoted here only to record that the
axis is open, and that the expected effect size (about ±0.03 AUC) sits at
the resolution limit of an unpaired comparison — which is why the plan's
Stage 2.5 makes every ladder comparison paired.

**Verdict on the four.** Benchmark construction: supported, and the
dominant effect. Label incompleteness: supported on the positive side,
unmeasured on the unjudged side. Representation: plausible and partially
measured — a real floor, visible in three places at once — but not separable
from the construction effect, with an untested mechanism. Aggregation:
unmeasured. In one sentence: *the benchmark's own construction explains the
low AUCs; the representation floor is the leading remaining hypothesis; the
positive set is known-incomplete; the aggregation axis has no harness
number.*

### What this data can and cannot answer

**It can:**

1. **Compare two scoring methods on the same cells**, paired: the
   discrimination (AUC on the reviewed set) next to the scale shift (the
   control's score level and rank position on the same 4,000 cells). The
   control is the yardstick that says whether a method separates more or
   merely re-scales.
2. **Measure a model's floor** — the level and the rank position it gives
   to ordinary same-document pairs — per document, on the control. It is a
   model-specific number and is not comparable across models.
3. **Describe the corpus**: the grid (44,467 cells), the coverage (3.42%),
   the per-document base rates (25.0% / 40.6% / 79.8%), the 70/225
   paragraphs that carry no positive record.
4. **Support a document-level holdout** (Stage 3's split): three documents,
   per-document strata, a control per document (NU 1,619 / CIDH 2,210 /
   AI 171).
5. **Re-use one control** across stages: the seed (42), the sampling rule,
   the per-document quotas and the cell ids are in every artifact, so a
   later stage measures against exactly these 4,000 cells.

**It cannot:**

1. **Say whether an unjudged cell is a true positive.** Unmeasured, and
   unmeasurable here (no human labelling). Hence the margin-half-only
   safe-negative rule, and no spot-check sheet.
2. **Say "153 paragraphs have no match".** The 153 is one model's top-1
   choice intersected with the file's coverage (arctic2: 55 given-positive /
   17 given-negative / 153 unjudged; voyage-4-nano 55 / 21 / 149;
   embeddinggemma 71 / 20 / 134) — model-dependent, and the three models
   disagree on it. The dataset-only counterpart is the 70 of 225 paragraphs
   that carry no positive record.
3. **Compare absolute scores across models**, or carry a threshold to
   another corpus, another model or UPR. The same 4,000 control cells score
   0.2194 / 0.2351 / 0.2436 under the three models, and the AUC is
   invariant under monotone rescaling.
4. **Measure embedding-model quality in the ordinary sense.** The reviewed
   negatives are another model's false positives, so the reviewed AUC
   measures how hard the set is and how much a model agrees with the
   ranking that produced it.
5. **Say anything about UPR.** OMCT is auxiliary data and the domain
   differs; no UPR metric is derivable from it. These findings annotate
   method choice only.
6. **Settle the representation mechanism** (the artifact holds no
   embedding-space measurement) **or the aggregation axis** (full text
   only).
7. **Treat the control as a negative set.** Its positive rate is
   unmeasured; it is a control and, from Stage 3 on, a training source —
   never ground truth, never pooled with subset A.

### The Stage-1 entry point this implies

**Attack first: the paragraph side's internal structure, measured against
the control rather than against the AUC.**

1. **The floor is the symptom the data actually attributes to the
   representation** (roughly one ordinary same-document cell in twelve to
   eighteen outscores the median true match; the control's p95 is above the
   positives' p50 for all three models), and segmentation is exactly the
   instrument that attacks a floor: a max over more chunks pushes every
   score up, so the question a chunked variant must answer is whether it
   raises the ordinary pairs as much as it raises the true matches. Only
   the control can answer that, because the control is rank-unbiased and
   the reviewed set is not.
2. **The reviewed AUC is the wrong entry point.** All three models sit
   within about 2.4 unpaired standard errors of chance there, so the set
   has almost no signal to amplify. Any Stage-1 comparison on it must be
   paired (the plan's Stage 2.5) and must be read as "on the reviewed
   set", never as "on the topic".
3. **The paragraph side is the untested half.** The question side's
   non-topical furniture (the "Answer choices:" block) is already removed
   from the Stage-0 baseline by the `strip_block` transform, and the
   Phase-2 probe already settled that stripping it raises the pair AUC for
   6 of 6 embedders. The paragraph side is the long, framed side — §0.1's
   indicative shape (mean 71 words / 478 characters per paragraph; "The
   Committee" in 33/225; not harness-re-derived) — and the plan's `atoms`
   chunker is the instrument for it.
4. **The reporting discipline Stage 1 must carry** (from here on): subset A
   and subset B separately, never pooled; the per-document macro next to
   every pooled number (pooled positive share 34.5% against a per-document
   macro of 48.5%); every method number with its control-side counterpart;
   paired statistics.

**What would falsify the representation hypothesis.** If a structural
variant lowers the control's floor *relative* to its shift on the verified
positives, the floor is representation-borne and the variant is worth
keeping. If it moves both equally, the variant has re-scaled the scores and
the representation is untouched. That is the test Stage 1 runs, and the
control set is what makes it decidable.

**Two implications for the later stages** (recorded here, not acted on in
this change set): the plan's Stage 1.1 splits into a question-side part the
Phase-2 probe has largely settled (the option block) and a paragraph-side
part with no harness number at all, so the entry point above is the
paragraph side, while the question-type features (Stage 1.1b) stay useful as
*features* for Stage 3; and the representation mechanism needs a new
measurement in the `analyze-omct` command, which is a separate change set
for the reviewer to approve.

### 2026-10-25 — `omct-analyze-arctic2-atoms` (Stage 1.1a arms, arctic2)

- **What:** the Stage-1.1a arms (TEXT_PROCESSING_PLAN.md section 3, Stage
  1.1a) with the third Stage-0 model (Ollama, `snowflake-arctic-embed2:latest`,
  digest 30685f516da4, 1024d). One run, six arms: the **baseline arm**
  (full text against full text — the Stage-0 scoring, the reference every
  other arm is compared with), the four **paragraph-atom arms** (the
  paragraph split into 1,742 sentence/clause atoms, folded into one pair score
  by `max` / `top_k_mean:2` / `top_k_mean:3` / `mean`) and the
  **question-transform arm** (the full text against the unstripped question,
  the free `keep` vs `strip_block` comparison). The question side is never
  chunked (out of scope). Cost: the atoms are 1,742 new vectors per model
  (the artifact's `sides.paragraph.n_chunks`) and the `keep` question side
  294; the baseline arm encodes **nothing** (225 + 294 cache hits) — the
  chunking version went into the *existing* cache-version slot, so the warm
  cache of the full-text arm stayed valid. (The artifact's cache counters
  are from the final, warm execution: 1,742 + 294 hits, 0 misses.)
- **Numbers** (OMCT auxiliary data, never a UPR metric; the control is the
  same 4,000 unjudged cells as the Stage-0 run):

  | arm | pair AUC | macro AUC | ΔAUC (95% CI, 2,000 paired resamples) | P(Δ>0) | de-shifted AUC | control mean (Δ) | floor-gap Δ |
  |---|---|---|---|---|---|---|---|
  | full_text (baseline) | 0.4680 | 0.5463 | — | — | — | 0.2436 | — |
  | full_text_keep | 0.4512 | 0.5467 | −0.0167 [−0.0329, −0.0005] | 0.023 | 0.4522 | 0.2452 (+0.0016) | +0.0031 |
  | atoms_max | 0.4865 | 0.5735 | +0.0186 [−0.0039, +0.0407] | 0.953 | 0.4698 | 0.2837 (+0.0401) | −0.0009 |
  | atoms_top_k_mean_2 | 0.4700 | 0.5496 | +0.0020 [−0.0184, +0.0222] | 0.569 | 0.4506 | 0.2592 (+0.0157) | −0.0135 |
  | atoms_top_k_mean_3 | 0.4628 | 0.5321 | −0.0052 [−0.0254, +0.0158] | 0.314 | 0.4428 | 0.2409 (−0.0027) | −0.0213 |
  | atoms_mean | 0.4341 | 0.5079 | −0.0338 [−0.0585, −0.0096] | 0.002 | 0.4179 | 0.1845 (−0.0590) | −0.0783 |

  - the **control-side counterpart**: the control's rank p50 barely moves —
    0.5021 (the baseline) against 0.5021 / 0.5021 / 0.5000 / 0.5021 / 0.4978
    for the five arms — so the arms move the score *level*, not the ordinary
    pairs' *rank position*;
  - the **top-fifth share** (C / A / B): baseline 0.6527 / 0.4930 / 0.1923;
    `atoms_max` 0.6546 / 0.4809 / 0.1985; `atoms_mean` 0.5840 / 0.4839 /
    0.1970;
  - the **hit@1** (the share of the group's pairs whose question is its
    paragraph's top candidate): positives 0.1050 → 0.1183 (`atoms_max`),
    0.1050 (`top2`), 0.0992 (`top3`), 0.0878 (`mean`); negatives 0.0171 →
    0.0171 / 0.0211 / 0.0191 / 0.0181 — the negatives' hit@1 does not fall
    with the positives', so nothing here separates the two groups;
  - the **per-document control-mean shift** (the plan's non-uniform caveat:
    NU / AI / CIDH): `atoms_max` +0.0221 / +0.0471 / +0.0528 — the inflation
    is not uniform, and it does not follow the paragraph length (NU has the
    longest paragraphs and the smallest shift);
  - the **per-document AUCs** (NU / AI / CIDH) beside the pooled number:
    baseline 0.4773 / 0.6209 / 0.5408 (pooled 0.4680, macro 0.5463) — the
    pooled figure is dominated by NU's 845 of the 1,520 reviewed pairs;
    `atoms_max`'s per-document AUCs are 0.5246 / 0.6867 / 0.5092 (macro
    0.5735, so the macro gain +0.0272 is larger than the pooled gain
    +0.0186) — two documents improve and one (CIDH) degrades;
  - a note on the de-shifted AUC: its **per-document** value equals the
    arm's per-document AUC (a constant shift inside a document cannot change
    that document's ranking) — the de-shifting is what corrects the
    *pooled* number, where the cross-document levels are at stake.
- **Verdict:** no arm improves the discrimination on the reviewed set: the
  only 95% intervals that exclude zero are negative (`full_text_keep`
  −0.0167 and `atoms_mean` −0.0338). The floor gap (the verified positives'
  mean minus the control's mean) *narrows* in four of the five arms
  (−0.0009 to −0.0783), so the ordinary same-document pairs did not fall
  relative to the true matches. The Stage-1 falsification test does not fire
  for this model.

### 2026-10-25 — `omct-analyze-voyage-4-nano-atoms` (Stage 1.1a arms, voyage-4-nano)

- **What:** the same six arms with the second Stage-0 model (HF,
  `voyageai/voyage-4-nano`, revision 433ece5e27a1, 2048d). Same seed, the
  same 4,000 control cells.
- **Numbers:**

  | arm | pair AUC | macro AUC | ΔAUC (95% CI) | P(Δ>0) | de-shifted AUC | control mean (Δ) | floor-gap Δ |
  |---|---|---|---|---|---|---|---|
  | full_text (baseline) | 0.5004 | 0.5588 | — | — | — | 0.2351 | — |
  | full_text_keep | 0.5174 | 0.5961 | +0.0170 [−0.0056, +0.0392] | 0.924 | 0.5286 | 0.3285 (+0.0934) | −0.0152 |
  | atoms_max | 0.4700 | 0.5144 | −0.0304 [−0.0522, −0.0092] | 0.003 | 0.4576 | 0.2940 (+0.0589) | −0.0147 |
  | atoms_top_k_mean_2 | 0.4497 | 0.4842 | −0.0507 [−0.0728, −0.0309] | 0.000 | 0.4352 | 0.2720 (+0.0369) | −0.0266 |
  | atoms_top_k_mean_3 | 0.4401 | 0.4679 | −0.0602 [−0.0827, −0.0394] | 0.000 | 0.4250 | 0.2563 (+0.0212) | −0.0341 |
  | atoms_mean | 0.4130 | 0.4513 | −0.0874 [−0.1130, −0.0629] | 0.000 | 0.3992 | 0.2119 (−0.0232) | −0.0874 |

  - the control's rank p50 stays at 0.5023 / 0.5023 / 0.4931 / 0.4958 /
    0.4931 (baseline 0.5023);
  - the top-fifth share (C / A / B): baseline 0.6679 / 0.4950 / 0.2003;
    `atoms_max` 0.5267 / 0.4598 / 0.2050; `atoms_mean` 0.3855 / 0.4297 /
    0.2042 — the atoms push the *positives* down the candidate list
    (0.6679 → 0.5267), and the negatives with them;
  - the hit@1: positives 0.1050 → 0.0706 / 0.0706 / 0.0630 / 0.0420; negatives
    0.0211 → 0.0231 / 0.0221 / 0.0251 / 0.0241 — the positives' hit@1 falls
    while the negatives' does not: the atoms lose the true matches first;
  - the per-document control-mean shift (NU / AI / CIDH): `atoms_max` +0.0380
    / +0.0512 / +0.0747 — again not uniform, and again not ordered by the
    paragraph length;
  - the per-document AUCs (NU / AI / CIDH): baseline 0.5027 / 0.6165 / 0.5573
    (pooled 0.5004, macro 0.5588).
- **Verdict:** the clearest negative result of the three: **every** atom arm
  is significantly worse on the reviewed set (the four intervals exclude
  zero, all negative, the largest −0.0874), and the floor gap narrows in
  every arm. The one arm with a positive de-shifted AUC above the baseline
  is the *question-side* one (`full_text_keep`, 0.5286 against 0.5004) — but
  its interval includes zero ([−0.0056, +0.0392]) and it comes with the
  largest scale shift in the run (the control mean +0.0934: the unstripped
  question lifts every score).

### 2026-10-25 — `omct-analyze-embeddinggemma-atoms` (Stage 1.1a arms, embeddinggemma)

- **What:** the same six arms with the first Stage-0 model (Ollama,
  `embeddinggemma:latest`, digest 82f094e4c0e1, 768d). Same seed, the same
  4,000 control cells.
- **Numbers:**

  | arm | pair AUC | macro AUC | ΔAUC (95% CI) | P(Δ>0) | de-shifted AUC | control mean (Δ) | floor-gap Δ |
  |---|---|---|---|---|---|---|---|
  | full_text (baseline) | 0.5371 | 0.6376 | — | — | — | 0.2194 | — |
  | full_text_keep | 0.5238 | 0.6240 | −0.0133 [−0.0296, +0.0018] | 0.046 | 0.5190 | 0.2201 (+0.0007) | +0.0004 |
  | atoms_max | 0.5526 | 0.6561 | +0.0155 [−0.0083, +0.0390] | 0.901 | 0.5430 | 0.2407 (+0.0212) | +0.0030 |
  | atoms_top_k_mean_2 | 0.5324 | 0.6358 | −0.0047 [−0.0259, +0.0165] | 0.328 | 0.5209 | 0.2154 (−0.0041) | −0.0101 |
  | atoms_top_k_mean_3 | 0.5264 | 0.6240 | −0.0107 [−0.0326, +0.0109] | 0.162 | 0.5158 | 0.1966 (−0.0228) | −0.0186 |
  | atoms_mean | 0.5388 | 0.6387 | +0.0016 [−0.0236, +0.0266] | 0.535 | 0.5318 | 0.1426 (−0.0768) | −0.0846 |

  - the control's rank p50 stays at 0.5069 / 0.5084 / 0.5098 / 0.5062 /
    0.5023 (baseline 0.5042);
  - the top-fifth share (C / A / B): baseline 0.6985 / 0.4347 / 0.1930;
    `atoms_max` 0.7271 / 0.4147 / 0.1888; `atoms_mean` 0.6469 / 0.3484 /
    0.1900;
  - the hit@1: positives 0.1355 → 0.1107 / 0.1145 / 0.1145 / 0.0992;
    negatives 0.0201 → 0.0171 / 0.0201 / 0.0141 / 0.0161;
  - the per-document control-mean shift (NU / AI / CIDH): `atoms_max` +0.0092
    / +0.0118 / +0.0307; `atoms_mean` −0.0777 / −0.0657 / −0.0771;
  - the per-document AUCs (NU / AI / CIDH): baseline 0.5231 / 0.7633 / 0.6263
    (pooled 0.5371, macro 0.6376).
- **Verdict:** the only **atom** arm of the whole Stage-1 run whose floor gap
  *widens* is this model's `atoms_max` (+0.0030), and its ΔAUC interval
  includes zero ([−0.0083, +0.0390]) — so it is the one candidate the
  falsification test does not rule out, and it is not significant.
  `atoms_mean` is the extreme case of the re-scaling story: the control's
  mean falls by 0.0768 while the discrimination is flat (+0.0016, the
  interval includes zero).

## Stage 1.1a — the paragraph side's atoms: the verdict (CS-3, 2026-10-25)

**The question.** CS-2's entry point: *does splitting the paragraph into
atoms lower the ordinary pairs' floor (the control) relative to its shift on
the true matches (subset C)?* If yes, the floor is representation-borne and
the variant is worth keeping. If both move equally, the method has merely
re-scaled the scores and the representation is untouched.

**The measurement.** Three runs (`omct-analyze-{arctic2,voyage-4-nano,
embeddinggemma}-atoms`), six arms each, all on the same 1,520 reviewed pairs
and the same 4,000 control cells, with the paired bootstrap of the AUC
difference (2,000 resamples, seed 42) and the plan's de-shifted AUC (the
control's **per-document** mean shift subtracted from the arm's reviewed-set
scores). The floor gap is the verified positives' mean score minus the
control's mean score; its change is the falsification statistic.

**The guard — the baseline arm reproduces Stage 0.** The full-text arm
inside each new run was compared with the Stage-0 artifact field by field
(the same script, both artifacts, never the raw pair file): the control cell
ids are identical (same cells, same order, 4,000 of them), and the baseline
arm's 1,520 per-pair scores and rank percentiles, the 4,000 control scores
and rank percentiles, the pair AUC, the per-document AUCs, the score
distributions, the negative-enrichment block and the top-1 provenance are
**identical** for all three models (0.4680 / 0.5004 / 0.5371; control means
0.2436 / 0.2351 / 0.2194). The only difference is a descriptive string: the
driver no longer takes "the first chunk" of a text but the arm's own chunks,
so the sides block now says "one chunk per text (the plain full-text
cosine)". The comparison is therefore *paired by construction*: same run,
same cells, same model.

**The answer: no — the falsification test does not fire.**

1. **The floor gap narrows in 8 of the 9 atom arms** (−0.0009 to −0.0874);
   it widens in exactly one (embeddinggemma's `atoms_max`, +0.0030, whose
   ΔAUC interval includes zero). The ordinary same-document pairs did not
   fall relative to the true matches — in most arms they rose relative to
   them.
2. **The mechanism is visible in the aggregation, and it is a re-scaling.**
   `max` takes a maximum over 7.7 atoms per paragraph on average, so it
   lifts every score (the control's mean +0.0212 / +0.0589 / +0.0401 for
   embeddinggemma / voyage-4-nano / arctic2) while the *rank* positions
   barely move (the control's rank p50 stays within 0.010 of the baseline's
   in every one of the fifteen arms). `mean` does the opposite: the control's
   mean falls (−0.0768 / −0.0232 / −0.0590) — but the true matches fall
   further, which is why the floor gap narrows by almost exactly the same
   amount.
3. **The discrimination does not follow.** Across the nine atom arms, only
   four have a positive ΔAUC at all (embeddinggemma's `atoms_max` +0.0155 and
   `atoms_mean` +0.0016, arctic2's `atoms_max` +0.0186 and
   `atoms_top_k_mean_2` +0.0020) and **none of the nine has a 95% interval
   that excludes zero on the positive side**. The five intervals that do
   exclude zero are all negative: voyage-4-nano's four atom arms (−0.0304,
   −0.0507, −0.0602, −0.0874) and arctic2's `atoms_mean` (−0.0338).
4. **The de-shifted AUC agrees** (the plan's rule: a discrimination gain
   counts only if it survives the scale shift). Against the baseline's
   pooled AUC, the de-shifted AUC is *below* it in 12 of the 15 arm
   comparisons; the three exceptions are embeddinggemma's `atoms_max`
   (0.5430 against 0.5371), arctic2's `atoms_max` (0.4698 against 0.4680) and
   voyage-4-nano's `full_text_keep` (0.5286 against 0.5004) — and the last
   one is the *question* side, not the paragraph side.

**What the run therefore says about the representation hypothesis.** The
paragraph side's internal structure is **not** the lever that lowers the
floor. The floor the Stage-0 write-up measured (ordinary same-document cells
scoring 0.2194–0.2436 on average, one ordinary cell in twelve to eighteen
outscoring the median true match) is not a "the paragraph is too coarse"
artefact: splitting the paragraph into atoms moves the floor *up* under a
max and *down* under a mean — but under a mean the true matches move down
further, so the floor does not fall *relative* to them. The
"same-report direction" mechanism therefore stays a **hypothesis** — this
change set measures the floor's behaviour under segmentation, not the
embedding space — and the measurement that could settle it (the share of
the similarity direction shared by all the paragraphs of a document) is still
an open change set in the plan's §5.

**The free question-side comparison (the `keep` vs `strip_block` arm).**
Model-dependent, and it does not reproduce the Phase-2 probe's "strip_block
wins on AUC 6/6": embeddinggemma −0.0133 [−0.0296, +0.0018] (P(Δ>0) 0.046),
voyage-4-nano **+0.0170** [−0.0056, +0.0392] (P(Δ>0) 0.924), arctic2 −0.0167
[−0.0329, −0.0005] (P(Δ>0) 0.023). So the unstripped question (with its
"Answer choices:" block) *helps* voyage-4-nano on this footing and hurts the
other two; the difference is not significant for the model that prefers it.
The control side shows why the question-side transform is a scale decision
and not a ranking decision: voyage-4-nano's control mean jumps +0.0934 (the
largest shift in the run) while its control rank p50 does not move at all
(0.5023 → 0.5023).

**What this run does not say.**

- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice only.
- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track), so the track's safe-negative rule
  still ships as the **margin half only** — no spot-check sheet.
- **Nothing about the embedding space.** The "same-report direction"
  mechanism is still a hypothesis; this run measures scores and rank
  positions, not directions.
- **Nothing about the plan's §0.1 ad-hoc chunking table** (0.4680 / 0.4708 /
  0.4507 / 0.4354). It is not re-derived here and it is not comparable: it
  measured word windows and a chunked question side with an unpaired AUC and
  no control; this run measures sentence/clause atoms with four aggregations,
  paired, against the control. The ad-hoc table stays indicative.
- **Nothing about the length↔shift correlation** the plan's caveat asks for
  (the paragraph lengths live in the raw pair file, which this track's log
  rule does not read). The per-document shifts are reported instead, and
  they show the inflation is not uniform — and not ordered by the paragraph
  length (NU has the longest paragraphs and the smallest shift).
- **Nothing about the other chunkers.** The word/sentence windows, the
  semantic boundaries and the question-side chunker are Stage 2's business;
  the aggregation axis now has a harness number only for the paragraph-side
  atoms.

**The next step.** Stage 2's ladder (fixed windows, sentence windows,
semantic boundaries) now has a *paired* yardstick and a control-side
counterpart to be measured against, and the paragraph-side atoms are
recorded as **not** helping on this data. The representation mechanism
(§5's open question) is the measurement that would still change the picture,
and it needs the reviewer's approval as a separate change set.


## Stage 2 — the segmentation ladder (CS-4, 2026-10-26)

**What.** The plan's Stage 2 in full: the **whole ladder**, not a reduced
grid, on the three Stage-0 models. Seven ladder variants × three models =
**21 runs** (`omct-analyze-<model>-<variant>`), each run holding the full-text
**baseline arm** (the Stage-0 scoring, the reference every other arm is
compared with) plus **six chunked arms** (one per aggregation rule). The
variants: `words15` / `words25` / `words40` (fixed word windows, exp03),
`sent2` (overlapping sentence windows, exp02), `semantic` (the semantic
boundaries, exp18(ii)) and `symwords25` / `symsent2` (both sides chunked,
exp10 / exp05). The arms: `max`, `top_k_mean:2`, `decayed_top_k:3:0.5:0.25`
(exp07), `local_global:0.5:max` and `local_global:0.75:max` (exp06) and
`peak:3:0.05` (exp08).

Every run uses the same seed (42) and therefore **the same 4,000 control
cells** as the Stage-0 and Stage-1 runs, and every arm is compared with the
baseline arm **inside the same run** — the pairing is by construction.

**Cost.** Each chunker needs its own vectors; the chunking version went into
the *existing* cache-version slot, so the warm cache stayed valid and the
full-text arm encoded **nothing** (225 + 294 cache hits per run). The new
vectors per model: `words15` 2,120 paragraph chunks (9.42 per paragraph),
`words25` 1,186 (5.27), `words40` 691 (3.07), `sent2` 409 (1.82), `semantic`
257–590 chunks per threshold over the six thresholds of the grid (plus the
598 atom vectors, shared by all thresholds), `symwords25` 1,186 paragraph +
657 question chunks, `symsent2` 409 + 338. Every run took 7–11 s.

**The guard.** The full-text arm inside every new run reproduces the Stage-0
artifact field by field: the same 4,000 control cell ids in the same order,
the same per-pair and per-cell scores and rank percentiles, the same AUCs
(0.4680 / 0.5004 / 0.5371) and control means (0.2436 / 0.2351 / 0.2194).
Checked with `tools/verify_analysis_baseline.py` (a new repo tool, the
CS-3 verification script generalised): **27 of 27 runs OK** (the three
Stage-0 runs, the three Stage-1 runs and the 21 new ones). The Stage-1 runs
were re-executed under the refactored driver and their arms and paired
statistics are **identical** to the CS-3 artifacts.

**The statistic that decides the comparison (new in this change set).** The
pooled AUC of the reviewed set compares **every** positive with **every**
negative, and on this data **58.7% of those pairs are cross-document** (41.3%
are within a document). A method that changes a document's score *level* —
and segmentation changes levels by construction — therefore moves the pooled
number without changing the ranking inside a document. The artifacts now
carry the split (`pair_auc_decomposition`: the within-document AUC and the
cross-document AUC) and a **paired bootstrap of the within-document AUC
difference** (`paired_within_group_auc_bootstrap`, stratified by document and
class, 2,000 resamples, seed 42). Both statistics are in the table
(`within_auc`, `cross_auc`, `within_d_auc`, `within_d_auc_ci`). The
within-document number is the one that transfers to a corpus without document
structure, so it is the number the verdict below is built on; the pooled
number is reported next to it, never alone.

**The swept parameter.** The semantic chunker's threshold is not a fixed
value: the plan's Stage 2.3 sweeps it **on fit-side documents only**. The
driver chooses it per document by the **document-level cross-fit** — for
every document, the threshold comes from the rotation that holds *that*
document out, chosen on the fit side (the other two documents) by the fit
side's pair AUC — so no document ever sees its own fitted value. The artifact
records the grid, the per-rotation fit-side and holdout numbers and the
chosen value (`threshold_selection`).

### The seven ladder variants (the 21 runs)

**words15** — fixed word windows, N=15, stride 7 (exp03)

- `omct-analyze-arctic2-words15` — baseline (full text): pooled AUC 0.4680, macro AUC 0.5463, within-document 0.5020, cross-document 0.4441, control mean 0.2436

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words15_max` | 0.4818 (0.5663) | +0.0238 [-0.0012, +0.0483] | +0.0139 [-0.0058, +0.0333] | 0.2972 (+0.0537) | +0.0011 | 9.42 |
  | `words15_top_k_mean_2` | 0.4628 (0.5446) | +0.0098 [-0.0143, +0.0341] | -0.0051 [-0.0245, +0.0140] | 0.2842 (+0.0407) | -0.0044 | 9.42 |
  | `words15_decayed_top_k_3_0.5_0.25` | 0.4557 (0.5348) | +0.0042 [-0.0199, +0.0282] | -0.0122 [-0.0317, +0.0061] | 0.2775 (+0.0340) | -0.0073 | 9.42 |
  | `words15_local_global_0.5_max` | 0.4728 (0.5574) | +0.0128 [+0.0013, +0.0241] | +0.0049 [-0.0045, +0.0142] | 0.2704 (+0.0268) | +0.0006 | 9.42 |
  | `words15_local_global_0.75_max` | 0.4766 (0.5639) | +0.0193 [+0.0010, +0.0371] | +0.0087 [-0.0056, +0.0231] | 0.2838 (+0.0403) | +0.0008 | 9.42 |
  | `words15_peak_3_0.05` | 0.4859 (0.5601) | +0.0275 [+0.0026, +0.0520] | +0.0180 [-0.0026, +0.0374] | 0.2788 (+0.0352) | +0.0015 | 9.42 |

- `omct-analyze-voyage-4-nano-words15` — baseline (full text): pooled AUC 0.5004, macro AUC 0.5588, within-document 0.5238, cross-document 0.4839, control mean 0.2351

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words15_max` | 0.4840 (0.5234) | -0.0184 [-0.0422, +0.0050] | -0.0164 [-0.0375, +0.0037] | 0.2980 (+0.0629) | -0.0046 | 9.42 |
  | `words15_top_k_mean_2` | 0.4730 (0.5121) | -0.0277 [-0.0508, -0.0048] | -0.0274 [-0.0477, -0.0077] | 0.2849 (+0.0498) | -0.0080 | 9.42 |
  | `words15_decayed_top_k_3_0.5_0.25` | 0.4641 (0.4993) | -0.0350 [-0.0578, -0.0120] | -0.0363 [-0.0567, -0.0167] | 0.2784 (+0.0433) | -0.0110 | 9.42 |
  | `words15_local_global_0.5_max` | 0.4912 (0.5396) | -0.0102 [-0.0228, +0.0022] | -0.0092 [-0.0199, +0.0011] | 0.2666 (+0.0315) | -0.0023 | 9.42 |
  | `words15_local_global_0.75_max` | 0.4875 (0.5304) | -0.0146 [-0.0333, +0.0037] | -0.0129 [-0.0292, +0.0026] | 0.2823 (+0.0472) | -0.0035 | 9.42 |
  | `words15_peak_3_0.05` | 0.4834 (0.5169) | -0.0203 [-0.0443, +0.0035] | -0.0170 [-0.0381, +0.0031] | 0.2801 (+0.0450) | -0.0049 | 9.42 |

- `omct-analyze-embeddinggemma-words15` — baseline (full text): pooled AUC 0.5371, macro AUC 0.6376, within-document 0.5632, cross-document 0.5188, control mean 0.2194

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words15_max` | 0.5611 (0.6648) | +0.0389 [+0.0134, +0.0643] | +0.0239 [+0.0022, +0.0461] | 0.2514 (+0.0319) | +0.0092 | 9.42 |
  | `words15_top_k_mean_2` | 0.5486 (0.6540) | +0.0308 [+0.0054, +0.0557] | +0.0115 [-0.0089, +0.0329] | 0.2374 (+0.0180) | +0.0035 | 9.42 |
  | `words15_decayed_top_k_3_0.5_0.25` | 0.5402 (0.6485) | +0.0256 [+0.0005, +0.0503] | +0.0031 [-0.0166, +0.0234] | 0.2306 (+0.0112) | +0.0000 | 9.42 |
  | `words15_local_global_0.5_max` | 0.5514 (0.6563) | +0.0270 [+0.0136, +0.0407] | +0.0142 [+0.0030, +0.0258] | 0.2354 (+0.0160) | +0.0046 | 9.42 |
  | `words15_local_global_0.75_max` | 0.5569 (0.6627) | +0.0360 [+0.0160, +0.0562] | +0.0197 [+0.0030, +0.0373] | 0.2434 (+0.0239) | +0.0069 | 9.42 |
  | `words15_peak_3_0.05` | 0.5624 (0.6659) | +0.0421 [+0.0164, +0.0682] | +0.0253 [+0.0029, +0.0477] | 0.2323 (+0.0129) | +0.0101 | 9.42 |

**words25** — fixed word windows, N=25, stride 12 (exp03)

- `omct-analyze-arctic2-words25` — baseline (full text): pooled AUC 0.4680, macro AUC 0.5463, within-document 0.5020, cross-document 0.4441, control mean 0.2436

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words25_max` | 0.4791 (0.5574) | +0.0292 [+0.0069, +0.0514] | +0.0111 [-0.0065, +0.0287] | 0.2829 (+0.0393) | +0.0062 | 5.27 |
  | `words25_top_k_mean_2` | 0.4721 (0.5471) | +0.0239 [+0.0016, +0.0457] | +0.0041 [-0.0139, +0.0216] | 0.2697 (+0.0261) | +0.0009 | 5.27 |
  | `words25_decayed_top_k_3_0.5_0.25` | 0.4730 (0.5504) | +0.0267 [+0.0054, +0.0472] | +0.0050 [-0.0121, +0.0213] | 0.2637 (+0.0201) | -0.0015 | 5.27 |
  | `words25_local_global_0.5_max` | 0.4728 (0.5549) | +0.0148 [+0.0040, +0.0257] | +0.0048 [-0.0036, +0.0133] | 0.2632 (+0.0197) | +0.0031 | 5.27 |
  | `words25_local_global_0.75_max` | 0.4759 (0.5570) | +0.0228 [+0.0064, +0.0393] | +0.0080 [-0.0053, +0.0212] | 0.2731 (+0.0295) | +0.0047 | 5.27 |
  | `words25_peak_3_0.05` | 0.4781 (0.5584) | +0.0240 [+0.0022, +0.0467] | +0.0102 [-0.0075, +0.0272] | 0.2662 (+0.0226) | +0.0046 | 5.27 |

- `omct-analyze-voyage-4-nano-words25` — baseline (full text): pooled AUC 0.5004, macro AUC 0.5588, within-document 0.5238, cross-document 0.4839, control mean 0.2351

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words25_max` | 0.4856 (0.5329) | -0.0154 [-0.0355, +0.0041] | -0.0148 [-0.0326, +0.0025] | 0.2764 (+0.0413) | +0.0023 | 5.27 |
  | `words25_top_k_mean_2` | 0.4809 (0.5188) | -0.0188 [-0.0385, +0.0012] | -0.0195 [-0.0376, -0.0022] | 0.2637 (+0.0286) | -0.0008 | 5.27 |
  | `words25_decayed_top_k_3_0.5_0.25` | 0.4768 (0.5132) | -0.0212 [-0.0402, -0.0020] | -0.0236 [-0.0407, -0.0074] | 0.2583 (+0.0232) | -0.0039 | 5.27 |
  | `words25_local_global_0.5_max` | 0.4925 (0.5474) | -0.0080 [-0.0184, +0.0023] | -0.0079 [-0.0173, +0.0008] | 0.2557 (+0.0206) | +0.0011 | 5.27 |
  | `words25_local_global_0.75_max` | 0.4885 (0.5400) | -0.0122 [-0.0279, +0.0030] | -0.0119 [-0.0256, +0.0014] | 0.2660 (+0.0309) | +0.0017 | 5.27 |
  | `words25_peak_3_0.05` | 0.4867 (0.5327) | -0.0127 [-0.0330, +0.0076] | -0.0137 [-0.0312, +0.0033] | 0.2604 (+0.0253) | +0.0011 | 5.27 |

- `omct-analyze-embeddinggemma-words25` — baseline (full text): pooled AUC 0.5371, macro AUC 0.6376, within-document 0.5632, cross-document 0.5188, control mean 0.2194

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words25_max` | 0.5602 (0.6733) | +0.0417 [+0.0179, +0.0649] | +0.0230 [+0.0056, +0.0420] | 0.2398 (+0.0203) | +0.0114 | 5.27 |
  | `words25_top_k_mean_2` | 0.5520 (0.6662) | +0.0365 [+0.0135, +0.0595] | +0.0149 [-0.0022, +0.0338] | 0.2271 (+0.0076) | +0.0056 | 5.27 |
  | `words25_decayed_top_k_3_0.5_0.25` | 0.5491 (0.6636) | +0.0361 [+0.0137, +0.0575] | +0.0120 [-0.0045, +0.0296] | 0.2213 (+0.0019) | +0.0016 | 5.27 |
  | `words25_local_global_0.5_max` | 0.5509 (0.6584) | +0.0263 [+0.0141, +0.0391] | +0.0138 [+0.0045, +0.0236] | 0.2296 (+0.0102) | +0.0057 | 5.27 |
  | `words25_local_global_0.75_max` | 0.5570 (0.6660) | +0.0366 [+0.0182, +0.0553] | +0.0199 [+0.0061, +0.0350] | 0.2347 (+0.0153) | +0.0086 | 5.27 |
  | `words25_peak_3_0.05` | 0.5611 (0.6754) | +0.0382 [+0.0157, +0.0619] | +0.0240 [+0.0056, +0.0430] | 0.2233 (+0.0039) | +0.0104 | 5.27 |

**words40** — fixed word windows, N=40, stride 20 (exp03)

- `omct-analyze-arctic2-words40` — baseline (full text): pooled AUC 0.4680, macro AUC 0.5463, within-document 0.5020, cross-document 0.4441, control mean 0.2436

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words40_max` | 0.4766 (0.5754) | +0.0250 [+0.0047, +0.0465] | +0.0086 [-0.0066, +0.0242] | 0.2655 (+0.0219) | +0.0105 | 3.07 |
  | `words40_top_k_mean_2` | 0.4763 (0.5768) | +0.0265 [+0.0080, +0.0458] | +0.0083 [-0.0064, +0.0232] | 0.2551 (+0.0116) | +0.0054 | 3.07 |
  | `words40_decayed_top_k_3_0.5_0.25` | 0.4740 (0.5766) | +0.0236 [+0.0061, +0.0424] | +0.0060 [-0.0080, +0.0200] | 0.2508 (+0.0073) | +0.0012 | 3.07 |
  | `words40_local_global_0.5_max` | 0.4717 (0.5614) | +0.0120 [+0.0023, +0.0223] | +0.0038 [-0.0035, +0.0113] | 0.2545 (+0.0110) | +0.0053 | 3.07 |
  | `words40_local_global_0.75_max` | 0.4742 (0.5690) | +0.0188 [+0.0036, +0.0349] | +0.0062 [-0.0053, +0.0179] | 0.2600 (+0.0164) | +0.0079 | 3.07 |
  | `words40_peak_3_0.05` | 0.4779 (0.5824) | +0.0224 [+0.0025, +0.0448] | +0.0100 [-0.0055, +0.0256] | 0.2523 (+0.0088) | +0.0068 | 3.07 |

- `omct-analyze-voyage-4-nano-words40` — baseline (full text): pooled AUC 0.5004, macro AUC 0.5588, within-document 0.5238, cross-document 0.4839, control mean 0.2351

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words40_max` | 0.4900 (0.5381) | -0.0104 [-0.0268, +0.0059] | -0.0103 [-0.0248, +0.0035] | 0.2570 (+0.0218) | +0.0101 | 3.07 |
  | `words40_top_k_mean_2` | 0.4891 (0.5362) | -0.0088 [-0.0242, +0.0069] | -0.0113 [-0.0254, +0.0022] | 0.2473 (+0.0122) | +0.0058 | 3.07 |
  | `words40_decayed_top_k_3_0.5_0.25` | 0.4865 (0.5362) | -0.0122 [-0.0269, +0.0021] | -0.0139 [-0.0272, -0.0015] | 0.2435 (+0.0084) | +0.0017 | 3.07 |
  | `words40_local_global_0.5_max` | 0.4954 (0.5491) | -0.0048 [-0.0131, +0.0034] | -0.0050 [-0.0123, +0.0019] | 0.2460 (+0.0109) | +0.0050 | 3.07 |
  | `words40_local_global_0.75_max` | 0.4927 (0.5436) | -0.0078 [-0.0201, +0.0046] | -0.0077 [-0.0184, +0.0028] | 0.2515 (+0.0164) | +0.0075 | 3.07 |
  | `words40_peak_3_0.05` | 0.4967 (0.5415) | -0.0056 [-0.0224, +0.0110] | -0.0037 [-0.0179, +0.0105] | 0.2443 (+0.0092) | +0.0076 | 3.07 |

- `omct-analyze-embeddinggemma-words40` — baseline (full text): pooled AUC 0.5371, macro AUC 0.6376, within-document 0.5632, cross-document 0.5188, control mean 0.2194

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words40_max` | 0.5474 (0.6603) | +0.0272 [+0.0065, +0.0475] | +0.0103 [-0.0050, +0.0263] | 0.2322 (+0.0127) | +0.0112 | 3.07 |
  | `words40_top_k_mean_2` | 0.5483 (0.6641) | +0.0314 [+0.0123, +0.0510] | +0.0112 [-0.0039, +0.0265] | 0.2218 (+0.0023) | +0.0054 | 3.07 |
  | `words40_decayed_top_k_3_0.5_0.25` | 0.5461 (0.6644) | +0.0304 [+0.0118, +0.0486] | +0.0089 [-0.0054, +0.0233] | 0.2177 (-0.0018) | +0.0017 | 3.07 |
  | `words40_local_global_0.5_max` | 0.5442 (0.6521) | +0.0184 [+0.0078, +0.0298] | +0.0071 [-0.0010, +0.0154] | 0.2258 (+0.0064) | +0.0056 | 3.07 |
  | `words40_local_global_0.75_max` | 0.5464 (0.6575) | +0.0241 [+0.0081, +0.0402] | +0.0093 [-0.0026, +0.0216] | 0.2290 (+0.0095) | +0.0084 | 3.07 |
  | `words40_peak_3_0.05` | 0.5490 (0.6635) | +0.0278 [+0.0077, +0.0484] | +0.0119 [-0.0032, +0.0279] | 0.2194 (+0.0000) | +0.0082 | 3.07 |

**sent2** — overlapping sentence windows, 2 sentences, step 1 (exp02)

- `omct-analyze-arctic2-sent2` — baseline (full text): pooled AUC 0.4680, macro AUC 0.5463, within-document 0.5020, cross-document 0.4441, control mean 0.2436

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `sent2s1_max` | 0.4797 (0.5570) | +0.0185 [+0.0042, +0.0335] | +0.0117 [+0.0004, +0.0235] | 0.2519 (+0.0083) | +0.0109 | 1.82 |
  | `sent2s1_top_k_mean_2` | 0.4758 (0.5665) | +0.0181 [+0.0056, +0.0313] | +0.0078 [-0.0021, +0.0175] | 0.2448 (+0.0012) | +0.0045 | 1.82 |
  | `sent2s1_decayed_top_k_3_0.5_0.25` | 0.4741 (0.5658) | +0.0119 [+0.0000, +0.0238] | +0.0061 [-0.0032, +0.0148] | 0.2430 (-0.0005) | -0.0008 | 1.82 |
  | `sent2s1_local_global_0.5_max` | 0.4738 (0.5531) | +0.0104 [+0.0033, +0.0180] | +0.0059 [+0.0001, +0.0118] | 0.2477 (+0.0042) | +0.0054 | 1.82 |
  | `sent2s1_local_global_0.75_max` | 0.4772 (0.5556) | +0.0151 [+0.0043, +0.0267] | +0.0092 [+0.0005, +0.0183] | 0.2498 (+0.0062) | +0.0082 | 1.82 |
  | `sent2s1_peak_3_0.05` | 0.4829 (0.5700) | +0.0171 [+0.0023, +0.0330] | +0.0149 [+0.0031, +0.0276] | 0.2446 (+0.0011) | +0.0061 | 1.82 |

- `omct-analyze-voyage-4-nano-sent2` — baseline (full text): pooled AUC 0.5004, macro AUC 0.5588, within-document 0.5238, cross-document 0.4839, control mean 0.2351

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `sent2s1_max` | 0.5059 (0.5516) | +0.0092 [-0.0029, +0.0208] | +0.0055 [-0.0051, +0.0156] | 0.2421 (+0.0070) | +0.0093 | 1.82 |
  | `sent2s1_top_k_mean_2` | 0.4976 (0.5464) | +0.0014 [-0.0086, +0.0112] | -0.0028 [-0.0113, +0.0056] | 0.2355 (+0.0004) | +0.0023 | 1.82 |
  | `sent2s1_decayed_top_k_3_0.5_0.25` | 0.4936 (0.5433) | -0.0068 [-0.0165, +0.0024] | -0.0068 [-0.0151, +0.0009] | 0.2341 (-0.0010) | -0.0026 | 1.82 |
  | `sent2s1_local_global_0.5_max` | 0.5030 (0.5547) | +0.0047 [-0.0012, +0.0105] | +0.0026 [-0.0027, +0.0077] | 0.2386 (+0.0035) | +0.0047 | 1.82 |
  | `sent2s1_local_global_0.75_max` | 0.5044 (0.5505) | +0.0069 [-0.0019, +0.0157] | +0.0040 [-0.0040, +0.0116] | 0.2404 (+0.0053) | +0.0070 | 1.82 |
  | `sent2s1_peak_3_0.05` | 0.5090 (0.5571) | +0.0069 [-0.0067, +0.0198] | +0.0086 [-0.0024, +0.0195] | 0.2353 (+0.0002) | +0.0045 | 1.82 |

- `omct-analyze-embeddinggemma-sent2` — baseline (full text): pooled AUC 0.5371, macro AUC 0.6376, within-document 0.5632, cross-document 0.5188, control mean 0.2194

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `sent2s1_max` | 0.5425 (0.6492) | +0.0205 [+0.0060, +0.0356] | +0.0053 [-0.0060, +0.0168] | 0.2235 (+0.0040) | +0.0098 | 1.82 |
  | `sent2s1_top_k_mean_2` | 0.5392 (0.6586) | +0.0209 [+0.0077, +0.0348] | +0.0020 [-0.0084, +0.0131] | 0.2165 (-0.0029) | +0.0032 | 1.82 |
  | `sent2s1_decayed_top_k_3_0.5_0.25` | 0.5384 (0.6529) | +0.0158 [+0.0038, +0.0282] | +0.0013 [-0.0085, +0.0106] | 0.2151 (-0.0044) | -0.0020 | 1.82 |
  | `sent2s1_local_global_0.5_max` | 0.5408 (0.6444) | +0.0130 [+0.0052, +0.0208] | +0.0037 [-0.0021, +0.0097] | 0.2214 (+0.0020) | +0.0049 | 1.82 |
  | `sent2s1_local_global_0.75_max` | 0.5420 (0.6466) | +0.0175 [+0.0062, +0.0291] | +0.0048 [-0.0038, +0.0136] | 0.2224 (+0.0030) | +0.0074 | 1.82 |
  | `sent2s1_peak_3_0.05` | 0.5484 (0.6566) | +0.0189 [+0.0042, +0.0340] | +0.0113 [-0.0007, +0.0231] | 0.2162 (-0.0032) | +0.0054 | 1.82 |

**semantic** — semantic boundaries: cut where the adjacent atoms' cosine drops (exp18(ii))

- `omct-analyze-arctic2-semantic` — baseline (full text): pooled AUC 0.4680, macro AUC 0.5463, within-document 0.5020, cross-document 0.4441, control mean 0.2436

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `semantic_max` | 0.4853 (0.5600) | +0.0195 [+0.0005, +0.0380] | +0.0174 [+0.0022, +0.0323] | 0.2617 (+0.0181) | +0.0056 | 1.94 |
  | `semantic_top_k_mean_2` | 0.5042 (0.5747) | -0.0029 [-0.0259, +0.0202] | +0.0363 [+0.0166, +0.0548] | 0.2244 (-0.0192) | -0.0138 | 1.94 |
  | `semantic_decayed_top_k_3_0.5_0.25` | 0.4902 (0.5346) | -0.0246 [-0.0480, -0.0018] | +0.0222 [+0.0014, +0.0409] | 0.2229 (-0.0207) | -0.0220 | 1.94 |
  | `semantic_local_global_0.5_max` | 0.4767 (0.5545) | +0.0115 [+0.0020, +0.0210] | +0.0087 [+0.0012, +0.0164] | 0.2526 (+0.0091) | +0.0028 | 1.94 |
  | `semantic_local_global_0.75_max` | 0.4812 (0.5583) | +0.0160 [+0.0020, +0.0298] | +0.0132 [+0.0020, +0.0246] | 0.2571 (+0.0136) | +0.0042 | 1.94 |
  | `semantic_peak_3_0.05` | 0.4789 (0.5541) | +0.0048 [-0.0106, +0.0196] | +0.0110 [-0.0030, +0.0251] | 0.2475 (+0.0039) | +0.0030 | 1.94 |

- `omct-analyze-voyage-4-nano-semantic` — baseline (full text): pooled AUC 0.5004, macro AUC 0.5588, within-document 0.5238, cross-document 0.4839, control mean 0.2351

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `semantic_max` | 0.5108 (0.5543) | +0.0062 [+0.0023, +0.0112] | +0.0104 [+0.0054, +0.0157] | 0.2348 (-0.0003) | +0.0006 | 1.65 |
  | `semantic_top_k_mean_2` | 0.5082 (0.5355) | -0.0569 [-0.0787, -0.0346] | +0.0078 [-0.0138, +0.0290] | 0.2209 (-0.0142) | -0.0336 | 1.65 |
  | `semantic_decayed_top_k_3_0.5_0.25` | 0.5011 (0.5283) | -0.0676 [-0.0900, -0.0441] | +0.0007 [-0.0220, +0.0225] | 0.2190 (-0.0161) | -0.0397 | 1.65 |
  | `semantic_local_global_0.5_max` | 0.5056 (0.5564) | +0.0029 [+0.0012, +0.0048] | +0.0052 [+0.0026, +0.0079] | 0.2350 (-0.0001) | +0.0003 | 1.65 |
  | `semantic_local_global_0.75_max` | 0.5080 (0.5536) | +0.0043 [+0.0016, +0.0078] | +0.0076 [+0.0038, +0.0116] | 0.2349 (-0.0002) | +0.0004 | 1.65 |
  | `semantic_peak_3_0.05` | 0.5138 (0.5624) | +0.0048 [-0.0048, +0.0147] | +0.0134 [+0.0027, +0.0245] | 0.2338 (-0.0013) | +0.0028 | 1.65 |

- `omct-analyze-embeddinggemma-semantic` — baseline (full text): pooled AUC 0.5371, macro AUC 0.6376, within-document 0.5632, cross-document 0.5188, control mean 0.2194

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `semantic_max` | 0.5549 (0.6610) | +0.0264 [+0.0073, +0.0443] | +0.0177 [+0.0025, +0.0336] | 0.2223 (+0.0028) | +0.0063 | 2.01 |
  | `semantic_top_k_mean_2` | 0.6131 (0.6405) | -0.0003 [-0.0237, +0.0241] | +0.0760 [+0.0508, +0.1004] | 0.1951 (-0.0244) | -0.0272 | 2.01 |
  | `semantic_decayed_top_k_3_0.5_0.25` | 0.5816 (0.6269) | -0.0253 [-0.0499, -0.0013] | +0.0445 [+0.0225, +0.0670] | 0.1950 (-0.0244) | -0.0180 | 2.01 |
  | `semantic_local_global_0.5_max` | 0.5467 (0.6523) | +0.0157 [+0.0059, +0.0253] | +0.0096 [+0.0018, +0.0177] | 0.2208 (+0.0014) | +0.0032 | 2.01 |
  | `semantic_local_global_0.75_max` | 0.5500 (0.6571) | +0.0205 [+0.0062, +0.0340] | +0.0129 [+0.0014, +0.0248] | 0.2215 (+0.0020) | +0.0048 | 2.01 |
  | `semantic_peak_3_0.05` | 0.5527 (0.6633) | +0.0238 [+0.0039, +0.0432] | +0.0155 [-0.0010, +0.0322] | 0.2097 (-0.0097) | +0.0084 | 2.01 |

**symwords25** — both sides word-chunked, N=25, stride 12 (exp10)

- `omct-analyze-arctic2-symwords25` — baseline (full text): pooled AUC 0.4680, macro AUC 0.5463, within-document 0.5020, cross-document 0.4441, control mean 0.2436

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words25_max` | 0.4357 (0.5112) | -0.0169 [-0.0402, +0.0084] | -0.0323 [-0.0522, -0.0134] | 0.2940 (+0.0504) | +0.0013 | 5.27 |
  | `words25_top_k_mean_2` | 0.4293 (0.4979) | -0.0236 [-0.0469, +0.0017] | -0.0386 [-0.0582, -0.0200] | 0.2808 (+0.0372) | -0.0036 | 5.27 |
  | `words25_decayed_top_k_3_0.5_0.25` | 0.4303 (0.4996) | -0.0211 [-0.0440, +0.0031] | -0.0376 [-0.0562, -0.0195] | 0.2748 (+0.0312) | -0.0062 | 5.27 |
  | `words25_local_global_0.5_max` | 0.4507 (0.5251) | -0.0077 [-0.0197, +0.0046] | -0.0172 [-0.0270, -0.0081] | 0.2688 (+0.0252) | +0.0006 | 5.27 |
  | `words25_local_global_0.75_max` | 0.4426 (0.5218) | -0.0123 [-0.0304, +0.0067] | -0.0254 [-0.0400, -0.0109] | 0.2814 (+0.0378) | +0.0010 | 5.27 |
  | `words25_peak_3_0.05` | 0.4369 (0.5097) | -0.0185 [-0.0420, +0.0070] | -0.0311 [-0.0512, -0.0121] | 0.2774 (+0.0338) | -0.0004 | 5.27 |

- `omct-analyze-voyage-4-nano-symwords25` — baseline (full text): pooled AUC 0.5004, macro AUC 0.5588, within-document 0.5238, cross-document 0.4839, control mean 0.2351

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words25_max` | 0.4489 (0.5145) | -0.0512 [-0.0731, -0.0295] | -0.0515 [-0.0703, -0.0332] | 0.2973 (+0.0622) | -0.0034 | 5.27 |
  | `words25_top_k_mean_2` | 0.4450 (0.5031) | -0.0542 [-0.0756, -0.0329] | -0.0554 [-0.0741, -0.0372] | 0.2846 (+0.0495) | -0.0064 | 5.27 |
  | `words25_decayed_top_k_3_0.5_0.25` | 0.4417 (0.4988) | -0.0563 [-0.0773, -0.0369] | -0.0587 [-0.0767, -0.0410] | 0.2791 (+0.0440) | -0.0097 | 5.27 |
  | `words25_local_global_0.5_max` | 0.4719 (0.5365) | -0.0287 [-0.0404, -0.0170] | -0.0285 [-0.0388, -0.0188] | 0.2662 (+0.0311) | -0.0017 | 5.27 |
  | `words25_local_global_0.75_max` | 0.4595 (0.5256) | -0.0415 [-0.0586, -0.0247] | -0.0409 [-0.0558, -0.0267] | 0.2817 (+0.0466) | -0.0026 | 5.27 |
  | `words25_peak_3_0.05` | 0.4498 (0.5126) | -0.0498 [-0.0717, -0.0281] | -0.0505 [-0.0695, -0.0322] | 0.2814 (+0.0463) | -0.0049 | 5.27 |

- `omct-analyze-embeddinggemma-symwords25` — baseline (full text): pooled AUC 0.5371, macro AUC 0.6376, within-document 0.5632, cross-document 0.5188, control mean 0.2194

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `words25_max` | 0.4960 (0.6119) | -0.0234 [-0.0482, +0.0024] | -0.0411 [-0.0611, -0.0207] | 0.2550 (+0.0355) | +0.0064 | 5.27 |
  | `words25_top_k_mean_2` | 0.4897 (0.6064) | -0.0265 [-0.0504, -0.0016] | -0.0474 [-0.0670, -0.0270] | 0.2422 (+0.0227) | +0.0004 | 5.27 |
  | `words25_decayed_top_k_3_0.5_0.25` | 0.4852 (0.5959) | -0.0289 [-0.0518, -0.0053] | -0.0520 [-0.0714, -0.0321] | 0.2364 (+0.0170) | -0.0036 | 5.27 |
  | `words25_local_global_0.5_max` | 0.5179 (0.6320) | -0.0074 [-0.0206, +0.0063] | -0.0192 [-0.0293, -0.0086] | 0.2372 (+0.0178) | +0.0032 | 5.27 |
  | `words25_local_global_0.75_max` | 0.5075 (0.6247) | -0.0134 [-0.0333, +0.0072] | -0.0297 [-0.0452, -0.0137] | 0.2461 (+0.0266) | +0.0048 | 5.27 |
  | `words25_peak_3_0.05` | 0.4965 (0.6181) | -0.0280 [-0.0525, -0.0018] | -0.0406 [-0.0605, -0.0200] | 0.2385 (+0.0191) | +0.0056 | 5.27 |

**symsent2** — both sides sentence-windowed, 2 sentences, step 1 (exp05)

- `omct-analyze-arctic2-symsent2` — baseline (full text): pooled AUC 0.4680, macro AUC 0.5463, within-document 0.5020, cross-document 0.4441, control mean 0.2436

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `sent2s1_max` | 0.4690 (0.5434) | +0.0076 [-0.0074, +0.0231] | +0.0011 [-0.0112, +0.0135] | 0.2532 (+0.0096) | +0.0100 | 1.82 |
  | `sent2s1_top_k_mean_2` | 0.4645 (0.5510) | +0.0059 [-0.0075, +0.0198] | -0.0035 [-0.0141, +0.0069] | 0.2461 (+0.0025) | +0.0035 | 1.82 |
  | `sent2s1_decayed_top_k_3_0.5_0.25` | 0.4620 (0.5499) | -0.0009 [-0.0134, +0.0119] | -0.0059 [-0.0158, +0.0037] | 0.2444 (+0.0008) | -0.0018 | 1.82 |
  | `sent2s1_local_global_0.5_max` | 0.4687 (0.5464) | +0.0050 [-0.0025, +0.0129] | +0.0008 [-0.0054, +0.0072] | 0.2484 (+0.0048) | +0.0050 | 1.82 |
  | `sent2s1_local_global_0.75_max` | 0.4693 (0.5453) | +0.0071 [-0.0041, +0.0190] | +0.0014 [-0.0079, +0.0109] | 0.2508 (+0.0072) | +0.0075 | 1.82 |
  | `sent2s1_peak_3_0.05` | 0.4733 (0.5579) | +0.0070 [-0.0087, +0.0232] | +0.0054 [-0.0070, +0.0187] | 0.2460 (+0.0024) | +0.0051 | 1.82 |

- `omct-analyze-voyage-4-nano-symsent2` — baseline (full text): pooled AUC 0.5004, macro AUC 0.5588, within-document 0.5238, cross-document 0.4839, control mean 0.2351

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `sent2s1_max` | 0.4994 (0.5487) | +0.0015 [-0.0113, +0.0144] | -0.0010 [-0.0125, +0.0099] | 0.2418 (+0.0067) | +0.0091 | 1.82 |
  | `sent2s1_top_k_mean_2` | 0.4917 (0.5461) | -0.0055 [-0.0163, +0.0056] | -0.0087 [-0.0183, +0.0006] | 0.2352 (+0.0001) | +0.0022 | 1.82 |
  | `sent2s1_decayed_top_k_3_0.5_0.25` | 0.4878 (0.5429) | -0.0135 [-0.0239, -0.0030] | -0.0126 [-0.0220, -0.0039] | 0.2338 (-0.0013) | -0.0027 | 1.82 |
  | `sent2s1_local_global_0.5_max` | 0.4993 (0.5548) | +0.0001 [-0.0063, +0.0063] | -0.0011 [-0.0070, +0.0042] | 0.2384 (+0.0033) | +0.0046 | 1.82 |
  | `sent2s1_local_global_0.75_max` | 0.4990 (0.5534) | +0.0005 [-0.0089, +0.0103] | -0.0014 [-0.0100, +0.0067] | 0.2401 (+0.0050) | +0.0069 | 1.82 |
  | `sent2s1_peak_3_0.05` | 0.5034 (0.5575) | -0.0000 [-0.0140, +0.0138] | +0.0030 [-0.0087, +0.0143] | 0.2350 (-0.0001) | +0.0046 | 1.82 |

- `omct-analyze-embeddinggemma-symsent2` — baseline (full text): pooled AUC 0.5371, macro AUC 0.6376, within-document 0.5632, cross-document 0.5188, control mean 0.2194

  | arm | pooled AUC (macro) | within-document ΔAUC (95% CI) | pooled ΔAUC (95% CI) | control mean (Δ) | floor-gap Δ | chunks/text |
  |---|---|---|---|---|---|---|
  | `sent2s1_max` | 0.5186 (0.6233) | -0.0037 [-0.0184, +0.0117] | -0.0186 [-0.0305, -0.0061] | 0.2258 (+0.0064) | +0.0083 | 1.82 |
  | `sent2s1_top_k_mean_2` | 0.5149 (0.6275) | -0.0039 [-0.0180, +0.0102] | -0.0222 [-0.0334, -0.0107] | 0.2189 (-0.0005) | +0.0018 | 1.82 |
  | `sent2s1_decayed_top_k_3_0.5_0.25` | 0.5131 (0.6218) | -0.0101 [-0.0229, +0.0032] | -0.0240 [-0.0344, -0.0135] | 0.2175 (-0.0020) | -0.0035 | 1.82 |
  | `sent2s1_local_global_0.5_max` | 0.5294 (0.6339) | +0.0013 [-0.0064, +0.0092] | -0.0077 [-0.0138, -0.0013] | 0.2226 (+0.0032) | +0.0042 | 1.82 |
  | `sent2s1_local_global_0.75_max` | 0.5242 (0.6286) | -0.0008 [-0.0122, +0.0111] | -0.0130 [-0.0222, -0.0034] | 0.2242 (+0.0048) | +0.0062 | 1.82 |
  | `sent2s1_peak_3_0.05` | 0.5230 (0.6311) | -0.0078 [-0.0229, +0.0079] | -0.0141 [-0.0267, -0.0013] | 0.2186 (-0.0009) | +0.0038 | 1.82 |


## Stage 2 — the verdict (CS-4, 2026-10-26)

**The question.** The plan's Stage 2 exit: *which segmentation + aggregation
wins where — or the honest "no chunking beats full text"?* The answer is
neither of the two simple answers: **chunking the paragraph side helps on two
of the three models, chunking the question side hurts on all three, and the
pooled AUC on the reviewed set is the wrong headline.**

**1. The one-sided ladder (paragraph chunked, question whole) improves the
within-document ranking on 2 of the 3 models.** Counted over the 42 arms of
each model (7 variants × 6 rules), the within-document ΔAUC's 95% interval
excludes zero:

| model | arms with a **positive** within-document interval | arms with a **negative** one | the best arm (within-document ΔAUC, 95% CI) |
|---|---|---|---|
| arctic2 | **24** | 1 | `words25_max` +0.0292 [+0.0069, +0.0514] |
| embeddinggemma | **28** | 4 | `words15_peak_3_0.05` +0.0421 [+0.0164, +0.0682] |
| voyage-4-nano | 3 | 12 | `sent2s1_max` +0.0092 [−0.0029, +0.0208] (not significant); the only significant ones are the semantic chunker's `max` (+0.0062 [+0.0023, +0.0112]) and its two local+global blends (+0.0029, +0.0043) |

The effect size is **+0.01 to +0.04 within-document AUC** — the same order as
the plan's expected ±0.03, but now measured *paired* and *per document*. It
is **not** a property of the model alone: voyage-4-nano's ladder is flat to
negative, so a segmentation recipe does not transfer across embedders without
being re-measured.

**2. The pooled AUC on the reviewed set is a misleading headline, and the
decomposition shows why.** The largest pooled gain in the whole ladder is
embeddinggemma's `semantic_top_k_mean_2`: **+0.0760 [+0.0508, +0.1004]**. Its
within-document ΔAUC is **−0.0003 [−0.0237, +0.0241]** and its cross-document
ΔAUC is **+0.1296**: the entire gain is the ranking *across* the three
documents, i.e. the documents' score levels moved, not the topical ranking
inside them. The same arm for arctic2: pooled +0.0363, within-document
−0.0029. The `decayed_top_k` arm is worse still: embeddinggemma's pooled
+0.0445 comes with a within-document **−0.0253 [−0.0499, −0.0013]** (significantly
*negative*), and voyage-4-nano's `semantic_decayed_top_k` has a within-document
**−0.0676 [−0.0900, −0.0441]** against a pooled +0.0007. A headline built on
the pooled AUC would have adopted the two arms that hurt the ranking inside a
document.

**3. The control side shows what the pooled number hides: the max-over-k
inflation.** The chunked arms lift the control's score *level* — and the lift
grows with the number of chunks per text (the plan's Stage 2.5 point):

| variant | chunks per paragraph | the control's mean shift (arctic2 / voyage-4-nano / embeddinggemma) |
|---|---|---|
| `words15` | 9.42 | +0.0537 / +0.0629 / +0.0319 |
| `words25` | 5.27 | +0.0393 / +0.0413 / +0.0203 |
| `words40` | 3.07 | +0.0219 / +0.0218 / +0.0127 |
| `sent2` | 1.82 | +0.0083 / +0.0070 / +0.0040 |
| `semantic` | 1.65–2.01 | +0.0181 / −0.0003 / +0.0028 (the `max` arm) |

The control's *rank* position barely moves (its rank p50 stays within ±0.01
of the baseline's in every arm), so the inflation is a level effect, not a
re-ranking of the ordinary pairs.

**4. The floor gap — the track's falsification statistic — fires in the
peak-like arms and fails in the diluted ones.** The floor gap (the verified
positives' mean minus the control's mean) *widens* in the arms that keep the
peak: `sent2s1_max` +0.0109 (arctic2) and +0.0098 (embeddinggemma),
`words40_max` +0.0105 / +0.0112, `words25_max` +0.0062 / +0.0114,
`words15_peak_3_0.05` +0.0015 / +0.0101. It *narrows* in the arms that
dilute the evidence over many chunks (`top_k_mean`, `decayed_top_k`) and in
the semantic variant's diluted arms (−0.0138 to −0.0397). So: the
paragraph's internal structure lowers the ordinary pairs' floor relative to
the true matches **only** when the aggregation keeps the peak instead of
averaging it away.

**5. The symmetric variants (both sides chunked) hurt — on all three
models.** `symwords25` is the worst arm family in the ladder: the
within-document ΔAUC is negative in every arm of every model
(arctic2 −0.0077 to −0.0236, voyage-4-nano −0.0287 to −0.0563,
embeddinggemma −0.0074 to −0.0289) and the pooled ΔAUC is significantly
negative in all 18 arms. `symsent2` is flat (the questions are short, so the
sentence windows give 2.6 chunks per question). The reading: the question
side's furniture (the "Answer choices:" block is already removed by the
`strip_block` transform) is not improved by chunking, and the max over a
question's chunks is noise. **Chunk the paragraph, never the question.**

**6. The swept semantic threshold does not transfer.** The fit side's choice
is unstable — the chosen threshold differs per document *and* per rule (e.g.
embeddinggemma's `max` arm chose 0.7 / 0.7 / 0.7, its `top_k_mean_2` chose
0.2 / 0.2 / 0.2, arctic2's `top_k_mean_2` chose 0.4 / 0.5 / 0.5) — and the
fit-side AUC does not predict the holdout's. The clearest case: arctic2's
`top_k_mean_2` with the NU document held out chose 0.4 on the fit side (fit
AUC 0.5886) and scored **0.4565** on the holdout, *below* the full-text
baseline's NU AUC of 0.4773, while the grid's best holdout value was 0.5663
at another threshold. With three documents, one rotation's fit side carries
two documents' worth of pairs, and the "best threshold" is largely noise.
**The semantic chunker is therefore not a reliable recipe**; where it helps
(as in arctic2's and embeddinggemma's `max` arms, +0.0195 and +0.0264
within-document with intervals excluding zero), it is the *peak* behaviour
that helps, not the boundary detection.

**The ranked statement (the plan's Stage-2 exit).** On the OMCT pairwise
data, with three models, 21 runs and 126 chunked arms, paired throughout:

1. **Chunk the paragraph, keep the question whole.** The paragraph-side
   windows are the only ladder step that helps *without a fitted parameter*
   (the semantic chunker's gain depends on a threshold that does not
   transfer). The best arms are `words25_max` on arctic2 (+0.0292
   [+0.0069, +0.0514]) and `words15_peak_3_0.05` on embeddinggemma (+0.0421
   [+0.0164, +0.0682]); `sent2` and `words40` are close behind and cost
   fewer chunks.
2. **The aggregation decides how much of the window gain survives.** With the
   window chunkers, the peak-like rules (`max`, `peak:3_0.05`) *and* the mean
   of the two best (`top_k_mean:2`) all carry a within-document gain (+0.018
   to +0.042 across arctic2's and embeddinggemma's window variants, with
   intervals excluding zero; the one exception is arctic2's
   `words15_top_k_mean_2`, +0.0098 [−0.0143, +0.0341]), while
   `decayed_top_k` (exp07's decay *and* its spread penalty) is the weakest of
   them: it is the only window rule that does not clear zero on arctic2's
   15-word variant (+0.0042 [−0.0199, +0.0282]) and it is negative on
   voyage-4-nano's window variants (−0.0350 to −0.0068). With the
   **semantic** chunker the order flips: the peak-like rules are the only
   ones that gain (+0.0195 / +0.0264) and the diluted ones lose (−0.0029
   to −0.0676) — a semantic chunk is a *topic-coherent* unit, so averaging
   two of them dilutes a match that sits in one of them.
3. **The local+global blend (exp06) is the cheapest way to take the window's
   gain.** It keeps a good part of the within-document gain while lifting the
   control's level far less than the plain `max` arm (arctic2's `words25`:
   control shift +0.0197 against `max`'s +0.0393, within-document +0.0148
   against +0.0292; embeddinggemma's `words25`: +0.0019 against +0.0203,
   +0.0263 against +0.0417).
4. **The more chunks, the bigger the inflation, for the same gain.** N=15
   (9.4 chunks per paragraph) and N=25 (5.3) are statistically
   indistinguishable from each other (their intervals overlap heavily) while
   N=15 inflates the control 1.4–1.6× as much as N=25 and 2.5–2.9× as much
   as N=40 (3.1 chunks). N=40 is the cheapest and nearly as good.
5. **The symmetric variants lose.** Chunking the question side is
   significantly harmful on all three models (point 5 above).
6. **The effect does not transfer across models.** Two of three models gain
   ~+0.02–0.04 within-document AUC; voyage-4-nano gains nothing. A
   segmentation recipe must be re-measured per embedder, and the winner
   (N=25 vs N=15 vs N=40) is model-dependent too.

**What this run does not say.**

- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice only.
- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track), so the track's safe-negative rule still
  ships as the **margin half only** — no spot-check sheet.
- **Nothing about the embedding space.** The "same-report direction"
  mechanism stays a hypothesis; this run measures scores and rank positions,
  not directions.
- **Nothing about the plan's §0.1 ad-hoc chunking table** (0.4680 / 0.4708 /
  0.4507 / 0.4354). It is not re-derived here and it is not comparable: it
  measured word windows and a chunked question side with an *unpaired* AUC
  and no control, and it pooled the documents' levels — the very effect this
  run measures. The ad-hoc table stays indicative.
- **Nothing about the length↔shift correlation** the plan's caveat asks for
  (the paragraph lengths live in the raw pair file, which this track's log
  rule does not read). The per-document shifts are reported instead.
- **Nothing about the question-type features** (Stage 1.1b, moved to
  Stage 3) or about a learned combiner (Stage 3): every arm here is a
  fixed rule.

**The next step.** The ladder's verdict is "chunk the paragraph, keep the
peak, don't chunk the question" — a *method* choice, not a representation
finding: the within-document gain is +0.02–0.04 AUC on a benchmark whose
models sit within ~2.4 unpaired SE of chance, and the representation
mechanism (plan §5's open question) is still unmeasured. Stage 3's feature
pack can now use the ladder's structure as a *feature* (the best chunk match,
the peak margin, the chunk count) rather than as a scoring rule, which is
where the remaining headroom is: the windows change the score, not the
embedding.

## Correction — the Stage-1 atom-arm count (CS-4, 2026-10-26)

An earlier entry in this log and three notes in `TEXT_PROCESSING_PLAN.md`
say that in the Stage-1.1a run "the floor gap narrows in **8 of the 9** atom
arms" (and "the **nine** atom arms"). The artifacts say **11 of the 12**: the
Stage-1 runs have **four** aggregation rules (`max`, `top_k_mean:2`,
`top_k_mean:3`, `mean`) on **three** models, so there are twelve atom arms,
and the floor gap narrows in eleven of them (the exception is
embeddinggemma's `atoms_max`, +0.0030, whose ΔAUC interval includes zero).

*Verified from the artifacts* (`results/omct-analyze-*-atoms/analysis.json`,
the `comparisons.<arm>.floor_gap.delta` field, re-read in the CS-4 runs' own
guard check). The **conclusion does not change**: the Stage-1 falsification
test did not fire, the four positive ΔAUCs are the same four, and none of
the twelve has a 95% interval that excludes zero on the positive side. Only
the count was wrong; the plan's notes are corrected in place (they are a
working document, not an append-only log) and this entry records the
correction.

## Stage 3.1 — the feature pack + the logistic combiner (CS-5, 2026-10-26)

**The question.** The plan's Stage 3.1: can a **table of numbers per pair**
(a feature row) plus a **learned scorer** (a logistic regression) rank the
OMCT pairs better than the hand-written scoring rules — and does the win
survive the plan's own floors (a document-level holdout, a label-permutation
chance floor)?

**The runs.** Three runs, one per Stage-0 model
(`experiments_text_processing/stage3_features_sweep.py`, the sweep
`stage3_features`):

- `omct-analyze-embeddinggemma-stage3`
- `omct-analyze-voyage-4-nano-stage3`
- `omct-analyze-arctic2-stage3`

Each run holds **all three rotations** (the out-of-fold protocol needs them
in one run: every judged pair and every control cell must be scored by the
model that never saw its document). The guard holds: all 30 runs (the 27
earlier ones and these three) reproduce the Stage-0 baseline arm field by
field (`tools/verify_analysis_baseline.py "omct-analyze-*"`).

**The protocol (as the reviewer settled it).**

- **The training data.** The fit side's **judged pairs** (the 524 verified
  positives and the 996 reviewer-verified hard negatives of the two fit
  documents) plus the fit side's **unjudged control cells** (subset B) as
  extra presumed negatives, down-weighted (**weight 0.3** — the control's
  true positive rate is unmeasured, so it must not dominate the fit). The
  control is never presented as ground truth.
- **The feature pack (v0).** 15 features: the full-text cosine; the best
  chunk cosine, the mean of the 2 and of the K=3 best chunk cosines, their
  spread and the peak margin; the soft lexical overlap in both directions
  and the char-3-gram cosine; the answer options' soft coverage; and the
  question-type one-hots (options present, a "no data" option, a yes/no
  question, a frequency question, the option count). **No length feature** —
  length is a document proxy on this data, so the lengths are *recorded* in
  the artifact and never fed to the model.
- **The combiner.** scikit-learn's logistic regression (C = 1, 1,000
  iterations, the features standardized — the scaler is fitted inside
  `fit()`, so the holdout never contributes to it).
- **The floors.** A **chance floor** (the fit side's whole label vector is
  permuted: the class counts and the weights stay, only the assignment
  changes — the model then has no label information at all) and a
  **control-role floor** (only the judged pairs' labels are permuted; the
  control cells keep their presumed-negative role). Five permutations per
  rotation, seed 42.

**1. The headline: the within-document AUC, on every held-out document.**
The within-document AUC is the headline (the plan's Stage-2 verdict: the
pooled AUC on this benchmark is structurally sensitive to the documents'
score levels). Paired against the full-text cosine (the Stage-0 baseline
arm) on the same held-out document, with the paired bootstrap's 95%
interval (2,000 resamples, seed 42):

| model | arm | baseline | Δ within-document AUC | 95% CI | pooled AUC (baseline) | Δ pooled | 95% CI | macro AUC (baseline) |
|---|---|---|---|---|---|---|---|---|
| embeddinggemma | **0.6655** | 0.5632 | **+0.1023** | [+0.0740, +0.1313] | 0.5831 (0.5371) | +0.0460 | [+0.0213, +0.0708] | 0.7156 (0.6376) |
| voyage-4-nano | **0.6723** | 0.5238 | **+0.1485** | [+0.1176, +0.1820] | 0.5805 (0.5004) | +0.0801 | [+0.0515, +0.1098] | 0.7052 (0.5588) |
| arctic2 | **0.6472** | 0.5020 | **+0.1452** | [+0.1143, +0.1797] | 0.5419 (0.4680) | +0.0740 | [+0.0443, +0.1033] | 0.6705 (0.5463) |

All three intervals exclude zero on the positive side. The per-document
numbers (the arm vs the baseline on the same held-out document) are in
every rotation of every model — 9 of 9 the arm is ahead:

| model | NU held out (arm / baseline) | AI held out | CIDH held out |
|---|---|---|---|
| embeddinggemma | 0.6189 / 0.5231 | 0.7867 / 0.7633 | 0.7411 / 0.6263 |
| voyage-4-nano | 0.6386 / 0.5027 | 0.7500 / 0.6165 | 0.7271 / 0.5573 |
| arctic2 | 0.6140 / 0.4773 | 0.6956 / 0.6209 | 0.7018 / 0.5408 |

**2. The chance floor lands at chance — a second floor does not, and it is
above the plain cosine.** Both floors are measured on the same statistic as
the headline (the within-document AUC: the pair-count-weighted mean of the
documents' own AUCs, so the comparison is like for like).

| model | chance floor | control-role floor | the full-text cosine | the arm |
|---|---|---|---|---|
| embeddinggemma | 0.5054 | 0.5788 | 0.5632 | **0.6655** |
| voyage-4-nano | 0.4936 | 0.5765 | 0.5238 | **0.6723** |
| arctic2 | 0.4968 | 0.5569 | 0.5020 | **0.6472** |

The **chance floor** (the fit side's whole label vector permuted: the class
counts and the weights stay, only the assignment changes — the model then has
no label information at all) sits at chance, as the plan requires. The
**control-role floor** (the same fit, but only the judged pairs' labels are
permuted and the control cells keep their presumed-negative role) sits at
**0.557–0.579** — *above* the plain cosine on all three models.

That is the finding of this run. A model that knows **nothing** about the
labels, but is told "the ordinary unjudged same-document cells are
negative", ranks the reviewed set better than the full-text cosine does. The
mechanism is the reviewed set's own construction: the reviewed pairs are the
ones a model put on top (so they are less "control-like" than the unjudged
cells), and the true matches sit at the top of that ranking while the hard
negatives sit lower.

So the arm's win decomposes into two parts (embeddinggemma; the other two
models are the same picture):

- **0.5632 → 0.5788 (+0.0156)**: what the control's presumed-negative role
  adds over the plain cosine — the training design's own, label-free signal.
- **0.5788 → 0.6655 (+0.0867)**: what the **label relation** adds on top —
  85% of the arm's total gain over the baseline (+0.1023).

The arm is ahead of the control-role floor in 9 of 9 rotations as well
(e.g. embeddinggemma: 0.6189 vs 0.5742 on NU, 0.7867 vs 0.5962 on AI,
0.7411 vs 0.5861 on CIDH).

**3. The control side: no inflation.** The arm's scores are probabilities and
the baseline's are cosines, so their *levels* are not comparable; the
**rank** numbers are. The control's rank percentile (0 = the top of the
paragraph's own candidate list) barely moves, and its top-fifth share moves
by less than half a point:

| model | control rank p50 (arm / baseline) | Δ | top-fifth share (arm / baseline) | Δ |
|---|---|---|---|---|
| embeddinggemma | 0.5023 / 0.5042 | −0.0019 | 0.1953 / 0.1930 | +0.0023 |
| voyage-4-nano | 0.4938 / 0.5023 | −0.0085 | 0.1963 / 0.2003 | −0.0040 |
| arctic2 | 0.4977 / 0.5021 | −0.0044 | 0.1965 / 0.1923 | +0.0043 |

The falsification statistic fires in the arm's favour: the standardized floor
gap (the verified positives' mean minus the control's mean, divided by the
control's spread) is **1.798 vs 1.730** (embeddinggemma), **1.913 vs 1.439**
(voyage-4-nano) and **1.668 vs 1.452** (arctic2). The de-shifted AUC (each
document's control level removed) keeps the gain: 0.6094 vs 0.5731, 0.5993
vs 0.5252, 0.5699 vs 0.5056.

**4. The caveat: the arm is better at ranking, worse at retrieving.** The
hit@k block is the one place the learned arm *loses*. The positives' hit@1
(the true match is the paragraph's own best candidate) falls from 0.1355 to
**0.0897** (embeddinggemma), from 0.1050 to **0.0821** (voyage-4-nano) and
from 0.1050 to **0.0859** (arctic2); the hit@10 follows on two of the three
models. The hard negatives' hit@1 falls too (0.0201 → 0.0090, 0.0211 →
0.0110, 0.0171 → 0.0100), so the arm is not simply "safer". Reading: a
bag-of-words overlap ranks a *pair* better than an embedding cosine does,
but it is a worse *retriever* — the cosine's best guess for a paragraph is
still a better candidate than the learned scorer's.

The fitted weights say what the arm leans on (the features are
standardized, so the coefficients are comparable). The largest ones are the
**pair-level** features — the full-text cosine, the lexical coverage in both
directions and the chunk cosines — and the question-type one-hots are
smaller (0.32–0.41 where they appear at all): the gain is **not** a
question-level prior. The weights are also unstable across the three fit
sides (the same feature flips sign between rotations), which is what a
logistic regression fitted on 675–1,421 judged pairs plus a few thousand
down-weighted presumed negatives does.

**5. The length capability landed (the plan's open question).** The command
now records the paragraph lengths (per document and per paragraph) and the
**length ↔ score correlation** of the control cells. For the baseline arm
(embeddinggemma) the Pearson correlation between a control cell's paragraph
length and its score is **+0.309 in NU, +0.259 in AI and +0.061 in CIDH**
(the Spearman values: +0.368 / +0.268 / +0.058) — the ordinary cells' score
already travels with the paragraph's length in two of the three documents,
and hardly at all in the third. So the plan's caveat ("the inflation is not
uniform") is now measurable: a chunked method's max-over-k inflation would
ride on that gradient, and it is document-dependent. The *shift* (the change
between a method and the baseline, cell by cell) still needs a second arm
and is a later change set.

**What this run does not say.**

- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice only.
- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track), so the track's safe-negative rule
  still ships as the **margin half only** — no spot-check sheet.
- **Nothing about the pooled AUC as a headline.** The out-of-fold matrix
  mixes **three models** (one per rotation), so its pooled number compares
  scores from different models; the within-document statistic and the
  per-document macro are the meaningful ones. The pooled numbers are
  reported beside them, never alone.
- **Nothing about which feature carries the win.** The pack's 15 features
  are fitted together; a group ablation (the lexical block, the chunk
  block, the question-type block) is not in this change set.

**The next step (CS-6).** The gradient-boosted combiner (the plan's 3.2),
the chunk-profile features (3.3), the random-vector floor, the feature-group
ablations and the calibration curves. The open question the Stage-3.1 run
adds: **how much of the +0.10 to +0.15 within-document gain survives if the
question-type features are removed** — the hit@k loss suggests the learned
ranking is not a retrieval ranking, and the ablation says whether that is
the type features or the lexical ones. A second question the run adds, for
the durable deliverable: the control-role floor shows that "the ordinary
unjudged cells are negative" is **by itself a better pair ranker than the
embedding cosine** on this benchmark — so a safe-negative rule built on such
a model is partly self-fulfilling, and the rule's margin has to be measured
against that floor, not against the cosine.

## Correction — the plan's paragraph-length figures (CS-5, 2026-10-26)

`TEXT_PROCESSING_PLAN.md` section 5 states the length ↔ shift question with
the figures "AI median 364 chars vs NU 863". The Stage-3.1 artifact's
`analysis.paragraph_lengths` (the same 225 paragraphs, measured by the
command) says **AI median 331 chars, NU median 375, CIDH median 514** (the
means: AI 342, NU 478, CIDH 524). The *claim* the figures supported — the
paragraph length tracks the document, so a length feature can memorize the
document instead of the topic — still holds (NU's paragraphs are longer than
AI's), but the stated margin was wrong. The plan's note is corrected in
place; this entry records the correction.

## Stage 3.1b — the feature-group ablation (CS-6a, 2026-10-26)

**The question.** The Stage-3.1 run left two questions open: (a) **which
block of the feature pack carries the gain** — the +0.10 to +0.15
within-document AUC the learned arm won over the full-text cosine — and (b)
**why the arm is a worse retriever** (its positives' hit@1 is 0.090 against
the cosine's 0.136 on embeddinggemma). The plan's open question named the
two candidates for (b): the question-type features (a question-level prior
that reorders a paragraph's candidates without looking at the paragraph) or
the lexical ones. The Stage-3.1 log could only *read* the fitted
coefficients; this change set *measures* it.

**The unit of the ablation is the feature group.** The v0 pack's 15 features
fall into four blocks of the same kind of signal:

| group | features | what it holds |
|---|---|---|
| `embedding` | 6 | the full-text cosine + the five chunk features |
| `lexical` | 3 | the soft lexical overlap (both ways) + the char-3-gram cosine |
| `question_part` | 1 | the answer options' soft coverage |
| `question_type` | 5 | the question-type one-hots |

The group is a **config knob** (`analysis.stage3.features.groups`, validated
at config time: an unknown, an empty or a repeated group fails before any
run), and the pack keeps only the selected groups' columns — the pack itself
computes every feature, the combiner sees the selection. Nothing else
changes: the same document-level holdout (three rotations, one out-of-fold
score matrix), the same training data (the fit side's judged pairs plus its
unjudged control cells as down-weighted presumed negatives), the same two
floors (the label-permutation chance floor and the control-role floor) and
the same paired statistics against the full-text cosine.

**The grid.** Eight group sets per model: `all` (the whole pack — the
reference), the leave-one-out set `no-qtype` / `no-qpart` / `no-lex` /
`no-embed`, and the single-block arms `embed-only` / `lex-only` /
`qtype-only`. Three Stage-0 models, so **24 runs**
(`omct-analyze-<model>-stage3-<groupset>`, the sweep
`experiments_text_processing/stage3_ablation_sweep.py`). Each run holds all
three rotations, because the out-of-fold protocol needs them together.

**The checks.** The `all` set reproduces the Stage-3.1 artifacts **field by
field** (every number identical; only the new metadata fields differ), and
the Stage-3.1 sweep was re-run for the table's new shape with the same
result. The guard holds: all **54 runs** reproduce the Stage-0 baseline arm
field by field (`tools/verify_analysis_baseline.py "omct-analyze-*"`).

**1. The headline: the within-document AUC per group set** (the Stage-3.1
arm is the `all` row; the paired CI against the full-text cosine is in the
second table).

| group set | features | embeddinggemma | voyage-4-nano | arctic2 |
|---|---|---|---|---|
| `all` | 15 | 0.6655 | 0.6723 | 0.6472 |
| `no-qtype` | 10 | 0.6632 | 0.6670 | 0.6462 |
| `no-qpart` | 14 | 0.6569 | 0.6653 | 0.6356 |
| `no-lex` | 12 | 0.6188 | 0.5650 | 0.5345 |
| `no-embed` | 9 | 0.6589 | 0.6589 | 0.6589 |
| `embed-only` | 6 | 0.6006 | 0.5436 | 0.5126 |
| `lex-only` | 3 | 0.6461 | 0.6461 | 0.6461 |
| `qtype-only` | 5 | 0.5713 | 0.5713 | 0.5713 |

The paired difference against the full-text cosine (the same held-out
documents, the paired bootstrap's 95% interval, 2,000 resamples, seed 42):

| group set | embeddinggemma | voyage-4-nano | arctic2 |
|---|---|---|---|
| `all` | +0.1023 [+0.0740, +0.1313] | +0.1485 [+0.1176, +0.1820] | +0.1452 [+0.1143, +0.1797] |
| `no-qtype` | +0.1000 [+0.0736, +0.1270] | +0.1432 [+0.1127, +0.1744] | +0.1443 [+0.1135, +0.1771] |
| `no-qpart` | +0.0937 [+0.0661, +0.1232] | +0.1415 [+0.1097, +0.1753] | +0.1336 [+0.1031, +0.1678] |
| `no-lex` | +0.0555 [+0.0286, +0.0824] | +0.0412 [+0.0157, +0.0650] | +0.0325 [+0.0061, +0.0578] |
| `no-embed` | +0.0956 [+0.0555, +0.1365] | +0.1351 [+0.0883, +0.1831] | +0.1569 [+0.1127, +0.2022] |
| `embed-only` | +0.0374 [+0.0133, +0.0610] | +0.0198 [+0.0034, +0.0349] | +0.0107 [−0.0096, +0.0299] |
| `lex-only` | +0.0829 [+0.0452, +0.1230] | +0.1223 [+0.0752, +0.1694] | +0.1441 [+0.1020, +0.1885] |
| `qtype-only` | +0.0081 [−0.0387, +0.0548] | +0.0475 [−0.0042, +0.0985] | +0.0694 [+0.0215, +0.1152] |

**2. The verdict on (a): the gain is the lexical block's, and the embedding
geometry adds nothing to it.**

- **Removing the question-type block costs almost nothing**: 0.0023
  (embeddinggemma), 0.0053 (voyage-4-nano), 0.0010 (arctic2) of the
  within-document AUC. The Stage-3.1 reading of the fitted weights ("the
  gain is not a question-level prior") was right, and now it is measured.
- **Removing the lexical block costs the most**: 0.0468, 0.1073, 0.1127.
  The `lex-only` arm — **three features** — reaches 0.6461 on all three
  models: 97% / 96% / 100% of the full pack's within-document AUC, and
  81% / 82% / 99% of its gain over the cosine.
- **Removing the embedding block costs nothing**: the `no-embed` arm reaches
  0.6589 on all three models, *above* the full pack on arctic2 (0.6472). So
  the six embedding features (the full-text cosine and the five chunk
  cosines) add no within-document ranking on top of the lexical block; on
  the two models where the full pack is ahead, the difference is 0.007 and
  0.013.
- The `embed-only` arm adds only +0.0374 / +0.0198 / +0.0107 over the
  cosine, and on arctic2 its interval includes zero.

The three arms that use **no embedding at all** (`no-embed`, `lex-only`,
`qtype-only`) give *identical* numbers on all three models. That is the
built-in check of the whole harness: the runs share the data, the control
cells, the rotations and the seeds, and only the embedding-dependent arms
move with the model.

**3. The same picture against the control-role floor** (the yardstick the
track's deliverable is measured against: the floor of an arm that knows
nothing about the labels but is told "the ordinary unjudged cells are
negative"). The table gives each arm's own control-role floor and the arm's
excess over it — the arm's *label-driven* part:

| group set | control-role floor (eg / voy / arc) | the arm's excess (eg / voy / arc) |
|---|---|---|
| `all` | 0.5788 / 0.5765 / 0.5569 | +0.0868 / +0.0958 / +0.0903 |
| `no-qtype` | 0.5686 / 0.5643 / 0.5490 | +0.0946 / +0.1027 / +0.0972 |
| `no-qpart` | 0.5993 / 0.6014 / 0.5770 | +0.0577 / +0.0639 / +0.0585 |
| `no-lex` | 0.5964 / 0.5362 / 0.5345 | +0.0223 / +0.0288 / −0.0000 |
| `no-embed` | 0.5392 / 0.5392 / 0.5392 | +0.1197 / +0.1197 / +0.1197 |
| `embed-only` | 0.5940 / 0.5173 / 0.5259 | +0.0066 / +0.0263 / −0.0133 |
| `lex-only` | 0.5632 / 0.5632 / 0.5632 | +0.0829 / +0.0829 / +0.0829 |
| `qtype-only` | 0.5358 / 0.5358 / 0.5358 | +0.0356 / +0.0356 / +0.0356 |

The `embed-only` arm is **at** its own control-role floor (0.6006 against
0.5940 on embeddinggemma) and **below** it on arctic2 (0.5126 against
0.5259). So the embedding geometry's own arm-level win over the cosine
(+0.0374 / +0.0198 / +0.0107) is *the training design's own label-free
signal*, not label information. The lexical block is the opposite: +0.0829
above its floor on every model.

**4. The verdict on (b): the hit@1 loss is the lexical block's, not the
question-type block's.** The positives' hit@1 (the true match is the
paragraph's own best candidate):

| group set | embeddinggemma | voyage-4-nano | arctic2 |
|---|---|---|---|
| `all` | 0.0897 (cosine 0.1355) | 0.0821 (cosine 0.1050) | 0.0859 (cosine 0.1050) |
| `no-qtype` | 0.0916 (cosine 0.1355) | 0.0821 (cosine 0.1050) | 0.0802 (cosine 0.1050) |
| `no-qpart` | 0.0897 (cosine 0.1355) | 0.0954 (cosine 0.1050) | 0.0706 (cosine 0.1050) |
| `no-lex` | 0.1279 (cosine 0.1355) | 0.1050 (cosine 0.1050) | 0.0935 (cosine 0.1050) |
| `no-embed` | 0.0344 (cosine 0.1355) | 0.0344 (cosine 0.1050) | 0.0344 (cosine 0.1050) |
| `embed-only` | 0.1317 (cosine 0.1355) | 0.1145 (cosine 0.1050) | 0.0916 (cosine 0.1050) |
| `lex-only` | 0.0324 (cosine 0.1355) | 0.0324 (cosine 0.1050) | 0.0324 (cosine 0.1050) |
| `qtype-only` | 0.0076 (cosine 0.1355) | 0.0076 (cosine 0.1050) | 0.0076 (cosine 0.1050) |

- **Dropping the lexical block restores the cosine's hit@1**: 0.1279 against
  the cosine's 0.1355 on embeddinggemma, and on voyage-4-nano it is exactly
  the cosine's 0.1050. Dropping the **question-type** block moves it by
  ≤0.006 (0.0897 → 0.0916, 0.0821 → 0.0821, 0.0859 → 0.0802).
- So the Stage-3.1 caveat ("a learned pair ranker is not a better
  retriever") is a statement about the **lexical** block: a bag-of-words
  overlap ranks a *pair* better than an embedding cosine does, and is a
  worse first guess for a paragraph.
- The `qtype-only` arm confirms the plan's *mechanism* and refutes its
  *size*: a question-level prior reorders a paragraph's candidates without
  looking at the paragraph, so its hit@1 collapses to 0.0076 — the worst of
  all the arms — but the question-type block is worth 0.001–0.005 of the
  within-document AUC, so it is not what costs the hit@1.

**5. The question-type arm, read as a question-level prior.** The
`qtype-only` arm reaches 0.5713 (the CI against the cosine excludes zero on
arctic2 only: +0.0694 [+0.0215, +0.1152]; on embeddinggemma and
voyage-4-nano it includes zero). Its **macro** AUC is 0.5324 — *below* the
cosine's on all three models (0.6376 / 0.5588 / 0.5463) — and it collapses
on the AI stratum (0.4358 against the cosine's 0.7633 / 0.6165 / 0.6209).
The leakage route is measured too: of the holdout's questions, 176 of 243
(NU), 45 of 51 (AI) and 182 of 217 (CIDH) are also judged by the fit side.
The features are pair-local (no corpus statistic enters them) and the
combiner's weight is fitted on the fit side only, but a question-type
weight reaches the holdout's copy of the same question — so this arm's
number is an upper bound on what a question-level prior can buy, not a
transferable gain.

**6. The per-rotation numbers** (the arm / the cosine's, on the same
held-out document). The AI stratum is the thin one (99 pairs, 51 questions),
and it is where every arm loses a rotation:

| group set | model | NU | AI | CIDH |
|---|---|---|---|---|
| `all` | embeddinggemma | 0.6189 / 0.5231 | 0.7867 / 0.7633 | 0.7411 / 0.6263 |
| `no-qtype` | embeddinggemma | 0.6115 / 0.5231 | 0.7911 / 0.7633 | 0.7471 / 0.6263 |
| `no-qpart` | embeddinggemma | 0.6149 / 0.5231 | 0.7677 / 0.7633 | 0.7251 / 0.6263 |
| `no-lex` | embeddinggemma | 0.5754 / 0.5231 | 0.7823 / 0.7633 | 0.6880 / 0.6263 |
| `no-embed` | embeddinggemma | 0.6366 / 0.5231 | 0.5734 / 0.7633 | 0.6978 / 0.6263 |
| `embed-only` | embeddinggemma | 0.5582 / 0.5231 | 0.7987 / 0.7633 | 0.6676 / 0.6263 |
| `lex-only` | embeddinggemma | 0.6291 / 0.5231 | 0.5646 / 0.7633 | 0.6761 / 0.6263 |
| `qtype-only` | embeddinggemma | 0.5473 / 0.5231 | 0.4358 / 0.7633 | 0.6141 / 0.6263 |

The arm is ahead of the cosine in 9 of 9 rotations for `all`, `no-qtype` and
`no-qpart`, in 8 of 9 for `no-lex` and `embed-only`, and in 6 of 9 for
`no-embed` and `lex-only` (the AI stratum) and 5 of 9 for `qtype-only`. The
other two models' per-rotation numbers are in the artifacts.

**7. The control side: no inflation.** The control's rank percentile (0 =
the top of the paragraph's own candidate list) barely moves for every arm:
the `all` arm by −0.0019 / −0.0085 / −0.0044, the single-block arms by
≤0.008, except the `lex-only` arm, which pushes the ordinary cells *down*
the candidate lists (+0.0119 / +0.0138 / +0.0141). So the lexical arm's
gain on the reviewed set is not an inflation of the control.

**What this run does not say.**

- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice only.
- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track), so the track's safe-negative rule
  still ships as the **margin half only**.
- **Nothing about a corpus with a different vocabulary.** The lexical block
  is a bag-of-words overlap over 294 questions; whether it transfers to a
  new corpus is open (the plan's §5).
- **Nothing new about the pooled AUC.** The `no-lex` arm's *pooled* AUC
  falls below the cosine's on all three models (0.5320 / 0.4934 / 0.4385
  against 0.5371 / 0.5004 / 0.4680) while its within-document gain
  survives — one more reason the within-document statistic is the headline
  on this benchmark.

**The next step (CS-6b).** The gradient-boosted combiner (the plan's 3.2)
behind the same `Combiner` port, with the group selection as a knob: the
ablation says the learner now has to earn something over a bag-of-words
overlap, so the comparison to watch is HistGB on the *lexical* block against
HistGB on the whole pack. Then 3.3 (the chunk-profile features), the
random-vector floor (the second sanity floor of the plan's Stage 3) and the
Stage-3 reporting (the calibration curves, the thresholds per document, and
the safe-negative margin measured against the control-role floor).

**A note on the comparison table.** The Stage-3 table shape grew with this
change set: a `groups` column, the two `hit@1` columns, and the two floor
columns renamed `perm_auc_within` / `ctrl_role_auc_within` because they now
carry the **within-document** weighted floors — the statistic the headline
is measured against, and the one the Stage-3.1 log entry already quoted (the
old columns carried the unweighted per-rotation means, which are not
like-for-like with the headline). The Stage-3.1 sweep was re-run for the new
shape; its artifacts reproduce the stored ones field by field, so the
Stage-3.1 numbers in the table did not move.

## Stage 3.2 — the gradient-boosted combiner (CS-6b, 2026-10-26)

**The question.** The plan's Stage 3.2: can a **gradient-boosted** learner
(scikit-learn's `HistGradientBoostingClassifier`) rank the OMCT pairs better
than the Stage-3.1 logistic regression? The Stage-3.1b ablation had already
narrowed the question: the learned arm's gain is the **lexical** block's (a
bag-of-words overlap), not the embedding geometry's and not the question-type
prior's. So the bar is explicit — a new learner has to beat a *three-feature*
logistic regression (0.6461 within-document AUC on all three models) and must
not cost more hit@1. A tree ensemble can only earn something if the signal is
not linear in the features, or if the blocks interact.

**The runs.** Nine runs (3 Stage-0 models x 3 feature-group sets,
`experiments_text_processing/stage3_histgb_sweep.py`, the sweep
`stage3_histgb`):

- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-histgb-all`
- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-histgb-lex-only`
- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-histgb-embed-only`

Each run holds all three rotations (the out-of-fold protocol needs them
together). The guard holds: all **63** runs (the 54 earlier ones and these
nine) reproduce the Stage-0 baseline arm field by field
(`tools/verify_analysis_baseline.py "omct-analyze-*"`).

**The protocol (unchanged from Stage 3.1).** The same document-level holdout
(fit on two documents, predict the third; one out-of-fold score matrix), the
same training data (the fit side's judged pairs plus its unjudged control
cells as down-weighted presumed negatives, weight 0.3), the same two floors
(the label-permutation chance floor and the control-role floor, 5 permutations
per rotation, seed 42) and the same paired statistics against the full-text
cosine (2,000 bootstrap resamples, seed 42).

**The learner.** scikit-learn's `HistGradientBoostingClassifier` with **its
defaults, explicitly pinned** (`max_iter` 100, `learning_rate` 0.1,
`max_leaf_nodes` 31, `min_samples_leaf` 20, `l2_regularization` 0,
`max_depth` `None`, `max_bins` 255, `random_state` 42) and **early stopping
off** — so the fit side's rows are the fit side's rows (no internal validation
split is carved out of them) and the fit is deterministic. No hyperparameter
grid: the track has already measured that a fit-side selection did not
transfer (the Stage-2 semantic threshold: the fit side's choice did not
predict the holdout's). A tree ensemble needs no standardization (the booster
bins the features itself), so — unlike the logistic arm — there is no scaler
that could leak the holdout into the fit.

**The logistic partners are the Stage-3.1b runs**
(`omct-analyze-<model>-stage3-{all,lex-only,embed-only}`): the same protocol,
the same control cells, the same rotations and the same seeds, so the two
learners' arms are read on the same footing.

**1. The headline: the within-document AUC, the booster beside the logistic
regression** (OMCT auxiliary data, the reviewed set's 1,520 judged pairs —
never a UPR metric). The CI is the arm's own paired bootstrap against the
full-text cosine on the same held-out documents:

| group set | model | HistGB | logistic | Δ (booster − logistic) | the booster's CI vs the cosine |
|---|---|---|---|---|---|
| `all` | embeddinggemma | 0.6323 | 0.6655 | **−0.0333** | +0.0690 [+0.0354, +0.1031] |
| `all` | voyage-4-nano | 0.6318 | 0.6723 | **−0.0405** | +0.1080 [+0.0744, +0.1433] |
| `all` | arctic2 | 0.5987 | 0.6472 | **−0.0485** | +0.0968 [+0.0601, +0.1353] |
| `lex-only` | all three | 0.6238 | 0.6461 | **−0.0223** | +0.0605 / +0.1000 / +0.1218 (eg / voy / arc) |
| `embed-only` | embeddinggemma | 0.5628 | 0.6006 | **−0.0378** | −0.0004 [−0.0326, +0.0347] |
| `embed-only` | voyage-4-nano | 0.5435 | 0.5436 | −0.0001 | +0.0197 [−0.0086, +0.0499] |
| `embed-only` | arctic2 | 0.5117 | 0.5126 | −0.0010 | +0.0097 [−0.0242, +0.0449] |

The booster is **below** the logistic regression on eight of the nine arms; on
the ninth (voyage-4-nano's `embed-only`) the two are tied to four decimals.
The `lex-only` row is identical on the three models — the same built-in check
the Stage-3.1b ablation used: the lexical features do not depend on the
embeddings, so the arm must not move with the model.

The booster still beats the full-text cosine on the two arms with the lexical
block in them (all three models' intervals exclude zero on the positive side
for `all` and `lex-only`), so it is a working learned arm — just not a better
one.

**2. The verdict: the learner is not the lever.** The paired evidence is the
per-rotation difference (both arms on the *same* held-out document, the same
pairs):

| group set | model | NU | AI | CIDH | ahead in |
|---|---|---|---|---|---|
| `all` | embeddinggemma | −0.0513 | −0.0215 | −0.0035 | 0/3 |
| `all` | voyage-4-nano | −0.0186 | −0.0152 | −0.0776 | 0/3 |
| `all` | arctic2 | −0.0501 | −0.0734 | −0.0454 | 0/3 |
| `lex-only` | all three | +0.0017 | −0.0057 | −0.0628 | 1/3 |
| `embed-only` | embeddinggemma | −0.0242 | −0.1025 | −0.0591 | 0/3 |
| `embed-only` | voyage-4-nano | +0.0275 | −0.0892 | −0.0444 | 1/3 |
| `embed-only` | arctic2 | +0.0202 | −0.0304 | −0.0358 | 1/3 |

The booster is ahead in **5 of the 27** per-rotation paired comparisons, and
in none of the nine `all`-pack comparisons. The reviewer's bar — beat a
three-feature logistic at 0.6461 — is not met: the booster's lexical arm
reaches 0.6238.

**3. The floors: where the difference actually sits.** Both floors again, on
the within-document statistic, plus each arm's excess over **its own**
control-role floor (the label-driven part; the yardstick the track's
deliverable is measured against):

| group set | model | the booster's control-role floor | the logistic's | the booster's excess | the logistic's excess |
|---|---|---|---|---|---|
| `all` | embeddinggemma | 0.5355 | 0.5788 | +0.0968 | +0.0868 |
| `all` | voyage-4-nano | 0.5276 | 0.5765 | +0.1042 | +0.0958 |
| `all` | arctic2 | 0.5113 | 0.5569 | +0.0874 | +0.0903 |
| `lex-only` | all three | 0.5450 | 0.5632 | +0.0788 | +0.0829 |
| `embed-only` | embeddinggemma | 0.5439 | 0.5940 | +0.0190 | +0.0066 |
| `embed-only` | voyage-4-nano | 0.5060 | 0.5173 | +0.0375 | +0.0263 |
| `embed-only` | arctic2 | 0.4883 | 0.5259 | +0.0233 | −0.0133 |

The chance floor lands at chance everywhere (0.480–0.496). The reading: the
booster's **label-driven** part is a wash with the logistic's (sometimes
higher, sometimes lower), and the logistic's advantage on the `all` and
`lex-only` arms comes mostly from a **higher control-role floor** — i.e. the
logistic regression is the better learner of the *training design's own*
label-free signal ("the ordinary unjudged cells are negative"), which is
exactly the signal the floor measures. (The mechanism behind that difference
is not measured here and is not claimed.)

**4. The hit@1: the booster retrieves worse.** The positives' hit@1 (the true
match is the paragraph's own best candidate):

| group set | model | HistGB | logistic | the cosine |
|---|---|---|---|---|
| `all` | embeddinggemma | 0.0611 | 0.0897 | 0.1355 |
| `all` | voyage-4-nano | 0.0840 | 0.0821 | 0.1050 |
| `all` | arctic2 | 0.0515 | 0.0859 | 0.1050 |
| `lex-only` | all three | 0.0153 | 0.0324 | 0.1355 / 0.1050 |
| `embed-only` | embeddinggemma | 0.0553 | 0.1317 | 0.1355 |
| `embed-only` | voyage-4-nano | 0.0515 | 0.1145 | 0.1050 |
| `embed-only` | arctic2 | 0.0515 | 0.0916 | 0.1050 |

The booster is the worse retriever on eight of the nine arms (the exception:
voyage-4-nano's `all` arm, +0.0019). On the lexical block it retrieves
*half* as well as the logistic (0.0153 against 0.0324) while ranking pairs
worse too — so the booster is worse on both halves of the job there. The
Stage-3.1 caveat ("a learned pair ranker is not a better retriever") is not
just confirmed but deepened: the *shape* of the learner matters for the
retrieval side too, and the booster is the worse shape.

**5. The importances: which features the trees used.** A tree ensemble has no
coefficients, so the artifact now carries the trees' **split usage** (the new
capability: per feature, the number of split nodes, the total gain, the
deepest level and the shares). It is a *usage* diagnostic — what the model
splits on — not an accuracy-based importance. On the `all` arms, the three
rotations together:

| model | embedding block | lexical block | question-part | question-type | the top single feature |
|---|---|---|---|---|---|
| embeddinggemma | 52.3% | 30.8% | 8.1% | 8.7% | `chunk_cosine_best` 16.9% |
| voyage-4-nano | 45.9% | 36.2% | 8.1% | 9.9% | `cosine_full` 15.9% |
| arctic2 | 42.6% | 38.0% | 9.0% | 10.3% | `lex_question_coverage` 16.1% |

Two readings. First, the booster's gain is **diffuse**: the top single
feature takes only 16–17% of it, against the logistic's concentrated
coefficients — a tree ensemble spreads its weight over the whole pack. Second,
the trees lean *harder* on the embedding block (43–52% of the gain) than the
lexical one, and the Stage-3.1b ablation showed that the embedding block adds
no label-driven signal. So the booster spends much of its capacity on the
block that carries the *training design's* signal — which is consistent with
its lower control-role floor and its lower label-driven excess.

**6. The control side: no inflation.** The control's rank percentile (0 = the
top of the paragraph's own candidate list) barely moves for any arm (−0.0126
to +0.0049), and the top-fifth share moves by ≤0.008. So the booster's
numbers on the reviewed set are not an inflation of the ordinary cells. The
de-shifted AUC (each document's control level removed) keeps the `all` arm's
gain (0.6072 / 0.6047 / 0.5741 against the baseline's 0.5731 / 0.5252 /
0.5056) but not the `embed-only` arm's (0.5233 / 0.5271 / 0.4910 — at or
below the baseline).

**What this run does not say.**

- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice only.
- **Nothing about a tuned booster.** The hyperparameters are scikit-learn's
  defaults, pinned; no grid was run. A *fit-side* grid would need a fit-side
  criterion that this track has already measured to be unidentifiable here
  (the Stage-2 semantic threshold), and a holdout-tuned grid would be a
  leakage.
- **Nothing about `lightgbm`.** The plan defers it until scikit-learn's
  HistGB is beaten; HistGB was not beaten, so the dependency is not proposed
  (and no new dependency is added at all).
- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track), so the track's safe-negative rule still
  ships as the **margin half only**.
- **No paired per-pair CI between the two learners.** The two learners' arms
  live in two runs, and the artifacts carry the per-rotation AUCs (not the
  per-pair scores), so the paired evidence is the per-rotation differences
  above plus each arm's own paired interval against the common reference (the
  cosine). A per-pair paired bootstrap between the arms would need the arms'
  per-pair out-of-fold scores in the artifact — a small additive block, and a
  later change set.
- **Nothing about the reviewed set's own construction.** The hard negatives'
  hit@1 also falls for the booster (0.0100 / 0.0090 / 0.0070 against the
  cosine's 0.0201 / 0.0211 / 0.0171), so the booster is not simply "safer" —
  but this benchmark's reviewed set is rank-selected (the Stage-0.2
  classification), and no arm's number here changes that.

**The next step (CS-6c).** The chunk-profile features (the plan's 3.3:
exp15/16's df and alignment-kind one-hots, exp17's ranked cross-chunk match
profiles) folded into the pack. The learner question is now settled for this
benchmark: the *features* are the lever, not the learner — so 3.3 goes after
new features with the logistic combiner, and the gradient-boosted combiner
stays in the registry as the second entry (available, not better). Then
CS-6d (the random-vector floor — the second sanity floor of the plan's
Stage 3) and CS-6e (the calibration curves, the thresholds per document, and
the safe-negative margin measured against the control-role floor).

## Stage 3.3 — the chunk-profile features (CS-6c, 2026-10-26)

**The question.** The plan's Stage 3.3: fold the **chunk-profile features**
into the pack — the features that say *where* the paragraph and the question
match (which piece of the paragraph aligns with which piece of the question,
how many times, how peaked), not how similar they are on average. The two
earlier change sets had narrowed the lever: the Stage-3.1b ablation showed the
learned arm's gain is the **lexical** block's, and the Stage-3.2 comparison
showed the **learner** is not the lever either. So the features were the only
thing left to try.

**The runs.** Nine runs (3 Stage-0 models × 3 group sets,
`experiments_text_processing/stage3_chunk_profile_sweep.py`, the sweep
`stage3_chunk_profile`):

- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-cp-base` — the
  v0 pack's four groups: the **control**;
- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-cp-full` — the
  v0 pack **plus** the new block;
- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-cp-profile` —
  the new block **alone**.

Each run holds all three rotations (the out-of-fold protocol needs them
together). The guard holds: all **72** runs reproduce the Stage-0 baseline arm
field by field (`tools/verify_analysis_baseline.py "omct-analyze-*"`).

**The protocol (unchanged from Stage 3.1).** The same document-level holdout
(fit on two documents, predict the third; one out-of-fold score matrix), the
same training data (the fit side's judged pairs plus its unjudged control
cells as down-weighted presumed negatives, weight 0.3), the same two floors
(the label-permutation chance floor and the control-role floor, 5 permutations
per rotation, seed 42) and the same paired statistics against the full-text
cosine (2,000 bootstrap resamples, seed 42). The combiner is the **logistic
regression** (CS-6b: the learner is not the lever).

**What was coded.** The new block is the pack's fifth group, `chunk_profile`
(10 features), and the four existing groups keep their exact meaning — so the
CS-6a arms still describe the same feature sets.

- The **ranked cross-chunk match profile** (exp17): both sides of a pair are
  cut into **sentence units** and into **word pieces** (5 words, tiling a
  sentence), every piece×piece cosine is computed, and the top three ranks of
  each granularity are kept (`cprof_sent_top1..3`, `cprof_ngram_top1..3`; a
  missing rank is padded with 0, the lineage's convention). The question's
  units are cut from the **transformed** question (the "Answer choices:" block
  stripped), like the pack's other embedding features, so the scale words stay
  out of the matches.
- The **alignment kinds** (exp16, with the taxonomy the reviewer chose on
  2026-10-26): which granularity produced the best alignment
  (`cprof_align_sent` / `cprof_align_ngram`) and whether the two aligned units
  share a content word (`cprof_align_lexical` /
  `cprof_align_semantic_only`).
- exp16's third feature, `log(1+df)` of the matched phrase **in the labeled
  corpus**, is **not** in the pack: it is a corpus-level statistic, the pack's
  contract is pair-local, a fit-side df would need a per-rotation feature
  matrix, and with ~150 fit-side paragraphs the counts would be 0/1/2. The
  reviewer's decision was to drop it.

The pack computes every feature and the combiner sees the selection, so the
diagnostics below cover all 25 columns in every run.

**1. The headline: the within-document AUC of the three arms** (OMCT auxiliary
data, the reviewed set's 1,520 judged pairs — never a UPR metric). The `base`
arm is the v0 pack; the reference is the full-text cosine.

| arm | embeddinggemma | voyage-4-nano | arctic2 |
|---|---|---|---|
| the full-text cosine (the reference) | 0.5632 | 0.5238 | 0.5020 |
| `base` (the v0 pack) | 0.6655 | 0.6723 | 0.6472 |
| `full` (v0 + the chunk profile) | **0.6722** | **0.6894** | 0.6423 |
| `profile` (the new block alone) | 0.5613 | 0.5324 | 0.4974 |

The `base` arm reproduces the CS-6a `all` arm **field for field** on every
number (`comparison`, `permutation_floor`, `out_of_fold`, `training`, `grid`,
`combiner`); the only differences are the two new blocks, the pack's name and
note, the `dropped_features` list (the new group's features are in it now)
and the new embedding sides' cache counters.

**2. The paired comparison between the two arms** — the new
`tools/compare_stage3_arms.py`, a paired per-pair bootstrap of the
within-document AUC difference (stratified by document and class, 2,000
resamples, seed 42) over the arms' per-pair out-of-fold scores, which the
Stage-3 block now carries:

| comparison | embeddinggemma | voyage-4-nano | arctic2 |
|---|---|---|---|
| `full` − `base` | +0.0066 [−0.0047, +0.0179] | **+0.0172 [+0.0048, +0.0298]** | −0.0049 [−0.0178, +0.0080] |
| `profile` − the cosine | −0.0019 [−0.0287, +0.0255] | +0.0086 [−0.0248, +0.0420] | −0.0046 [−0.0328, +0.0213] |

The per-rotation paired differences say the same thing (the same held-out
document, the same pairs, `full` − `base`): +0.0004 / −0.0139 / +0.0175
(embeddinggemma, NU/AI/CIDH), +0.0158 / +0.0304 / +0.0191 (voyage-4-nano),
+0.0047 / +0.0335 / −0.0217 (arctic2) — **7 of the 9** favour the full arm.

**3. The verdict: the chunk profile is not the lever either.** The block alone
is **indistinguishable from the full-text cosine** on all three models (every
interval above includes zero; 4 of the 9 per-rotation comparisons against the
cosine are positive). In the full pack it buys a **small gain that is
established on one of the three models only** (voyage-4-nano, +0.0172, the
interval excluding zero; embeddinggemma's +0.0066 and arctic2's −0.0049 both
include zero).

**4. The floors: the block brings a label-free signal with it.** The
within-document floors of the three arms (the chance floor permutes the whole
label vector, the control-role floor permutes only the judged pairs' labels
and keeps the control cells' presumed-negative role):

| arm | chance floor (eg / voy / arc) | control-role floor | the arm's excess over it |
|---|---|---|---|
| `base` | 0.5054 / 0.4936 / 0.4968 | 0.5788 / 0.5765 / 0.5569 | +0.0868 / +0.0958 / +0.0903 |
| `full` | **0.5248 / 0.5138 / 0.5313** | 0.5317 / 0.5372 / 0.5232 | +0.1405 / +0.1522 / +0.1191 |
| `profile` | 0.5324 / 0.5238 / 0.5343 | 0.5010 / 0.4532 / 0.4750 | +0.0603 / +0.0792 / +0.0224 |

The `base` arm's chance floor sits **at chance** (0.494–0.505), as the plan
requires. The two arms that carry the chunk profile sit **above** it
(0.514–0.534): with the profile's features, a model that has **no label
information at all** ranks the reviewed set better than chance. The block
therefore carries a signal that is not label-driven — the reviewed set is
rank-selected (the Stage-0.2 classification), and the profile's features pick
that structure up.

The "excess over the control-role floor" is **not** comparable across arms
with different feature sets: the floor is itself a property of the arm's
features (the full arm's control-role floor is *lower* than the base's, 0.53
against 0.58, so its excess looks bigger). The primary statement is the
paired within-document AUC difference above; the floors are the diagnostic
that says how much of an arm's number is label-driven.

**5. The per-feature diagnostics: what each new feature actually measures.**
The Stage-3 block now carries, for **every** feature the pack computes, its
own discrimination number (the within-document AUC of the feature used as a
score), its control-side counterpart (the ordinary cells' level and rank
position) and its correlation with the paragraph's length. The new block's own
rows (the `base` arm's artifact, so the numbers are the same everywhere):

| feature | within AUC (eg / voy / arc) | control mean (eg) | length correlation (eg / voy / arc) |
|---|---|---|---|
| `cprof_sent_top1` | 0.5642 / 0.5042 / 0.4988 | 0.2352 | +0.189 / +0.114 / +0.199 |
| `cprof_sent_top2` | 0.5222 / 0.4557 / 0.4770 | 0.1577 | +0.431 / +0.309 / +0.460 |
| `cprof_sent_top3` | 0.4928 / 0.4632 / 0.4531 | 0.0978 | +0.522 / +0.456 / +0.560 |
| `cprof_ngram_top1` | 0.4728 / 0.4376 / 0.4830 | 0.3980 | +0.306 / +0.279 / +0.294 |
| `cprof_ngram_top2` | 0.4550 / 0.4240 / 0.4608 | 0.3523 | +0.399 / +0.350 / +0.386 |
| `cprof_ngram_top3` | 0.4607 / 0.4262 / 0.4377 | 0.3277 | +0.444 / +0.392 / +0.431 |
| `cprof_align_sent` | 0.5104 / 0.5085 / 0.5060 | 0.0092 | −0.070 / −0.048 / −0.070 |
| `cprof_align_ngram` | 0.4896 / 0.4915 / 0.4940 | 0.9908 | +0.070 / +0.048 / +0.070 |
| `cprof_align_lexical` | 0.4691 / 0.4758 / 0.4762 | 0.2500 | +0.130 / +0.125 / +0.149 |
| `cprof_align_semantic_only` | 0.5309 / 0.5242 / 0.5238 | 0.7500 | −0.130 / −0.125 / −0.149 |

Three readings.

- The block's best feature is a **"best local match"** signal
  (`cprof_sent_top1`), and it is *weaker* than what the v0 pack already has:
  on embeddinggemma 0.5642 against `chunk_cosine_best`'s 0.6049. The profile
  therefore re-measures the v0's best chunk cosine at a finer granularity
  without adding anything.
- The **lower ranks are length proxies**: the deeper the rank, the more the
  feature travels with the paragraph's length (+0.43 to +0.56 for
  `cprof_sent_top3`), and the more the value is a "the paragraph has more
  pieces" signal than a topical one. A length proxy memorizes the document
  instead of the topic — the plan's leakage rule (ii) — and the v0 pack has
  no such feature. This is the risk the reviewer flagged, and it is measured
  here, not assumed.
- The **alignment kinds are at chance** (0.469–0.531) and their control-side
  means show how lopsided they are: the ordinary cells' best alignment is a
  word-piece match 99% of the time (`cprof_align_ngram`'s control mean
  0.9908) and a sentence match 1% of the time (`cprof_align_sent` 0.0092), so
  the kind is nearly a constant of the corpus rather than a property of the
  pair — a fitted model can read it as a document-level prior at best.

**6. The retrieval side: the full arm is a worse retriever.** The positives'
hit@1 (the true match is the paragraph's own best candidate):

| arm | embeddinggemma | voyage-4-nano | arctic2 |
|---|---|---|---|
| the full-text cosine | 0.1355 | 0.1050 | 0.1050 |
| `base` | 0.0897 | 0.0821 | 0.0859 |
| `full` | 0.0802 | 0.0763 | 0.0649 |
| `profile` | 0.0840 | 0.0916 | 0.0878 |

The Stage-3.1 caveat ("a learned pair ranker is not automatically a better
retriever") deepens: adding the chunk profile costs another 0.010–0.021 of
hit@1 on embeddinggemma and arctic2, and the block alone is no better than
the cosine's neighbourhood. The hard negatives move the other way — the
`profile` arm's negatives' hit@1 roughly **doubles** (0.0181 / 0.0221 / 0.0211
against the base's 0.0090 / 0.0110 / 0.0100) and their rank position rises
(the negatives' rank percentile p50 0.2412 / 0.2109 / 0.2428 against the
base's 0.3745 / 0.3261 / 0.3779): the block alone is **less safe** for the
reviewed hard negatives. The full arm pushes them slightly *down* instead
(0.4156 / 0.3413 / 0.3894) — the one point in the block's favour for the
track's deliverable.

**7. The control side: no inflation.** The ordinary cells' rank percentile
moves by ≤ 0.002 for every arm (−0.0085 to +0.0019) and their top-fifth
share by ≤ 0.013. So no arm's number on the reviewed set is an inflation of
the ordinary cells.

**8. The length check.** The Stage-0 caveat asks how much of a method's scale
shift travels with the paragraph's length. The correlation between a control
cell's paragraph length and its score, for the learned arm and for the cosine
(the reference), and the same correlation for the **rank shift** (the arm's
rank percentile minus the cosine's — scale-free, so the two arms compare
without their different units):

| arm | embeddinggemma | voyage-4-nano | arctic2 |
|---|---|---|---|
| the cosine (the reference) | +0.162 | +0.031 | +0.094 |
| `base` | +0.245 | +0.275 | +0.366 |
| `full` | +0.205 | +0.212 | +0.282 |
| `profile` | +0.030 | −0.060 | +0.051 |
| the rank shift (`full` − the cosine) | −0.010 | −0.019 | −0.025 |

The learned arm's score travels more with the length than the cosine's (the
lexical block's doing — it is a bag-of-words overlap, and the longer the
paragraph the more words it can share). Adding the chunk profile **lowers**
that correlation (0.245 → 0.205, 0.275 → 0.212, 0.366 → 0.282) and the
length ↔ rank-shift correlation is ≈ 0 for every arm: the profile block does
not add a length-driven shift, and the arm's own length correlation is not
what the gain comes from.

**What this run does not say.**

- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice only.
- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track), so the track's safe-negative rule still
  ships as the **margin half only**.
- **Nothing about a corpus with a different vocabulary.** The lexical block
  is a bag-of-words overlap over 294 questions, and the chunk profile is
  measured on the same corpus; whether either transfers is open (the plan's
  §5).
- **Nothing about a better taxonomy for the alignment kinds.** The taxonomy
  the reviewer chose (the granularity × the lexicalness) is at chance here;
  whether another taxonomy (e.g. which *question part* the best alignment
  lands in) would say something is open, and it would need the question's
  options inside the profile's question side.
- **Nothing about the df.** It was dropped, not measured: a fit-side df would
  need a per-rotation feature matrix, and the counts on ~150 fit-side
  paragraphs would be 0/1/2.

**The next step (CS-6d).** The Stage-3 **random-vector floor** — the second
sanity floor of the plan's Stage 3 (the label-permutation one is done): the
arm fitted on random vectors instead of the model's embeddings, so a "win" can
be checked against what the *training design* alone buys. Then CS-6e (the
calibration curves, the thresholds per document, and the safe-negative margin
measured against the control-role floor).

**A note on the reporting.** Two blocks were added to the Stage-3 artifact
(this change set): the **per-feature diagnostics** (every feature's own
discrimination number, control-side counterpart and length correlation) and
the **length check** (the arm's length ↔ score correlation beside the
cosine's, and the length ↔ rank-shift correlation — the plan's §5 open item
about the shift, which needed a second arm in the same run). The Stage-3
block also gained the arms' **per-pair out-of-fold scores**
(`out_of_fold_scores`), which is what makes the paired per-pair comparison
between two arms of different runs possible (the CS-6b open question); the
new tool `tools/compare_stage3_arms.py` reads them.

## Stage 3 — the random-vector floor (the plan's second sanity floor) (CS-6d, 2026-10-26)

**The question.** The plan's Stage 3 asks for **two** sanity floors, so that a
"win" can be checked against chance: the **label-permutation** floor (done in
CS-5 — the fit side's labels are destroyed) and the **random-vector** floor:
*the arm is fitted on random vectors instead of the model's embeddings, so a
"win" can be checked against what the training design alone buys*. Two
earlier change sets had left the question open: the Stage-3.1b ablation
measured that the embedding geometry adds no label-driven signal, and the
Stage-3.3 chunk profile is **entirely** embedding-derived, so the floor is
also the sharpest statement available about that block.

**What the floor replaces.** The feature pack's **embedding sides** only: the
full-text cosine, the chunk cosines and the chunk profile's unit embeddings.
The pack's lexical, question-part and question-type features never used an
embedding, so they stay real. The run's **baseline arm** (the full-text
cosine every arm is compared against) keeps the model's embeddings, so the
track's guard still holds and the arm-vs-cosine comparison stays meaningful.

**The random vectors.** A seeded random unit vector of the **model's
dimension** per `(kind, text)`: `sha256(seed | kind | text)` seeds a
`numpy.random.default_rng`, the vector is drawn from `uniform(-1, 1)` and
L2-normalized. The seed is the run's own `config.seed` (42); the recipe is
recorded in the artifact (`stage3.feature_embeddings`); the vectors are
**never** written to the embedding cache, so they cannot pollute the model's
cache.

**The runs.** Nine runs (3 Stage-0 models x 3 group sets,
`experiments_text_processing/stage3_random_floor_sweep.py`, the sweep
`stage3_random_floor`):

- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-rand-base` — the
  v0 pack's four groups;
- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-rand-full` — the
  v0 pack **plus** the chunk profile;
- `omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-rand-profile` —
  the chunk profile **alone**.

The three group sets are the Stage-3.3 ones, so every CS-6c arm gets its
random-vector floor. The protocol is unchanged from Stage 3.1: the same
document-level holdout (three rotations, one out-of-fold score matrix), the
same training data (the fit side's judged pairs plus its unjudged control
cells as down-weighted presumed negatives, weight 0.3), the same two floors
(the label-permutation chance floor and the control-role floor, 5
permutations per rotation, seed 42), the same paired statistics against the
full-text cosine (2,000 bootstrap resamples, seed 42) and the same combiner
(the logistic regression — the learner is not the lever, CS-6b). The guard
holds: all **81** runs reproduce the Stage-0 baseline arm field by field
(`tools/verify_analysis_baseline.py "omct-analyze-*"`).

**1. The headline: the within-document AUC, the real arm beside its
random-vector floor** (OMCT auxiliary data, the reviewed set's 1,520 judged
pairs — never a UPR metric). The CI is the paired per-pair bootstrap of the
two arms' out-of-fold scores (`tools/compare_stage3_arms.py`, stratified by
document and class, 2,000 resamples, seed 42):

| group set | model | the real arm | the random floor | Δ (real − random) | the paired 95% CI |
|---|---|---|---|---|---|
| `base` | embeddinggemma | 0.6655 | 0.6568 | +0.0088 | [−0.0362, +0.0203] |
| `base` | voyage-4-nano | 0.6723 | 0.6588 | +0.0135 | [−0.0392, +0.0117] |
| `base` | arctic2 | 0.6472 | 0.6565 | −0.0093 | [−0.0120, +0.0319] |
| `full` | embeddinggemma | 0.6722 | 0.6617 | +0.0105 | [−0.0383, +0.0174] |
| `full` | voyage-4-nano | 0.6894 | 0.6630 | +0.0264 | [−0.0519, −0.0011] |
| `full` | arctic2 | 0.6423 | 0.6596 | −0.0173 | [−0.0063, +0.0416] |
| `profile` | embeddinggemma | 0.5613 | 0.4733 | +0.0880 | [−0.1331, −0.0445] |
| `profile` | voyage-4-nano | 0.5324 | 0.4830 | +0.0494 | [−0.0901, −0.0077] |
| `profile` | arctic2 | 0.4974 | 0.4706 | +0.0268 | [−0.0730, +0.0203] |

(The full-text cosine, the reference: 0.5632 / 0.5238 / 0.5020.)

**2. The verdict on the v0 pack and on the full pack: the embedding geometry
adds nothing to the ranking.** The random floor is **indistinguishable** from
the real arm on five of the six arms — all three `base` arms and `full` on
embeddinggemma and arctic2; every one of those five intervals includes zero.
On arctic2 the random floor is even *higher* than the real arm on both group
sets (+0.0093 and +0.0173, both including zero). The sixth (voyage-4-nano's
`full`) is the only arm where the real arm is significantly ahead
(+0.0264 [−0.0519, −0.0011]) — and that is the arm the Stage-3.3 chunk
profile was already established on.

The random floor's `base` arms give 0.6568 / 0.6588 / 0.6565 — within 0.0023
of each other across the three models, the built-in check the Stage-3.1b
ablation used for its embedding-free arms (0.6589 on all three models; the
small spread is the random vectors' dimension, which differs per model). And
they land **on** that arm: with noise in the embedding slots, the v0 pack
*is* the embedding-free pack.

**3. The verdict on the chunk profile: the block's whole signal is the
embedding geometry — and it was never the lever.** The random floor of the
profile block alone sits **at or below chance**: 0.4733 / 0.4830 / 0.4706
(the arm's own chance floor: 0.5067 / 0.5182 / 0.5011). The real block is
significantly ahead of its own floor on two of the three models
(embeddinggemma +0.0880 [−0.1331, −0.0445], voyage-4-nano +0.0494
[−0.0901, −0.0077]; arctic2's +0.0268 [−0.0730, +0.0203] includes zero). So
the block is a re-measurement of the embedding geometry and nothing else —
which is the same conclusion the Stage-3.3 paired comparison reached from the
other side (the real block was already indistinguishable from the plain
cosine). Two independent routes, one verdict: **the chunk profile is not the
lever.**

**4. The floors: the embedding features carry the training design's prior,
not label-driven signal.** The within-document floors (the chance floor
permutes the whole label vector, the control-role floor permutes only the
judged pairs' labels and keeps the control cells' presumed-negative role):

| arm | the real arm: chance / control-role | the random floor: chance / control-role |
|---|---|---|
| `base` (eg / voy / arc) | 0.5054 / 0.4936 / 0.4968 — 0.5788 / 0.5765 / 0.5569 | 0.4877 / 0.4856 / 0.4885 — 0.5434 / 0.5410 / 0.5431 |
| `full` (eg / voy / arc) | 0.5248 / 0.5138 / 0.5313 — 0.5317 / 0.5372 / 0.5232 | 0.5005 / 0.5103 / 0.4945 — 0.5402 / 0.5409 / 0.5395 |
| `profile` (eg / voy / arc) | 0.5324 / 0.5238 / 0.5343 — 0.5010 / 0.4532 / 0.4750 | 0.5067 / 0.5182 / 0.5011 — 0.4355 / 0.4361 / 0.4357 |

The random arms' chance floors sit at chance, as the plan requires. The
**excess over the control-role floor** — the label-driven part, the yardstick
the track's deliverable is measured against:

| arm | the real arm | the random floor |
|---|---|---|
| `base` (eg / voy / arc) | +0.0868 / +0.0958 / +0.0903 | +0.1133 / +0.1177 / +0.1134 |
| `full` | +0.1405 / +0.1522 / +0.1191 | +0.1215 / +0.1221 / +0.1201 |
| `profile` | +0.0603 / +0.0792 / +0.0224 | +0.0378 / +0.0470 / +0.0349 |

The v0 pack's **label-driven part does not grow with the embedding
geometry**: the real arm's excess is +0.0868 / +0.0958 / +0.0903 and the
random floor's is +0.1133 / +0.1177 / +0.1134 — with noise in the embedding
slots the label-driven part is if anything *larger*. What the embedding
features add is the opposite: the real `base` arm's control-role floor is
**higher** than the random floor's (0.5569–0.5788 against 0.5410–0.5434), so
part of the real arm's number is the **training design's own prior** picked
up through the embedding features — the reviewed set is rank-selected
(the Stage-0.2 classification), and the embedding geometry picks that
structure up. That is the sharpest form of the Stage-3.1b ablation's
finding, now measured with a floor instead of a column removal: the
embedding features carry the benchmark's *construction*, not its labels.

(The excess is not comparable *across* arms with different feature sets — the
floor is itself a property of the arm's features. The primary statement is
the paired difference in §1.)

**5. The hit@1: the embedding geometry is the retrieval signal, the lexical
block is the ranking signal.** The positives' hit@1 (how often the pair's
question is the paragraph's best candidate):

| arm | the real arm (eg / voy / arc) | the random floor | the full-text cosine |
|---|---|---|---|
| `base` | 0.0897 / 0.0821 / 0.0859 | 0.0401 / 0.0420 / 0.0382 | 0.1355 / 0.1050 / 0.1050 |
| `full` | 0.0802 / 0.0763 / 0.0649 | 0.0363 / 0.0382 / 0.0267 | " |
| `profile` | 0.0840 / 0.0916 / 0.0878 | 0.0076 / 0.0057 / 0.0153 | " |

The random floor's hit@1 collapses to 0.027–0.042 for the v0 pack and the
full pack, and to 0.006–0.015 for the profile block alone — the level of (or
below) the Stage-3.1b embedding-free arms (the `no-embed` arm's 0.0344 and
the `lex-only` arm's 0.0324 on embeddinggemma). So the two roles separate
cleanly:

- the **lexical** block is what makes the learned arm a better *pair ranker*
  than the cosine (it survives the floor: the random floor's `base` arm is
  still 0.657 against the cosine's 0.563 on embeddinggemma);
- the **embedding geometry** is what makes anything a *retriever* at all
  (the floor's hit@1 is a third of the cosine's), and the cosine is still
  the better retriever than the learned arm.

**6. The per-feature check: the floor replaced exactly the embedding sides.**
The nine artifacts carry the per-feature diagnostics for every column the
pack computes. In the random runs, the **nine non-embedding features are
bit-identical** to the real runs — `lex_question_coverage` 0.6147,
`lex_paragraph_coverage` 0.3653, `lex_char_ngram_cosine` 0.4460,
`qpart_options_coverage` 0.4952 and the five question-type one-hots (0.4494–
0.5767), the same numbers in all three models, as they must be: they do not
use an embedding. The embedding features' own within-document AUC collapses to
chance (`cosine_full` 0.5632 → 0.4906, `chunk_cosine_best` 0.6049 → 0.5267
on embeddinggemma), and the profile's features do the same
(`cprof_sent_top1` 0.5642 → 0.4712). That is the internal proof that the
floor is a like-for-like replacement and nothing more.

**7. The control side: no inflation.** The random arms' control rank
percentile sits at chance (p50 0.5021–0.5115) — as it must, since the random
vectors carry no signal — and the real arms' beside it (0.4938–0.5062).

**What this run does not say.**

- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice only.
- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track), so the track's safe-negative rule still
  ships as the **margin half only**.
- **Nothing about "the" null.** The floor is **one draw** (seed 42): a
  different seed would give a slightly different number. The five `base` /
  `full` arms' intervals are wide enough that the conclusion does not rest on
  the draw, but a single seed is a single draw.
- **Nothing about a better feature family.** The floor says the *embedding
  geometry* carries no label-driven signal on this benchmark; it does not say
  that no feature could (the plan's §5 question about the corpus as the
  ceiling stays open).
- **Nothing about a corpus with a different vocabulary.** Same caveat as
  Stage 3.3: the lexical block is a bag-of-words overlap over 294 questions.

**The next step (CS-6e).** The Stage-3 **reporting**: the calibration curves,
the thresholds per document, and the safe-negative margin measured against
the **control-role floor** (the reviewer's decision: the track's durable
deliverable is measured against that floor, not against the plain cosine; the
rule still ships as the margin half only). The floor measured here says what
that margin has to beat: 0.5410–0.5434 for a pack with noise in the embedding
slots, 0.5569–0.5788 for the real one.

**A note on the reporting.** The Stage-3 artifact gained one block in this
change set: `stage3.feature_embeddings` — the provenance of the pack's
embedding sides (the mode, the seed, the dimension and the draw rule), so a
reader knows which space an arm's embedding features live in. The run-level
config echo carries the mode as well (`analysis.stage3.embeddings`).

## Stage 3 — the reporting: the calibration, the thresholds and the safe-negative margin (CS-6e, 2026-10-26)

**The question.** The plan's Stage 3 carries a *reporting* discipline —
"calibration curves + thresholds per document; pooled and per-document macro;
both negative subsets separately" — and the track's named deliverable
(plan §8) is *"a measured 'which unjudged pairs are safe negatives' rule"*.
CS-6e measures both: the reliability diagram of the learned arm's
probability, the **per-document threshold** (chosen on the fit side, applied
to the held-out document), and the **safe-negative margin** measured against
the **control-role floor** (the reviewer's decision of 2026-10-26: the
deliverable is measured against that floor, not against the plain cosine;
the rule still ships as the **margin half only**).

**What was run.** Twelve runs (`experiments_text_processing/stage3_reporting_sweep.py`,
the sweep `stage3_reporting`): 3 Stage-0 models x 2 feature-group sets
(`base` = the v0 pack's four groups, `full` = the v0 pack plus the chunk
profile) x 2 **fit-side criteria** of the per-document threshold
(`floor_gap`, `youden`) —
`omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage3-report-{base,full}-{floor_gap,youden}`.

Nothing else changed: the same document-level holdout (three rotations, one
out-of-fold score matrix), the same training data (the fit side's judged
pairs plus its unjudged control cells as down-weighted presumed negatives),
the same two floors, the same combiner. The guard holds: all **93** runs
reproduce the Stage-0 baseline arm field by field
(`tools/verify_analysis_baseline.py "omct-analyze-*"`).

**One new discipline: the fit side's own rows are scored by an *inner*
cross-fit.** The threshold and the margin are **fit-side choices**, and the
fit side's rows have to be scored by *some* model. The out-of-fold matrix
would have done it with a leak: the model that scored a fit-side document
was trained on the holdout. So the fit side's rows are scored one level in —
the same document-level holdout, applied to the two fit-side documents — and
every fit-side number (the threshold's criterion, the margin's criterion) is
computed on those. No model that saw the holdout produced a number the
holdout is judged on.

**1. The calibration: the arm's probability is not a probability, and the
cosine's calibration map is worse.** The expected calibration error (the
count-weighted mean gap between the predicted probability and the observed
positive rate, over ten equal-width bins) and the Brier score, for the
learned arm's own probability and for the full-text cosine put on a
probability scale by a **fit-side** calibration map (a logistic regression
on the cosine as its single feature):

| arm | the arm's ECE (pooled / per-document macro) | the arm's Brier | the cosine's ECE (pooled) | the cosine's Brier |
|---|---|---|---|---|
| embeddinggemma `base` | 0.1806 / 0.3327 | 0.2589 | 0.2132 | 0.2776 |
| embeddinggemma `full` | 0.1749 / 0.3274 | 0.2601 | 0.2132 | 0.2776 |
| voyage-4-nano `base` | 0.1875 / 0.3344 | 0.2641 | 0.2097 | 0.2777 |
| voyage-4-nano `full` | 0.1891 / 0.3278 | 0.2637 | 0.2097 | 0.2777 |
| arctic2 `base` | 0.1875 / 0.3488 | 0.2650 | 0.2226 | 0.2783 |
| arctic2 `full` | 0.1816 / 0.3413 | 0.2640 | 0.2226 | 0.2783 |

Three things to read here.

- **The arm is miscalibrated everywhere.** An ECE of 0.18–0.19 (pooled) and
  0.33–0.35 (per document) means the arm's probabilities are off by roughly
  a third of the scale. A threshold on this probability is a threshold on a
  number that does not mean "probability".
- **The cosine's own calibration map is *worse* than the arm** (ECE 0.210–0.223
  against 0.175–0.189; the Brier 0.278 against 0.259–0.265). Putting the
  geometry on a probability scale does not make it a probability.
- **The pooled ECE flatters both arms** (0.18 pooled against a 0.33 macro).
  The pooled diagram mixes the three documents' calibrations, so a reader who
  only looks at the pooled number sees a better-calibrated model than exists.
  This is the reviewer's own caveat, measured: a single global cut over the
  out-of-fold matrix is **not** like-for-like. The miscalibration is worst on
  one document in every run — the per-document ECEs are 0.20–0.24 for two
  documents and **0.52–0.62** for the third (the AI document: 99 pairs, 80%
  positive), where the arm says ~0.5 where the observed rate is ~0.8.

**2. The per-document threshold: the `floor_gap` criterion transfers, the
`youden` criterion does not.** The threshold is chosen on the **fit side**
(the other two documents) and applied to the held-out one. The two criteria
differ in the negative class the false-positive rate is measured on:
`floor_gap` measures it on the **unjudged control cells** (the rule's target
class) and `youden` on the **reviewer-verified hard negatives** (the reviewed
set's own ground truth).

| arm | the macro recall of the per-document thresholds (`floor_gap`) | (`youden`) | the pooled out-of-fold recall (`floor_gap` / `youden`) |
|---|---|---|---|
| embeddinggemma `base` | 0.4720 | 0.3738 | 0.4084 / 0.4599 |
| embeddinggemma `full` | 0.4769 | 0.4103 | 0.5477 / 0.5878 |
| voyage-4-nano `base` | 0.4666 | 0.2457 | 0.3435 / 0.2176 |
| voyage-4-nano `full` | 0.6247 | 0.3843 | 0.6813 / 0.3550 |
| arctic2 `base` | 0.3159 | 0.0443 | 0.2767 / 0.0267 |
| arctic2 `full` | 0.3872 | 0.1122 | 0.3302 / 0.0782 |

The **`floor_gap` criterion wins on 6 of 6 arms** on the like-for-like
statistic (the per-document macro recall of the holdout's operating point).
The `youden` criterion is the one that looks principled — it separates the
positives from the *ground-truth* hard negatives — and it is the one that
fails: the hard negatives sit **above** the positives (the label anomaly the
track designed around from the start), so a threshold that separates the
positives from them lands far too high for the positives. On arctic2 the
holdout's recall at the `youden` threshold is 0.02–0.06. The rule for the
track's deliverable is therefore the **control-side** criterion, not the
reviewed-set one.

**The per-document choice does not transfer everywhere.** For the NU
document (845 pairs, a 25% positive rate — the lowest base rate) the
fit-side threshold is much higher than for the other two, and the holdout's
recall at it is 0.02–0.29 at a precision of 0.87–1.00: the fit side's choice
for that document is *not* transferable, the same failure the Stage-2
semantic threshold showed. The pooled out-of-fold cut is reported beside the
per-document ones and is **not** like-for-like (three calibrations in one
matrix); the per-document thresholds are the ones to read.

**3. The safe-negative margin rule: the reach is measured, and the quality
signal is blind.** The rule is a **margin rule**, never an absolute cut (the
track's rule: the hard negatives sit above the positives' 95th percentile, so
no absolute cut separates them, and a margin is measured *inside* a
paragraph's own candidate list, so a document's score level never enters it).
A cell is a **safe negative** when it sits at least the chosen margin below
its own paragraph's best candidate; the margin is chosen on the fit side by
the rule's own criterion (the coverage over the fit side's unjudged cells
minus the harm on the fit side's verified positives) and applied to the
held-out document.

Three variants, measured on the **whole 42,947-cell unjudged grid** (not
only the 4,000-cell control sample) with the **harm** on the 524 verified
positives — the only ground-truth number the rule can be judged on:

| rule | the reach over the unjudged grid | the harm on the verified positives | the control-role floor's harm **at the same reach** |
|---|---|---|---|
| the cosine's margin (the geometry) | 0.7763–0.8286 | 0.3130–0.3378 | — (not a fitted model: no floor) |
| the arm's margin (the ranking) | 0.5895–0.6539 | 0.2615–0.4008 | 0.2489–0.3060 |
| **both margins** (retrieve with the cosine, rank with the arm) | **0.4999–0.5950** | **0.1088–0.1927** | **0.1286–0.1788** |

(embeddinggemma / voyage-4-nano / arctic2, the `base` and the `full` group
sets; the numbers are the ranges over the six arms. The floor's number is
read off the **floor's own curve of the same rule** at the real arm's reach,
per document, then averaged over the three documents.)

**The two-part rule is the best of the three**: it reaches 50–60% of the
unjudged space (21,471–25,555 cells of the 42,947) while discarding
11–19% of the verified positives, where the geometry alone reaches much
further (78–83%) at a much higher cost (31–34%) and the arm's margin alone
reaches 59–65% at 26–40%. This is CS-6d's "retrieve with the cosine, rank
with the learned arm" in the shape of a rule.

**And the verdict on its quality: the measurement cannot establish it.** At
the same reach, the **control-role floor** — the model fitted with the
judged pairs' labels permuted, which knows nothing about the labels — is as
good as the real arm: the floor's harm at the real arm's reach is
0.1286–0.1788 against the real arm's 0.1088–0.1927, and the real arm is
better on 0–3 of the three documents per arm (voyage-4-nano's `full` arm is
the only one where it is better on all three). The arm-margin variant is
worse: the floor matches or beats it on 4 of the 6 arms.

The reason is the benchmark's construction, and it is the same artefact the
Stage-0.2 classification found: the reviewed set is **rank-selected** (the
pairs a model put on top, which a human reviewed), so the verified positives
sit near the top of their paragraph's candidate list **whatever model
produces the scores**. A margin rule protects the cells near the top, so it
protects the positives — even a label-blind one. The harm on the verified
positives therefore measures the *construction* of the reviewed set, not the
rule's topical quality, and it cannot tell a good rule from a bad one.

**So the durable deliverable ships with the claim it can carry.** The rule's
**reach** is measured (50–60% of the unjudged grid), its **cost on the known
positives** is measured (11–19% of them), and its **precision on the unjudged
cells is unmeasured** — the unjudged cells' true positive rate stays
unmeasured, no human labelling in this track, and the floor comparison says
the one available proxy is blind. The rule ships as the **margin half only**:
a recipe that says which unjudged cells can be *provisionally* treated as
negatives, with a stated reach and a stated cost, and **no precision claim**.

**What this run does not say.**

- **Nothing about the unjudged cells' positive rate.** It stays unmeasured
  (no human labelling in this track). The rule ships as the margin half
  only; no spot-check sheet is proposed.
- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice and characterize the deliverable, they are never a UPR
  metric.
- **Nothing about the rule's precision on the unjudged cells** — the reach
  and the cost on the known positives are all that is measured, and the floor
  comparison says the cost is not evidence of topical quality.
- **Nothing about a different corpus.** Same caveat as Stage 3.3: the
  lexical block is a bag-of-words overlap over 294 questions, and the
  margins are measured inside one document's candidate list.
- **Nothing about a second seed.** The floor is five permutation draws per
  rotation (their mean is reported), and the margin's fit-side choice is one
  grid of quantiles.
- **Nothing about a different feature family** (the corpus-as-ceiling
  question of §5 stays open).

**The next step.** The track's Stage 3 is now measured end to end: the arm's
quality (CS-5/CS-6a/CS-6b/CS-6c/CS-6d), the reporting (this change set) and
the deliverable's measurement. What remains open is the track's §8 question
— *how the recipe is used downstream* (PLAN's Phase 6 reranker slot and
Phase 9 fine-tuning pairs) — and the one thing the track cannot measure
without human labelling: whether the unjudged cells the rule clears really
are negatives. Stage 4 (the LLM experiments) and the UPR port are the
reviewer's call.

**A note on the reporting.** The Stage-3 artifact gained three blocks in this
change set (`stage3.reporting`): `calibration` (the reliability diagram of
the arm's probability and of the cosine's fit-side calibration map, pooled
and per document, with the bin counts, the ECE and the Brier score),
`thresholds` (each document's fit-side choice, its holdout's operating point,
and the pooled out-of-fold cut beside them) and `safe_negative` (the three
rule variants, the reach over the unjudged grid, the harm on the verified
positives, the control-role floor's counterpart and the explicit statement
that the unjudged cells' true positive rate is unmeasured). The reporting is
**opt-in** (`analysis.stage3.reporting.enabled`, the default `false`), so
every earlier run's artifact keeps its exact content: re-running the
Stage-3.3 sweep reproduces its numbers field for field (verified — only the
config echo carries the new key).

## Stage 3.5 — the decision models: the probe (CS-7a, 2026-10-26)

**The question.** The plan's Stage 3.5 asks the question the whole track has
been circling: **is the reviewed set learnable at all?** Every embedder sits at
chance on it — the within-document AUC **0.5632** (embeddinggemma), **0.5238**
(voyage-4-nano) and **0.5020** (arctic2) — and the reviewed set is
**rank-selected** (exp18's model chose which pairs got reviewed), so those
numbers may be measuring the selection rather than the labels. A **decision
model** — a frozen, zero-shot scorer that answers a typed question with a
**probability** — is the first *independent* ranker on this benchmark, and it
attacks the deliverable's weakest link at the same time: a calibrated
probability means the safe-negative rule can take a **fixed** cut instead of a
fitted, non-transferable threshold (CS-6e's open question).

**What was run.** Eight runs
(`experiments_text_processing/stage35_decision_sweep.py`, the sweep
`stage35_decision`): 2 models x 2 state shapes x 2 question types —
`omct-decide-{tev1,laya}-{pair,paragraph}-{noul,choice}`. The scope is
`judged`: the 1,520 reviewed pairs plus the **same seeded 4,000-cell unjudged
control** the Stage-0 runs draw, so the control side is comparable. The
**synthetic competence control** (100 constructed positive pairs against 100
unrelated ones) runs beside every arm.

**The infrastructure (new).** The `Decider` port, the `/v1/systemone` client
(a raw HTTP call — the endpoint is in neither the Ollama CLI nor its
Python/JS libraries), the decision models' **own** disk cache (its own root
under `cache/`, keyed by the model id, the digest, the prompt version, the
state and the question specs — never the embedding cache), the **call
planner** (the state's shape and the model's input budget decide the
grouping) and the pure side (`text_processors/decision/`: the question specs,
the plan, the competence control, the measurement). The models are pinned by
id **and digest** and the run **fails** on a mismatch. The prompt is a config
value, not a code constant.

### 1. The verdict: the reviewed set IS learnable, and tev1 beats every embedding arm

The headline is the **within-document AUC** (the pooled number sits beside it,
never alone: 58.7% of the reviewed set's positive-negative pairs are
cross-document, so the pooled number moves with the documents' score levels).

| arm | within-doc AUC | pooled AUC | per-doc macro | ECE (pooled / macro) | Brier | the floor gap | the competence AUC | the calls |
|---|---|---|---|---|---|---|---|---|
| **tev1-pair-choice** | **0.7767** | 0.7691 | 0.8028 | **0.0891** / 0.2009 | 0.1881 | 0.3943 | 0.9992 | 5,520 + 200 |
| tev1-paragraph-choice | 0.7708 | 0.7689 | 0.8021 | 0.1128 / 0.2235 | 0.1932 | 0.3532 | 0.9996 | 339 + 100 |
| tev1-pair-noul | 0.7604 | 0.7575 | 0.7846 | 0.1128 / 0.2364 | 0.1966 | 0.3064 | 0.9996 | 5,520 + 200 |
| tev1-paragraph-noul | 0.7244 | 0.7350 | 0.7605 | 0.1365 / 0.2443 | 0.2064 | 0.2689 | 0.9996 | 339 + 100 |
| laya-paragraph-noul | 0.5992 | 0.6049 | 0.6089 | 0.1188 / 0.2142 | 0.2330 | 0.2013 | 0.9443 | 2,048 + 100 |
| laya-pair-noul | 0.5984 | 0.5982 | 0.6215 | 0.1863 / 0.2574 | 0.2551 | 0.2466 | 0.9475 | 5,520 + 200 |
| laya-pair-choice | 0.5700 | 0.5890 | 0.6323 | 0.2423 / 0.2305 | 0.2944 | 0.1733 | 0.9671 | 5,520 + 200 |
| laya-paragraph-choice | 0.5555 | 0.5665 | 0.6138 | 0.2355 / 0.2535 | 0.2929 | 0.1438 | 0.9547 | 2,048 + 100 |

(The calls are the arm's own plus the competence control's; the first run
served them from a cold cache, and the re-run reproduced every number exactly
from the cache.)

The comparison that matters (all on the same 1,520 reviewed pairs of the
**OMCT** data):

| scorer | within-doc AUC | pooled AUC | the ECE (pooled) | the positives' hit@1 |
|---|---|---|---|---|
| the full-text cosine (the Stage-0 arms) | 0.5020–0.5632 | 0.4680–0.5371 | — | 0.1050–0.1355 |
| the track's learned arm (the v0 pack, CS-5) | 0.6472–0.6723 | 0.5419–0.5831 | 0.175–0.189 | 0.0821–0.0897 |
| the control-role floor (the learned arm's yardstick) | 0.5569–0.5788 | — | — | — |
| **tev1 (the best arm)** | **0.7767** | **0.7691** | **0.0891** | — |
| laya (the best arm) | 0.5992 | 0.6049 | 0.1188 | — |

**tev1 is far above every other scorer on this benchmark** — above the
embedders by 0.21–0.27 of within-document AUC and above the track's own
learned arm by 0.10. **laya is not**: its best arm (0.5992) beats the cosine
(0.5020–0.5632) but loses to the track's learned arm (0.6472–0.6723). So the
plan's decision tree lands on its first branch — *well above chance* — **for
tev1**, and on a middle branch for laya.

**The control side agrees.** The floor gap (the verified positives' mean minus
the control cells' mean) is 0.269–0.394 for tev1 (the positives average
0.337–0.465, the ordinary cells 0.061–0.071) against 0.144–0.247 for laya.
The reviewed set's positives are **not** the ordinary cells, and an
independent, label-blind-free scorer separates them.

**What this changes.** The track's Stage-0.2 classification said the reviewed
set's *construction* dominates its signal ("the benchmark is construction-
dominated"). That is now **too strong**: the labels are learnable, and the
learner is a zero-shot model with no knowledge of the corpus. What the
construction *does* do is make the **embedding** geometry useless as a
ranker on this set — the rank-selection means an embedder's score on a
reviewed pair mostly measures *where the pair sat in exp18's ranking*, not
whether the pair is a true match. The deliverable's quality claim was
blocked by that; the blocker was the *scorer*, not the labels.

### 2. The competence control passes decisively — the models are the right shape

The synthetic control (a paragraph and a question **sharing** a distinctive
phrase against an **unrelated** pair) reaches an AUC of **0.944–0.9996** on
every arm: the positives average 0.776–0.924, the negatives 0.046–0.379. So
both models can do the task, and the reviewed set's labels are topically
separable — the ambiguity the plan flagged ("*the data is at the ceiling* vs.
*the model is the wrong shape*") is resolved: **the data is not at the
ceiling**.

### 3. The prompt grid: the two-option `choice` wins for tev1, the `noul` wins for laya

The docs warn that a `noul` question can get "stuck on no". That is
measurable here, and it is **model-dependent**:

- **tev1**: the `choice` type beats the `noul` in both shapes (the
  within-document AUC 0.7767 vs. 0.7604 in the `pair` shape, 0.7708 vs.
  0.7244 in the `paragraph` shape), and it is also the better-calibrated arm
  (the ECE 0.0891 vs. 0.1128 pooled).
- **laya**: the `noul` wins in both shapes (0.5984 vs. 0.5700, 0.5992 vs.
  0.5555) — the opposite of the warning, and laya's `choice` arms are the
  worst-calibrated of the eight (the ECE 0.2355–0.2423 pooled).

**The state's shape costs, but it is nearly free in quality.** The `pair`
shape (the pair in one state) needs **5,520 calls** per arm; the `paragraph`
shape (the paragraph alone as the state, the question in the instructions)
needs **339** (tev1) or **2,048** (laya) — a **16× / 2.7×** cheaper arm for
0.006 (tev1, `choice`) to 0.036 (tev1, `noul`) of within-document AUC. For
the grid scope (CS-7b, the whole 44,467-cell grid) that difference is the
difference between minutes and hours.

### 4. The calibration: tev1's probability is a probability (and it is better calibrated than the track's own arm)

tev1's card says its calibration is **untested**. It is now measured, on the
reviewed set: the expected calibration error **0.0891–0.1365 pooled** and
**0.2009–0.2443 per-document macro**, the Brier score **0.1881–0.2064**. The
track's own learned arm (CS-6e) had the ECE 0.175–0.189 pooled and
0.327–0.349 per-document macro at a Brier of 0.259–0.265 — so **tev1's
probability is better calibrated than the logistic arm's**, and the pooled
ECE still flatters it (the AI document's own ECE is 0.342–0.416, the same
effect CS-6e measured on the learned arm). laya's arms are the opposite:
0.1188–0.2423 pooled, and its `choice` arms are the worst of the eight.

This is what CS-7b needs: a **fixed 0.5 cut** is only meaningful on a
calibrated probability, and tev1's is the one to cut.

### 5. The input budget: laya's context is 512 tokens, and that costs it cells

The server refuses a prompt longer than the model's context. The measured
limits (2026-10-26) are **512 tokens for laya** and **~2,500 for tev1**; the
corpus's longest paragraph (2,255 characters) and longest question (1,560)
do not fit laya's budget. The call planner therefore groups the cells into
calls that fit the budget and **truncates the state's paragraph** (never the
question, never the instructions) when a cell does not fit: **441–448 of the
5,520 cells (8.0–8.1%)** were truncated for laya, **0** for tev1. Every
truncated cell is flagged in the artifact (`truncation` and the per-cell
`paragraph_truncated`).

The truncation is not random: it hits the **longest** paragraphs, so laya's
arm is measured on a systematically shorter view of the corpus's most
content-rich paragraphs. That is part of why laya's arms are the weaker ones,
and it is not separable from the model difference in this change set.

### What this run does not say.

- **Nothing about the unjudged cells' positive rate.** The probe measures the
  reviewed set; the safe-negative rule's precision is still unmeasured (no
  human labelling in this track).
- **Nothing comparable about the rank-based numbers.** The probe's candidate
  list is the **scored** cells of a paragraph (the reviewed pairs plus its
  control cells), not the paragraph's full same-document question list, so
  the rank percentiles, the hit@k and the top-1 provenance are **not**
  comparable with the embedding arms' (which rank against the full list).
  The AUCs are comparable (the same 1,520 pairs, the same labels). The grid
  scope (CS-7b) is what makes the rank numbers comparable.
- **Nothing about the grid.** The probe scores the reviewed set and the
  control sample, not the 44,467-cell grid.
- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice, they are never a UPR metric.
- **Nothing about a second seed or a second prompt wording.** The prompt
  grid is one task instruction, one `noul` question and one two-option
  `choice`; the models are also **not deterministic** (the same call returns
  a slightly different probability on a re-run — the within-document AUC
  moves in the third decimal), which is why the **cache** is what makes a
  re-run exactly reproducible.

**The next step.** CS-7b (Stage 3.5.2): the decision arm on the **whole
same-document grid**, with the safe-negative rule re-measured with tev1's
probability as the ranking signal and a **fixed** cut (0.5) beside the
fit-side-threshold comparison, and the control-role floor's counterpart
beside both. The probe has settled which arm to carry: **tev1 with the
two-option `choice`** (the best AUC, the best calibration, and the
`paragraph` shape is 16× cheaper for 0.006 of AUC).

## Stage 3.5.2 — the decision arm on the grid + the rule's decider (CS-7b, 2026-10-26)

**The question.** The plan's Stage 3.5.2 turns the deliverable's rule around:
the safe-negative rule gets a **label-driven** scorer behind it. The decision
model scores the **whole same-document grid** (44,467 cells) and becomes an
*arm* of `analyze-omct` — the same cells, the same candidate lists, the same
control, the paired comparison against the full-text cosine — and the rule is
re-measured with the decision model's probability as the ranking signal.

**What was run.** Three runs
(`experiments_text_processing/stage35_grid_sweep.py`, the sweep
`stage35_grid`): the three Stage-0 models as the cosine's reference, each
with the **decision arm** (`tev1`, the `paragraph` shape, the two-option
`choice` — the CS-7a winner) scoring the whole grid, plus its
**label-blind counterpart** (the same model, the same prompts, but every
cell's question replaced by another question of the same document) —
`omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage35-grid`. The
decision arm's numbers are the same in the three runs (it does not depend on
an embedder); the *reference* differs, so the paired comparison against the
cosine is per model.

**1. The decision arm beats every other scorer on the reviewed set.** The
within-document AUC is **0.7701** (the pooled 0.7688, the per-document macro
0.8014) — against the full-text cosine's 0.5020–0.5632 and the track's own
learned arm's 0.6472–0.6723. The paired difference against the cosine is
**+0.2069 / +0.2463 / +0.2681** (embeddinggemma / voyage-4-nano / arctic2),
all with 95% intervals excluding zero, and the **de-shifted** AUC (the
control's per-document mean shift removed) stays at 0.7592–0.7639: the gain
survives the scale shift, so it is a ranking gain, not a re-scaling. The
floor gap (the verified positives' mean minus the ordinary cells' mean) is
**0.3530** against the cosine's 0.1276–0.1442.

The *retrieval* side is a different story: the positives' hit@1 is **0.1317**
against the cosine's 0.1050–0.1355 — the decision model ranks a *pair* much
better but picks a paragraph's first candidate about the same. (The rank
percentiles do improve: the positives' median rank 0.0507 against the
cosine's 0.078–0.097.) The cost: **1,805 calls** for the grid in the
`paragraph` shape — the `pair` shape would have needed 44,467.

**2. The rule with the decision model: the same cost, far more reach.** The
rule's numbers over the **whole 42,947-cell unjudged grid**, with the harm on
the **524 verified positives** and the label-blind counterpart beside it:

| rule (the decision model's probability) | the reach | the harm | the label-blind counterpart's harm |
|---|---|---|---|
| the **fit-side threshold** | **0.8313** | **0.1851** | 0.4147 |
| the **margin rule** (the cell below its paragraph's best) | 0.7952 | 0.3015 | 0.5336 |
| the **fixed 0.5 cut** | 0.9789 | 0.6050 | 0.8035 |

Beside the **learned arm's** rule, measured in the same three runs (CS-6e's
numbers):

| rule (the learned arm's probability) | the reach | the harm | the control-role floor's harm at the same reach |
|---|---|---|---|
| the **two-part rule** (the cosine's margin AND the arm's margin) | 0.4999–0.5950 | 0.1355–0.1927 | 0.1286–0.1638 |
| the arm's margin | 0.5895–0.6539 | 0.2824–0.4008 | 0.2489–0.2847 |
| the cosine's margin | 0.7763–0.8286 | 0.3130–0.3378 | — (the geometry has no floor) |

**The best form of the rule now reaches 83% of the unjudged space at the harm
the old best form paid** (0.1851 against 0.1355–0.1927) — where the old best
form reached 50–60%. That is 1.4–1.7× the reach at the same cost, and it is
the first version of the rule whose reach is *not* bought with the harm.

**3. The harm is label-driven — the certification CS-6e could not give.** The
label-blind counterpart (the same model, the wrong pairings) discards
**41.5%** of the verified positives at the same reach where the real arm
discards **18.5%** (and 80.4% against 60.5% at the fixed cut). CS-6e could
not certify the harm because the *control-role floor* — a model that knows
nothing about the labels — reproduced it; here the label-blind scorer is
**2.2× worse** on the same cells at the same reach. So the harm on the
verified positives is a **label-driven number**, not a measurement of the
reviewed set's construction ✓✓ — the deliverable's cost claim is finally a
*quality* claim and not only a reach claim.

**4. The fixed 0.5 cut is NOT the good operating point — a negative result
about the plan's premise.** The plan hoped that a calibrated model would let
the rule take a **fixed** cut. The measurement says otherwise: at 0.5 the
rule clears 97.9% of the unjudged grid but discards **60.5%** of the verified
positives. The model *is* calibrated (CS-7a: the ECE 0.089–0.137 pooled),
but its probabilities are compressed toward the low end: it answers "no" for
most pairs, including 60% of the **true** ones (the positives' mean
probability is 0.41). Calibration fixes the *meaning* of the probability; it
does not remove the need for a **fitted** operating point. The fit-side
threshold (the harm 0.1851 at the reach 0.8313) is the form that ships, and
its choice transfers (it is chosen on two documents and applied to the
third ✓).

**What this run does not say.**

- **Nothing about the unjudged cells' positive rate.** The rule's *reach*
  and its *harm* on the verified positives are measured; its *precision* on
  the unjudged cells is not (no human labelling in this track).
- **Nothing about a *second* decision model on the grid.** The arm is `tev1`
  only (the CS-7a winner); `laya`'s best arm (0.5992) is below the track's
  learned arm, so it was not carried to the grid.
- **Nothing about the pair shape on the grid.** The `pair` shape is the
  better of the two on the reviewed set (0.7767 vs. 0.7708) but it needs
  44,467 calls instead of 1,805; the grid scope is where that difference
  decides.
- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice, they are never a UPR metric.
- **Nothing about the models' non-determinism** (CS-7a): the same call
  returns a slightly different probability on a re-run, so the *cache* is
  what makes a re-run exactly reproducible (the re-run of this sweep
  reproduced every number).

**The next step.** CS-7c (Stage 3.5.3): the shortlist rerank — the cosine's
top-K candidates per paragraph re-decided by the decision model, with the
positives' hit@1 / hit@K against the cosine's and the learned arm's. That is
PLAN's Phase-6 slot, and the decision arm's numbers here (the hit@1 0.1317,
the rank percentiles) are the reference it has to beat.

## Stage 3.5.3 — the shortlist rerank (CS-7c, 2026-10-26)

**The question.** The plan's Stage 3.5.3 is the track's **Phase-6 slot**: the
cosine's **top-K candidates** per paragraph are re-decided by the decision
model, and the metric is the reviewed set's positives' **hit@1 / hit@K** —
the number PLAN's cross-encoder reranker needs. The decision models are
frozen and zero-shot, so nothing leaks (the leakage rule (i) is about
*fitted* question-level priors; a frozen model's question text carries no
OMCT label).

**What was run.** Three runs
(`experiments_text_processing/stage35_rerank_sweep.py`, the sweep
`stage35_rerank`): the three Stage-0 models as the cosine's reference, each
with the decision arm of CS-7b —
`omct-analyze-{embeddinggemma,voyage-4-nano,arctic2}-stage35-rerank`. The
rerank's numbers come from the grid the decision arm **already scored**, so
the sweep cost **no new model calls** (38 s, all cache hits).

**1. The rerank beats both of its references.** The positives' hit@1 (the
share of the true matches that end up **first** among their paragraph's
candidates):

| the rerank's shortlist | arctic2 | voyage-4-nano | embeddinggemma |
|---|---|---|---|
| the cosine's own hit@1 (the geometry alone) | 0.1050 | 0.1050 | 0.1355 |
| the decision model's own hit@1 (rerank *everything*) | 0.1317 | 0.1317 | 0.1317 |
| the **rerank** at K=5 | 0.1431 | 0.1240 | 0.1393 |
| the **rerank** at K=10 | **0.1565** | 0.1374 | 0.1393 |
| the **rerank** at K=20 | 0.1469 | **0.1431** | 0.1355 |
| the **rerank** at K=50 | 0.1336 | 0.1412 | 0.1336 |
| the track's learned arm's hit@1 (CS-6a) | 0.0859 | 0.0821 | 0.0897 |

The hybrid is **above the decision model's own full-grid ranking** on all
three embedders (0.1565 vs. 0.1317 on arctic2) and above the cosine's own
on all three — the classic retrieve-and-rerank result, and the plan's
CS-6d shape ("retrieve with the cosine, rank with the learned arm") with a
*decision model* as the ranker. The per-document macro beside it: 0.1792
(arctic2, K=10), 0.1578 (voyage-4-nano, K=20), 0.1658 (embeddinggemma,
K=5–10).

The K that wins differs by the cosine (10 for arctic2, 20 for
voyage-4-nano, 5–10 for embeddinggemma) — the shortlist is the cosine's own,
so the rerank inherits its retrieval quality.

**2. The cost is a rounding error against the ceiling.** The shortlist's own
recall (the positives' hit@K — the ceiling a rerank cannot beat, since a
rerank reorders the shortlist but does not enlarge it) is 0.3073 at K=5,
0.4256 at K=10, 0.5229 at K=20 and 0.7042 at K=50 (arctic2). The rerank
recovers a *third* of that ceiling at K=10 (0.1565 of 0.4256). The price is
the shortlist's cells: 225 candidates at K=1 and 2,250 at K=10 — against the
44,467 cells the full grid costs (the decision calls: ~90 against 1,805, i.e.
**20× cheaper** than scoring everything).

**3. The rerank lowers the hard negatives' rise, too.** The reviewer-verified
hard negatives' hit@1 *falls* with the rerank (arctic2: 0.0171 at K=1 →
0.0100 at K=10) and the ordinary control cells' hit@1 falls with it
(0.0045 → 0.0022). So the rerank is not trading the positives against the
negatives: it lifts the true matches *and* pushes the reviewed set's hard
negatives down — the opposite of what the embedding arms do (CS-6d: the
learned arm ranks pairs better but retrieves worse).

**What this run does not say.**

- **Nothing about the precision** of the reranked shortlist: the unjudged
  cells' true positive rate stays unmeasured (no human labelling in this
  track), so the rerank's numbers are about the *reviewed* set's positives.
- **Nothing about a K chosen on data**: the winning K is read off the same
  reviewed set the metric is measured on, so it is an upper bound on what a
  fit-side K would deliver (the K grid would have to be chosen on the fit
  side, as the rule's parameters are).
- **Nothing about a second decision model** (the arm is `tev1` only) and
  **nothing about the `pair` shape** (the grid scope's cost decided that:
  1,805 calls against 44,467).
- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice, they are never a UPR metric.

**The next step.** The plan's Stage 3.5 is measured end to end: the probe
(CS-7a), the arm and the rule (CS-7b), the shortlist rerank (CS-7c). The
track's remaining rungs are the **Stage-2 ones** — the chunk header (CS-7d)
and the late chunking (CS-7e) — and they now have a much better yardstick
than the embedding arms: the decision model's arm (the within-document AUC
0.7701) and the shortlist rerank (the positives' hit@1 0.1565) are the
numbers a new segmentation rung has to beat, not the cosine's 0.5020-0.5632
and 0.1050-0.1355.

## Stage 2.6 — the chunk header (CS-7d, 2026-10-26)

**The question.** The plan's Stage 2.6 is the classic "contextual chunk
header" idea in the one form that is legal on this data: the header is the
paragraph's **own first sentence** — pair-local, label-free, no document
identity, and it keeps one embedding per chunked text (a *question-stem*
header would need one embedding per (chunk, question) pair: ~175,000 instead
of ~500 per model). The other headings are ruled out by the track's leakage
rules: the document title is document identity (rule ii) and the question
number is a corpus statistic (rule i).

**What was run.** Six runs
(`experiments_text_processing/stage26_header_sweep.py`, the sweep
`stage26_header`): 3 Stage-0 models x 2 header modes, with the Stage-2
verdict's paragraph-side window (25 words, stride 12) and two aggregation
rules (`max`, `peak:3:0.05`) — each run carrying the **no-header arm of the
same run** beside the header arm, so the header is measured against the
no-header arm **paired by construction** (the same cells, the same candidate
lists). The comparison is the repo tool
`tools/compare_arms_in_run.py` (the paired bootstrap of the AUC difference
between two arms of one run, read from the artifact).

The two modes are both measured (the reviewer's rule: measure the
alternatives rather than picking one): `others` prepends the header to every
chunk **except the first** (the first chunk already starts with the header's
text, so prefixing it would only duplicate it) and `all` prepends it to every
chunk (the literal reading; the first chunk then carries its first sentence
twice).

**1. The ranking side: the header is not neutral, it is negative.** The
within-document AUC, the header arm against the no-header arm of the same run
(the paired bootstrap, 2,000 resamples, seed 42):

| model | mode | the no-header arm | the header arm | the paired difference (95% CI) |
|---|---|---|---|---|
| embeddinggemma | others | 0.6049 | 0.5734 | **-0.0315** [-0.0431, -0.0204] |
| embeddinggemma | all | 0.6049 | 0.5749 | **-0.0300** [-0.0416, -0.0191] |
| voyage-4-nano | others | 0.5084 | 0.5130 | +0.0045 [-0.0059, +0.0149] |
| voyage-4-nano | all | 0.5084 | 0.5148 | +0.0064 [-0.0046, +0.0170] |
| arctic2 | others | 0.5311 | 0.5225 | -0.0087 [-0.0199, +0.0025] |
| arctic2 | all | 0.5311 | 0.5242 | -0.0069 [-0.0187, +0.0046] |

The pre-registered hypothesis said the ranking gain would be ≈ 0 (the chunk
geometry carries no label-driven ranking signal, CS-6d's random-vector
floor). It is ≈ 0 on two models and **negative** on the third: on
embeddinggemma the header's own sentence *dilutes* the chunk's content, and
the loss is significant (both intervals exclude zero). The `peak` arm tells
the same story (embeddinggemma -0.0313 / -0.0300).

**2. The retrieval side: the payoff does not materialize.** The positives'
**hit@1** (the pre-registered hypothesis's target):

| model | mode | the no-header arm | the header arm | the paired counts (the header only / the no-header only) |
|---|---|---|---|---|
| embeddinggemma | others | 0.1374 | 0.1336 | 14 / 16 |
| embeddinggemma | all | 0.1374 | 0.1336 | 16 / 18 |
| voyage-4-nano | others | 0.0916 | 0.0954 | 12 / 10 |
| voyage-4-nano | all | 0.0916 | **0.1011** | 13 / 8 |
| arctic2 | others | 0.1011 | 0.1031 | 7 / 6 |
| arctic2 | all | 0.1011 | 0.1031 | 9 / 8 |

The best arm (voyage-4-nano, the `all` mode) gains **+0.0095** of hit@1 —
real but marginal, and *below* the arm it started from on the other models
(the embedding arms' own numbers: 0.0916-0.1374). For scale: the decision
model's own hit@1 is **0.1317** and the shortlist rerank's **0.1565**
(CS-7b/CS-7c). So the header's retrieval gain does not reach the systems
that do not need a segmentation change at all.

**3. The two modes are the same.** `others` and `all` differ by at most
0.0015 of within-document AUC and by one hit@1 count — so the header's
*content* is what matters (or rather: neither the context nor the
duplication changes the ranking). The default (`others`) is kept: it does
not duplicate the header's text in the first chunk.

**The verdict: the chunk header does not ship.** It is not neutral (it is
significantly negative on one of the three models) and its retrieval payoff
(+0.0095 at best) is an order of magnitude below what the decision models
deliver. With CS-7a–CS-7c in hand, the track's yardstick for a segmentation
change is no longer the cosine's 0.5020–0.5632 within-document AUC and its
0.1050–0.1355 hit@1, but the **decision arm's 0.7701** and the **rerank's
0.1565** — and the header reaches neither.

**What this run does not say.**

- **Nothing about a question-stem header.** The plan rules it out on the
  leakage rules (rule i) *and* on the cost (~175,000 embeddings per model);
  it was not measured.
- **Nothing about the header's effect on the decision models.** The decision
  models read the raw text, so a chunking change cannot touch them (the
  header is an embedding-side lever by construction).
- **Nothing about other segmentations.** The header is one rung: the late
  chunking (Stage 2.7) is the other, and it is a *geometry* change, not a
  text change.
- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice, they are never a UPR metric.

**The next step.** CS-7e (Stage 2.7): the **late chunking** — the document
embedded once and the token-level embeddings pooled per chunk, so every chunk
carries the document's context. It is HF-only (Ollama's `/api/embed`
cannot return token-level embeddings), so the comparison is **within-model**
(late-chunked voyage-4-nano against early-chunked voyage-4-nano, the same
cells, the same seeds).

## Stage 2.7 — the late chunking (CS-7e, 2026-10-26)

**The question.** The plan's Stage 2.7 is the last rung of the segmentation
ladder: instead of embedding each chunk separately, embed the **whole
document once** and pool its **token-level** embeddings per chunk, so every
chunk carries the document's context.

Two costs were stated up front in the plan, and both bound the arm:

- it needs **token-level** embeddings, which the `Embedder` port did not
  expose and which Ollama's `/api/embed` **cannot** return — so the arm is
  **HF-only** (the Stage-0 models are `embeddinggemma` (Ollama),
  `voyage-4-nano` (HF) and `arctic2` (Ollama), so only `voyage-4-nano`
  qualifies) and the comparison is **within-model**;
- it needs the chunker's **character spans** (the `Chunker` port returned
  strings), so the tokens could be pooled per chunk — plus a graceful
  "not available for this backend" path.

**What was run.** Two runs
(`experiments_text_processing/stage27_late_sweep.py`, the sweep
`stage27_late`), both on `voyage-4-nano`:

- `omct-analyze-voyage-4-nano-late-words25` — the paragraph side
  late-chunked, the question side whole (as the early-chunked reference
  does);
- `omct-analyze-voyage-4-nano-late-symwords25` — the symmetric reading: both
  sides late-chunked.

Each run carries the **early-chunked arm of the same run** beside the late
one (the Stage-2.6 precedent), so the late-vs-early comparison is **paired
by construction**: the same cells, the same candidate lists, the same
control, the same seeds. The published early reference
(`omct-analyze-voyage-4-nano-words25`, the Stage-2 ladder run) is the
cross-run check of that arm, and it reproduces it **exactly** (the paired
difference is 0.0000 on the pooled AUC, the within-document AUC and the
positives' hit@1 — the repo tool `tools/compare_arms_in_run.py`, extended
with an optional `--run-b` for the cross-run comparison).

The geometry is the **same** on both sides of the comparison (25-word
windows, stride 12 — the Stage-2 verdict's paragraph-side window), so the
only difference between the two arms is *where the chunk's vector comes
from*: its own embedding (early) or the document's token-level embeddings
pooled over the chunk's character span (late). The two aggregation rules
are the plan's early-chunked reference: `max` and `peak:3:0.05`.

**The pooling rules, both measured** (the reviewer's rule: measure the
alternatives rather than pick one): `mean` (the model's own formula,
`Normalize(mean(tokens))`, restricted to the chunk) and `norm` (every token
vector L2-normalized *before* the mean, so each token weighs the same).

**1. The ranking side: the late geometry wins, and the pre-registered
hypothesis is wrong.** The within-document AUC (the track's headline
statistic), the late arm against the **early arm of the same run** (the
paired bootstrap, 2,000 resamples, seed 42):

| run | arm | the early arm | the late arm | the paired difference (95% CI) |
|---|---|---|---|---|
| late-words25 | mean / max | 0.5084 | **0.5944** | **+0.0859** [+0.0650, +0.1074] |
| late-words25 | mean / peak | 0.5111 | **0.5940** | **+0.0829** [+0.0605, +0.1047] |
| late-words25 | norm / max | 0.5084 | 0.5683 | +0.0599 [+0.0421, +0.0793] |
| late-words25 | norm / peak | 0.5111 | 0.5623 | +0.0511 [+0.0321, +0.0702] |
| late-symwords25 | mean / max | 0.4725 | **0.5599** | **+0.0874** [+0.0662, +0.1080] |
| late-symwords25 | mean / peak | 0.4740 | **0.5618** | **+0.0878** [+0.0651, +0.1103] |
| late-symwords25 | norm / max | 0.4725 | 0.4999 | +0.0273 [+0.0053, +0.0492] |
| late-symwords25 | norm / peak | 0.4740 | 0.4907 | +0.0167 [-0.0061, +0.0387] |

The plan's pre-registered hypothesis (the same as Stage 2.6's) said the
chunk geometry carries no label-driven **ranking** signal, so the expected
within-document AUC gain was ≈ 0. It is **not** ≈ 0: it is +0.086 on the
paragraph side, with the interval excluding zero, on the model where the
whole early-chunking ladder gained nothing (CS-4: "voyage-4-nano gains
nothing — 3 of 42 arms positive"). The hypothesis is **falsified on the
ranking side**, and the symmetric reading is no better than the paragraph
side alone.

**2. The retrieval side: the payoff is there, and it is the best
embedding-side hit@1 of the track on this model.** The positives' hit@1:

| run | arm | the early arm | the late arm | the paired counts (the late arm only / the early arm only) |
|---|---|---|---|---|
| late-words25 | mean / max | 0.0916 | **0.1240** | 26 / 9 |
| late-words25 | mean / peak | 0.0897 | **0.1260** | 30 / 11 |
| late-words25 | norm / max | 0.0916 | 0.1240 | 24 / 7 |
| late-words25 | norm / peak | 0.0897 | 0.1221 | 25 / 8 |
| late-symwords25 | mean / max | 0.0954 | 0.1069 | 20 / 14 |
| late-symwords25 | mean / peak | 0.0954 | 0.0973 | 20 / 19 |

The full-text cosine's own hit@1 is 0.1050 on this model, so the late arm
is above the cosine (+0.0210) and above the early chunking (+0.0344). The
positives' median rank percentile falls from 0.1429 to 0.0783. For scale:
the decision model's own hit@1 is 0.1317 and the shortlist rerank's 0.1565 —
the late chunking reaches neither, but it is the first **embedding-side**
arm of the track that comes close to the decision model on the retrieval
side.

**3. The control side says the gain is a ranking gain, not a scale shift.**
The track's reporting rule: no discrimination number without the control
side next to it. The early chunking lifts the control's score level by
**+0.0413** (the max-over-k inflation the plan's Stage 2.5 named) and the
late chunking by **+0.0011** — the late arm's gain survives the
de-shifting (the de-shifted AUC 0.5642 against the early arm's 0.4786).
The ordinary control cells' rank position does not move (the rank-p50
delta -0.0065) and their hit@1 stays at 0.0037, while the **reviewed hard
negatives' hit@1 falls** (0.0241 → 0.0171) as the positives' rises — the
same signature the shortlist rerank had (CS-7c), and the opposite of what
the early chunking does.

**4. The full arms of the primary run** (the pooled AUC and the
per-document macro beside the within-document number; the control-side
counterpart beside both):

| arm | pooled AUC | macro AUC | within-doc AUC | de-shifted AUC | hit@1 (positives) | hit@1 (hard neg.) | hit@1 (control) | the control's mean shift |
|---|---|---|---|---|---|---|---|---|
| `full_text` (the baseline) | 0.5004 | 0.5588 | 0.5238 | — | 0.1050 | 0.0211 | 0.0030 | — |
| `words25_max` (early) | 0.4856 | 0.5329 | 0.5084 | 0.4786 | 0.0916 | 0.0241 | 0.0035 | +0.0413 |
| `words25_peak_3_0.05` (early) | 0.4867 | 0.5327 | 0.5111 | 0.4814 | 0.0897 | 0.0251 | 0.0035 | +0.0253 |
| `late_mean_words25_max` | **0.5678** | **0.6312** | **0.5944** | **0.5642** | 0.1240 | 0.0171 | 0.0037 | +0.0011 |
| `late_mean_words25_peak_3_0.05` | 0.5684 | 0.6271 | 0.5940 | 0.5674 | 0.1260 | 0.0141 | 0.0030 | -0.0148 |
| `late_norm_words25_max` | 0.5445 | 0.6050 | 0.5683 | 0.5428 | 0.1240 | 0.0211 | 0.0030 | +0.0136 |
| `late_norm_words25_peak_3_0.05` | 0.5406 | 0.6019 | 0.5623 | 0.5415 | 0.1221 | 0.0201 | 0.0032 | -0.0019 |

The per-document AUCs of the primary run's best arm (the late `mean`/`max`)
are 0.5755 (Colombia), 0.6943 (CIDH) and 0.6240 (NU), against the early
arm's 0.4799 / 0.5639 / 0.5551 — the gain is in **all three** documents,
so it is not one document's level moving.

**5. The two pooling rules: `mean` wins.** The model's own formula beats
the per-token-normalized variant by +0.0260 of within-document AUC
(+0.0859 against +0.0599) and, on the symmetric run, the `norm` variant's
gain is small and one of its two intervals includes zero. The reading: the
*relative* magnitude of a token carries information (a token that stands
out in the document should pull the chunk's vector), and flattening it
before the mean throws that away.

**6. The provenance the artifact carries.** The token-level path is checked
inside every run: the **token coverage** is 1.000 (every one of the 20,251
content tokens of the 225 paragraphs lands in at least one chunk — the
windows tile the text), and the **consistency check** compares the
document's own pooled vector with the model's own embedding of the same
text: the "chunks' view" (the content tokens only — no chunk can own the
prompt prefix's tokens) reaches a cosine of **0.9918**, and the "model's
formula" (the content tokens plus the prefix's own, which is what the model
itself averages) reaches **0.99998**. The token-level path therefore
reproduces the model's own embedding, and the gap of the "chunks' view" is
the prompt prefix's contribution, as expected.

**7. The cost.** The late side embeds **225 documents** (one forward pass
each) where the early side embeds **1,186 chunks** — 5.3x fewer encoder
calls on the paragraph side, at the price of longer inputs and a token-level
pass that is **not cached** (a late-chunked vector is a function of the
whole document, so caching it under the chunk's key would be wrong; the
plan's cache-key layout is unchanged). The whole run takes 12.4 s with a
warm cache.

**The verdict: the late chunking is the ladder's one winner — and it is
still not the deliverable's scorer.** It is the first segmentation rung
that improves the **ranking** side significantly (the pre-registered
hypothesis expected ≈ 0), it is the best embedding-side arm of the track on
this model (the within-document AUC 0.5944 against the cosine's 0.5238 and
the early chunking's 0.5084), and it is the best embedding-side retriever
(the positives' hit@1 0.1260 against the cosine's 0.1050 and the early
chunking's 0.0916) — with a gain that is *not* a scale shift and that
pushes the reviewed hard negatives down while the ordinary cells stay
put.

But it does not reach the systems that do not need a segmentation change
at all: the decision model's arm is at the within-document AUC **0.7701**
and the shortlist rerank at the positives' hit@1 **0.1565**, and the
decision model's own hit@1 (0.1317) is already above the late chunking's
0.1260. The track's own learned arm (0.6472-0.6723 within-document) is
also above it — but most of that arm's gain is label-driven (its
control-role floor is 0.557-0.579), while the late chunking is a
**geometry change with no fitting at all**. So the late chunking is the
best *representation-side* arm the track has, and the decision model is
still the deliverable's scorer.

**What this run does not say.**

- **Nothing about the other two models.** The arm is HF-only, and only
  `voyage-4-nano` is an HF model among the Stage-0 three. Whether the gain
  transfers to `embeddinggemma` and `arctic2` is **unmeasured** (it would
  need a backend that can return token-level embeddings).
- **Nothing about the reviewed set's construction.** The reviewed set is
  rank-selected (§0.1), so a geometry change can pick up the construction
  and not the labels. The control side is the mitigation the track has: the
  ordinary cells do not move, and the hard negatives fall while the
  positives rise — but the track cannot *prove* the gain is a topic signal.
- **Nothing about the late chunking on long documents.** The corpus's
  paragraphs average 71 words; the arm's whole point is the document's
  context, and a document short enough to be one chunk has no context to
  carry. The measurement says nothing about the UPR scale (151 chars per
  text — the plan's Stage 5 says the UPR-side chunking axis is near-vacuous
  by construction).
- **Nothing about a hybrid.** The late chunking's chunk vector and the
  early chunk's own embedding were not combined; the arm measures the
  pooling alone.
- **Nothing about UPR.** OMCT is auxiliary data; these numbers annotate
  method choice, they are never a UPR metric.

**The next step.** The plan's Stage 2 is now complete (2.1–2.7): the
ladder's ranked statement is in the track log's "Stage 2 — the verdict"
section, and the two open rungs are closed — the chunk header does not
ship, the late chunking does (as the best embedding-side arm, not as the
deliverable's scorer). What is left in the plan is the reviewer's call:
**Stage 4** (the LLM experiments: the exp09-style rerank and the
question/side enrichment) and **Stage 5** (the UPR port). One thing the
late chunking does say to both: the embedding axis's remaining job is
cheap retrieval at scale, and the late geometry is the best the embedding
side can do without a model that reads the text (the decision models).

## CS-8a — the UPR side: the command, the plumbing, the UPR-side background (2026-10-26)

**The run.** `upr-analyze-champion` — the text-processing track's UPR
artifact (`results/upr-analyze-champion/analysis.json`), the **whole-text
reference arm** of the standing champion: embeddinggemma, the fusion
(a=0.5 knn / b=1.0 prototype / c=1.0 zero-shot, knn k=10) and the `top_k k=2`
decision rule, on the **val** split. The arm is the main pipeline's own
scoring — the same predictor, the same two embedding runs, the same decision
rule — so the artifact's numbers *are* the pipeline's numbers.

**The guard (the UPR analogue of `tools/verify_analysis_baseline.py`).** The
reference arm reproduces the matching `evaluate` run
(`upr-eval-fusionc-embeddinggemma-k10-c1`) **field by field** — the overall
report, the per-bucket reports, the record count and the label count. The
check runs **inside** the command (a mismatch is a hard error, never a
warning) and is re-checkable from the artifacts with the new repo tool
`tools/verify_upr_baseline.py` (`uv run python tools/verify_upr_baseline.py
"upr-analyze-*"` → OK).

**What is new on the UPR side: the background needs no sample.** On OMCT the
track had to *draw* a control (the unjudged same-document cells, with a seed
and a per-document quota). UPR's grid is small and fully enumerable, so the
**non-gold cells are the background**: 43,568 cells against the split's 1,312
gold cells, complete, with **no sampling, no seed and no quota** (the artifact
records `sampling: null` and `seed: null` and says why). Every UPR number
below is therefore exact — there is no sampling error to argue about.

**The headline numbers (val; the reference arm).**

| view | number |
|---|---|
| decision (top_k k=2) | micro-F1 **0.5479**, macro-F1 0.5006, sample-F1 0.5269, coverage 0.868 |
| ranking | P@5 0.2770, R@10 0.7126, nDCG@5 0.6725, shortlist R@50 0.7423 |
| gold side | rank p50 **0.0147**, top-fifth share 0.7599, hit@1 0.3666, hit@10 0.7462 |
| background side | rank p50 **0.5000**, top-fifth share 0.1892, mean 0.5526, p50 0.5396 |
| provenance | the top-1 label is gold for **481/660** texts (0.7288); the mean gold labels in the top-5 is 1.385 and in the top-10 is 1.483 (of 1.988 gold labels per text) |

The **background's rank p50 is exactly 0.5000** — the expected value for a
random cell in a 68-label list, and a useful check of the rank machinery
itself. The gold side is not near chance: the median gold label is the
**best** of its text's 68 (rank p50 0.0147) and three quarters of the gold
cells sit in the top fifth of their list.

**Per label bucket** (the frequency buckets; the 8 label-holdout labels are
the `unseen` ones):

| bucket | labels | micro-F1 | macro-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 |
|---|---|---|---|---|---|---|---|---|
| head | 16 | 0.6953 | 0.6701 | 1.000 | 0.2849 | 1.0000 | 0.9558 | 1.0000 |
| torso | 26 | 0.5121 | 0.5383 | 1.000 | 0.2329 | 0.9859 | 0.8857 | 1.0000 |
| tail | 18 | 0.5287 | 0.5181 | 0.944 | 0.2064 | 1.0000 | 0.9153 | 1.0000 |
| unseen | 8 | **0.0000** | 0.0000 | **0.000** | 0.1531 | 1.0000 | 0.6575 | **0.034** |

**The unseen labels are not discoverable by the champion's geometry, and the
per-label distributions say why.** The `unseen` labels are never suggested
(coverage 0.000) and their gold cells sit outside the top-50 shortlist
(R@50 0.034). The artifact's per-label score distributions make the mechanism
explicit: for the head label "Detention" the gold cells average **2.242**
against the background's 0.626 (a well-separated label), while for the unseen
"General" the gold cells average **0.0946** against the background's **0.0881**
— the two distributions are the same. The zero-shot term (c=1.0 on the
`about` label template) carries the unseen labels at chance, not at a
disadvantage; there is simply no signal in this geometry for them.

**Per text-length bucket** (the axis the chunking hypothesis is about; the
val split's texts average 153 characters):

| bucket | texts | micro-F1 | macro-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 |
|---|---|---|---|---|---|---|---|---|
| 0-99 | 155 | 0.5307 | 0.4481 | 0.788 | 0.2245 | 0.7005 | 0.6678 | 0.7167 |
| 100-199 | 366 | 0.5620 | 0.5142 | 0.868 | 0.2770 | 0.7115 | 0.6742 | 0.7429 |
| 200-299 | 114 | 0.5422 | 0.4650 | 0.781 | 0.3404 | 0.7398 | 0.6869 | 0.7778 |
| 300+ | 25 | **0.4727** | 0.4498 | 0.643 | 0.3120 | 0.6800 | **0.6104** | 0.7300 |

**The long-text tail is where the champion is weakest** (micro-F1 0.4727 and
nDCG@5 0.6104 against the corpus's 0.5479 / 0.6725) — the starting point of
the CS-8b hypothesis ("a possible small win on the retrieval side for the
long-text tail"). Note the tail is small (25 texts), so its numbers carry a
wide error bar: the pooled number would hide it, which is exactly why the
length buckets are reported separately from now on.

**What this run does not say.**

- **Nothing about any chunking.** This is the whole-text reference arm; the
  ladder, the chunk header, the late chunking and the symmetric variants are
  CS-8b / CS-8c.
- **Nothing about the feature pack, the combiner or the floors** (CS-8d /
  CS-8e) and nothing about the decision models on UPR (CS-9).
- **Nothing about `test`.** Only the val split was touched; `test` stays
  reserved for the final adopted config.
- **Nothing about the calibration.** The artifact reports the discrimination
  and the background side, not the calibration block (CS-8f).
- **No OMCT claim.** Everything above is a UPR number on the val split; the
  OMCT side's findings (the late chunking's win, the lexical block's gain)
  are method-choice evidence only and are re-measured here, not transferred.

**The next step.** CS-8b: the segmentation ladder on UPR, measured against
this arm in the same run (the pairing is by construction), with the
length buckets reported separately and the background's level and rank shift
beside every headline number.

## CS-8b — the segmentation ladder on UPR (2026-10-26)

**The run.** `upr-analyze-ladder` — the whole ladder on the UPR val split
(`results/upr-analyze-ladder/analysis.json`): the whole-text reference arm
(the standing champion: embeddinggemma, the fusion a=0.5 knn / b=1.0
prototype / c=1.0 zero-shot with knn k=10, `top_k k=2`) beside **153 arms**
in **13 families**: the **pair-scoring** family (the text's chunks against the
zero-shot label vectors), the **retrieval** family (every chunk is a query
against the train-only index) and the **symmetric** family (the label side
chunked too), over six chunkers (the atoms, the word windows N = 15 / 25 / 40,
the sentence windows, the semantic boundaries) and eight folding rules (`max`,
`top_k_mean:2`, `top_k_mean:3`, `mean`, `decayed_top_k:3:0.5:0.25`,
`local_global:0.5|max`, `local_global:0.75|max`, `peak:3:0.05`), plus the
fusion weights re-fitted on val (the `_refit` arms). The guard holds: the
reference arm reproduces `upr-eval-fusionc-embeddinggemma-k10-c1` field by
field.

**The statistic.** The ladder's comparisons are **paired**: a paired bootstrap
over the val **texts** (2,000 resamples, seed 42) of the micro-F1 difference
(at the run's decision rule) and of the nDCG@5 difference between an arm and
the whole-text reference arm, pooled **and per text-length bucket**. This is
the function CS-8f planned ("one new function"); it was pulled forward into
CS-8b because the ladder's verdict needs an interval, and it lives in
`eval/multilabel.py` (`paired_metric_bootstrap`, next to the multi-label
metrics it bootstraps).

### The verdict: the pooled null is *almost* right — the effect is in the long texts

**The pre-registered expectation was "a null on the ranking side, a possible
small win on the retrieval side for the long-text tail".** The first half
holds (pooled, the ladder is a near-null: the best un-refitted arm is
+0.0137 of micro-F1) and the second half is **confirmed**:

| arm (retrieval family) | pooled Δ micro-F1 | 95% interval | 0-99 | 100-199 | 200-299 | 300+ |
|---|---|---|---|---|---|---|
| reference (whole text) | — | — | 0.5307 | 0.5620 | 0.5422 | 0.4727 |
| `retr_atoms_max` | +0.0106 | [-0.0023, +0.0227] | +0.0000 | -0.0027 | **+0.0393** [+0.0039, +0.0757] | **+0.1091** [+0.0194, +0.2069] |
| `retr_atoms_top_k_mean_2` | +0.0122 | [+0.0023, +0.0235] | +0.0000 | +0.0069 | +0.0314 | +0.0545 [+0.0000, +0.1132] |
| `retr_words15_max` | +0.0008 | [-0.0123, +0.0145] | +0.0000 | -0.0192 | +0.0314 | **+0.1273** [+0.0377, +0.2264] |
| `retr_words25_max` | +0.0076 | [-0.0008, +0.0168] | +0.0000 | -0.0041 | +0.0236 | **+0.1273** [+0.0367, +0.2281] |
| `retr_words40_max` | +0.0030 | [+0.0008, +0.0061] | +0.0000 | +0.0000 | +0.0039 | +0.0545 [+0.0000, +0.1189] |
| `retr_sent2_max` | +0.0000 | [0.0000, 0.0000] | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `retr_semantic_max` | +0.0000 | [0.0000, 0.0000] | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

(The per-bucket cells are the paired Δ micro-F1 with their 95% intervals; the
bucket's own absolute micro-F1 is in the first row.)

**The effect is exactly where a text is actually split.** The two chunkers
that do not split UPR's texts (the sentence windows and the semantic
boundaries: 1.00 chunks per text) give **exactly zero** in every bucket and
every rule — the ladder's null, measured. The `words40` chunker splits only
the longest texts (1.07 chunks per text) and gives exactly zero in the three
short buckets and +0.0545 in the 300+ bucket. Every gain is a long-text gain.

**The chunker matters more than the rule, and the sentence boundaries beat
the word windows.** Pooled, the best un-refitted arm is the atoms chunker
(+0.0106 to +0.0137, the interval excluding zero in six of its eight rules),
while the word windows at the same or a higher chunk count are a wash
(`words15`: +0.0008 pooled). In the long-text tail the two are equal
(+0.109 vs +0.127). This is the **opposite** of the OMCT verdict, where the
word windows won and the atoms were only the Stage-1 arm: on UPR the
boundary *quality* (a sentence boundary) matters more than the boundary
*density*.

**The scale shift is there, and the ranking gain survives it.** The `max` rule
takes a maximum over more candidates, so the background's score level rises
(+0.0505 for `retr_atoms_max`) while the background's **rank** position does
not move at all (Δ rank p50 = 0.0000) and the gold cells' rank p50 does not
move either (0.0147 in both arms). The gain is a ranking gain, not a
re-scaling — the track's reporting rule, satisfied.

### The pair-scoring family: much worse at the decision, much better at the shortlist

The **pair-scoring** family (the zero-shot label vectors only) is
significantly **worse** at the decision view: micro-F1 0.4134-0.4514 against
the champion's 0.5479, every paired interval excluding zero (-0.0965 to
-0.1345). But its **shortlist recall is much better**: the gold cells' R@50
is **0.8658-0.8701** against the champion's **0.7423**.

That is a finding about the champion itself, not about the chunking: the
champion's kNN and prototype terms **push 12.7 points of gold cells out of
the top-50 shortlist** that the pure zero-shot geometry keeps. The decision
view (the top-2) and the shortlist view (the top-50) are pulling in opposite
directions, and PLAN's Phase 6 reranker slot lives on the shortlist side —
so the zero-shot geometry is the better *candidate generator* even though it
is the worse *classifier*.

### The symmetric family: degenerate, exactly as predicted

The symmetric arms (the label side chunked too) are **identical** to the
one-sided pair-scoring arms (e.g. `sym_words15_max` and `pair_words15_max`:
micro-F1 0.4483, R@50 0.8691, and the background's mean shift agrees to
1e-10). The artifact records the degeneracy: **0 of the 68 labels** produce
more than one chunk (the labels are 1-3 words), so the label side is the
identity and the symmetric family is the ladder's null-check — as the
reviewer's "the labels are only 1-3 words" said it would be. Chunking the
label does nothing here, and the OMCT verdict ("never chunk the question")
transfers.

### The semantic threshold: selected on the fit side, and the grid is flat

The semantic chunker's threshold is chosen on the **fit side** (the train
split, 1,676 records) with the fit side's own label-scoring metric (the
micro-F1 at the run's decision rule, in the **pair-scoring** geometry — the
retrieval geometry's fit-side score would contain the text's own train
example, a self-match leak). The grid's criterion is flat: 0.4609 / 0.4609 /
0.4609 / 0.4606 / 0.4606 / 0.4606 for the thresholds 0.2 … 0.7, and the
chunk counts barely move (1,685-1,704 chunks for 1,676 records). All three
cross-fit folds choose the same threshold (0.2) and the held-out criterion
matches the reference's to the fourth decimal. The semantic boundaries find
**no** boundaries in UPR's one-topic recommendations — the expected null, now
measured.

### The weight re-fit: a val-selected number, and it moves

The fusion weights re-fitted on val for the chunked components (a = 0 … 2, c
= 0 … 1, b = 1; 20 combinations per rule) add **+0.0030 to +0.0167** of
micro-F1 on top of the un-refitted arms, and they move the background's score
level a lot (-0.17 for the families whose chunker does not split). These
numbers are **selected on val**, so they are optimistically biased; the
un-refitted arm is the comparable one. The blend is checked against the
family's own predictor before the grid is explored, so the refit cannot
silently use a different scoring path.

**What this run does not say.**

- **Nothing about the feature pack, the combiner or the floors** (CS-8d /
  CS-8e): no learned scorer is involved here.
- **Nothing about the chunk header, the late chunking or the label-side
  transform** (CS-8c).
- **Nothing about the calibration** (CS-8f) and nothing about the decision
  models on UPR (CS-9).
- **Nothing about `test`** — only val was touched.
- **The long-text tail is 25 texts.** Its intervals are wide (the
  `words15` arm's +0.1273 has a ±0.09 half-width) and the effect is measured
  on one split. It is a *consistent* pattern across five chunkers and four
  rules (the gain is in the long bucket in every arm that splits those
  texts), not a single noisy number — but it is 25 texts.
- **The `local_global` arms are not a pure chunking effect**: they blend the
  chunk-level score with the *whole-text* score, so they are a different
  method, not a segmentation of the same one.

**The next step.** CS-8c: the text-side extras (the chunk header, the late
chunking — HF-only, so the within-model comparison — and the label-side
transform), each measured against this ladder's arms.

## CS-8c — the text-side extras and the label-side transform (2026-10-26)

Two runs, both on the UPR val split, both against the same discipline as
CS-8b (the **paired** bootstrap over the val texts, pooled and per length
bucket, beside the background's level and rank shift):

- `upr-analyze-extras` — the **chunk header** (both modes) and the **label-side
  transform** (the bare label name vs the config template), on the champion's
  geometry (embeddinggemma). The guard holds: the reference arm reproduces
  `upr-eval-fusionc-embeddinggemma-k10-c1` field by field.
- `upr-analyze-late` — the **late chunking** (the text embedded once, its
  token-level embeddings pooled per chunk), on `voyage-4-nano` — the arm is
  **HF-only** (Ollama's `/api/embed` cannot return token-level embeddings), so
  the comparison is **within-model**: the late-chunked arm against the
  early-chunked arm of the same run, with the voyage champion as the
  reference. The guard holds against
  `upr-eval-fusionc-voyage-4-nano-k10-c1` (the voyage reference: micro-F1
  0.5395, the shortlist R@50 0.8193).

### The chunk header: it does not ship on UPR either

The header is the text's own first sentence, prepended to every chunk (the
inner chunker: the atoms, the CS-8b ladder's best chunker on UPR).

| arm | Δ micro-F1 (paired) | 95% interval |
|---|---|---|
| `header_atoms_others_max` | -0.0030 | [-0.0114, +0.0053] |
| `header_atoms_others_mean` | **-0.0091** | [-0.0168, -0.0015] |
| `header_atoms_others_local_global:0.75:max` | +0.0015 | [-0.0053, +0.0076] |
| `header_atoms_all_max` | -0.0061 | [-0.0146, +0.0030] |
| `header_atoms_all_mean` | -0.0068 | [-0.0150, +0.0015] |

The header is a **null to negative** on UPR, exactly as on OMCT (where it
cost -0.0315 of within-document AUC): the only positive arm is the
local+global blend, which mixes the chunked score with the whole-text score
and is a wash. The two modes differ by at most 0.0037. The header's
hypothesis (the chunk geometry carries no label-driven ranking signal) holds
on a second corpus.

### The label-side transform: the template wins, decisively

The label text is either the **bare name** ("Detention") or the config
**template** ("This recommendation is about Detention.") — the UPR analogue of
OMCT's free `keep` vs `strip_block` comparison. Both variants are in the same
run (each family points at its own prior label run), in the pair-scoring
family where the label side carries the whole signal:

| label text | micro-F1 | Δ vs the champion (paired) | 95% interval | shortlist R@50 |
|---|---|---|---|---|
| the `about` template (the champion) | 0.4134-0.4377 | -0.1102 to -0.1345 | (all exclude zero) | 0.8655-0.8703 |
| the bare name | 0.3731-0.3951 | **-0.1527 to -0.1748** | (all exclude zero) | 0.8533-0.8612 |

The **template is clearly better** than the bare name (-0.05 to -0.06 of
micro-F1 between the two variants, and +0.012 to +0.017 of shortlist
recall). On OMCT the same question was model-dependent ("the unstripped
question helped one model and hurt two"); on UPR there is one answer: the
label's *sentence* is a better query than the label's bare noun — the
template's context tells the model what kind of text it is looking for.

### The late chunking: the model's own pooling formula wins again, but the effect is small

The late chunking embeds the text **once** and pools its token-level
embeddings per chunk. The consistency check is in the artifact: the cosine
between the text's pooled token vector and the model's own embedding of the
same text is **0.9468** (the token coverage 1.0, no chunk without a token),
so the token-level path lives in the model's own space.

| arm (voyage-4-nano) | Δ micro-F1 | 95% interval | Δ nDCG@5 | shortlist R@50 |
|---|---|---|---|---|
| the early-chunked arm (`words25_max`) | +0.0023 | [-0.0046, +0.0091] | +0.0016 | 0.8158 |
| the late chunking, the model's formula (`mean`) | +0.0038 | [-0.0068, +0.0137] | **+0.0075** | **0.8374** |
| the late chunking, per-token-normalized (`norm`) | **-0.0076** | [-0.0215, +0.0046] | **-0.0108** | 0.7934 |

**The model's own pooling formula beats the per-token-normalized variant**
again (on OMCT: +0.0260 of within-document AUC; here: +0.0114 of micro-F1
and +0.0183 of nDCG@5 between the two poolings), and the late chunking
beats the early one on the **ranking** side (the nDCG@5 and the shortlist
recall) but not on the decision side — and none of the intervals excludes
zero. In the **long-text tail** (300+ characters) the late chunking's
shortlist recall is 0.9133 against the early arm's 0.8533 (+0.06), which is
where its mechanism (the document's context in every chunk) should bite.

The late arm's score **level** is not comparable with the early arm's (the
background's mean shifts by -0.14: a pooled token vector is not the model's
sentence embedding, it is the average of the text's tokens). The ranking
metrics (the nDCG@5, the shortlist recall) are the scale-free ones, and they
are the ones to read.

**What this run does not say.**

- **Nothing about the feature pack, the combiner or the floors** (CS-8d /
  CS-8e): no learned scorer is involved here.
- **Nothing about the calibration** (CS-8f) and nothing about the decision
  models on UPR (CS-9).
- **Nothing about `test`** — only val was touched.
- **The late chunking is one model.** It is HF-only, so only
  `voyage-4-nano` is measured; the champion (`embeddinggemma`, Ollama) cannot
  run the arm at all. The within-model comparison is against the voyage
  champion, not against the champion of the pipeline.
- **The header was measured on one inner chunker** (the atoms). The
  OMCT-style "the header on the word windows" variant is not re-run here: the
  two chunkers' effects on UPR are already measured in CS-8b, and the header
  was negative on both of them there.

**The next step.** CS-8d: the UPR feature pack (the global and chunk
cosines, the lexical overlap, the char n-grams, the description coverage, the
chunk profile), the logistic combiner fitted on train, and the three floors —
the track's most interesting UPR question, whether the **lexical** block's
gain (OMCT's finding) transfers to a 68-label human-rights vocabulary.

## CS-8d — the UPR feature pack, the logistic combiner and the three floors (2026-10-26)

**The run.** `upr-analyze-learned` — the track's UPR artifact
(`results/upr-analyze-learned/analysis.json`) on the **val** split (660 texts,
68 labels, 1,312 gold cells, 43,568 background cells). The whole-text
**reference arm** (the standing champion: embeddinggemma, the fusion
a=0.5 knn / b=1.0 prototype / c=1.0 zero-shot with knn k=10, `top_k k=2`) is
computed in the same run, so the pairing is by construction; the guard holds
(the reference arm reproduces `upr-eval-fusionc-embeddinggemma-k10-c1` field by
field, and `uv run python tools/verify_upr_baseline.py "upr-analyze-*"` is
green for all five UPR runs).

**The pack** (`text_processors/features/upr_features.py`, the UPR analogue of
the OMCT pack): the global cosine, the best-chunk cosine, the mean of the 2
and of the K=3 best chunk cosines, their spread and the peak margin (the
**embedding** group); the soft lexical overlap in both directions (the label's
**name** ↔ the text) and the char-3-gram cosine (the **lexical** group). The
chunker is the **atoms** (the CS-8b ladder's best chunker on UPR). The pack
computes a grid, not a pair list: its output is a cube of
`(n_texts, n_labels, n_features)`, one row per (text, label) cell.

**The gap: the label catalog.** The pack's two label-side blocks — the
**label-description coverage** and the **label-type features** (the label's
parent category) — need a **label catalog** (a name / description / parent per
label, `paths.label_catalog`). The repo has none (`label_catalog: null`), so
the run packs the two catalog-free groups only. The catalog-gated groups are
*implemented* behind the config hook (`upr_analysis.learned.groups`), and
selecting them without a catalog is a config-time error; the artifact records
the gap (`learned.pack.label_catalog`). **This is the stated gap of CS-8d.**

**The combiner** (the track's `Combiner` port, the logistic regression): fitted
on **train** only (the train grid's 113,968 rows — 3,087 gold cells at weight
1 and 110,881 non-gold cells as down-weighted presumed negatives at weight
0.3, as on OMCT) and evaluated on **val**. The scaler is fitted inside the
fit, so the evaluated split never touches the model.

### The headline (val; the UPR harness's own metrics)

| arm | micro-F1 | macro-F1 | sample-F1 | coverage | P@5 | R@10 | nDCG@5 | shortlist R@50 | hit@1 | gold rank p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| the reference arm (the whole-text champion) | **0.5479** | 0.5006 | 0.5269 | 0.868 | 0.2770 | 0.7126 | **0.6725** | 0.7423 | 0.3666 | 0.0147 |
| `learned_logistic` (the pack + the logistic) | 0.4749 | 0.4842 | 0.4622 | **0.971** | 0.2542 | 0.7066 | 0.6070 | **0.8692** | 0.3064 | 0.0294 |
| `learned_logistic_random` (the random-vector floor) | 0.3837 | 0.3875 | 0.3739 | 0.941 | 0.2106 | 0.6033 | 0.4990 | 0.8723 | 0.2492 | 0.0441 |

**The paired bootstrap** (2,000 resamples, seed 42, the **text** as the
resampling unit, against the reference arm of the same run): the micro-F1
difference is **-0.0729 [-0.1010, -0.0454]** and the nDCG@5 difference
**-0.0654 [-0.0960, -0.0338]** — both intervals exclude zero, so the learned
arm is **significantly worse than the champion at the decision view**. Per
length bucket (the paired micro-F1): 0-99 **-0.1372** [-0.1964, -0.0761],
100-199 **-0.0658** [-0.1034, -0.0277], 200-299 -0.0275 [-0.0933, +0.0359],
300+ -0.0545 [-0.2163, +0.0952] — the loss is in the short texts, and the two
long buckets' intervals include zero.

### The verdict: a worse classifier, a much better candidate generator

**The pack does not beat the champion at the decision view, and it is a far
better shortlist generator.** The shortlist recall@50 is **0.8692 against the
champion's 0.7423** (+0.127) — the same shape as the CS-8b pair-scoring
family, and the same fact about the champion: its kNN and prototype terms push
gold cells out of the top-50 that the zero-shot geometry keeps. PLAN's
Phase-6 reranker slot lives on the **shortlist** side, so this is the track's
most relevant UPR result so far. The arm's **coverage** is 0.971 against the
champion's 0.868: it suggests more of the split's gold-bearing labels
somewhere (a diversity win) and pays for it in precision (the gold cells' rank
p50 degrades from 0.0147 to 0.0294).

**The lexical block transfers — and it is the ranking signal.** The
**random-vector floor** (the pack's embedding sides replaced with seeded noise,
the lexical features left real) keeps **0.3837** of the arm's 0.4749
(-0.0912, the paired interval against the reference arm: -0.1641 [-0.1955,
-0.1333]) and its shortlist recall is **unchanged** (0.8723 against the real
arm's 0.8692). So on UPR, as on OMCT, the **lexical** signal is the ranking
signal and the **embedding geometry** contributes (almost) nothing to the
ranking — on a completely different vocabulary (68 human-rights labels
against OMCT's 294 questions). The pre-registered question ("does the lexical
signal transfer?") is answered: **yes** — but the lexical signal alone does
**not** beat the champion's fusion at the decision view.

### The floors: the UPR grid carries no construction signal

| floor | construction | micro-F1 (5 draws) | mean | nDCG@5 (mean) |
|---|---|---|---|---|
| chance (the label-permutation floor) | the fit side's whole label vector permuted (the class counts and the weights stay; only the assignment changes) | 0.0448 … 0.3777 | 0.2292 | 0.3066 |
| control-role | the fit side's gold cells' positives relocated into the non-gold space (the non-gold cells keep their presumed-negative role; the original gold cells become negatives) | 0.0274 … 0.0426 | 0.0333 | 0.0590 |
| random-vector | the pack's embedding sides replaced with seeded random unit vectors | 0.3837 (one draw) | — | 0.4990 |

**The control-role floor is at chance** (0.027-0.043, against the random
top-2 expectation of ~0.029). So on UPR the training design's own prior ("the
ordinary cells are negative") buys **nothing** — the opposite of OMCT, where
the control-role floor reached 0.557-0.579 (above the plain cosine) because
the reviewed set was **rank-selected**. UPR's grid is not rank-selected, so
there is no construction signal for a label-blind model to pick up. The
arm's advantage over its floor is therefore **entirely label-driven**.

**The chance floor is a distribution, not a number** (0.045-0.378 over the
five draws), and that is the finding, not a defect: the pack's feature space
has a dominant **topical** axis, so a linear model fitted on a *random* 2.7%
target still finds that axis — with a random sign. The label-blind model's
val AUC moves between 0.44 and 0.66 across draws, and the top-2 decision
amplifies that spread. The honest reading: the arm (0.4749) is above the
**best** chance draw (0.3777), but the chance floor's *mean* is a weak
summary and its spread is wide. (On OMCT the same construction landed on a
tight 0.494-0.505 — the same fit on a corpus whose feature space has no
dominant axis.)

### The unseen labels: the pack discovers five of the eight

The label-holdout labels (the eight `unseen` ones) are **never** in train, so
only a label-agnostic mechanism can find them. The champion's geometry cannot
(its unseen bucket: micro-F1 0.0000, coverage 0.000, shortlist R@50 0.034).
**The pack's soft lexical overlap can**: the unseen bucket's micro-F1
**0.4758**, the coverage **0.875**, the shortlist R@50 **0.7211**.

The per-label view (the artifact's per-label distributions) says *which*:

| unseen label | the pack: the gold cells' mean vs the background's | the reference arm |
|---|---|---|
| Gender-based violence | 0.5521 vs 0.0751 | 0.3960 vs 0.2411 |
| Freedom of association and peaceful assembly | 0.5385 vs 0.0781 | 0.3805 vs 0.2659 |
| Freedom of religion and belief | 0.5246 vs 0.0539 | 0.3705 vs 0.2200 |
| Asylum-seekers - refugees | 0.4502 vs 0.0142 | 0.3443 vs 0.1526 |
| Special procedures | 0.4002 vs 0.0372 | 0.2685 vs 0.1751 |
| ESC rights - general | 0.0917 vs 0.0633 | 0.2405 vs 0.2328 |
| General | 0.0065 vs 0.0120 | 0.0946 vs 0.0881 |
| Other | 0.0043 vs 0.0046 | 0.0759 vs 0.0679 |

The five **specific** unseen labels ("Gender-based violence",
"Freedom of association…", "Asylum-seekers - refugees", "Special
procedures") are found with a wide separation; the three **generic** ones
("General", "Other", "ESC rights - general") are not — for "General" the gold
cells even score *below* the background (0.0065 vs 0.0120). The mechanism is
the vocabulary, not the geometry: a label whose name is a distinct phrase
matches the texts that are about it, whatever the model has never seen.

**What this run does not say.**

- **Nothing about the chunk-profile block, the gradient-boosted combiner or
  the feature-group ablation** (CS-8e): the pack runs with two groups and the
  logistic combiner only.
- **Nothing about the calibration or the decision-rule comparison** (CS-8f).
- **Nothing about the decision models on UPR** (CS-9).
- **Nothing about `test`** — only val was touched.
- **The floors are five draws.** The chance floor's spread is wide (above),
  so its mean is a weak summary; the control-role floor is tight.
- **The arm's level numbers are not comparable with the champion's**: the
  combiner's output is a probability, the champion's a cosine. Only the
  rank-based numbers (the gold cells' rank percentile, the shortlist recall)
  and the paired intervals are comparable. The background's level shift
  (-0.4882) is a scale artefact, and its **rank** shift is 0.0000.
- **The pack has no length feature** (the length is measurable on UPR, so it
  is recorded as a diagnostic and never fed to the model) and **no
  label-frequency feature** (the label frequency is a corpus statistic; the
  pack's spec has no frequency column, and the head/torso/tail/unseen buckets
  already report its effect).

**The next step.** CS-8e: the two "not the lever" checks re-run on UPR — the
gradient-boosted combiner behind the same port, the chunk-profile block as the
pack's fifth group, the feature-group ablation (the groups are a config knob)
and the per-feature diagnostics — with the **lexical** block's transfer as the
headline question, which this change set has answered.

## CS-8e — the chunk profile, the feature-group ablation and the boosted combiner (2026-10-26)

**Ten runs**, all on the UPR **val** split, all in the same harness as CS-8d
(the whole-text reference arm in the same run, so the pairing is by
construction; the guard holds for all fifteen UPR runs).

- `upr-analyze-learned-cp-{base,full,profile}` — the **chunk profile** as the
  pack's fifth group (the same three arms as OMCT's CS-6c: the base pack, the
  full pack, the profile alone).
- `upr-analyze-learned-abl-{embed,lex,noembed,nolex}` — the **feature-group
  ablation** (the groups are a config knob: the pack never changes, only the
  columns the combiner sees).
- `upr-analyze-learned-histgb-{full,lex-only,embed-only}` — the
  **gradient-boosted combiner** (sklearn's `HistGradientBoostingClassifier`,
  the pinned defaults, no grid) behind the same port.

The `cp-base` run reproduces the CS-8d run's numbers exactly (micro-F1
0.4749, the shortlist R@50 0.8692) — the two sweeps' cross-check.

### The ablation: the lexical block does **not** carry the gain here

| arm (the group set) | micro-F1 | macro-F1 | nDCG@5 | shortlist R@50 | the random-vector floor |
|---|---|---|---|---|---|
| the reference arm (the whole-text champion) | **0.5479** | 0.5006 | **0.6725** | 0.7423 | — |
| `base` (embedding + lexical) | 0.4749 | 0.4842 | 0.6070 | 0.8692 | 0.3837 |
| `full` (embedding + lexical + profile) | 0.4704 | 0.4846 | 0.6074 | 0.8721 | 0.3815 |
| `profile` alone (the chunk profile) | 0.4726 | 0.4960 | 0.6122 | **0.8734** | 0.1421 |
| `embed` (the embedding block alone) | 0.4460 | 0.4479 | 0.5878 | 0.8639 | 0.0258 |
| `lex` (the lexical block alone) | 0.3837 | 0.3904 | 0.4985 | 0.8673 | — (the pack *is* the floor) |
| `noembed` (lexical + profile) | 0.4749 | 0.4902 | 0.6070 | 0.8746 | — |
| `nolex` (embedding + profile) | 0.4726 | 0.4967 | 0.6148 | 0.8734 | 0.1459 |

**The pre-registered question — "does the lexical signal transfer?" — is
answered: it transfers, but it is *not* the dominant block on this
vocabulary.** On OMCT the lexical block was the signal (the embedding block
alone was at its own floor). On UPR the ranking is the other way round: the
**embedding block alone** (0.4460) is *above* the **lexical block alone**
(0.3837), and the lexical block adds **+0.0289** on top of the embedding
block (0.4460 → 0.4749). The lexical block's own contribution to the pack is
what the random-vector floor measures: the base pack loses 0.0912 of micro-F1
when the embedding sides are replaced with noise (0.4749 → 0.3837) — so the
lexical features carry 81% of the pack's number, but the pack's number is
below the champion's in any case.

**The chunk profile is not the lever** (the OMCT verdict transfers): in the
full pack it costs **-0.0045** of micro-F1 (0.4749 → 0.4704, a wash), and its
random-vector floor (0.1421 against the real arm's 0.4726) says its signal
*is* the embedding geometry re-measured at a finer granularity. The profile
*alone* is a good arm (0.4726, the best shortlist of the pack family at
0.8734) — but `nolex` (the embedding block + the profile) gives **exactly**
the profile's numbers (0.4726), i.e. the embedding block adds nothing on top
of the profile: the two are the same signal measured twice.

**The per-feature diagnostics** (the CS-6c precedent, on the UPR grid: every
feature's own AUC against the gold cells, its background-side counterpart and
its length correlation — the `cp-full` artifact) say the same thing from the
feature level:

| feature | its own AUC (val) | the gold's mean vs the background's | the length correlation |
|---|---|---|---|
| `cprof_ngram_top1` | 0.7927 | 0.3950 vs 0.2909 | +0.259 |
| `chunk_cosine_best` | 0.7884 | 0.3379 vs 0.2348 | +0.197 |
| `chunk_cosine_top2_mean` | 0.7835 | 0.3049 vs 0.2181 | +0.113 |
| `cosine_full` | 0.7818 | 0.2988 vs 0.2030 | -0.110 |
| `cprof_sent_top1` | 0.7812 | 0.2993 vs 0.2035 | -0.103 |
| `lex_label_coverage` | 0.7566 | 0.5159 vs 0.2085 | +0.274 |
| `lex_char_ngram_cosine` | 0.7484 | 0.1994 vs 0.0911 | +0.096 |
| `lex_text_coverage` | 0.7174 | 0.1084 vs 0.0480 | -0.007 |
| `cprof_align_lexical` | 0.7177 | 0.5404 vs 0.1050 | +0.065 |
| `cprof_ngram_top2` | 0.7222 | 0.2942 vs 0.2370 | **+0.451** |
| `cprof_ngram_top3` | 0.6621 | 0.2311 vs 0.1897 | **+0.605** |
| `chunk_cosine_spread` | 0.6005 | 0.0384 vs 0.0202 | +0.371 |
| `chunk_peak_margin` | 0.5994 | 0.0434 vs 0.0223 | +0.393 |
| `cprof_sent_top2` / `cprof_sent_top3` | 0.5010 / 0.5008 | ~0 | +0.130 / +0.058 |
| `cprof_align_sent` / `cprof_align_ngram` | 0.5144 / 0.4856 | ~0 | -0.206 / +0.206 |
| `cprof_align_semantic_only` | 0.2823 | 0.4596 vs 0.8950 | -0.065 |

Three readings:

1. **The profile's rank columns re-measure the embedding geometry** (the
   top-1 ranks are within 0.01 of the chunk cosines) and its **lower ranks
   are length proxies** (the correlation grows from +0.259 at the top-1 rank
   to **+0.605** at the top-3 rank) — the CS-6c finding, reproduced on UPR.
2. **The profile's second sentence is at chance** (0.5010): UPR's texts have
   one or two sentences, so the second and third sentence ranks carry
   nothing. The profile's *granularity* kinds (`align_sent` 0.5144,
   `align_ngram` 0.4856) are at chance too.
3. **The profile's one new signal is lexical**: `cprof_align_lexical` (the
   aligned unit shares a content word with the label's name) reaches
   **0.7177**, and its complement `cprof_align_semantic_only` 0.2823. That is
   the *lexical block re-measured at the unit level* — which is why the
   profile adds nothing to the pack that already has it.

### The boosted combiner: on UPR the learner *does* matter (a little)

| arm (the group set) | the logistic's micro-F1 | the booster's micro-F1 | the logistic's R@50 | the booster's R@50 |
|---|---|---|---|---|
| `full` (embedding + lexical + profile) | 0.4704 | **0.4818** | 0.8721 | **0.8909** |
| `lex` (the lexical block alone) | 0.3837 | **0.4043** | 0.8673 | **0.8991** |
| `embed` (the embedding block alone) | **0.4460** | 0.4293 | 0.8639 | 0.8697 |

On OMCT the booster was *below* the logistic on 8 of the 9 arms ("the learner
is not the lever"); on UPR it is **above** on the two packs that carry the
lexical block (+0.0114 and +0.0206 of micro-F1) and below on the
embedding-only arm (-0.0167). So the OMCT verdict **does not transfer** — but
the booster still does not beat the champion at the decision view (0.4818 vs
0.5479; the paired interval against the reference arm: -0.0661 [-0.0924,
-0.0390]).

**The booster's shortlist recall is the best the track has measured on UPR**
(0.8909-0.8991 against the champion's 0.7423 and the logistic's 0.8692-0.8746)
— the shortlist side, which is where PLAN's Phase-6 reranker slot lives.

**And the booster's chance floor is tight** (micro-F1 0.0388-0.0616 against
the logistic's 0.0540-0.3465): a tree ensemble fitted on a *random* 2.7%
target does not find the feature space's dominant axis the way a linear model
does — so the CS-8d finding ("the chance floor is a distribution, because the
feature space has a dominant topical axis") is a property of the **linear**
learner, not of the pack. The control-role floor stays at chance for the
booster too (0.0122-0.0274).

**What this change set does not say.**

- **Nothing about the calibration or the decision-rule comparison** (CS-8f).
- **Nothing about the decision models on UPR** (CS-9) and nothing about
  `test` — only val was touched.
- **The booster's arms are three group sets**, not the whole ladder: the
  OMCT precedent (`all`, `lex-only`, `embed-only`) with UPR's group names.
- **The two learners' arms live in different runs**, so they are compared
  through their own paired intervals against the shared reference arm (the
  whole-text champion) and not through a paired per-pair bootstrap between the
  two runs (the OMCT track's `tools/compare_stage3_arms.py` reads the
  out-of-fold scores; the UPR driver does not write them, so the UPR
  comparison is the per-arm number beside the shared reference).
- **The per-feature diagnostics are on the evaluated split and the fit side**,
  and they say what a feature is worth *alone*; they do not say what the
  combiner used it for (the combiner's coefficients are in the artifact).

**The next step.** CS-8f: the reporting blocks — the calibration (the ECE,
the Brier score, the reliability diagram, pooled and per bucket) and the
decision-rule comparison (`top_k`, `score_threshold`, `top_k_min_score`) on
the best arm.

## CS-8f — the reporting blocks: the calibration and the decision rules (2026-10-26)

**Two runs**, both on the UPR **val** split, both on the CS-8e grid's **best
arm** (the gradient-boosted combiner on the full pack — the best decision-view
number and the best shortlist recall the track has measured on UPR), with the
**logistic** on the same pack as the second run so the two learners'
calibration can be read side by side:

- `upr-analyze-learned-report-histgb`
- `upr-analyze-learned-report-logistic`

The guard holds for all seventeen UPR runs. The threshold rules' cut is
chosen on the **fit side** (the train split) by the **Youden** separation
(positives against the non-gold cells — on UPR the negative class is the
ordinary cells, so this is the "floor gap" analogue of the OMCT reporting,
not OMCT's "youden against the hard negatives"), so the evaluated split never
picks the operating point.

### The calibration: the learned arm's probability is a probability

| arm | the pooled ECE | the Brier | the per-bucket ECE (head / torso / tail / unseen) | the per-length ECE (0-99 / 100-199 / 200-299 / 300+) |
|---|---|---|---|---|
| the boosted combiner | **0.0323** | **0.0239** | 0.0259 / 0.0417 / 0.0407 / **0.0121** | 0.0263 / 0.0315 / 0.0389 / **0.0517** |
| the logistic regression | 0.0436 | 0.0275 | 0.0274 / 0.0568 / 0.0577 / **0.0118** | 0.0338 / 0.0416 / 0.0578 / **0.0705** |

The learned arm's probability is **well calibrated** — the pooled ECE 0.0323
(the booster) and 0.0436 (the logistic) against the OMCT learned arm's
0.175-0.189 and its per-document macro of 0.327-0.349. The calibration is
best on the **unseen** labels (the ECE 0.012) and worst on the **long texts**
(0.0517/0.0705), i.e. the long texts are where the arm's probability is
least trustworthy — the same bucket the CS-8b ladder found the champion's
weakest.

**And the booster is better calibrated than the logistic** (0.0323 against
0.0436) — the opposite of the usual expectation, and the second way in which
the learner matters on UPR (the first is the decision view, CS-8e).

### The decision rules: a budget rule still wins

| rule | the booster's micro-F1 | the logistic's micro-F1 | the booster's coverage (the predicted/gold cells) |
|---|---|---|---|
| `top_k` (k=2, the run's rule) | 0.4818 | 0.4704 | 1.006 (1,320 cells) |
| `score_threshold` (the fit side's cut, 0.0924 / 0.0868) | 0.2465 | 0.1773 | 4.621 / 7.046 (6,063 / 9,245 cells) |
| `top_k_min_score` (the top-2 with the fit side's cut as a floor) | **0.4945** | **0.4800** | 0.936 / 0.941 (1,228 / 1,234 cells) |

**The best operating point is the top-k with the fit side's cut as a floor**:
it keeps the top-2 budget and drops the cells whose probability is below the
fit side's threshold, and it beats the plain top-k on both learners (+0.0127
for the booster, +0.0096 for the logistic). The **pure threshold rule is much
worse** (0.2465 / 0.1773): the probabilities are **compressed** toward the
base rate (the grid's positive rate is 2.9%), so a fixed cut predicts 4.6-7.0
times the gold cells. The same compression the CS-7b decision-model arm
showed on OMCT ("the model is calibrated but its probabilities are
compressed, so a fitted threshold is still needed") — here the fitted
threshold is useful only as a **floor** under a budget rule, not as the rule
itself.

**What this change set does not say.**

- **Nothing about the decision models on UPR** (CS-9) and nothing about
  `test` — only val was touched.
- **The calibration is measured on the evaluated split**, so it is a
  diagnostic of the arm, not a fit-side quantity; the threshold *is* fit-side.
- **The two learners' arms live in two runs**, so the calibration comparison
  is the two numbers side by side (the same protocol the CS-6b/6c arms used:
  each arm against the shared reference arm).
- **The reliability diagram's bins are equal width over [0, 1]** (10 bins),
  and the ECE is the count-weighted mean absolute gap between the mean
  predicted probability and the observed rate in each bin.

**The next step.** CS-9: the decision models on UPR — a **measurement**, not
a pipeline component: `tev1` and `laya` through the existing `Decider` port
and the `/v1/systemone` client, the UPR-native question type ("is this text
about <label>?") with the `noul`/`choice` types and both prompt shapes, the
synthetic competence control beside every arm, the grid arm and the
safe-negative rule in its three forms with the label-blind counterpart beside
them, and the shortlist rerank. **CS-10 (the LLM stage) waits for the
reviewer's explicit OK** — it is not started.

## CS-9 — the decision models on UPR (a measurement) (2026-10-26)

**The runs.** Five, all on the UPR **val** split, through the existing
`Decider` port and the `/v1/systemone` client (the model id **and** digest
pinned, the prompt version recorded, the calls cached — a re-run of the sweep
reproduces every number exactly from the cache, in 0.3 s per run):

| run | model | shape / type | texts | cells | calls |
|---|---|---|---|---|---|
| `upr-decide-tev1-paragraph-noul` | tev1 | paragraph / noul | 150 | 10,200 | 300 + 300 |
| `upr-decide-tev1-paragraph-choice` | tev1 | paragraph / choice | 150 | 10,200 | 300 + 300 |
| `upr-decide-laya-paragraph-noul` | laya | paragraph / noul | 150 | 10,200 | 602 + 602 |
| `upr-decide-laya-paragraph-choice` | laya | paragraph / choice | 150 | 10,200 | 602 + 602 |
| `upr-decide-tev1-pair-choice` | tev1 | pair / choice | 20 | 1,360 | 1,360 + 1,360 |

**The scope, and the deviation from the plan's wording (the reviewer's call).**
The plan's CS-9 lists the **grid arm** (the whole (text x label) grid), the
safe-negative rule in its three forms and the shortlist rerank. The cost of
the full grid is now **measured**, and it is the reason for the deviation: the
`paragraph` shape is ~2 calls per text (~1,320 calls per model for the val
grid), and the `pair` shape is **one call per (text, label) cell** — the
pair-shape arm made 1,360 calls for 20 texts, i.e. **68 calls per text**
against the paragraph shape's 2, so the full val grid in that shape is ~44,880
calls per arm per model. This change set therefore measures a **seeded sample**
(150 of the 660 val texts for the paragraph shape, 20 for the pair shape), and
**the full grid, the safe-negative rule and the shortlist rerank are left for a
follow-up (CS-9b)** with that cost in front of it. The sample is the same
texts for every arm (so the arms are paired), and the artifact carries the
**pipeline's own scoring of the same cells** (the champion's scores restricted
to the sample's texts), so every probe number has its reference beside it on
the same texts.

### The headline (val, the sample's 150 texts; the champion on the same texts)

| scorer | micro-F1 | macro-F1 | nDCG@5 | shortlist R@50 | hit@1 | the gold's rank p50 |
|---|---|---|---|---|---|---|
| **the reference arm (the whole-text champion)** | 0.5415 | 0.4966 | 0.6565 | 0.7322 | 0.3510 | 0.0147 |
| **tev1, paragraph / noul** | **0.5482** | **0.5713** | **0.7055** | **0.9978** | 0.3344 | 0.0147 |
| tev1, paragraph / choice | 0.5349 | 0.5638 | 0.6861 | 0.9978 | 0.3278 | 0.0147 |
| laya, paragraph / noul | 0.2226 | 0.2087 | 0.3376 | 0.9956 | 0.1457 | 0.1029 |
| laya, paragraph / choice | 0.1960 | 0.1815 | 0.2810 | 0.9728 | 0.1060 | 0.1324 |
| the label-blind counterpart (tev1, noul) | **0.0365** | — | 0.0789 | 0.7778 | 0.0199 | — |

**tev1 beats the champion on the decision view (+0.0067 of micro-F1) and on
the ranking view (+0.0490 of nDCG@5), and it is a near-perfect *candidate
generator*: the shortlist recall@50 is 0.9978 against the champion's 0.7322 —
the decision model puts the gold label in the text's top-50 of 68 labels
almost always, where the champion's kNN and prototype terms lose 27 points of
gold cells.** That is the most Phase-6-relevant number the track has measured
on UPR.

**The label-blind counterpart is at chance** (the micro-F1 0.0365 against the
random top-2 expectation of ~0.029, the nDCG@5 0.0789), so the decision
model's quality is **label-driven**: it is not a construction artefact of UPR's
label space. (The same yardstick on OMCT is what turned the safe-negative
rule's cost into a quality claim, CS-7b.)

**The competence control passes decisively** (the AUC 0.9816 for tev1's `noul`
arm, 0.9896 for its `choice` arm, 0.9008 / 0.8840 for laya's), so a
chance-level reading is never the ambiguity here: the models are the right
shape.

**laya is much weaker than tev1** (the micro-F1 0.2226 / 0.1960 against
tev1's 0.5482 / 0.5349) — the same ordering as on OMCT (CS-7a). But laya's
**shortlist** is nearly as good (0.9956 / 0.9728): it ranks the gold label
inside the top-50 but rarely at the top (the hit@1 0.1457 / 0.1060 against
tev1's 0.3344), so its loss is a *top-1* loss, not a *ranking* loss.

**The prompt grid, on UPR: the `noul` type beats the `choice` type for tev1**
(the micro-F1 0.5482 against 0.5349) — the **opposite** of OMCT, where the
`choice` won (CS-7a). The `choice` type is however **better calibrated**
(the ECE 0.1553 against 0.1971, the Brier 0.0735 against 0.0933). For laya
the `choice` type is worse on every count (the micro-F1 0.1960, the ECE
**0.3994**, the Brier 0.2436).

### The unseen labels: the decision model finds what the champion cannot

| bucket | the champion's micro-F1 / coverage / nDCG@5 | tev1 (noul) | the label-blind counterpart |
|---|---|---|---|
| head | 0.6903 / 1.000 / 0.9831 | 0.6271 / 1.000 / 0.8981 | 0.1425 / 1.000 / 0.2977 |
| torso | 0.4713 / 0.958 / 0.8505 | 0.5000 / 0.958 / 0.8195 | 0.0213 / 1.000 / 0.1397 |
| tail | 0.5882 / 0.769 / 0.9330 | 0.5333 / 1.000 / 0.9385 | 0.0308 / 1.000 / 0.2357 |
| **unseen** | **0.0000 / 0.000 / 0.6046** | **0.4706 / 1.000 / 0.8737** | 0.1090 / 1.000 / 0.4456 |

**The decision model discovers the eight label-holdout labels** (the unseen
bucket: the micro-F1 0.4706 against the champion's 0.0000, the coverage 1.000
against 0.000, the nDCG@5 0.8737 against 0.6046) — and the label-blind
counterpart does not (0.1090), so the discovery is the model's own reading
of the text, not a construction artefact. This is the strongest UPR argument
for a decision model in the loop: the champion's geometry **cannot** suggest
a label it has never seen (the CS-8a verdict), and neither can the CS-8d
feature pack without its lexical block — but a decision model can, because it
reads the *text* against the *label's name*.

The decision model is *worse* on the head bucket (0.6271 against the
champion's 0.6903) and better on the torso and the unseen ones: the two
scorers are complementary, and the champion's head labels (the frequent ones)
are exactly the ones the kNN and prototype terms get right.

### The probability compression (measured, not assumed)

| arm | the gold cells' mean probability | the background's | the gold above 0.5 | the background above 0.5 | the ECE |
|---|---|---|---|---|---|
| tev1, noul | 0.7578 | 0.2105 | 0.8675 | 0.1430 | 0.1971 |
| tev1, choice | 0.7371 | 0.1681 | 0.8510 | 0.1116 | 0.1553 |
| laya, noul | 0.4852 | 0.1813 | 0.5232 | 0.1053 | 0.1607 |
| laya, choice | 0.7241 | **0.4200** | 0.8543 | **0.3716** | 0.3994 |

**tev1 is not compressed** (the gold cells' mean 0.7578, 87% of them above
0.5) — the opposite of the CS-7b decision arm on OMCT (the positives' mean
0.41, "the model is calibrated but its probabilities are compressed"). So on
UPR a **fixed 0.5 cut is a meaningful operating point** for tev1, which the
OMCT stage could not say. **laya is the compressed one** (its `noul` arm's
gold mean 0.4852, 52% above 0.5) and its `choice` arm is compressed the other
way (it says "yes" too often: 37% of the background cells are above 0.5),
which is why its calibration is the worst of the four (the ECE 0.3994).

**What this change set does not say.**

- **Nothing about the full grid, the safe-negative rule or the shortlist
  rerank** (CS-9b): the cost of the grid is measured and stated above, and
  the rule needs the grid.
- **Nothing about `test`** — only val was touched.
- **The sample's numbers are the sample's**: 150 texts (302 gold cells) for
  the paragraph shape and 20 texts (40 gold cells) for the pair shape, so the
  pair shape's numbers are indicative only (its job here is the shape's
  **cost**: 68 calls per text against 2).
- **The decision models are frozen and zero-shot**: they carry no UPR label,
  so nothing leaks; and a decision model only *scores* — it never relabels,
  curates or corrects the data.
- **The non-gold labels' true positive rate is unmeasured**, so these numbers
  are a reach and a cost, never a precision claim. The label-blind
  counterpart is the yardstick that says how much of the cost is the model's
  own label-blind signal, and the competence control says the model is the
  right shape.

**The next step.** CS-9b: the full-grid arm, the safe-negative rule in its
three forms (the fixed 0.5 cut, the fit-side threshold, the margin rule) with
the label-blind counterpart beside them, and the shortlist rerank — the
numbers the Phase-6 slot needs. **CS-10 (the LLM stage) still waits for the
reviewer's explicit OK.**

**Two bookkeeping notes on the track's UPR table** (the only hand edit of the
table so far, and it is a *removal*, never a number): the CS-9 sweep's first
run wrote its rows before the row helper carried the run id, the split and the
scope, so ten rows with an empty run id sat next to the ten correct ones;
they were **superseded and removed** (the table's upsert keys on the run id,
so it could not replace them itself). The scratch smoke run (four texts, a
temporary config) had written a section of its own; its section and its
`results/upr-decide-smoke` folder were removed as well. No number was edited.

## CS-11 — the label catalog: the description channel and the sibling backoff (2026-10-27)

**The catalog.** `catalogs/upr_labels.jsonl` (68 entries: `name`,
`description`, `parent`) with its provenance in
`catalogs/upr_labels.meta.json`. It is **LLM-written** (the reviewer's
decision): the writer saw the **68 label names and the public human-rights
taxonomy only** — never a UPR record, never a label-frequency statistic — and
the catalog was **frozen and committed before the runs that use it** (the
meta file records the sha256, the rules and the freeze). The taxonomy has
**12 parents**; the label space is covered exactly (68 of 68, no extra).

**The runs.** Seven, all on the UPR **val** split (the track's table carries
the rows; the guard is green for every `analyze-upr` run):

| run | what |
|---|---|
| `upr-labelembed-embeddinggemma-desc` | the input artifact: the 68 label **descriptions** embedded as the label-side query (68 embeddings) |
| `upr-eval-fusionc-embeddinggemma-k10-c1-desc` | the guard's reference for the description label side (the champion with the description vectors) |
| `upr-analyze-learned-cat-control` | the **control**: the CS-8f grid's best arm, re-run in this sweep (it **reproduces the CS-8f artifact's metrics exactly** — the reproducibility check) |
| `upr-analyze-learned-cat-desc` | the label query = the **description** |
| `upr-analyze-learned-cat-feat` | the **catalog-gated feature groups** on (`label_part`, `label_type`) |
| `upr-analyze-learned-cat-backoff` | the **sibling-max backoff** + its control |
| `upr-analyze-learned-cat-desc-backoff` | the description query + the backoff |

**The comparisons are paired across runs** (the resampling unit is the text;
2,000 resamples, seed 42) through the new repo tool
`tools/compare_upr_arms.py`: each run's artifact now carries every arm's
**per-text** summary (the decision view's (tp, fp, fn) counts and the ranking
view's per-text nDCG, overall and per label bucket), and the tool bootstraps
the difference of two arms of two different runs on the same resampled texts.

### The headline (val; the learned arm; the control is the same pack without the catalog's channels)

| arm | micro-F1 | macro-F1 | nDCG@5 | shortlist R@50 | unseen micro-F1 | unseen nDCG@5 |
|---|---|---|---|---|---|---|
| the champion (the whole-text reference arm) | 0.5479 | 0.5006 | 0.6725 | 0.7423 | 0.0000 | 0.6575 |
| **the control** (the learned arm) | 0.4818 | 0.4652 | 0.6137 | 0.8909 | 0.4469 | 0.6641 |
| the description query | **0.5084** | 0.4872 | 0.6557 | 0.8864 | 0.4690 | 0.6650 |
| the catalog-gated groups | **0.5076** | **0.5077** | **0.6643** | **0.8955** | **0.4824** | **0.6950** |

### The paired differences against the control (the text is the resampling unit)

| scope | the description query: micro-F1 | the catalog-gated groups: micro-F1 |
|---|---|---|
| **pooled** | **+0.0266** [+0.0084, +0.0435] | **+0.0258** [+0.0091, +0.0433] |
| pooled nDCG@5 | +0.0420 [+0.0259, +0.0577] | +0.0506 [+0.0339, +0.0658] |
| head | +0.0444 [+0.0137, +0.0732] | −0.0013 [−0.0288, +0.0254] |
| torso | −0.0022 [−0.0351, +0.0276] | +0.0437 [+0.0133, +0.0743] |
| tail | +0.0590 [−0.0011, +0.1211] | +0.0516 [+0.0037, +0.1013] |
| **unseen** | +0.0221 [−0.0195, +0.0630] | +0.0355 [−0.0070, +0.0782] |
| unseen nDCG@5 | +0.0010 [−0.0250, +0.0276] | **+0.0309** [+0.0088, +0.0535] |

**Verdict 1 — the description channel is a real, label-driven win.** Both
readings of "the description as its own channel" beat the control, and the
pooled interval **excludes zero** in both: the description as the label's
**query** (+0.0266 of micro-F1) and the catalog-gated **features** (the
description's coverage + the parent one-hots, +0.0258). The gain is
**label-driven** (the arms' own control-role floors stay at chance, as in
CS-8d) and it is the **pooled** number that carries it: the **unseen**
bucket's rise is in the same direction in both arms (+0.0221 and +0.0355) but
its own interval **includes zero** (the unseen bucket is eight labels and
~100 gold cells on val — a wide interval), so the pre-registered condition
("the unseen bucket must rise with the interval excluding zero") is **not
met** and only the catalog-gated arm's unseen **nDCG** interval excludes
zero. The honest claim is therefore: *the description adds a signal the name
does not have, and it adds it everywhere; it is not yet shown to add it
specifically to the held-out labels.*

**Verdict 2 — the description is a weaker channel than the name, pooled.**
The per-feature diagnostics say so directly: the **name's** lexical coverage
ranks the cells at the AUC **0.7566** and the **description's** coverage at
**0.7067** — the scratch's "the description loses where the name is already
distinctive", measured. The pack gains because the two channels are
*complementary*, not because the description dominates. The per-label
structure of the scratch's claim ("the description beats the name where the
name is weak") is **not** measured here: the diagnostics are pooled per
feature, and the per-label breakdown is an open follow-up.

**Verdict 3 — the sibling-max backoff does not ship.** The mechanism *acts*
(its report is in the artifact): the unsupported labels' scores change on
20–81% of the cells (λ = 0.5) with a mean lift of +0.011 to +0.083, and the
two catch-alls ("General", "Other") have **no supported sibling** — a null by
construction, exactly as the plan predicted. But the **decision view gets
worse**: the unsupported arm's unseen bucket falls by **−0.0356**
[−0.0579, −0.0117] (λ = 0.5) and by **−0.1080** [−0.1505, −0.0626] (λ = 1),
and the pooled number falls too (−0.0053 [−0.0121, +0.0015] at λ = 0.5,
−0.0623 [−0.0807, −0.0454] at λ = 1). The **control** (the same rule on the
fit side's eight rarest *supported* labels) falls too (−0.0091
[−0.0150, −0.0038] at λ = 0.5, −0.0904 [−0.1061, −0.0752] at λ = 1), so the
backoff has the *fallback* signature (it hurts the labels that have their
own support) — but it does **not** buy the held-out labels anything. The
reason is visible in the rule itself: the backoff takes the **max** over the
siblings, so it transfers a sibling's *peak* relevance, not its relevance to
this text, and the label is then suggested in the texts it does not belong
to. The pre-registered falsification ("the backoff must raise the unseen
labels that have a supported sibling") **fires**: the backoff does not
ship, and the honest claim is "the sibling-max backoff as specified
transfers relevance peaks, not relevance". The combination arm agrees: the
description query + the backoff falls −0.0091 [−0.0167, −0.0015] against the
description query alone.

**Verdict 4 — the parent one-hots are (nearly) empty on their own.** The 12
`labeltype_*` features rank the cells at the AUC 0.4705–0.5433 — a label's
parent category carries almost no information about whether a *cell* is
gold. Their contribution in the `cat-feat` arm is therefore the *parent-level
intercept* they share with the label's siblings (the taxonomy's own
structure), not a per-cell signal. That is why the catalog's win lives in
the **description** channel and not in the parent one-hots.

**What this change set does not say.**

- **Nothing about `test`** — only val was touched.
- **Nothing about the label catalog's *own* quality** — the descriptions are
  one LLM's reading of the label names; the frozen catalog is the input, and
  a second writer's catalog is not measured.
- **The per-label structure of the description's advantage is not measured**
  (the diagnostics are pooled per feature).
- The **unseen** bucket's rise rests on ~100 gold cells, so its interval is
  wide; the pooled intervals are the solid part of the claim.
- The catalog is **label-side knowledge**; the raw data was not touched, and
  nothing was relabelled.

**The next step.** CS-12 (the listwise second stage, whose question's
`criteria` field carries these descriptions) and CS-13 (the decision
probability as a pack feature). **CS-10 (the LLM stage) still waits for the
reviewer's explicit OK.**





## CS-9b — the full-val grid, the safe-negative rule and the shortlist rerank (2026-10-27)

*(This entry is appended after the CS-11 one: the CS-9b run was still in
flight — 11,984 decision calls — when the CS-11 entry was written.)*

**The runs.** Two, on the UPR **val** split's **whole grid** (660 texts x 68
labels = 44,880 cells), the champion's arms as the reference:

| run | the arm |
|---|---|
| `upr-decide-tev1-paragraph-noul-grid` | tev1, the `paragraph` shape, the `noul` type |
| `upr-decide-tev1-paragraph-choice-grid` | tev1, the `paragraph` shape, the `choice` type |

**The cost:** 5,992 calls per arm — the val grid (1,320), its label-blind
counterpart (1,320) and the **fit side** (the train grid's 3,352 calls, which
the rule's operating points are fitted on). The `laya` arms and the `pair`
shape stay at their CS-9 samples (a second decision model is closed as a
lever, and the `pair` shape's full grid is 44,880 calls per arm for a shape
whose cost is already measured).

### The headline (val, the whole grid)

| arm | micro-F1 | nDCG@5 | shortlist R@50 | **unseen** micro-F1 | unseen coverage | ECE | the blind counterpart's micro-F1 |
|---|---|---|---|---|---|---|---|
| the champion (the whole-text reference) | 0.5479 | 0.6725 | 0.7423 | **0.0000** | 0.000 | — | — |
| **tev1, `noul`** | **0.5502** | **0.7247** | **0.9987** | **0.5267** | 1.000 | 0.1982 | 0.0517 |
| tev1, `choice` | 0.5494 | 0.7176 | 0.9987 | 0.5106 | 1.000 | 0.1557 | 0.0540 |

**The decision model beats the champion on the full val split** — the micro-F1
0.5502 against 0.5479, the nDCG@5 0.7247 against 0.6725, the shortlist
recall@50 0.9987 against 0.7423 — and it does so while **finding the eight
held-out labels** (the unseen bucket 0.5267 against the champion's 0.0000).
The label-blind counterparts are at chance (0.0517 and 0.0540), so the
quality is label-driven. The CS-9 sample's findings **hold at full-val
scale**: the `noul` type beats the `choice` type (0.5502 against 0.5494, the
nDCG@5 0.7247 against 0.7176) while the `choice` type is **better
calibrated** (the ECE 0.1557 against 0.1982); and tev1's probabilities are
**not compressed** on UPR (the gold cells' mean 0.7543 against the
background's 0.2116), so a fixed 0.5 cut is a meaningful operating point.

### The safe-negative rule (the `noul` arm; the `choice` arm in brackets)

| form | the coverage (the reach) | the harm (the gold cells discarded) | the blind counterpart's coverage | the blind counterpart's harm |
|---|---|---|---|---|
| the **fixed 0.5 cut** | 0.8584 (0.8908) | 0.1288 (0.1555) | 0.8396 | **0.7553** (0.7782) |
| the **fit-side threshold** (0.6070) | 0.9328 (0.9242) | 0.2012 (0.1928) | 0.9142 | **0.8270** (0.8239) |
| the **margin rule** (0.2823) | 0.9143 (0.9171) | 0.1502 (0.1532) | 0.8941 | **0.8247** (0.8285) |

(the harm at the **matched** coverage: the blind 0.8505 for the fit-side
threshold and 0.8475 for the margin rule — against the real arm's 0.2012 and
0.1502.)

**Verdict 1 — the rule's cost is a quality claim on UPR too.** The
label-blind counterpart discards **75-85%** of the gold cells where the real
arm discards 13-20%, at a *similar* reach. That is the CS-7b OMCT
signature, and it is what turns the UPR rule's cost from "a construction
artefact" into a quality number — the deliverable now ships a **reach claim
and a cost claim that is label-driven** on UPR as well (the background cells'
true positive rate stays unmeasured, so there is still **no precision
claim**).

**Verdict 2 — the best form on UPR is the margin rule, not the fit-side
threshold.** The OMCT verdict (the fit-side threshold is the best form) does
**not** transfer: the fit-side threshold is the *most aggressive* form here
(the coverage 0.9328 at the harm 0.2012), and the rule's own trade-off
criterion (the coverage minus the harm) prefers the **margin rule** (0.9143
at the harm 0.1502 — 0.7641 against the fixed cut's 0.7296 and the fit-side
threshold's 0.7316). The pre-registered expectation is therefore **not
met**, and the honest reading is: *on UPR the probabilities are not
compressed, so a fitted threshold over-clears; the margin rule is the
shape that trades reach against harm best.*

### The shortlist rerank (the champion's top-K, re-ordered by the decision model)

| K | the shortlist's own recall (the ceiling) | the champion's hit@1 | the decision model's own hit@1 | the **reranked** hit@1 |
|---|---|---|---|---|
| 5 | 0.6966 | 0.3666 | 0.3483 | 0.3521 |
| 10 | 0.7462 | 0.3666 | 0.3483 | 0.3384 |
| 20 | 0.7652 | 0.3666 | 0.3483 | 0.3293 |
| 50 | 0.7797 | 0.3666 | 0.3483 | 0.3239 |

**Verdict 3 — the rerank does not ship, and it says where the ceiling is.**
The pre-registered falsification **fires**: the reranked hit@1 (0.3521 at
K = 5) is *below* the champion's own (0.3666) at every K, so the decision
model is **not** a better ranker of the champion's shortlist than the
champion's own geometry. But the table also says something more useful:
**the shortlist's own recall is the binding constraint** (0.7462 at K = 10 —
even a *perfect* reranker of the champion's top-10 could only reach 74.6%),
while the decision model **alone** reaches 0.9987. So the Phase-6 slot
should not be fed the champion's shortlist: a reranker worth having has to
be fed the *decision model's* candidates (or the union's — CS-14's number),
and it has to be a **trained** ranker, because the zero-shot decision model
is a good candidate generator and a weak one.

**What this change set does not say.** Nothing about `test` (only val was
touched); nothing about a *precision* claim for the rule (the background's
true positive rate is unmeasured); nothing about the `laya` arms or the
`pair` shape at full-val scale (they stay at their CS-9 samples, and the
exclusion is the reviewer's to reverse).

## CS-12 — the listwise second decision stage (2026-10-27)

**The runs.** Two, both on the UPR **val** split's **whole grid** (660 texts x
68 labels = 44,880 cells), beside the pointwise arm they build on:

| run | the second round's criteria |
|---|---|
| `upr-decide-tev1-paragraph-noul-listwise` | the CS-11 catalog's **descriptions** (the main arm) |
| `upr-decide-tev1-paragraph-noul-listwise-nodesc` | the label's **own name** (the control: it isolates the *listwise mechanism* from the *descriptions*) |

**The mechanism.** Round 1 is the existing 68 pointwise `noul` questions per
text (unchanged, and **free**: the calls are CS-9b's cache hits). Round 2 is
**one** call per text asking "which of these 20 topics is the text about?",
the 20 options being the round-1 top-20. The arm's ranking is a **rank
fusion** (the round-2 order over the candidates, then the round-1 order for
the rest) — parameter-free, so nothing is fitted on val — and its score
matrix is the negative final rank (so the calibration and the probability
compression are reported as **not applicable**, as the plan says).

**The feasibility check came first**: a 20-option `choice` had never been
exercised against the server, so the shape was smoke-tested on **3 texts**
(`tools/smoke_upr_listwise.py`): the server accepts it, the per-option
probabilities come back, and **no** call was truncated (the planner counts the
options *and* the criteria in the input budget — the plan's requirement). The
real runs: **0 truncated cells** out of 660, 1,320 calls per arm (the real
one and its label-blind counterpart).

### The numbers (val, the whole grid)

| arm | micro-F1 | macro-F1 | nDCG@5 | shortlist R@50 | hit@1 | unseen micro-F1 |
|---|---|---|---|---|---|---|
| the pointwise arm (the reference) | **0.5502** | 0.5609 | 0.7247 | 0.9987 | 0.3483 | **0.5267** |
| the listwise (the descriptions) | 0.5205 | 0.5368 | 0.7272 | 0.9987 | **0.3613** | 0.4659 |
| the listwise (the names, the control) | 0.5327 | — | 0.7373 | 0.9987 | — | 0.5195 |
| the label-blind counterpart (the descriptions) | 0.0433 | 0.0349 | 0.0792 | 0.7850 | 0.0168 | 0.0263 |

### The paired differences against the pointwise arm (the text is the resampling unit)

| scope | the listwise (the descriptions): micro-F1 | the listwise (the names): micro-F1 |
|---|---|---|
| **pooled** | **−0.0296** [−0.0535, −0.0038] | −0.0175 [−0.0386, +0.0054] |
| pooled nDCG@5 | +0.0025 [−0.0206, +0.0270] | +0.0126 [−0.0088, +0.0348] |
| head | −0.0260 [−0.0606, +0.0099] | −0.0196 [−0.0513, +0.0144] |
| torso | −0.0368 [−0.0846, +0.0107] | −0.0108 [−0.0519, +0.0320] |
| tail | +0.0441 [−0.0309, +0.1172] | −0.0483 [−0.1040, +0.0107] |
| **unseen** | **−0.0609** [−0.1106, −0.0078] | −0.0072 [−0.0559, +0.0460] |

**Verdict — the listwise stage does not ship.** The pre-registered
falsification **fires on both clauses**: the listwise arm's micro-F1 does
**not** beat the pointwise arm's (it is 0.0296 *lower*, with the interval
excluding zero) and its **unseen** bucket falls by 0.0609 [−0.1106, −0.0078].
The mechanism is not inert — the top-2 changes on most texts (the
diagnostic the plan asked for) — it is *harmful*. Two things in the numbers
say why, and they are worth the next session's attention:

- **the first pick improves and the second pick does not.** The listwise's
  **hit@1** is 0.3613 against the pointwise arm's 0.3483 (**+0.0130**): the
  explicit choice finds the gold label more often *first*. But the *second*
  pick degrades enough to lose the top-2 decision view (and the unseen
  bucket's). So the listwise is a better *first* guess and a worse *pair*.
- **the descriptions make it worse, not better.** The control (the same
  mechanism with the label's **name** as the criterion) is the milder arm:
  the pooled difference −0.0175 (the interval includes zero) against the
  descriptions' −0.0296, and the unseen −0.0072 against −0.0609. So the
  catalog's descriptions, which help the pack (CS-11), *hurt* the listwise
  question: a 20-way choice with 20 long descriptions is a harder question
  for the model than a 20-way choice with 20 short ones.

**The order of the reviewer's hypothesis, restated by the measurement.** The
hypothesis was "the decision model is excellent at *finding* the candidates
and much weaker at *ordering* them". What the measurement says is sharper:
the model's **explicit** ordering (a 20-way choice) is **worse** than its
**implicit** one (the 68 independent probabilities' order) — the pointwise
arm's top-2 is the better pair. So on UPR the decision model is a very good
candidate generator (the shortlist R@50 0.9987 against the champion's
0.7423) and should not be asked to do the ordering. The listwise's
label-blind counterpart is at chance (0.0433), so its quality is
label-driven — the shape works, the mechanism just does not help.

**What this change set does not say.** Nothing about `test` (only val was
touched); nothing about a *smaller* candidate list (the 20 was the plan's
pre-registered number — a shorter list is a different question and would need
its own pre-registration); nothing about a *decoding* change (the round-2
answer's per-option probabilities are used as they are). The 20-option
question's cost (1,320 calls per arm) is the price of a negative result, and
the arm's calls are cached, so a re-read is free.

the union candidate generator, as a number (2026-10-27)



## CS-13 — the decision probability as a feature in the UPR pack (2026-10-27)

**The runs.** Four, all on the UPR **val** split, the pack fitted on **train**
(the main pipeline's protocol) and the guard green for every run:

| run | the feature | the combiner |
|---|---|---|
| `upr-analyze-learned-dec-histgb` | the decision probability (the real pairing) | the gradient-boosted combiner |
| `upr-analyze-learned-dec-logistic` | the decision probability (the real pairing) | the logistic combiner |
| `upr-analyze-learned-dec-blind-histgb` | the decision probability of the **label-blind counterpart** | the gradient-boosted combiner |
| `upr-analyze-learned-dec-blind-logistic` | the decision probability of the **label-blind counterpart** | the logistic combiner |

The control is `upr-analyze-learned-cat-control` (the same pack, the same
combiner, **without** the feature). The feature is the decision model's
probability for the (text, label) cell — **cell-local** by construction (the
pack never calls the model: the CLI scores the cells at the edge and passes
the matrix in), with no corpus statistic, no document identity and no length.

**The cost:** the train and val probabilities are **CS-9b's cache hits**
(the rule's fit side needs them anyway), so the only new calls are the
label-blind counterpart's **fit side** (3,352 calls) — the blind val side and
both real sides were already paid for.

### The headline (val; the learned arm; the control is the same pack without the feature)

| arm | micro-F1 | macro-F1 | nDCG@5 | shortlist R@50 | **unseen** micro-F1 | unseen nDCG@5 |
|---|---|---|---|---|---|---|
| the champion (the whole-text reference arm) | 0.5479 | 0.5006 | 0.6725 | 0.7423 | **0.0000** | 0.6575 |
| the control (the learned arm) | 0.4818 | 0.4652 | 0.6137 | 0.8909 | 0.4469 | 0.6641 |
| **+ the decision probability (the booster)** | **0.5965** | **0.5839** | **0.7567** | 0.9856 | **0.5697** | 0.7923 |
| + the decision probability (the logistic) | 0.5760 | 0.5837 | 0.7407 | 0.9992 | 0.5485 | 0.8352 |
| + the **label-blind** probability (the booster) | 0.4977 | — | 0.6380 | 0.8869 | 0.4738 | 0.6646 |
| + the **label-blind** probability (the logistic) | 0.4719 | — | 0.6034 | 0.8723 | 0.4656 | 0.6688 |

### The paired differences against the control (the text is the resampling unit)

| scope | the booster: micro-F1 | the logistic: micro-F1 |
|---|---|---|
| **pooled** | **+0.1147** [+0.0964, +0.1349] | +0.0942 [+0.0758, +0.1127] |
| pooled nDCG@5 | +0.1430 [+0.1243, +0.1620] | +0.1270 [+0.1087, +0.1453] |
| head | +0.1432 [+0.1117, +0.1747] | +0.1324 [+0.1024, +0.1636] |
| torso | +0.0707 [+0.0351, +0.1070] | +0.0540 [+0.0177, +0.0888] |
| tail | +0.1241 [+0.0600, +0.1880] | +0.1007 [+0.0420, +0.1629] |
| **unseen** | **+0.1228** [+0.0763, +0.1718] | +0.1016 [+0.0560, +0.1455] |
| unseen nDCG@5 | +0.1282 [+0.0994, +0.1592] | +0.1712 [+0.1378, +0.2063] |

**Verdict — the hypothesis is confirmed, and it is the track's largest UPR
result.** Every pre-registered clause holds:

- the augmented pack's **unseen** bucket is **not below** the control's (0.5697
  against 0.4469 — and against the reviewer's reference of 0.4758 for the
  logistic pack), and the **paired** interval excludes zero (+0.1228
  [+0.0763, +0.1718]);
- its shortlist R@50 (0.9856) is not below the control's (0.8909);
- the augmented pack's own **control-role floor** stays **at chance** (the
  micro-F1 0.0120 over the five draws — the CS-8d level was 0.027-0.043), so
  the gain is **label-driven** and not the training design's own prior;
- the **label-blind** counterpart's arm reaches only 0.4977 (+0.0160
  [+0.0038, +0.0279] over the control) against the real arm's +0.1147 — so
  **93% of the feature's contribution is the model's label reading** and the
  yardstick confirms it.

**The mechanism, as the measurement states it.** The hypothesis was that the
decision probability transfers to the labels the train split never saw,
because it is computed from the label's **name** — identically for a held-out
label — so the combiner's fitted weight applies there too. The **fit side has
zero positives for the eight unseen labels** (0 of 3,087 train gold cells),
so the transfer is the whole story: the unseen bucket rises by 0.1228
([+0.0763, +0.1718]) and the *pooled* number by 0.1147, and the fit side
never saw the labels it is being judged on. The **random-vector floor** says
the rest of the story: with the embedding sides replaced by noise, the
augmented arm keeps **0.5798** (the unseen 0.5534) — so the decision
probability carries the ranking and the **embedding geometry contributes
almost nothing** (0.5965 against 0.5798). That is the CS-6d's OMCT finding
("the lexical block is the ranking signal, the embedding geometry is the
retrieval signal") in a third voice: on UPR the *decision model* is the
ranking signal, and the geometry is not.

**What this means for the track's headline.** The augmented arm's micro-F1
(0.5965) is above the champion's (0.5479) — the first UPR number the track
has measured that beats the main pipeline's whole-text scorer at the
decision view, and it does so while **finding the eight held-out labels**
(the unseen bucket 0.5697 against the champion's 0.0000). The price is
stated: the feature needs the decision model in the loop (2 calls per text,
1,320 calls for the val split, ~3,352 for the fit side) — that is a *scoring*
cost, not a *retrieval* cost, and the Phase-6 slot decision is the
reviewer's.

**What this change set does not say.** Nothing about `test` (only val was
touched — the one test shot for an adopted config is still owed); nothing
about the feature's *cost* in a production setting (the calls are cached
here, and a fresh run pays 1,320 calls per 660 texts); nothing about a
decision model *other* than tev1 (laya stays closed as a lever); and
nothing about the feature's own calibration (the arm's probability is the
combiner's, not the decision model's).

the union candidate generator, as a number (2026-10-27)

**The run.** One: `upr-decide-tev1-paragraph-noul-grid-union` (the whole val
grid; the decision calls are CS-9b's cache hits, so the run costs **zero**
new model calls). The union is the per-row **z-max** of the champion's and
the decision model's scores — and it is a **measurement, not a component**:
the union is not built and enters no pipeline (the 2026-10-22 trim put
"union shortlists" out of the track's menu; they are main-plan Phase 4/6
machinery). The number is for PLAN.

### The recall curve (val, the whole grid; the share of the gold cells inside the text's top-k)

| scorer | recall@1 | recall@3 | recall@5 | **recall@10** | nDCG@5 | shortlist R@20 | shortlist R@50 | hit@1 |
|---|---|---|---|---|---|---|---|---|
| the champion (the geometry) | 0.4139 | 0.6295 | 0.6773 | 0.7126 | 0.6725 | 0.7260 | 0.7423 | 0.3666 |
| the decision model | 0.3953 | 0.6928 | 0.7904 | 0.8845 | 0.7247 | 0.9538 | 0.9987 | 0.3483 |
| **the union** | **0.4192** | **0.7241** | **0.8355** | **0.9120** | **0.7693** | 0.9718 | 0.9970 | 0.3697 |
| the union with the **label-blind** scores (the control) | 0.4134 | 0.5811 | 0.6218 | 0.6947 | 0.6307 | 0.7553 | 0.8337 | 0.3659 |

**Verdict — the union's gain is real and label-driven, and it is a number for
PLAN, not a component.** The union's **recall@10 = 0.9120** against the
decision model's 0.8845 and the champion's 0.7126: a candidate generator
built on both would put the gold label in a ten-label shortlist for **91%**
of the texts, where the champion's geometry manages 71% and the decision
model alone 88%. The **control** is the yardstick: the same z-max with the
decision model's **label-blind** scores reaches only 0.6947 — *below* the
champion's own 0.7126 — so the union's gain is the model's label reading and
not a construction artefact of the z-normalization.

**But the union's *decision* view has a scale artefact, and it is worth
stating plainly** (the union's top-2 decision view is in the artifact, and it
is *not* a pipeline proposal):

| the union's top-2 decision view | pooled | head | torso | tail | **unseen** |
|---|---|---|---|---|---|
| the union's micro-F1 | 0.5843 | 0.7332 | 0.5341 | 0.4901 | **0.2540** |
| the decision model's micro-F1 | 0.5502 | 0.6510 | 0.4842 | 0.4359 | **0.5267** |

The union's pooled micro-F1 (0.5843) is the best number the track has
measured on UPR — above the champion's 0.5479 — and its **unseen** bucket
(0.2540) is *half* the decision model's (0.5267). The reason is the z-max's
scale: the champion's per-text z-scores are more spread out than the
decision model's, so the max is biased toward the champion's scale, and the
champion's scores for a held-out label are noise (its unseen coverage is
0.000). A union of two *calibrated* scores is not a union of two *z-scored*
ones. So the union's honest reading is the one the plan asked for: **it is
a candidate generator's number (the recall curve), not a scorer's** — and
the number is the recall@10 0.9120, with the decision view's unseen-bucket
artefact stated beside it.

**What this change set does not say.** Nothing about `test`; nothing about
*building* the union (the track's menu excludes it, and this run is a
measurement); nothing about a z-normalization other than the per-row one
(the per-text z-score is a choice, and a different one would move the
numbers — the recall curve is the robust part, the decision view is not).

## CS-14 — the union candidate generator, as a number (2026-10-27)

**The run.** One: `upr-decide-tev1-paragraph-noul-grid-union` (the whole val
grid; the decision calls are CS-9b's cache hits, so the run costs **zero**
new model calls). The union is the per-row **z-max** of the champion's and
the decision model's scores — and it is a **measurement, not a component**:
the union is not built and enters no pipeline (the 2026-10-22 trim put
"union shortlists" out of the track's menu; they are main-plan Phase 4/6
machinery). The number is for PLAN.

### The recall curve (val, the whole grid; the share of the gold cells inside the text's top-k)

| scorer | recall@1 | recall@3 | recall@5 | **recall@10** | nDCG@5 | shortlist R@20 | shortlist R@50 | hit@1 |
|---|---|---|---|---|---|---|---|---|
| the champion (the geometry) | 0.4139 | 0.6295 | 0.6773 | 0.7126 | 0.6725 | 0.7260 | 0.7423 | 0.3666 |
| the decision model | 0.3953 | 0.6928 | 0.7904 | 0.8845 | 0.7247 | 0.9538 | 0.9987 | 0.3483 |
| **the union** | **0.4192** | **0.7241** | **0.8355** | **0.9120** | **0.7693** | 0.9718 | 0.9970 | 0.3697 |
| the union with the **label-blind** scores (the control) | 0.4134 | 0.5811 | 0.6218 | 0.6947 | 0.6307 | 0.7553 | 0.8337 | 0.3659 |

**Verdict — the union's gain is real and label-driven, and it is a number for
PLAN, not a component.** The union's **recall@10 = 0.9120** against the
decision model's 0.8845 and the champion's 0.7126: a candidate generator
built on both would put the gold label in a ten-label shortlist for **91%**
of the texts, where the champion's geometry manages 71% and the decision
model alone 88%. The **control** is the yardstick: the same z-max with the
decision model's **label-blind** scores reaches only 0.6947 — *below* the
champion's own 0.7126 — so the union's gain is the model's label reading and
not a construction artefact of the z-normalization.

**But the union's *decision* view has a scale artefact, and it is worth
stating plainly** (the union's top-2 decision view is in the artifact, and it
is *not* a pipeline proposal):

| the union's top-2 decision view | pooled | head | torso | tail | **unseen** |
|---|---|---|---|---|---|
| the union's micro-F1 | 0.5843 | 0.7332 | 0.5341 | 0.4901 | **0.2540** |
| the decision model's micro-F1 | 0.5502 | 0.6510 | 0.4842 | 0.4359 | **0.5267** |

The union's pooled micro-F1 (0.5843) is the best number the track has
measured on UPR — above the champion's 0.5479 — and its **unseen** bucket
(0.2540) is *half* the decision model's (0.5267). The reason is the z-max's
scale: the champion's per-text z-scores are more spread out than the
decision model's, so the max is biased toward the champion's scale, and the
champion's scores for a held-out label are noise (its unseen coverage is
0.000). A union of two *calibrated* scores is not a union of two *z-scored*
ones. So the union's honest reading is the one the plan asked for: **it is
a candidate generator's number (the recall curve), not a scorer's** — and
the number is the recall@10 0.9120, with the decision view's unseen-bucket
artefact stated beside it.

**What this change set does not say.** Nothing about `test`; nothing about
*building* the union (the track's menu excludes it, and this run is a
measurement); nothing about a z-normalization other than the per-row one
(the per-text z-score is a choice, and a different one would move the
numbers — the recall curve is the robust part, the decision view is not).
