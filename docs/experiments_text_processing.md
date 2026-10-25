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
