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

