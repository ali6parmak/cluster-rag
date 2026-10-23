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

