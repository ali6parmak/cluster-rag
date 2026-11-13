# TOPIC_CLASSIFICATION_PLAN.md — the EURLEX-4K topic-classification track

Satellite to `AGENTS.md` and `PLAN.md` (created 2026-11-13). **This is the live
context for a new session**: why the track exists, the rules, what has been
tried, and what is open. The detailed pre-registrations, tables and verdicts
are in `docs/topic_classification.md`; every number below is copied from there
or from `results/eurlex-*/metrics.json`. The detailed doc is the log; this file
is the map.

## 0. Why this track exists

- **The product question.** The main pipeline suggests labels by retrieving and
  scoring candidates. On a label space of **4,271** EUROVOC concepts, the
  interesting question is no longer "what is the top-2?" but **"can a cheap
  generator put the right labels on a shortlist of ~50, and how do we rank
  them?"**
- **The data.** `data/eurlex_4k/` (read-only, gitignored): 57,000 EU legal
  documents (45,000 train / 6,000 dev / 6,000 test), multi-label, numeric or
  alphanumeric EUROVOC IDs, and a 7,201-row concept-title vocabulary.
- **The reference, not the target.** The UPR track measured embeddings 0.5479,
  dictionary alone 0.5243 (~96% of embeddings), boosting + the dictionary column
  0.5502, the fit-side mask 0.5874 and the best pack 0.5965 (micro-F1@2 on UPR
  val). Those are **references**: EURLEX has ~4,271 labels, so micro-F1@2 is
  the wrong instrument. The track uses recall@k, R-precision, nDCG@k and
  predicted cardinality.
- **The metric change.** Primary: **R@50** (shortlist breadth), **R-precision**
  (cardinality-aware precision), **nDCG@10** (ordering at the top). Secondary:
  R@5, R@10, R@20, R@100, nDCG@50. Report coverage@50, predicted cardinality
  and per-bucket (head / torso / tail / unseen) numbers beside them.

## 1. Governance and rules

- **Data is read-only.** `data/eurlex_4k/` is never edited, cleaned or
  re-split silently. The splits are the provided ones.
- **`train` = index/fit, `dev` = tuning and evaluation, `test` = held out.**
  No test number is reported until the final shot; the 74 labels that occur
  only in test are excluded from dev evaluation.
- **The honest evaluation is `family_new`**, not the random split: dev
  documents whose title-skeleton family is absent from train. The random split
  has 26.1% near-duplicate documents and is reported beside the family slice.
- **One change set per commit.** Pre-register the mechanism, the shape of the
  number and the falsification before the run; write the artifact, the doc
  section, run `ruff` + `pytest`, commit, report, stop.
- **Every number comes from an artifact** in `results/eurlex-*`; the detailed
  doc copies it. No hand-written tables.
- **No LLM in the loop without flagging the cost.** No label-holdout mask until
  the held-out label set is defined and approved.
- **Dependency direction is one-way.** `src/cluster_rag/topic_classification/`
  may import `cluster_rag`; `cluster_rag` must not import it.
- **Pure cores stay pure.** The metrics and any future helpers are numpy +
  standard library; model clients (embedders, LLMs) live at the edge and are
  wired in the experiment scripts.

## 2. The dataset in one screen

| split | documents | distinct labels | labels/document (mean / median) |
|---|---:|---:|---:|
| train | 45,000 | 4,108 | 5.07 / 5 |
| dev | 6,000 | 2,515 | 5.04 / 5 |
| test | 6,000 | 2,517 | 5.06 / 5 |

- **Label space.** 4,271 distinct IDs in the union; **4,197** are present in
  train+dev (the dev-evaluable space); 74 occur only in test. The vocabulary
  file is a **superset** (7,201 rows, 0 duplicates, all titled); the join has
  **0 unmatched**. **163** labels have no train example dataset-wide; **89**
  of them are in dev (96 dev gold assignments).
- **Support.** 709 labels have exactly one train document; 2,267 have fewer
  than 10; 441 have at least 100. Buckets: head ≥100 (441), torso 10–99
  (1,400), tail 1–9 (2,267), unseen 0 (163 dataset-wide / 89 dev).
- **Text.** Median document 2,693 characters / 447 words; recitals carry 55%
  of the characters, main body 31%, title 6%, header 7.8%. The median is small
  enough for a single embedding; the long tail needs chunking.
- **Leakage.** 26.1% of dev/test documents have a train neighbour at TF-IDF
  cosine ≥0.90; 2,056 title-skeleton families cross splits (12,371 documents);
  1,945 documents cite a document held in another split. The clean slice (best
  train cosine <0.90) and `family_new` are the honest views.
- Full audit: `docs/topic_classification.md` §1–7 and
  `results/eurlex-stats/stats.json`.

## 3. Naming and where things live

| thing | path |
|---|---|
| this plan (context / map) | `TOPIC_CLASSIFICATION_PLAN.md` |
| detailed log (audit, pre-registrations, results) | `docs/topic_classification.md` |
| package (pure metrics and future helpers) | `src/cluster_rag/topic_classification/` |
| experiments | `experiments_topic_classification/` |
| artifacts (gitignored) | `results/eurlex-*` |
| run ids | `eurlex-<detail>` (e.g. `eurlex-shortlist`, `eurlex-fusion`) |
| embedding cache | `cache/` (MiniLM, version `eurlex-4k-v1`; labels `eurlex-4k-label-v1`) |

## 4. What has been tried (executed change sets)

| ID | what | key result | artifact | detailed doc | commit |
|---|---|---|---|---|---|
| TC-1 | **Data audit** — schema, label join, support, cardinality, text, leakage | 4,271 labels; 0 unmatched; 89 dev zero-shot; 26.1% near-dup; 2,056 cross-split families | `results/eurlex-stats/stats.json` | §1–7 | `fa3e83e` |
| TC-2 | **Dictionary vs embedding k-NN shortlist** — the cheap baselines | k-NN **0.9168** R@50 / 0.5715 R-precision / 0.7070 nDCG@10; dictionary primary 0.5388 / 0.3810 / 0.4835; count 0.8171 but head-heavy; prior 0.3203 | `results/eurlex-shortlist/metrics.json` | §8 | `e5d4fd0` |
| TC-3 | **Family-held-out evaluation** — is 0.917 family memorisation? | k-NN on `family_new` **0.8944 / 0.4869 / 0.6364** (clean **0.8862**); family effect 0.0224 all-dev, **0.0774** seen-vs-new; dictionary best `family_new` 0.7668, 0.1276 behind | `results/eurlex-family-heldout/metrics.json` | §9 | `df5825a` |
| TC-4 | **Label-name zero-shot arm** — the only signal for labels with no examples | label-name alone 0.2148 overall, **0.1042** unseen R@50; zero-shot-only **0.8678**; rescue 0.0104 (1/96); untuned fusion 0.8806 (−0.036) | `results/eurlex-labelname/metrics.json` | §10 | `d599a29` |
| TC-5 | **Dictionary as a feature** — one-parameter z-score fusion | primary CV R@50 0.9171 vs 0.9168, **+0.0003** (CI includes 0), R-precision **−0.0204**, nDCG **−0.0190**; log-odds +0.0029/+0.0037; count +0.0010; stage 2 not licensed | `results/eurlex-fusion/metrics.json` (+ `scores/`) | §11 | `8bf5d5d` |
| TC-6 | **Calibrated zero-shot channel** — rank/quantile-map the title scores | **null**: best α=1 gives 9/96 unseen (0.0938) and −0.00046 overall; CV selects α=0; zero-shot unsolved | `results/eurlex-zeroshot-calibrated/metrics.json` | §12 | `235a3ac` |

## 5. What we learned

- **The embedding k-NN is the shortlist generator.** The random-split 0.917 is
  largely real: on genuinely new families it is **0.894 R@50**, and on the
  clean new-family slice **0.886**. That is the baseline to beat.
- **The dictionary is weak alone and does not add through a global weight.**
  Its count arm reaches 0.817 R@50 but is head-heavy (tail 0.122); the primary
  z-scored arm is 0.539. The one-parameter fusion is a null for R@50 and costs
  ordering. The UPR “dictionary column” evidence does not reproduce in this
  cheap form.
- **Zero-shot labels are unsolved.** Titles carry a real signal (0.868 when
  restricted to the 89 zero-shot labels; 10/96 in the full top-50), but raw
  replacement finds 1/96 and quantile calibration finds 9/96; both are
  scale-bound and neither clears the 0.10 unseen bar.
- **The family effect is real but not dominant.** Family-seen dev documents
  are easier (0.972 R@50) than family-new (0.894), but 71% of dev is already
  family-new, so the all-dev average overstates new-family performance by
  ~0.022. Report `family_new` as the honest number.
- **The near-duplicate clean slice is a useful secondary, not the primary
  evaluation.** `family_new` is the main honest slice; clean is reported beside
  it.

## 6. Open options (prioritised)

1. **Better document embedding** — the current MiniLM truncates at 256 tokens;
   the median document is ~447 words. Try a longer-context model, a
   **chunk-mean** embedding, or a multi-view (title/header/recitals/body)
   embedding. Highest expected gain.
2. **Lexical/BM25 k-NN + reciprocal-rank fusion** — a different error profile;
   cheap and the UPR lexical route was strong.
3. **Cross-encoder rerank of the top-50** — target R-precision / nDCG@10, not
   R@50. A small cached ms-marco MiniLM or a legal-domain reranker.
4. **Learned candidate pack** — union of k-NN and dictionary top-50, features
   (k-NN score, dictionary scores, label frequency, label-name cosine, label
   co-occurrence, doc length), trained on train candidates with family-grouped
   cross-fit. The one-parameter gate failed, so this needs its own
   pre-registration and a separate justification.
5. **Label co-occurrence smoothing** — boost labels that co-occur with the
   top-scoring labels; cheap, tail-oriented.
6. **Zero-shot quota or learned calibrator** — a different insertion rule than
   raw replacement or quantile mapping; the title signal exists but does not
   survive the full ranking.
7. **Fine-tune the embedding** — contrastive training with train labels and
   hard negatives; likely the biggest gain, but a real training run.
8. **Cluster routing / label pruning / threshold calibration** — efficiency and
   product output; less likely to move R@50.
9. **Ship the current k-NN** — 0.894 new-family R@50 is already strong; the
   remaining work is marginal unless the product needs the tail or zero-shot.

## 7. Open questions

- Does a longer-context or chunked embedding improve the **0.894** new-family
  baseline, and by how much?
- Can a learned candidate pack beat the k-NN despite the one-parameter fusion
  null? If so, which features carry the signal?
- Can zero-shot labels be solved by a **quota** or a learned calibrator, or is
  the title signal fundamentally too weak at 4,271 labels?
- Is `family_new` the final evaluation protocol, and what is the one **test**
  shot for an adopted config?
- The vocabulary has **titles only** (no descriptions or hierarchy). Is a
  EUROVOC parent/description source worth adding, or is that out of scope?
- The dev-evaluable label space is 4,197; the dataset union is 4,271. Keep the
  74 test-only labels excluded from dev, or include them as floored columns?

## 8. Milestones

- **M1 — audit** (done): dataset, label space, leakage.
- **M2 — cheap shortlist baselines** (done): dictionary vs k-NN, metrics.
- **M3 — trustworthy evaluation** (done): family-held-out and clean slices.
- **M4 — zero-shot attempts** (done): label-name and calibrated channel, both
  nulls.
- **M5 — representation experiment** (open): longer-context / chunked /
  multi-view embedding.
- **M6 — rerank or pack** (open): cross-encoder or learned candidate combiner.
- **M7 — final test shot** (open): one adopted config, test once.

## 9. Next change sets (not queued)

- **TC-7 — representation:** pre-register a longer-context or chunk-mean
  embedding, evaluate on dev / `family_new` / clean, same metrics. This is the
  highest expected gain.
- **TC-8 — rerank:** cross-encoder on the top-50, measure R-precision/nDCG and
  the cost.
- **TC-9 — pack:** learned candidate combiner, only with a new pre-registration
  and a clear rationale after TC-5's null.
- **TC-10 — test:** one shot for the adopted config.

Each starts only on the reviewer's word; one change set per commit.

## 10. The durable deliverable

A **shortlist generator for 4,271 labels** with a trustworthy evaluation:
- the honest **new-family baseline** (k-NN 0.894 R@50 / 0.4869 R-precision /
  0.6364 nDCG@10; clean 0.886 / 0.4744 / 0.6241);
- the audit, the metrics module, the pre-registrations and the artifacts;
- a clear statement of what does **not** work (dictionary alone, one-parameter
  fusion, raw/calibrated label-name zero-shot) so the next session does not
  repeat it.

## 11. Cross-references

- Detailed log and full tables: `docs/topic_classification.md` §1–12.
- Metrics module: `src/cluster_rag/topic_classification/metrics.py`.
- Experiment scripts: `experiments_topic_classification/`.
- Artifacts: `results/eurlex-stats/`, `results/eurlex-shortlist/`,
  `results/eurlex-family-heldout/`, `results/eurlex-labelname/`,
  `results/eurlex-fusion/`, `results/eurlex-zeroshot-calibrated/`.
- The old text-processing track (UPR/OMCT) is separate: `TEXT_PROCESSING_PLAN.md`
  and `docs/experiments_text_processing.md`. Do not mix its numbers with
  EURLEX.
