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
| embedding cache | `cache/` (MiniLM, version `eurlex-4k-v1`; chunk view `eurlex-4k-chunk200-v1`; field views `eurlex-4k-views-v1`; labels `eurlex-4k-label-v1`) |

## 4. What has been tried (executed change sets)

| ID | what | key result | artifact | detailed doc | commit |
|---|---|---|---|---|---|
| TC-1 | **Data audit** — schema, label join, support, cardinality, text, leakage | 4,271 labels; 0 unmatched; 89 dev zero-shot; 26.1% near-dup; 2,056 cross-split families | `results/eurlex-stats/stats.json` | §1–7 | `fa3e83e` |
| TC-2 | **Dictionary vs embedding k-NN shortlist** — the cheap baselines | k-NN **0.9168** R@50 / 0.5715 R-precision / 0.7070 nDCG@10; dictionary primary 0.5388 / 0.3810 / 0.4835; count 0.8171 but head-heavy; prior 0.3203 | `results/eurlex-shortlist/metrics.json` | §8 | `e5d4fd0` |
| TC-3 | **Family-held-out evaluation** — is 0.917 family memorisation? | k-NN on `family_new` **0.8944 / 0.4869 / 0.6364** (clean **0.8862**); family effect 0.0224 all-dev, **0.0774** seen-vs-new; dictionary best `family_new` 0.7668, 0.1276 behind | `results/eurlex-family-heldout/metrics.json` | §9 | `df5825a` |
| TC-4 | **Label-name zero-shot arm** — the only signal for labels with no examples | label-name alone 0.2148 overall, **0.1042** unseen R@50; zero-shot-only **0.8678**; rescue 0.0104 (1/96); untuned fusion 0.8806 (−0.036) | `results/eurlex-labelname/metrics.json` | §10 | `d599a29` |
| TC-5 | **Dictionary as a feature** — one-parameter z-score fusion | primary CV R@50 0.9171 vs 0.9168, **+0.0003** (CI includes 0), R-precision **−0.0204**, nDCG **−0.0190**; log-odds +0.0029/+0.0037; count +0.0010; stage 2 not licensed | `results/eurlex-fusion/metrics.json` (+ `scores/`) | §11 | `8bf5d5d` |
| TC-6 | **Calibrated zero-shot channel** — rank/quantile-map the title scores | **null**: best α=1 gives 9/96 unseen (0.0938) and −0.00046 overall; CV selects α=0; zero-shot unsolved | `results/eurlex-zeroshot-calibrated/metrics.json` | §12 | `235a3ac` |
| TC-7 | **Chunk-mean document embedding** (menu item 1) — 200-word non-overlapping chunks, mean-pooled, same k-NN | **null (negative)**: family_new R@50 **0.8887 vs 0.8944** (Δ **−0.0057 [−0.0094, −0.0020]**), R-precision −0.0192, nDCG@10 −0.0220; worse on every slice; 42% of 200-word chunks still exceed 256 tokens | `results/eurlex-embed-chunkmean/metrics.json` | §13 | this commit |
| TC-8 | **Multi-view embedding** (menu item 2) — the four fields as separate k-NN views; vector mean vs rank fusion | **primary null / secondary win**: view mean family_new 0.8942 vs 0.8944 (Δ −0.0002 [−0.0032, +0.0029]); **rank fusion 0.9055** (Δ **+0.0112 [+0.0080, +0.0145]**), R-precision +0.0220, nDCG@10 +0.0241, tail +0.05; `secondary_win_only`, **not adopted** | `results/eurlex-multiview/metrics.json` | §14 | this commit |
| TC-9 | **Lexical k-NN + RRF** (menu item 3) — TF-IDF (word 1–2 gram) k-NN fused with the embedding k-NN | **win**: fusion family_new **0.9165** vs 0.8944 (Δ **+0.0222 [+0.0195, +0.0250]**), R-precision +0.0560, nDCG@10 +0.0633; but the **lexical arm alone is 0.9138** (Δ +0.0194) with tail **0.5404** vs the fusion's 0.4830 — the fusion adds only +0.0028 over lexical and dilutes its tail | `results/eurlex-lexical-rrf/metrics.json` | §15 | this commit |
| TC-10 | **Label co-occurrence smoothing** (menu item 4) — top-10 labels vote for their train-only co-occurrence neighbours, α by family-grouped CV | **null**: the embedding base's CV selects **α=0 in all 5 folds** (Δ 0.0000); the grid falls monotonically (0.8944 → 0.8536 at α=1); the lexical base's best α=0.05 buys +0.0007 (CI touches 0) and loses ordering; the tail never rises | `results/eurlex-cooccurrence/metrics.json` | §16 | this commit |

## 5. What we learned

- **The embedding k-NN was the first shortlist generator — and the baseline to
  beat.** The random-split 0.917 is largely real: on genuinely new families it
  is **0.894 R@50**, and on the clean new-family slice **0.886**. TC-9's lexical
  k-NN later beats it on the honest slice (§15).
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
- **Chunk-mean is not the representation lever.** Replacing the whole-document
  view with 200-word chunk means loses R@50 on every slice (−0.0057 on
  `family_new`, interval entirely below zero) and loses ~0.02 R-precision and
  nDCG@10. The 200-word chunk is itself above the model's 256-token context on
  42% of inputs, so it does not even fully remove the truncation; the mean also
  dilutes the front-loaded title/header. The whole-document baseline stands.
- **Rank fusion of views is the first positive shortlist result, but it is a
  secondary-arm win.** The four fields are each worse alone than the
  concatenated document, yet fusing their rankings reaches `family_new`
  **0.9055 R@50** (+0.0112, interval excluding 0), +0.022 R-precision and
  +0.024 nDCG@10, and lifts the tail 0.4011 → 0.4511. Averaging the view
  vectors instead is flat — the same dilution failure as TC-7. Because the
  pre-registered primary (view mean) is a null, the item is
  `secondary_win_only` and the method is **not adopted** without the reviewer.
- **The lexical k-NN is the stronger single shortlist generator.** On the honest
  `family_new` slice TF-IDF reaches **0.9138 R@50** against the embedding's
  0.8944 (+0.0194) and **0.5404** tail recall against 0.4011; RRF with the
  embedding gives **0.9165** (+0.0222 over the embedding) but only +0.0028 over
  the lexical arm, and it *lowers* the tail to 0.4830 — equal-weight fusion
  favours the labels both views agree on, i.e. the head. On EURLEX, as on UPR,
  the words carry most of the shortlist signal.
- **Label co-occurrence smoothing is a null.** The train-only co-occurrence
  matrix's top-10 boost is a weaker ranking than the k-NN's own, so adding it
  only dilutes: family-grouped CV selects α = 0 on the embedding base, and the
  grid falls monotonically with α; the tail does not rise at any α. A PMI
  matrix or a learned combiner is untested, but the cheap additive form is out.

## 6. What we are going to try (the ordered menu)

This is the agreed order. The next session starts at **item 1** and does not
skip ahead. Each item is one change set: pre-register it (mechanism, metric,
falsification, cost), run it, write the artifact and the doc section, run
`ruff` + `pytest`, commit, report, stop. Every item is compared with the
current k-NN on `all_dev`, `family_new` and `family_new_clean`:

> **Baseline to beat:** k-NN `family_new` **0.8944 R@50 / 0.4869 R-precision /
> 0.6364 nDCG@10**; clean **0.8862 / 0.4744 / 0.6241**.

### Cheap / quick

1. **Better document embedding.** — **TC-7 executed 2026-11-13: NULL.** The
   200-word chunk-mean is worse than the whole-document baseline on every slice
   and loses 0.0192 R-precision / 0.0220 nDCG@10; do not continue this variant
   as-is (see `docs/topic_classification.md` §13). The longer-context-model half
   of the item was **not** run.
   - **What:** replace the 256-token MiniLM whole-document view with a
     **chunk-mean** embedding (split into ~200-word chunks, embed each,
     mean-pool, normalize) or a longer-context model (e5-base 512 tokens,
     bge-m3 / nomic 8k). Start with chunk-mean MiniLM — it uses the existing
     model and cache.
   - **Why:** the current model truncates the median 447-word document; the
     representation is the most likely lever.
   - **Evaluate:** dev / `family_new` / clean; R@50, R-precision, nDCG@10.
   - **Cost:** chunk-mean ~20–30 min CPU; a longer-context model 1–2 h CPU.
2. **Multi-view embedding.** — **TC-8 executed 2026-11-13: PRIMARY NULL, SECONDARY WIN (`secondary_win_only`).** Averaging the four view vectors is flat (family_new Δ R@50 −0.0002); **rank fusion** of the four per-view k-NNs reaches **0.9055 R@50** (Δ **+0.0112 [+0.0080, +0.0145]**), R-precision +0.0220 and nDCG@10 +0.0241, with the tail bucket +0.05. It is not adopted without the reviewer because the pre-registered primary arm failed (see `docs/topic_classification.md` §14).
   - **What:** embed title, header, recitals and main_body separately; combine
     them (mean or max) into one document vector, or keep separate k-NN views
     and fuse their rankings.
   - **Why:** the title/header are clean topic signals; the body carries the
     detail; the current truncated view mixes them.
   - **Evaluate:** same slices; compare with item 1.
   - **Cost:** cheap with cached MiniLM; needs its own cache version.
3. **Lexical / BM25 k-NN + reciprocal-rank fusion.** — **TC-9 executed 2026-11-13: WIN.** The RRF fusion clears the bar on every slice (`family_new` R@50 **0.9165**, Δ **+0.0222 [+0.0195, +0.0250]**, R-precision +0.0560, nDCG@10 +0.0633). The decomposition matters: the **TF-IDF k-NN alone is 0.9138** on `family_new` (above the embedding's 0.8944) with tail **0.5404** vs 0.4011, and the fusion adds only **+0.0028** R@50 over it while **reducing the tail to 0.4830** (see `docs/topic_classification.md` §15). Not adopted; the reviewer chooses.
   - **What:** build a TF-IDF/BM25 k-NN over train, take its top-50, and
     combine it with the embedding k-NN by **reciprocal-rank fusion** (no
     weight to tune).
   - **Why:** a different error profile; the UPR lexical route was nearly as
     good as embeddings; cheap.
   - **Evaluate:** RRF vs k-NN on all slices; report whether it helps the tail.
   - **Cost:** minutes.
4. **Label co-occurrence smoothing.** — **TC-10 executed 2026-11-13: NULL.** The family-grouped CV selects **α = 0 in every fold** on the embedding base; the full-dev grid falls monotonically with α (`family_new` 0.8944 → 0.8536 at α = 1), and the stronger lexical base's best α (0.05) buys +0.0007 R@50 (interval touches 0) while losing ordering and tail recall. The z-scored co-occurrence boost is a weaker version of the k-NN's own ranking (see `docs/topic_classification.md` §16).
   - **What:** train a label co-occurrence matrix; after the k-NN score, add
     `α · co-occurrence(top labels)`; tune α by family-grouped CV.
   - **Why:** labels co-occur; a strong head label can lift its neighbours;
     tail-oriented.
   - **Evaluate:** all slices, with the tail bucket called out.
   - **Cost:** minutes.
5. **Calibrated zero-shot channel.**
   - **What:** a **quota** (reserve N shortlist slots for zero-shot labels
     ranked by title cosine) or a learned calibrator — a different insertion
     rule than the raw replacement and quantile mapping that already failed.
   - **Why:** the title signal exists (0.868 zero-shot-only R@50) but does not
     survive the full ranking (TC-6 null).
   - **Evaluate:** unseen hits / 96; overall R@50 loss ≤ 0.01.
   - **Cost:** minutes.
6. **Cardinality / threshold calibration.**
   - **What:** turn scores into predicted label sets at a global (or
     per-bucket) threshold tuned to ~5 labels/doc; report precision, recall
     and predicted cardinality.
   - **Why:** product output; the shortlist is not the final label set.
   - **Evaluate:** predicted cardinality and set metrics; R@50 is unchanged.
   - **Cost:** minutes.

### Medium

7. **Cross-encoder rerank of the top-50.**
   - **What:** rerank the k-NN top-50 with a cross-encoder (cached ms-marco
     MiniLM or a legal-domain model) over (document, label title) pairs.
   - **Why:** target R-precision / nDCG@10; the k-NN ordering is the weak
     point, and the top-100 recall is 0.9473, so the shortlist ceiling is high.
   - **Evaluate:** rerank vs k-NN on the same shortlist.
   - **Cost:** 6k × 50 pairs; CPU minutes–hours.
8. **Learned candidate pack.**
   - **What:** union of the k-NN and dictionary top-50; features = k-NN score,
     dictionary arms, label frequency, label-name cosine, label co-occurrence,
     document length (and the cross-encoder score if item 7 exists); train a
     logistic / HistGB model on **train candidates** with a family-grouped
     cross-fit.
   - **Why:** the UPR “dictionary column” lived inside a learned pack, not a
     global weight; the TC-5 null does not rule out a learned combination.
   - **Evaluate:** held-out dev / `family_new`; must beat the k-NN by the
     agreed bar.
   - **Cost:** medium; needs its own pre-registration and a leak-free
     train-candidate construction.
9. **Cluster routing / label pruning.**
   - **What:** cluster train documents; route a dev document to its nearest
     clusters and rank only the labels represented there (or prune labels by
     cluster).
   - **Why:** efficiency at 4,271 labels; may improve tail precision by
     removing globally popular but irrelevant labels.
   - **Evaluate:** R@50 and cost against the full ranking.
   - **Cost:** cheap–medium.
10. **Hard-negative mining.**
    - **What:** use near-duplicate/template families as hard negatives for the
      pack (item 8), the cross-encoder (item 7) or the fine-tune (item 11);
      build a family-disjoint training split.
    - **Why:** makes the learned models learn the label, not the template; the
      audit already flags 26.1% near-duplicates.
    - **Evaluate:** as part of the model it feeds, not as a standalone
      shortlist.
    - **Cost:** medium.

### Expensive

11. **Fine-tune the embedding.**
    - **What:** contrastive training on train labels with hard negatives;
      evaluate on `family_new`.
    - **Why:** likely the biggest gain; the current model is generic.
    - **Evaluate:** same slices; one test shot later.
    - **Cost:** GPU / training run; needs explicit approval.
12. **LLM rerank / zero-shot label assignment.**
    - **What:** an LLM reranks the top-50 or picks labels from the titles.
    - **Why:** strongest ordering; handles ambiguity.
    - **Evaluate:** R-precision / nDCG and cost; latency reported.
    - **Cost:** high; needs explicit approval.

### Also valid

13. **Ship the current k-NN.** The 0.894 new-family baseline is strong; if the
    product does not need the tail or the zero-shot labels, stop and ship it.

## 7. Open questions

- Does a longer-context model (or a chunk size that fits the 256-token context,
  or a front-loaded title view) improve the **0.894** new-family baseline? The
  **chunked** half is answered — 200-word chunk-mean is a null (§13); the
  longer-context half is still open.
- Does the multi-view **rank-fusion** gain survive a different fusion rule,
  other `top_k`/`offset` values, or a different view set — and which views
  drive it? TC-8 is a secondary-arm win, not an adopted method (§14).
- Can a tail-aware fusion (weighted RRF, a view-specific weight, or a lexical
  term in a learned pack) keep the lexical tail gain while adding the
  embedding's breadth? TC-9's equal-weight RRF loses 0.057 tail recall
  against the lexical arm alone (§15).
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
- **M5 — representation** (item 1: **null**; item 2: primary **null**, rank
  fusion **secondary win**, not adopted): menu items 1–2.
- **M6 — candidate quality / rerank** (item 3: **win**, lexical arm strongest; item 4:
  **null**; items 5–12 open): menu items 3–12.
- **M7 — ship / final test** (open): menu item 13, then one test shot.

## 9. How the next session should proceed

The menu in **§6 is the order**. Start at **item 1**; do not skip ahead and do
not queue several items.

For every item:

1. Write the pre-registration in `docs/topic_classification.md`: the mechanism,
   the shape of the number, the falsification conditions, the cost, and the
   slices (`all_dev`, `family_new`, `family_new_clean`) it will be read on.
2. Run the experiment; write `results/eurlex-<detail>/metrics.json`.
3. Append the result table and the verdict to the same doc section.
4. Run `ruff check`, `ruff format --check`, and `pytest`; commit with one clear
   message; report and **stop**.
5. If the item is a null, record it and wait for the reviewer's word before the
   next item.

The current baseline to beat (from §5) is the k-NN on `family_new`: **0.8944
R@50 / 0.4869 R-precision / 0.6364 nDCG@10**. The first change set, **TC-7 =
menu item 1** (better document embedding), is **executed and null** (§13):
chunk-mean at 200 words loses 0.0057 R@50 on `family_new`. **TC-8 = menu item 2**
(multi-view embedding) is also executed: vector mean is flat, but **rank fusion**
clears the bar on `family_new` (+0.0112 R@50 [+0.0080, +0.0145], +0.022
R-precision, +0.024 nDCG@10) and is recorded as `secondary_win_only` — **not
adopted** without the reviewer. **TC-9 = menu item 3** (lexical k-NN + RRF) is also
executed and **wins its bar** (`family_new` +0.0222 R@50 [+0.0195, +0.0250]), but
the decomposition shows the lexical arm alone (0.9138) is the strongest single
retriever and the fusion adds only +0.0028 over it while diluting the tail. The
**TC-10 = menu item 4** (label co-occurrence smoothing) is also executed and is a
**null**: the family-grouped CV selects α = 0 on the embedding base and the grid
falls monotonically with α. The next change set is the reviewer's choice; menu
item 5 is not queued.

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
  `results/eurlex-fusion/`, `results/eurlex-zeroshot-calibrated/`,
  `results/eurlex-embed-chunkmean/`, `results/eurlex-multiview/`,
  `results/eurlex-lexical-rrf/`, `results/eurlex-cooccurrence/`.
- The old text-processing track (UPR/OMCT) is separate: `TEXT_PROCESSING_PLAN.md`
  and `docs/experiments_text_processing.md`. Do not mix its numbers with
  EURLEX.
