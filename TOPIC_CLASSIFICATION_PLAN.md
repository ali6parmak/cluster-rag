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
| label catalog (EUROVOC broader terms) | `catalogs/eurlex_labels.jsonl` (+ `.meta.json`; built from the official SKOS, TC-19) |
| fine-tuned embedding (TC-17) | `results/eurlex-embed-finetune/model/full/` (dev-tuned, not adopted) |
| TC-17 candidate ranking (TC-18) | `results/eurlex-embed-finetune/ranked_dev.npz` (the pack's top-64, exported deterministically) |

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
| TC-11 | **Zero-shot quota** (menu item 5) — reserve the bottom-5 slots of the top-50 for the best zero-shot labels by title cosine | **win**: unseen **34/96 (0.3542)** vs 0/96 for every seen-label arm, with all_dev R@50 −0.0060 and family_new −0.0081 (both inside the 0.01 cap); R-precision/nDCG@10 unchanged by construction; the lexical/RRF bases reach 47/96 at N=10 but fail the family_new guard (0.0121/0.0124) | `results/eurlex-zero-shot-quota/metrics.json` | §17 | this commit |
| TC-12 | **Cardinality / threshold calibration** (menu item 6) — global score threshold at the train mean cardinality vs fixed top-5 | **win (primary), base-dependent**: embedding threshold micro-F1 **0.4865 vs 0.4675** (Δ **+0.0190 [+0.0150, +0.0233]**), all_dev +0.0283, cardinality 5.071; the gain is micro-level (sample-F1/macro-F1 fall, tail stays 0); lexical top-5 beats its threshold (−0.0246) and RRF gains only +0.0062 (below bar) | `results/eurlex-threshold-calibration/metrics.json` | §18 | this commit |
| TC-13 | **Cross-encoder rerank of the lexical top-50** (menu item 7) — ms-marco MiniLM L-6 (primary) / L-12 (secondary), rerank within the top-50 | **null (strong negative)**: `family_new` nDCG@10 **0.6902 → 0.4360** (Δ **−0.2543 [−0.2626, −0.2463]**) and R-precision −0.2276 with L-6; L-12 is worse (**−0.3783**); R@50 unchanged (0.9138) — the rerank destroys the ordering, it does not improve it | `results/eurlex-cross-encoder-rerank/metrics.json` | §19 | this commit |
| TC-14 | **Learned candidate pack** (menu item 8) — out-of-fold train candidates, 11 features, HistGB; union of embedding/lexical/frequency top-50 | **win**: `family_new` R@50 **0.9278** vs lexical 0.9138 (Δ **+0.0141 [+0.0119, +0.0163]**), R-precision +0.0404, nDCG@10 +0.0440, tail 0.5404 → 0.5681; also beats RRF (Δ +0.0113); the z-scored dictionary arms were not features (scope) | `results/eurlex-candidate-pack/metrics.json` | §20 | this commit |
| TC-15 | **Cluster routing / label pruning** (menu item 9) — MiniBatchKMeans K=500 on train docs, route dev to the C nearest clusters, rank their labels first | **efficiency_only (primary)**: C=3 ranks ~250 of 4,197 labels (5.96 %) at `family_new` R@50 −0.0005 [−0.0025, +0.0014], R-precision +0.0073, nDCG@10 +0.0069, tail 0.5447; C=1 (2.38 % space, 91.4 % coverage) loses 0.0294 R@50 | `results/eurlex-cluster-routing/metrics.json` | §21 | this commit |
| TC-16 | **Hard-negative discipline** (menu item 10) — the pack trained with family-grouped folds (A) vs random folds (B) | **null**: B puts a same-family sibling in 27.7 % of documents' indices (A: 0 %), yet A beats B on `family_new` R@50 by only **+0.0006 [+0.00003, +0.0011]** — the discipline is not measurably needed, and the pack's gain is not template leakage (it is kept as the leak-free construction) | `results/eurlex-hard-negative-discipline/metrics.json` | §22 | this commit |
| TC-17 | **Fine-tune the embedding** (menu item 11) — five family-grouped fold models + one full-train MiniLM contrastive fine-tune (in-batch + base-kNN/family hard negatives); TC-14's pack with the fine-tuned embedding channel | **`ordering_win_only`**: the pack reaches `family_new` **0.9341 / 0.6181 / 0.7692** vs the control's 0.9278 / 0.5818 / 0.7343 — R@50 **+0.0062 [+0.0045, +0.0080]** (below the +0.01 bar), R-precision **+0.0363**, nDCG@10 **+0.0349**, tail 0.5681 → 0.5819; the fine-tuned k-NN is 0.9104 (beats the base MiniLM by +0.0160, still behind the lexical 0.9138); the control reproduces TC-14 exactly; 900 s GPU | `results/eurlex-embed-finetune/metrics.json` | §24 | this commit |
| TC-18 | **LLM rerank** (menu item 12) — one `tev1` decision call per dev document scoring the TC-17 pack's top-64 as `noul` questions, reranked within the 64 | **null (strong negative)**: `family_new` nDCG@10 **0.7670 → 0.5667** (Δ **−0.2002 [−0.2072, −0.1937]**), R-precision **−0.1940**, R@50 −0.0017; R@100 and coverage@64 are unchanged, so the failure is **ordering** — the second off-the-shelf reranker to collapse after TC-13; Stage 2 (label-blind) not run per the stop rule; 1.9 h, 6,000 calls | `results/eurlex-llm-rerank/metrics.json` | §25 | this commit |
| TC-19 | **EUROVOC parent channel** (off-menu) — the four immediate-`skos:broader` features (support, subtree rate, parent-title cosine, consistency) from the pinned official SKOS in the TC-14 pack (11 → 15), plus a parent-aware zero-shot quota | **null**: `family_new` R@50 **0.9287 vs 0.9278** (Δ **+0.0009 [+0.0003, +0.0015]**, far below the +0.01 bar), R-precision **+0.0123**, nDCG@10 **+0.00945** (just under the +0.01 ordering bar), tail 0.5681 → 0.5723; the **quota leg gains** **34 → 37/96** at an all-dev R@50 loss of 0.0060 (secondary, does not decide); catalog reproduces §23's join exactly; CPU 568 s | `results/eurlex-parent-channel/metrics.json` | §23 | this commit |

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
- **The zero-shot quota solves the zero-shot shortlist.** Reserving the bottom 5
  slots of the top-50 for the best zero-shot titles puts **34/96** dev
  assignments on the shortlist (0.3542), against 0/96 for every seen-label
  arm and 9/96 for the best calibrated blend. It costs 0.0060 all-dev R@50
  and 0.0081 `family_new` R@50 (inside the 0.01 cap) and leaves R-precision
  and nDCG@10 untouched; the price is the tail (0.4011 → 0.3617). The
  stronger lexical base reaches 47/96 but breaches the `family_new` cap.
- **The cardinality threshold helps the embedding output, but only at the micro
  level, and it is base-dependent.** A global threshold at the train mean
  (5.074) lifts `family_new` micro-F1 **+0.019** over top-5 on the embedding
  base (all_dev +0.028), at 5.071 labels/doc, but sample-F1 and macro-F1 fall
  and the per-document cardinality error grows (MAE 1.27 → 2.13); the tail
  still gets zero predictions. On the lexical ranking fixed top-5 is better
  (−0.025), and RRF gains only +0.006. The decision rule interacts with the
  ranking; no single set predictor is chosen.
- **An off-the-shelf cross-encoder is not a usable reranker here.** Reranking the
  lexical top-50 with ms-marco MiniLM **halves** the top-10 hit rate: nDCG@10
  0.6902 → 0.4360 (`family_new`, Δ −0.2543), R-precision −0.2276, and the
  larger L-12 is worse (−0.3783). The candidate set does not move (R@50
  0.9138), so the failure is purely ordering, and the role mismatch (the long
  document as query, the short label as passage) is the plausible cause. A
  task-fitted reranker (NLI or fine-tuned) is a different pre-registration;
  menu item 11 (fine-tune) needs approval.
- **The learned candidate pack is the track's best shortlist.** A per-candidate
  HistGB over 11 features (embedding / lexical / RRF scores, label rate,
  label-name cosine, document length, bucket, retriever ranks), trained on
  out-of-fold train candidates, reaches **0.9278 R@50 / 0.5818 R-precision /
  0.7343 nDCG@10** on `family_new` — +0.0141 R@50, +0.0404 R-precision and
  +0.0440 nDCG@10 over the strongest single retriever, with the tail rising
  too (0.5404 → 0.5681). Unlike the global combinations (TC-5, TC-10), the
  learned pack can use the retrievers' disagreement. The dictionary arms are
  not in it (cost); a dictionary-in-pack ablation is the open follow-up.
- **Cluster routing is a cost lever, not a metric lever.** Routing to the 3
  nearest clusters ranks ~6 % of the labels with the `family_new` shortlist
  unchanged (R@50 −0.0005) and a small ordering gain (+0.007); a single
  cluster (2.4 % of the space) loses 0.029 R@50. Useful for a label-side
  budget, not a new best.
- **The hard-negative discipline is a null (and the pack is not template
  leakage).** Training the pack with 27.7 % of documents seeing a same-family
  sibling in their retrieval index moves `family_new` R@50 by 0.0006; if the
  model had leaned on the template it would have inflated the training rows and
  collapsed on `family_new`. The family-grouped split stays as the leak-free
  construction. Hard negatives for a fine-tune are untested.
- **Fine-tuning lifts the embedding channel, but the pack's gain is an
  ordering win, not a shortlist win.** Six MiniLM fine-tunes (five family-grouped
  fold models + one full-train, in-batch + hard negatives) raise the embedding
  k-NN **+0.0160 R@50** over the base model (0.9104 vs 0.8944 on `family_new`,
  interval excluding 0) and the pack that consumes the fine-tuned channel
  **+0.0062 R@50 / +0.0363 R-precision / +0.0349 nDCG@10** over the
  TC-14 control (0.9341 vs 0.9278), with the tail 0.5681 → 0.5819. The R@50
  gain misses the +0.01 bar, so the outcome is **`ordering_win_only`**:
  fine-tuning sharpens the top of the ranking but buys only ~40 % of the
  breadth gain the pack itself bought over the lexical arm, and the fine-tuned
  embedding is still behind the lexical k-NN as a single retriever (0.9104 vs
  0.9138). Not adopted.
- **The EUROVOC hierarchy is a precision/ordering feature, not a breadth
  feature.** The four parent features lift the pack's `family_new`
  R-precision **+0.0123** and nDCG@10 **+0.00945** (both intervals excluding 0)
  and the tail 0.5681 → 0.5723, but R@50 only **+0.0009** and coverage not at
  all: a shortlist that already holds a parent tends to hold its children, so
  the hierarchy re-ranks more than it widens. Both bars miss (R@50 +0.01,
  nDCG +0.01 by a hair), so the item is a **null**. Where the parent is the
  only evidence — the zero-shot quota — it does add **3/96** unseen
  assignments (34 → 37) at an all-dev R@50 loss of 0.0060; that is a positive
  secondary, not an adoption.
- **An off-the-shelf decision model is not a reranker here either.** `tev1` on
  the TC-17 pack's top-64 collapses `family_new` nDCG@10 **0.7670 → 0.5667**
  (Δ **−0.2002**) and R-precision **−0.1940** while R@50 barely moves
  (−0.0017) and R@100 is identical: the shortlist is unchanged, the ordering is
  destroyed — the same signature as TC-13's cross-encoder. The 4B binary
  decision model has no topical-ranking training; its probability tracks label
  breadth/title shape more than the document. The pre-registered label-blind
  control was **not run** because the bar failed. The pack's own ordering
  remains the best.

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
5. **Calibrated zero-shot channel.** — **TC-11 executed 2026-11-13: WIN (the quota).** Reserving the bottom **5** of the top-50 for the best zero-shot titles puts **34/96** unseen assignments on the shortlist (0.3542) at an all_dev R@50 cost of 0.0060 and a `family_new` cost of 0.0081 — inside the 0.01 cap, with R-precision/nDCG@10 unchanged by construction. The lexical/RRF bases reach 47/96 at N=10 but cost 0.0121/0.0124 on `family_new` and are not wins. Not adopted; the reviewer chooses (see `docs/topic_classification.md` §17).
   - **What:** a **quota** (reserve N shortlist slots for zero-shot labels
     ranked by title cosine) or a learned calibrator — a different insertion
     rule than the raw replacement and quantile mapping that already failed.
   - **Why:** the title signal exists (0.868 zero-shot-only R@50) but does not
     survive the full ranking (TC-6 null).
   - **Evaluate:** unseen hits / 96; overall R@50 loss ≤ 0.01.
   - **Cost:** minutes.
6. **Cardinality / threshold calibration.** — **TC-12 executed 2026-11-13: WIN (primary), base-dependent.** A global threshold at the train mean cardinality (5.074) beats top-5 on the embedding base: `family_new` micro-F1 **0.4865 vs 0.4675** (Δ **+0.0190 [+0.0150, +0.0233]**), all_dev +0.0283, realised cardinality 5.071. It is a **micro-level** gain (sample-F1 0.466 → 0.455, macro-F1 0.156 → 0.136, cardinality MAE 1.27 → 2.13, tail still 0), and it does not generalise across bases: the lexical top-5 beats its threshold (−0.0246) and RRF gains only +0.0062. Not adopted; the reviewer chooses (§18).
   - **What:** turn scores into predicted label sets at a global (or
     per-bucket) threshold tuned to ~5 labels/doc; report precision, recall
     and predicted cardinality.
   - **Why:** product output; the shortlist is not the final label set.
   - **Evaluate:** predicted cardinality and set metrics; R@50 is unchanged.
   - **Cost:** minutes.

### Medium

7. **Cross-encoder rerank of the top-50.** — **TC-13 executed 2026-11-13: NULL (strong negative).** The lexical top-50 reranked by ms-marco MiniLM loses **0.2543 nDCG@10** on `family_new` (0.6902 → 0.4360, interval entirely below zero) and 0.2276 R-precision; the bigger L-12 is worse (−0.3783). R@50 is unchanged by construction. The off-the-shelf web-relevance cross-encoder is unusable in the document-as-query / label-as-passage role assignment (see `docs/topic_classification.md` §19); a task-fitted reranker would be a new, more expensive pre-registration.
   - **What:** rerank the k-NN top-50 with a cross-encoder (cached ms-marco
     MiniLM or a legal-domain model) over (document, label title) pairs.
   - **Why:** target R-precision / nDCG@10; the k-NN ordering is the weak
     point, and the top-100 recall is 0.9473, so the shortlist ceiling is high.
   - **Evaluate:** rerank vs k-NN on the same shortlist.
   - **Cost:** 6k × 50 pairs; CPU minutes–hours.
8. **Learned candidate pack.** — **TC-14 executed 2026-11-13: WIN.** The HistGB pack trained on out-of-fold train candidates beats the lexical k-NN on `family_new` by **+0.0141 R@50** (0.9278 vs 0.9138, interval excluding 0), **+0.0404 R-precision** and **+0.0440 nDCG@10**, and lifts the tail 0.5404 → 0.5681; it also beats the RRF fusion (+0.0113 R@50). The menu's z-scored dictionary arms are **not** features (the pre-registered scope decision); the pack's gain comes from the two retrievers, the frequency prior and the label metadata. Not adopted; the reviewer chooses (§20).
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
9. **Cluster routing / label pruning.** — **TC-15 executed 2026-11-13: `efficiency_only` (not a shortlist win).** Routing to the **3** nearest clusters ranks ~250 of 4,197 labels (5.96 %) with the `family_new` shortlist unchanged (R@50 −0.0005 [−0.0025, +0.0014]) and a small ordering gain (R-precision +0.0073, nDCG@10 +0.0069); the tail rises 0.5404 → 0.5447. C=1 (2.38 % of the space, 91.4 % gold coverage) loses 0.0294 R@50. A cost lever, not a metric lever; not adopted (§21).
   - **What:** cluster train documents; route a dev document to its nearest
     clusters and rank only the labels represented there (or prune labels by
     cluster).
   - **Why:** efficiency at 4,271 labels; may improve tail precision by
     removing globally popular but irrelevant labels.
   - **Evaluate:** R@50 and cost against the full ranking.
   - **Cost:** cheap–medium.
10. **Hard-negative mining.** — **TC-16 executed 2026-11-13: NULL.** The TC-14 pack
    already uses the family-disjoint split, so the item became the discipline's
    ablation: the same pack trained with **random** folds (27.7 % of documents get a
    same-family sibling in their index) is only **0.0006 R@50** behind on
    `family_new` — far below the 0.01 bar. The discipline is not measurably needed at
    this feature set, but it is kept as the leak-free construction, and the pack's
    gain is **not** template leakage. Hard negatives for a fine-tune (item 11) remain
    untested (§22).
    - **What:** use near-duplicate/template families as hard negatives for the
      pack (item 8), the cross-encoder (item 7) or the fine-tune (item 11);
      build a family-disjoint training split.
    - **Why:** makes the learned models learn the label, not the template; the
      audit already flags 26.1% near-duplicates.
    - **Evaluate:** as part of the model it feeds, not as a standalone
      shortlist.
    - **Cost:** medium.

### Expensive

11. **Fine-tune the embedding.** — **TC-17 executed 2026-11-14: `ordering_win_only`.** Five family-grouped fold models plus one full-train MiniLM contrastive fine-tune (in-batch + hard negatives). The fine-tuned k-NN beats the base MiniLM by **+0.0160 R@50** (0.9104 vs 0.8944) but is still behind the lexical arm (0.9138, −0.0034); the TC-14 pack with the fine-tuned channel gains **+0.0062 R@50** (below the +0.01 bar), **+0.0363 R-precision** and **+0.0349 nDCG@10** (0.9341 / 0.6181 / 0.7692 on `family_new`), tail 0.5681 → 0.5819. The control reproduces TC-14 exactly; 900 s GPU. Not adopted; the reviewer chooses (§24).
    - **What:** contrastive training on train labels with hard negatives;
      evaluate on `family_new`.
    - **Why:** likely the biggest gain; the current model is generic.
    - **Evaluate:** same slices; one test shot later.
    - **Cost:** GPU / training run; needs explicit approval.
12. **LLM rerank / zero-shot label assignment.** — **TC-18 executed 2026-11-14: null (strong negative).** `tev1` re-scored the TC-17 pack's top-64, one call per document: `family_new` nDCG@10 **0.7670 → 0.5667** (Δ **−0.2002 [−0.2072, −0.1937]**), R-precision **−0.1940**, R@50 −0.0017 — the candidate set is unchanged (R@100 identical), so it is a pure ordering failure, the same signature as TC-13. The pre-registered label-blind Stage 2 was not run (the bar failed). Not adopted; the reviewer decides (§25).
    - **What:** an LLM reranks the top-50 or picks labels from the titles.
    - **Why:** strongest ordering; handles ambiguity.
    - **Evaluate:** R-precision / nDCG and cost; latency reported.
    - **Cost:** high; needs explicit approval.

### Also valid

13. **Ship the current k-NN.** The 0.894 new-family baseline is strong; if the
    product does not need the tail or the zero-shot labels, stop and ship it.

### Off-menu (pre-registered, not run)

**TC-19 — the EUROVOC parent channel** (pre-registered 2026-11-14, detailed in
`docs/topic_classification.md` §23). Add the official EUROVOC `skos:broader`
hierarchy to the pack as four features — parent support, subtree rate,
parent-title cosine and parent consistency — through a committed
`catalogs/eurlex_labels.jsonl`; a secondary leg ranks the zero-shot quota by
parent evidence (baseline 34/96). Two arms, same folds: TC-14's 11-feature pack
vs the same pack + 4 parent features. A win needs `family_new` R@50 +0.01 over
the 11-feature pack, interval excluding 0, no R-precision/nDCG loss beyond 0.01;
otherwise null. LLM-free (the English definitions are recorded but unused),
~20 min CPU. **Status: executed 2026-11-14 — null** (R@50 +0.0009, nDCG@10
+0.00945, both under the 0.01 bars; the quota leg gains 34 → 37/96). §23.

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
- The learned pack **does** beat the k-NN by the shortlist bar (§20). Which
  features carry its gain (the retriever scores, the label rate, the name cosine,
  the ranks) is **not measured yet** — a feature ablation is the open follow-up,
  as is a dictionary-in-pack variant (the z-scored arms were omitted for cost).
- The **quota** answers the zero-shot question for the shortlist (34/96, §17);
  is a learned calibrator or a learned insertion ranker worth it, and can the
  quota's tail cost (0.4011 → 0.3617 on `family_new`) be recovered without
  giving back the unseen gain?
- The fine-tune's pack gain is an **ordering** gain (+0.0349 nDCG@10) with a
  sub-bar breadth gain (+0.0062 R@50, §24). Is `ordering_win_only` enough to
  adopt a GPU-trained embedding, and would more training (more epochs, hard
  negatives mined with the fine-tuned model, a larger base model) clear the
  +0.01 shortlist bar — or is the pack's remaining breadth lexical?
- **Two off-the-shelf rerankers collapse the pack's ordering** (TC-13's
  cross-encoder, TC-18's `tev1`). Is the top of the pack's ranking improvable at
  all by a zero-shot scorer, or would it need a reranker fine-tuned on
  (document, label) pairs — and is that worth a GPU run?
- Is `family_new` the final evaluation protocol, and what is the one **test**
  shot for an adopted config?
- The vocabulary ships **titles only**. **TC-19** tested the official EUROVOC
  hierarchy (§23): it is a precision/ordering feature, not a breadth one
  (+0.0009 R@50, +0.00945 nDCG@10 — a null), with a positive quota secondary
  (34 → 37/96). The description channel stays open: official definitions cover
  15 % (660 labels), and an LLM-written catalog (UPR CS-11 style) would need
  approval; is a description feature worth testing on the 660 covered labels?
- The dev-evaluable label space is 4,197; the dataset union is 4,271. Keep the
  74 test-only labels excluded from dev, or include them as floored columns?

## 8. Milestones

- **M1 — audit** (done): dataset, label space, leakage.
- **M2 — cheap shortlist baselines** (done): dictionary vs k-NN, metrics.
- **M3 — trustworthy evaluation** (done): family-held-out and clean slices.
- **M4 — zero-shot attempts** (done): label-name and calibrated channel nulls;
  the **quota wins** (34/96, §17).
- **M5 — representation** (item 1: **null**; item 2: primary **null**, rank
  fusion **secondary win**, not adopted): menu items 1–2.
- **M6 — candidate quality / rerank** (item 3: **win**, lexical arm strongest; item 4:
  **null**; item 5: **win**, zero-shot quota; item 6: **win (primary), base-dependent**;
  item 7: **null (strong negative)**; item 8: **win**, the track's best shortlist;
  item 9: **efficiency_only**; item 10: **null**, the discipline is not needed;
  item 11: **`ordering_win_only`**, the fine-tune sharpens ordering but misses
  the shortlist bar; item 12: **null**, the decision model collapses the
  ordering; off-menu TC-19: **null**, the parent channel is precision-only):
  menu items 3–12.
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
falls monotonically with α. **TC-11 = menu item 5** (zero-shot quota) is also
executed and **wins**: the bottom-5 quota puts **34/96** unseen assignments on the
shortlist at an all-dev R@50 cost of 0.0060 and a `family_new` cost of 0.0081,
inside the 0.01 cap. **TC-12 = menu item 6** (cardinality / threshold calibration)
is also executed and **wins on the primary base**: the global threshold lifts
`family_new` micro-F1 by **+0.019** over top-5 (all_dev +0.028) at 5.071
labels/doc, but the gain is micro-level and does not generalise to the lexical
base. **TC-13 = menu item 7** (cross-encoder rerank) is also executed and is a
**strong negative**: ms-marco halves the ordering (nDCG@10 0.6902 → 0.4360 with
L-6, worse with L-12) while leaving R@50 unchanged, so the off-the-shelf
web-relevance model is not a usable reranker in this role assignment. The next
change set is the reviewer's choice. **TC-14 = menu item 8** (learned candidate
pack) is also executed and **wins**: the out-of-fold HistGB pack reaches
**0.9278 R@50 / 0.5818 R-precision / 0.7343 nDCG@10** on `family_new`, beating the
lexical k-NN by +0.0141 R@50, +0.0404 R-precision and +0.0440 nDCG@10, with the
tail up (0.5404 → 0.5681). It is the track's best shortlist so far, but it is a
dev-tuned model and not adopted. **TC-15 = menu item 9** (cluster routing) is also
executed and is an **`efficiency_only`** result: 3-cluster routing ranks ~6 % of
the labels with the shortlist unchanged (R@50 −0.0005) and +0.007 ordering.
**TC-16 = menu item 10** (hard-negative discipline) is also executed and is a
**null**: training the pack with random folds (27.7 % template exposure) costs only
0.0006 R@50 on `family_new`, so the family-disjoint split is not measurably needed
— but it is kept, and the pack's gain is not template leakage. **TC-17 = menu item
11** (fine-tune the embedding) is also executed and is an **`ordering_win_only`**:
the pack with the fine-tuned embedding reaches **0.9341 / 0.6181 / 0.7692** on
`family_new` against the control's 0.9278 / 0.5818 / 0.7343 — R@50 **+0.0062**
(below the +0.01 bar), R-precision **+0.0363**, nDCG@10 **+0.0349**, tail
0.5681 → 0.5819 — and the fine-tuned k-NN (0.9104) beats the base MiniLM
(+0.0160) but not the lexical arm (0.9138); 900 s GPU. The next change set is
the reviewer's choice; menu item 12 (LLM rerank) needs explicit cost approval.
**TC-19** (off-menu, the EUROVOC parent channel) is also executed and is a
**null** (`family_new` R@50 +0.0009, nDCG@10 +0.00945 — both bars miss; the
parent-aware quota gains 34 → 37/96 as a positive secondary, §23).
**TC-18 = menu item 12** (the LLM rerank) is also executed and is a **null
(strong negative)**: `tev1` collapses the TC-17 pack's ordering (nDCG@10
0.7670 → 0.5667, R-precision −0.1940), with the candidate set unchanged — the
pre-registered label-blind Stage 2 was not run. The menu is now exhausted
except **item 13: ship / final test**, which is the reviewer's decision: choose
the final config (the TC-17 pack is the best shortlist; the TC-11 quota is the
zero-shot route), then pre-register the **one** test shot and run it. The test
split has never been read.

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
  `results/eurlex-lexical-rrf/`, `results/eurlex-cooccurrence/`,
  `results/eurlex-zero-shot-quota/`, `results/eurlex-threshold-calibration/`,
  `results/eurlex-cross-encoder-rerank/`, `results/eurlex-candidate-pack/`,
  `results/eurlex-cluster-routing/`, `results/eurlex-hard-negative-discipline/`,
  `results/eurlex-embed-finetune/`, `results/eurlex-parent-channel/`,
  `results/eurlex-llm-rerank/`.
- The old text-processing track (UPR/OMCT) is separate: `TEXT_PROCESSING_PLAN.md`
  and `docs/experiments_text_processing.md`. Do not mix its numbers with
  EURLEX.
