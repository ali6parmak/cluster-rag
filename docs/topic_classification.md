# Topic classification — the EURLEX-4K track

**Status: data audit (change set 2), 2026-11-12.** The numbers below are measured
from the files in `data/eurlex_4k/` and copied from
`results/eurlex-stats/stats.json`, written by
`experiments_topic_classification/audit_eurlex.py`. The data is raw and
read-only; nothing here re-splits or edits it. The audit does **not** produce a
model metric.

The question this track asks is the product question: with ~4,271 labels, can a
cheap label-side signal (the corpus dictionary) put the right labels on a
shortlist, and how does that compare with an embedding k-NN? Micro-F1@2 is the
wrong instrument at this label count; the experiment will use recall@k,
precision@k, nDCG@k and predicted cardinality, tuned on validation.

## 1. Dataset and schema

| split | documents | Regulation | Decision | Directive |
|---|---:|---:|---:|---:|
| train | 45,000 | 29,600 | 13,627 | 1,773 |
| dev | 6,000 | 3,964 | 1,819 | 217 |
| test | 6,000 | 3,967 | 1,793 | 240 |
| **total** | **57,000** | **37,531** | **17,239** | **2,230** |

Every row has the same seven keys and the same types:

| key | type | notes |
|---|---|---|
| `celex_id` | string | unique across the three splits (0 duplicates) |
| `document_type` | string | Regulation / Decision / Directive |
| `title` | string | never empty |
| `header` | string | never empty; repeats the title in 65.7% of documents |
| `recitals` | string | two empty; the longest field by character count |
| `main_body` | list of strings | paragraphs, not a string; never empty |
| `eurovoc_concepts` | list of strings | numeric or alphanumeric label IDs; no duplicate IDs within a document |

The three files are the only split files; there is no separate id list or split
manifest.

## 2. Label space, vocabulary join, and support

The vocabulary `data/eurlex_4k/eurovoc_concepts.jsonl` is a **superset** of the
labels used by the splits:

| fact | value |
|---|---:|
| vocabulary rows | 7,201 |
| distinct vocabulary IDs | 7,201 (0 duplicates) |
| vocabulary rows with a title | 7,201 (0 empty) |
| vocabulary IDs alphanumeric (e.g. `c_4b33c289`) | 235 |
| **distinct IDs used by the splits** | **4,271** |
| **matched to the vocabulary** | **4,271** |
| **unmatched IDs** | **0** |
| labels with a non-empty title | 4,271 |
| vocabulary IDs not used by the splits | 2,930 |
| alphanumeric IDs used by the splits | 1 (`c_871b5612`, "staff regulations (EU)"; 24 documents) |

The join is on **strings**, so `3474` and `c_871b5612` are both keys; the
superset is expected and is not a mismatch. Every split ID has a title.

**Support** here means the number of **train** documents that carry the label.
The audit's working categories are:

- **frequent**: train support **≥ 10** — 1,841 labels;
- **few-shot**: train support **1–9** — 2,267 labels;
- **zero-shot**: train support **0**, present in dev/test — 163 labels.

For continuity with the earlier plan, the same support gives head / torso /
tail / unseen as **441 / 1,400 / 2,267 / 163**. The full support histogram:

| train support | labels |
|---|---:|
| 0 | 163 |
| 1 | 709 |
| 2–4 | 865 |
| 5–9 | 693 |
| 10–19 | 556 |
| 20–49 | 539 |
| 50–99 | 305 |
| 100–499 | 347 |
| 500+ | 94 |
| **total** | **4,271** |

So **709 labels have exactly one train document**, 1,097 have at most two, and
2,267 have fewer than ten. The head is thin: 441 labels carry most of the mass,
and the top label (`import`) has 3,645 train positives.

### Distribution on the splits

| split | distinct labels | frequent | few-shot | zero-shot | docs touched by frequent | docs touched by few-shot | docs touched by zero-shot | label assignments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 4,108 | 1,841 | 2,267 | 0 | 44,986 | 5,829 | 0 | 228,322 |
| dev | 2,515 | 1,708 | 718 | 89 | 5,996 | 785 | 87 | 30,240 |
| test | 2,517 | 1,719 | 713 | 85 | 5,995 | 775 | 89 | 30,356 |

The "docs touched" columns overlap (a document can carry labels from more than
one category), so they do not sum to the split size. Zero-shot labels are rare
in assignment terms — 96 of 30,240 dev assignments (0.3%) and 96 of 30,356 test
assignments — but they are the labels a seen-label method cannot retrieve at
all.

## 3. Cardinality

| split | mean | median | p90 | p95 | p99 | max | documents with 1 label |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5.074 | 5 | 7 | 8 | 10 | 26 | 239 |
| dev | 5.040 | 5 | 7 | 8 | 10 | 16 | 25 |
| test | 5.059 | 5 | 7 | 8 | 10 | 19 | 43 |
| **overall** | **5.069** | **5** | **7** | **8** | **10** | **26** | **307** |

The distribution is tightly centred (3–6 labels cover most documents); 766
documents carry 10 or more labels. A top-2 prediction is therefore far below
the mean cardinality — another reason recall@k and predicted cardinality are
the right headline.

## 4. Text and fields

| field | chars mean / median / p95 | words mean / median | share of characters |
|---|---:|---:|---:|
| title | 219 / 205 / 358 | 36 / 34 | 6.0% |
| header | 284 / 270 / 437 | 47 / 44 | 7.8% |
| recitals | 1,999 / 1,489 / 5,048 | 327 / 243 | 55.0% |
| main_body | 1,135 / 536 / 4,445 | 190 / 91 | 31.2% |
| **whole document** | **3,640 / 2,693 / 9,722** | **600 / 447** | **100%** |

The whole document's maximum is 21,641 characters; `main_body` has a mean of
3.03 paragraphs (median 2, p95 8, max 17). The content is mostly recitals and
main body; the title and header are short and partly duplicated. The longest
inputs will need chunking or truncation, but the median document is only ~2,700
characters, so a single whole-text embedding is a reasonable first baseline.

## 5. Leakage

The splits are random, and EU legal acts are formulaic. The audit measures three
kinds of overlap, all on the **provided** splits (nothing is re-split).

### Exact duplicates

| field | duplicate groups | cross-split groups | documents with a cross-split twin |
|---|---:|---:|---:|
| title | 23 | 10 | 25 |
| header | 17 | 4 | 8 |
| recitals | 943 | 549 | **8,566** (15.0%) |
| main_body | 704 | 361 | **3,804** (6.7%) |
| whole document | 0 | 0 | 0 |

There is **no exact whole-document duplicate across splits**, but 15% of
documents share an identical recitals block with a document in another split,
and 6.7% share an identical main body. These are recurring instruments
(standard import values, export refunds, fishing prohibitions) that differ in
dates, numbers or a final article.

### Near duplicates

TF-IDF cosine, train as the index, each dev/test document's best train match:

| threshold | full text (dev/test docs) | title only |
|---|---:|---:|
| ≥ 0.95 | 735 (6.1%) | 79 (0.7%) |
| ≥ 0.90 | **3,135 (26.1%)** | 661 (5.5%) |
| ≥ 0.80 | 6,141 (51.2%) | 4,394 (36.6%) |
| ≥ 0.70 | 7,888 (65.7%) | 8,154 (68.0%) |

A quarter of dev/test documents have a train neighbour at cosine ≥ 0.90 on the
whole text. The examples are template families, e.g. `32012R0003` ↔
`32012R0008` (cosine 1.0) and `32011R0001` ↔ `32011R0007`. Title-only near
duplicates are rarer (5.5% at ≥ 0.90), so the boilerplate — not the title — is
what makes them close.

A title-skeleton grouping (digits replaced by `#`) confirms the scale:
**12,371 documents (21.7%)** sit in **2,056** template families that span
splits, including the standard-import-value and fishing-prohibition families.

### Citation / amendment links

- 1,945 documents (3.4%) cite at least one document held in **another** split,
  across 1,961 citation edges: train→dev 457, train→test 475, dev→train 465,
  test→train 446, dev→test 58, test→dev 60.
- 16,827 titles contain "amending", 5,533 "implementing", 1,039 "repealing";
  971 / 275 / 103 of those respectively also cite a cross-split document.
- `celex_id` itself is unique across splits (0 duplicates), so a CELEX-based
  split audit would not have caught the template overlap.

**Consequence for evaluation.** The random split is not a clean "new document"
test. A shortlist number on dev will be optimistic for the ~26% of documents
with a ≥ 0.90 full-text neighbour in train. The first experiment should report
the overall number **and** a "clean" slice (best train cosine < 0.90), and the
audit's near-duplicate score should travel with every dev number. A future
document-level or family-level split is the cleaner fix; it is not part of this
change set.

## 6. Sample documents

| split | CELEX | type | cardinality | labels (id — title) |
|---|---|---|---:|---|
| train | `32014R0727` | Regulation | 7 | 1402 export industry · 2771 originating product · 3191 import (EU) · 5055 organic acid · 519 anti-dumping duty · 5969 China · 5971 chlorine |
| train | `32006D0201` | Decision | 4 | 1355 Indonesia · 1474 agreement (EU) · 5744 fact-finding mission · c_871b5612 staff regulations (EU) |
| dev | `31999R1400` | Regulation | 5 | 1565 milk · 2443 common agricultural policy · 2644 intervention price · 2676 target price · 4860 butter |
| test | `32011D0690` | Decision | 17 | 1166 Ghana · 1338 India · 1445 veterinary inspection · 1644 import licence · 1729 Madagascar · 1841 Mauritius · 1907 honey · 2121 egg · 2531 food contamination · 2718 fishery product · 3191 import (EU) · 4164 Singapore · 4580 Turkey · 4747 live poultry · 4841 Belize · 6268 surveillance concerning imports · 6569 food safety |
| dev | `32005R1139` | Regulation | 3 | 2067 budgetary classification · 2658 flavouring · 2948 aromatic plant |

Titles: `32014R0727` is an anti-dumping "new exporter" review; `32006D0201`
extends the EU monitoring mission agreement with Indonesia (and carries the
alphanumeric label); `31999R1400` fixes milk and butter intervention prices;
`32011D0690` amends the third-country residue-plan annex; `32005R1139`
classifies goods in the Combined Nomenclature.

## 7. What this means for the first experiment

1. **The label side is now usable.** All 4,271 used labels have titles; there
   are no descriptions. That makes a label-name embedding possible (a zero-shot
   route for the 163 labels with no train example), but the first experiment
   stays with the two cheap shortlist generators the plan asked for: the
   train-only corpus dictionary and an embedding k-NN.
2. **Both cheap generators are "seen-label" methods.** A label with zero train
   documents cannot be retrieved by a k-NN over train documents or by a
   dictionary built from train documents. Report the 163 zero-shot labels
   separately; their 96 dev assignments are the ceiling neither method can
   reach.
3. **Evaluate the shortlist, not top-2.** The product question is "is the right
   label in the top-k?"; report recall@50, precision@10, nDCG@50, coverage and
   predicted cardinality, by head / torso / tail / unseen.
4. **Carry the leakage score.** Report the dev numbers overall and on the
   clean slice (best train cosine < 0.90); the audit's near-duplicate rate is
   part of the result, not a footnote.
5. **Split roles.** `train` is the index/fit split; `dev` is the
   tuning/product-like split; `test` stays held out. The audit does not re-split.

## 8. First experiment — dictionary vs embedding k-NN shortlist

**Pre-registered 2026-11-13, before the run.** Artifact:
`results/eurlex-shortlist/metrics.json`; script:
`experiments_topic_classification/shortlist_baselines.py`.

**Question.** Can the train-only corpus dictionary produce a useful shortlist
at 4,271 labels, and how does it compare with a cheap embedding k-NN?

**Split.** `train` is the index/fit split (45,000 documents), `dev` the
evaluation and tuning split (6,000), `test` held out. No resplit, no test.

**Methods.**

1. **Dictionary** — the corpus map of the train split (`build_corpus_map`,
   imported from `text_processors/`, not copied). Primary arm
   `sum_zscore_positive`: sum the map's z-scores over the document's content
   words, keeping only positive evidence. Secondary arms `sum_logodds` and
   `support_count` (the measure-free control). A label with no train example
   carries the row floor, so the 163 zero-shot labels cannot enter a shortlist.
2. **Embedding k-NN** — MiniLM-L6-v2 (revision
   `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`), the whole document text,
   `document` kind, cosine, `k = 100` neighbours, similarity-weighted label
   votes (power 1.0), via the pipeline's `knn_vote_scores`. The model's context
   is 256 tokens, so it truncates long documents; the dictionary reads the
   whole text.
3. **Frequency prior** — rank labels by their train document frequency, the
   same ranking for every document; the label-blind floor.

**Metrics.** Primary: **recall@50** (macro over documents), **R-precision**
(macro precision at `|gold|`), **nDCG@10** (binary relevance). Secondary:
recall@5, recall@10, recall@20, recall@100, nDCG@50. Diagnostics: coverage@50
(share of documents with at least one gold in the top 50), mean gold
cardinality, and **label-level recall@50 by bucket** (head ≥100 train
documents, torso 10–99, tail 1–9, unseen 0). Ties break by label column order
(a stable descending sort). The definitions live in
`src/cluster_rag/topic_classification/metrics.py`.

**Leakage control.** The audit found that 26.1% of dev/test documents have a
train neighbour at TF-IDF cosine ≥0.90. Every method is reported **overall**
and on the **clean slice** (best train cosine < 0.90), using the audit's
near-duplicate text view.

**Pre-registered reading (the bars, fixed before the run).**

- **Beat the prior.** The dictionary must beat the frequency prior's R@50 by
  at least **0.10**; otherwise the "dictionary" is only the label prior.
- **Useful absolute level.** Dictionary R@50 **≥ 0.80** is useful for a
  downstream reranker; **0.50–0.80** is marginal; **< 0.50** is not useful at
  this label count.
- **Cheap alternative.** The dictionary is a viable cheap alternative to the
  embedding k-NN if its R@50 is within **0.10** of the embedding's. If the
  embedding is more than 0.10 ahead, the embedding is the shortlist generator
  to use for the next step.
- **Leakage.** If the dictionary's overall R@50 exceeds its clean-slice R@50
  by more than **0.05**, the overall number is inflated by near-duplicates and
  the clean slice is the headline.
- **Breadth vs ordering.** R-precision and nDCG@10 carry no separate bar, but
  a method that wins R@50 while collapsing R-precision is reported as a
  breadth-only shortlist.
- **No auto-adopt.** The numbers are the evidence; the reviewer decides the
  next step.

**Cost.** Dictionary: CPU, ~1–2 minutes. Embedding: one MiniLM pass over
51,000 documents, ~10 minutes on this CPU, no GPU, no LLM. The k-NN and metrics
are a few minutes.

**Not tested here.** `test`; the label-holdout mask (needs a held-out label
set); the decision model (LLM cost); label-name embeddings (titles are
available but not used in this arm); chunked embeddings (the MiniLM context is
the current embedding view).

### Result (2026-11-13)

Artifact: `results/eurlex-shortlist/metrics.json` (run `eurlex-shortlist`, 1,452 s,
no LLM, embedding cache 0 hits / 51,000 misses). The experiment's score columns
are the **4,197 labels present in train + dev**. The audit's dataset-wide union
is 4,271; the 74 labels that occur only in `test` are deliberately excluded —
`test` is untouched and they have no dev gold, so they would be floored and
could not change any dev metric. The dev unseen bucket is therefore 89 labels
(the dev-only zero-shot ones), not 163.

**Primary metrics (overall / clean slice, macro over documents):**

| method | R@50 | R-precision | nDCG@10 | coverage@50 |
|---|---:|---:|---:|---:|
| frequency prior | 0.3203 / 0.2680 | 0.0769 / 0.0420 | 0.1037 / 0.0680 | 0.6867 |
| dictionary `sum_zscore_positive` (**pre-registered primary**) | **0.5388** / 0.4846 | **0.3810** / 0.3149 | **0.4835** / 0.4142 | 0.8887 |
| dictionary `sum_logodds` | 0.6714 / 0.6095 | 0.3569 / 0.2924 | 0.4571 / 0.3867 | 0.9430 |
| dictionary `support_count` (measure-free) | **0.8171** / 0.7694 | 0.3376 / 0.3140 | 0.4508 / 0.4180 | 0.9803 |
| embedding k-NN (MiniLM, k=100) | **0.9168** / 0.8979 | **0.5715** / 0.5079 | **0.7070** / 0.6545 | 0.9978 |

**Secondary metrics (overall):**

| method | R@5 | R@10 | R@20 | R@100 | nDCG@50 |
|---|---:|---:|---:|---:|---:|
| frequency prior | 0.0720 | 0.1289 | 0.1929 | 0.4604 | 0.1737 |
| dictionary `sum_zscore_positive` | 0.3801 | 0.4860 | 0.5263 | 0.5527 | 0.5058 |
| dictionary `sum_logodds` | 0.3598 | 0.4854 | 0.5734 | 0.7358 | 0.5278 |
| dictionary `support_count` | 0.3473 | 0.5194 | 0.6797 | 0.8777 | 0.5670 |
| embedding k-NN | 0.5623 | 0.7230 | 0.8306 | 0.9473 | 0.7838 |

**Label-level recall@50 by train-support bucket (overall):**

| method | head (441) | torso (1,400) | tail (2,267) | unseen (89) |
|---|---:|---:|---:|---:|
| frequency prior | 0.4167 | 0.0000 | 0.0000 | 0.0000 |
| dictionary `sum_zscore_positive` | 0.5943 | 0.3376 | 0.1686 | 0.0000 |
| dictionary `sum_logodds` | 0.6624 | 0.6836 | 0.5464 | 0.0000 |
| dictionary `support_count` | 0.9228 | 0.5036 | 0.1223 | 0.0000 |
| embedding k-NN | 0.9614 | 0.8068 | **0.4093** | 0.0000 |

**The bars, read out.**

- **Beat the prior: passed by every dictionary arm.** R@50 gains over the
  frequency prior: z-score +0.2185, log-odds +0.3511, count +0.4968.
- **Useful absolute level: only the count arm passes, and it is breadth-only.**
  The pre-registered primary (`sum_zscore_positive`) is **marginal** at 0.5388
  (0.50–0.80 band); `sum_logodds` is also marginal at 0.6714; only
  `support_count` clears 0.80 (0.8171). But the count arm's R-precision
  (0.3376) and nDCG@10 (0.4508) are **below** the z-scored arm's
  (0.3810 / 0.4835) and far below the embedding's (0.5715 / 0.7070). Its
  head recall is 0.9228 against a tail recall of 0.1223: it is a head-heavy
  breadth shortlist, not a balanced one.
- **Cheap alternative: no on the clean slice.** The count arm is 0.0997 behind
  the embedding overall — just inside the 0.10 band — but 0.1285 behind on the
  clean slice, and its R-precision gap is 0.2339. The pre-registered primary is
  0.3780 behind overall. The embedding is the shortlist generator at this label
  count.
- **Leakage: the dictionary's overall number is inflated.** Clean-slice drops:
  z-score 0.0542 (> 0.05, so the clean number is the headline), log-odds
  0.0619 (> 0.05), count 0.0477, embedding 0.0189. The dictionary reads the
  whole document and benefits more from the template duplicates than the
  256-token embedding does.
- **Breadth vs ordering.** The count arm wins breadth (R@50 0.8171) but loses
  ordering (nDCG@10 0.4508; R-precision 0.3376). The z-scored arm is the
  dictionary's best orderer (nDCG@10 0.4835; R-precision 0.3810) and its worst
  at breadth. The signed log-odds is the most balanced across buckets
  (torso 0.6836, tail 0.5464) but below the count arm on R@50.

**Reading.** The pre-registered dictionary arm does **not** produce a useful
shortlist at this label count: 0.5388 R@50 and 0.3810 R-precision. The
measure-free count reaches 0.8171 R@50, but it does so by ranking the head
labels, and it collapses on the tail (0.1223), on R-precision and on nDCG@10.
The embedding k-NN is materially better on every headline metric and much
better on the tail (0.4093). The dictionary is not a substitute for the
embedding shortlist here; it may still be a useful **feature/channel** in a
later pack, which is the UPR precedent (the dictionary's column inside the
pack, not the dictionary alone).

**What the result does not say.** Nothing about `test`; nothing about a
label-holdout mask; nothing about the decision model; nothing about label-name
embeddings (titles are available but this arm does not use them); nothing about
chunked embeddings (the MiniLM context is 256 tokens). The next step is the
reviewer's.

## 9. Family-held-out evaluation (pre-registered 2026-11-13, before the run)

**Question.** The random dev split has 26.1% of documents with a train near
neighbour at TF-IDF cosine ≥ 0.90, and its families are not held out. Does the
embedding k-NN still beat the dictionary on documents whose **title-skeleton
family never appears in the train index**, and how much of the 0.9168 R@50 is
family memorisation?

**Family definition.** The audit's title skeleton, unchanged: lowercase, digits
→ `#`, punctuation removed, whitespace collapsed (`title_skeleton` in
`experiments_topic_classification/audit_eurlex.py`). The evaluation slice is
every dev document whose skeleton is **absent from train** — a family never in
the index. The complement (family seen in train) is reported separately so the
family effect is visible.

**Slices.**

- `all_dev` — all 6,000 dev documents (the 0.9168 comparator).
- `family_new` — dev documents whose family is absent from train.
- `family_seen` — the complement.
- `family_new_clean` — `family_new` ∩ the audit's near-duplicate clean slice
  (best train TF-IDF cosine < 0.90); the strongest "new document" view.
- `family_new_seen_labels` — `family_new`, restricted to gold labels that have
  at least one train example (the zero-shot labels removed), so family
  generalisation is not confused with label absence.

**Methods.** The same three dictionary arms (`sum_zscore_positive`,
`sum_logodds`, `support_count`), the same MiniLM k-NN (k=100, power 1.0, cached
vectors) and the frequency prior. No re-embedding; no new model.

**Metrics.** The same primary and secondary set as section 8 (recall@50,
R-precision, nDCG@10; recall@5/10/20/100; nDCG@50; coverage@50; label-level
recall@50 by bucket), computed per slice.

**Pre-registered reading.**

- **Family effect.** If `kNN R@50(all_dev) − kNN R@50(family_new)` is at least
  **0.05**, the random-split number is materially family-driven, and the
  `family_new` number becomes the headline for the new-family claim.
- **Embedding vs dictionary on new families.** The embedding is a clear winner
  on `family_new` if its R@50 exceeds the best dictionary arm's by at least
  **0.10**; a smaller gap makes the dictionary a viable new-family
  alternative. The same comparison is read on `family_new_clean` and on
  R-precision / nDCG@10.
- **Zero-shot share.** If `family_new_seen_labels` exceeds `family_new` by more
  than **0.05** R@50, label absence (not family overlap) is a material part of
  the family-new drop; report both.
- **No auto-adopt.** The numbers are the evidence; the reviewer decides the
  next step.

**Cost.** Dictionary rebuild (~34 s) + scoring the 6,000 dev documents on the
three arms (the dominant cost, ~15 min) + cached MiniLM vectors + the k-NN
(~15 s) + the near-duplicate slice (~30 s). No LLM, no new embedding pass.

### Result (2026-11-13)

Artifact: `results/eurlex-family-heldout/metrics.json` (run
`eurlex-family-heldout`, 990 s; the MiniLM cache had 51,000 hits / 0 misses, no
LLM). The slice is an **evaluation filter**, not a resplit: the index is still
the original train split, and `family_new` is the dev documents whose title
skeleton is absent from it.

**Slice sizes.**

| slice | documents | notes |
|---|---:|---|
| all dev | 6,000 | the 0.9168 comparator |
| family_new | 4,260 | family absent from train |
| family_seen | 1,740 | family present in train |
| family_new_clean | 3,764 | family_new ∩ near-dup clean slice |
| family_new_seen_labels | 4,260 | family_new, zero-shot gold labels removed |

The family-new slice carries 21,994 gold assignments; only **88 (0.4%)** are
zero-shot labels, so label absence is not the story. The seen-label view has
the same 4,260 documents.

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| method | all dev | family_new | family_seen | family_new_clean |
|---|---|---|---|---|
| frequency prior | 0.3203 / 0.0769 / 0.1037 | 0.2420 / 0.0262 / 0.0524 | 0.5122 / 0.2011 / 0.2292 | 0.2347 / 0.0252 / 0.0505 |
| dictionary `sum_zscore_positive` | 0.5388 / 0.3810 / 0.4835 | 0.4784 / 0.3023 / 0.4033 | 0.6868 / 0.5737 / 0.6798 | 0.4638 / 0.2867 / 0.3845 |
| dictionary `sum_logodds` | 0.6714 / 0.3569 / 0.4571 | 0.5906 / 0.2732 / 0.3640 | 0.8693 / 0.5620 / 0.6850 | 0.5723 / 0.2586 / 0.3470 |
| dictionary `support_count` | 0.8171 / 0.3376 / 0.4508 | 0.7668 / 0.3276 / 0.4295 | 0.9404 / 0.3621 / 0.5032 | 0.7443 / 0.3112 / 0.4095 |
| embedding k-NN | 0.9168 / 0.5715 / 0.7070 | **0.8944 / 0.4869 / 0.6364** | **0.9718 / 0.7786 / 0.8797** | **0.8862 / 0.4744 / 0.6241** |

**Family effect (R@50).**

| method | all dev − family_new | family_seen − family_new |
|---|---:|---:|
| frequency prior | 0.0784 | 0.2702 |
| dictionary `sum_zscore_positive` | 0.0604 | 0.2084 |
| dictionary `sum_logodds` | 0.0808 | 0.2787 |
| dictionary `support_count` | 0.0503 | 0.1736 |
| embedding k-NN | **0.0224** | **0.0774** |

**The bars, read out.**

- **Pre-registered family effect: not met on the all-dev average.** The k-NN's
  `all_dev − family_new` is **0.0224**, below the 0.05 bar. But the direct
  family comparison is larger: **family_seen 0.9718 vs family_new 0.8944 =
  0.0774**. The all-dev average is diluted because 71% of dev documents are
  already family-new. So the random-split 0.9168 overstates the new-family
  number by ~0.022; the **new-family R@50 is 0.8944**, and the
  new-family-and-clean number is **0.8862**.
- **Embedding vs dictionary on new families: clear winner.** The best
  dictionary arm (`support_count`, R@50 0.7668) is **0.1276** behind the
  embedding on family_new and **0.1420** behind on family_new_clean — both past
  the 0.10 bar. On R-precision the gap is 0.1593 (0.4869 vs 0.3276); on
  nDCG@10 it is 0.2069 (0.6364 vs 0.4295). The dictionary is not a substitute
  on genuinely new families.
- **The dictionary is more family-sensitive than the embedding.** Its all-dev
  − family_new drop is 0.0503–0.0808 (versus the embedding's 0.0224), and its
  family_seen − family_new gap is 0.17–0.28 (versus 0.077). The count arm's
  family_new tail recall is **0.112**, still head-heavy; `sum_logodds` is the
  most balanced dictionary arm (tail 0.539) but far below the embedding on
  R-precision and nDCG.
- **Zero-shot label absence is not the driver.** It is 0.4% of family-new
  assignments; removing those labels changes the k-NN by +0.0027 R@50
  (0.8970 seen-label vs 0.8944 overall), the z-scored dictionary by +0.0013,
  and the count arm by +0.0019.

**Reading.** The 0.917 random-split number is **not** mostly family
memorisation: on documents whose family was never in the index the embedding
still reaches **0.8944 R@50, 0.4869 R-precision, 0.6364 nDCG@10** (0.8862 /
0.4744 / 0.6241 on the clean slice). That is the number to quote for the
new-family claim. The embedding remains clearly ahead of the dictionary, and
the dictionary's shortlist degrades more when the family is new. The next step
is the reviewer's; the zero-shot label-name arm and the fusion/pack step are
still open.

## 10. Label-name embeddings — the zero-shot arm (pre-registered 2026-11-13, before the run)

**Question.** Both current shortlist generators score **0.0000** on the 89 dev
zero-shot labels: the k-NN can only vote for labels attached to train
documents, and the dictionary has empty tables for them. The vocabulary now
gives every label a title. Can a label-name embedding give those labels their
only signal, and does it improve the shortlist?

**Label titles.** The `eurovoc_concepts.jsonl` title for each of the 4,197
train+dev labels (the 74 test-only labels stay excluded, as in sections 8–9).
Raw titles, no template. Embedded with the same MiniLM-L6-v2, `label` kind,
cosine to the dev document vectors already cached from section 8.

**Arms.**

1. **Label-name alone** — score every dev document against every label title;
   a third shortlist generator.
2. **k-NN + zero-shot rescue** — keep the k-NN scores for labels with at least
   one train example; replace the scores of the 89 zero-shot labels with their
   label-name cosine. This tests the targeted use of the arm without changing
   the seen-label ranking.
3. **Untuned equal-weight fusion** (secondary) — per-row z-score the k-NN and
   label-name matrices and add them. No weight is tuned; it is a read on
   whether label names help beyond the rescue.

**Metrics.** The same primary and secondary set as section 8 (R@50,
R-precision, nDCG@10; recall@5/10/20/100; nDCG@50; coverage@50; bucket
recall@50). The **unseen bucket** is the headline this time. For the
label-name arm, a **zero-shot-only ranking** is added: restricted to the 89
zero-shot labels, for the 87 dev documents that carry one, rank only those
labels and report recall@5/10/50. That separates "the title signal exists"
from "it survives the full 4,197-label ranking".

**Pre-registered reading.**

- **The zero-shot route must be non-zero.** Label-name alone must reach a
  non-zero unseen-bucket recall@50 (any positive value); 0.0000 means the
  titles carry no usable signal for the labels with no examples.
- **The rescue must not cost the seen ranking.** Its overall R@50 must stay
  within **0.01** of the k-NN's 0.9168 (section 8, all dev) and its
  unseen-bucket recall@50 must be positive.
- **The untuned fusion earns a follow-up only if** it beats the k-NN's R@50 by
  at least **0.01**; otherwise it is a null and the next step is the
  dictionary-as-feature fusion (step 3), not this one.
- **No auto-adopt.** The numbers are the evidence; the reviewer decides.

**Cost.** 4,197 title embeddings (seconds), the cached dev document vectors,
one matrix product, the k-NN scores (seconds). No LLM, no new model, no
re-embedding of documents.

### Result (2026-11-13)

Artifact: `results/eurlex-labelname/metrics.json` (run `eurlex-labelname`, 25 s;
4,197 title embeddings, 6,000 cached document vectors, no LLM). The label
space is the same 4,197 train+dev labels; the zero-shot set is the **89 dev
labels** with no train example, carrying **96 gold assignments** across **87
documents**.

**Overall metrics (all dev):**

| method | R@50 | R-precision | nDCG@10 | unseen R@50 | coverage@50 |
|---|---:|---:|---:|---:|---:|
| frequency prior | 0.3203 | 0.0769 | 0.1037 | 0.0000 | 0.6867 |
| embedding k-NN | **0.9168** | 0.5715 | 0.7070 | 0.0000 | 0.9978 |
| label-name alone | 0.2148 | 0.0543 | 0.0802 | **0.1042** | 0.6698 |
| k-NN + zero-shot rescue | 0.9168 (−0.00002) | 0.5715 | 0.7070 | 0.0104 | 0.9978 |
| k-NN + label-name fusion (untuned) | 0.8806 (−0.0362) | **0.5754** | **0.7113** | 0.0104 | 0.9977 |

**Label-name alone, zero-shot-only ranking** (89 labels, 87 documents with an
unseen gold label): R@5 0.3563, R@10 0.4770, R@20 0.6322, **R@50 0.8678**,
R@100 1.0; R-precision 0.1264; nDCG@10 0.2888, nDCG@50 0.3792; coverage@50
0.8736.

**The bars, read out.**

- **The zero-shot route is non-zero: passed.** Label-name alone reaches
  **0.1042** unseen-bucket R@50 in the full 4,197-label ranking, and
  **0.8678** when restricted to the 89 zero-shot labels. The titles carry a
  real signal — the only signal any of these arms has for those labels.
- **The rescue does not cost the seen ranking, but is practically a null.**
  Its overall R@50 is 0.9168 (−0.00002), so the “within 0.01” bar is passed; its
  unseen-bucket R@50 is positive (0.0104), so the letter of the bar is passed
  too. But **1 of 96** unseen gold assignments enters the full top-50. The raw
  label-name cosine is not calibrated to the k-NN’s vote scale: the unseen
  scores are too small to displace the seen labels’ k-NN scores.
- **The untuned fusion is not a win.** It gains a little ordering (R-precision
  +0.0039, nDCG@10 +0.0043) but loses **0.0362 R@50**, well short of the
  +0.01 bar. The equal-weight z-score fusion is a breadth loss.
- **Label-name alone is not a third general shortlist generator**: 0.2148 R@50,
  below the frequency prior’s 0.3203. It is a **zero-shot specialist**, not a
  replacement for the k-NN.

**Reading.** The title signal is real but **scale-bound**: within the
zero-shot labels it ranks well (0.868 R@50), but in the full ranking its
cosine scores sit below the k-NN’s vote scores, so the naive rescue finds one
unseen label in 96. The next step, if the reviewer wants the zero-shot route,
is a **calibrated** zero-shot channel — map the unseen labels’ title scores
into the seen score distribution (rank/quantile mapping) instead of replacing
raw cosines. The untuned fusion does not earn a follow-up; the dictionary-as-
feature fusion (step 3) remains the next real pack step.

## 11. Dictionary as a feature — one-parameter fusion (pre-registered 2026-11-13, before the run)

**Question.** The UPR table’s “boosting + dictionary column” asks whether the
dictionary helps inside a model even when it is weak alone. The cheapest form
is one parameter: z-score the dictionary and the k-NN scores per document and
add them, `fusion = z(k-NN) + w · z(dictionary)`. Does any `w > 0` beat the
k-NN alone (`w = 0`)?

**Scores.** The same 4,197-label dev evaluation as sections 8–10. Three
dictionary arms: `sum_zscore_positive` (the section-8 primary),
`sum_logodds` and `support_count`. The k-NN is the same MiniLM k=100 vote
score. Each matrix is z-scored **per document** (mean 0, std 1; a zero-variance
row becomes zeros), then combined at weight `w`.

**Weight grid and selection.** `w ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0,
1.5, 2.0}`. Because dev is the only tuning set and the evaluation must stay
trustworthy, the weight is selected by **5-fold group cross-validation on
dev**, with the folds grouped by the audit’s title skeleton so a document’s
weight is never selected on a document from its own family. For each fold, the
weight maximises R@50 on the other folds (ties go to the smaller weight, i.e.
towards the k-NN); every dev document is then scored by its held-out fold’s
weight, and the pooled matrix is evaluated once. The full-dev grid is reported
beside the cross-validated number, clearly marked as tuned.

**Metrics.** The same primary and secondary set as section 8, plus the
family_new and family_new_clean slices of section 9. A paired bootstrap over
documents (2,000 resamples) gives the 95 % interval of the fusion-minus-k-NN
difference for R@50, R-precision and nDCG@10.

**Pre-registered reading.**

- **A fusion win** requires the cross-validated primary arm’s R@50 to improve
  by at least **0.01** over the k-NN, with the paired 95 % interval excluding
  zero, and the same direction on `family_new`. If R@50 does not move but
  R-precision / nDCG@10 do, it is an **ordering gain**, not a shortlist gain,
  and the candidate-level combiner (stage 2) is not licensed by this result.
- **A collapse on any bucket** (head / torso / tail / unseen) is reported even
  if the pooled number rises.
- **No auto-adopt.** The reviewer decides; stage 2 (train-candidate combiner)
  starts only if the fusion gate passes and the reviewer asks for it.

**Cost.** Recompute the dictionary matrices (`~15 min`), the cached MiniLM
k-NN (`~15 s`), and the fusion grid (seconds). No LLM, no new model. The score
matrices are saved under `results/eurlex-fusion/scores/` for a possible stage 2.

### Result (2026-11-13)

Artifact: `results/eurlex-fusion/metrics.json` (run `eurlex-fusion`, 1,072 s;
the score matrices are saved under `results/eurlex-fusion/scores/`). The
weight is selected by 5-fold group CV on dev (whole families held out), so the
CV number is not tuned on the evaluated documents.

**Cross-validated fusion vs the k-NN (all dev; Δ is fusion − k-NN, 2,000-resample paired bootstrap):**

| dictionary arm | CV R@50 | Δ R@50 [95 % CI] | Δ R-precision [95 % CI] | Δ nDCG@10 [95 % CI] | selected weights |
|---|---:|---|---|---|---|
| `sum_zscore_positive` (primary) | 0.9171 | **+0.0003 [−0.0009, +0.0015]** | **−0.0204 [−0.0233, −0.0176]** | **−0.0190 [−0.0212, −0.0168]** | 1.0, 0.75, 0.75, 0.05, 0.75 |
| `sum_logodds` | 0.9173 | +0.0005 [−0.0008, +0.0018] | +0.0029 [+0.0019, +0.0039] | +0.0037 [+0.0030, +0.0043] | 0.2, 0.2, 0.3, 0.3, 0.3 |
| `support_count` | 0.9178 | +0.0010 [+0.0004, +0.0016] | +0.0002 [+0.00004, +0.0004] | +0.0001 [−0.00001, +0.0003] | 0.05 × 5 |
| k-NN alone | 0.9168 | — | — | — | 0 |

The primary arm’s best **full-dev grid** weight is 0.75, R@50 0.9174 (tuned on
dev, so not the headline); its CV family-new R@50 is 0.8945 against the k-NN’s
0.8944. The z-scored fusion’s CV bucket recalls are head 0.9621, torso 0.8063,
tail 0.4034, unseen 0; the k-NN’s are 0.9614 / 0.8068 / 0.4093 / 0. The
dictionary does not lift the tail.

**The bars, read out.**

- **A fusion win: not met.** The primary arm’s CV R@50 gain is **+0.0003**,
  far below the +0.01 bar, and its 95 % interval includes zero. It also loses
  **0.0204 R-precision** and **0.0190 nDCG@10** — the selected weights trade
  ordering for a negligible breadth change.
- **The other arms are not wins either.** `sum_logodds` is a small **ordering
  gain** (+0.0029 R-precision, +0.0037 nDCG@10) with flat R@50 (+0.0005,
  interval includes zero); `support_count` gains +0.0010 R@50 but its
  R-precision is flat and its nDCG interval includes zero. Both are well below
  the 0.01 shortlist bar.
- **The full grid confirms the shape.** At small weights the primary arm’s
  ordering edges up (w = 0.1: R-precision 0.5731, nDCG@10 0.7089, R@50
  0.9169), and `sum_logodds` at w = 0.3–0.5 reaches R-precision 0.5747–0.5761
  and nDCG@10 0.7112–0.7140. These are **ordering** gains of a few thousandths,
  not shortlist gains; no weight reaches the +0.01 R@50 bar.
- **Stage 2 is not licensed.** The candidate-level combiner was gated on a
  +0.01 CV R@50 primary-arm gain; the observed gain is +0.0003 with an interval
  including zero. The one-parameter fusion is a null for the shortlist
  question.

**Reading.** At 4,197 labels the dictionary adds essentially nothing to the
k-NN shortlist through a single global weight. The UPR “dictionary column”
evidence came from a learned pack with many features and a different label
space; this cheap fusion does not reproduce it. A learned candidate combiner
could still be worth a separate experiment, but it is not licensed by this
gate, and the evidence here says the dictionary's value is not a one-parameter
re-ranking of the k-NN. The calibrated zero-shot channel remains the other
open follow-up.
