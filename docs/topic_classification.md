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
