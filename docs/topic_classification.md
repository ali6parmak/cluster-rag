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
