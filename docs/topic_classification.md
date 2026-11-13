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

## 12. Calibrated zero-shot channel (pre-registered 2026-11-13, before the run)

**Question.** The label-name signal exists but is scale-bound: raw title cosine
does not enter the k-NN’s top-50 (section 10: 1 of 96 unseen gold assignments).
Can a **rank/quantile calibration** put the zero-shot labels on the k-NN score
scale without hurting the seen labels?

**The calibration.** For each dev document:

1. Compute the label-title cosine for all 4,197 labels (seen and zero-shot).
2. For each zero-shot label, take `q` = the percentile of its title cosine
   **among all 4,197 label-title cosines of that document**.
3. Map `q` to the document’s k-NN score distribution: `calibrated =
   quantile(seen-label k-NN scores, q)`.
4. Blend into the k-NN scores with a single strength `α`:
   `final = kNN + α · calibrated` on the zero-shot columns only; seen labels
   are unchanged. `α = 0` is the k-NN baseline; `α = 1` is the full calibration.

This is the rank/quantile mapping suggested by section 10, not a raw cosine
replacement.

**Selection and evaluation.** `α ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}`, selected by
5-fold group CV on dev (whole title-skeleton families held out); every document
is scored by the weight of its held-out fold, and the pooled matrix is
evaluated once. The full-dev grid is reported beside the CV number. The
family_new and family_new_clean slices of section 9 are reported too.

**Metrics.** The same primary and secondary set, with the **unseen bucket** as
the headline. Report the unseen hits at 50 (of 96 assignments), the seen-bucket
recalls, and the overall R@50 / R-precision / nDCG@10. A paired bootstrap over
documents gives the 95 % interval of the overall R@50 difference from the
k-NN; the unseen-bucket count is reported directly (96 assignments, so it is
read as a count, not a precise rate).

**Pre-registered reading.**

- **Zero-shot win:** unseen-bucket R@50 **≥ 0.10** (at least the label-name
  alone’s 0.1042) with an overall R@50 loss **≤ 0.01**. If the calibration
  costs more than 0.01 overall, it is not a net win, even if the unseen bucket
  rises.
- **Overall win:** CV R@50 improvement **≥ 0.01** over the k-NN; a smaller
  change is not a shortlist win.
- **A null:** if the selected `α` is 0, or the unseen bucket stays below 0.10,
  the calibrated channel does not ship; report the null.
- **No auto-adopt.** The reviewer decides.

**Cost.** The cached title and document vectors, the cached k-NN scores
(seconds), the quantile mapping (seconds). No dictionary rebuild, no LLM, no
new model.

### Result (2026-11-13)

Artifact: `results/eurlex-zeroshot-calibrated/metrics.json` (run
`eurlex-zeroshot-calibrated`, 55 s; cached vectors, cached k-NN, no LLM).

**Full-dev grid (calibration strength α):**

| α | overall R@50 | unseen hits / 96 | unseen R@50 | head R@50 | tail R@50 |
|---:|---:|---:|---:|---:|---:|
| 0 (k-NN) | 0.9168 | 0 | 0.0000 | 0.9614 | 0.4093 |
| 0.1 | 0.9168 | 0 | 0.0000 | 0.9614 | 0.4093 |
| 0.25 | 0.9168 | 0 | 0.0000 | 0.9613 | 0.4093 |
| 0.5 | 0.9167 | 2 | 0.0208 | 0.9613 | 0.4083 |
| 0.75 | 0.9164 | 6 | 0.0625 | 0.9612 | 0.4034 |
| 1.0 | 0.9163 | 9 | 0.0938 | 0.9611 | 0.4024 |

**Grouped CV:** R@50 0.9167, Δ vs k-NN **−0.00009 [−0.0002, 0.0000]**; pooled
unseen hits **0**; selected α = 0, 0, 0, 0, 0.5 across the five folds. The
full-dev grid also selects α = 0 on overall R@50.

**The bars, read out.**

- **Zero-shot win: not met, by one hit.** At full strength (α = 1) the
  calibration puts **9 of 96** unseen gold assignments in the top 50
  (unseen R@50 0.0938), just below the pre-registered **0.10** bar and slightly
  below the raw label-name alone (10 of 96, 0.1042 in section 10). The overall
  R@50 cost is small (0.9163 vs 0.9168, −0.00046), so the loss condition
  passes; the unseen-bucket condition does not.
- **Overall win: not met.** No α improves overall R@50; the grid’s best is
  α = 0.
- **The CV selects the k-NN.** Because the overall R@50 effect is tiny and
  slightly negative, four of five folds choose α = 0; the pooled CV unseen
  hits are 0. The calibration is not stable enough to be selected.
- **The calibration does not beat the raw label-name arm.** Raw replacement
  gave 0.1042 unseen; quantile calibration at full strength gives 0.0938 and
  costs 0.00046 overall. The rank mapping does not add signal beyond the raw
  cosine — it only moves the competition from the title scale to the k-NN
  scale.

**Reading.** The calibrated zero-shot channel is a **null**. The title signal
is real (section 10), but neither raw replacement nor this quantile calibration
makes it survive the 4,197-label ranking strongly enough to clear the 0.10
unseen bar; the best full-strength arm misses by one hit and is slightly worse
than label-name alone. The zero-shot labels remain unsolved in the current
shortlist methods. A different insertion rule (a quota or a learned
calibrator) would be a new pre-registration, not a continuation of this one.

## 13. Chunk-mean document embedding — TC-7 (pre-registered 2026-11-13, before the run)

**Question.** The whole-document MiniLM view truncates at the model's 256-token
context, and the median EURLEX document is 447 words: the embedding sees roughly
half of it. Does replacing one truncated whole-document vector with the **mean of
non-overlapping 200-word chunk embeddings** improve the k-NN shortlist,
especially on `family_new`?

**Mechanism (fixed before the run).**

- **Text view:** the same `full_text` as the baseline — `title`, `header`,
  `recitals` and `main_body`, newline-joined.
- **Chunking:** split on whitespace into consecutive, **non-overlapping**
  windows of exactly **200 words**; the last window carries the remainder
  (shorter). A document of at most 200 words is one chunk.
- **Embedding:** MiniLM-L6-v2 (revision
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`), the cached model, `document`
  kind, each chunk embedded independently.
- **Pooling:** arithmetic mean of the chunk vectors, then L2-normalize, so the
  document vector is unit-norm and the k-NN's dot product is a cosine.
- **k-NN:** identical to the baseline — cosine, `k = 100` neighbours,
  similarity-weighted label votes (power 1.0), train = the 45,000-document index,
  dev = the 6,000 evaluated documents.
- **Cache:** baseline `eurlex-4k-v1` (whole documents); treatment
  `eurlex-4k-chunk200-v1` (chunks). The version slot is part of the cache key,
  so the two views never collide.
- **No tuning.** The chunk size is fixed at **200 words** by this
  pre-registration; it is not tuned on dev in this change set, and no other
  chunk-size or pooling variant is run.

**Slices.** `all_dev` (6,000), `family_new` (dev documents whose title skeleton
is absent from train; 4,260 in section 9), and `family_new_clean`
(`family_new` ∩ best train TF-IDF cosine < 0.90; 3,764). The baseline is
recomputed in the same run from its warm cache and reported beside the
treatment on every slice; test is never read.

**Metrics.** Primary: R@50, R-precision, nDCG@10. Secondary: R@5, R@10, R@20,
R@100, nDCG@50. Diagnostics: coverage@50 and label-level recall@50 by
head / torso / tail / unseen. The **paired bootstrap over documents** (2,000
resamples, percentile 95 % interval) gives the interval of the treatment-minus-
baseline R@50 difference on each slice; the same resampled document indices are
used for both methods.

**Pre-registered bar.**

- A **win** requires the treatment to improve `family_new` **R@50 by at least
  0.01**, with the paired 95 % bootstrap interval **excluding 0**, and to lose
  **R-precision and nDCG@10 by no more than 0.01** on `family_new`.
- Otherwise it is a **null**: record it and do not proceed to menu item 2.

**Cost.** ~174,000 chunk embeddings (154,061 train + 20,237 dev; mean 2.7
chunks/document), a forward pass of the cached MiniLM on CPU; ~25–40 min plus
the cache writes, the two k-NN passes and the evaluation. No new model
download, no LLM, no dictionary rebuild.

**Not tested here.** Other chunk sizes, overlap, other pooling rules, longer-
context embedding models, multi-view embeddings (menu item 2), `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-embed-chunkmean/metrics.json` (run
`eurlex-embed-chunkmean`, **1,394.6 s**, no LLM). The baseline was re-scored in
the same run from the warm `eurlex-4k-v1` cache (**51,000 hits / 0 misses**) and
reproduces section 9 exactly (all dev 0.9168, family_new 0.8944, clean 0.8862
R@50). The treatment embedded **174,298 chunks** (3.42 per document on average,
max 18; 154,061 train + 20,237 dev) under `eurlex-4k-chunk200-v1` (**3,977 cache
hits / 170,321 misses**; the hits are boilerplate chunks already cached by a
smoke pass on a document subset). Train = 45,000-document index, dev = 6,000 evaluated; test unread.

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| method | all_dev | family_new | family_new_clean |
|---|---|---|---|
| baseline — whole document, MiniLM, k=100 | **0.9168 / 0.5715 / 0.7070** | **0.8944 / 0.4869 / 0.6364** | **0.8862 / 0.4744 / 0.6241** |
| treatment — 200-word chunk mean, same k-NN | 0.9125 / 0.5536 / 0.6881 | 0.8887 / 0.4677 / 0.6145 | 0.8800 / 0.4534 / 0.5995 |

**Secondary metrics by slice (R@5 / R@10 / R@20 / R@100 / nDCG@50):**

| slice | method | R@5 | R@10 | R@20 | R@100 | nDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| all_dev | baseline | **0.5623** | **0.7230** | **0.8306** | **0.9473** | **0.7838** |
| all_dev | treatment | 0.5459 | 0.7105 | 0.8220 | 0.9453 | 0.7678 |
| family_new | baseline | **0.4839** | **0.6524** | **0.7843** | **0.9342** | **0.7323** |
| family_new | treatment | 0.4648 | 0.6364 | 0.7721 | 0.9316 | 0.7139 |
| family_new_clean | baseline | **0.4742** | **0.6408** | **0.7721** | **0.9287** | **0.7210** |
| family_new_clean | treatment | 0.4526 | 0.6228 | 0.7593 | 0.9257 | 0.7005 |

**Diagnostics (coverage@50; label-level recall@50 by head / torso / tail /
unseen; mean gold cardinality):**

| slice | method | coverage@50 | head | torso | tail | unseen | cardinality |
|---|---|---:|---:|---:|---:|---:|---:|
| all_dev | baseline | **0.9978** | **0.9614** | **0.8068** | **0.4093** | 0.0 | 5.04 |
| all_dev | treatment | 0.9968 | 0.9608 | 0.7931 | 0.3895 | 0.0 | 5.04 |
| family_new | baseline | **0.9972** | **0.9489** | **0.7995** | **0.4011** | 0.0 | 5.163 |
| family_new | treatment | 0.9958 | 0.9481 | 0.7859 | 0.3819 | 0.0 | 5.163 |
| family_new_clean | baseline | **0.9968** | **0.9441** | **0.7872** | **0.3868** | 0.0 | 5.134 |
| family_new_clean | treatment | 0.9952 | 0.9435 | 0.7727 | 0.3649 | 0.0 | 5.134 |

**Paired bootstrap over documents (treatment − baseline R@50, 2,000
resamples):**

| slice | documents | Δ R@50 | 95 % interval | excludes 0 |
|---|---:|---:|---|---|
| all_dev | 6,000 | **−0.0043** | [−0.0071, −0.0015] | yes (negative) |
| family_new | 4,260 | **−0.0057** | [−0.0094, −0.0020] | yes (negative) |
| family_new_clean | 3,764 | **−0.0062** | [−0.0102, −0.0022] | yes (negative) |

**The bar, read out.**

- **R@50 gain ≥ 0.01 on family_new: not met.** The treatment **loses** 0.0057
  (0.8887 vs 0.8944); the interval is entirely below zero.
- **Interval excluding 0: not met in the required direction.** The interval
  excludes zero, but on the negative side: the loss is real, not noise.
- **R-precision loss ≤ 0.01: not met.** It loses **0.0192** (0.4677 vs 0.4869).
- **nDCG@10 loss ≤ 0.01: not met.** It loses **0.0220** (0.6145 vs 0.6364).
- **`outcome`: null.** No condition holds. The treatment is worse on every
  primary metric on every slice, and worse on all secondary metrics except
  R@100 (a near-tie: 0.9453 vs 0.9473 on all dev).

**Reading.** Chunk-mean is a **null**, and a negative one: it does not merely
fail to clear the +0.01 bar, it loses 0.4–0.6 points of R@50 and roughly two
points of R-precision and nDCG@10 on every slice. The loss is ordering-heavy
(R-precision and nDCG fall ~3–4× more than R@50) and slightly stronger on the
clean slice than on all dev, so near-duplicate memorisation does not explain it.
The head bucket is almost unchanged (0.9614 → 0.9608 on all dev) while the torso
and tail fall (0.8068 → 0.7931, 0.4093 → 0.3895): pooling flattens the ranking
mostly where the label support is thin.

Two mechanisms are consistent with the numbers, but this run does **not**
separate them:

1. **The chunk size did not remove the truncation it was meant to remove.** A
   post-hoc token count (read-only, no metric changed; recorded as
   `token_diagnostics` in the artifact) shows the whole document
   has a median of **556 tokens** and 98.7 % of documents exceed MiniLM's 256
   tokens — the baseline sees the front of the document only. But 200 words is
   a median of **250 tokens** and **42 % of chunks still exceed 256 tokens**, so
   the treatment is itself truncated on two-fifths of its inputs.
2. **The mean dilutes the title/header.** The whole-document view is front-
   loaded: title + header (~83 words) occupy a large share of the 256-token
   window the baseline actually reads. The chunk mean gives that signal 1/3.4
   of the weight and mixes in boilerplate recital chunks, which is a plausible
   reason the clean topic signal is weaker.

A chunk size that fits the 256-token context (or the front-loaded title kept as
its own view) would be a **different pre-registration**, not a re-read of this
one; the chunk size was fixed here at 200 words and is not re-tuned. The null
does not license menu item 2 by itself; it is the reviewer's call whether the
representation thread continues.

**What the result does not say.** Nothing about `test`; nothing about other
chunk sizes, overlap or pooling rules; nothing about longer-context models;
nothing about multi-view embeddings (menu item 2).

## 14. Multi-view document embedding — TC-8 (pre-registered 2026-11-13, before the run)

**Question.** The whole-document view concatenates four fields of very different
quality: a short clean `title` + `header` and long noisy `recitals` +
`main_body`. MiniLM's 256-token window is filled mostly by title+header+the
start of the recitals, so the fields are not weighted, they are *truncated into
place*. Does embedding the fields separately and combining them improve the
k-NN shortlist on `family_new`?

**Mechanism (fixed before the run).**

- **Views:** four per document, embedded as-is — `title`, `header`, `recitals`,
  and `main_body` (paragraphs joined by newlines). This is the same source text
  the baseline's `full_text` joins; only the packaging differs. A view longer
  than the model's 256-token context (typically the recitals) is truncated by
  the model, not chunked.
- **Embedding:** MiniLM-L6-v2 (revision
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`), the cached model, `document`
  kind, cache version **`eurlex-4k-views-v1`** (one entry per distinct field
  text).
- **Blank views:** two dataset documents have empty recitals. A blank view is
  **absent**: it is excluded from the mean, casts no k-NN votes and adds no RRF
  rank.
- **Treatment A — view mean (pre-registered primary):** L2-normalize each
  present view vector, take the arithmetic mean, L2-normalize the mean, and run
  the shared k-NN (`k = 100`, power 1.0) exactly as the baseline does.
- **Treatment B — rank fusion (secondary):** one k-NN per view (same
  `k = 100`, power 1.0), then **reciprocal-rank fusion** with `top_k = 100`
  (each view's top-100 labels contribute) and `offset = 60` (rank `r`
  contributes `1/(60 + r)`). No weight is tuned. The rank rule matters because
  the views' vote sums are not on a comparable scale.
- **Best single view (diagnostic only):** the four single-view k-NNs are
  reported to show which field carries the signal. They are **not** competing
  treatments, and no post-hoc selection of the best view is licensed by this
  run.

**Slices.** `all_dev` (6,000), `family_new` (4,260) and `family_new_clean`
(3,764), exactly as sections 9 and 13. The baseline is re-scored in the same run
from the warm `eurlex-4k-v1` cache and reported beside both treatments; test is
never read.

**Metrics.** Primary: R@50, R-precision, nDCG@10. Secondary: R@5, R@10, R@20,
R@100, nDCG@50. Diagnostics: coverage@50 and label-level recall@50 by
head / torso / tail / unseen. A **paired bootstrap over documents** (2,000
resamples, percentile 95 % interval) gives the treatment-minus-baseline R@50
difference on each slice, with the same resampled rows for both methods.

**Pre-registered bar.** The **view mean** is the arm that decides the item:

- A **win** requires the view mean to improve `family_new` **R@50 by at least
  0.01**, with the paired 95 % interval **excluding 0**, and to lose
  **R-precision and nDCG@10 by no more than 0.01** on `family_new`.
- The **rank fusion** is read against the same bar as a secondary arm. A win on
  the fusion alone is recorded as **`secondary_win_only`**: it does not adopt
  the method without the reviewer, because the primary arm is the registered
  decision.
- Otherwise the item is a **null**: record it and do not proceed to menu item 3.

**Cost.** 204,000 view embeddings (4 views × 51,000 documents, deduplicated by
text in the cache), a MiniLM CPU pass of roughly 15–30 min: title/header/
main_body are short, the recitals are truncated at 256 tokens. Then six k-NN
passes (baseline + 4 views + view mean; ~2 min), the RRF fusion (seconds), the
evaluation and the bootstrap. No new model, no LLM.

**Not tested here.** Other combination rules (max, weighted mean), other view
sets, chunking inside a view, a longer-context model (the other half of menu
item 1), and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-multiview/metrics.json` (run `eurlex-multiview`, no
LLM). Provenance: the first pass embedded **199,010 view texts in 907.2 s**
(4,988 cache hits) and crashed **after** the evaluation on a reporting bug, so
it wrote no artifact; the final run is the same deterministic computation on
the now-warm view cache (**203,998 hits / 0 misses, 134.9 s**). The baseline is
re-scored in the run from `eurlex-4k-v1` (**51,000 hits / 0 misses**) and
reproduces section 9 exactly. 203,998 view vectors (4 views × 51,000 documents,
deduplicated by text); two documents have blank recitals (absent view). Train =
45,000-document index, dev = 6,000 evaluated; test unread.

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| method | all_dev | family_new | family_new_clean |
|---|---|---|---|
| baseline — whole document | 0.9168 / 0.5715 / 0.7070 | **0.8944** / 0.4869 / 0.6364 | 0.8862 / 0.4744 / 0.6241 |
| view mean (primary) | 0.9168 / 0.5727 / 0.7092 | 0.8942 / 0.4894 / 0.6397 | 0.8863 / 0.4776 / 0.6272 |
| **rank fusion (secondary)** | **0.9253 / 0.5891 / 0.7277** | **0.9055 / 0.5089 / 0.6605** | **0.8980 / 0.4958 / 0.6464** |

**Secondary metrics by slice (R@5 / R@10 / R@20 / R@100 / nDCG@50):**

| slice | method | R@5 | R@10 | R@20 | R@100 | nDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| all_dev | baseline | 0.5623 | 0.7230 | 0.8306 | 0.9473 | 0.7838 |
| all_dev | view mean | 0.5645 | 0.7253 | 0.8309 | 0.9477 | 0.7850 |
| all_dev | **rank fusion** | **0.5833** | **0.7508** | **0.8528** | **0.9500** | **0.7979** |
| family_new | baseline | 0.4839 | 0.6524 | 0.7843 | 0.9342 | 0.7323 |
| family_new | view mean | 0.4865 | 0.6552 | 0.7844 | 0.9343 | 0.7343 |
| family_new | **rank fusion** | **0.5083** | **0.6849** | **0.8120** | **0.9371** | **0.7492** |
| family_new_clean | baseline | 0.4742 | 0.6408 | 0.7721 | 0.9287 | 0.7210 |
| family_new_clean | view mean | 0.4775 | 0.6432 | 0.7723 | 0.9285 | 0.7231 |
| family_new_clean | **rank fusion** | **0.4981** | **0.6714** | **0.7988** | **0.9313** | **0.7371** |

**Diagnostics (coverage@50; label-level recall@50 by head / torso / tail /
unseen):**

| slice | method | coverage@50 | head | torso | tail | unseen |
|---|---|---:|---:|---:|---:|---:|
| all_dev | baseline | 0.9978 | 0.9614 | 0.8068 | 0.4093 | 0.0 |
| all_dev | view mean | 0.9975 | 0.9622 | 0.7993 | 0.4201 | 0.0 |
| all_dev | **rank fusion** | **0.9980** | **0.9685** | **0.8161** | **0.4566** | 0.0 |
| family_new | baseline | 0.9972 | 0.9489 | 0.7995 | 0.4011 | 0.0 |
| family_new | view mean | 0.9967 | 0.9501 | 0.7913 | 0.4085 | 0.0 |
| family_new | **rank fusion** | **0.9974** | **0.9588** | **0.8086** | **0.4511** | 0.0 |
| family_new_clean | baseline | 0.9968 | 0.9441 | 0.7872 | 0.3868 | 0.0 |
| family_new_clean | view mean | 0.9963 | 0.9454 | 0.7795 | 0.3926 | 0.0 |
| family_new_clean | **rank fusion** | **0.9971** | **0.9551** | **0.7954** | **0.4353** | 0.0 |

**Single-view k-NN diagnostics on `family_new` (R@50 / R-precision /
nDCG@10)** — the fields are not competing treatments, only a read on where the
signal is:

| view | family_new | alone vs whole document |
|---|---|---|
| whole document (baseline) | **0.8944 / 0.4869 / 0.6364** | — |
| title | 0.8739 / 0.4794 / 0.6264 | −0.0205 R@50 |
| header | 0.8701 / 0.4731 / 0.6162 | −0.0243 R@50 |
| recitals | 0.8820 / 0.4574 / 0.6053 | −0.0123 R@50 |
| main_body | 0.7889 / 0.3779 / 0.4982 | −0.1055 R@50 |

**Paired bootstrap over documents (treatment − baseline R@50, 2,000
resamples):**

| treatment | all_dev | family_new | family_new_clean |
|---|---|---|---|
| view mean | +0.00001 [−0.0022, +0.0023] | −0.0002 [−0.0032, +0.0029] | +0.0001 [−0.0032, +0.0036] |
| rank fusion | **+0.0085 [+0.0060, +0.0110]** | **+0.0112 [+0.0080, +0.0145]** | **+0.0117 [+0.0082, +0.0153]** |

**The bar, read out.**

- **Primary (view mean): null.** `family_new` R@50 changes by **−0.0002**, far
  below the +0.01 bar, with the interval straddling zero. It does not lose
  R-precision (+0.0025) or nDCG@10 (+0.0032) — a tiny ordering gain, but not a
  shortlist gain. On `all_dev` and clean it is flat to four decimals.
- **Secondary (rank fusion): the bar is met.** `family_new` R@50 **+0.0112**
  with the interval entirely above zero; R-precision **+0.0220** and nDCG@10
  **+0.0241** both improve (no loss). The same direction holds on `all_dev`
  (+0.0085) and clean (+0.0117). The preregistered outcome is
  **`secondary_win_only`**: by the pre-registration this does **not** adopt the
  method without the reviewer, because the primary arm decides the item.
- **No bucket collapses.** The fusion improves every bucket: head 0.9489 →
  0.9588, torso 0.7995 → 0.8086, tail **0.4011 → 0.4511** on `family_new`.
  The unseen bucket stays 0.0 (no arm can retrieve labels with no train
  example). Coverage@50 rises slightly (0.9972 → 0.9974).

**Reading.** Combining the views by **averaging their vectors** is a flat null —
the same failure mode as TC-7: a unit-weight mean dilutes whichever view is
confident. Combining them by **fusing their rankings** works: each field is
individually worse than the concatenated document (title 0.8739, recitals
0.8820, header 0.8701, main_body 0.7889 R@50 on `family_new` against 0.8944),
yet their rank fusion reaches **0.9055** — so the gain is **complementarity, not
one strong view**. The fusion keeps each view's own top-100, and the different
views disagree in a way that covers more of the tail: the tail recall rises by
0.05, the largest bucket move.

This is the first positive result of the track, and it is a **secondary-arm
win**: +0.0112 R@50 / +0.022 R-precision / +0.024 nDCG@10 on the honest
`family_new` slice, with a paired interval excluding zero. It does not license
menu item 3 by itself, and it does not license adopting rank fusion without the
reviewer: the pre-registered primary (view mean) is a null.

**What the result does not say.** Nothing about `test`; nothing about other
fusion rules (weighted RRF, score z-scoring), other `top_k`/`offset` values
(fixed here, not tuned), other view sets, chunking inside a view, or a
longer-context model. The four single-view numbers are diagnostics — no
post-hoc view selection is licensed.
