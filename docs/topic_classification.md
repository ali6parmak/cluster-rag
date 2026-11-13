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

## 15. Lexical k-NN + reciprocal-rank fusion — TC-9 (pre-registered 2026-11-13, before the run)

**Question.** The embedding k-NN is the current shortlist generator (0.8944
R@50 on `family_new`). A lexical k-NN over the same text has a different error
profile — exact word overlap rather than a 384-dimensional semantic space. Does
fusing the two rankings improve the shortlist, and does it help the tail?

**Mechanism (fixed before the run).**

- **Baseline — embedding k-NN:** the whole-document MiniLM view, `k = 100`,
  similarity-weighted votes (power 1.0), served from the warm `eurlex-4k-v1`
  cache and re-scored in this run. This is the comparator, not a treatment.
- **Lexical view:** the same `full_text` (title + header + recitals +
  main_body) as the embedding baseline. TF-IDF with word 1–2 grams,
  `min_df = 2`, `max_features = 200,000`, `sublinear_tf = True`,
  `lowercase = True`, L2-normalized rows. Fitted on **train only** and applied
  to train and dev. This is the TF-IDF stand-in for BM25 that needs no new
  dependency (scikit-learn is already a core dependency); BM25 proper is not
  tested here.
- **Lexical k-NN:** cosine similarity (the dot product of unit rows), the same
  `k = 100` and power 1.0, and the **same voting rule** as the embedding k-NN
  (stable descending sort, similarities clipped at 0), applied to the
  batch-wise similarity block because the 45,000 × 200,000 TF-IDF matrix cannot
  be densified.
- **Fusion (the treatment): reciprocal-rank fusion** of the embedding ranking
  and the lexical ranking — equal weight, `top_k = 50` per view (the plan's
  "take its top-50"), `offset = 60` (rank `r` contributes `1/(60 + r)`). No
  weight is tuned. The fused candidate pool is the union of the two top-50
  lists, so R@100 is capped by that union; R@50 is the primary and is not
  affected by the cap.
- **No new model, no re-embedding, no LLM.**

**Slices.** `all_dev` (6,000), `family_new` (4,260) and `family_new_clean`
(3,764), exactly as sections 9, 13 and 14. Test is never read.

**Metrics.** Primary: R@50, R-precision, nDCG@10. Secondary: R@5, R@10, R@20,
R@100, nDCG@50. Diagnostics: coverage@50 and label-level recall@50 by
head / torso / tail / unseen. The **lexical k-NN alone** is reported as a
diagnostic — its error profile and tail recall are the reason the fusion is
interesting, but it is **not** a competing treatment. A paired bootstrap over
documents (2,000 resamples) gives the fusion-minus-baseline R@50 difference on
each slice.

**Pre-registered bar.**

- A **win** requires the fusion to improve `family_new` **R@50 by at least
  0.01**, with the paired 95 % interval **excluding 0**, and to lose
  **R-precision and nDCG@10 by no more than 0.01** on `family_new`.
- Otherwise it is a **null**: record it and do not proceed to menu item 4.
- **No auto-adopt.** Even a win is a result for the reviewer, not an adoption.

**Cost.** TF-IDF fit and transform over 51,000 documents (~1–3 min), the
6,000 × 45,000 sparse cosine in batches (~2–5 min), the metrics and bootstrap
(~2 min). Minutes total; no GPU, no LLM, no model download.

**Not tested here.** BM25 proper, other n-gram ranges or vocabulary sizes,
other `top_k`/`offset` values, weighted fusion, and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-lexical-rrf/metrics.json` (run `eurlex-lexical-rrf`,
**108.8 s**, no LLM). The embedding baseline was re-scored from the warm
`eurlex-4k-v1` cache (51,000 hits / 0 misses) and reproduces section 9 exactly.
TF-IDF: vocabulary **200,000** (the `max_features` cap binds), **23,170,645**
train non-zeros and **3,023,996** dev non-zeros, `min_df=2`, word 1–2 grams,
sublinear TF. Train = 45,000-document index, dev = 6,000 evaluated; test unread.

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| method | all_dev | family_new | family_new_clean |
|---|---|---|---|
| baseline — embedding k-NN | 0.9168 / 0.5715 / 0.7070 | 0.8944 / 0.4869 / 0.6364 | 0.8862 / 0.4744 / 0.6241 |
| lexical TF-IDF k-NN (diagnostic) | 0.9317 / 0.6152 / 0.7517 | 0.9138 / 0.5415 / 0.6902 | 0.9065 / 0.5293 / 0.6771 |
| **RRF fusion (treatment)** | **0.9337 / 0.6164 / 0.7581** | **0.9165 / 0.5428 / 0.6997** | **0.9094 / 0.5317 / 0.6882** |

**Secondary metrics by slice (R@5 / R@10 / R@20 / R@100 / nDCG@50):**

| slice | method | R@5 | R@10 | R@20 | R@100 | nDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| all_dev | baseline | 0.5623 | 0.7230 | 0.8306 | 0.9473 | 0.7838 |
| all_dev | lexical | 0.6054 | 0.7684 | 0.8678 | **0.9512** | 0.8178 |
| all_dev | **fusion** | **0.6112** | **0.7764** | **0.8740** | 0.9481 | **0.8220** |
| family_new | baseline | 0.4839 | 0.6524 | 0.7843 | 0.9342 | 0.7323 |
| family_new | lexical | 0.5366 | 0.7068 | 0.8304 | **0.9390** | 0.7739 |
| family_new | **fusion** | **0.5435** | **0.7187** | **0.8397** | 0.9355 | **0.7800** |
| family_new_clean | baseline | 0.4742 | 0.6408 | 0.7721 | 0.9287 | 0.7210 |
| family_new_clean | lexical | 0.5272 | 0.6936 | 0.8181 | **0.9344** | 0.7628 |
| family_new_clean | **fusion** | **0.5360** | **0.7075** | **0.8298** | 0.9299 | **0.7697** |

The fusion's R@100 is below the other two arms by construction: the fused
candidate pool is the union of the two top-50 lists, so it cannot reach a label
that neither view ranked in its top 50.

**Diagnostics (coverage@50; label-level recall@50 by head / torso / tail /
unseen):**

| slice | method | coverage@50 | head | torso | tail | unseen |
|---|---|---:|---:|---:|---:|---:|
| all_dev | baseline | 0.9978 | 0.9614 | 0.8068 | 0.4093 | 0.0 |
| all_dev | lexical | 0.9985 | 0.9682 | 0.8475 | **0.5414** | 0.0 |
| all_dev | **fusion** | **0.9987** | **0.9720** | 0.8454 | 0.4862 | 0.0 |
| family_new | baseline | 0.9972 | 0.9489 | 0.7995 | 0.4011 | 0.0 |
| family_new | lexical | 0.9981 | 0.9578 | **0.8426** | **0.5404** | 0.0 |
| family_new | **fusion** | **0.9984** | **0.9633** | 0.8403 | 0.4830 | 0.0 |
| family_new_clean | baseline | 0.9968 | 0.9441 | 0.7872 | 0.3868 | 0.0 |
| family_new_clean | lexical | 0.9979 | 0.9541 | **0.8312** | **0.5208** | 0.0 |
| family_new_clean | **fusion** | **0.9981** | **0.9598** | 0.8284 | 0.4642 | 0.0 |

**Paired bootstrap over documents (2,000 resamples):**

| comparison | all_dev | family_new | family_new_clean |
|---|---|---|---|
| fusion − embedding (the bar) | **+0.0169 [+0.0149, +0.0189]** | **+0.0222 [+0.0195, +0.0250]** | **+0.0231 [+0.0202, +0.0260]** |
| lexical − embedding (diagnostic) | +0.0149 [+0.0121, +0.0177] | +0.0194 [+0.0157, +0.0231] | +0.0203 [+0.0161, +0.0243] |
| fusion − lexical (post-hoc) | +0.0019 [+0.0002, +0.0037] | +0.0028 [+0.0004, +0.0053] | +0.0028 [+0.0001, +0.0055] |

**The bar, read out.**

- **All four conditions are met.** `family_new` R@50 **+0.0222** with the
  interval entirely above zero; R-precision **+0.0560** and nDCG@10 **+0.0633**
  both improve (no loss). The same direction holds on `all_dev` (+0.0169) and
  clean (+0.0231).
- **The lexical arm is not a weak add-on — it is the stronger single
  retriever.** On `family_new` it reaches **0.9138** R@50, **+0.0194** over the
  embedding baseline, and **0.5404** tail recall against the embedding's
  **0.4011**. Its R-precision (+0.055) and nDCG@10 (+0.054) are also ahead. The
  fusion's win over the baseline is mostly this arm.
- **RRF adds a little over its stronger arm.** Fusion − lexical is **+0.0028
  R@50 [+0.0004, +0.0053]** on `family_new` (small but positive), +0.0014
  R-precision and +0.0095 nDCG@10. The extra is mostly ordering, not breadth.
- **The fusion does not help the tail relative to the lexical arm.** The
  lexical tail is **0.5404**; the fusion's is **0.4830** (−0.057). Equal-weight
  RRF rewards labels both views agree on, which are disproportionately head
  labels, so it dilutes the lexical arm's tail advantage. Against the
  *embedding* baseline the fusion still improves the tail (+0.082).

**Reading.** The plan asked whether a lexical k-NN fused by RRF helps, and
whether it helps the tail. The answer is **yes against the embedding baseline**
— the fusion clears the pre-registered bar on every slice, driven by a much
stronger honest-slice and tail signal — but the honest decomposition is more
interesting than the fused number: **the TF-IDF k-NN alone beats the embedding
k-NN on `family_new`** (0.9138 vs 0.8944, +0.0194) and roughly doubles its tail
recall (0.5404 vs 0.4011), and RRF adds only **+0.0028** R@50 over it while
*reducing* the tail to 0.4830. On EURLEX, as on UPR, the words carry most of the
shortlist signal; the embedding's advantage is not where the track assumed.

This is the first change set that **wins its pre-registered bar outright**. It
is not an adoption: the reviewer decides what to take forward, and the
decomposition (lexical alone vs fusion) is the decision that matters.

**What the result does not say.** Nothing about `test`; nothing about BM25
proper, other n-gram ranges/vocabulary caps, other `top_k`/`offset` values or
weighted fusion; nothing about a lexical-plus-embedding learned combiner; and
nothing about the tail dilution as a fixable weighting problem.

## 16. Label co-occurrence smoothing — TC-10 (pre-registered 2026-11-13, before the run)

**Question.** The k-NN votes label by label; a label with many neighbours votes
for itself and nothing for the labels that normally travel with it. EURLEX
documents carry ~5 labels each, and some labels co-occur far more than chance.
Does adding a **train-only label co-occurrence boost** to the k-NN scores
improve the shortlist — and does it help the tail, as the plan expects?

**Mechanism (fixed before the run).**

- **Primary base — embedding k-NN:** the whole-document MiniLM view, `k = 100`,
  power 1.0, re-scored from the warm `eurlex-4k-v1` cache. This is the track's
  standing baseline and the base the pre-registered bar is read on.
- **Secondary bases (exploratory):** the TF-IDF k-NN of TC-9 and the TC-9
  RRF(embedding, lexical) scores, smoothed by the same rule. They are reported
  because the lexical arm is the stronger retriever (§15), but they **do not**
  decide the item.
- **Co-occurrence matrix (train only):** `C[a, b]` = the number of train
  documents carrying both labels `a` and `b`, `a != b` (zero diagonal), and
  `P[b | a] = C[a, b] / sum_b C[a, b]` (row-conditional). A label with no train
  example keeps an all-zero row, so the smoothing cannot invent a zero-shot
  candidate.
- **Boost:** for each document, take its **top `M = 10`** labels by base score;
  their non-negative base scores are renormalised to sum 1 and used as weights
  over the co-occurrence rows: `boost = sum_a w_a * P[., a]`. A document whose
  top-10 scores are all zero gets an all-zero boost.
- **Smoothing:** per-row z-score the base and the boost (the TC-5 `row_zscore`
  convention), then `smoothed = z(base) + alpha * z(boost)`. At `alpha = 0`
  the ranking is exactly the base's.
- **Alpha selection:** `alpha ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}`,
  selected by **5-fold group cross-validation on dev**, folds grouped by the
  title skeleton (whole families held out), exactly as TC-5. Each dev document
  is scored by its held-out fold's alpha, and the pooled matrix is evaluated
  once. The full-dev grid is reported beside the CV number as tuned.

**Slices.** `all_dev` (6,000), `family_new` (4,260), `family_new_clean`
(3,764). Test is never read.

**Metrics.** Primary: R@50, R-precision, nDCG@10. Secondary: R@5, R@10, R@20,
R@100, nDCG@50. Diagnostics: coverage@50 and label-level recall@50 by
head / torso / tail / unseen, with the **tail** called out. A paired bootstrap
over documents (2,000 resamples) gives the smoothed-minus-base R@50 difference
on each slice for each base.

**Pre-registered bar.**

- A **win** requires the primary (embedding) base's cross-validated smoothed
  scores to improve `family_new` **R@50 by at least 0.01**, with the paired
  95 % interval **excluding 0**, and to lose **R-precision and nDCG@10 by no
  more than 0.01** on `family_new`.
- A collapse on any bucket (head / torso / tail / unseen) is reported even if
  the pooled number rises.
- Otherwise it is a **null**: record it and do not proceed to menu item 5.
- **No auto-adopt.** The reviewer decides.

**Cost.** The co-occurrence matrix from the train labels (seconds), three
boosts (top-10 × dense matrix, ~15 s each), the 5-fold × 8-alpha grid on dev
(~2 min), the metrics and bootstrap. No new model, no re-embedding pass beyond
the cache, no LLM.

**Not tested here.** PMI or other co-occurrence normalisations, other `M`
values, a learned combiner over co-occurrence features, and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-cooccurrence/metrics.json` (run `eurlex-cooccurrence`,
**178.2 s**, no LLM; embedding cache 51,000 hits / 0 misses). The train-only
co-occurrence matrix has **4,107** labels with a row (of 4,197), **224,424**
directed non-zero pairs. The 89 dev zero-shot labels have no row (zero train
support), and one train label appears only in a single-label document, so 90 of
the 4,197 labels cannot be boosted by construction. All three bases were
re-scored in the run and reproduce their sections: embedding
0.8944, lexical 0.9138, RRF 0.9165 `family_new` R@50.

**Selected alphas (5-fold family-grouped CV, one per fold):**

| base | fold alphas | CV-smoothed family_new R@50 vs base |
|---|---|---|
| embedding k-NN (primary) | **0, 0, 0, 0, 0** | 0.8944 vs 0.8944 (Δ **0.0000**) |
| lexical k-NN (exploratory) | 0.05, 0.05, 0.05, 0.05, 0.05 | 0.9144 vs 0.9138 (Δ +0.0007) |
| RRF fusion (exploratory) | 0, 0.05, 0, 0, 0 | 0.9162 vs 0.9165 (Δ −0.0003) |

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| base | slice | unsmoothed | CV-smoothed |
|---|---|---|---|
| embedding | all_dev | 0.9168 / 0.5715 / 0.7070 | 0.9168 / 0.5715 / 0.7070 |
| embedding | family_new | 0.8944 / 0.4869 / 0.6364 | 0.8944 / 0.4869 / 0.6364 |
| embedding | family_new_clean | 0.8862 / 0.4744 / 0.6241 | 0.8862 / 0.4744 / 0.6241 |
| lexical | all_dev | 0.9317 / 0.6152 / 0.7517 | 0.9327 / 0.6125 / 0.7491 |
| lexical | family_new | 0.9138 / 0.5415 / 0.6902 | 0.9144 / 0.5383 / 0.6869 |
| lexical | family_new_clean | 0.9065 / 0.5293 / 0.6771 | 0.9071 / 0.5260 / 0.6736 |
| RRF | all_dev | 0.9337 / 0.6164 / 0.7581 | 0.9336 / 0.6107 / 0.7528 |
| RRF | family_new | 0.9165 / 0.5428 / 0.6997 | 0.9162 / 0.5386 / 0.6941 |
| RRF | family_new_clean | 0.9094 / 0.5317 / 0.6882 | 0.9090 / 0.5278 / 0.6825 |

**Full-dev R@50 grid on `family_new` (tuned on dev, reported for shape):**

| alpha | embedding | lexical | RRF |
|---:|---:|---:|---:|
| 0 | **0.8944** | 0.9138 | **0.9165** |
| 0.05 | 0.8906 | **0.9144** | 0.9164 |
| 0.1 | 0.8878 | 0.9137 | 0.9161 |
| 0.2 | 0.8831 | 0.9102 | 0.9154 |
| 0.3 | 0.8786 | 0.9064 | 0.9144 |
| 0.5 | 0.8719 | 0.8998 | 0.9124 |
| 0.75 | 0.8610 | 0.8918 | 0.9100 |
| 1.0 | 0.8536 | 0.8838 | 0.9076 |

**Tail bucket (`family_new` label-level recall@50):** embedding **0.4011 →
0.4011** (α = 0); lexical **0.5404 → 0.5330**; RRF **0.4830 → 0.4819**. At
larger α the tail falls with the rest; there is no α in the grid where it rises.

**Paired bootstrap over documents (CV-smoothed − base R@50, 2,000 resamples):**

| base | all_dev | family_new | family_new_clean |
|---|---|---|---|
| embedding | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| lexical | +0.0009 [+0.0003, +0.0016] | +0.0007 [−0.0001, +0.0015] | +0.0006 [−0.0003, +0.0015] |
| RRF | −0.0001 [−0.0004, +0.0001] | −0.0003 [−0.0006, −0.0001] | −0.0003 [−0.0007, −0.0001] |

**The bar, read out.**

- **R@50 gain ≥ 0.01 on the primary base: not met.** The CV selects **α = 0 in
  every fold**, so the smoothed embedding scores are the baseline exactly (the
  delta is 0 by construction, not a rounding artifact).
- **Interval excluding 0: not met** (the interval is [0, 0]).
- **The grid explains the selection.** For the embedding base, R@50 falls
  monotonically from 0.9168 at α = 0 to 0.8856 at α = 1 (all dev), and
  `family_new` from 0.8944 to 0.8536. There is no interior optimum: the boost
  is not aligned with the k-NN's ranking, only a weaker version of it.
- **The stronger bases do not rescue it.** Lexical's best α (0.05) buys
  +0.0007 `family_new` R@50 with an interval touching zero, and costs
  R-precision (−0.0032) and nDCG@10 (−0.0033); the RRF base is slightly
  *negative*. Both are exploratory and neither passes the bar.
- **The tail hypothesis fails.** The plan expected a head label to lift its
  tail neighbours; instead the tail falls at every non-zero α on the lexical
  base (0.5404 → 0.5330) and is flat on the embedding base (α = 0).

**Reading.** Co-occurrence smoothing is a **null**. At 4,197 labels the
co-occurrence matrix is broad and head-dominated, and its z-scored boost mostly
re-weights labels the k-NN already found: the boost's own ranking is weaker
than the k-NN's, so adding it can only dilute. The 5-fold family-grouped CV
recognises this and selects α = 0, which is the honest outcome rather than a
knob to force. The co-occurrence idea is not refuted in every form — a PMI
matrix, a per-bucket alpha, or a learned combiner over co-occurrence features
remain untested — but this cheap additive form does not license menu item 5.

**What the result does not say.** Nothing about `test`; nothing about PMI or
other normalisations, other `M`, per-bucket alphas, or a learned combiner; and
nothing about whether co-occurrence helps a *reranker* rather than a shortlist.

## 17. Zero-shot quota — TC-11 (pre-registered 2026-11-13, before the run)

**Question.** TC-6 showed the title signal is real but does not survive the full
ranking when it is *blended* into the k-NN scores: the calibrated channel found
9 of 96 unseen assignments (0.0938) and missed the 0.10 bar by one hit. A
**quota** is a different insertion rule: instead of competing on the score
scale, reserve N shortlist slots for the N best zero-shot labels by title
cosine. Does that put zero-shot labels on the shortlist without costing the
overall R@50?

**Mechanism (fixed before the run).**

- **Primary base — embedding k-NN:** the whole-document MiniLM view, `k = 100`,
  power 1.0, re-scored from the warm `eurlex-4k-v1` cache. The TF-IDF k-NN and
  the TC-9 RRF fusion are **exploratory** secondary bases; they do not decide
  the item.
- **Zero-shot set:** the **89** dev labels with zero train support (96 gold
  assignments across 87 dev documents). The label-title vectors are the cached
  `eurlex-4k-label-v1` embeddings; each dev document is scored by cosine to
  those 89 titles only, and the candidates are ranked per document by that
  cosine.
- **Insertion rule (the quota):** reserve the **bottom N slots of the top-50** —
  the N reserved labels replace the base ranking's weakest entries (positions
  `50-N+1 .. 50`), and the displaced entries move to positions `51 .. 50+N`.
  A candidate already ranked inside the top-50 is skipped (it is already on the
  shortlist). The reserved labels are the top-N by title cosine.
- **Why the bottom:** by construction a reserved label can never enter
  R-precision (`k = |gold| ≈ 5`) or the nDCG@10 window, so the quota can only
  cost the documents whose gold sat in the displaced positions, and it can only
  help the unseen bucket. This is the minimal-cost placement, not a reranker.
- **Quota selection:** `N ∈ {0, 1, 2, 3, 5, 8, 10, 15, 20}`, selected by
  **5-fold group cross-validation on dev**, folds grouped by the title skeleton
  (whole families held out). On each fold's training documents the selected N
  **maximises the unseen-bucket R@50 subject to an all-dev R@50 loss ≤ 0.01**
  against the base; ties go to the smaller N, and if no N qualifies the fold
  selects 0. Each dev document is then scored by its held-out fold's N and the
  pooled matrix is evaluated once. The full-dev grid is reported as tuned.
- **No new model, no LLM**; the vectors are cached.

**Slices.** `all_dev` (6,000), `family_new` (4,260), `family_new_clean`
(3,764). Test is never read.

**Metrics.** Primary: R@50, R-precision, nDCG@10. Secondary: R@5, R@10, R@20,
R@100, nDCG@50. Diagnostics: coverage@50 and label-level recall@50 by
head / torso / tail / **unseen** — the unseen bucket (hits / 96 on `all_dev`)
is the headline. A paired bootstrap over documents (2,000 resamples) gives the
quota-minus-base R@50 difference on each slice.

**Pre-registered bar.**

- A **win** requires, on the pooled CV matrix: the **unseen-bucket R@50 ≥
  0.10** (at least 10 of the 96 `all_dev` assignments), with an **`all_dev`
  R@50 loss ≤ 0.01** *and* a **`family_new` R@50 loss ≤ 0.01** against the
  base. A quota that buys the unseen bucket by damaging the honest slice is not
  a win.
- Otherwise it is a **null**: record it and do not proceed to menu item 6.
- **No auto-adopt.** The reviewer decides.

**Cost.** The cached title and document vectors, the 6,000 × 89 cosine matrix
(seconds), the quota insertion over the grid (a few minutes on CPU), the
5-fold selection, and the bootstrap. No model download, no LLM.

**Not tested here.** A quota on the group/classifier labels beyond the 89
zero-shot ones, a learned insertion ranker, other `N` grids, a quota that
inserts inside the top-10, and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-zero-shot-quota/metrics.json` (run
`eurlex-zero-shot-quota`, **133.9 s**, no LLM). Vectors: documents 51,000 cache
hits / 0 misses (`eurlex-4k-v1`), label titles 4,197 hits / 0 misses
(`eurlex-4k-label-v1`). The base rankings reproduce their sections: embedding
0.9168, lexical 0.9317, RRF 0.9337 `all_dev` R@50; every base scores **0/96** on
the unseen bucket before the quota.

**Selected quotas (5-fold family-grouped CV):** the embedding base selects
**5** in all five folds; the lexical and RRF bases select **10** in all five.

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| base | slice | before quota | after quota (CV) |
|---|---|---|---|
| embedding | all_dev | 0.9168 / 0.5715 / 0.7070 | 0.9108 / 0.5715 / 0.7070 |
| embedding | family_new | 0.8944 / 0.4869 / 0.6364 | 0.8862 / 0.4869 / 0.6364 |
| embedding | family_new_clean | 0.8862 / 0.4744 / 0.6241 | 0.8778 / 0.4744 / 0.6241 |
| lexical | all_dev | 0.9317 / 0.6152 / 0.7517 | 0.9227 / 0.6152 / 0.7517 |
| lexical | family_new | 0.9138 / 0.5415 / 0.6902 | 0.9016 / 0.5415 / 0.6902 |
| RRF | all_dev | 0.9337 / 0.6164 / 0.7581 | 0.9241 / 0.6164 / 0.7581 |
| RRF | family_new | 0.9165 / 0.5428 / 0.6997 | 0.9041 / 0.5428 / 0.6997 |

R-precision and nDCG@10 are **identical** before and after: the reserved labels
sit at positions 46–50, outside both windows. R@5, R@10 and R@20 are identical
too; only R@50, R@100, nDCG@50 and coverage move.

**Diagnostics (coverage@50; label-level recall@50 by head / torso / tail /
unseen; the unseen column is hits / assignments):**

| base | slice | coverage@50 | head | torso | tail | unseen |
|---|---|---|---|---|---|---|
| embedding | all_dev | 0.9978 → **0.9980** | 0.9614 → 0.9573 | 0.8068 → 0.7893 | 0.4093 → 0.3718 | **0/96 → 34/96 (0.3542)** |
| embedding | family_new | 0.9972 → 0.9972 | 0.9489 → 0.9434 | 0.7995 → 0.7803 | 0.4011 → 0.3617 | **0/88 → 31/88 (0.3523)** |
| embedding | family_new_clean | 0.9968 → 0.9968 | 0.9441 → 0.9381 | 0.7872 → 0.7673 | 0.3868 → 0.3464 | **0/86 → 31/86 (0.3605)** |
| lexical | all_dev | 0.9985 → 0.9983 | 0.9682 → 0.9613 | 0.8475 → 0.8269 | 0.5414 → 0.4813 | **0/96 → 47/96 (0.4896)** |
| RRF | all_dev | 0.9987 → 0.9988 | 0.9720 → 0.9649 | 0.8454 → 0.8227 | 0.4862 → 0.4290 | **0/96 → 47/96 (0.4896)** |

**Full-dev grid (tuned; `family_new` R@50 and the unseen bucket):**

| N | embedding: unseen | embedding: all_dev / family_new R@50 | lexical: unseen | lexical: family_new R@50 |
|---:|---:|---|---:|---:|
| 0 | 0/96 | 0.9168 / 0.8944 | 0/96 | 0.9138 |
| 1 | 12/96 | 0.9158 / 0.8930 | 12/96 | 0.9131 |
| 2 | 17/96 | 0.9144 / 0.8911 | 17/96 | 0.9125 |
| 3 | 27/96 | 0.9132 / 0.8895 | 27/96 | 0.9115 |
| **5** | **34/96** | **0.9108 / 0.8862** | 34/96 | 0.9094 |
| 8 | 41/96 | 0.9064 / 0.8805 | 41/96 | 0.9055 |
| 10 | 47/96 | 0.9017 / 0.8743 | 47/96 | 0.9016 |
| 15 | 58/96 | 0.8931 / 0.8637 | 58/96 | 0.8924 |
| 20 | 62/96 | 0.8783 / 0.8443 | 62/96 | 0.8790 |

**Paired bootstrap over documents (quota − base R@50, 2,000 resamples):**

| base | all_dev | family_new | family_new_clean |
|---|---|---|---|
| embedding (the bar) | −0.0060 [−0.0071, −0.0049] | −0.0081 [−0.0096, −0.0067] | −0.0085 [−0.0102, −0.0069] |
| lexical (exploratory) | −0.0090 [−0.0104, −0.0077] | −0.0121 [−0.0140, −0.0103] | −0.0128 [−0.0148, −0.0109] |
| RRF (exploratory) | −0.0096 [−0.0109, −0.0082] | −0.0124 [−0.0143, −0.0106] | −0.0128 [−0.0147, −0.0108] |

**The bar, read out.**

- **Unseen R@50 ≥ 0.10: passed, with room.** **34 of 96** unseen assignments
  enter the shortlist on `all_dev` (0.3542) — against 0/96 for the base and
  9/96 for the best calibrated blend (TC-6). 31 of 88 within `family_new`
  (0.3523) and 31 of 86 on the clean slice (0.3605).
- **`all_dev` R@50 loss ≤ 0.01: passed.** It costs **0.0060** (0.9168 →
  0.9108), interval inside the cap.
- **`family_new` R@50 loss ≤ 0.01: passed.** It costs **0.0081** (0.8944 →
  0.8862), inside the cap — the guard I added to the bar is what the
  exploratory bases fail: the lexical and RRF bases find **47/96** with
  `N = 10`, but their `family_new` losses are **0.0121** and **0.0124**,
  outside the 0.01 cap, so they are **not** wins.
- **`outcome`: win** on the primary (embedding) base.

**Reading.** The quota works where blending failed. TC-4/TC-6 tried to make the
title cosine compete on the seen-label score scale and found 1/96 and 9/96; the
quota does not compete — it **reserves** — and finds **34/96** while staying
inside the loss cap. The placement is the whole trick: the reserved labels sit
at positions 46–50, so R-precision and nDCG@10 are untouched and the cost is
confined to the weakest shortlist entries; the price shows up as 0.6–0.9 points
of R@50 and ~4 points of tail recall (0.4011 → 0.3617 on `family_new`), not as
an ordering loss. The zero-shot labels are no longer unsolved: **a third of
their dev assignments now reach the shortlist**.

The exploratory bases are instructive: the stronger lexical base can reach
47/96, but at a `family_new` cost of 0.0121 — the quota simply consumes more of
its (already better) shortlist. The pre-registered guard stops that trade, and
the embedding base is the one that clears all three conditions.

**What the result does not say.** Nothing about `test`; nothing about a learned
insertion ranker, a quota inside the top-10, other `N` grids, or a quota for
labels beyond the 89 zero-shot ones; and nothing about whether the tail loss can
be recovered by combining the quota with the lexical arm's stronger tail.

## 18. Cardinality / threshold calibration — TC-12 (pre-registered 2026-11-13, before the run)

**Question.** Every item so far scores a **shortlist**; the product needs a
**label set**. The gold cardinality varies (mean 5.07, max 16 on dev), so a
fixed top-k forces the same set size on every document. Does a global score
threshold — calibrated to the train split's mean cardinality, so the set size
adapts per document — beat fixed top-5 on micro-F1 at the same average
cardinality?

**Mechanism (fixed before the run).**

- **Primary base — embedding k-NN:** the whole-document MiniLM view, `k = 100`,
  power 1.0, re-scored from the warm `eurlex-4k-v1` cache. The TF-IDF k-NN and
  the TC-9 RRF fusion are **exploratory** bases; they do not decide the item.
- **Gold:** the dev gold matrix over the 4,197 train+dev labels (the same space
  as every section since 8).
- **Control — top-5:** the five highest-scoring labels per document. This is
  the fixed-cardinality operating point at the train mean (5.074 labels/doc).
- **Treatment — global score threshold:** for a set of calibration documents,
  the predicted-cell budget is `round(5.074 × n)` (the **train** split's mean
  cardinality), and `tau` is the score at that global rank in the calibration
  scores; every label with `score >= tau` is predicted for its document. Ties at
  `tau` are all predicted, so the realised mean is reported, not assumed.
- **Threshold selection:** 5-fold group cross-validation on dev, folds grouped
  by the title skeleton (whole families held out). Each fold's `tau` is
  calibrated on the other four folds' documents and applied to the held-out
  documents; the pooled decision matrix is evaluated once. The full-dev
  calibration is reported beside it as **tuned**.
- **No per-document or per-bucket thresholds** in this change set; the plan
  allows "global (or per-bucket)" and this pre-registration fixes the global
  form only. No new model, no LLM.

**Slices.** `all_dev` (6,000), `family_new` (4,260), `family_new_clean`
(3,764). Test is never read.

**Metrics.** Primary: **micro-F1** of the decision matrix (the product metric).
Secondary: macro-F1, sample-F1, micro precision and micro recall. Diagnostics:
predicted cardinality (mean, median, mean absolute error against the gold
cardinality), and label-level precision/recall of the predicted sets by
head / torso / tail / unseen. The shortlist **R@50 is reported unchanged** (the
decision does not touch the ranking). A **paired bootstrap over documents**
(2,000 resamples) gives the interval of the threshold-minus-top-5 micro-F1 and
micro-recall differences, recomputing the pooled TP/FP/FN counts on each
resample.

**Pre-registered bar.**

- A **win** requires the threshold to improve `family_new` **micro-F1 by at
  least 0.01** over top-5, with the paired 95 % interval **excluding 0**, and
  **micro-recall not to drop by more than 0.01**. The realised `all_dev` mean
  cardinality must land within **±0.25** of the 5.074 target; if the
  calibration misses its own target, the reading is a null regardless of the F1.
- Otherwise it is a **null**: record it and do not proceed to menu item 7.
- **No auto-adopt.** The reviewer decides.

**Cost.** Re-scored cached embeddings, three base rankings, the threshold
calibration (a partition per fold), the decision metrics and the bootstrap.
Minutes on CPU; no model download, no LLM.

**Not tested here.** Per-bucket or per-document thresholds, a learned
cardinality predictor, F1-optimal threshold selection (the calibration targets
the **cardinality**, not the metric), and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-threshold-calibration/metrics.json` (run
`eurlex-threshold-calibration`, **167.4 s**, no LLM; embedding cache 51,000
hits / 0 misses). Target cardinality = the train mean **5.0738** labels/doc.
Thresholds: embedding ~**17.74–17.91** (per fold), lexical ~**6.73–6.80**,
RRF **0.0301**. The base rankings and their shortlist R@50 are unchanged
(embedding 0.9168 / 0.8944 / 0.8862 on the three slices; the decision never
touches the ranking).

**Primary metrics — embedding k-NN (top-5 control vs calibrated threshold):**

| slice | decision | micro P | micro R | **micro F1** | macro F1 | sample F1 | cardinality | MAE |
|---|---|---|---|---|---|---|---|---|
| all_dev | top-5 | 0.5363 | 0.5321 | 0.5342 | 0.1600 | 0.5326 | 5.000 | 1.291 |
| all_dev | **threshold** | 0.5608 | 0.5643 | **0.5625** | 0.1457 | 0.5331 | 5.071 | 1.992 |
| family_new | top-5 | 0.4752 | 0.4602 | 0.4675 | 0.1558 | 0.4663 | 5.000 | 1.267 |
| family_new | **threshold** | 0.5081 | 0.4667 | **0.4865** | 0.1360 | 0.4546 | 4.742 | 2.125 |
| family_new_clean | top-5 | 0.4621 | 0.4500 | 0.4560 | 0.1559 | 0.4553 | 5.000 | 1.247 |
| family_new_clean | **threshold** | 0.4998 | 0.4482 | **0.4726** | 0.1362 | 0.4426 | 4.604 | 2.133 |

**Exploratory bases, family_new (micro F1 / precision / recall / cardinality):**

| base | top-5 | threshold | Δ F1 |
|---|---|---|---|
| lexical k-NN | **0.5192** / 0.5277 / 0.5110 / 5.000 | 0.4946 / 0.5339 / 0.4607 / 4.454 | **−0.0246** |
| RRF fusion | 0.5249 / 0.5335 / 0.5166 / 5.000 | **0.5311** / 0.5385 / 0.5239 / 5.023 | +0.0062 |

**Per-bucket label-level micro-F1, embedding base, family_new:**

| decision | head | torso | tail | unseen |
|---|---|---|---|---|
| top-5 | 0.5190 | 0.2982 | 0.0000 | 0.0000 |
| threshold | **0.5470** | 0.2559 | 0.0000 | 0.0000 |

**Full-dev grid (`family_new` micro-F1 by target cardinality):**

| target | embedding | lexical |
|---:|---:|---:|
| 2 | 0.3387 | 0.2873 |
| 3 | 0.4308 | 0.4029 |
| 4 | 0.4729 | 0.4670 |
| **5** | **0.4862** | 0.4937 |
| 6 | 0.4855 | **0.4982** |
| 7 | 0.4748 | 0.4940 |
| 8 | 0.4612 | 0.4821 |
| 10 | 0.4299 | 0.4525 |

**Paired bootstrap over documents (threshold − top-5):**

| base | slice | Δ micro-F1 [95 % CI] | Δ micro-recall |
|---|---|---|---|
| embedding | all_dev | **+0.0283 [+0.0248, +0.0318]** | +0.0322 |
| embedding | family_new | **+0.0190 [+0.0150, +0.0233]** | +0.0065 |
| embedding | family_new_clean | **+0.0167 [+0.0122, +0.0211]** | −0.0018 |
| lexical | family_new | −0.0246 [−0.0297, −0.0195] | −0.0504 |
| RRF | family_new | +0.0062 [+0.0032, +0.0090] | +0.0073 |

**The bar, read out.**

- **Micro-F1 gain ≥ 0.01: passed.** **+0.0190** on `family_new` (+0.0283
  `all_dev`), interval entirely above zero.
- **Interval excluding 0: passed.**
- **Micro-recall loss ≤ 0.01: passed.** Recall **rises** by +0.0065 on
  `family_new`; it is flat on the clean slice (−0.0018).
- **Realised cardinality within ±0.25 of 5.074: passed on `all_dev`** (5.071).
  On `family_new` the realised cardinality is **4.742**: the fold calibrations
  see the easier family-seen documents and the threshold then predicts fewer
  labels on the harder new families. That is reported, not hidden; the bar's
  calibration condition was pre-registered on `all_dev`.
- **`outcome`: win** on the primary (embedding) base. The lexical base is
  **negative** (−0.0246: its top-5 is already the better set) and the RRF base
  gains **+0.0062**, below the 0.01 bar — both null.

**Reading.** The calibrated threshold beats top-5 on the embedding base, and
the mechanism is visible: it **reallocates** the set size rather than adding
signal. Precision rises (0.475 → 0.508 on `family_new`) while recall is roughly
flat, so micro-F1 gains **+0.019**; on new-family documents the same threshold
yields 4.74 labels/doc instead of 5. But the gain is **micro-level only**: the
per-document numbers fall (sample-F1 0.4663 → 0.4546, macro-F1 0.1558 →
0.1360) and the cardinality error rises (MAE 1.267 → 2.125) because the
threshold gives some documents larger sets and others smaller. The tail still
gets **zero** predictions from the embedding base — the threshold at ~5
labels/doc cannot reach labels whose scores sit far below the cut.

The exploratory bases split the lesson: on the stronger lexical ranking, fixed
top-5 is the better set predictor at every cardinality in the grid (the
threshold only redistributes an already-good ranking and loses 0.025 F1); on
the RRF ranking the threshold edges up +0.006, not enough to clear the bar. So
**the decision rule interacts with the base**, and "which set predictor to
ship" is not settled by this item — the embedding threshold wins its own
comparison, while the lexical top-5 remains the best `family_new` set
(0.5192) and the RRF threshold the best absolute micro-F1 (0.5311).

**What the result does not say.** Nothing about `test`; nothing about
per-bucket or per-document thresholds, a learned cardinality predictor, or an
F1-optimal threshold (the calibration targets the **cardinality**); and nothing
about recovering the tail, which no ~5-label operating point reaches.

## 19. Cross-encoder rerank of the lexical top-50 — TC-13 (pre-registered 2026-11-13, before the run)

**Question.** The lexical TF-IDF k-NN is the strongest single ranking on the
honest slice (0.9138 R@50 on `family_new`), but its **ordering** is the weak
point: R-precision 0.5415 and nDCG@10 0.6902. The top-50 is already ~91 %
complete, so the remaining gain is not breadth — it is putting the right labels
first. Does a cross-encoder rerank of the lexical top-50 improve the ordering?

**Mechanism (fixed before the run).**

- **Base — lexical k-NN:** the TC-9 arm — TF-IDF word 1–2 grams, `min_df = 2`,
  `max_features = 200,000`, sublinear TF, cosine, `k = 100`, power 1.0 — fitted
  on train and recomputed in this run. The rerank is compared with **this same
  base**, not with the embedding k-NN.
- **Candidates:** the base's **top-50** per dev document.
- **Pair:** query = the document's `full_text` (the same title + header +
  recitals + main_body view as every section), document = the candidate
  **label title**. The pair is truncated at the model's 512-token budget; the
  client records how many inputs it truncated.
- **Primary model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision
  `c5ee24cb16019beea0893ab7796b1df96625c6b8`, the existing `cross_encoder`
  kind (sigmoid of the classification logit), `float32`, `max_length = 512`,
  `batch_size = 128`, `device = cuda`.
- **Secondary model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`, revision
  `7b0235231ca2674cb8ca8f022859a6eba2b1c968`, same settings except
  `batch_size = 64`. It is a **secondary** read with the same bar; a win on
  L-12 alone is recorded as `secondary_win_only` and is not adopted.
- **Rerank:** sort the 50 candidates by cross-encoder score descending, ties
  broken by the base rank; positions 51+ keep the base order. **R@50 is
  unchanged by construction** (the candidate set is the same); the targets are
  nDCG@10 and R-precision.
- **Scores** are cached in the rerankers' own cache (`cache/rerank/`, separate
  from the embedding cache). No LLM, no test data.

**Slices.** `all_dev` (6,000), `family_new` (4,260), `family_new_clean`
(3,764).

**Metrics.** Primary: **nDCG@10** (ordering). Guard: **R-precision** (must not
fall by more than 0.005). Check: **R@50 identical** to the base (within
1e-9). Secondary: R@5, R@10, R@20, R@100, nDCG@50. Diagnostics: coverage@50
and label-level recall@50 by head / torso / tail / unseen. A **paired bootstrap
over documents** (2,000 resamples) gives the intervals of the nDCG@10 and
R-precision differences.

**Pre-registered bar.**

- A **win** requires the L-6 rerank to improve `family_new` **nDCG@10 by at
  least 0.01**, with the paired 95 % interval **excluding 0**, and to lose
  **R-precision by no more than 0.005** on `family_new`.
- **R@50 must be unchanged**; any change means the candidate set moved and the
  reading is void.
- The L-12 read uses the same bar. If only L-12 wins, the outcome is
  `secondary_win_only`.
- Otherwise it is a **null**: record it and do not proceed to menu item 8.
- **No auto-adopt.** The reviewer decides.

**Cost.** 300,000 pairs per model. Measured on this machine's GPU (RTX 4070 Ti
Super, 16 GB): ~850 pairs/s at 512 tokens → **~6 min per model**; the base,
slices, evaluation and bootstrap add ~10 min CPU. The run needs GPU device
access, which the command sandbox's private `/dev` does not provide by default.
No model download (both cross-encoders are already in the local HF cache).

**Not tested here.** Other bases (the embedding or RRF top-50), other candidate
depths, other reranker families (listwise, LLM yes/no), a fine-tuned
cross-encoder, and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-cross-encoder-rerank/metrics.json` (run
`eurlex-cross-encoder-rerank`, **1,157 s**, no LLM). Both rerankers ran on the
GPU (RTX 4070 Ti Super): L-6 **405 s**, L-12 **661 s** for 300,000 pairs each
(peak 1,184 MB; **174,779 of 300,000 pairs** — 58 % — truncated at 512 tokens;
6,000 cache misses and 0 hits per model on the first scoring pass). The base
ranking reproduces TC-9 exactly: lexical `all_dev` 0.9317 R@50 / 0.7517
nDCG@10 / 0.6152 R-precision; `family_new` 0.9138 / 0.6902 / 0.5415. R@50,
coverage and the bucket recalls are **identical** before and after both
reranks — the candidate set is unchanged, so the whole effect is the ordering
inside the top-50.

**Primary metrics by slice (nDCG@10 / R-precision / R@50):**

| method | all_dev | family_new | family_new_clean |
|---|---|---|---|
| baseline — lexical top-50 | **0.7517 / 0.6152 / 0.9317** | **0.6902 / 0.5415 / 0.9138** | **0.6771 / 0.5293 / 0.9065** |
| L-6 rerank (primary) | 0.4872 / 0.3563 / 0.9317 | 0.4360 / 0.3139 / 0.9138 | 0.4329 / 0.3111 / 0.9065 |
| L-12 rerank (secondary) | 0.3199 / 0.2234 / 0.9317 | 0.3119 / 0.2207 / 0.9138 | 0.3093 / 0.2184 / 0.9065 |

**Secondary metrics by slice (R@5 / R@10 / R@20 / R@100 / nDCG@50):**

| slice | method | R@5 | R@10 | R@20 | R@100 | nDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| all_dev | baseline | **0.6054** | **0.7684** | **0.8678** | 0.9512 | **0.8178** |
| all_dev | L-6 | 0.3691 | 0.5409 | 0.7248 | 0.9512 | 0.6383 |
| all_dev | L-12 | 0.2307 | 0.3622 | 0.5615 | 0.9512 | 0.5280 |
| family_new | baseline | **0.5366** | **0.7068** | **0.8304** | 0.9390 | **0.7739** |
| family_new | L-6 | 0.3199 | 0.4880 | 0.6855 | 0.9390 | 0.6012 |
| family_new | L-12 | 0.2230 | 0.3558 | 0.5624 | 0.9390 | 0.5193 |
| family_new_clean | baseline | **0.5272** | **0.6936** | **0.8181** | 0.9344 | **0.7628** |
| family_new_clean | L-6 | 0.3178 | 0.4842 | 0.6797 | 0.9344 | 0.5963 |
| family_new_clean | L-12 | 0.2211 | 0.3540 | 0.5590 | 0.9344 | 0.5141 |

**Paired bootstrap over documents (rerank − baseline):**

| model | slice | Δ nDCG@10 [95 % CI] | Δ R-precision [95 % CI] |
|---|---|---|---|
| L-6 | all_dev | **−0.2646 [−0.2712, −0.2577]** | −0.2588 [−0.2659, −0.2519] |
| L-6 | family_new | **−0.2543 [−0.2626, −0.2463]** | −0.2276 [−0.2363, −0.2192] |
| L-6 | family_new_clean | −0.2442 [−0.2527, −0.2355] | −0.2182 [−0.2270, −0.2089] |
| L-12 | family_new | **−0.3783 [−0.3872, −0.3692]** | −0.3207 [−0.3292, −0.3124] |
| L-12 | all_dev | −0.4319 [−0.4396, −0.4240] | −0.3918 [−0.3997, −0.3837] |

**The bar, read out.**

- **nDCG@10 gain ≥ 0.01: not met — it is a loss of 0.254.** The interval is
  entirely below zero; the rerank more than a third of the baseline's nDCG@10
  on `family_new`.
- **R-precision loss ≤ 0.005: not met.** It loses **0.2276** (L-6) and
  **0.3207** (L-12).
- **R@50 unchanged: passed** (0.9138 exactly; the candidate set is fixed).
- **Both models are null**, and the larger model is *worse* (L-12 −0.378 vs
  L-6 −0.254): the failure is not capacity.
- **`outcome`: null.**

**Reading.** This is not a neutral rerank — it is **destructive**. The lexical
base puts a gold label in the top 10 for 0.71 of documents; after the ms-marco
rerank that falls to **0.49** (L-6) and **0.36** (L-12), while the top-50 still
contains the same 91 % of gold. The model is reordering a nearly complete
candidate set into a much worse order.

The most plausible mechanism is the **role mismatch the pre-registration
flagged**: ms-marco is trained for a short query → long passage; here the
document is the query and the label title is the passage, and the scores are
near-uniform (the smoke pass returned ~0.004 for generic titles). A bigger
ms-marco does not fix that — it makes it worse. The 58 % truncation at 512
tokens also means the query side sees only the document's front, but the
lexical base reads the whole text, so truncation alone does not explain a
collapse.

The item is therefore answered for the plan's cheap option: **an off-the-shelf
web-relevance cross-encoder is not a usable reranker for EUROVOC labels in this
role assignment.** A task-fitted reranker (an NLI/entailment model scoring
"this text is about {label}", or a cross-encoder fine-tuned on EURLEX pairs)
would be a different, more expensive pre-registration — and fine-tuning is menu
item 11, which needs explicit approval.

**What the result does not say.** Nothing about `test`; nothing about a
task-fitted reranker or a listwise/LLM reranker; nothing about reranking the
embedding or RRF top-50 instead of the lexical one (unlikely to rescue an
off-distribution scorer); nothing about a shorter query view.

## 20. Learned candidate pack — TC-14 (pre-registered 2026-11-13, before the run)

**Question.** Every cheap *global* combination so far has been a null (TC-5's
one-parameter fusion, TC-10's co-occurrence boost), and the cross-encoder
rerank was destructive (TC-13). Does a **learned per-candidate model** — trained
on train candidates and applied to dev — order the shortlist better than the
strongest single retriever? The UPR precedent is that the dictionary's value
lived inside a learned pack, not a global weight.

**Mechanism (fixed before the run).**

- **Candidate union (per document):** the union of the **top-50** lists from
  (a) the embedding k-NN, (b) the lexical TF-IDF k-NN, and (c) the train
  frequency prior. The union is ≤ 150 labels per document.
- **Features (11, identical at train and dev time):**
  1. `embedding_score` — the k-NN vote sum;
  2. `lexical_score` — the TF-IDF k-NN vote sum;
  3. `rrf_score` — reciprocal-rank fusion of the two rankings (top-50,
     offset 60), the TC-9 signal;
  4. `label_rate` — the label's document frequency **divided by the index
     size** (so the out-of-fold train index and the full train index are on
     one scale);
  5. `label_cosine` — the document vector's cosine to the label title
     (cached embeddings, titles only);
  6. `doc_words` — the document's length in words;
  7–9. `bucket_head` / `bucket_torso` / `bucket_tail` — one-hot train-support
     bucket;
  10–11. `rank_embedding` / `rank_lexical` — the candidate's rank in each
     retriever, clipped at 50.
- **Deliberate omission — the dictionary arms.** The menu lists the z-scored
  dictionary among the features. A **leak-free, fold-specific** dictionary for
  the 45,000 train candidates is not affordable with the current scorer (the
  dev-only scoring already costs ~15 min per arm; a five-fold train scoring is
  hours). The frequency prior (feature 4) is the dictionary's cheap count arm;
  the z-scored arms are **not** in this pack. This is a scope decision recorded
  before the run, not a result.
- **Training data — the leak-free cross-fit.** The train split is split into
  **5 folds grouped by the title skeleton** (whole families held out). For each
  fold, the index is the **other four folds**: the embedding k-NN, the TF-IDF
  (refitted on the index), and the frequency prior are built from the index
  only, and the held-out documents' candidates/features are generated from it.
  A train document therefore never retrieves from its own family, and its
  features never see its own labels. The target is "the candidate is one of
  the document's gold labels". The model is fitted once on the pooled
  out-of-fold rows.
- **Dev data:** the same features computed against the **full train index**
  (the pipeline's normal retrieval). No dev label enters any feature or the
  training set.
- **Model:** `HistGradientBoostingClassifier` with fixed parameters
  (`max_iter=200`, `learning_rate=0.1`, `max_leaf_nodes=31`,
  `min_samples_leaf=100`, `early_stopping=False`, `random_state=42`). No
  hyper-parameter search.
- **Pack ranking:** the dev document's candidates sorted by the model score
  (ties by the candidate's RRF rank), followed by the labels outside the union
  in RRF order.
- **Comparators:** the **lexical k-NN** (the strongest single retriever,
  `family_new` 0.9138 R@50 / 0.5415 R-precision / 0.6902 nDCG@10) is the
  primary comparator; the **RRF fusion** (0.9165 / 0.5428 / 0.6997) is
  reported beside it, and the embedding k-NN is the reference baseline.

**Slices.** `all_dev` (6,000), `family_new` (4,260), `family_new_clean`
(3,764). Test is never read.

**Metrics.** Primary: R@50, R-precision, nDCG@10. Secondary: R@5, R@10, R@20,
R@100, nDCG@50. Diagnostics: coverage@50 and label-level recall@50 by
head / torso / tail / unseen. A paired bootstrap over documents (2,000
resamples) gives the pack-minus-comparator intervals.

**Pre-registered bar.**

- **Shortlist win:** the pack must improve `family_new` **R@50 by at least
  0.01** over the **lexical** comparator, with the paired 95 % interval
  excluding 0, and lose **R-precision and nDCG@10 by no more than 0.01**.
- **Ordering win (secondary):** if R@50 does not move but nDCG@10 improves by
  at least **0.01** with the interval excluding 0 and R@50 loses no more than
  0.01, the outcome is `ordering_win_only` — an ordering gain, not a shortlist
  win.
- Otherwise it is a **null**: record it and do not proceed to menu item 9.
- **No auto-adopt.** The reviewer decides.

**Cost.** Five out-of-fold retrievals (TF-IDF refits ~40 s each; sparse cosine
~1 min each; embedding k-NN ~30 s each), the feature rows (~5 M candidate rows),
the HistGB fit, and the dev scoring. Roughly 20–30 min on CPU; no GPU, no LLM,
no model download.

**Not tested here.** The z-scored dictionary arms, co-occurrence features, a
logistic variant (the plan allows logistic or HistGB; HistGB is fixed here),
hyper-parameter search, a candidate set drawn from the dev folds, and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-candidate-pack/metrics.json` (run
`eurlex-candidate-pack`, **543.5 s**, no LLM; document cache 51,000 hits, label
cache 4,197). Training: five out-of-fold folds, **4,746,172 candidate rows**
(4.51 % positive, ~105.4 candidates/document), 84 s per fold. Dev: **632,709
candidate rows** (105.45 per document). The comparator rankings reproduce their
sections exactly (lexical `family_new` 0.9138 / 0.5415 / 0.6902; RRF 0.9165 /
0.5428 / 0.6997).

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| method | all_dev | family_new | family_new_clean |
|---|---|---|---|
| embedding k-NN (reference) | 0.9168 / 0.5715 / 0.7070 | 0.8944 / 0.4869 / 0.6364 | 0.8862 / 0.4744 / 0.6241 |
| lexical k-NN (primary comparator) | 0.9317 / 0.6152 / 0.7517 | 0.9138 / 0.5415 / 0.6902 | 0.9065 / 0.5293 / 0.6771 |
| RRF fusion (secondary comparator) | 0.9337 / 0.6164 / 0.7581 | 0.9165 / 0.5428 / 0.6997 | 0.9094 / 0.5317 / 0.6882 |
| **candidate pack (HistGB)** | **0.9422 / 0.6488 / 0.7861** | **0.9278 / 0.5818 / 0.7343** | **0.9218 / 0.5710 / 0.7229** |

**Secondary metrics by slice (R@5 / R@10 / R@20 / R@100 / nDCG@50):**

| slice | method | R@5 | R@10 | R@20 | R@100 | nDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| all_dev | lexical | 0.6054 | 0.7684 | 0.8678 | **0.9512** | 0.8178 |
| all_dev | RRF | 0.6112 | 0.7764 | 0.8740 | 0.9481 | 0.8220 |
| all_dev | **pack** | **0.6398** | **0.8049** | **0.8950** | 0.9494 | **0.8427** |
| family_new | lexical | 0.5366 | 0.7068 | 0.8304 | **0.9390** | 0.7739 |
| family_new | RRF | 0.5435 | 0.7187 | 0.8397 | 0.9355 | 0.7800 |
| family_new | **pack** | **0.5786** | **0.7546** | **0.8670** | 0.9370 | **0.8058** |
| family_new_clean | lexical | 0.5272 | 0.6936 | 0.8181 | **0.9344** | 0.7628 |
| family_new_clean | RRF | 0.5360 | 0.7075 | 0.8298 | 0.9299 | 0.7697 |
| family_new_clean | **pack** | **0.5708** | **0.7429** | **0.8574** | 0.9314 | **0.7964** |

**Diagnostics (coverage@50; label-level recall@50 by head / torso / tail /
unseen):**

| slice | method | coverage@50 | head | torso | tail | unseen |
|---|---|---:|---:|---:|---:|---:|
| all_dev | lexical | 0.9985 | 0.9682 | 0.8475 | 0.5414 | 0.0 |
| all_dev | RRF | 0.9987 | 0.9720 | 0.8454 | 0.4862 | 0.0 |
| all_dev | **pack** | **0.9988** | **0.9759** | **0.8637** | **0.5690** | 0.0 |
| family_new | lexical | 0.9981 | 0.9578 | 0.8426 | 0.5404 | 0.0 |
| family_new | RRF | 0.9984 | 0.9633 | 0.8403 | 0.4830 | 0.0 |
| family_new | **pack** | **0.9986** | **0.9689** | **0.8605** | **0.5681** | 0.0 |
| family_new_clean | lexical | 0.9979 | 0.9541 | 0.8312 | 0.5208 | 0.0 |
| family_new_clean | RRF | 0.9981 | 0.9598 | 0.8284 | 0.4642 | 0.0 |
| family_new_clean | **pack** | **0.9984** | **0.9661** | **0.8504** | **0.5508** | 0.0 |

The unseen bucket stays 0.0 for every arm — the pack's candidates are all
seen labels (the union's retrievers cannot score a label with no train
example); the zero-shot quota (TC-11) remains the separate route.

**Paired bootstrap over documents (2,000 resamples):**

| comparator | slice | Δ R@50 [95 % CI] | Δ R-precision | Δ nDCG@10 |
|---|---|---|---|---|
| lexical | all_dev | **+0.0104 [+0.0089, +0.0121]** | +0.0336 | +0.0343 |
| lexical | family_new | **+0.0141 [+0.0119, +0.0163]** | **+0.0404** | **+0.0440** |
| lexical | family_new_clean | **+0.0153 [+0.0129, +0.0177]** | +0.0417 | +0.0458 |
| RRF | all_dev | +0.0085 [+0.0071, +0.0100] | +0.0324 | +0.0279 |
| RRF | family_new | **+0.0113 [+0.0094, +0.0132]** | +0.0390 | +0.0346 |

**The bar, read out.**

- **R@50 gain ≥ 0.01 over the lexical k-NN: passed.** **+0.0141** on
  `family_new` (0.9278 vs 0.9138), interval entirely above zero; +0.0104 on
  `all_dev` and +0.0153 on the clean slice.
- **Interval excluding 0: passed.**
- **R-precision and nDCG@10 loss ≤ 0.01: passed on the upside** — both
  **improve** (+0.0404 and +0.0440 on `family_new`).
- **Ordering bar: passed too** (nDCG@10 +0.0440, interval excluding 0).
- **`outcome`: win** — the first arm to beat the strongest single retriever on
  **both breadth and ordering**, and the tail rises as well (0.5404 → 0.5681
  on `family_new`).
- Against the RRF fusion the pack also wins (+0.0113 R@50, +0.0390
  R-precision, +0.0346 nDCG@10 on `family_new`).

**Reading.** The learned pack works where the global combinations failed
(TC-5, TC-10): a per-candidate model can use the retrievers' **disagreement**
— a label that the lexical ranking ranks low but the embedding ranks high, or
that the frequency prior knows is common, is worth promoting — and the label
metadata (name cosine, bucket) tells it when. The gain is large and consistent
across every slice and every metric: +0.014 R@50, +0.040 R-precision, +0.044
nDCG@10 on `family_new`. The tail improves too (the pack's top-50 now reaches
0.568 of tail assignments vs 0.540 for lexical), because the candidate union
and the features give rare labels a second chance.

Two honest qualifications. First, the pack is trained on **out-of-fold train
candidates** (the 4/5 index) and evaluated on dev candidates from the **full
train index**: the features are on a slightly different index scale, and the
`label_rate` feature is normalised by the index size exactly to keep that
comparable. Second, the menu's **z-scored dictionary arms are not in the pack**
(the pre-registered scope decision): the pack's gain comes from the two
retrievers, the frequency prior and the label metadata — not from the
dictionary. A dictionary-in-pack variant would need the fold-specific scoring
this change set could not afford.

**What the result does not say.** Nothing about `test`; nothing about the
dictionary arms as features, co-occurrence features, a logistic variant or any
hyper-parameter search; nothing about refitting the pack for a production
snapshot; and nothing about the zero-shot labels, which still need the quota.
The pack is a **dev-tuned model**, so the final test shot (menu item 13) is
still a separate decision.

## 21. Cluster routing / label pruning — TC-15 (pre-registered 2026-11-13, before the run)

**Question.** The full ranking scores all **4,197** labels for every document.
If a document is first routed to its nearest clusters of **train documents**,
and only the labels those clusters actually carry are ranked, does the
shortlist stay as good (or improve, by removing globally popular but
irrelevant head labels) while the label space shrinks? The item is explicitly
about **R@50 and cost against the full ranking**.

**Mechanism (fixed before the run).**

- **Base ranking:** the lexical TF-IDF k-NN (TC-9), the strongest single
  retriever — `all_dev` 0.9317 R@50 / 0.6152 R-precision / 0.7517 nDCG@10;
  `family_new` 0.9138 / 0.5415 / 0.6902. The learned pack (TC-14) is the
  current best overall but is **not** the base here: routing is measured
  against the full ranking of the same retriever, and combining routing with
  the pack would be a separate change set.
- **Clustering:** MiniLM whole-document vectors (cached), **MiniBatchKMeans**
  with `K = 500` clusters on the 45,000 train vectors (`batch_size=4096`,
  `n_init=3`, `max_iter=100`, `random_state=42`).
- **Cluster label sets:** a cluster's labels are the labels carried by **at
  least one** train document assigned to it (no support threshold).
- **Routing:** a dev document is routed to the **C** clusters whose centroids
  have the highest cosine to the document vector. **C = 3 is the primary**;
  **C = 1 is a secondary**, more aggressive read. No C is tuned.
- **Pruned ranking:** the union of the routed clusters' labels is the
  **routed label set**; the ranking places those labels first, ordered by the
  lexical score, then the remaining labels ordered by the lexical score. (The
  ranking stays a full permutation, so every metric is comparable; the
  pruning is a mask on what can enter the shortlist.)
- **No new model, no LLM, no GPU**; the embeddings are cached.

**Slices.** `all_dev` (6,000), `family_new` (4,260), `family_new_clean`
(3,764). Test is never read.

**Metrics.** Primary: R@50, R-precision, nDCG@10. Secondary: R@5, R@10, R@20,
R@100, nDCG@50. Diagnostics: coverage@50, label-level recall@50 by head /
torso / tail / unseen, and — the routing-specific ones — the **routed label
space** (mean and median labels per document, and its share of 4,197) and
**routing recall** (the share of the document's gold labels that are in the
routed set). A paired bootstrap over documents (2,000 resamples) gives the
routed-minus-full intervals.

**Pre-registered reading.**

- **Shortlist win:** the C = 3 routed ranking must improve `family_new`
  **R@50 by at least 0.01** with the paired 95 % interval excluding 0, lose
  R-precision and nDCG@10 by no more than 0.01, **and** the mean routed label
  space must be **at most 50 %** of the 4,197 labels (otherwise it is not
  pruning).
- **`efficiency_only`:** no shortlist win, but the `family_new` R@50 loss is
  **≤ 0.005** with a mean routed label space **≤ 25 %** — a practical pruning
  result, not a metric win.
- The C = 1 read uses the same win conditions; if only C = 1 wins, the outcome
  is `secondary_win_only`.
- Otherwise it is a **null**: record it and do not proceed to menu item 10.
- **No auto-adopt.** The reviewer decides.

**Cost.** MiniBatchKMeans on 45,000 × 384 (~1–3 min), the cluster label sets
(seconds), the routing and the two pruned rankings (seconds), the cached
lexical base and the metrics (~5 min). Minutes total; no GPU, no LLM, no model
download.

**Not tested here.** Other K, other C, other clusterings (TF-IDF, HDBSCAN,
label-space clustering), routing combined with the learned pack, a support
threshold on the cluster label sets, and `test`.

### Result (2026-11-13)

Artifact: `results/eurlex-cluster-routing/metrics.json` (run
`eurlex-cluster-routing`, **99.0 s**, no LLM; document cache 51,000 hits).
MiniBatchKMeans K=500 on the 45,000 train vectors: **5.4 s** (inertia 8,017.5),
cluster sizes 3–427, cluster label sets **3–619 labels** (mean 77.8). Routing
6,000 dev documents: **0.1 s**. The full lexical base reproduces TC-9 exactly
(`family_new` 0.9138 / 0.5415 / 0.6902).

**Routed label space and routing recall:**

| route | mean labels | median | share of 4,197 | routing recall (all_dev, micro) | family_new | family_new_clean |
|---|---:|---:|---:|---:|---:|---:|
| C = 1 | 99.8 | 74 | **2.38 %** | 0.9141 | 0.8905 | 0.8798 |
| C = 3 | 250.1 | 206 | **5.96 %** | **0.9636** | 0.9534 | 0.9488 |

**Primary metrics by slice (R@50 / R-precision / nDCG@10):**

| method | all_dev | family_new | family_new_clean |
|---|---|---|---|
| full lexical ranking | **0.9317 / 0.6152 / 0.7517** | **0.9138 / 0.5415 / 0.6902** | **0.9065 / 0.5293 / 0.6771** |
| routed, C = 3 (primary) | 0.9315 / 0.6209 / 0.7572 | 0.9132 / 0.5488 / 0.6971 | 0.9058 / 0.5373 / 0.6843 |
| routed, C = 1 (secondary) | 0.9100 / 0.6240 / 0.7597 | 0.8843 / 0.5514 / 0.6995 | 0.8738 / 0.5393 / 0.6859 |

**Secondary metrics, C = 3 vs full (R@5 / R@10 / R@20 / R@100 / nDCG@50):**

| slice | method | R@5 | R@10 | R@20 | R@100 | nDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| all_dev | full | 0.6054 | 0.7684 | 0.8678 | **0.9512** | 0.8178 |
| all_dev | routed C=3 | **0.6116** | **0.7748** | **0.8711** | 0.9487 | **0.8206** |
| family_new | full | 0.5366 | 0.7068 | 0.8304 | **0.9390** | 0.7739 |
| family_new | routed C=3 | **0.5441** | **0.7150** | **0.8342** | 0.9349 | **0.7773** |

**Diagnostics (C = 3 vs full; coverage@50 and label-level recall@50 by bucket,
`family_new`):**

| method | coverage@50 | head | torso | tail | unseen |
|---|---|---:|---:|---:|---:|
| full | 0.9981 | 0.9578 | 0.8426 | 0.5404 | 0.0 |
| routed C=3 | 0.9981 | 0.9573 | 0.8392 | **0.5447** | 0.0 |

**Paired bootstrap over documents (routed − full):**

| route | slice | Δ R@50 [95 % CI] | Δ R-precision | Δ nDCG@10 |
|---|---|---|---|---|
| C = 3 | all_dev | −0.0002 [−0.0017, +0.0013] | +0.0057 | +0.0055 |
| C = 3 | family_new | **−0.0005 [−0.0025, +0.0014]** | +0.0073 | +0.0069 |
| C = 3 | family_new_clean | −0.0007 [−0.0028, +0.0014] | +0.0080 | +0.0072 |
| C = 1 | family_new | **−0.0294 [−0.0332, −0.0255]** | +0.0099 | +0.0092 |

**The pre-registered reading, read out.**

- **Shortlist win: not met.** C = 3's R@50 change is **−0.0005**, far below
  the +0.01 bar, and the interval straddles zero. The ordering does not reach
  the +0.01 ordering bar either (+0.0069 nDCG@10).
- **`efficiency_only`: met.** The `family_new` R@50 loss is **0.0005 ≤ 0.005**
  and the routed space is **5.96 % ≤ 25 %**. The primary outcome is
  **`efficiency_only`**.
- **C = 1 is a null.** Routing to a single cluster keeps only 2.4 % of the
  label space, covers 91.4 % of gold, and loses **0.0294 R@50** — too
  aggressive, even though its R-precision and nDCG@10 rise (+0.0099 / +0.0092).
- **The routing recall is the ceiling.** C = 3 ranks 96.4 % of gold
  assignments (95.3 % on `family_new`); the routed R@50 (0.9132) sits just
  below the full ranking, so the coverage ceiling is not the binding factor at
  depth 50.

**Reading.** Cluster routing is a **cost lever, not a metric lever**. At C = 3
it ranks about **250 of 4,197 labels per document** (a 17× reduction) with the
shortlist unchanged to the fourth decimal and a small ordering gain
(+0.007 R-precision/nDCG@10), and the tail slightly improves (0.5404 → 0.5447)
because pruning removes globally popular head labels rather than rare ones.
C = 1 shows where the lever breaks: 2.4 % of the space and 91.4 % coverage cost
0.029 R@50.

The practical value is label-side: a product ranking only the routed labels
sorts ~6 % of the columns; the k-NN's own score computation is unchanged, so
the end-to-end saving is on the label side, not on retrieval. The result is not
a shortlist win and is not adopted; combining routing with the learned pack
(the current best, which ranks the union of retrievers) would be a separate
change set.

**What the result does not say.** Nothing about `test`; nothing about other K,
other C, TF-IDF/HDBSCAN clustering, a support threshold on the cluster label
sets, or routing inside the pack.
