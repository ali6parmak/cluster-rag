# AGENTS.md

Instructions for AI coding agents (and humans) working in this repo. **Read the "START HERE" section below first** — it indexes the measured numbers and the file that answers each question, so a single question never costs a whole-file read. Then read `PLAN.md`, which defines the phases and the current scope.

## START HERE

**The one-paragraph version.** The pipeline embeds a text, retrieves similar labeled examples and candidate labels, fuses their scores and suggests labels; feedback updates the system online and an active learner picks what a human should review next. Two tracks run side by side: the **main pipeline** (`PLAN.md`) and the **text-processing / segmentation track** (`TEXT_PROCESSING_PLAN.md`). The track's evidence trail is `docs/experiments_text_processing.md` (42 sections, append-only) and its curated tables are `docs/experiments_table_text_processing_omct.md` and `docs/experiments_table_text_processing_upr.md`. The track's plan is split in two: the **live** plan holds the open work, and `docs/text_processing_plan_archive.md` holds the executed pre-registrations and the questions they answered (moved verbatim, append-only).

**The two datasets — never mix them.** A `upr-` run is the **UPR** human-rights recommendation data (train / val / test; the text-processing track has only ever touched **val** — `test` is reserved). An `omct-` run is the **auxiliary OMCT pairwise data** (1,520 reviewed pairs plus a seeded 4,000-cell unjudged control): it is method-choice evidence and **never** a UPR metric.

### The headline numbers

Every row names the dataset, the split, the value, the artifact it comes from and the log section that explains it. All of them are reproduced by the two final sweeps (`experiments_text_processing/omct_final_sweep.py`, `experiments_text_processing/upr_final_sweep.py`).

| # | what | dataset / split | number | artifact | log § |
|---|---|---|---|---|---|
| U1 | The **standing champion** — the pipeline's own whole-text scoring — micro-F1 | UPR **val** | **0.5479** | `results/upr-analyze-final-champion/analysis.json` | §CS-8a |
| U2 | The **best UPR number** — the learned feature pack + the decision model's probability: micro-F1 / unseen bucket / shortlist R@50 | UPR **val** | **0.5965** / **0.5697** / **0.9856** | `results/upr-analyze-final-decision-feature/analysis.json` | §CS-13 |
| U3 | The **decision model alone** (`tev1`, the paragraph shape, the `noul` question): micro-F1 / shortlist R@50 / recall@10 | UPR **val** | 0.5502 / 0.9987 / 0.8845 | `results/upr-decide-tev1-paragraph-noul-grid/decisions.json` | §CS-9b |
| U4 | The **union candidate generator** — a *measurement, not a component*: recall@10 / its unseen bucket's micro-F1 | UPR **val** | 0.9120 / 0.2540 (the decision model's own 0.5267) | `results/upr-decide-tev1-paragraph-noul-grid-union/decisions.json` | §CS-14 |
| U5 | The **learned pack** alone, without the decision feature: shortlist R@50 / micro-F1 | UPR **val** | 0.8909 / 0.4818 | `results/upr-analyze-final-learned/analysis.json` | §CS-8d, §CS-8e |
| U6 | The **label catalog's channels** (the description + the parent) against the same pack without them: Δ micro-F1 | UPR **val** | **+0.0258 [+0.0091, +0.0433]** | `results/upr-analyze-final-catalog/analysis.json` | §CS-11 |
| U7 | **Chunking the long texts** (the 300+ character bucket): Δ micro-F1 — the chunkers that do not split UPR's texts give exactly 0 in every bucket | UPR **val** | **+0.109 … +0.127** | `results/upr-analyze-ladder/analysis.json` | §CS-8b |
| U8 | The **pure zero-shot geometry** as a *candidate generator*: shortlist R@50 (the champion's 0.7423) | UPR **val** | **0.8691** | `results/upr-analyze-final-ladder/analysis.json` | §CS-8b |
| U9 | The **lexical floor** — TF-IDF (word 1–2 grams) + one-vs-rest logistic regression, no embedding at all: micro-F1 / the tail bucket / shortlist R@50 | UPR **val** | **0.3845** / 0.0301 / 0.7205 | `results/upr-eval-lexical-tfidf-val/metrics.json` | §CS-29 |
| U10 | The **corpus map as a classifier** — the train split's own word↔label weights, no embedding and no model call: micro-F1 / the seen buckets / the tail bucket | UPR **val** | **0.5243** / 0.5915 / 0.5435 | `results/upr-eval-corpus-map-sum-zscore-positive-val/metrics.json` | §CS-30, §CS-31 |
| U11 | The **corpus map as a pack feature**: the pack + the map's column against the same pack without it (paired Δ micro-F1) | UPR **val** | **+0.0684 [+0.0441, +0.0915]** (the pack reaches 0.5502) | `results/upr-analyze-cmap/analysis.json` | §CS-32 |
| U12 | The **class-restricted corpus maps** against the full map: the generic class dropped / the distinctive-only map (micro-F1) | UPR **val** | **0.3815 / 0.0798** against the full map's **0.5243** | `results/upr-eval-corpus-map-sum-zscore-positive-{nogeneric,distinctive}-val/metrics.json` | §CS-33 |
| U13 | The **corpus-mined lexicon** as a label-side channel: the mined words' coverage as a pack feature (the AUC over the val grid's cells) / its seen-bucket Δ against the plain pack / the head-to-head against the name+description channel | UPR **val** | **0.8109** / **+0.0507 [+0.0312, +0.0708]** / **+0.0272 [+0.0044, +0.0502]** (only one of three seen buckets clears — the bar is not met) | `results/upr-analyze-learned-lexicon-minedfeat/analysis.json` | §CS-35 |
| U14 | The **corpus map's stability** — the share of its 18,707 links that survive the resampling / the **filtered** map as a classifier (the fit side's out-of-fold) | UPR **val** | **0.2763** (at τ = 0.75; **0.1256** at τ = 0.90) / **0.6544** against the full map's **0.6613** — the filter is a null | `results/upr-hitmap-corpus/hitmap.json` | §CS-36 |
| U15 | The **map's abstention** ("unknown" instead of "impossible"): the eight held-out labels' unseen micro-F1 with the abstention / the paired unseen-view difference | UPR **val** | **0.0000** / **+0.0000** — the abstention is a null: the "impossible" verdict was never in the encoding | `results/upr-analyze-cmap-abstain/analysis.json` | §CS-37 |
| U16 | The **label-pair relation**: the sibling pairs' mean vocabulary Jaccard above the other pairs' / the **vocabulary-less** class's size | UPR **val** (the train split's map) | **+0.0413 [+0.0372, +0.0451]** / **13** labels — the corpus's relation **is** the taxonomy's | `results/upr-hitmap-corpus/hitmap.json` | §CS-38 |
| U17 | The **phrase table** (the map over the corpus's bigrams): its generic share against the word table's / the union's fit side's out-of-fold against the word map's | UPR **val** (the train split's map) | **15.5 %** against **43.6 %** / **0.6663** against **0.6613** — the unit change is a null (the phrases are too rare to be a table) | `results/upr-eval-corpus-map-sum-zscore-positive-phrase-val/metrics.json` | §CS-39 |
| U18 | The **lexical kNN** (the words alone as a retriever): its shortlist R@50 / micro-F1 — against the embedding kNN's and the champion's | UPR **val** | **0.9437** / **0.4605** (the embedding kNN's 0.9576 / 0.5388; the champion's 0.7423 / 0.5479) — the words alone retrieve nearly as well as the embedding, so **PLAN.md's BM25 checkbox is live**; and the champion's fusion is the *worse* candidate generator (0.7423, barely above the 0.7353 chance level) | `results/upr-eval-lexical-knn-k10-val/metrics.json` | §CS-41 |
| U19 | The **fit-side mask** (the dictionary's tables emptied for 8 seed-chosen seen labels during the fit pass only): the unseen bucket with the mask / the seen-view Δ against the pack without the column / the pooled micro-F1 | UPR **val** | **0.2947** / **+0.1478 [+0.1236, +0.1710]** / **0.5874** — the eight recover only part-way (the registered 0.45 bar is **not** met), but the mask costs the seen buckets nothing and the pooled number is the track's **second-best pack number** (CS-13's 0.5965 is first) | `results/upr-analyze-cmap-fitmask/analysis.json` | §CS-42 |
| M1 | The **decision model** on the same-document grid: **within-document** AUC (the pooled AUC 0.7688 goes beside it, never alone) | OMCT | **0.7701** | `results/omct-analyze-final-*/analysis.json` | §Stage 3.5.2 |
| M2 | The plain full-text cosine, for comparison | OMCT | 0.5020–0.5632 | same | §Stage 3.5.2 |
| M3 | The **learned feature pack**: within-document AUC | OMCT | 0.6472–0.6723 | same | §Stage 3.1 |
| M4 | The **safe-negative rule** (the fit-side threshold): reach over the unjudged grid / harm on the verified positives | OMCT | **0.8313 / 0.1851** | same | §Stage 3.5.2 |
| M5 | …the **label-blind counterpart's** harm at the same reach — so the harm is label-driven | OMCT | 0.4147 | same | §Stage 3.5.2 |
| M6 | The **shortlist rerank**: the positives' hit@1 (the cosine's own hit@1) | OMCT | **0.1565** (arctic2, K=10) / 0.1050 | same | §Stage 3.5.3 |
| M7 | The **late chunking**: Δ within-document AUC — voyage-4-nano only (it needs token-level embeddings) | OMCT | **+0.0859** | `results/omct-analyze-final-voyage-4-nano/analysis.json` | §Stage 2.7 |

`§CS-n` = the section of `docs/experiments_text_processing.md`; `§Stage n` = the same log's OMCT sections.

### Question → file → section

| question | where to look |
|---|---|
| What is this project, and what state is it in? | `PLAN.md` §1 and §4 · `AGENTS.md` "Project summary" |
| What is the best number on UPR, and what produces it? | the table above → `docs/experiments_text_processing.md` §CS-13 |
| What exactly is the champion config (the pipeline's own scoring)? | `results/upr-analyze-final-champion/config.yaml` → §CS-8a |
| Where do run artifacts live, and what is inside one? | `AGENTS.md` "Runs and where things live" |
| What does a `run_id` mean, and how do I find a run's log entry? | `AGENTS.md` "Runs and where things live" |
| How do I run an experiment? | `AGENTS.md` "Running things" → `experiments_text_processing/{omct,upr}_final_sweep.py` |
| Where are the curated result tables? | `docs/experiments_table_text_processing_{omct,upr}.md` → `## OMCT — final_results`, `## UPR — final_results` |
| Where is the full research trail (all 42 log sections)? | `docs/experiments_text_processing.md` → its table of contents, at the top |
| Where is the track's pre-registration record (the executed plan)? | `docs/text_processing_plan_archive.md` — the moved, verbatim plan: §0.1, §3's stages, §5's answered questions, §7's change sets, and PLAN.md's Phase-4 sweep record |
| What did chunking / segmentation show? (OMCT) | §Stage 2, §Stage 2 — the verdict, §Stage 2.6, §Stage 2.7 |
| What did chunking / segmentation show? (UPR) | §CS-8b, §CS-8c |
| Is the embedding model the lever, or are the features? | §Stage 3.1b (OMCT) · §CS-8d, §CS-8e (UPR) |
| What is the learned feature pack, and what are its three floors? | §CS-8d · `src/cluster_rag/text_processors/combining/` |
| What are the decision models, and what do they buy? | §Stage 3.5 (the probe) · §CS-9 (UPR) · §CS-9b (the full-val grid) |
| What is the safe-negative rule — the track's deliverable? | §Stage 3.5.2 · §CS-9b · §CS-14 · `TEXT_PROCESSING_PLAN.md` §8 |
| What is the label catalog? | §CS-11 · `catalogs/upr_labels.jsonl` (+ `catalogs/upr_labels.meta.json`) |
| What does a **plain lexical model** reach on this corpus (the round's floor)? | the track log's §CS-29 (TF-IDF + one-vs-rest, micro-F1 0.3845 — the tail collapses, the shortlist is on par with the champion's) |
| What does a **pure corpus statistic** reach (no embedding, no model call)? | the track log's §CS-30 and §CS-31 (the corpus map as a classifier: micro-F1 **0.5243** against the champion's 0.5479 — 90% of it; the map's log-odds weighting buys +0.2226 over a plain word count, and the z-scored weights add a further +0.0319) |
| What does the corpus map add **inside** the learned pack? | the track log's §CS-32 (a paired **+0.0684 [+0.0441, +0.0915]** of micro-F1 over the same pack without it — but it is a seen-labels feature: it costs the unseen bucket 0.4867) |
| What kind of **words** affect the label, in which way? | the track log's §CS-33 (the corpus map's own word classes: the **distinctive** words are 100 % positive with a mean log-odds of **+4.27** against the **generic** class's 81 % and **+0.84** — but the class-restricted map is a **much worse** classifier: 0.3815 and 0.0798 against the full map's 0.5243: the strongest words are rare, the register is everywhere) |
| Which words does the **model** itself lean on? | the track log's §CS-33 (the surrogate's own readout, the same split: the **name words** are 82.9 % positive with a mean coefficient of **+0.208**, an order of magnitude above every class mean — the model's habit is the label's own name, the corpus's habit is the register) |
| Does the **grammatical** class of a word matter (real POS tags)? | the track log's §CS-34 (**no** — a null: the nouns and the verbs are nearly identical to the map (+2.0044 against +1.9856) and the noun-only map is materially worse than the full map: 0.4840 against 0.5243, a paired **+0.0403 [+0.0258, +0.0545]**) |
| Does the **corpus-mined lexicon** work as a label-side channel (the PIFA idea)? | the track log's §CS-35 (**not** by the pre-registered bar: the mined words are the **best cell-level signal in the pack** — the AUC **0.8109** against the description's 0.7067 — and the channel buys **+0.0507 [+0.0312, +0.0708]** of seen-bucket micro-F1 over the plain pack, but the head-to-head against the name+description channel does not separate them (only one of three seen buckets clears), so the lexicon does not become a catalog channel; the expansion route is a null for the eight) |
| How much of the **corpus map** is sampling noise? | the track log's §CS-36 (the **stability map**: 500 bootstrap resamples of the train split; the survival curve's **cliff at 63 %** — 1 − e⁻¹, where a single-text link belongs — and **not one** of the map's 12,483 single-text links survives τ = 0.75; but the **filtered** map is a **worse** classifier (the fit side's out-of-fold 0.6544 against the full map's 0.6613), so the filter is a null and the **readout** is the deliverable; the map's head words keep their place in only 50 % of the resamples) |
| Can the map say **"I don't know"** instead of "impossible"? | the track log's §CS-37 (**no** — a null, and the reading is the finding: the abstention ("no evidence" as NaN instead of the floor) leaves the eight held-out labels at 0.0000, a paired **+0.0000**; a map with no word↔label information at all collapses them the same way, so it is the **channel's silence on the fit side** — the eight are the labels the map is silent about *and* the labels that are never positive — that costs them, not the encoding) |
| What is the **label space's own geometry** — the lexical twins, the neighbours? | the track log's §CS-38 (**the corpus rediscovers the taxonomy**: the sibling pairs — the labels that share a catalog parent — are closer in vocabulary, **+0.0413 [+0.0372, +0.0451]**, and **11 of the corpus's 20 closest pairs** are siblings; there are **no lexical twins** at all, and the **"no vocabulary of their own"** class holds **13** labels, the eight held-out ones plus five whose own words lie almost entirely inside one neighbour's table) |
| Do **phrases** beat their own words? | the track log's §CS-39 (**no** — a null: the phrase table's generic share is **15.5 %** against the word table's **43.6 %** (the phrases are too rare to sit in many tables), the union's val gain is not established, and the **blind** union gains *more* on the fit side (+0.0240 against +0.0050) — the phrases add noise; the register is real but it is a **fragment**: the top interactions are "on its", "with an", "and will") |
| What did the zero-shot rerankers do? | §CS-15 (as scorers) · §CS-16 (as a pack feature) |
| What did the hit map show (which words carry the signal)? | the track log's §CS-18 (the topic-side map + the causal ablation) · §CS-19 (the text-side map + the surrogate attribution) · the archive's §7.1.4 / §7.1.5 |
| What does the corpus itself associate with each label (the plain statistics)? | the track log's §CS-23 (the corpus map: every word for every label of the **train** split, with the counts, the lift and the log-odds) · the archive's §7.1 (the corpus map) · `results/upr-hitmap-corpus/hitmap.json` |
| How do I see the hit map for **one text**? | `tools/hitmap_view.py` (the terminal viewer: the words and what each contributes — the lexical map's own decomposition *and* the surrogate's fitted coefficient — who the words say it is, what the model says, what is true, what if a word is removed, and the corpus's own words for the topic; every block names the question it answers, and `--colour` forces the colours on when piping) → the track log's §CS-20 and §CS-25 |
| What does the map say about **two** labels — why A and not B? | `tools/hitmap_view.py` (the **contrastive** block: the words that separate a pair of labels, the corpus map's two tables read against each other — CS-40) → the track log's §CS-40 |
| How much of the **retrieval** is words alone? | the track log's §CS-41 (the **lexical kNN**: the TF-IDF neighbours over the train split, voted by the pipeline's own kNN machinery — the shortlist R@50 **0.9437** against the embedding kNN's 0.9576, and the champion's fusion's 0.7423) → `results/upr-eval-lexical-knn-k10-val/metrics.json` |
| Can the dictionary stop saying **"impossible"** when it is silent? | the track log's §CS-42 (**not fully** — the fit-side mask (8 seed-chosen *seen* labels' tables emptied during the fit pass only) lifts the eight held-out labels **0.0000 → 0.2947** and five of them come back, but the registered 0.45 bar is not met; the mask costs the seen buckets nothing (they **gain** +0.1478 [+0.1236, +0.1710] against the control) and the pooled number **0.5874** is the track's second-best pack number; the three that stay down are the two catch-alls and the broad ESC category) |
| What is still open, and what is the next step? | `TEXT_PROCESSING_PLAN.md` §5 (the unanswered questions) and §7 (CS-17, the hit map's next round CS-21/CS-22, the viewer's round two CS-23 … CS-26, the lexical methods round CS-29 … CS-35, and the map's quality round CS-36 … CS-41 — **both rounds are executed**; the live thread is now the **main plan's deferred BM25 checkbox**, which CS-41's lexical kNN made live, and the **fit-side round CS-42 … CS-46** — **CS-42 is executed, CS-43 is the open one**) · `PLAN.md` §7 |
| What are the main pipeline's phases, and which are done? | `PLAN.md` §4 |
| What are the coding rules and the pure-core boundary? | `AGENTS.md` "Coding conventions" · `TEXT_PROCESSING_PLAN.md` §2 and §4 |
| How is leakage prevented? | `AGENTS.md` "Data and evaluation discipline" · `TEXT_PROCESSING_PLAN.md` §4 |
| How do I add a command or a sweep? | `AGENTS.md` "Running things" · `src/cluster_rag/commands/` · `src/cluster_rag/sweep.py` · `src/cluster_rag/cli.py` |
| How do I check that a run reproduced? | `tools/verify_analysis_baseline.py` · `tools/verify_upr_baseline.py` |

### The house rules in one screen

The full text lives in the sections below; these are the ones that get broken by accident.

- **Every number comes from an artifact in `results/` first**, then the log. No hand-written tables.
- **The logs are append-only** and the comparison tables are auto-appended by the CLI commands — **never hand-edit a table row**.
- **The two negative subsets are never pooled** (A = the reviewer-verified hard negatives, B = the unjudged control), and **OMCT is never a UPR metric**.
- **UPR `test` is reserved.** The text-processing track has only ever touched `val`.
- **Raw data is read-only**; an LLM may score, never relabel.
- **Small, focused changes**, one change set per commit; explain before a large one (new dependency, schema change, or a refactor over three modules) and wait.
- **One change set at a time, with a stop between them.** When a change set is done (its numbers in an artifact, its log section written, its checkbox ticked, the commit made, the gate green), **stop and report** — what changed, the numbers, what the reviewer has to decide, what the next change set would be. The next change set starts only on the reviewer's word; do not queue them.

## Project summary
Retrieval-augmented multi-label topic classification. Texts are embedded (Hugging Face or Ollama models), similar labeled examples and candidate labels are retrieved (optionally routed through clusters), scores are fused or reranked, and labels are suggested. User feedback updates the system online, and active learning chooses which items to review.

## Working agreement
1. **Follow PLAN.md phase order.** Do not start a phase before the exit criteria of its dependencies are met. The Phase 3 evaluation harness comes before any modeling work.
2. **Small, focused changes.** One task (one PLAN checkbox) per change set. When a task is done, tick its checkbox in `PLAN.md`.
3. **Explain before large changes.** For new dependencies, schema changes, or refactors touching more than 3 modules, describe the plan and wait for approval.
4. **Never claim results you did not measure.** Every metric you report must come from a run in `results/` and appear in `docs/experiments.md`.
5. **If something is ambiguous, ask.** Do not guess label semantics, data formats or thresholds. Add open questions to `PLAN.md` section 7.
6. **Explain plainly and state the scope.** When discussing concepts, results or reviews: one concept per paragraph, short sentences, define jargon on first use, and pair formal definitions with a small concrete example. Always say which dataset a claim or metric applies to — the UPR train/val/test data or the auxiliary OMCT pairwise data (`omct-` runs) — and which split where relevant. Never leave the dataset implicit for the reader to guess.

## Repository layout (target)

    .
    ├── PLAN.md
    ├── AGENTS.md
    ├── TEXT_PROCESSING_PLAN.md  # satellite track: text processing / segmentation
    ├── pyproject.toml
    ├── configs/                 # the single base config (paths, defaults)
    │   └── base.yaml            # experiments live in experiments/*.py sweeps, authored in Python
    ├── catalogs/                # committed label catalogs (name / description / parent)
    │   └── upr_labels.jsonl     # the text-processing track's CS-11 catalog (+ its meta file)
    ├── data/                    # gitignored
    │   ├── raw/                 # original JSONL (read-only)
    │   ├── external/            # auxiliary datasets from collaborators (see PLAN §2)
    │   ├── processed/
    │   └── splits/              # fixed split id lists (commit these, they are small)
    ├── cache/                   # gitignored: embedding cache
    ├── results/                 # gitignored: one folder per run_id
    ├── docs/
    │   ├── experiments.md            # append-only experiment log
    │   ├── experiments_table_upr.md  # UPR comparison tables, auto-appended by CLI commands
    │   ├── experiments_table_omct.md # OMCT comparison tables, auto-appended by CLI commands
    │   ├── experiments_text_processing.md            # text-processing track log (append-only)
    │   ├── experiments_table_text_processing_omct.md  # track tables, auto-appended
    │   ├── experiments_table_text_processing_upr.md   # track tables (UPR stage)
    │   └── text_processing_plan_archive.md  # the track plan's executed pre-registrations (append-only)
    ├── notebooks/               # exploration only, no logic that others import
    ├── experiments/             # experiment sweep files: typed Cells -> resolved configs (`cluster-rag sweep --file`)
    ├── experiments_text_processing/  # the text-processing track's sweep files
    ├── tools/                   # repo tools: verification scripts over the run artifacts
    ├── src/cluster_rag/
    │   ├── config.py
    │   ├── data.py              # Record model, loading, validation, splits
    │   ├── preprocess.py
    │   ├── pairs.py             # auxiliary PairRecord model + loader/validation
    │   ├── dataset_stats.py     # UPR dataset statistics (pure functions)
    │   ├── omct_probe.py        # omct probe driver (numpy + eval metrics only)
    │   ├── omct_analysis/        # track's Stage-0 analysis driver (analyze-omct)
    │   │   ├── helpers.py       # the vocabulary + the leaf helpers
    │   │   ├── view.py          # the dataset view + the encoding
    │   │   ├── arms.py          # the ladder's arms + the comparison
    │   │   ├── stage3.py        # the learned pack + its reporting
    │   │   ├── decision.py      # the decision models + the safe-negative rule
    │   │   └── report.py        # the artifact one run writes
    │   ├── omct_decision.py     # track's Stage-3.5 decision-probe driver (decide-omct)
    │   ├── upr_analysis.py      # track's UPR analysis driver (analyze-upr)
    │   ├── sweep.py             # SweepCell, define_sweep (sweep cells -> configs)
    │   ├── experiment_table.py  # markdown comparison tables (upserted rows)
    │   ├── text_processors/     # text-processing track: chunkers, features, combiners (pure)
    │   │   ├── aggregation.py   # the chunk-level aggregation rules (Stage 1.1a)
    │   │   ├── calibration.py   # the Stage-3 reporting's pure core: calibration, thresholds, margins
    │   │   ├── control.py       # the same-document grid + the seeded control sample (subset B)
    │   │   ├── decision/        # the decision models' pure side (Stage 3.5): the question
    │   │   │                   # specs, the call plan, the competence control, the measurement
    │   │   ├── rerank/          # the rerankers' pure side (CS-15): the candidate sets,
    │   │   │                   # their ceilings, the rank fusion, the measurement
    │   ├── embed/
    │   │   ├── base.py          # Embedder protocol
    │   │   ├── hf.py            # sentence-transformers backend
    │   │   ├── ollama.py        # Ollama /api/embed backend
    │   │   ├── random.py        # seeded random vectors (the track's Stage-3 floor)
    │   │   └── cache.py
    │   ├── decision/            # the decision models at the edge (Stage 3.5): the Decider
    │   │   ├── base.py          #   port + the answer shapes
    │   │   ├── systemone.py     #   the /v1/systemone client (a raw HTTP call)
    │   │   └── cache.py         #   the decision models' own disk cache
    │   ├── index.py             # ExampleIndex, LabelIndex
    │   ├── retrieve/            # knn.py, prototypes.py, zeroshot.py, fusion.py, threshold.py
    │   ├── cluster.py           # discovery + routing
    │   ├── rerank/              # the rerankers at the edge (CS-15): base.py (the port),
    │   │                        # hf.py (the HF client), cache.py (their own disk cache)
    │   ├── feedback/            # store.py (SQLite), updates.py (memory, negatives, Rocchio)
    │   ├── active.py            # selection strategies
    │   ├── eval/                # metrics.py, run.py, simulate.py
    │   ├── commands/            # the CLI's commands, one module per family
    │   │   ├── _common.py       # the shared helpers (the run id, the table append, the guard)
    │   │   ├── stats.py         # stats, stats --omct, split
    │   │   ├── embed.py         # embed, labelembed
    │   │   ├── omct.py          # probe-omct, analyze-omct, decide-omct
    │   │   ├── upr.py           # evaluate, analyze-upr, decide-upr, rerank-upr
    │   │   └── sweep.py         # the sweep runner
    │   └── cli.py               # the argparse layer + the dispatch (thin)
    └── tests/

## Environment
- Python 3.11+. Use `uv` (preferred) or `pip` with a virtualenv.
- Setup: `uv sync` (or `pip install -e ".[dev]"`).
- Core deps: `numpy`, `pandas`, `scikit-learn`, `sentence-transformers`, `httpx` (or `ollama`), `pydantic`, `pyyaml`, `umap-learn`, `hdbscan`, `iterative-stratification`, `nltk` (the track's POS tagger, CS-34), `pytest`, `ruff`.
- Optional deps, added only when their phase starts: `faiss-cpu` / `hnswlib`, `bertopic`, `setfit`, `rank-bm25`, `fastapi`, `streamlit`, `mlflow`.

### Ollama
- Default host `http://localhost:11434`, overridable with the env var `OLLAMA_HOST`.
- Embeddings: `POST /api/embed` with `{"model": ..., "input": [texts]}`. Send batches, and handle connection errors with a clear message ("is `ollama serve` running? did you `ollama pull <model>`?").
- Record the model name **and digest** (`ollama show` / `/api/tags`) in the run metadata.
- LLM reranking: use `format: "json"` or a JSON schema where supported, and always check the output with pydantic.

### Hugging Face
- Use `sentence-transformers`, pin the model `revision` in the config when possible, and respect `HF_HOME` for the cache.
- Always set `normalize_embeddings=True` (or normalize manually) so that dot product = cosine similarity.

### Model prefixes (critical)
Many embedding models expect task prefixes, and a wrong prefix silently hurts quality. Prefixes live in config per model, for example:
- e5 family: `"query: "` for texts being classified, `"passage: "` for indexed docs (or `"query: "` for both in symmetric tasks; test both).
- nomic-embed-text: `"classification: "`, `"clustering: "`, `"search_query: "`, `"search_document: "`.
- Instruction models (bge, gte-Qwen, qwen embedding, etc.): follow the model card.
Check the prefixes against the model card before running experiments.

## Coding conventions
- Type hints everywhere, and docstrings on public functions. Run `ruff check` and `ruff format` over `src`, `tests` *and* `experiments/` before finishing (sweep files live outside src).
- Embeddings are `np.float32`, shape `(n, d)`, L2-normalized. Assert this at index boundaries.
- Pure functions for scoring and metrics, with no hidden global state. Components take a config object.
- Dependency direction: `cli.py` (the argparse layer and the dispatch) and `commands/` (the commands themselves) orchestrate, component modules hold the logic, and pure cores (`data.py`, `dataset_stats.py`, `pairs.py`, `retrieve/`, `eval/`, `active.py`) stay free of model/I/O client imports (`torch`, `sentence-transformers`, `httpx`, `faiss`, `sqlite3`). Only `embed/*` and `feedback/*` touch models, caches and the database. Ports are plain `typing.Protocol` (e.g. `embed/base.py`), so tests can substitute fakes without network access or model downloads.
- The text-processing track's package (`text_processors/`) is a pure core too: it may import `cluster_rag.embed.base` (Protocol + numpy) and nothing heavier — no `torch`/`sentence-transformers`/`httpx`/`sqlite3`. Model clients (embedders, Ollama LLMs) are wired in CLI code, never inside the package. A test enforces this. One exception, reviewed and recorded in `TEXT_PROCESSING_PLAN.md` §2: the track's *combiners* (`text_processors/combining/logistic.py`, `text_processors/combining/histgb.py`) import scikit-learn — a core dependency of the main pipeline, with no model download, no network and no I/O. The track's *decision* side (`text_processors/decision/`) stays pure the same way: the question specs, the call plan, the competence control and the measurement are pure, and the model client (`cluster_rag/decision/`, the `/v1/systemone` caller) lives at the edge.
- The **part-of-speech tagger** (the track's CS-34) is a model client too (it downloads a model): it lives at the edge in `src/cluster_rag/pos/` (the `base.py` port, the `nltk_tagger.py` backend, its own disk cache), and the track's package imports the **`cluster_rag.pos.base` port** only.
- **scikit-learn is not a model client** (no model download, no network, no I/O), so it is allowed in the component modules — it is not on the forbidden list above. The track's pure package keeps its one documented exception (`combining/`); the main pipeline's `predictors/` gained its first scikit-learn import with `lexical_tfidf` (the track's CS-29 lexical floor, `TEXT_PROCESSING_PLAN.md` §7.4).
- No hard-coded paths, thresholds or model names in the logic. Put them in `configs/`.
- Pass explicit seeds (`numpy`, `random`, `torch`, UMAP `random_state`).
- Logging uses the `logging` module, not `print`, in library code.
- Never modify files in `data/raw/`.

## Data and evaluation discipline (no leakage)
- Splits are fixed once (Phase 1) and loaded from `data/splits/`. Never re-split silently.
- The indexes (examples, prototypes, clusters, thresholds) are built from **train only**. Val is for tuning, and **test is touched only for final reported numbers.**
- The label-holdout split must exclude the held-out labels everywhere in train artifacts, including label prototypes built from examples (zero-shot name/description embeddings are allowed).
- The feedback simulator may reveal gold labels only for items the active learner selected, one at a time, and must log every revealed item.
- Always report per frequency bucket (head / torso / tail / unseen) as well as overall numbers.
- Auxiliary pairwise data in `data/external/` (e.g. OMCT GTI paragraph-question pairs) is for model selection, threshold calibration and optional reranker/fine-tune training only. It never enters UPR train artifacts and is never reported as a UPR metric; its runs get the `omct-` run_id prefix (AGENTS.md "Runs and where things live").

## Runs and where things live

Every run (one CLI command execution or sweep cell) is identified by its
`run_id`; all its artifacts live in `results/<run_id>/`.

- **Run-id grammar:** `<dataset>-<command>-<detail>[-<split>]` — e.g.
  `upr-embed-e5-base-v2`, `upr-eval-mf-topk2-val`, `omct-stats-base`,
  `omct-probe2-arctic2`, `omct-decide-tev1-pair-noul`. Datasets: `upr` (the
  UPR train/val/test data) and `omct` (the auxiliary OMCT pairwise data).
- **Prefixes per command:** the OMCT commands (`probe-omct`, `analyze-omct`,
  `decide-omct`, and `stats --omct`) add the `omct-` prefix themselves at run time — configs and sweep
  cells carry the bare detail (`run_id: stats-base` → `results/omct-stats-base/`);
  a run_id that already carries a dataset prefix is warned about (it would
  double up). The UPR commands (`stats`/`split`/`embed`/`evaluate`) take the
  run_id as configured and only **warn** when it lacks `upr-` (never a hard
  error).
- **`results/<run_id>/` contents:** the frozen resolved `config.yaml` plus the
  command's artifact (`stats.json`, `split_report.json`, `embeddings.npz` +
  `meta.json`, `probe_report.json`, `metrics.json`).
- **Doc files:** `docs/experiments.md` is the append-only narrative log (one
  row/section per run with revisions/digests and findings — old rows are never
  edited). The comparison tables are per dataset:
  `docs/experiments_table_upr.md` and `docs/experiments_table_omct.md`,
  auto-appended by the CLI commands and upserted by run key (the latest run
  per key wins). Never edit table rows by hand. A sweep owns its section
  heading (`## UPR — <sweep-label>` / `## OMCT — <sweep-label>` plus
  `… - groups` for the probe's per-document breakdown); standalone runs land
  directly under `## <UPR|OMCT> — <command>`.
- **The text-processing track owns its own pair of tables and its own log**
  (`TEXT_PROCESSING_PLAN.md`): `docs/experiments_text_processing.md` (the
  track's append-only narrative log) and
  `docs/experiments_table_text_processing_omct.md` /
  `docs/experiments_table_text_processing_upr.md` (auto-appended, upserted by
  run key). Which file a command writes to is **command-derived, never a
  heuristic**: every command maps to a `(dataset, track)` pair in
  `experiment_table.py` (`COMMAND_TARGETS`), and the track picks the file
  (main pipeline → the per-dataset files; text processing → the track's
  files). The config keys are `paths.experiment_table_upr` /
  `_omct` and `paths.experiment_table_text_processing_omct` / `_upr`
  (unset → the append is skipped). The track's sweeps live in
  `experiments_text_processing/`; the sweep loader keeps `experiments/`
  importable so `embedder_pool.MODELS` stays shared.
- **Legacy ids (pre-2026-10-20):** runs recorded under the old naming
  (`base`, `embed-*`, `ev-*`, `aux-*`) were migrated to `upr-`/`omct-` ids in
  Phase 3.5 after a verified field-for-field re-run (see the migration note at
  the end of `docs/experiments.md`); their old result folders were deleted.
  Existing log rows keep the legacy ids they were written with — when a doc
  row predates 2026-10-20 and names another prefix, see that mapping note
  instead of renaming anything.

## Running things (target CLI)

    uv run cluster-rag stats      --config configs/base.yaml
    uv run cluster-rag split      --config configs/base.yaml
    uv run cluster-rag sweep      --file experiments/omct_probe_sweep.py
    uv run cluster-rag embed      --config <yaml with an embed: section>
    uv run cluster-rag probe-omct --config <yaml with an embed: section>
    uv run cluster-rag decide-omct --config <yaml with an analysis.decision: section>
    uv run cluster-rag evaluate   --config <yaml with an eval: section> --split val
    uv run cluster-rag clusters   --config <yaml with a clusters: section>
    uv run cluster-rag simulate   --config <yaml with a simulate: section>

`uv run` auto-syncs the environment first. Experiments are **defined only in
Python sweep files** (`experiments/*.py`, typed cells resolved by
`define_sweep` over `configs/base.yaml`); the only authored YAML is
`configs/base.yaml` (paths + defaults + the stats/split standalone entry).
The equivalent low-level form is `python -m cluster_rag.cli <command> ...`
(same commands and flags); the `cluster-rag` console script is defined in
`pyproject.toml` under `[project.scripts]`.

Each command writes to `results/<run_id>/`: `config.yaml` (resolved), `metrics.json`, and any artifacts.

## Testing
- Unit tests for metrics (hand-computed small cases), fusion and thresholds, Rocchio updates, the feedback store, and cache keys.
- Use a tiny fixture dataset (`tests/fixtures/sample.jsonl`, about 20 rows) and a **fake embedder** (deterministic hash-based vectors) so tests need neither network access nor model downloads.
- Ollama/HF integration tests are marked `@pytest.mark.integration` and skipped by default.
- Integration tests skip by default; run them explicitly with `uv run pytest -o adopts= -m integration -q`.

## Experiment log format (`docs/experiments.md`)
Append one row per run:

    | date | run_id | phase | embedder | preprocess | method | split | P@5 | R@10 | micro-F1 | macro-F1 | shortlist R@50 | notes |

Column cells may be omitted ("n/a") for stats/split runs. The auto-appended
comparison tables live in the per-dataset table files (see "Runs and where
things live"); do not write their rows by hand.

The text-processing track keeps the same discipline in its own log,
`docs/experiments_text_processing.md` (append-only, one short section per
run: date, run_id, what, numbers, verdict), with its numbers in
`docs/experiments_table_text_processing_omct.md` /
`docs/experiments_table_text_processing_upr.md`. OMCT findings there
annotate method choice; they are never a UPR metric.

## Definition of done (per task)
- [ ] Code + tests, and `pytest` and `ruff` pass.
- [ ] Config entries added, nothing hard-coded.
- [ ] If the task produces metrics: run saved in `results/` and a row added to `docs/experiments.md`.
- [ ] `PLAN.md` checkbox ticked, with any new open questions noted.
- [ ] Short summary of what changed and what the next step is.
