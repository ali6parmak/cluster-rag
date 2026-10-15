# PLAN.md: Retrieval-Augmented Multi-Label Topic Classification with Feedback

## 1. Goal

Build a system that suggests topic labels for short texts (e.g. UPR-style human-rights recommendations). It should:

- work with **thousands of labels** (likely 2000+, long-tailed, and not all known yet),
- treat labeling as **retrieval**: find similar labeled examples and candidate labels, then score or rerank them,
- use **clustering** to explore the data and to narrow the search coarse-to-fine,
- **improve from user feedback** (accept / reject / add label) without full retraining,
- use **active learning** to choose which items a human should review next.

### Non-goals (for now)
- Production deployment, auth, multi-user scaling.
- Training large models from scratch.

## 2. Data

Input format is JSONL, one record per line:

    {"id": "...", "text": "...", "language": "en", "labels": ["Discrimination", "Sexual Orientation and Gender Identity"]}

Assumptions to verify in Phase 1:
- Multi-label, so a record can have 1..n labels.
- The label set in the dataset is **incomplete** compared with the real taxonomy.
- There may be several languages (the `language` field).
- Texts are formulaic ("Continue to...", "Ensure that..."), which may bias embeddings.

Optional extra resource: a label catalog file (`labels.jsonl`) with `name`, an optional `description`, and an optional `parent`. Even labels with zero examples can be retrieved through it.

## 3. Target architecture

    Text -> Preprocess -> Embed -> [Cluster routing: top-n clusters]
         -> Retrieve: (a) top-k similar labeled examples (Example Index)
                      (b) top-m candidate labels       (Label Index)
         -> Score fusion (kNN votes + prototype sim + zero-shot sim - negative memory)
         -> [Optional rerank: cross-encoder or LLM with retrieved few-shot examples]
         -> Suggestions (labels + scores + evidence)
         -> User feedback -> Feedback Store -> online index/prototype updates
                                            -> periodic fine-tuning

Core components:
| Component | Responsibility |
|---|---|
| Embedding backends | One interface for Hugging Face (sentence-transformers) and Ollama (`/api/embed`). Results are cached on disk. |
| Example Index | Vectors of labeled texts plus their label sets. Starts as numpy and moves to FAISS/hnswlib if needed. |
| Label Index | One vector per label: a blend of name/description embedding and the mean of its examples (prototype). |
| Cluster tree | Hierarchical clusters over examples and/or labels, used for routing and discovery. |
| Scorer | Fuses signals into per-label scores and applies thresholds. |
| Reranker (optional) | Cross-encoder or a local LLM via Ollama that picks final labels from a shortlist. |
| Feedback Store | SQLite log of feedback events. It is the source of truth for online updates. |
| Active learner | Picks items for human review. |
| Evaluation harness | Offline metrics plus a simulated-user feedback loop. |

## 4. Phases

Each phase lists **tasks**, **deliverables** and **exit criteria**. Do them in order, because later phases depend on the evaluation harness from Phase 3.

### Phase 0: Project setup
- [x] Create the repo skeleton (see AGENTS.md "Repository layout").
- [x] Set up `pyproject.toml`, ruff, pytest and a config system (YAML + pydantic or dataclasses).
- [x] Add `.gitignore` entries for `data/`, `cache/`, `results/` artifacts and model files.
- [x] Write a smoke test that loads the sample JSONL.

**Exit:** `pytest` passes and the CLI works (`uv run cluster-rag --help`, equivalently `python -m cluster_rag.cli --help`).

### Phase 1: Data exploration and splits
- [ ] Loader with validation (pydantic `Record` model) that reports and drops malformed rows.
- [ ] Stats: number of records, number of unique labels, labels per record, label frequency distribution (head / torso / tail buckets), label co-occurrence, languages, text lengths.
- [ ] Prefix analysis: most common leading 1-3 words ("Continue to", "Ensure that", ...).
- [ ] Multi-label **iterative stratified** split into train / val / test (e.g. `iterative-stratification` or `scikit-multilearn`). Fix and save the split ids.
- [ ] Create a "label-holdout" split: some labels removed entirely from train, to test zero-shot and new-label discovery.

**Deliverables:** `notebooks/01_eda.ipynb` or a script report, plus `data/splits/*.json`.
**Exit:** The splits are reproducible and the label statistics are written to `docs/experiments.md`.

### Phase 2: Embedding layer
- [ ] `Embedder` protocol: `encode(texts, kind: "query"|"document"|"label") -> np.ndarray` (L2-normalized float32).
- [ ] `HFEmbedder` (sentence-transformers) and `OllamaEmbedder` (HTTP `POST /api/embed`, batched).
- [ ] Model-specific prefix handling (e.g. e5 `query:`/`passage:`, nomic `classification:`/`clustering:`, instruction-style prompts for bge/gte/qwen-embedding). Keep the prefixes in config, never hard-coded in the logic.
- [ ] Disk cache keyed by `(model_id, prefix, preprocessing_version, sha1(text))`.
- [ ] Preprocessing variants: `raw`, `strip_leading_phrase` (remove formulaic openers), `lowercase`.
- [ ] Candidate models to compare (pick 3-5): `intfloat/multilingual-e5-base|large`, `BAAI/bge-m3`, `sentence-transformers/all-MiniLM-L6-v2` (fast baseline), Ollama `embeddinggemma`, `mxbai-embed-large`, `bge-m3`, `snowflake-arctic-embed`.

**Exit:** The full dataset embeds with at least 2 backends and cache hits skip re-encoding.

### Phase 3: Evaluation harness (build before modeling)
- [ ] Metrics: P@k, R@k (k = 1, 3, 5, 10), nDCG@k, micro-F1, macro-F1, sample-F1, label coverage. Report each per frequency bucket (head / torso / tail / unseen).
- [ ] **Shortlist recall@m** (m = 20, 50, 100): is the gold label anywhere in the retrieved candidates? This is the ceiling for any reranker.
- [ ] Trivial baselines: most-frequent labels and random.
- [ ] Results written as JSON to `results/<run_id>/metrics.json` together with the resolved config, plus a one-line summary appended to `docs/experiments.md`.

**Exit:** One command evaluates any predictor on val or test.

### Phase 4: Retrieval baselines (no clustering yet)
- [ ] **kNN voting:** top-k neighbors in the Example Index, with labels scored by similarity-weighted votes (optionally sim^p or softmax with temperature).
- [ ] **Label prototypes:** mean of the example embeddings per label.
- [ ] **Zero-shot label embeddings:** embed label names and descriptions, optionally with templates such as "This recommendation is about {label}."
- [ ] **Fusion:** `score = a*knn + b*proto + c*zeroshot`. Tune the weights on val.
- [ ] **Thresholding:** global, per-label (calibrated on val) and top-k with a minimum score. Compare them.
- [ ] Optional lexical signal: BM25 over texts and label names, for hybrid retrieval.
- [ ] Ablations: embedding model x preprocessing x k x fusion weights.

**Exit:** The best config is chosen on val, with test results reported once. Expect kNN + prototype fusion to clearly beat the trivial baselines.

### Phase 5: Clustering (discovery and routing)
- [ ] Discovery: UMAP + HDBSCAN (or BERTopic) on the embeddings. For each cluster, report size, label purity / entropy, top labels and c-TF-IDF keywords.
- [ ] Analyze the mixed clusters (overlapping labels?) and the outliers (missing labels?). On the label-holdout split, check whether held-out labels show up as coherent clusters.
- [ ] Routing: build a 2-level hierarchy (e.g. k-means over examples or over label prototypes). At query time, route to the **top-n clusters** and retrieve only inside them.
- [ ] Measure shortlist recall loss and speedup for routing vs flat retrieval, as a function of n.
- [ ] Cluster-level propagation: in pure, dense clusters, propagate the majority labels to unlabeled members with high confidence, and check the result on val.

**Exit:** A documented decision on whether routing is worth it at the current data size, plus a cluster report artifact.

### Phase 6: Reranking (optional, after Phase 4 is solid)
- [ ] Cross-encoder reranker on (text, label name/description) pairs. Zero-shot first, then fine-tuned with training pairs.
- [ ] LLM reranker via Ollama (e.g. a 7-14B instruct model) with the prompt = text + shortlist (labels + descriptions) + top retrieved examples as few-shot. It must return **strict JSON** (`{"labels": [...], "confidence": {...}}`), checked against the schema, with retry on parse failure.
- [ ] Compare against fusion alone: quality gain vs latency and cost.

**Exit:** The reranker is enabled only if it improves val F1 / P@k by a meaningful margin.

### Phase 7: Feedback store and online learning
- [ ] SQLite schema `feedback(id, item_id, text, label, action, source, user, ts, model_version)` with `action` ∈ {accept, reject, add, remove}.
- [ ] Online update rules (no retraining):
  - accept/add: the item goes into the Example Index with its confirmed labels.
  - reject: the (text embedding, label) pair goes into the **negative memory**, and similar future texts get a penalty for that label.
  - Rocchio update of label prototypes (`a*proto + b*mean(accepted) - c*mean(rejected)`, then renormalize).
  - Corrected items become few-shot candidates for the LLM reranker.
- [ ] Every update can be replayed from the event log (rebuild the indexes from scratch at any time).
- [ ] **Simulated user** for evaluation: an oracle that answers with the gold labels, optionally with configurable noise.
- [ ] Learning curve: metric vs number of feedback events, with and without each update rule.

**Exit:** The simulation shows metrics improving with feedback, and the indexes can be rebuilt from the log.

### Phase 8: Active learning
- [ ] Selection strategies: random (baseline), uncertainty (small margin between top-2 scores, or a score near the threshold), cluster representatives (medoids of large unlabeled clusters), boundary items, outliers, and a diversity-aware batch mix.
- [ ] Exploration: occasionally show lower-ranked labels (epsilon-greedy or Thompson sampling) to avoid feedback loops.
- [ ] Compare the strategies in the simulator with an equal labeling budget.

**Exit:** At least one strategy beats random on the learning curve.

### Phase 9: Embedding / model fine-tuning
- [ ] Build training data from the labels plus accumulated feedback: (text, positive label/text, hard negative). Hard negatives come from rejected suggestions and confusable labels.
- [ ] Options: SetFit, sentence-transformers `MultipleNegativesRankingLoss`, or a bi-encoder over (text, label description) pairs.
- [ ] Retrain on a schedule (e.g. every N feedback events), re-embed, and compare with the previous model version.

**Exit:** The fine-tuned embedder improves shortlist recall and tail-label macro-F1 on test.

### Phase 10: Interface (optional)
- [ ] FastAPI endpoints: `/suggest`, `/feedback`, `/next_items`.
- [ ] Minimal Streamlit or Gradio UI: show a text, the suggested labels with evidence (nearest examples), and accept/reject/add buttons.

## 5. Experiment tracking
- Every run gets a `run_id`, a frozen config and metrics in `results/`, plus one line in `docs/experiments.md`.
- Fix all random seeds. Record model ids and versions (HF revision or Ollama model digest).
- Optional: MLflow or a simple CSV leaderboard.

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Embeddings group texts by formulaic openers rather than topic | Preprocessing variants and instruction prefixes; check in the Phase 5 cluster report |
| Long tail / unseen labels | Zero-shot label embeddings, label descriptions, label-holdout evaluation |
| One global threshold fits badly | Per-label calibration, top-k + minimum score |
| Feedback loop bias | Exploration and the "add label" action; log impressions as well as clicks |
| Noisy feedback | Weight by source, cap the effect of a single event, replayable log |
| Data leakage | Strict split discipline (see AGENTS.md) |

## 7. Open questions
- Is the full label taxonomy available with descriptions and hierarchy?
- How many languages are there, and what is the volume per language?
- Latency budget per suggestion (decides whether an LLM rerank is feasible)?
- Expected data size (decides between numpy, FAISS and hnswlib)?

## 8. Milestones
1. **M1:** Phases 0-3, the evaluation harness with baselines.
2. **M2:** Phase 4, a strong retrieval baseline.
3. **M3:** Phases 5-6, the clustering report and reranker decision.
4. **M4:** Phases 7-8, feedback + active learning shown working in simulation.
5. **M5:** Phases 9-10, fine-tuning and a demo UI.
