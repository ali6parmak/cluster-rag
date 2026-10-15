# AGENTS.md

Instructions for AI coding agents (and humans) working in this repo. Read `PLAN.md` first, since it defines the phases and the current scope.

## Project summary
Retrieval-augmented multi-label topic classification. Texts are embedded (Hugging Face or Ollama models), similar labeled examples and candidate labels are retrieved (optionally routed through clusters), scores are fused or reranked, and labels are suggested. User feedback updates the system online, and active learning chooses which items to review.

## Working agreement
1. **Follow PLAN.md phase order.** Do not start a phase before the exit criteria of its dependencies are met. The Phase 3 evaluation harness comes before any modeling work.
2. **Small, focused changes.** One task (one PLAN checkbox) per change set. When a task is done, tick its checkbox in `PLAN.md`.
3. **Explain before large changes.** For new dependencies, schema changes, or refactors touching more than 3 modules, describe the plan and wait for approval.
4. **Never claim results you did not measure.** Every metric you report must come from a run in `results/` and appear in `docs/experiments.md`.
5. **If something is ambiguous, ask.** Do not guess label semantics, data formats or thresholds. Add open questions to `PLAN.md` section 7.

## Repository layout (target)

    .
    ├── PLAN.md
    ├── AGENTS.md
    ├── pyproject.toml
    ├── configs/                 # YAML experiment configs
    │   ├── base.yaml
    │   └── experiments/*.yaml
    ├── data/                    # gitignored
    │   ├── raw/                 # original JSONL (read-only)
    │   ├── processed/
    │   └── splits/              # fixed split id lists (commit these, they are small)
    ├── cache/                   # gitignored: embedding cache
    ├── results/                 # gitignored except summaries: one folder per run_id
    ├── docs/
    │   └── experiments.md       # append-only experiment log
    ├── notebooks/               # exploration only, no logic that others import
    ├── src/cluster_rag/
    │   ├── config.py
    │   ├── data.py              # Record model, loading, validation, splits
    │   ├── preprocess.py
    │   ├── embed/
    │   │   ├── base.py          # Embedder protocol
    │   │   ├── hf.py            # sentence-transformers backend
    │   │   ├── ollama.py        # Ollama /api/embed backend
    │   │   └── cache.py
    │   ├── index.py             # ExampleIndex, LabelIndex
    │   ├── retrieve/            # knn.py, prototypes.py, zeroshot.py, fusion.py, threshold.py
    │   ├── cluster.py           # discovery + routing
    │   ├── rerank/              # cross_encoder.py, llm.py
    │   ├── feedback/            # store.py (SQLite), updates.py (memory, negatives, Rocchio)
    │   ├── active.py            # selection strategies
    │   ├── eval/                # metrics.py, run.py, simulate.py
    │   └── cli.py
    └── tests/

## Environment
- Python 3.11+. Use `uv` (preferred) or `pip` with a virtualenv.
- Setup: `uv sync` (or `pip install -e ".[dev]"`).
- Core deps: `numpy`, `pandas`, `scikit-learn`, `sentence-transformers`, `httpx` (or `ollama`), `pydantic`, `pyyaml`, `umap-learn`, `hdbscan`, `iterative-stratification`, `pytest`, `ruff`.
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
- Type hints everywhere, and docstrings on public functions. Run `ruff check` and `ruff format` before finishing.
- Embeddings are `np.float32`, shape `(n, d)`, L2-normalized. Assert this at index boundaries.
- Pure functions for scoring and metrics, with no hidden global state. Components take a config object.
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

## Running things (target CLI)

    uv run cluster-rag stats      --config configs/base.yaml
    uv run cluster-rag split      --config configs/base.yaml
    uv run cluster-rag embed      --config configs/experiments/e5_base.yaml
    uv run cluster-rag evaluate   --config configs/experiments/knn_proto.yaml --split val
    uv run cluster-rag clusters   --config configs/experiments/hdbscan.yaml
    uv run cluster-rag simulate   --config configs/experiments/feedback_sim.yaml

`uv run` auto-syncs the environment first. The equivalent low-level form is `python -m cluster_rag.cli <command> ...` (same commands and flags); the `cluster-rag` console script is defined in `pyproject.toml` under `[project.scripts]`.

Each command writes to `results/<run_id>/`: `config.yaml` (resolved), `metrics.json`, and any artifacts.

## Testing
- Unit tests for metrics (hand-computed small cases), fusion and thresholds, Rocchio updates, the feedback store, and cache keys.
- Use a tiny fixture dataset (`tests/fixtures/sample.jsonl`, about 20 rows) and a **fake embedder** (deterministic hash-based vectors) so tests need neither network access nor model downloads.
- Ollama/HF integration tests are marked `@pytest.mark.integration` and skipped by default.

## Experiment log format (`docs/experiments.md`)
Append one row per run:

    | date | run_id | phase | embedder | preprocess | method | split | P@5 | R@10 | micro-F1 | macro-F1 | shortlist R@50 | notes |

## Definition of done (per task)
- [ ] Code + tests, and `pytest` and `ruff` pass.
- [ ] Config entries added, nothing hard-coded.
- [ ] If the task produces metrics: run saved in `results/` and a row added to `docs/experiments.md`.
- [ ] `PLAN.md` checkbox ticked, with any new open questions noted.
- [ ] Short summary of what changed and what the next step is.
