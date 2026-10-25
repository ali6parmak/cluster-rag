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
6. **Explain plainly and state the scope.** When discussing concepts, results or reviews: one concept per paragraph, short sentences, define jargon on first use, and pair formal definitions with a small concrete example. Always say which dataset a claim or metric applies to — the UPR train/val/test data or the auxiliary OMCT pairwise data (`omct-` runs) — and which split where relevant. Never leave the dataset implicit for the reader to guess.

## Repository layout (target)

    .
    ├── PLAN.md
    ├── AGENTS.md
    ├── TEXT_PROCESSING_PLAN.md  # satellite track: text processing / segmentation
    ├── pyproject.toml
    ├── configs/                 # the single base config (paths, defaults)
    │   └── base.yaml            # experiments live in experiments/*.py sweeps, authored in Python
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
    │   └── experiments_table_text_processing_upr.md   # track tables (UPR stage)
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
    │   ├── omct_analysis.py     # track's Stage-0 analysis driver (analyze-omct)
    │   ├── sweep.py             # SweepCell, define_sweep (sweep cells -> configs)
    │   ├── experiment_table.py  # markdown comparison tables (upserted rows)
    │   ├── text_processors/     # text-processing track: chunkers, features, combiners (pure)
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
- Type hints everywhere, and docstrings on public functions. Run `ruff check` and `ruff format` over `src`, `tests` *and* `experiments/` before finishing (sweep files live outside src).
- Embeddings are `np.float32`, shape `(n, d)`, L2-normalized. Assert this at index boundaries.
- Pure functions for scoring and metrics, with no hidden global state. Components take a config object.
- Dependency direction: `cli.py` orchestrates, component modules hold the logic, and pure cores (`data.py`, `dataset_stats.py`, `pairs.py`, `retrieve/`, `eval/`, `active.py`) stay free of model/I/O client imports (`torch`, `sentence-transformers`, `httpx`, `faiss`, `sqlite3`). Only `embed/*` and `feedback/*` touch models, caches and the database. Ports are plain `typing.Protocol` (e.g. `embed/base.py`), so tests can substitute fakes without network access or model downloads.
- The text-processing track's package (`text_processors/`) is a pure core too: it may import `cluster_rag.embed.base` (Protocol + numpy) and nothing heavier — no `torch`/`sentence-transformers`/`httpx`/`sqlite3`. Model clients (embedders, Ollama LLMs) are wired in CLI code, never inside the package. A test enforces this. One exception, reviewed and recorded in `TEXT_PROCESSING_PLAN.md` §2: the track's *combiner* (`text_processors/combining/logistic.py`) imports scikit-learn — a core dependency of the main pipeline, with no model download, no network and no I/O.
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
  `omct-probe2-arctic2`. Datasets: `upr` (the UPR train/val/test data) and
  `omct` (the auxiliary OMCT pairwise data).
- **Prefixes per command:** the OMCT commands (`probe-omct`, and `stats
  --omct`) add the `omct-` prefix themselves at run time — configs and sweep
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
