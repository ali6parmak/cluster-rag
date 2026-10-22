# Text-processing track sweeps

Sweep files for the text-processing / segmentation track
(`TEXT_PROCESSING_PLAN.md`). They live here, parallel to `experiments/`, so
the main pipeline's sweeps stay where they are (reviewer decision,
2026-10-22).

Two rules for a sweep file in this directory:

1. **The shared embedder pool stays shared.** `experiments/embedder_pool.py`
   holds the card-verified model configs (prefixes, pinned revisions, backend
   knobs) for the whole pool. The `sweep` command's loader puts both this
   directory *and* `experiments/` on `sys.path` while a sweep file executes,
   so a sweep here can simply `from embedder_pool import MODELS` — the same
   models, prefixes and revisions as the main pipeline's sweeps, so the two
   lines stay comparable.
2. **A sweep is one dataset, one track.** `define_sweep(cells, dataset="omct",
   label="<sweep-label>")` resolves the cells over `configs/base.yaml`. The
   cells' *command* decides the comparison-table file: the track's commands
   own `docs/experiments_table_text_processing_omct.md`, the main pipeline's
   commands the per-dataset files. Run ids keep the `omct-`/`upr-` dataset
   prefixes, so a track row is still visibly auxiliary.

The first sweep file (CS-1, `analyze-omct` — Stage 0.1's artifact command)
lands here.
