"""OMCT threshold-evidence sweep (Phase 4): pair score distributions.

PLAN Phase 4 thresholding names the OMCT pair score distributions as the
threshold-calibration evidence (strictly auxiliary: what judged same-topic vs
not-same-topic cosines look like on the collaborator pair data — never a UPR
metric, its operating points are calibrated on UPR val). `omct_probe` now
carries a `score_distributions` block per transform variant (judged pair
cosines by label + the "should answer nothing" rows' best same-document
similarity); these four cells re-run the probe for the four UPR-val
semifinalists to produce that artifact (warm: all paragraph/question vectors
are cache hits).

Section "## OMCT — thresholds"; run ids `omct-probe-thresholds-<model>`.

    uv run cluster-rag sweep --file experiments/omct_threshold_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.sweep import ProbeCell, define_sweep

SEMIFINALISTS: tuple[str, ...] = ("embeddinggemma", "jina-v5", "voyage-4-nano", "arctic2")

cells: list[ProbeCell] = [
    ProbeCell(label=f"probe-thresholds-{detail}", embed=MODELS[detail]) for detail in SEMIFINALISTS
]

SWEEP = define_sweep(cells, dataset="omct", label="thresholds")
