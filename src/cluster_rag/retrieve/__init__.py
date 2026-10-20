"""Retrieval scoring cores (Phase 4): pure functions over embedding matrices.

These implement the scoring math consumed by the predictors
(:mod:`cluster_rag.predictors`) and, later, the fusion scorer: pure numpy,
train-only inputs, no model clients and no I/O (AGENTS.md dependency-direction
rule). Every consumer asserts the embedding contract (float32, 2-D, unit rows)
at its own boundary before calling in here.
"""
