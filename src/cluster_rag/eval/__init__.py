"""Evaluation metrics (Phases 3+); pure functions, numpy only.

The pairwise-retrieval subset here (:func:`roc_auc`,
:func:`pairwise_retrieval_metrics`, :func:`bucket_pair_statistics`) is
exercised by the Phase 2 auxiliary probe. The full P@k/R@k/F1 set arrives in
Phase 3. Everything takes plain arrays and returns JSON-serializable dicts,
with no file I/O, model clients or global state (AGENTS.md
dependency-direction rule).
"""
