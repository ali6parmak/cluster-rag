"""The CLI's command implementations, one module per family.

``cluster_rag.cli`` keeps the argparse layer and the dispatch — the thin
shell a user touches. The commands themselves live here:

- ``stats`` — ``stats``, ``stats --omct``, ``split``
- ``embed`` — ``embed``, ``labelembed``
- ``omct`` — ``probe-omct``, ``analyze-omct``, ``decide-omct``
- ``upr`` — ``evaluate``, ``analyze-upr``, ``decide-upr``, ``rerank-upr``
- ``sweep`` — the sweep runner
- ``_common`` — the shared helpers (the run-id plumbing, the table append,
  the whole-text guard)
"""
