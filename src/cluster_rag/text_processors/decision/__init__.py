"""The decision models' pure side (the text-processing track's Stage 3.5).

Two pure modules:

- :mod:`cluster_rag.text_processors.decision.questions` — the typed question
  specs (``noul`` / ``choice``) and the prompt template's rendering;
- :mod:`cluster_rag.text_processors.decision.probe` — the cells, the call
  plan, the **synthetic competence control** and the measurement blocks.

Pure core (AGENTS.md dependency direction): numpy and the standard library
only — no torch, sentence-transformers, httpx or sqlite3. The model client
(the ``/v1/systemone`` caller) lives in :mod:`cluster_rag.decision`, at the
edge, and the CLI wires it in.
"""

from cluster_rag.text_processors.decision.probe import (
    GROUP_COMPETENCE,
    GROUP_CONTROL,
    GROUP_JUDGED,
    GROUP_UNJUDGED,
    STATE_PAIR,
    STATE_PARAGRAPH,
    STATE_SHAPES,
    DecisionCall,
    DecisionCell,
    build_competence_cells,
    measure_competence_control,
    measure_decision_arm,
    plan_calls,
)
from cluster_rag.text_processors.decision.questions import (
    QUESTION_PLACEHOLDER,
    QUESTION_TYPE_CHOICE,
    QUESTION_TYPE_NOUL,
    QUESTION_TYPES,
    QuestionSpec,
    build_question_spec,
    render_instructions,
)

__all__ = [
    "GROUP_COMPETENCE",
    "GROUP_CONTROL",
    "GROUP_JUDGED",
    "GROUP_UNJUDGED",
    "QUESTION_PLACEHOLDER",
    "QUESTION_TYPES",
    "QUESTION_TYPE_CHOICE",
    "QUESTION_TYPE_NOUL",
    "STATE_PAIR",
    "STATE_PARAGRAPH",
    "STATE_SHAPES",
    "DecisionCall",
    "DecisionCell",
    "QuestionSpec",
    "build_competence_cells",
    "build_question_spec",
    "measure_competence_control",
    "measure_decision_arm",
    "plan_calls",
    "render_instructions",
]
