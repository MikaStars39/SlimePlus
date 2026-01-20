

"""
`mika_eval.data.tasks` is a convenience namespace that:
- exposes the shared dataset registry / helpers
- imports all `load_*` dataset loaders so callers can `getattr(tasks, "load_xxx")`

Important: loader modules must import shared helpers from `mika_eval.tasks.base`
to avoid circular imports during package initialization.
"""

from mika_eval.tasks.base import (
    DATASETS, 
    get_answer_text, 
    get_question_text, 
    load_dataset_from_hf,
)

__all__ = [
    "DATASETS",
    "get_question_text",
    "get_answer_text",
    "load_dataset_from_hf",
]