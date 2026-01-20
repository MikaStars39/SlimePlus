

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
    prepare_pass_at_k_jsonl,
)

# Import loaders (kept at module scope so they're available via `mika_eval.data.tasks.load_xxx`)
from mika_eval.tasks.aime2024 import load_aime2024
from mika_eval.tasks.aime2025 import load_aime2025
from mika_eval.tasks.amc2023 import load_amc2023
from mika_eval.tasks.hmmt2025 import load_hmmt2025
from mika_eval.tasks.math500 import load_math500
from mika_eval.tasks.minerva import load_minerva
from mika_eval.tasks.mmlu_pro import load_mmlu_pro
from mika_eval.tasks.ifeval import load_ifeval
from mika_eval.tasks.gpqa_diamond import load_gpqa_diamond
from mika_eval.tasks.ceval import load_ceval
from mika_eval.tasks.DAPO_Math_17k_Processed import load_DAPO_Math_17k_Processed
from mika_eval.tasks.ifbench import load_ifbench

__all__ = [
    "DATASETS",
    "get_question_text",
    "get_answer_text",
    "load_dataset_from_hf",
    "load_aime2024",
    "load_aime2025",
    "load_amc23",
    "load_amc2023",
    "load_math500",
    "load_minerva",
    "load_hmmt25",
    "load_hmmt2025",
    "load_mmlu_pro",
    "load_ifeval",
    "load_gpqa_diamond",
    "load_ceval",
    "load_DAPO_Math_17k_Processed",
    "load_ifbench",
]