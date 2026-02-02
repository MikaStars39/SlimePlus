

"""
`open_gym.data.tasks` is a convenience namespace that:
- exposes the shared dataset registry / helpers
- imports all `load_*` dataset loaders so callers can `getattr(tasks, "load_xxx")`

Important: loader modules must import shared helpers from `open_gym.tasks.base`
to avoid circular imports during package initialization.
"""

from open_gym.tasks.base import (
    DATASETS, 
    get_answer_text, 
    get_question_text, 
    load_dataset_from_hf,
)

from .process_func.aime2024 import load_aime2024
from .process_func.aime2025 import load_aime2025
from .process_func.amc2023 import load_amc2023
from .process_func.math500 import load_math500
from .process_func.minerva import load_minerva
from .process_func.hmmt2025 import load_hmmt2025
from .process_func.gpqa_diamond import load_gpqa_diamond
from .process_func.ceval import load_ceval
from .process_func.DAPO_Math_17k_Processed import load_DAPO_Math_17k_Processed
from .process_func.ifeval import load_ifeval
from .process_func.ifbench import load_ifbench
from .process_func.mmlu_pro import load_mmlu_pro
from .process_func.ceval import load_ceval

__all__ = [
    "DATASETS",
    "get_question_text",
    "get_answer_text",
    "load_dataset_from_hf",
    "load_aime2024",
    "load_aime2025",
    "load_amc2023",
    "load_math500",
    "load_minerva",
    "load_hmmt2025",
    "load_gpqa_diamond",
    "load_ceval",
    "load_DAPO_Math_17k_Processed",
    "load_ifeval",
    "load_ifbench",
    "load_mmlu_pro",
    "load_ceval",
]