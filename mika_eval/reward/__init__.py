# ------------------------------- Nemo Gym -------------------------------

from .math_with_judge.score import score_fn as math_score_fn
from .code_gen.score import score_fn as code_score_fn
from .mcqa.score import score_fn as mcqa_score_fn
from .instruction_following.score import score_fn as if_score_fn
from .structured_outputs.score import score_fn as so_score_fn
from .workplace_assistant.score import score_fn as wa_score_fn

# Map data_source values (from parquet) to score functions
score_fn_dict = {
    # Primary data_source names (from parquet)
    "nemogym_math": math_score_fn,
    "nemogym_code": code_score_fn,
    "nemogym_mcqa": mcqa_score_fn,
    "nemogym_if": if_score_fn,
    "nemogym_structured": so_score_fn,
    "nemogym_workplace": wa_score_fn,
}


def get_score_fn(data_source: str):
    """
    Get score function by data_source.

    Args:
        data_source: The data_source value from the parquet file (e.g., "nemogym_math")

    Returns:
        The corresponding score function

    Raises:
        KeyError: If data_source is not recognized
    """
    if data_source not in score_fn_dict:
        raise KeyError(
            f"Unknown data_source: {data_source}. "
            f"Available: {list(score_fn_dict.keys())}"
        )
    return score_fn_dict[data_source]

from .reward import judge_router
from .score import eval_results

__all__ = [
    
    # nemo gym
    "math_score_fn",
    "code_score_fn",
    "mcqa_score_fn",
    "if_score_fn",
    "so_score_fn",
    "wa_score_fn",
    "score_fn_dict",
    "get_score_fn",

    # mika eval
    "judge_router",
    "eval_results",
]