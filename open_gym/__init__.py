"""
MikaEval: A flexible evaluation framework for LLMs.
"""

__version__ = "0.1.0"

# Backend engines
from .backend import BatchInferenceEngine, OnlineServingEngine

# Reward functions
from .reward import judge_router, eval_results

__all__ = [
    # Version
    "__version__",
    # Backend
    "BatchInferenceEngine",
    "OnlineServingEngine",
    # Reward
    "judge_router",
    "eval_results",
]
