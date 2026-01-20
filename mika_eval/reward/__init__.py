from .reward import judge_router
from .score import eval_results

# judge_router is a router that can judge single input 
#
# eval_results read an eval_result file and judge + computing metrics
#

__all__ = ["judge_router", "eval_results"]