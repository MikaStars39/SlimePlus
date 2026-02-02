# IFBench evaluation module
from .if_bench import ifbench_judge, ifbench_judge_loose, calculate_scores
from .instructions_registry import INSTRUCTION_DICT

__all__ = ['ifbench_judge', 'ifbench_judge_loose', 'calculate_scores', 'INSTRUCTION_DICT']
