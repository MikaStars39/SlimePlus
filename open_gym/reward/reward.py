from typing import Dict

from open_gym.reward.if_eval.if_eval import if_judge
from open_gym.reward.math.math_verify_reward import math_judge
from open_gym.reward.gpqa.gpqa_verify_reward import gpqa_judge

# ----------------------- IMPORTANT: judge router -----------------------
# rule-based judge router that manage the judging process

def judge_router(
    response: str,
    label: str = "",
    source: str = None,
    **kwargs
) -> Dict:

    if "ifeval" in source.lower():
        #
        # ifeval return:
        # return {
        #     'instruction_count': len(instructions),
        #     'instruction_pass_cnt': instruction_pass_cnt,
        #     'pass': prompt_level_pass_flag
        # }
        #
        return if_judge(response, **kwargs)
    elif "gpqa" in source.lower():
        return gpqa_judge(response, label, **kwargs)
    else:
        #
        # math return:
        # return {
        #     "pred": pred_ans,
        #     "pass": True if score == 1.0 else False
        # }
        #
        return math_judge(response, label, **kwargs)

if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported (see eval.py). "
        "Run `python eval.py ...` instead."
    )
