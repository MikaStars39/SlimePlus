from math_verify import parse, verify
from typing import Tuple
from src.utils import _extract_answer


def grade_answer(solution_str: str, ground_truth: str) -> Tuple[float, float]:
    try: 
        ground_truth = parse(ground_truth)
        solution = parse(solution_str)
        if verify(ground_truth, solution):
            return 1.0, 1.0
        else:
            return 0.0, 1.0
    except Exception as e:
        print(f"Error: {e}")
        return 0.0, 0.0

def math_judge(
    instance: dict
) -> dict:
    label = instance.get("label", "")
    raw_eval_res = instance.get("response", "") 

    pred_ans = _extract_answer(raw_eval_res)
    if pred_ans is None:
        instance["pred"] = pred_ans
        instance["pass"] = False

    score = grade_answer(f"${pred_ans}$", f"${label}$")

    instance["pred"] = pred_ans
    instance["pass"] = True if score == 1.0 else False

    return instance


if __name__ == "__main__":
    # Parse the gold and answer
    # If you know that gold will only contain latex or expr (no latex env), use
    # parse(gold, extraction_config=[LatexExtractionConfig()]) or parse(gold, extraction_config=[ExprExtractionConfig()])

    gold = "${1,3} \\cup {2,4}$"
    answer = "${1,2,3,4}$"

    # Order here is important!
    print(grade_answer(answer, gold))