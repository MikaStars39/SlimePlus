from math_verify import parse, verify
from typing import Tuple

def grade_answer(solution_str: str, ground_truth: str) -> Tuple[float, float]:
    try: 
        ground_truth = parse(ground_truth)
        solution = parse(solution_str)
        if verify(ground_truth, solution):
            return 1.0, 1.0
        else:
            return 0.0, 1.0
    except:
        print(f"Error: {e}")
        return 0.0, 0.0

if __name__ == "__main__":
    # Parse the gold and answer
    # If you know that gold will only contain latex or expr (no latex env), use
    # parse(gold, extraction_config=[LatexExtractionConfig()]) or parse(gold, extraction_config=[ExprExtractionConfig()])

    gold = "${1,3} \\cup {2,4}$"
    answer = "${1,2,3,4}$"

    # Order here is important!
    print(grade_answer(answer, gold))