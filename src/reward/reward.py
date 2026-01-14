import re
import json
from typing import Dict, List
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset

from src.reward.if_eval.if_eval import if_judge
from src.reward.math.math_verify_reward import math_judge

def _calculate_matrics(
    updated_items: List[Dict]
) -> Dict[str, Dict[str, float]]:
    # Calculate final metrics
    raw_data = {}
    for item in updated_items:
        ds_name = item.get("source", "unknown")
        q_id = item.get("question_id", "unknown")
        score = 1.0 if item.get("pass", False) else 0.0
        raw_data.setdefault(ds_name, {}).setdefault(q_id, []).append(score)

    final_results = {}
    for ds_name, q_map in raw_data.items():
        all_scores = []
        pass_at_k_scores = []

        for q_id, scores in q_map.items():
            all_scores.extend(scores)
            pass_at_k_scores.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

        final_results[ds_name] = {
            "avg_k": sum(all_scores) / len(all_scores) if all_scores else 0,
            "pass_k": sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0
        }

    # Overall metrics across all datasets
    overall_all_scores = []
    overall_pass_at_k_scores = []
    for ds_name, q_map in raw_data.items():
        for q_id, scores in q_map.items():
            overall_all_scores.extend(scores)
            overall_pass_at_k_scores.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

    final_results["overall"] = {
        "avg_k": sum(overall_all_scores) / len(overall_all_scores) if overall_all_scores else 0,
        "pass_k": sum(overall_pass_at_k_scores) / len(overall_pass_at_k_scores) if overall_pass_at_k_scores else 0,
    }

    return final_results

def judge_router(
    instance: Dict
) -> Dict:
    if instance.get("eval_type", "ifeval") == "ifeval":
        return if_judge(instance)
    else:
        return math_judge(instance)

def eval_results(
    eval_output_file: Path,
    final_eval_output_file: Path,
    n_proc: int = 32
) -> Dict[str, Dict[str, float]]:
    
    results = load_dataset("json", data_files=eval_output_file)
    results = results.map(judge_router, num_proc=n_proc)

    # ------------------ calculate the metrics and return ------------------ 
    results = _calculate_matrics(list(results))
    with open(final_eval_output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return metrics

if __name__ == "__main__":
    test_res = "The answer is \\boxed{m/frac{2}{3}}"
    print(f"Extracted: {extract_answer(test_res)}")
    print(f"Reward: {get_reward(extract_answer(test_res), 'm/frac{{2}}{{3}}', 'dapo')}")
