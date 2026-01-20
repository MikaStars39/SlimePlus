import json
import logging
from typing import Dict, List
from pathlib import Path

from datasets import load_dataset
from src.reward.reward import judge_router

def instance_judge(
    instance: Dict
) -> Dict:
    # ------------------ get the instance information ------------------
    response = instance.get("response", "")
    label = instance.get("label", "")
    source = instance.get("source", None)
    
    other_kwargs = {
        k: v for k, v in instance.items() if k not in ["response", "label", "source"]
    }
    
    # ------------------ judge the instance ------------------
    judge_res = judge_router(
        response=response,
        label=label,
        source=source,
        **other_kwargs
    )
    
    # ------------------ update the instance ------------------
    instance.update(judge_res)
    return instance


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

def eval_results(
    eval_output_file: Path,
    score_output_file: Path,
    final_eval_output_file: Path,
    n_proc: int = 32
) -> Dict[str, Dict[str, float]]:
    
    logging.info(f"Scoring eval results from {eval_output_file} (num_proc={n_proc})...")
    
    # ------------------ load the results ------------------
    results = load_dataset("json", data_files=str(eval_output_file), split="train")
    logging.info(f"Loaded {len(results)} records; running judge_router...")
    
    results = results.map(
        instance_judge,
        num_proc=n_proc,
        load_from_cache_file=False,
        desc="instance_judge",
    )
    logging.info("Judging complete; computing metrics...")

    items = list(results)
    passing_keys = {
        (item.get("source", "unknown"), item.get("question_id", "unknown"))
        for item in items
        if item.get("pass", False)
    }
    for item in items:
        key = (item.get("source", "unknown"), item.get("question_id", "unknown"))
        item["pass_at_k"] = key in passing_keys

    # ------------------ save the results to a jsonl file ------------------
    with open(score_output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"Saved score results to {score_output_file}")

    # ------------------ calculate the metrics and return ------------------ 
    metrics = _calculate_matrics(items)
    with open(final_eval_output_file, "w", encoding="utf-8") as f:
        for ds_name, ds_metrics in metrics.items():
            f.write(json.dumps([ds_name, ds_metrics], ensure_ascii=False) + "\n")
    logging.info(f"Saved final metrics to {final_eval_output_file}")
    
    return metrics