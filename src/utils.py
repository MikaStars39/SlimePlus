import logging
import sys
import time
import json
from typing import Any, Dict, List
from pathlib import Path

def setup_logging(result_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "eval.log"
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("eval")

def calculate_and_print_metrics(eval_output_file: Path, cache_dir: str = None):
    """Calculate and print Avg@K and Pass@K metrics."""
    from src.reward.reward import get_reward, extract_answer
    
    if not eval_output_file.exists():
        logging.error(f"Eval results not found: {eval_output_file}")
        return

    logging.info(f"Calculating metrics from {eval_output_file}...")
    
    # dataset_metrics = { dataset_name: { question_id: [is_correct_float, ...] } }
    dataset_metrics = {}
    
    with open(eval_output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            ds_name = item.get("source", "unknown")
            q_id = item.get("question_id", "unknown")
            label = item.get("label", "")
            
            # Extract clean answer from LLM extraction response
            raw_eval_res = item.get("response", "")
            pred_ans = extract_answer(raw_eval_res)
            
            # Select reward type
            reward_type = "f1" if ds_name == "gpqa_diamond" else "dapo"
            
            # Compute score
            score = float(get_reward(pred_ans, label, reward_type))
            
            if ds_name not in dataset_metrics:
                dataset_metrics[ds_name] = {}
            if q_id not in dataset_metrics[ds_name]:
                dataset_metrics[ds_name][q_id] = []
            dataset_metrics[ds_name][q_id].append(score)

    # Print Report
    print("\n" + "="*60)
    print(f"{'Dataset':<25} | {'Avg@K':<12} | {'Pass@K':<12}")
    print("-" * 60)
    
    for ds_name, q_map in dataset_metrics.items():
        all_scores = []
        pass_at_k_scores = []
        for q_id, scores in q_map.items():
            all_scores.extend(scores)
            # Pass@k is 1 if any sample is fully correct (score == 1.0)
            # This ensures Avg@1 == Pass@1 when K=1
            pass_at_k_scores.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)
        
        avg_k = sum(all_scores) / len(all_scores) if all_scores else 0
        pass_k = sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0
        print(f"{ds_name:<25} | {avg_k:>11.2%} | {pass_k:>11.2%}")
    print("="*60 + "\n")
