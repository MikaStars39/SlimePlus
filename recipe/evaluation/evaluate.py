"""
Evaluate model responses and calculate accuracy metrics.

Usage:
    python evaluate.py --input results.jsonl --output-dir /path/to/output
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from open_gym.reward.reward import judge_router


def judge_instance(instance: Dict) -> Dict:
    """Judge a single instance using the appropriate judge (math, ifeval, etc.)."""
    result = judge_router(
        response=instance.get("response", ""),
        label=instance.get("label", ""),
        source=instance.get("source", ""),
        **{k: v for k, v in instance.items() if k not in ["response", "label", "source"]}
    )
    instance.update(result)
    return instance


def calculate_metrics(items: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Calculate avg@k and pass@k metrics per dataset and overall."""
    # Group scores by dataset and question_id
    grouped = {}
    for item in items:
        ds_name = item.get("source", "unknown")
        q_id = item.get("question_id", "unknown")
        score = 1.0 if item.get("pass", False) else 0.0
        grouped.setdefault(ds_name, {}).setdefault(q_id, []).append(score)

    # Calculate metrics per dataset
    results = {}
    for ds_name, questions in grouped.items():
        all_scores = []
        pass_at_k = []
        
        for q_id, scores in questions.items():
            all_scores.extend(scores)
            pass_at_k.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

        results[ds_name] = {
            "avg_k": sum(all_scores) / len(all_scores) if all_scores else 0,
            "pass_k": sum(pass_at_k) / len(pass_at_k) if pass_at_k else 0,
            "n_questions": len(questions),
            "n_samples": len(all_scores),
        }

    # Overall metrics
    all_scores = []
    all_pass_k = []
    for ds_name, questions in grouped.items():
        for q_id, scores in questions.items():
            all_scores.extend(scores)
            all_pass_k.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

    results["overall"] = {
        "avg_k": sum(all_scores) / len(all_scores) if all_scores else 0,
        "pass_k": sum(all_pass_k) / len(all_pass_k) if all_pass_k else 0,
        "n_questions": len(all_pass_k),
        "n_samples": len(all_scores),
    }

    return results


def print_metrics(metrics: Dict[str, Dict[str, float]]):
    """Pretty print metrics table."""
    print("\n" + "=" * 70)
    print(f"{'Dataset':<25} | {'Avg@K':>10} | {'Pass@K':>10} | {'Questions':>10}")
    print("-" * 70)

    # Print overall first
    if "overall" in metrics:
        m = metrics["overall"]
        print(f"{'OVERALL':<25} | {m['avg_k']:>10.2%} | {m['pass_k']:>10.2%} | {m['n_questions']:>10}")
        print("-" * 70)

    # Print per-dataset
    for ds_name, m in sorted(metrics.items()):
        if ds_name == "overall":
            continue
        print(f"{ds_name:<25} | {m['avg_k']:>10.2%} | {m['pass_k']:>10.2%} | {m['n_questions']:>10}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate responses and calculate metrics.")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input JSONL with 'response' and 'label' fields")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Output directory for results")
    parser.add_argument("--num-proc", type=int, default=32, 
                        help="Number of parallel processes for judging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scores_file = output_dir / "scores.jsonl"
    metrics_file = output_dir / "metrics.json"

    # Load and judge
    logging.info(f"Loading data from {args.input}...")
    dataset = load_dataset("json", data_files=args.input, split="train")
    logging.info(f"Loaded {len(dataset)} samples, running judges...")

    judged = dataset.map(
        judge_instance,
        num_proc=args.num_proc,
        load_from_cache_file=False,
        desc="Judging",
    )

    # Mark pass@k
    items = list(judged)
    passing = {
        (item.get("source"), item.get("question_id"))
        for item in items if item.get("pass", False)
    }
    for item in items:
        item["pass_at_k"] = (item.get("source"), item.get("question_id")) in passing

    # Save scored results
    with open(scores_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"Saved scored results to {scores_file}")

    # Calculate and save metrics
    metrics = calculate_metrics(items)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved metrics to {metrics_file}")

    # Print results
    print_metrics(metrics)


if __name__ == "__main__":
    main()
