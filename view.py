import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import statistics

def parse_args():
    parser = argparse.ArgumentParser(description="View evaluation results and generate a summary table.")
    parser.add_argument("result_dir", type=str, help="The directory containing evaluation results (e.g., outputs/debug)")
    return parser.parse_args()

def get_dataset_stats(dataset_dir: Path) -> Dict[str, Any]:
    result_file = dataset_dir / "result.jsonl"
    if not result_file.exists():
        return None

    scores = []
    format_scores = []
    
    with result_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "avg" in data:
                    scores.append(data["avg"])
                if "format_score_avg" in data:
                    format_scores.append(data["format_score_avg"])
            except json.JSONDecodeError:
                continue

    if not scores:
        return None

    return {
        "dataset": dataset_dir.name,
        "count": len(scores),
        "accuracy": statistics.mean(scores),
        "format_accuracy": statistics.mean(format_scores) if format_scores else 0.0,
    }

def main():
    args = parse_args()
    result_path = Path(args.result_dir)
    
    if not result_path.exists():
        print(f"Error: Directory {args.result_dir} does not exist.")
        return

    stats_list = []
    
    # Iterate through subdirectories
    for item in result_path.iterdir():
        if item.is_dir() and (item / "result.jsonl").exists():
            stats = get_dataset_stats(item)
            if stats:
                stats_list.append(stats)
    
    if not stats_list:
        print("No evaluation results found.")
        return

    # Sort by dataset name
    stats_list.sort(key=lambda x: x["dataset"])

    # Generate Markdown Table
    print(f"\n# Evaluation Summary: {result_path.name}\n")
    print("| Dataset | Count | Accuracy (Pass@1) | Format Accuracy |")
    print("| :--- | :--- | :--- | :--- |")
    
    total_count = 0
    total_accuracy_sum = 0
    total_format_sum = 0

    for stats in stats_list:
        acc_pct = f"{stats['accuracy'] * 100:.2f}%"
        fmt_pct = f"{stats['format_accuracy'] * 100:.2f}%"
        print(f"| {stats['dataset']} | {stats['count']} | {acc_pct} | {fmt_pct} |")
        
        total_count += stats['count']
        total_accuracy_sum += stats['accuracy'] * stats['count']
        total_format_sum += stats['format_accuracy'] * stats['count']

    if len(stats_list) > 1:
        avg_acc = f"{(total_accuracy_sum / total_count) * 100:.2f}%"
        avg_fmt = f"{(total_format_sum / total_count) * 100:.2f}%"
        print(f"| **Average** | **{total_count}** | **{avg_acc}** | **{avg_fmt}** |")
    
    print("\n")

if __name__ == "__main__":
    main()

