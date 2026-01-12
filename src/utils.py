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

def display_metrics_report(metrics_data: Dict[str, Dict[str, float]]):
    if not metrics_data:
        print("No metrics data to display.")
        return

    print("\n" + "="*60)
    print(f"{'Dataset':<25} | {'Avg@K':<12} | {'Pass@K':<12}")
    print("-" * 60)

    # 优先打印 overall（如果存在）
    if "overall" in metrics_data:
        stats = metrics_data["overall"]
        avg_k = stats["avg_k"]
        pass_k = stats["pass_k"]
        print(f"{'overall':<25} | {avg_k:>11.2%} | {pass_k:>11.2%}")
        print("-" * 60)

    for ds_name, stats in metrics_data.items():
        if ds_name == "overall":
            continue
        avg_k = stats["avg_k"]
        pass_k = stats["pass_k"]
        print(f"{ds_name:<25} | {avg_k:>11.2%} | {pass_k:>11.2%}")
        
    print("="*60 + "\n")

def calculate_and_print_metrics(eval_output_file: Path, cache_dir: str = None):
    from src.reward.reward import extract_metrics_from_file
    results = extract_metrics_from_file(eval_output_file)
    display_metrics_report(results)
