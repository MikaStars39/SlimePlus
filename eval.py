import argparse
from pathlib import Path

from src.task_manager import TaskManager
from src.utils import setup_logging

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MikaEval: Offline Inference and Evaluation")
    
    # Execution Mode
    parser.add_argument("--mode", choices=["prepare", "infer", "llm-eval", "metrics", "all"], default="infer")
    
    # Paths
    parser.add_argument("--result-dir", required=True, help="Directory for output results.")
    parser.add_argument("--model", required=True, help="Base model path for inference.")
    parser.add_argument("--eval-model", default=None, help="Model path for LLM-based answer extraction (Step 3). Defaults to --model if not set.")
    parser.add_argument("--dataset", default="aime2024", help="Dataset name/abbreviation.")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for datasets.")
    
    # SGLang Engine Config
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference.")
    
    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)

    # LLM Extraction (Step 3) Sampling Params
    # Extraction should be deterministic + short to avoid repetitive / rambling outputs.
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    
    # Template
    parser.add_argument("--prompt-format", default="slime", help="Prompt template to use.")
    parser.add_argument("--max-concurrency", type=int, default=2000, help="Max concurrency for inference.")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    setup_logging(result_dir)
    task_manager = TaskManager(args=args, result_dir=result_dir)

    # ------------------------------ 1. Prepare ------------------------------
    if args.mode in ["all", "prepare"]:
        task_manager.prepare_data()
    
    # ------------------------------ 2. Inference ------------------------------
    if args.mode in ["all", "infer"]:
        task_manager.inference()

    # ------------------------------ 3. LLM Eval ------------------------------
    if args.mode in ["all", "llm-eval"]:
        task_manager.llm_evaluation()

    # ------------------------------ 4. Calculate Metrics ------------------------------
    if args.mode in ["all", "metrics"]:
        task_manager.calculate_metrics()

if __name__ == "__main__":
    main()
