from pathlib import Path

from src.config import parse_eval_args
from src.task_manager import TaskManager
from src.utils import setup_logging


def main() -> None:
    args = parse_eval_args()
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
