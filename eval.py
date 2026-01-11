import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from pprint import pprint

# 必须在所有自定义导入之前，甚至在 logging 之前
sys.setrecursionlimit(100000)

from src.backend.offline import run_offline_async_inference

def log_info(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | INFO | {msg}")

def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluation entry script, supports model merging, vLLM startup, and multi-dataset evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "infer", "llm-eval", "rule-eval"],
        default="all",
        help="Execution mode: 'all' (infer+llm-eval), 'infer' (inference only), 'llm-eval' (LLM extraction + evaluation), 'rule-eval' (Rule extraction + evaluation).",
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory for intermediate processes and result output.",
    )
    parser.add_argument("--model", required=True, help="Base model name or path.")
    parser.add_argument(
        "--adapter",
        default="",
        help="LoRA/PEFT adapter path, leave empty for no merge.",
    )
    parser.add_argument(
        "--dataset",
        default="aime2024",
        help="Dataset abbreviation to evaluate, comma separated (e.g., aime2024).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory containing benchmark datasets. If specified, will look for datasets in subfolders named after dataset abbreviations (e.g., cache_dir/aime_2024/).",
    )
    parser.add_argument(
        "--prompt-format",
        default="lighteval",
        help="Prompt format template to use.",
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=1,
        help="Number of rollouts to generate per sample.",
    )
    parser.add_argument(
        "--serve-port", type=int, default=8000, help="First vLLM backend port number."
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Number of data parallel backends (start multiple vLLMs).",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size passed to vLLM."
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Verify needed GPU count before running, error if insufficient.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization limit passed to vLLM (0~1), controls memory usage per card.",
    )
    parser.add_argument(
        "--seed", type=float, default=None, help="Generation random seed."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Generation temperature."
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=131072, help="Generation length."
    )
    parser.add_argument(
        "--dtype", default="auto", help="Model dtype, used during merging."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Whether to trust remote code."
    )
    parser.add_argument(
        "--served-model-name", default="eval-model", help="Model name exposed by vLLM."
    )
    parser.add_argument(
        "--api-key", default="dummy", help="API Key for OpenAI compatible interface."
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=3600.0,
        help="Timeout for a single request.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="For debugging, limit number of evaluation samples.",
    )
    parser.add_argument(
        "--max-num-request",
        type=int,
        default=None,
        help="Max number of concurrent requests per data parallel (DP) vLLM backend.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for offline inference.",
    )

    args, unknown = parser.parse_known_args()

    return args

def main() -> None:
    # ------------------- 0. parsing args and preparing logger --------------------

    args = parse_args()
    result_dir = Path(args.result_dir)
    need_pass1 = args.mode in ["all", "infer"]
    need_pass2 = args.mode in ["all", "llm-eval"]

    # ------------------- 1. prepare data --------------------
    
    if "infer" in args.mode:
        from src.data.data import prepare_pass_at_k_jsonl
        data_file = result_dir / "data.jsonl" 
        prepare_pass_at_k_jsonl(
            config_str=args.dataset,
            output_file=data_file,
            cache_dir=args.cache_dir,
        )

    # ------------------- 2. offline inference --------------------

    if "infer" in args.mode:
        output_file = result_dir / "inference_results.jsonl"

        # check if output_file exists
        if Path(output_file).exists():
            log_info(f"Output file {output_file} already exists, skipping offline inference")
        else:
            asyncio.run(run_offline_async_inference(
                input_file=data_file,
                output_file=output_file, 
                model_path=args.model, 
                dp_size=args.dp_size,
                tp_size=args.tp_size,
                batch_size=args.batch_size,
                mem_fraction_static=args.gpu_memory_utilization,
                sampling_params={
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                },
            ))
    
    if "llm-eval" in args.mode:
        from src.data.extract import prepare_extraction_data

        infer_file = result_dir / "inference_results.jsonl"
        eval_input_file = result_dir / "eval_input.jsonl"
        eval_output_file = result_dir / "eval_results.jsonl"

        if Path(eval_output_file).exists():
            log_info(f"Eval output {eval_output_file} exists, skipping extraction.")
        else:
            # Prepare extraction prompts from inference results
            log_info(f"Preparing extraction prompts from {infer_file}...")
            prepare_extraction_data(infer_file, eval_input_file)

            # Run LLM to extract answers
            asyncio.run(run_offline_async_inference(
                input_file=eval_input_file,
                output_file=eval_output_file,
                model_path=args.model,
                dp_size=args.dp_size,
                tp_size=args.tp_size,
                batch_size=args.batch_size,
                mem_fraction_static=args.gpu_memory_utilization,
                sampling_params={
                    "temperature": 0.0,  # Greedy for extraction
                    "max_new_tokens": 512,
                },
            ))
    
    # ------------------- 3. evaluation --------------------
        


if __name__ == "__main__":
    main()