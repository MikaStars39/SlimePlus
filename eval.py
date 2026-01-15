import argparse
import asyncio
import logging
import os
from pathlib import Path
from src.utils import setup_logging, merge_two_jsonl_file
from src.backend.offline import run_offline_async_inference

def prepare_data(
    args: argparse.Namespace, 
    result_dir: Path
) -> Path:
    """
    Step 1: Prepare dataset jsonl + apply prompt/chat template for inference.

    Returns:
        Path to the formatted jsonl file used as inference input.
    """
    # --------------- 1.1 prepare data ---------------
    from src.tasks import prepare_pass_at_k_jsonl

    data_file = result_dir / "data.jsonl"
    logging.info(f"Preparing data for {args.dataset}...")
    prepare_pass_at_k_jsonl(
        config_str=args.dataset,
        output_file=str(data_file),
        cache_dir=args.cache_dir,
    )

    # --------------- 1.2 apply prompt template ---------------
    from src.utils.template import apply_template_to_jsonl

    formatted_input_file = result_dir / "data.chat.jsonl"
    logging.info(f"Applying prompt/chat template for inference (format={args.prompt_format})...")
    apply_template_to_jsonl(
        input_file=str(data_file),
        output_file=str(formatted_input_file),
        model_path=str(args.model),
        user_template=args.prompt_format,
    )
    return formatted_input_file

def inference(
    args: argparse.Namespace, 
    result_dir: Path, 
) -> Path:
    """
    Step 2: Run offline inference for the main model.

    Returns:
        Path to inference_results.jsonl
    """
    infer_input_file = result_dir / "data.chat.jsonl"
    output_file = result_dir / "inference_results.jsonl"
    if output_file.exists():
        logging.info(f"Inference results exist at {output_file}, skipping.")
        return output_file

    asyncio.run(
        run_offline_async_inference(
            input_file=str(infer_input_file),
            output_file=str(output_file),
            model_path=args.model,
            dp_size=args.dp_size,
            tp_size=args.tp_size,
            mem_fraction_static=args.gpu_memory_utilization,
            sampling_params={
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            },
        )
    )
    return output_file

def llm_evaluation(
    args: argparse.Namespace, 
    result_dir: Path
) -> Path:
    """
    Step 3: LLM extraction (answer extraction) to produce eval_results.jsonl.

    Returns:
        Path to eval_results.jsonl
    """
    # --------------- 3.1 prepare paths ---------------
    from src.llm_judge.extract import prepare_extraction_data

    infer_file = result_dir / "inference_results.jsonl"
    eval_input_file = result_dir / "eval_input.jsonl"
    eval_output_file = result_dir / "eval_results.jsonl"
    no_eval_output_file = result_dir / "no_eval_results.jsonl"

    # --eval-model defaults to --model if not provided
    eval_model_path = args.eval_model or args.model
    logging.info(f"Using model {eval_model_path} for extraction.")

    # --------------- 3.2 if exists jump ---------------
    if eval_output_file.exists():
        logging.info(f"Eval results exist at {eval_output_file}, skipping.")
        return eval_output_file

    # --------------- 3.3 extract the model answer from the json file ---------------
    logging.info("Extracting answers using LLM...")
    prepare_extraction_data(
        input_file=infer_file,
        output_file=eval_input_file,
        output_no_eval_file=no_eval_output_file,
    )

    # --------------- 3.5 prepare infer templates ---------------
    if os.path.getsize(eval_input_file) == 0:
        logging.info(f"Input file {eval_input_file} is empty, skipping LLM extraction inference.")
    else:
        from src.utils.template import apply_template_to_jsonl

        eval_chat_input_file = result_dir / "eval_input.chat.jsonl"  # save here
        logging.info("Applying chat template to eval_input.jsonl for LLM extraction...")
        apply_template_to_jsonl(
            input_file=str(eval_input_file),
            output_file=str(eval_chat_input_file),
            model_path=str(eval_model_path),
            user_template="auto",
        )

        # --------------- 3.6 run inference ---------------
        asyncio.run(
            run_offline_async_inference(
                input_file=str(eval_chat_input_file),
                output_file=str(eval_output_file),
                model_path=eval_model_path,
                dp_size=args.dp_size,
                tp_size=args.tp_size,
                mem_fraction_static=args.gpu_memory_utilization,
                sampling_params={
                    "temperature": args.eval_temperature,
                    "top_p": args.eval_top_p,
                    "max_new_tokens": args.eval_max_new_tokens,
                },
            )
        )
        logging.info(f"Inference completed for {eval_chat_input_file}")

    # --------------- 3.7 merge eval and no eval into one file ---------------
    merge_two_jsonl_file(
        file1_path=eval_output_file,
        file2_path=no_eval_output_file,
        output_path=eval_output_file,
    )
    return eval_output_file

def calculate_metrics(
    args: argparse.Namespace, 
    result_dir: Path, 
) -> Path:
    """
    Step 4: Calculate accuracy/metrics and write final.jsonl

    Returns:
        Path to final.jsonl
    """
    # --------------- 4.1 eval the results and save ---------------
    from src.reward.reward import eval_results

    final_eval_output_file = result_dir / "final.jsonl"
    eval_output_file = result_dir / "eval_results.jsonl"
    eval_results(
        eval_output_file=eval_output_file,
        final_eval_output_file=final_eval_output_file,
    )
    return final_eval_output_file

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

    # ------------------------------ 1. Prepare ------------------------------
    if args.mode in ["all", "prepare"]:
        infer_input_file = prepare_data(args=args, result_dir=result_dir)
    
    # ------------------------------ 2. Inference ------------------------------
    if args.mode in ["all", "infer"]:
        infer_output_file = inference(args=args, result_dir=result_dir)

    # ------------------------------ 3. LLM Eval ------------------------------
    if args.mode in ["all", "llm-eval"]:
        eval_output_file = llm_evaluation(args=args, result_dir=result_dir)

    # ------------------------------ 4. Calculate Metrics ------------------------------
    if args.mode in ["all", "metrics"]:
        calculate_metrics(args=args, result_dir=result_dir)

if __name__ == "__main__":
    main()
