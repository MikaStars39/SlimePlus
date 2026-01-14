import argparse
import asyncio
import sys
import logging
import json
from pathlib import Path
from src.utils import setup_logging
from src.backend.offline import run_offline_async_inference

# Required for deep recursion in some datasets
sys.setrecursionlimit(100000)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MikaEval: Offline Inference and Evaluation")
    
    # Execution Mode
    parser.add_argument("--mode", choices=["infer", "llm-eval", "all"], default="infer")
    
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

    # ------------------------------ 1. Prepare Data ------------------------------ 
    if args.mode in ["all", "infer"]:
        from src.data.data import prepare_pass_at_k_jsonl
        data_file = result_dir / "data.jsonl" 
        logging.info(f"Preparing data for {args.dataset}...")
        prepare_pass_at_k_jsonl(
            config_str=args.dataset,
            output_file=data_file,
            cache_dir=args.cache_dir,
        )
        # Apply prompt template + (if supported) model chat template before inference.
        # `prepare_pass_at_k_jsonl` writes raw questions into `prompt`; many instruct/chat models
        # behave much better if we wrap the prompt using the official tokenizer chat template.
        from src.data.template import apply_template_to_jsonl, PROMPT_TEMPLATES
        infer_input_file = data_file
        formatted_input_file = result_dir / "data.chat.jsonl"
        logging.info(f"Applying prompt/chat template for inference (format={args.prompt_format})...")
        apply_template_to_jsonl(
            input_file=str(data_file),
            output_file=str(formatted_input_file),
            model_path=str(args.model),
            user_template=PROMPT_TEMPLATES.get(args.prompt_format)
        )
        infer_input_file = formatted_input_file

    # ------------------------------ 2. Inference ------------------------------ 
    if args.mode in ["all", "infer"]:
        output_file = result_dir / "inference_results.jsonl"
        if output_file.exists():
            logging.info(f"Inference results exist at {output_file}, skipping.")
        else:
            asyncio.run(run_offline_async_inference(
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
            ))
    
    # ------------------------------ 3. LLM Extraction ------------------------------ 
    if args.mode in ["all", "llm-eval"]:
        from src.data.extract import prepare_extraction_data
        infer_file = result_dir / "inference_results.jsonl"
        eval_input_file = result_dir / "eval_input.jsonl"
        eval_output_file = result_dir / "eval_results.jsonl"

        if eval_output_file.exists():
            logging.info(f"Eval results exist at {eval_output_file}, skipping.")
        else:
            logging.info(f"Extracting answers using LLM...")
            prepare_extraction_data(infer_file, eval_input_file)
            
            # Use separate eval model if specified, else fallback to base model
            eval_model_path = args.eval_model if args.eval_model else args.model
            logging.info(f"Using model {eval_model_path} for extraction.")

            eval_infer_input_file = eval_input_file
            from src.data.template import apply_template_to_jsonl
            eval_chat_input_file = result_dir / "eval_input.chat.jsonl"
            logging.info("Applying chat template to eval_input.jsonl for LLM extraction...")
            apply_template_to_jsonl(
                input_file=str(eval_input_file),
                output_file=str(eval_chat_input_file),
                model_path=str(eval_model_path),
                user_template="{problem}"
            )
            eval_infer_input_file = eval_chat_input_file

            asyncio.run(run_offline_async_inference(
                input_file=str(eval_infer_input_file),
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
            ))

    # ------------------------------ 4. Calculate Accuracy ------------------------------ 
    if args.mode in ["all", "llm-eval"]:
        from src.reward.reward import extract_metrics_from_file
        results = extract_metrics_from_file(eval_output_file)
        
        from src.utils import calculate_and_print_metrics
        calculate_and_print_metrics(eval_output_file, cache_dir=args.cache_dir)

if __name__ == "__main__":
    main()
