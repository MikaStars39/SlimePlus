"""
Batch inference using SGLang engine.

Usage:
    python inference.py --input data.chat.jsonl --output results.jsonl \
        --model /path/to/model --tp-size 8 --temperature 0.6
"""

import argparse
import asyncio
import logging
import warnings
import multiprocessing.resource_tracker

from open_gym.backend import BatchInferenceEngine

# Suppress resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker:.*")
_original_unregister = multiprocessing.resource_tracker._resource_tracker.unregister
def _safe_unregister(name, rtype):
    try:
        _original_unregister(name, rtype)
    except KeyError:
        pass
multiprocessing.resource_tracker._resource_tracker.unregister = _safe_unregister


async def run_inference(args):
    """Run batch inference pipeline."""
    engine_args = {
        "model_path": args.model,
        "dp_size": args.dp_size,
        "tp_size": args.tp_size,
        "max_inflight": args.max_concurrency,
        "mem_fraction_static": args.gpu_mem,
    }
    
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_tokens,
    }

    logging.info(f"Starting inference with model: {args.model}")
    logging.info(f"Engine config: tp={args.tp_size}, dp={args.dp_size}, concurrency={args.max_concurrency}")
    logging.info(f"Sampling: temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    async with BatchInferenceEngine(**engine_args) as engine:
        await engine.run(
            input_file=args.input,
            output_file=args.output,
            sampling_params=sampling_params,
            resume=args.resume,
        )


def main():
    parser = argparse.ArgumentParser(description="Batch inference with SGLang.")
    
    # I/O
    parser.add_argument("--input", type=str, required=True, 
                        help="Input JSONL file with 'prompt' field")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output JSONL file with 'response' field")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from existing output file")
    
    # Model
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the model")
    parser.add_argument("--tp-size", type=int, default=1, 
                        help="Tensor parallel size")
    parser.add_argument("--dp-size", type=int, default=1, 
                        help="Data parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.9, 
                        help="GPU memory utilization")
    parser.add_argument("--max-concurrency", type=int, default=1024, 
                        help="Max concurrent requests")
    
    # Sampling
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, 
                        help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=4096, 
                        help="Max new tokens to generate")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
