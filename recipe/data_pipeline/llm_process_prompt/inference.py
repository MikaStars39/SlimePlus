import asyncio
import argparse
import warnings
import multiprocessing.resource_tracker

from mika_eval.backend import BatchInferenceEngine

warnings.filterwarnings("ignore", message="resource_tracker: process died unexpectedly")
warnings.filterwarnings("ignore", message="resource_tracker:.*")

_original_unregister = multiprocessing.resource_tracker._resource_tracker.unregister
def _safe_unregister(name, rtype):
    try:
        _original_unregister(name, rtype)
    except KeyError:
        pass
multiprocessing.resource_tracker._resource_tracker.unregister = _safe_unregister

# ------ Logic --------
async def run_batch_inference(args):
    """
    Standard Batch Inference execution.
    """
    engine_args = {
        "model_path": args.model_path,
        "dp_size": args.dp_size,
        "tp_size": args.tp_size,
        "max_inflight": args.max_concurrency,
        "mem_fraction_static": args.gpu_mem,
    }
    
    sampling_params = {
        "temperature": args.temp,
        "top_p": args.top_p,
        "max_new_tokens": args.max_tokens,
    }

    async with BatchInferenceEngine(**engine_args) as engine:
        await engine.run(
            input_file=args.input,
            output_file=args.output,
            sampling_params=sampling_params,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference: Run LLM batch processing.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--max_concurrency", type=int, default=128)
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)

    args = parser.parse_args()
    asyncio.run(run_batch_inference(args))