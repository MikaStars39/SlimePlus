import json
import asyncio
import sglang as sgl
from tqdm.asyncio import tqdm
import sys

sys.setrecursionlimit(100000)

async def run_offline_async_inference(
    input_file: str, 
    output_file: str, 
    model_path: str, 
    dp_size: int,
    tp_size: int,
    batch_size: int,
    mem_fraction_static: float,
    sampling_params: dict,
):
    """
    Perform offline inference with Data Parallelism and streaming file I/O.
    
    Args:
        input_file: Path to the input .jsonl file.
        output_file: Path to the output .jsonl file.
        model_path: Path to the model weights.
        batch_size: Number of prompts to send to the engine at once.
    """
    
    # 1. Initialize SGLang Engine with Data Parallelism
    # dp_size=8 will replicate the model across 8 GPUs for higher throughput.
    # tp_size=1 assumes the model fits on a single GPU. 
    # If the model is huge (e.g., 70B), use tp_size=2, dp_size=4 for 8 GPUs total.
    llm = sgl.Engine(
        model_path=model_path,
        dp_size=dp_size, 
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static, # Reserved memory for KV cache
        # disable_cuda_graph=True, # Disable CUDA Graph to avoid RecursionError
        log_level="error"
    )

    # 2. Stream Processing: Read and write line by line to handle large files
    # We use standard 'with open' as it's simpler and sufficient for offline tasks
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        
        batch_data = []
        print(f"Starting inference using {dp_size} GPUs (DP)...")
        
        for line in f_in:
            if not line.strip():
                continue
            
            batch_data.append(json.loads(line))
            
            # 3. When batch is full, trigger asynchronous inference
            if len(batch_data) >= batch_size:
                prompts = [item["prompt"] for item in batch_data]
                
                # SGLang distributes these prompts across all 8 DP workers automatically
                outputs = await llm.async_generate(prompts, sampling_params)
                
                # 4. Write results immediately to disk
                for item, output in zip(batch_data, outputs):
                    item["response"] = output["text"]
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                f_out.flush() # Ensure data is written even if script crashes
                batch_data = [] # Clear the batch for next set of lines

        # 5. Handle the remaining items in the last batch
        if batch_data:
            prompts = [item["prompt"] for item in batch_data]
            outputs = await llm.async_generate(prompts, sampling_params)
            for item, output in zip(batch_data, outputs):
                item["response"] = output["text"]
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Clean up resources
    llm.shutdown()
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    model_path = "/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/20260110_173129_gspo_qwen30ba3b/iter_0000223_hf"
    input_file = "examples/debug.jsonl"
    output_file = "examples/debug_output.jsonl"
    batch_size = 256
    dp_size = 1
    tp_size = 1
    mem_fraction_static = 0.8
    sampling_params = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 32000,
    }

    # Run async main function with asyncio.run
    asyncio.run(
        run_offline_async_inference(
            input_file=input_file, 
            output_file=output_file, 
            model_path=model_path, 
            dp_size=dp_size,
            tp_size=tp_size,
            batch_size=batch_size,
            mem_fraction_static=mem_fraction_static,
            sampling_params=sampling_params,
        )
    )
