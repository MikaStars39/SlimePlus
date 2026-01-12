import json
import asyncio
import sglang as sgl
from tqdm.asyncio import tqdm

async def run_offline_async_inference(
    input_file: str, 
    output_file: str, 
    model_path: str, 
    chunk_size: int = 512,
    dp_size: int = 1,
    tp_size: int = 1,
    mem_fraction_static: float = 0.90,
    sampling_params: dict = None
):
    if sampling_params is None:
        sampling_params = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 2048}

    print(f"Initializing Engine with model: {model_path}")
    
    # 1. Initialize Engine
    # SGLang Engine handles continuous batching internally.
    llm = sgl.Engine(
        model_path=model_path,
        dp_size=dp_size, 
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        log_level="error",
        disable_radix_cache=True, # Set based on your specific needs (e.g., usually True for eval/benchmarks)
        trust_remote_code=True
    )

    # 2. Wrapper for single item generation
    # This binds the result back to the original item dictionary.
    async def generate_wrapper(item):
        output = await llm.async_generate(item["prompt"], sampling_params)
        item["response"] = output["text"]
        return item

    # 3. Batch Processor
    async def process_chunk(batch_data, f_out, pbar):
        # Create a list of tasks for the whole chunk
        tasks = [generate_wrapper(item) for item in batch_data]
        
        # 'as_completed' yields tasks as soon as they finish, regardless of order.
        # This allows immediate writing to disk.
        for task in asyncio.as_completed(tasks):
            try:
                result_item = await task
                
                # Write immediately
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush() # Ensure data is written to disk immediately
                
                # Update progress bar
                pbar.update(1)
            except Exception as e:
                print(f"Error processing item: {e}")

    # 4. Main Logic
    print("Counting total lines...")
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8') if _.strip())
    print(f"Starting Inference on {total_lines} items...")

    current_batch = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        pbar = tqdm(total=total_lines, desc="Inference")
        
        for line in f_in:
            if not line.strip(): continue
            
            try:
                data = json.loads(line)
                current_batch.append(data)
                
                # Process when chunk is full
                if len(current_batch) >= chunk_size:
                    await process_chunk(current_batch, f_out, pbar)
                    current_batch = [] # Reset buffer
                    
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line[:50]}...")

        # Process remaining items
        if current_batch:
            await process_chunk(current_batch, f_out, pbar)
            
        pbar.close()

    llm.shutdown()
    print(f"\nDone! Results saved to {output_file}")

# Execution Entry Point
if __name__ == "__main__":
    # Define parameters here
    params = {
        "temperature": 0.6, 
        "top_p": 0.9, 
        "max_new_tokens": 30000 
    }

    asyncio.run(run_offline_async_inference(
        input_file="/mnt/llm-train/users/explore-train/qingyu/MikaEval/outputs/20260110_173129_gspo_qwen30ba3b_0000223_slime_new/data.jsonl",
        output_file="/mnt/llm-train/users/explore-train/qingyu/MikaEval/outputs/20260110_173129_gspo_qwen30ba3b_0000223_slime_new/inference_results.jsonl",
        model_path="/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/20260110_173129_gspo_qwen30ba3b/iter_0000223_hf",
        chunk_size=512,
        dp_size=8,
        tp_size=1,
        mem_fraction_static=0.9,
        sampling_params=params
    ))