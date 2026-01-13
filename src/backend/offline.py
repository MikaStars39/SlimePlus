import json
import asyncio
import sglang as sgl
from tqdm.asyncio import tqdm
import time
from typing import Optional, Tuple, Any, Dict

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

    def _extract_token_counts(output: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Best-effort extraction of token usage from SGLang outputs.
        Works across potential schema variants (meta_info / usage).
        Returns (prompt_tokens, completion_tokens, total_tokens) with None if unavailable.
        """
        if not isinstance(output, dict):
            return None, None, None

        meta = output.get("meta_info")
        if isinstance(meta, dict):
            pt = meta.get("prompt_tokens", meta.get("input_tokens"))
            ct = meta.get("completion_tokens", meta.get("output_tokens"))
            tt = meta.get("total_tokens")
            return (
                int(pt) if isinstance(pt, (int, float)) else None,
                int(ct) if isinstance(ct, (int, float)) else None,
                int(tt) if isinstance(tt, (int, float)) else None,
            )

        usage = output.get("usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens", usage.get("input_tokens"))
            ct = usage.get("completion_tokens", usage.get("output_tokens"))
            tt = usage.get("total_tokens")
            return (
                int(pt) if isinstance(pt, (int, float)) else None,
                int(ct) if isinstance(ct, (int, float)) else None,
                int(tt) if isinstance(tt, (int, float)) else None,
            )

        return None, None, None
    
    # 1. Initialize Engine
    # SGLang Engine handles continuous batching internally.
    llm = sgl.Engine(
        model_path=model_path,
        dp_size=dp_size, 
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        log_level="info",
        disable_radix_cache=True, # Set based on your specific needs (e.g., usually True for eval/benchmarks)
        trust_remote_code=True
    )

    # Throughput accounting
    start_t = time.perf_counter()
    acc_prompt_tokens = 0
    acc_completion_tokens = 0
    acc_total_tokens = 0
    saw_any_token_counts = False

    # 2. Wrapper for single item generation
    # This binds the result back to the original item dictionary.
    async def generate_wrapper(item):
        output = await llm.async_generate(item["prompt"], sampling_params)
        item["response"] = output["text"]
        # Token usage (best-effort)
        pt, ct, tt = _extract_token_counts(output)
        item["_token_usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
        return item

    # 3. Batch Processor
    async def process_chunk(batch_data, f_out, pbar):
        nonlocal acc_prompt_tokens, acc_completion_tokens, acc_total_tokens, saw_any_token_counts
        # Create a list of tasks for the whole chunk
        tasks = [generate_wrapper(item) for item in batch_data]
        
        # 'as_completed' yields tasks as soon as they finish, regardless of order.
        # This allows immediate writing to disk.
        for task in asyncio.as_completed(tasks):
            try:
                result_item = await task
                
                # Write immediately
                # 默认不把 _token_usage 落盘（避免污染结果文件）
                token_usage = result_item.pop("_token_usage", None)
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush() # Ensure data is written to disk immediately
                
                # Update progress bar
                pbar.update(1)

                # Update throughput stats
                if isinstance(token_usage, dict):
                    pt = token_usage.get("prompt_tokens")
                    ct = token_usage.get("completion_tokens")
                    tt = token_usage.get("total_tokens")
                    if isinstance(pt, int):
                        acc_prompt_tokens += pt
                        saw_any_token_counts = True
                    if isinstance(ct, int):
                        acc_completion_tokens += ct
                        saw_any_token_counts = True
                    if isinstance(tt, int):
                        acc_total_tokens += tt
                        saw_any_token_counts = True

                elapsed = max(1e-9, time.perf_counter() - start_t)
                items_per_s = pbar.n / elapsed
                if saw_any_token_counts:
                    # Prefer total_tokens if present, else prompt+completion
                    total_tokens = acc_total_tokens if acc_total_tokens > 0 else (acc_prompt_tokens + acc_completion_tokens)
                    tok_per_s = total_tokens / elapsed
                    pbar.set_postfix({"items/s": f"{items_per_s:.2f}", "tok/s": f"{tok_per_s:.2f}"})
                else:
                    pbar.set_postfix({"items/s": f"{items_per_s:.2f}"})
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
    elapsed = max(1e-9, time.perf_counter() - start_t)
    items_per_s = total_lines / elapsed if total_lines else 0.0
    if saw_any_token_counts:
        total_tokens = acc_total_tokens if acc_total_tokens > 0 else (acc_prompt_tokens + acc_completion_tokens)
        tok_per_s = total_tokens / elapsed
        print(f"Throughput: {items_per_s:.2f} items/s, {tok_per_s:.2f} tokens/s (elapsed {elapsed:.2f}s)")
    else:
        print(f"Throughput: {items_per_s:.2f} items/s (elapsed {elapsed:.2f}s). token统计在当前返回结构中不可用。")
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