import json
import asyncio
import os
import sglang as sgl
from tqdm.asyncio import tqdm
import time
from typing import Optional, Tuple, Any, Dict

async def run_offline_async_inference(
    input_file: str, 
    output_file: str, 
    model_path: str, 
    chunk_size: int = 512,
    max_inflight: Optional[int] = None,
    dp_size: int = 1,
    tp_size: int = 1,
    mem_fraction_static: float = 0.90,
    sampling_params: dict = None,
    flush_every_n: int = 1,
    flush_interval_s: Optional[float] = None,
    resume: bool = False,
    # Speculative Decoding Parameters
    speculative_algorithm: Optional[str] = None,
    speculative_draft_model_path: Optional[str] = None,
    speculative_num_steps: Optional[int] = None,
    speculative_eagle_topk: Optional[int] = None,
    speculative_num_draft_tokens: Optional[int] = None,
):

    if sampling_params is None:
        raise ValueError("sampling_params is required")

    # Set environment variable for longer context if needed
    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

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
        trust_remote_code=True,
        # Pass speculative decoding parameters
        speculative_algorithm=speculative_algorithm,
        speculative_draft_model_path=speculative_draft_model_path,
        speculative_num_steps=speculative_num_steps,
        speculative_eagle_topk=speculative_eagle_topk,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
    )

    # Throughput accounting
    start_t = time.perf_counter()
    acc_prompt_tokens = 0
    acc_completion_tokens = 0
    acc_total_tokens = 0
    saw_any_token_counts = False
    pending_writes = 0
    last_flush_t = time.perf_counter()

    # 2. Wrapper for single item generation
    # This binds the result back to the original item dictionary.
    async def generate_wrapper(item):
        # SGLang sampling params can vary slightly across versions/builds.
        # If the user passes an unsupported key (e.g., stop / penalties), fail once and retry
        # with a conservative filtered param set to keep long eval jobs from crashing.
        async def _generate_with_fallback(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return await llm.async_generate(prompt, params)
            except Exception as e:
                # Best-effort fallback: drop a few common optional keys and retry once.
                drop_keys = {
                    "stop",
                    "stop_token_ids",
                    "repetition_penalty",
                    "frequency_penalty",
                    "presence_penalty",
                    "min_new_tokens",
                }
                filtered = {k: v for k, v in (params or {}).items() if k not in drop_keys}
                if filtered == (params or {}):
                    raise
                print(
                    f"[offline] async_generate failed with optional sampling params; retrying without "
                    f"{sorted(list(drop_keys))}. Error: {e}"
                )
                return await llm.async_generate(prompt, filtered)

        output = await _generate_with_fallback(item["prompt"], sampling_params)
        item["response"] = output["text"]
        # Token usage (best-effort)
        pt, ct, tt = _extract_token_counts(output)
        item["_token_usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
        return item

    # 3. Streaming Processor (producer-consumer) to reduce batching bubbles
    if max_inflight is None:
        max_inflight = chunk_size
    max_inflight = max(1, int(max_inflight))
    queue_max = max(1, max_inflight * 2)
    work_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
    write_lock = asyncio.Lock()

    async def _write_result(result_item, f_out, pbar):
        nonlocal acc_prompt_tokens, acc_completion_tokens, acc_total_tokens, saw_any_token_counts
        nonlocal pending_writes, last_flush_t
        # 默认不把 _token_usage 落盘（避免污染结果文件）
        token_usage = result_item.pop("_token_usage", None)
        f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        pending_writes += 1
        do_flush = False
        if flush_every_n <= 1:
            do_flush = True
        elif pending_writes >= flush_every_n:
            do_flush = True
        if flush_interval_s is not None and flush_interval_s >= 0:
            if (time.perf_counter() - last_flush_t) >= flush_interval_s:
                do_flush = True
        if do_flush:
            f_out.flush() # Ensure data is written to disk
            pending_writes = 0
            last_flush_t = time.perf_counter()

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

    async def producer(f_in, num_workers: int, existing_ids: set):
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if existing_ids and data.get("id") in existing_ids:
                    continue
                await work_queue.put(data)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line[:50]}...")
        for _ in range(num_workers):
            await work_queue.put(None)

    async def worker(f_out, pbar):
        while True:
            item = await work_queue.get()
            if item is None:
                work_queue.task_done()
                break
            try:
                result_item = await generate_wrapper(item)
                async with write_lock:
                    await _write_result(result_item, f_out, pbar)
            except Exception as e:
                print(f"Error processing item: {e}")
            finally:
                work_queue.task_done()

    # 4. Main Logic
    def _count_nonempty_lines(path: str) -> int:
        return sum(1 for _ in open(path, 'r', encoding='utf-8') if _.strip())

    def _load_existing_ids(path: str) -> set:
        ids = set()
        if not os.path.exists(path):
            return ids
        with open(path, "r", encoding="utf-8") as f_exist:
            for line in f_exist:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("response") is not None and row.get("id") is not None:
                    ids.add(row.get("id"))
        return ids

    print("Counting total lines...")
    total_lines = _count_nonempty_lines(input_file)
    existing_ids = _load_existing_ids(output_file) if resume else set()
    remaining_lines = max(0, total_lines - len(existing_ids))
    if resume and existing_ids:
        print(f"Resuming from {len(existing_ids)} completed items...")
    print(f"Starting Inference on {remaining_lines} items (total {total_lines})...")

    if resume and remaining_lines == 0:
        print("Nothing to do (output already complete).")
        llm.shutdown()
        return

    out_mode = "a" if resume and existing_ids else "w"
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, out_mode, encoding='utf-8') as f_out:
        
        pbar = tqdm(total=remaining_lines, desc="Inference")

        num_workers = max_inflight
        producer_task = asyncio.create_task(producer(f_in, num_workers, existing_ids))
        workers = [asyncio.create_task(worker(f_out, pbar)) for _ in range(num_workers)]

        await producer_task
        await work_queue.join()
        for w in workers:
            await w

        pbar.close()

    llm.shutdown()
    elapsed = max(1e-9, time.perf_counter() - start_t)
    items_per_s = total_lines / elapsed if total_lines else 0.0
    if saw_any_token_counts:
        total_tokens = acc_total_tokens if acc_total_tokens > 0 else (acc_prompt_tokens + acc_completion_tokens)
        tok_per_s = total_tokens / elapsed
        print(f"Throughput: {items_per_s:.2f} items/s, {tok_per_s:.2f} tokens/s (elapsed {elapsed:.2f}s)")
    else:
        print(f"Throughput: {items_per_s:.2f} items/s (elapsed {elapsed:.2f}s).")
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
        sampling_params=params,
        # Example Speculative Decoding Config:
        # speculative_algorithm="EAGLE3",
        # speculative_draft_model_path="lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex",
        # speculative_num_steps=3,
        # speculative_eagle_topk=1,
        # speculative_num_draft_tokens=4,
    ))