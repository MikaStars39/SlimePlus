import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import aiohttp
from typing import List, Tuple, Iterable, Dict, Any, Set, Optional
from pathlib import Path
from src.utils import prepare_prompt, ProgressVisualizer, StageContext
from src.data import load_dataset_from_hf


PROMPT_TEMPLATES = {
    "lighteval": """{problem} Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "open-r1": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),
    "extraction": """
Please extract the final answer from the following response. The answer should be put inside \\boxed{{}}. 

Response:
{response}
""".strip(),
}


def extract_vllm_args(unknown: List[str]) -> Tuple[List[str], List[str]]:
    vllm_args: List[str] = []
    leftover: List[str] = []
    idx = 0
    while idx < len(unknown):
        token = unknown[idx]
        if token.startswith("--vllm-"):
            stripped = "--" + token[len("--vllm-") :]
            if "=" in token:
                _, value = token.split("=", 1)
                vllm_args.extend([stripped, value])
            elif idx + 1 < len(unknown) and not unknown[idx + 1].startswith("-"):
                vllm_args.extend([stripped, unknown[idx + 1]])
                idx += 1
            else:
                vllm_args.append(stripped)
        else:
            leftover.append(token)
        idx += 1
    return vllm_args, leftover


def build_vllm_command(
    model_path: Path, port: int, args: argparse.Namespace, vllm_args: List[str]
) -> List[str]:
    dp_size = max(1, args.dp_size)
    max_concurrent_per_dp = max(1, args.max_num_request // dp_size)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        args.served_model_name,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(args.tp_size),
        "--max-num-seqs",
        str(max_concurrent_per_dp),
    ]

    # Solution: append --gpu-memory-utilization to vLLM command, default 0.95, configurable via CLI.
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(vllm_args)
    return cmd


def pipe_to_logger(
    stream: Iterable[str], logger: logging.Logger, level: int, prefix: str
) -> None:
    for line in stream:
        logger.log(level, "%s%s", prefix, line.rstrip("\n"))


def start_vllm_processes(
    model_path: Path,
    args: argparse.Namespace,
    vllm_args: List[str],
    logger: logging.Logger,
) -> Tuple[List[subprocess.Popen], List[int]]:
    ports: List[int] = []
    processes: List[subprocess.Popen] = []
    env = os.environ.copy()
    dp_size = max(1, args.dp_size)

    for rank in range(dp_size):
        # Calculate GPU ID range for current process
        start_gpu_id = rank * args.tp_size
        end_gpu_id = start_gpu_id + args.tp_size
        gpu_ids = list(range(start_gpu_id, end_gpu_id))

        # Check for out-of-bounds (based on args.num_gpus or simple logic, assuming user config is correct)

        env_local = env.copy()
        env_local["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        port = args.serve_port + rank
        cmd = build_vllm_command(model_path, port, args, vllm_args)
        logger.info(
            "Starting vLLM backend [%d/%d], port %d, GPUs=%s, command: %s",
            rank + 1,
            dp_size,
            port,
            gpu_ids,
            " ".join(cmd),
        )
        proc = subprocess.Popen(
            cmd,
            env=env_local,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        processes.append(proc)
        ports.append(port)
        if proc.stdout:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stdout, logger, logging.INFO, f"[vllm:{port}] "),
                daemon=True,
            ).start()
        if proc.stderr:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stderr, logger, logging.ERROR, f"[vllm:{port}] "),
                daemon=True,
            ).start()
    return processes, ports


def stop_vllm_processes(
    processes: List[subprocess.Popen], logger: logging.Logger
) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                logger.info("Attempting to terminate vLLM process (pid=%d).", proc.pid)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Exception during process termination (pid=%d): %s", proc.pid, exc
                )
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass


def wait_for_vllm_ready(
    port: int, process: subprocess.Popen, timeout: float, logger: logging.Logger
) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            logger.error("vLLM process (pid=%d) exited prematurely.", process.pid)
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("vLLM at port %d is ready.", port)
                    return True
        except Exception:
            time.sleep(2)
    logger.error("Timeout waiting for vLLM at port %d.", port)
    return False


async def generate_with_vllm_async(
    session: aiohttp.ClientSession, prompt: str, port: int, args: argparse.Namespace
) -> str:
    """Async version of vLLM generation function for concurrent requests."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": 1,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    try:
        async with session.post(
            url, json=payload, headers=headers, timeout=timeout
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM returned HTTP error: {response.status}")
            content = await response.json()
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"vLLM connection failed: {exc}") from exc

    try:
        return content["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse vLLM response: {content}") from exc


async def run_batch_inference(
    tasks: List[Dict[str, Any]],
    ports: List[int],
    semaphores: Dict[int, asyncio.Semaphore],
    args: argparse.Namespace,
    logger: logging.Logger,
    output_file: Optional[Path] = None,
    visualizer: Optional[ProgressVisualizer] = None,
) -> List[Dict[str, Any]]:
    """
    Generic batch inference interface.
    tasks: List of dicts. Each dict must contain "prompt".
           If "port_idx" is present, it will be used to select the port from ports list.
           Otherwise, it will be assigned automatically (round-robin).
           All other fields in the dict will be included in the result record.
    """
    if not tasks:
        return []

    results: List[Dict[str, Any]] = []
    file_lock = asyncio.Lock()
    ports_cycle = len(ports)

    async def generate_one_task(
        task: Dict[str, Any],
        task_idx: int,
        session: aiohttp.ClientSession,
    ) -> None:
        prompt = task["prompt"]
        port_idx = task.get("port_idx")
        if port_idx is None:
            port_idx = task_idx % ports_cycle

        port = ports[port_idx]
        semaphore = semaphores[port]

        # Prepare the record for output
        record = task.copy()
        # We don't want port_idx in the final saved record if it was just for routing
        # record.pop("port_idx", None)

        async with semaphore:
            try:
                response = await generate_with_vllm_async(
                    session, prompt, port, args
                )
                record["response"] = response
            except Exception as exc:
                logger.error(
                    "Generation failed for task %d (port=%d): %s",
                    task_idx,
                    port,
                    exc,
                )
                return

        results.append(record)

        if output_file:
            async with file_lock:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if visualizer:
            pid = record.get("problem_id")
            rid = record.get("rollout_id")
            if pid is not None and rid is not None:
                await visualizer.update(pid, rid)

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [
            generate_one_task(task, i, session)
            for i, task in enumerate(tasks)
        ]
        await asyncio.gather(*coros)

    return results


async def generate_responses(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    ports: List[int],
    logger: logging.Logger,
    semaphores: Dict[int, asyncio.Semaphore],
    do_generation: bool = True,
    do_judge_extract: bool = False,
) -> None:
    """
    Asynchronously generate responses and save to outputs.jsonl.
    Implementation: Read existing outputs.jsonl to build cache, only generate missing entries.
    Generated results are appended to outputs.jsonl in real-time.
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "outputs.jsonl"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    ds = None

    # Pass 1: Original Generation
    if do_generation:
        with StageContext(logger, f"C.1[{dataset_name}]", "Reading cached output (Pass 1)"):
            cache: Set[Tuple[int, int]] = set()

            if output_file.exists():
                with output_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if (
                                "problem_id" in data
                                and "rollout_id" in data
                                and "response" in data
                                and data["response"] != ""
                                and int(data["rollout_id"]) < rollout_n
                            ):
                                cache.add((data["problem_id"], data["rollout_id"]))
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON line in outputs.jsonl, skipped.")

            logger.info("Loaded cache entries for generation: %d", len(cache))

        with StageContext(logger, f"C.2[{dataset_name}]", "Preparing generation tasks"):
            ds = load_dataset_from_hf(dataset_name, args.cache_dir)
            tasks_to_process: List[Dict[str, Any]] = []
            ports_cycle = len(ports)
            prompt_template = PROMPT_TEMPLATES[args.prompt_format]

            for idx, sample in enumerate(ds):
                prompt = prepare_prompt(dataset_name, sample, prompt_template)
                for rollout_id in range(rollout_n):
                    if (idx, rollout_id) in cache:
                        continue
                    port_idx = (idx * rollout_n + rollout_id) % ports_cycle
                    tasks_to_process.append(
                        {
                            "problem_id": idx,
                            "rollout_id": rollout_id,
                            "prompt": prompt,
                            "port_idx": port_idx,
                        }
                    )

            logger.info("New requests to generate: %d", len(tasks_to_process))
            visualizer = ProgressVisualizer(
                dataset_dir / "process_gen.txt", len(ds), rollout_n, cache
            )

            if tasks_to_process:
                with StageContext(logger, f"C.3[{dataset_name}]", "Parallel Generation"):
                    await run_batch_inference(
                        tasks=tasks_to_process,
                        ports=ports,
                        semaphores=semaphores,
                        args=args,
                        logger=logger,
                        output_file=output_file,
                        visualizer=visualizer,
                    )
            else:
                logger.info("All generation requests exist in cache.")
            visualizer.cleanup()

    # Pass 2: LLM-as-a-judge Extraction (Optional)
    if do_judge_extract:
        judge_file = dataset_dir / "judge.jsonl"
        with StageContext(logger, f"C.4[{dataset_name}]", "Reading cached output (Pass 2: Extraction)"):
            extraction_cache: Set[Tuple[int, int]] = set()
            responses_to_extract: List[Dict[str, Any]] = []

            # We need to load the output file to get responses
            all_records: Dict[Tuple[int, int], Dict[str, Any]] = {}
            if output_file.exists():
                with output_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            pid, rid = data.get("problem_id"), data.get("rollout_id")
                            if pid is None or rid is None:
                                continue
                            if (pid, rid) not in all_records:
                                all_records[(pid, rid)] = data
                            else:
                                all_records[(pid, rid)].update(data)
                        except json.JSONDecodeError:
                            pass

            # Load existing judge cache
            if judge_file.exists():
                with judge_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            pid, rid = data.get("problem_id"), data.get("rollout_id")
                            if pid is not None and rid is not None:
                                extraction_cache.add((pid, rid))
                        except json.JSONDecodeError:
                            pass

            for (pid, rid), data in all_records.items():
                if (pid, rid) in extraction_cache:
                    continue
                if "response" in data and data["response"]:
                    responses_to_extract.append(data)

            logger.info("Loaded judge cache entries: %d", len(extraction_cache))
            logger.info("New requests to extract with judge: %d", len(responses_to_extract))

        if responses_to_extract:
            with StageContext(logger, f"C.5[{dataset_name}]", "LLM-as-a-judge Answer Extraction"):
                if ds is None:
                    ds = load_dataset_from_hf(dataset_name, args.cache_dir)
                
                from src.grader import extract_answer
                extraction_tasks: List[Dict[str, Any]] = []
                extract_template = PROMPT_TEMPLATES["extraction"]
                ports_cycle = len(ports)

                for i, record in enumerate(responses_to_extract):
                    pid, rid = record["problem_id"], record["rollout_id"]
                    prompt = extract_template.format(response=record["response"])
                    port_idx = (pid * rollout_n + rid) % ports_cycle
                    
                    task = {
                        "problem_id": pid,
                        "rollout_id": rid,
                        "prompt": prompt,
                        "port_idx": port_idx,
                    }
                    extraction_tasks.append(task)

                visualizer_extract = ProgressVisualizer(
                    dataset_dir / "process_extract.txt", len(ds), rollout_n, extraction_cache
                )

                results = await run_batch_inference(
                    tasks=extraction_tasks,
                    ports=ports,
                    semaphores=semaphores,
                    args=args,
                    logger=logger,
                    output_file=None,
                    visualizer=visualizer_extract,
                )

                # Now save to judge.jsonl
                with judge_file.open("a", encoding="utf-8") as f:
                    for res in results:
                        llm_output = res["response"]
                        # Extract content inside \boxed{}
                        boxed_content = extract_answer(llm_output)
                        record = {
                            "problem_id": res["problem_id"],
                            "rollout_id": res["rollout_id"],
                            "judge_response": llm_output,
                            "extracted_response": boxed_content,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                visualizer_extract.cleanup()
        else:
            logger.info("All judge extraction requests exist in cache.")

    logger.info(
        "Dataset %s processing complete, results saved to %s",
        dataset_name,
        output_file,
    )