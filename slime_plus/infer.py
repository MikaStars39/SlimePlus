import asyncio
import json
import logging
import os
import time
from pathlib import Path

import ray

from slime.rollout.sglang_rollout import generate
from slime.utils.http_utils import init_http_client
from slime.utils.types import Sample
from slime_plus.data import StreamingRolloutDataSource

logger = logging.getLogger(__name__)


def _estimate_total_prompts(input_path: str):
    """
    Best-effort estimate for total prompt count.
    Returns None when the format is unsupported for cheap counting.
    """
    if not input_path:
        return None

    lower_path = input_path.lower()
    if lower_path.endswith(".jsonl"):
        total = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
        return total

    if lower_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, list):
                return len(data)
        return None

    return None


@ray.remote
class AsyncRolloutWorker:
    def __init__(self, args):
        self.args = args
        # NOTE: HTTP client initialization is deferred to run() to ensure it happens
        # within the active asyncio event loop (Ray async actor context).

    def _build_sampling_params(self):
        """Constructs sampling parameters based on the arguments."""
        return {
            "temperature": self.args.rollout_temperature,
            "top_p": self.args.rollout_top_p,
            "top_k": self.args.rollout_top_k,
            "max_new_tokens": self.args.rollout_max_response_len,
            "stop": self.args.rollout_stop,
            "stop_token_ids": self.args.rollout_stop_token_ids,
            "skip_special_tokens": self.args.rollout_skip_special_tokens,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

    async def _producer(self, sample_queue: asyncio.Queue, batch_size: int, data_source: StreamingRolloutDataSource):
        """Fetches data from the data source and pushes it into the sample queue."""
        try:
            while True:
                # Ray ObjectRef is awaitable under asyncio.
                groups = await data_source.get_samples.remote(batch_size)
                
                # Stop producing when the data source is exhausted
                if not groups:
                    logger.info("Producer: Data source exhausted, stopping producer.")
                    break
                
                # Flatten the groups and put individual samples into the queue
                for group in groups:
                    for sample in group:
                        await sample_queue.put(sample)
        except Exception as e:
            logger.error(f"Error in producer: {e}")
            raise

    async def _consumer(self, sample_queue: asyncio.Queue, result_queue: asyncio.Queue, sampling_params: dict):
        """Consumes samples from the queue, runs inference, and pushes results to the result queue."""
        try:
            while True:
                sample = await sample_queue.get()
                
                # Exit gracefully when receiving the poison pill (None)
                if sample is None:
                    sample_queue.task_done()
                    break
                
                try:
                    # Execute the model inference
                    result = await generate(self.args, sample, sampling_params.copy())
                    await result_queue.put(result)
                except Exception as e:
                    logger.error(f"Error in consumer: {e}, with sample: {getattr(sample, 'index', 'Unknown')}")
                    # Mark sample as failed and pass the error forward
                    sample.status = Sample.Status.FAILED 
                    setattr(sample, "error_msg", str(e))
                    await result_queue.put(sample)
                finally:
                    # Notify the queue that the item has been processed
                    sample_queue.task_done()
        except Exception as e:
            logger.error(f"Fatal error in consumer: {e}")

    async def _collector(
        self,
        result_queue: asyncio.Queue,
        results: list[dict],
        sink_actor,
        sink_flush_size: int,
        sink_batch: list,
        pending_sink_refs: list,
        max_pending_sink_writes: int,
    ):
        """Collects inference results and flushes them to the remote sink actor in batches."""
        while True:
            res = await result_queue.get()
            
            # Exit gracefully when receiving the poison pill (None)
            if res is None:
                break
            
            # If no sink actor is provided, store results in memory
            if sink_actor is None:
                results.append(res)
                continue

            # Convert result to dict if necessary
            item = res.to_dict() if hasattr(res, "to_dict") else res
            sink_batch.append(item)
            
            # Flush to remote sink actor when batch size is reached
            if len(sink_batch) >= sink_flush_size:
                pending_sink_refs.append(sink_actor.write_batch.remote(sink_batch.copy()))
                sink_batch.clear()
                if len(pending_sink_refs) >= max_pending_sink_writes:
                    # Backpressure: wait for a chunk of writes to finish so refs do not grow unbounded.
                    num_to_wait = max(1, len(pending_sink_refs) // 2)
                    done_refs, remaining_refs = await asyncio.to_thread(
                        ray.wait,
                        pending_sink_refs,
                        num_returns=num_to_wait,
                        timeout=None,
                    )
                    await asyncio.gather(*done_refs)
                    pending_sink_refs[:] = remaining_refs

    async def run(self, data_source: StreamingRolloutDataSource, sink_actor=None):
        """Main execution method that orchestrates the producer-consumer pipeline."""
        
        # Initialize HTTP client here to ensure it runs within the active asyncio event loop
        init_http_client(self.args)

        # Setup configurations
        concurrency = getattr(self.args, "concurrency", 10)
        batch_size = getattr(self.args, "batch_size", 10)
        sink_flush_size = getattr(self.args, "sink_flush_size", 32)
        max_pending_sink_writes = max(1, getattr(self.args, "max_pending_sink_writes", 64))

        # Initialize queues and params
        sample_queue = asyncio.Queue(maxsize=concurrency * 2)
        result_queue = asyncio.Queue()
        sampling_params = self._build_sampling_params()

        # Shared states for the collector
        results = []
        pending_sink_refs = []
        sink_batch = []

        # 1. Start the Producer task
        producer_task = asyncio.create_task(self._producer(sample_queue, batch_size, data_source))

        # 2. Start the Consumer tasks
        consumer_tasks = []
        for _ in range(concurrency):
            consumer_task = asyncio.create_task(self._consumer(sample_queue, result_queue, sampling_params))
            consumer_tasks.append(consumer_task)

        # 3. Start the Collector task
        collector_task = asyncio.create_task(self._collector(
            result_queue,
            results,
            sink_actor,
            sink_flush_size,
            sink_batch,
            pending_sink_refs,
            max_pending_sink_writes,
        ))

        # --------------- Pipeline Synchronization ---------------
        run_error = None
        graceful_shutdown = False
        try:
            # Wait for the producer to finish fetching all data
            await producer_task

            # Wait for consumers to process all items currently in the sample_queue
            await sample_queue.join()
            graceful_shutdown = True
        except Exception as e:
            run_error = e
            logger.exception(f"Pipeline failed before completion: {e}")
        finally:
            if graceful_shutdown:
                # Normal path: let consumers exit after all queued work is done.
                for _ in range(concurrency):
                    await sample_queue.put(None)
                await asyncio.gather(*consumer_tasks, return_exceptions=True)
            else:
                # Failure path: force-stop all workers to avoid deadlocks.
                if not producer_task.done():
                    producer_task.cancel()
                for consumer_task in consumer_tasks:
                    if not consumer_task.done():
                        consumer_task.cancel()
                await asyncio.gather(*consumer_tasks, return_exceptions=True)

            # Collector exits after draining all currently queued results.
            await result_queue.put(None)
            await asyncio.gather(collector_task, return_exceptions=True)

        # --------------- Final Cleanup ---------------
        
        # Flush any remaining items in the sink_batch that didn't reach the flush threshold
        if sink_actor is not None:
            if sink_batch:
                pending_sink_refs.append(sink_actor.write_batch.remote(sink_batch.copy()))
            
            # Await all pending sink RPC writes to finish.
            if pending_sink_refs:
                await asyncio.gather(*pending_sink_refs)

        if run_error is not None:
            raise run_error

        logger.info(f"Completed rollout with {len(results)} in-memory samples.")
        return results


@ray.remote
class JsonlSink:
    def __init__(self, output_path: str, flush_every: int = 32):
        self.output_path = output_path
        self.flush_every = max(1, flush_every)
        self.total_written = 0
        self.pending_since_flush = 0
        self._progress_every = 100
        self._next_progress_at = 100
        self._resume_processed = 0
        self._total_expected_samples = None
        self._started_at = time.monotonic()

        output_parent = Path(output_path).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")

    def configure_progress(
        self,
        resume_processed: int,
        total_expected_samples: int = None,
        progress_every: int = 100,
    ):
        self._resume_processed = max(0, int(resume_processed))
        self._total_expected_samples = (
            int(total_expected_samples) if total_expected_samples is not None else None
        )
        self._progress_every = max(1, int(progress_every))
        self._next_progress_at = self._progress_every

    def _log_progress_if_needed(self):
        while self.total_written >= self._next_progress_at:
            elapsed = max(1e-6, time.monotonic() - self._started_at)
            tps = self.total_written / elapsed
            overall_processed = self._resume_processed + self.total_written
            if self._total_expected_samples and self._total_expected_samples > 0:
                pct = overall_processed / self._total_expected_samples * 100.0
                logger.info(
                    "-------------- Progress: %.2f%% (%s/%s), new=%s, tps=%.2f ---------------",
                    pct,
                    overall_processed,
                    self._total_expected_samples,
                    self.total_written,
                    tps,
                )
            else:
                logger.info(
                    "-------------- Progress: processed=%s (new=%s), total=unknown, tps=%.2f --------------",
                    overall_processed,
                    self.total_written,
                    tps,
                )
            self._next_progress_at += self._progress_every

    def read_resume_state(self, n_samples_per_prompt: int) -> dict:
        processed_samples = 0
        if os.path.exists(self.output_path):
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        processed_samples += 1
        return {
            "processed_samples": processed_samples,
            "processed_prompts": processed_samples // n_samples_per_prompt,
            "sample_remainder": processed_samples % n_samples_per_prompt,
        }

    def write_batch(self, samples: list[dict]) -> int:
        if not samples:
            return self.total_written

        for sample in samples:
            self._file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            self.total_written += 1
            self.pending_since_flush += 1
            self._log_progress_if_needed()
            if self.pending_since_flush >= self.flush_every:
                self._file.flush()
                os.fsync(self._file.fileno())
                self.pending_since_flush = 0

        return self.total_written

    def stats(self) -> dict:
        return {"output_path": self.output_path, "total_written": self.total_written}

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self.pending_since_flush = 0
        self._file.close()


async def run_streaming_inference(args):
    output_jsonl = args.plus_output_path or os.path.join(args.save, "plus_output.jsonl")
    flush_every = max(1, args.plus_flush_every)
    num_workers = max(1, args.plus_num_workers)

    sink_actor = JsonlSink.options(num_cpus=1, num_gpus=0).remote(output_jsonl, flush_every)
    resume_state = await sink_actor.read_resume_state.remote(args.n_samples_per_prompt)
    logger.info(
        "Resume state: samples=%s prompts=%s remainder=%s",
        resume_state["processed_samples"],
        resume_state["processed_prompts"],
        resume_state["sample_remainder"],
    )

    total_prompts = _estimate_total_prompts(args.plus_input_path)
    total_expected_samples = (
        total_prompts * args.n_samples_per_prompt if total_prompts is not None else None
    )

    await sink_actor.configure_progress.remote(
        resume_processed=resume_state["processed_samples"],
        total_expected_samples=total_expected_samples,
        progress_every=100,
    )

    if total_expected_samples is not None:
        logger.info(
            "Progress tracking enabled: total_expected_samples=%s, resume_processed=%s, log_every=%s",
            total_expected_samples,
            resume_state["processed_samples"],
            100,
        )
    else:
        logger.info(
            "Progress tracking enabled with unknown total (input format does not support cheap counting). log_every=%s",
            100,
        )

    # Re-create data source with resume offsets.
    data_source = StreamingRolloutDataSource.options(num_cpus=1, num_gpus=0).remote(
        args.plus_input_path,
        args,
        resume_state["processed_prompts"],
        resume_state["sample_remainder"],
        resume_state["processed_samples"],
    )

    # Inject runtime knobs used by AsyncRolloutWorker.
    args.concurrency = max(1, args.plus_worker_concurrency)
    args.batch_size = args.plus_worker_batch_size
    args.sink_flush_size = max(1, args.plus_sink_flush_size)
    args.max_pending_sink_writes = max(1, getattr(args, "plus_max_pending_sink_writes", 64))

    worker_actors = [
        AsyncRolloutWorker.options(num_cpus=1, num_gpus=0).remote(args) for _ in range(num_workers)
    ]
    tasks = [actor.run.remote(data_source, sink_actor) for actor in worker_actors]
    try:
        await asyncio.gather(*tasks)
    finally:
        try:
            await sink_actor.close.remote()
        except Exception:
            logger.exception("Failed to close JsonlSink cleanly.")

    sink_stats = await sink_actor.stats.remote()
    logger.info(
        "Streaming inference finished. Output: %s (%s samples)",
        sink_stats["output_path"],
        sink_stats["total_written"],
    )