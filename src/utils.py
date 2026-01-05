# logging_utils.py
import logging
import sys
import time
import torch
import gc
import argparse

from typing import Any, Dict, Tuple, Set
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datasets import load_dataset
import asyncio


# ------------------ model utils ---------------------


def resolve_torch_dtype(dtype: Any) -> Any:
    """
    Parse dtype string to torch.dtype, supporting auto/common aliases,
    compatible with older Transformers lacking get_torch_dtype.
    """
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized == "auto":
            return None
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"Unsupported dtype: {dtype}")


def merge_model_if_needed(
    args: argparse.Namespace, result_dir: Path, logger: logging.Logger
) -> Path:
    if not args.adapter:
        logger.info("No adapter provided, using base model directly: %s", args.model)
        return Path(args.model)

    output_dir = result_dir / "model"
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("Detected existing merged model directory, reusing: %s", output_dir)
        return output_dir

    adapter_path = Path(args.adapter).resolve()
    if not adapter_path.exists():
        raise ValueError(f"Adapter path does not exist: {adapter_path}")

    # Check for critical files to prevent peft from mistaking it for a huggingface repo id
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(f"adapter_config.json not found in {adapter_path}")

    torch_dtype = resolve_torch_dtype(args.dtype)
    logger.info("Loading base model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("Loading LoRA/PEFT adapter: %s", adapter_path)
    # Explicitly cast to str
    model = PeftModel.from_pretrained(model, str(adapter_path), device_map="cuda")
    logger.info("Executing merge_and_unload, writing LoRA weights into base model.")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Actively release memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ------------------ logging utils ---------------------


class StreamToLogger:
    """Redirect stdout/stderr to logger to ensure output is recorded in both file and console."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, buffer: str) -> None:
        self._buffer += buffer
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.logger.log(self.level, line)

    def flush(self) -> None:
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""


def setup_logging(result_dir: Path) -> logging.Logger:
    result_dir.mkdir(parents=True, exist_ok=True)
    latest_log_path = result_dir / "latest_run.log"
    log_path = result_dir / "logs" / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    latest_file_handler = logging.FileHandler(
        latest_log_path, mode="w", encoding="utf-8"
    )
    latest_file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(latest_file_handler)
    logging.root.addHandler(console_handler)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    stdout_logger.propagate = True
    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    stderr_logger.propagate = True
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    return logging.getLogger("eval_all")


class StageContext:
    """Stage logging context to mark start/end and failure scenarios."""

    def __init__(
        self,
        logger: logging.Logger,
        stage_id: int | str,
        name: str,
        emoji_start: str = "🚀",
        emoji_end: str = "🏁",
        emoji_fail: str = "💥",
    ) -> None:
        self.logger = logger
        self.stage_id = str(stage_id)
        self.name = name
        self.emoji_start = emoji_start
        self.emoji_end = emoji_end
        self.emoji_fail = emoji_fail

    def __enter__(self) -> "StageContext":
        self.logger.info(
            "%s Stage %s started: %s", self.emoji_start, self.stage_id, self.name
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if exc_type is None:
            self.logger.info(
                "%s Stage %s ended: %s", self.emoji_end, self.stage_id, self.name
            )
        else:
            self.logger.error(
                "%s Stage %s failed: %s, Error: %s",
                self.emoji_fail,
                self.stage_id,
                self.name,
                exc,
            )


# ------------------ dataset utils ---------------------

DATASETS = {
    "aime2024": ("HuggingFaceH4/aime_2024", "train"),
    "aime2025": ("yentinglin/aime_2025", "train"),
    "amc2023": ("zwhe99/amc23", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "minerva": ("math-ai/minervamath", "test"),
    "hmmt2025": ("FlagEval/HMMT_2025", "train"),
}


def load_dataset_from_hf(dataset_name: str, cache_dir: str = None):
    if dataset_name in DATASETS:
        if cache_dir is not None:
            # Use the HuggingFace dataset name to construct cache path
            hf_name, _ = DATASETS[dataset_name]
            cache_dataset_name = hf_name.split("/")[-1]  # Extract dataset name from HF path
            cache_path = Path(cache_dir) / cache_dataset_name
            if cache_path.exists():
                # Load from local cache directory
                return load_dataset(str(cache_path), split="train")
        # Fall back to HuggingFace
        hf_name, split = DATASETS[dataset_name]
        return load_dataset(hf_name, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def prepare_prompt(
    dataset_name: str, sample: Dict[str, Any], prompt_template: str
) -> str:
    """Construct model input prompt based on sample, modify as needed."""
    problem = None
    if "problem" in sample:
        problem = sample["problem"]
    elif "question" in sample:
        problem = sample["question"]
    elif "prompt" in sample:
        problem = sample["prompt"]
    else:
        raise ValueError(f"Unsupported sample format: {sample}")
    return prompt_template.format(problem=problem)


class ProgressVisualizer:
    def __init__(
        self,
        filepath: Path,
        problem_n: int,
        rollout_n: int,
        completed: Set[Tuple[int, int]],
    ) -> None:
        self.filepath = filepath
        self.problem_n = problem_n
        self.rollout_n = rollout_n
        # Row: rollout_id, Col: problem_id
        self.grid = [["." for _ in range(problem_n)] for _ in range(rollout_n)]
        for pid, rid in completed:
            if 0 <= rid < rollout_n and 0 <= pid < problem_n:
                self.grid[rid][pid] = "X"
        self.lock = asyncio.Lock()
        self._write_sync()

    def _write_sync(self) -> None:
        try:
            with self.filepath.open("w", encoding="utf-8") as f:
                for row in self.grid:
                    f.write("".join(row) + "\n")
        except Exception:
            pass

    async def update(self, problem_id: int, rollout_id: int) -> None:
        if 0 <= rollout_id < self.rollout_n and 0 <= problem_id < self.problem_n:
            async with self.lock:
                if self.grid[rollout_id][problem_id] != "X":
                    self.grid[rollout_id][problem_id] = "X"
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._write_sync
                    )

    def cleanup(self) -> None:
        try:
            if self.filepath.exists():
                self.filepath.unlink()
        except Exception:
            pass