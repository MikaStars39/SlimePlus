import logging
import os
from pathlib import Path
from datasets import load_dataset

import open_gym.tasks as tasks

from .process_func import *

DATASETS = {
# --------------------------- math ------------------------------

    "aime2024": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "need_llm_extract": False,  # if need llm to extract answer
        "eval_type": "math",
        "process_func": "load_aime2024",
    },
    "aime2025": {
        "hf_name": "yentinglin/aime_2025",
        "split": "train",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_aime2025",
    },
    "amc2023": {
        "hf_name": "zwhe99/amc23",
        "split": "test",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_amc2023",
    },
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_math500",
    },
    "minerva": {
        "hf_name": "math-ai/minervamath",
        "split": "test",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_minerva",
    },
    "hmmt2025": {
        "hf_name": "FlagEval/HMMT_2025",
        "split": "train",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_hmmt2025",
    },
    "beyond_aime": {
        "hf_name": "ByteDance-Seed/BeyondAIME",
        "split": "test",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_beyond_aime",
    },

# --------------------------- GeneralQA ------------------------------

    "gpqa_diamond": {
        "hf_name": "fingertap/GPQA-Diamond",
        "split": "test",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_gpqa_diamond",
    },

    "mmlu_pro": {
        "hf_name": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_mmlu_pro",
    },

    "ceval": {
        "hf_name": "ceval/ceval-exam",
        "split": "test",
        "need_llm_extract": False,
        "eval_type": "math",
        "process_func": "load_ceval",
    },

# --------------------------- Instruction Following ------------------------------

    "ifeval": {
        "hf_name": "google/IFEval",
        "split": "train",
        "need_llm_extract": False,
        "eval_type": "ifeval",
        "process_func": "load_ifeval",
    },

    "ifbench": {
        "hf_name": "allenai/IFBench_test",
        "split": "train",
        "need_llm_extract": False,
        "eval_type": "ifbench",
        "process_func": "load_ifbench"
    },
}

def get_question_text(row):
    """Identify the question/problem column in different dataset schemas."""
    for key in ["problem", "question", "prompt", "instruction"]:
        if key in row:
            return row[key]
    raise KeyError(f"Could not find a question column in row: {row.keys()}")


def get_answer_text(row):
    """Identify the answer/solution column in different dataset schemas."""
    for key in ["answer", "solution", "label", "target", "correct_answer", "gold_answer"]:
        if key in row:
            return str(row[key])
    return ""


def load_dataset_from_hf(dataset_name: str, cache_dir: str = None):
    """Loads a dataset from HuggingFace or local cache."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if cache_dir is not None:
        cache_dataset_name = DATASETS[dataset_name]["hf_name"].split("/")[-1]
        cache_path = Path(cache_dir) / cache_dataset_name
        logging.info(f"Cache path: {cache_path}")
        logging.info(f"Cache path exists: {cache_path.exists()}")
        if cache_path.exists():
            try:
                return load_dataset(str(cache_path), split=DATASETS[dataset_name]["split"])
            except Exception as e:
                logging.info(f"Cache loading failed, falling back to HF: {e}")

    return load_dataset(DATASETS[dataset_name]["hf_name"], split=DATASETS[dataset_name]["split"])
