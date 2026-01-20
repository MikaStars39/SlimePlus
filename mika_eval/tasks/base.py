import logging
import os
from pathlib import Path
from datasets import load_dataset

import src.tasks as tasks

DATASETS = {
# --------------------------- math ------------------------------

    "aime2024": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "custom_args": [],  # custom args from the dataset
        "need_llm_extract": False,  # if need llm to extract answer
        "eval_type": "math",
    },
    "aime2025": {
        "hf_name": "yentinglin/aime_2025",
        "split": "train",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },
    "amc2023": {
        "hf_name": "zwhe99/amc23",
        "split": "test",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },
    "minerva": {
        "hf_name": "math-ai/minervamath",
        "split": "test",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },
    "hmmt2025": {
        "hf_name": "FlagEval/HMMT_2025",
        "split": "train",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },
    "imo_answerbench": {
        "hf_name": "Hwilner/imo-answerbench",
        "split": "train",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },
    "beyond_aime": {
        "hf_name": "ByteDance-Seed/BeyondAIME",
        "split": "test",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },

# --------------------------- GeneralQA ------------------------------

    "gpqa_diamond": {
        "hf_name": "fingertap/GPQA-Diamond",
        "split": "test",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },

    "mmlu_pro": {
        "hf_name": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },

    "ceval": {
        "hf_name": "ceval/ceval-exam",
        "split": "test",
        "custom_args": [],
        "need_llm_extract": False,
        "eval_type": "math",
    },

# --------------------------- Instruction Following ------------------------------

    "ifeval": {
        "hf_name": "google/IFEval",
        "split": "train",
        "need_llm_extract": False,
        "eval_type": "ifeval",
    },

    "ifbench": {
        "hf_name": "allenai/IFBench_test",
        "split": "train",
        "custom_args": ["instruction_id_list", "kwargs"],
        "need_llm_extract": False,
        "eval_type": "ifbench",
    },

# --------------------------- Training Data ------------------------------

    "DAPO_Math_17k_Processed": {
        "hf_name": "open-r1/DAPO-Math-17k-Processed",
        "split": "train",
        "custom_args": [],
        "need_llm_extract": True,
        "eval_type": "math",
    }
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

def prepare_pass_at_k_jsonl(
    config_str: str, 
    output_file: str, 
    cache_dir: str = None
):
    """
    Parses config_str (e.g., 'aime2024@32,math500@4') and generates a JSONL file 
    where each question is repeated k times for Pass@k sampling.
    """
    dataset_configs = []
    for item in config_str.split(","):
        name, k_val = item.split("@")
        dataset_configs.append((name.strip(), int(k_val.strip())))

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for ds_name, k in dataset_configs:

            logging.info(f"Processing {ds_name} (repeat {k} times)...")
            
            loader_name = f"load_{ds_name.replace('-', '_')}"
            loader = getattr(tasks, loader_name, None)
            if loader is None:
                raise ValueError(
                    f"Could not find loader '{loader_name}' for dataset '{ds_name}'. "
                    f"Please implement '{loader_name}(dataset_name, cache_dir, k, f_out)'."
                )
            loader(ds_name, cache_dir, k, f_out)

    logging.info(f"Successfully generated {output_file}.")
