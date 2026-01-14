

DATASETS = {
    "aime2024": {
        "hf_name": "HuggingFaceH4/aime_2024", 
        "split": "train", 
        "custom_args": [], # custom args from the dataset
        "need_llm_extract": False, # if need llm to extract answer
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
    "gpqa_diamond": {
        "hf_name": "fingertap/GPQA-Diamond",
        "split": "test",
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
    "ifeval": {
        "hf_name": "google/IFEval",
        "split": "train",
        "custom_args": ["instruction_id_list","kwargs"],
        "need_llm_extract": False,
        "eval_type": "ifeval",
    }
}

