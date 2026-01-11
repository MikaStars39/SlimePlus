from pathlib import Path
from datasets import load_dataset

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