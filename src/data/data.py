import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Configuration for supported datasets
DATASETS = {
    "aime2024": ("HuggingFaceH4/aime_2024", "train"),
    "aime2025": ("yentinglin/aime_2025", "train"),
    "amc2023": ("zwhe99/amc23", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "minerva": ("math-ai/minervamath", "test"),
    "hmmt2025": ("FlagEval/HMMT_2025", "train"),
    "gpqa_diamond": ("hendrydong/gpqa_diamond", "test"),
    "imo-answerbench": ("Hwilner/imo-answerbench", "train"),
}

def load_dataset_from_hf(dataset_name: str, cache_dir: str = None):
    """Loads a dataset from HuggingFace or local cache."""
    if dataset_name in DATASETS:
        hf_name, split = DATASETS[dataset_name]
        if cache_dir is not None:
            cache_dataset_name = hf_name.split("/")[-1]
            cache_path = Path(cache_dir) / cache_dataset_name
            if cache_path.exists():
                try:
                    return load_dataset(str(cache_path), split=split)
                except Exception as e:
                    print(f"Cache loading failed, falling back to HF: {e}")
        return load_dataset(hf_name, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_question_text(row):
    """Identify the question/problem column in different dataset schemas."""
    # List of common column names for the prompt/question
    for key in ["problem", "question", "prompt", "instruction"]:
        if key in row:
            return row[key]
    raise KeyError(f"Could not find a question column in row: {row.keys()}")

def prepare_pass_at_k_jsonl(config_str: str, output_file: str, cache_dir: str = None):
    """
    Parses config_str (e.g., 'aime2024@32,math500@4') and generates a JSONL file 
    where each question is repeated k times for Pass@k sampling.
    """
    # 1. Parse the configuration string
    # Result: [('aime2024', 32), ('math500', 4), ...]
    dataset_configs = []
    for item in config_str.split(","):
        name, k_val = item.split("@")
        dataset_configs.append((name.strip(), int(k_val.strip())))

    total_records = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for ds_name, k in dataset_configs:
            print(f"Processing {ds_name} (repeat {k} times)...")
            
            # 2. Load the dataset
            data = load_dataset_from_hf(ds_name, cache_dir)
            
            # 3. Iterate through rows and repeat k times
            for idx, row in enumerate(tqdm(data, desc=f"Loading {ds_name}")):
                question = get_question_text(row)
                
                for sample_idx in range(k):
                    # Create a unique ID for each attempt
                    # Format: {dataset}_{original_index}_{attempt_index}
                    unique_id = f"{ds_name}_{idx}_{sample_idx}"
                    
                    record = {
                        "id": unique_id,
                        "source": ds_name,
                        "prompt": question,  # 'prompt' key matches the offline engine script
                        "sample_index": sample_idx
                    }
                    
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1

    print(f"Successfully generated {output_file} with {total_records} total records.")

if __name__ == "__main__":
    # The input configuration string
    config_str = "aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
    output_file = "outputs/debug/prepared_inference_data.jsonl"
    cache_dir = "/mnt/llm-train/users/explore-train/qingyu/MikaEval/.cache"
    
    prepare_pass_at_k_jsonl(
        config_str=config_str, 
        output_file=output_file, 
        cache_dir=cache_dir,
    )