"""
Prepare benchmark data for pass@k evaluation.

Usage:
    python prepare_data.py --dataset aime2024@32,math500@16 \
        --cache-dir /path/to/cache --out-dir /path/to/output \
        --model /path/to/model --prompt-format slime
"""

import argparse
import json
import logging
import os

import open_gym.tasks as tasks

from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from open_gym.tasks import DATASETS
from open_gym.utils.template import PROMPT_TEMPLATES, SYSTEM_PROMPT_TEMPLATES


def prepare_dataset(
    dataset_name: str,
    cache_dir: str,
    k: int,
    f_out,
):
    func_name = DATASETS[dataset_name]["process_func"]
    func = getattr(tasks, func_name)
    func(dataset_name, cache_dir, k, f_out)


def apply_chat_template(
    input_file: str,
    output_file: str,
    model_path: str,
    prompt_format: str = "blank",
    system_prompt: str = None,
):
    """Apply chat template using model's tokenizer."""
    logging.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    user_template = PROMPT_TEMPLATES.get(prompt_format, PROMPT_TEMPLATES["blank"])

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, desc="Applying template"):
            if not line.strip():
                continue
            
            data = json.loads(line)
            formatted_content = user_template.replace("{problem}", data.get("prompt", ""))

            messages = [{"role": "user", "content": formatted_content}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            try:
                final_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logging.warning(f"Chat template failed, using raw prompt: {e}")
                final_prompt = formatted_content

            data["prompt"] = final_prompt
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    logging.info(f"Saved formatted data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data for pass@k evaluation.")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset spec, e.g., aime2024@32,math500@16")
    parser.add_argument("--cache-dir", type=str, default=None, 
                        help="Cache directory for HuggingFace datasets")
    parser.add_argument("--out-dir", type=str, required=True, 
                        help="Output directory for prepared data")
    parser.add_argument("--model", type=str, default=None, 
                        help="Model path for chat template (optional)")
    parser.add_argument("--prompt-format", type=str, default="blank", 
                        help="Prompt template format (slime, lighteval, blank, etc.)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = out_dir / "data.jsonl"
    formatted_file = out_dir / "data.chat.jsonl"

    # Step 1: Parse dataset configs and prepare raw data
    dataset_configs = []
    for item in args.dataset.split(","):
        name, k_val = item.strip().split("@")
        dataset_configs.append((name.strip(), int(k_val.strip())))

    logging.info(f"Preparing data for: {dataset_configs}")
    
    with open(data_file, "w", encoding="utf-8") as f_out:
        for ds_name, k in dataset_configs:
            logging.info(f"Processing {ds_name} with k={k}...")
            prepare_dataset(ds_name, args.cache_dir, k, f_out)

    logging.info(f"Raw data saved to {data_file}")

    # Step 2: Apply chat template if model path provided
    if args.model:
        apply_chat_template(
            input_file=str(data_file),
            output_file=str(formatted_file),
            model_path=args.model,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
        )
    else:
        logging.info("No model path provided, skipping chat template application.")
        logging.info(f"Use the raw data file: {data_file}")


if __name__ == "__main__":
    main()
