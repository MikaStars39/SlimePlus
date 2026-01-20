import sys
import json
import os
import argparse
from datasets import load_dataset

# ------ Utility Functions --------
def calculate_word_overlap(text1, text2):
    """
    Calculate the ratio of shared words between two strings.
    Using set intersection to determine similarity.
    """
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1:
        return 0.0
    
    intersection = words1.intersection(words2)
    # Ratio relative to the target prompt's word count
    return len(intersection) / len(words1)

# ------ Argument Parsing --------
def parse_args():
    """
    Standard Linux-style CLI argument parsing.
    Exposes parameters for easy automation and piping.
    """
    parser = argparse.ArgumentParser(description="Deduplicate JSONL datasets based on Prompt and Label similarity.")
    parser.add_argument("target", help="Path to the target .jsonl file to clean.")
    parser.add_argument("refs", nargs="+", help="One or more reference .jsonl files to check against.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Word overlap threshold (default: 0.5)")
    parser.add_argument("--proc", type=int, default=32, help="Number of CPU processes for mapping (default: 32)")
    return parser.parse_args()

# ------ Core Logic --------
def main():
    args = parse_args()

    # 1. Path Management
    file_dir = os.path.dirname(args.target)
    base_name = os.path.splitext(os.path.basename(args.target))[0]
    cleaned_path = os.path.join(file_dir, f"{base_name}_cleaned.jsonl")
    removed_path = os.path.join(file_dir, f"{base_name}_removed.jsonl")

    # 2. Build Reference Indices
    # strict_prompts: Set for O(1) exact match lookup
    # label_map: Dict mapping labels to list of prompts for similarity check
    strict_prompts = set()
    label_map = {}

    print(f"[*] Building reference index from {len(args.refs)} files...")
    for path in args.refs:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    p, l = data.get('prompt'), data.get('label')
                    if p:
                        strict_prompts.add(p)
                        if l:
                            if l not in label_map:
                                label_map[l] = []
                            label_map[l].append(p)
                except (json.JSONDecodeError, KeyError):
                    continue

    # 3. Load Target Dataset
    dataset = load_dataset("json", data_files=args.target, split="train")

    # 4. Map Function with Combined Logic
    def check_duplicate(example):
        prompt = example.get('prompt', "")
        label = example.get('label')

        # Rule 1: Strict prompt matching
        if prompt in strict_prompts:
            example['is_dup'] = True
            return example

        # Rule 2: Label match + Word overlap > 50%
        if label in label_map:
            # Check against all prompts that share the same label
            for ref_prompt in label_map[label]:
                if calculate_word_overlap(prompt, ref_prompt) > args.threshold:
                    example['is_dup'] = True
                    return example

        example['is_dup'] = False
        return example

    # ------ Processing and IO --------
    print(f"[*] Processing dataset using {args.proc} cores...")
    processed_ds = dataset.map(check_duplicate, num_proc=args.proc)

    # Split dataset based on the helper column
    cleaned_ds = processed_ds.filter(lambda x: not x['is_dup'], num_proc=args.proc)
    removed_ds = processed_ds.filter(lambda x: x['is_dup'], num_proc=args.proc)

    # Export to JSONL (Linux principle: use plain text)
    cleaned_ds.remove_columns(['is_dup']).to_json(cleaned_path, force_ascii=False)
    removed_ds.remove_columns(['is_dup']).to_json(removed_path, force_ascii=False)

    print("-" * 30)
    print(f"Success!")
    print(f"Cleaned: {cleaned_path} ({len(cleaned_ds)} samples)")
    print(f"Removed: {removed_path} ({len(removed_ds)} samples)")
    print("-" * 30)

if __name__ == "__main__":
    main()

"""
python dedup.py target.jsonl ref.jsonl --threshold 0.6 --proc 16
"""