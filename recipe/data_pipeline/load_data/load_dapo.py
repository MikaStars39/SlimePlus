from tqdm import tqdm
import json
from io import TextIOWrapper
import os
from datasets import load_dataset

def load_klearreasoner(
    dataset_name: str,
    k: int,
    f_out: TextIOWrapper,
):
    """Load the AIME 2024 dataset."""
    dataset = load_dataset(dataset_name, split="train",)
    
    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}")):
        
        question = row["prompt"]
        row.pop("prompt")
        answer = row["solution"]
        row.pop("solution")

        for sample_idx in range(k):
            # Create a unique ID for each attempt
            # Format: {dataset}_{original_index}_{attempt_index}
            unique_id = f"DAPO-Math-17k-Processed_{idx}_{sample_idx}"
            
            record = {
                "id": unique_id,
                "question_id": f"DAPO-Math-17k-Processed_{idx}",
                "source": "DAPO-Math-17k-Processed",
                "prompt": question,
                "sample_index": sample_idx,
                "need_llm_extract": True,
                "label": answer,
                "user_template_type": "blank",
                "system_prompt_type": None,
                **row
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    output_dir = "/mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed"

    # first make dictionary
    os.makedirs(output_dir, exist_ok=True)
    
    load_klearreasoner(
        dataset_name="/mnt/llm-train/users/explore-train/qingyu/.cache/DAPO-Math-17k-Processed",
        k=1,
        f_out=open(os.path.join(output_dir, "data.jsonl"), "w"),
    )