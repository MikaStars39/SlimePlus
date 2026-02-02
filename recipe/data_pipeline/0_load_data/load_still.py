from tqdm import tqdm
import json
from io import TextIOWrapper
import os
from datasets import load_dataset

def load_still(
    dataset_name: str,
    k: int,
    f_out: TextIOWrapper,
):
    """Load the AIME 2024 dataset."""
    dataset = load_dataset(dataset_name, split="train")
    
    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}")):
        
        question = row["question"]
        answer = row["answer"]

        for sample_idx in range(k):
            # Create a unique ID for each attempt
            # Format: {dataset}_{original_index}_{attempt_index}
            unique_id = f"STILL-3-Preview-RL-Data_{idx}_{sample_idx}"
            
            record = {
                "id": unique_id,
                "question_id": f"STILL-3-Preview-RL-Data_{idx}",
                "source": "STILL-3-Preview-RL-Data",
                "prompt": question,
                "sample_index": sample_idx,
                "need_llm_extract": True,
                "label": answer,
                "user_template_type": "blank",
                "system_prompt_type": None,
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    output_dir = "/mnt/llm-train/users/explore-train/qingyu/data/STILL-3-Preview-RL-Data"
    load_still(
        dataset_name="/mnt/llm-train/users/explore-train/qingyu/.cache/STILL-3-Preview-RL-Data",
        k=1,
        f_out=open(os.path.join(output_dir, "data.jsonl"), "w"),
    )