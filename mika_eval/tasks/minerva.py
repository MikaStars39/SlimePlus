from datasets import load_dataset
from tqdm import tqdm
import json
from io import TextIOWrapper

from mika_eval.tasks.base import DATASETS, get_question_text, get_answer_text, load_dataset_from_hf


def load_minerva(
    dataset_name: str,
    cache_dir: str,
    k: int,
    f_out: TextIOWrapper,
):
    dataset = load_dataset_from_hf(dataset_name, cache_dir)
    
    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}")):
        
        question = get_question_text(row)
        answer = get_answer_text(row)

        for sample_idx in range(k):
            # Create a unique ID for each attempt
            # Format: {dataset}_{original_index}_{attempt_index}
            unique_id = f"{dataset_name}_{idx}_{sample_idx}"
            
            record = {
                "id": unique_id,
                "question_id": f"{dataset_name}_{idx}",
                "source": dataset_name,
                "prompt": question,  # 'prompt' key matches the offline engine script
                "sample_index": sample_idx,
                "need_llm_extract": DATASETS[dataset_name]["need_llm_extract"],
                "label": answer,
                
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
