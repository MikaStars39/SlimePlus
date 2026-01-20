import logging
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
import json
from io import TextIOWrapper

from mika_eval.tasks.base import DATASETS, get_question_text, get_answer_text, load_dataset_from_hf

def load_ceval(
    dataset_name: str,
    cache_dir: str,
    k: int,
    f_out: TextIOWrapper,
):
    """
    C-Eval (ceval/ceval-exam) has many subsets (dataset configs). We load the `test`
    split for each subset, then build a multiple-choice prompt with A/B/C/D.
    """

    hf_name = DATASETS[dataset_name]["hf_name"]
    split = DATASETS[dataset_name]["split"]

    def _load_subset(subset_name: str | None):
        # Keep compatibility with this repo's "local cache_dir contains dataset repo" behavior
        if cache_dir is not None:
            cache_dataset_name = hf_name.split("/")[-1]
            cache_path = Path(cache_dir) / cache_dataset_name
            logging.info(f"Cache path: {cache_path}")
            logging.info(f"Cache path exists: {cache_path.exists()}")
            if cache_path.exists():
                try:
                    if subset_name is None:
                        return load_dataset(str(cache_path), split=split)
                    return load_dataset(str(cache_path), subset_name, split=split)
                except Exception as e:
                    logging.info(f"Cache loading failed, falling back to HF: {e}")

        if subset_name is None:
            return load_dataset(hf_name, split=split, cache_dir=cache_dir)
        return load_dataset(hf_name, subset_name, split=split, cache_dir=cache_dir)

    # Try to enumerate all subsets(configs) for ceval; fall back to the default loader if not possible.
    try:
        subset_names = get_dataset_config_names(hf_name)
    except Exception as e:
        logging.info(f"Could not enumerate subsets for {hf_name}: {e}")
        subset_names = []

    if not subset_names:
        # Fallback: treat as a normal single-split dataset.
        dataset = load_dataset_from_hf(dataset_name, cache_dir)
        subset_names = [None]
        datasets_iter = [(None, dataset)]
    else:
        datasets_iter = [(subset, _load_subset(subset)) for subset in subset_names]

    for subset_name, dataset in datasets_iter:
        subset_tag = subset_name if subset_name is not None else dataset_name
        for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}/{subset_tag}")):
            # question + A/B/C/D
            question = row["question"] if "question" in row else get_question_text(row)
            options = []
            for letter in ["A", "B", "C", "D"]:
                if letter in row:
                    options.append(f"{letter}. {row[letter]}")
            if options:
                prompt = f"{question}\n\nOptions:\n" + "\n".join(options)
            else:
                prompt = question

            answer = row["answer"] if "answer" in row else get_answer_text(row)

            for sample_idx in range(k):
                unique_id = f"{dataset_name}_{subset_tag}_{idx}_{sample_idx}"

                record = {
                    "id": unique_id,
                    "question_id": f"{dataset_name}_{subset_tag}_{idx}",
                    "source": dataset_name,
                    "subset": subset_name,
                    "prompt": prompt,  # 'prompt' key matches the offline engine script
                    "sample_index": sample_idx,
                    "need_llm_extract": DATASETS[dataset_name]["need_llm_extract"],
                    "label": str(answer),
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
