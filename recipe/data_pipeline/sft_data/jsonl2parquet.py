from datasets import load_dataset
import argparse

def convert_jsonl_to_parquet(
    jsonl_file_name: str,
    parquet_file_name: str,
):
    ds = load_dataset("json", data_files=jsonl_file_name)

    def convert(sample):
        conversations = sample["conversations"]

        def convert_role(role):
            if role == "human":
                return "user"
            elif role == "gpt":
                return "assistant"
            elif role == "system":
                return "system"
            else:
                raise ValueError(f"Unknown role: {role}")

        messages = [
            {
                "role": convert_role(turn["from"]),
                "content": turn["value"],
            }
            for turn in conversations
        ]

        return {"messages": messages}

    ds = ds.map(convert, num_proc=16)
    ds = ds.remove_columns([col for col in ds.column_names['train'] if col != 'messages'])
    ds['train'].to_parquet(parquet_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jsonl2parquet")
    parser.add_argument("--jsonl-file-name", required=True)
    parser.add_argument("--parquet-file-name", required=True)

    args = parser.parse_args()

    convert_jsonl_to_parquet(
        jsonl_file_name=args.jsonl_file_name,
        parquet_file_name=args.parquet_file_name,
    )

"""
python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/data_pipeline/sft_data/jsonl2parquet.py \
    --jsonl-file-name /mnt/llm-train/users/explore-train/wangzhenfang8/datasets/MCP/0103/exp7_0324gen_filtered_less_4096_using_23w_2ep_joyai_1w.jsonl \
    --parquet-file-name /mnt/llm-train/users/explore-train/qingyu/data/sft/test_1w.parquet
"""