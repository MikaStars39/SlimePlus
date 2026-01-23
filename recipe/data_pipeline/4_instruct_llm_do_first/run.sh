# dapo
python recipe/data_pipeline/4_instruct_llm_do_first/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_relabel_post.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_pre.jsonl \
    --k 4 \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/4_instruct_llm_do_first/llm_judge_prepare.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_judge_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_judge_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_judge_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 8192

python recipe/data_pipeline/4_instruct_llm_do_first/llm_judge_extract.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_pre.jsonl \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_judge_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_judge_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_judge_failed.jsonl

python recipe/data_pipeline/4_instruct_llm_do_first/calculate_pass_at_k.py \
    --rollout /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-stem/raw_instruct_filter_judge_post.jsonl \
    --reference /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_post.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-stem/raw_instruct_filter_pass_at_k.jsonl

python recipe/data_pipeline/4_instruct_llm_do_first/filter_by_pass_at_k.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_pass_at_k.jsonl \
    --threshold 0.51 \
    --output-high /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_4b_easy.jsonl \
    --output-low /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_4b_hard.jsonl
    