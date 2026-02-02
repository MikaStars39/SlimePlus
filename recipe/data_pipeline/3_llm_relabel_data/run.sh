# dapo
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_relabel_failed.jsonl

# INTELLECT-3-RL-Math
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_failed.jsonl

# INTELLECT-3-RL-Science
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_relabel_failed.jsonl

# KlearReasoner-MathSub-30K
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_post.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_post.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_relabel_failed.jsonl

# MegaScience-stem
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_relabel_failed.jsonl

# Polaris-Dataset-53K
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_relabel_failed.jsonl

# STILL-3-Preview-RL-Data
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_relabel_failed.jsonl

# WebInstruct-verified-math
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_failed.jsonl

# WebInstruct-verified-stem
python recipe/data_pipeline/3_llm_relabel_data/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_relabel_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/3_llm_relabel_data/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_relabel_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_relabel_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/3_llm_relabel_data/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_deduped.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_relabel_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_relabel_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_relabel_failed.jsonl