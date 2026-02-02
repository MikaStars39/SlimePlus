# dapo
python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_verifiable_failed.jsonl

# -------------------

python recipe/data_pipeline/5_llm_judge_if_verifiable/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_instruct_filter_4b_hard.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_verifiable_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/5_llm_judge_if_verifiable/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_verifiable_pre.jsonl  \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_verifiable_infer.jsonl  \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 16384

python recipe/data_pipeline/5_llm_judge_if_verifiable/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_instruct_filter_4b_hard.jsonl  \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_verifiable_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_verifiable_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_verifiable_failed.jsonl