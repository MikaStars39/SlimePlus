python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Science \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/MegaScience-stem \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/Polaris-Dataset-53K \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/STILL-3-Preview-RL-Data \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-math \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507

python recipe/data_pipeline/4_instruct_llm_do_first/run_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem/raw_relabel_post.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/stage_1/WebInstruct-verified-stem \
    --k 10 \
    --threshold 0.51 \
    --qwen-4b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507 \
    --qwen-30b /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507