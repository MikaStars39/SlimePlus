python recipe/data_pipeline/llm_process_prompt/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_deduped.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507

python recipe/data_pipeline/llm_process_prompt/inference.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_pre.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_infer.jsonl \
    --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507 \
    --tp_size 1 \
    --dp_size 8 \
    --max_concurrency 1024 \
    --max_tokens 2048

python recipe/data_pipeline/llm_process_prompt/postprocess.py \
    --original /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_deduped.jsonl \
    --response /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_post.jsonl \
    --failed /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_failed.jsonl

head -n 100 /mnt/llm-train/users/explore-train/qingyu/data/stage_1/KlearReasoner-MathSub-30K/raw_process_prompt_post.jsonl > /mnt/llm-train/users/explore-train/qingyu/data/temp.jsonl