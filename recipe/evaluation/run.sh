#!/bin/bash
MODEL_PATH=/mnt/llm-train/users/explore-train/qingyu/.cache/GLM-4.7-Flash-FP8
CACHE_DIR="/mnt/llm-train/users/explore-train/qingyu/.cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/eval_outputs/${TIMESTAMP}_glm_4.7_flash_fp8"

# Step 1: Prepare data (load benchmarks and apply chat template)
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/prepare_data.py \
    --dataset "aime2024@8,aime2025@8,math500@4,gpqa_diamond@4" \
    --cache-dir "$CACHE_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --prompt-format "lighteval"

# Step 2: Run batch inference
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/inference.py \
    --input "$OUTPUT_DIR/data.chat.jsonl" \
    --output "$OUTPUT_DIR/results.jsonl" \
    --model "$MODEL_PATH" \
    --tp-size 1 \
    --dp-size 8 \
    --temperature 0.6 \
    --top-p 1 \
    --max-tokens 32768 \
    --resume

python /mnt/llm-train/users/explore-train/qingyu/slimulation/slimulation/backend/online.py \
    --input "$OUTPUT_DIR/data.chat.jsonl" \
    --output "$OUTPUT_DIR/results.jsonl" \
    --model "xxxx" \
    --api-key "xxxxxxxxx" \
    --base-url "http://6.30.3.162:32607/v1" \
    --temperature 0.6 \
    --top-p 1 \
    --concurrency 1024 \
    --max-tokens 32768

# Step 3: Evaluate and calculate metrics
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/evaluate.py \
    --input "$OUTPUT_DIR/results.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --num-proc 32
