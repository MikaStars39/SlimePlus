#!/bin/bash
# Complete evaluation pipeline example
# Usage: bash run.sh

# MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/sft-s1-lr1e-5-DECAY_SAMPLES-0212_1456"
# MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/Qwen3-8B-ODA-Math-460k_opd/iter_0000059_hf"
# MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/Qwen3-8B-Base_opd_sft_rl/iter_0000019_hf"
# MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/20260128_084356_opd_40b_intellect_teach_temp_1_5/iter_0000135_hf"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

PROJECT_DIR=/mnt/llm-train/users/explore-train/qingyu/slime

PYTHONPATH=/root/Megatron-LM python \
    ${PROJECT_DIR}/tools/convert_torch_dist_to_hf.py \
    --input-dir /mnt/llm-train/users/explore-train/qingyu/ckpt/20260131_070627_self_distillation/iter_0000015 \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/ckpt/20260131_070627_self_distillation/iter_0000015_hf \
    --origin-hf-dir /mnt/llm-train/users/explore-train/wangzhenfang8/hf_outputs/40b/sft-postrain-exp-v5-baselong-e2_lr5e-5_minlr5e-6/13638/ \
    --vocab-size 151936 \
    --chunk-size 10000000000

# MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/20260131_052150_self_distillation/iter_0000031_hf"
MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/20260131_070627_self_distillation/iter_0000015_hf"
CACHE_DIR="/mnt/llm-train/users/explore-train/qingyu/.cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/eval_outputs/${TIMESTAMP}_intellect_iter_0000015"

# Step 1: Prepare data (load benchmarks and apply chat template)
python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/prepare_data.py \
    --dataset "aime2024@8,aime2025@8,math500@4" \
    --cache-dir "$CACHE_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --prompt-format "lighteval" \
    --system-prompt "You are JoyAI, a large language model trained by JD (京东). Answer as concisely as possible."

# Step 2: Run batch inference
python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/inference.py \
    --input "$OUTPUT_DIR/data.chat.jsonl" \
    --output "$OUTPUT_DIR/results.jsonl" \
    --model "$MODEL_PATH" \
    --tp-size 1 \
    --dp-size 8 \
    --temperature 0.7 \
    --top-p 0.8 \
    --max-tokens 32768 \
    --resume
    
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/eval_outputs/20260131_065729_8b_opt_sft"

# Step 3: Evaluate and calculate metrics
python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/evaluate.py \
    --input "$OUTPUT_DIR/results.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --num-proc 32
