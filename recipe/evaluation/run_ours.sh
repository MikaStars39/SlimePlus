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
ORIGINAL_PATH=/mnt/llm-train/users/explore-train/qingyu/ckpt/20260205_055742_table_v2/iter_0000031

PYTHONPATH=/root/Megatron-LM python \
    ${PROJECT_DIR}/tools/convert_torch_dist_to_hf.py \
    --input-dir $ORIGINAL_PATH \
    --output-dir ${ORIGINAL_PATH}_hf \
    --origin-hf-dir /mnt/llm-train/users/explore-train/qingyu/ckpt/sft-16k-v2 \
    --vocab-size 129280 \
    --chunk-size 10000000000

# MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/20260131_052150_self_distillation/iter_0000031_hf"
MODEL_PATH=${ORIGINAL_PATH}_hf
MODEL_PATH=/mnt/llm-train/users/explore-train/qingyu/ckpt/sft-train-exp-s2-v2-bases1v2-e2_lr1e-5_minlr1e-6/736
CACHE_DIR="/mnt/llm-train/users/explore-train/qingyu/.cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/eval_outputs/${TIMESTAMP}_12524"

# Step 1: Prepare data (load benchmarks and apply chat template)
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/prepare_data.py \
    --dataset "math500@4" \
    --cache-dir "$CACHE_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --prompt-format "lighteval" \
    --system-prompt "You are JoyAI, a large language model trained by JD (京东). Answer as concisely as possible."

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

# Step 3: Evaluate and calculate metrics
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/evaluate.py \
    --input "$OUTPUT_DIR/results.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --num-proc 32
