#! /bin/bash

set -exo pipefail
ulimit -n 65535


export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_DIR="."
BASE_MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/20260110_173129_gspo_qwen30ba3b/iter_0000223_hf"
DATASET="aime2024@32"
CACHE_DIR="${PROJECT_DIR}/.cache"
TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="1024"
DP_SIZE=1
TP_SIZE=1
MAX_NUM_REQUEST=2000
GPU_MEMORY_UTILIZATION=0.9
DTYPE="bfloat16"
SERVE_PORT=8000
MODE="infer" # infer, rule-eval, llm-eval
OUTPUT_DIR="${PROJECT_DIR}/outputs/20260110_173129_gspo_qwen30ba3b_0000223_slime"
PROMPT_FORMAT="slime"

function infer() {
  
  RESULT_DIR="$1" # where to save the results
  MODEL_DIR="$2" # where to load the model
  ADAPTER_DIR="$3" # where to load the adapter e.g., lora

  mkdir -p "${RESULT_DIR}"
  
  python "${PROJECT_DIR}/eval.py" \
    --result-dir "${RESULT_DIR}" \
    --model "${MODEL_DIR}" \
    --adapter "${ADAPTER_DIR}" \
    --dataset "${DATASET}" \
    --serve-port "${SERVE_PORT}" \
    --dp-size "${DP_SIZE}" \
    --tp-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "42" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-num-request "${MAX_NUM_REQUEST}" \
    --dtype "${DTYPE}" \
    --mode "${MODE}" \
    --prompt-format "${PROMPT_FORMAT}" \
    ${CACHE_DIR:+--cache-dir "${CACHE_DIR}"} 2>&1 | tee "${RESULT_DIR}/eval.log";
}

set +e

infer \
  "${OUTPUT_DIR}" \
  "${BASE_MODEL_PATH}" \
  ""