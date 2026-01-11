#! /bin/bash

set -exo pipefail
ulimit -n 65535


# export HF_ENDPOINT="https://hf-mirror.com"
# export VLLM_LOGGING_LEVEL="DEBUG"

PROJECT_DIR="."
# BASE_MODEL_PATH="/mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s3"
BASE_MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/MikaEval/.cache/Qwen3-4B-Instruct-2507" # for judge
# DATASET="aime2024@512,aime2025@512,amc2023@32,math500@8,minerva@8,hmmt2025@32"
DATASET="aime2024@2" # debug

CACHE_DIR="${PROJECT_DIR}/.cache"
# Cache directory for benchmark datasets (optional)
# If specified, datasets will be loaded from subfolders like CACHE_DIR/aime_2024/, aime_2025/, etc.

TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"
DP_SIZE=8
TP_SIZE=1
MAX_NUM_REQUEST=2000
GPU_MEMORY_UTILIZATION=0.95
DTYPE="bfloat16"
SERVE_PORT=8000
MODE="llm-eval" # infer, eval or all

function kill_vllm_processes() {
  pkill -9 python || true;
  pkill -9 -f "vllm.entrypoints.openai.api_server" || true;
  pkill -9 -f "VLLM::EngineCore" || true;
  sleep 1;
  pkill -9 python || true;
  pkill -9 -f "vllm.entrypoints.openai.api_server" || true;
  pkill -9 -f "VLLM::EngineCore" || true;
}

function eval_model_with_adapter() {
  kill_vllm_processes;
  
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
    ${CACHE_DIR:+--cache-dir "${CACHE_DIR}"} 2>&1 | tee "${RESULT_DIR}/eval.log";
}

set +e

eval_model_with_adapter \
  "${PROJECT_DIR}/outputs/debug" \
  "${BASE_MODEL_PATH}" \
  ""