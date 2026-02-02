#! /bin/bash

set -exo pipefail
ulimit -n 65535

export HF_ENDPOINT="https://hf-mirror.com"
export NLTK_DATA="." # can be empty

BASE_MODEL_PATH="Your Model Path"
EVAL_MODEL_PATH="Your Eval Model Path" 
DATASET="null@1"
CACHE_DIR="/mnt/llm-train/users/explore-train/qingyu/.cache"
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/KlearReasoner" # Output
MODE="infer" # Evaluation Mode: "all" runs data prep, inference, llm-extraction, and metrics calculation
DP_SIZE=8
TP_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
MAX_CONCURRENCY=256
MAX_NEW_TOKENS="30000"
TEMPERATURE="0.7"
TOP_P="0.9"
EVAL_MAX_NEW_TOKENS=1024
PROPMT_FORMAT="auto"

mkdir -p "${OUTPUT_DIR}"

python eval.py \
  --mode "${MODE}" \
  --result-dir "${OUTPUT_DIR}" \
  --model "${BASE_MODEL_PATH}" \
  --eval-model "${EVAL_MODEL_PATH}" \
  --dataset "${DATASET}" \
  --cache-dir "${CACHE_DIR}" \
  --dp-size "${DP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --prompt-format "${PROPMT_FORMAT}" \
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"
