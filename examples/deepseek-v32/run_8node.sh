#!/bin/bash
set -ex

# ---------------------- preprocess ----------------------------
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export FLASHINFER_DISABLE_VERSION_CHECK=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---------------------- env and args ----------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT=/mnt/llm-train/users/explore-train/qingyu/SlimePlus
SLIME_REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/slime_original
PYTHONPATH=/mnt/llm-train/users/explore-train/qingyu/slime_original:/root/Megatron-LM

source "${PROJECT_ROOT}/examples/utils/null_args.sh"

# ---------------------- running arguments ----------------------------

CKPT_ARGS=(
   --hf-checkpoint /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507
)

ROLLOUT_ARGS=(
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
)

SGLANG_ARGS=(
   --rollout-num-gpus 64 # total rollout gpus i.e., nnodes * per_node_gpus

   # --prefill-num-servers 2
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   
   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

   # enable deepep for sglang
   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto

   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 3
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 4

   --sglang-mem-fraction-static 0.7
)

PLUS_ARGS=(
  --plus-input-path /mnt/llm-train/users/explore-train/qingyu/.cache/dapo-math-17k/dapo-math-17k.jsonl
  --plus-output-path /mnt/llm-train/users/explore-train/qingyu/SlimePlus/output/plus_output.jsonl
  --plus-num-workers 4
  --plus-flush-every 1000
  --plus-worker-concurrency 1000
  --plus-worker-batch-size 128
  --plus-sink-flush-size 128
)

# ---------------------- ray runtime ----------------------------

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_DISABLE_VERSION_CHECK\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
  }
}"

# ---------------------- submit job ----------------------------

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${PROJECT_ROOT}/run.py" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PLUS_ARGS[@]}
