# 如何评测

评测总体分为3个步骤：
 - 准备评测所需的数据集（如果不能联网下载的话）
 - 设置推理参数，使用vllm 推理出每个题目对应的答案
 - 设置评估参数，使用rule-based提取或者使用llm提取出对应答案，并计算相应指标

所有任务都可以通过shell脚本来实现，全程只需要`bash xxx`就可以

examples里有全部的对应脚本，可以对应学习

## 熟悉脚本

脚本的结构大致如下：
```bash
#! /bin/bash

set -exo pipefail
ulimit -n 65535

# export HF_ENDPOINT="https://hf-mirror.com"
# export VLLM_LOGGING_LEVEL="DEBUG"

PROJECT_DIR="." 
BASE_MODEL_PATH="/mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s3"
# BASE_MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/MikaEval/.cache/Qwen3-4B-Instruct-2507" # for judge

DATASET="aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
# DATASET="aime2024@2" # debug

CACHE_DIR="${PROJECT_DIR}/.cache"
# Cache directory for benchmark datasets (optional)
# If specified, datasets will be loaded from subfolders like CACHE_DIR/aime_2024/, aime_2025/, etc.

TEMPERATURE="0.7" # temperature
TOP_P="0.9" # top p
MAX_NEW_TOKENS="31744" # how many new tokens w/o prompts
DP_SIZE=8 # dp, vllm
TP_SIZE=1 # tp, vllm
MAX_NUM_REQUEST=2000 # how many new requests can make
GPU_MEMORY_UTILIZATION=0.95
DTYPE="bfloat16"
SERVE_PORT=8000
MODE="infer" # infer, rule-eval, llm-eval
```
超参数，功能都已经做了相应的注释

## 准备评测所需的数据集

