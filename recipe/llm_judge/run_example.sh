#!/bin/bash
# LLM Judge Scoring System - Example Run Script

# Configuration paths
INPUT_FILE="/mnt/llm-train/users/explore-train/wangzhenfang8/codes/generate/data/used_for_dpo_obj/0131-v5/used_for_dpo_obj.jsonl"
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/dpo"
JUDGE_MODEL="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8"

# Run complete pipeline
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/run_judge_pipeline.py \
    --input "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --judge-model "${JUDGE_MODEL}"

pip install -e /mnt/llm-train/users/explore-train/qingyu/slimulation --no-deps -i https://mirrors.jd.com/pypi/web/simple

python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/inference.py \
    --input "/mnt/llm-train/users/explore-train/qingyu/data/dpo/shards_16/shard_0.jsonl" \
    --output "/mnt/llm-train/users/explore-train/qingyu/data/dpo/responses/response_0.jsonl" \
    --model_path "/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507" \
    --tp_size 1 \
    --dp_size 8 \
    --max_tokens 32768、

python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/shard_jsonl.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/dpo/judge_prepared.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/dpo/shards_24 \
    --num-shards 24 \
    --num-readers 48

# 1. 截取 100 条
head -n 100 /mnt/llm-train/users/explore-train/qingyu/data/dpo/shards_16/shard_0.jsonl > shard_test_small.jsonl

# 2. 运行推理测试
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/inference.py \
    --input shard_test_small.jsonl \
    --output shard_test_small_res.jsonl \
    --model_path "/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507" \
    --tp_size 2 \
    --dp_size 4 \
    --max_concurrency 128 \
    --max_tokens 2048

#!/bin/bash

# 1. 定义 Pod 列表 (按顺序排列，共16个)
pods=(
    "dpo-data-0-6vd8m"
    "dpo-data-1-qpcgs"
    "dpo-data-2-sk8rp"
    "dpo-data-3-gj6wd"
    "dpo-data-4-hl59h"
    "dpo-data-5-rp6nz"
    "dpo-data-6-blg5d"
    "dpo-data-7-ch5qw"
    "dpo-data-8-8cgfz"
    "dpo-data-9-9f6vs"
    "dpo-data-10-q9vm6"
    "dpo-data-11-64mbg"
    "dpo-data-12-bprcz"
    "dpo-data-13-92rkn"
    "dpo-data-14-zm5dl"
    "dpo-data-15-j4hzj"
    "dpo-data-16-4fhn9"
    "dpo-data-17-gwkkg"
    "dpo-data-18-25k2x"
    "dpo-data-19-q92sh"
    "dpo-data-20-2rtp6"
    "dpo-data-21-6pmsb"
    "dpo-data-22-t45rt"
    "dpo-data-23-jg5m5"
)

# for pod in "${pods[@]}"; do echo "Cleaning $pod..."; kt exec $pod -- pkill -9 -f sglang || echo "No sglang process found on $pod"; done

# 2. 循环启动任务
for i in "${!pods[@]}"; do
    pod_name=${pods[$i]}
    shard_id=$i
    
    echo "正在为 Pod [$pod_name] 分配任务: Shard $shard_id ..."

    # 使用 nohup 或后台运行模式执行，避免 kubectl 断开导致任务终止
    kt exec $pod_name -- bash -c "pip install -e /mnt/llm-train/users/explore-train/qingyu/slimulation --no-deps -i https://mirrors.jd.com/pypi/web/simple && \
        python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/inference.py \
        --input \"/mnt/llm-train/users/explore-train/qingyu/data/dpo/shards_16/shard_${shard_id}.jsonl\" \
        --output \"/mnt/llm-train/users/explore-train/qingyu/data/dpo/responses/response_${shard_id}.jsonl\" \
        --model_path \"/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507\" \
        --tp_size 1 \
        --dp_size 8 \
        --max_concurrency 1024 \
        --max_tokens 32768
    " > "log_shard_${shard_id}.log" 2>&1 &

    # 稍微等一秒，避免同时并发请求 kubectl API 过载
    sleep 1
done

echo "所有任务已在后台提交，请通过 'jobs' 命令或查看 log 文件监控进度。"