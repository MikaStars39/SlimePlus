# 如何评测

评测总体分为3个步骤：
 - 准备评测所需的数据集（如果不能联网下载的话，已经cache了现在支持的，可以暂时不管这一步）
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
超参数，功能都已经做了相应的注释。例如，我们想要评测一个路径为`/mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s3`的模型，我们要做的事是：
- 复制一份 `examples/infer.sh` 并且改成你自己的名字
- `BASE_MODEL_PATH`改为这个路径
- `OUTPUT_DIR`写一个你想要储存的路径
然后直接run这个就可以了，比如`bash scripts/coldstart_qwen_30b/infer_qwen3-30b-s1-0103.sh`
这样就会自动推理出每个问题的回答

然后我们需要做judge，也就是看每个答案对不对，同样复制一份`examples/llm_based_eval.sh`，然后按照之前的改法改好对应的路径，注意`OUTPUT_DIR`需要一样，因为judge的时候会去这个下面找推理的结果

judge完以后，只需要`python view.py [你的output dir]`就可以自动生成如下类似的markdown表格：

```
| Dataset | Count | Accuracy (Pass@1) | Pass@Max | Format Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| aime2024 | 30 | 34.48% | 53.33% | 100.00% |
| aime2025 | 30 | 25.31% | 40.00% | 100.00% |
| amc2023 | 40 | 73.59% | 85.00% | 100.00% |
| hmmt2025 | 30 | 13.33% | 33.33% | 100.00% |
| math500 | 500 | 86.10% | 87.40% | 100.00% |
| minerva | 272 | 33.46% | 33.46% | 100.00% |
| **Average** | **902** | **63.51%** | **66.52%** | **100.00%** |
```


/mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s3
| Dataset | Count | Accuracy (Pass@1) | Pass@Max | Format Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| aime2024 | 30 | 34.48% | 53.33% | 100.00% |
| aime2025 | 30 | 25.31% | 40.00% | 100.00% |
| amc2023 | 40 | 73.59% | 85.00% | 100.00% |
| hmmt2025 | 30 | 13.33% | 33.33% | 100.00% |
| math500 | 500 | 86.10% | 87.40% | 100.00% |
| minerva | 272 | 33.46% | 33.46% | 100.00% |
| **Average** | **902** | **63.51%** | **66.52%** | **100.00%** |

/mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s1-0103
| Dataset | Count | Accuracy (Pass@1) | Pass@Max | Format Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| aime2024 | 30 | 50.31% | 76.67% | 100.00% |
| aime2025 | 30 | 37.19% | 46.67% | 100.00% |
| amc2023 | 40 | 82.81% | 85.00% | 100.00% |
| hmmt2025 | 30 | 23.96% | 40.00% | 100.00% |
| math500 | 500 | 92.35% | 94.40% | 100.00% |
| minerva | 272 | 38.60% | 40.81% | 100.00% |
| **Average** | **902** | **70.21%** | **73.84%** | **100.00%** |

/mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s1-0103-1
| Dataset | Count | Accuracy (Pass@1) | Pass@Max | Format Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| aime2024 | 30 | 54.37% | 70.00% | 100.00% |
| aime2025 | 30 | 31.15% | 46.67% | 100.00% |
| amc2023 | 40 | 83.20% | 97.50% | 100.00% |
| hmmt2025 | 30 | 24.06% | 40.00% | 100.00% |
| math500 | 500 | 91.95% | 95.00% | 100.00% |
| minerva | 272 | 38.42% | 39.71% | 100.00% |
| **Average** | **902** | **69.89%** | **74.17%** | **100.00%** |

/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/20260110_173129_gspo_qwen30ba3b/iter_0000223_hf
| Dataset | Count | Accuracy (Pass@1) | Pass@Max | Format Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| aime2024 | 30 | 52.71% | 70.00% | 100.00% |
| aime2025 | 30 | 35.73% | 46.67% | 100.00% |
| amc2023 | 40 | 78.36% | 85.00% | 100.00% |
| hmmt2025 | 30 | 23.75% | 43.33% | 100.00% |
| math500 | 500 | 89.70% | 91.80% | 100.00% |
| minerva | 272 | 39.25% | 39.71% | 99.63% |
| **Average** | **902** | **68.76%** | **71.95%** | **99.89%** |


