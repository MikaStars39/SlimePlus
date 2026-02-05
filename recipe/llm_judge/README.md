# LLM Judge 评分系统

使用大型语言模型对多个模型输出进行多维度自动评分。

## 功能特点

- **多维度评分**: 从正确性（50分）、逻辑性（25分）、清晰度（15分）、完整性（10分）四个维度评估
- **批量处理**: 支持一次处理多个输出（如JoyAI_output_0-7）
- **结构化输出**: JSON格式的评分结果，便于后续分析
- **错误处理**: 自动记录评分失败的案例

## 文件说明

- `prepare_judge.py`: 准备评估数据，将输入转换为LLM评分格式
- `inference.py`: 批量推理引擎
- `extract_scores.py`: 从LLM响应中提取并聚合评分结果
- `run_judge_pipeline.py`: 一键运行完整流程

## 输入格式

输入JSONL文件，每行包含：

```json
{
    "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "问题内容"},
        {"from": "gpt", "value": "参考答案"}
    ],
    "JoyAI_output_0": "模型输出1",
    "JoyAI_output_1": "模型输出2",
    ...
    "JoyAI_output_7": "模型输出8"
}
```

## 输出格式

输出JSONL文件，每行包含：

```json
{
    "original_idx": 0,
    "question": "问题内容",
    "reference_answer": "参考答案",
    "scores": {
        "JoyAI_output_0": {
            "correctness": 45,
            "logic": 22,
            "clarity": 13,
            "completeness": 9,
            "total_score": 89,
            "brief_comment": "回答正确且逻辑清晰..."
        },
        "JoyAI_output_1": { ... },
        ...
    }
}
```

## 使用方法

### 方法1: 一键运行（推荐）

```bash
python run_judge_pipeline.py \
    --input /path/to/input.jsonl \
    --output-dir /path/to/output \
    --judge-model /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8
```

### 方法2: 分步运行

#### 步骤1: 准备数据

```bash
python prepare_judge.py \
    --input input.jsonl \
    --output prepared.jsonl \
    --tokenizer /path/to/tokenizer
```

#### 步骤2: LLM推理

```bash
python inference.py \
    --input prepared.jsonl \
    --output inference.jsonl \
    --model_path /path/to/judge/model \
    --tp_size 2 \
    --dp_size 4 \
    --max_tokens 2048
```

#### 步骤3: 提取评分

```bash
python extract_scores.py \
    --prepared prepared.jsonl \
    --response inference.jsonl \
    --output scores.jsonl \
    --failed failed.jsonl
```

## 评分标准说明

### 1. 正确性 (50分)

- **45-50分**: 答案完全正确，所有关键步骤准确无误
- **35-44分**: 答案基本正确，有少量小错误
- **25-34分**: 答案部分正确，有明显错误但思路对
- **15-24分**: 答案错误较多，只有少部分正确
- **0-14分**: 答案基本错误或完全错误

### 2. 逻辑性 (25分)

- **20-25分**: 推理过程严密，逻辑完全正确
- **15-19分**: 推理过程基本正确，有少量跳跃
- **10-14分**: 推理过程部分合理，有明显漏洞
- **5-9分**: 推理过程混乱，逻辑错误较多
- **0-4分**: 没有合理的推理过程

### 3. 清晰度 (15分)

- **12-15分**: 表达清晰，格式规范，易于理解
- **9-11分**: 表达基本清晰，格式尚可
- **6-8分**: 表达有些混乱，但能看懂
- **3-5分**: 表达混乱，难以理解
- **0-2分**: 表达极其混乱

### 4. 完整性 (10分)

- **8-10分**: 完整回答所有问题，步骤齐全
- **6-7分**: 基本完整，缺少少量内容
- **4-5分**: 不够完整，缺失明显
- **2-3分**: 很不完整，缺失严重
- **0-1分**: 几乎没有完整性

## 配置说明

### 推理参数

在 `run_judge_pipeline.py` 中可以调整：

- `tp_size`: Tensor并行大小（默认2）
- `dp_size`: Data并行大小（默认4）
- `max_concurrency`: 最大并发数（默认512）
- `max_tokens`: 最大生成token数（默认2048）
- `temp`: 温度参数（默认0.3，较低温度使评分更稳定）

### 评分提示词

在 `prepare_judge.py` 中的 `JUDGE_SYSTEM_PROMPT` 可以自定义评分标准和提示词。

## 注意事项

1. **输入数据完整性**: 确保每条数据都有 `conversations` 字段和相应的 `JoyAI_output_*` 字段
2. **模型性能**: 建议使用30B以上的模型作为评判模型，以保证评分质量
3. **失败处理**: 检查 `judge_failed.jsonl` 文件了解评分失败的案例
4. **资源需求**: 根据GPU显存调整 `tp_size` 和 `dp_size` 参数

## 示例

使用默认配置运行：

```bash
python run_judge_pipeline.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/temp.jsonl \
    --output-dir /mnt/llm-train/users/explore-train/qingyu/data/judge_output
```

## 故障排查

### 问题1: JSON解析失败

检查输入文件格式是否正确，每行是否是有效的JSON。

### 问题2: 评分提取失败率高

- 检查 `judge_failed.jsonl` 中的失败案例
- 可能需要调整提示词或温度参数
- 确保模型正确加载

### 问题3: 显存不足

- 降低 `tp_size` 和 `dp_size`
- 减少 `max_concurrency`
- 使用较小的模型

## 性能优化

1. **并行度调整**: 根据GPU数量调整 `tp_size` 和 `dp_size`
2. **批量大小**: 调整 `max_concurrency` 平衡速度和显存使用
3. **温度参数**: 评分任务建议使用较低温度（0.1-0.5）

## 输出分析

可以使用以下Python代码分析评分结果：

```python
import json
import numpy as np

scores_by_output = {f"JoyAI_output_{i}": [] for i in range(8)}

with open("judge_scores.jsonl") as f:
    for line in f:
        data = json.loads(line)
        for output_name, score_data in data["scores"].items():
            scores_by_output[output_name].append(score_data["total_score"])

# 计算平均分
for output_name, scores in scores_by_output.items():
    if scores:
        print(f"{output_name}: 平均分 {np.mean(scores):.2f}, 标准差 {np.std(scores):.2f}")
```
