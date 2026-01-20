# 如何支持新数据集 (How to Support a New Dataset)

本文档介绍了如何在 MikaEval 中添加对新评测数据集的支持。主要涉及三个步骤：注册数据集、实现加载器、以及导出加载器。

## 步骤 1: 在 `base.py` 中注册数据集

打开 `MikaEval/src/tasks/base.py`，在 `DATASETS` 字典中添加你的数据集配置。

```python
DATASETS = {
    # ... 现有数据集 ...
    "your_dataset_name": {
        "hf_name": "huggingface_repo/name",  # HuggingFace 数据集路径或本地路径
        "split": "test",                    # 数据集划分 (如 test, train)
        "custom_args": [],                  # 可选：自定义参数
        "need_llm_extract": False,           # 是否需要 LLM 提取答案 (通常数学题为 False)
        "eval_type": "math",                # 评测类型 (目前支持 "math" 和 "ifeval")
    },
}
```

## 步骤 2: 创建加载器文件

在 `MikaEval/src/tasks/` 目录下创建一个新的 Python 文件（例如 `your_dataset_name.py`），并实现加载逻辑。建议参考 `MikaEval/src/tasks/math500.py`。

核心任务是实现一个加载函数，其命名必须遵循 `load_{dataset_name}` 的格式（其中 `dataset_name` 是你在 `base.py` 中定义的键，且连字符 `-` 会被自动替换为下划线 `_`）。该函数负责将原始数据转换为统一的 JSONL 格式。

```python
from tqdm import tqdm
import json
from io import TextIOWrapper
from mika_eval.tasks.base import DATASETS, get_question_text, get_answer_text, load_dataset_from_hf

def load_your_dataset_name(
    dataset_name: str,
    cache_dir: str,
    k: int,
    f_out: TextIOWrapper,
):
    # 1. 加载原始数据集
    dataset = load_dataset_from_hf(dataset_name, cache_dir)
    
    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}")):
        # 2. 提取问题和答案 (可以使用 base.py 中的辅助函数)
        question = get_question_text(row)
        answer = get_answer_text(row)

        # 3. 处理 Pass@k (重复写入 k 次)
        for sample_idx in range(k):
            unique_id = f"{dataset_name}_{idx}_{sample_idx}"
            
            record = {
                "id": unique_id,
                "question_id": f"{dataset_name}_{idx}",
                "source": dataset_name,
                "prompt": question,  # 必须使用 'prompt' 键
                "sample_index": sample_idx,
                "need_llm_extract": DATASETS[dataset_name]["need_llm_extract"],
                "label": answer,
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
```

## 步骤 3: 在 `__init__.py` 中导出

打开 `MikaEval/src/tasks/__init__.py`，导入你新创建的加载函数并将其添加到 `__all__` 列表中。

```python
# ... 其它导入 ...
from mika_eval.tasks.your_dataset_name import load_your_dataset_name

__all__ = [
    # ... 其它函数 ...
    "load_your_dataset_name",
]
```

## 步骤 4 (可选): 更新评测逻辑

如果你的数据集不属于 "math" 或 "ifeval" 类型，可能需要修改 `MikaEval/src/reward/reward.py` 中的 `judge_router` 函数，以支持新的判分逻辑。

```python
def judge_router(instance: Dict) -> Dict:
    source = instance.get("source", "unknown").lower()
    if "ifeval" in source:
        return if_judge(instance)
    elif "your_special_dataset" in source:
        return your_custom_judge(instance)
    else:
        return math_judge(instance)
```

## 测试

完成后，你可以尝试运行数据准备脚本来验证：

```bash
python -m mika_eval.tasks.base --config "your_dataset_name@1" --output_file "test_data.jsonl"
```
