

### 使用运行时命令配置

你可以根据需要使用以下命令启动特定的步骤：

| 想要运行的步骤 | 对应的命令行参数 |
| --- | --- |
| **步骤 1 (准备数据)** | `python main.py --mode prepare --result-dir ./res --model ...` |
| **步骤 2 (模型推理)** | `python main.py --mode infer --result-dir ./res --model ...` |
| **步骤 3 (LLM 提取)** | `python main.py --mode llm-eval --result-dir ./res --model ...` |
| **步骤 4 (计算指标)** | `python main.py --mode metrics --result-dir ./res --model ...` |


### ⚠️ 关键注意事项

由于这四个步骤是**链式依赖**的，如果你单独运行某一步，必须确保上一步生成的产出文件已经存在于 `--result-dir` 目录中：

1. **运行 Step 2 前**：目录下必须有 `data.chat.jsonl`。
2. **运行 Step 3 前**：目录下必须有 `inference_results.jsonl`。
3. **运行 Step 4 前**：目录下必须有 `eval_results.jsonl`。

**代码中的保护机制：**
你现在的代码里其实已经有一部分“跳过”逻辑（例如 `if output_file.exists(): return`），这意味着即使你运行 `--mode all`，如果之前已经跑完了前两步，它会自动跳过那些已经产生结果文件的步骤。