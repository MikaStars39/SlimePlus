# LLM Judge System Instructions

## 📋 System Overview

This is a multi-dimensional LLM automated scoring system, modeled after `4_instruct_llm_do_first`, used to evaluate the quality of multiple model outputs.

### Comparison with 4_instruct_llm_do_first

| Feature | 4_instruct_llm_do_first | llm_judge |
|------|-------------------------|-----------|
| **Purpose** | Generate multiple rollouts and judge correctness | Multi-dimensional scoring for multiple outputs |
| **Input** | Single Q&A data | Q&A data + 8 model outputs |
| **Process** | Expand → Inference → Judge → Pass@k Stats | Prepare → Inference → Extract Scores |
| **Output** | CORRECT/INCORRECT | Detailed 0-100 scoring |
| **Dimensions** | Single correctness | 4 dimensions (Correctness, Logic, Clarity, Completeness) |

## 🎯 Core Features

### Scoring Dimensions (Total 100 points)

1. **Correctness (50 points)** - Most important
   - Is the final answer correct?
   - Are key steps accurate?
   - Is the calculation result correct?

2. **Logic (25 points)**
   - Is the reasoning process coherent?
   - Is each derivation step reasonable?
   - Are there any logical errors?

3. **Clarity (15 points)**
   - Is the expression clear?
   - Are steps easy to understand?
   - Is the format standardized?

4. **Completeness (10 points)**
   - Is the question fully answered?
   - Are any steps missing?
   - Is the answer explicit?

## 📁 File Structure

```
llm_judge/
├── README.md                    # Main documentation (Chinese)
├── QUICKSTART.md                # Quick start guide
├── Instructions.md              # This file
├── prepare_judge.py             # Step 1: Prepare evaluation data
├── inference.py                 # Step 2: LLM inference
├── extract_scores.py            # Step 3: Extract scores
├── run_judge_pipeline.py        # One-click run for complete pipeline
├── analyze_scores.py            # Analyze scoring results
├── run_example.sh               # Example run script
└── test_data_example.jsonl      # Sample test data
```

## 🚀 Quick Usage

### Method 1: One-Click Run (Recommended)

```bash
cd /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge

python run_judge_pipeline.py \
    --input /path/to/your/data.jsonl \
    --output-dir /path/to/output \
    --judge-model /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8
```

### Method 2: Use Example Script

1. Edit `run_example.sh`, modify input/output paths.
2. Run: `./run_example.sh`

### Method 3: Step-by-Step Execution

```bash
# Step 1: Prepare data
python prepare_judge.py \
    --input /mnt/llm-train/users/explore-train/wangzhenfang8/codes/generate/data/used_for_dpo_obj/0131-v5/used_for_dpo_obj.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/dpo/prepared.jsonl \
    --tokenizer /path/to/model

# Step 2: LLM inference
python inference.py \
    --input prepared.jsonl \
    --output inference.jsonl \
    --model_path /path/to/model

# Step 3: Extract scores
python extract_scores.py \
    --prepared prepared.jsonl \
    --response inference.jsonl \
    --output scores.jsonl \
    --failed failed.jsonl
```

## 📊 View Results

### View Scoring File

```bash
# View the first scoring result
cat judge_scores.jsonl | head -n 1 | jq .

# Count total distribution
cat judge_scores.jsonl | jq '.scores[].total_score' | sort -n
```

### Generate Analysis Report

```bash
python analyze_scores.py --scores judge_scores.jsonl
```

Output example:
```
================================================================================
LLM Scoring Results Analysis Report
================================================================================

Total samples: 100
Total evaluations: 800

--------------------------------------------------------------------------------
Model Output Ranking (by Mean Total Score)
--------------------------------------------------------------------------------
 1. JoyAI_output_2      - Mean Score:  87.50
 2. JoyAI_output_0      - Mean Score:  85.30
 3. JoyAI_output_5      - Mean Score:  83.20
...
```

### Export to CSV

```bash
python analyze_scores.py \
    --scores judge_scores.jsonl \
    --export-csv output.csv
```

Further analysis in Excel or pandas:
```python
import pandas as pd
df = pd.read_csv('output.csv')
print(df.groupby('output_name')['total_score'].describe())
```

## 📝 Input Data Format

Input JSONL file, one JSON object per line:

```json
{
    "conversations": [
        {"from": "system", "value": "System prompt"},
        {"from": "human", "value": "Question content"},
        {"from": "gpt", "value": "Reference answer"}
    ],
    "JoyAI_output_0": "Model output 1",
    "JoyAI_output_1": "Model output 2",
    "JoyAI_output_2": "Model output 3",
    "JoyAI_output_3": "Model output 4",
    "JoyAI_output_4": "Model output 5",
    "JoyAI_output_5": "Model output 6",
    "JoyAI_output_6": "Model output 7",
    "JoyAI_output_7": "Model output 8"
}
```

**Notes:**
- The `conversations` field must contain the question and reference answer.
- `JoyAI_output_*` fields are model outputs to be scored.
- It can contain only partial outputs (e.g., only 0-3), the system will handle it automatically.

## 📤 Output Data Format

Scoring result JSONL file, one JSON object per line:

```json
{
    "original_idx": 0,
    "question": "Calculate 2 + 2?",
    "reference_answer": "2 + 2 = 4",
    "scores": {
        "JoyAI_output_0": {
            "correctness": 48,
            "logic": 23,
            "clarity": 14,
            "completeness": 9,
            "total_score": 94,
            "brief_comment": "Answer is correct, expression is clear, logic is complete."
        },
        "JoyAI_output_1": {
            "correctness": 45,
            "logic": 20,
            "clarity": 12,
            "completeness": 8,
            "total_score": 85,
            "brief_comment": "Answer is correct but lacks detailed steps."
        }
    }
}
```

## ⚙️ Configuration Adjustments

### Modify Scoring Criteria

Edit `JUDGE_SYSTEM_PROMPT` in `prepare_judge.py`:

```python
JUDGE_SYSTEM_PROMPT = """
### Scoring Criteria
Please score from the following four dimensions (total 100 points):

1. **Correctness (50 points)** - Most important dimension
   [Modify scoring details here]

2. **Logic (25 points)**
   [Modify scoring details here]
...
"""
```

### Adjust Inference Parameters

Edit the `step2_inference` method in `run_judge_pipeline.py`:

```python
cmd = [
    ...
    '--tp_size', '2',              # Tensor Parallel
    '--dp_size', '4',              # Data Parallel
    '--max_concurrency', '512',    # Concurrency
    '--max_tokens', '2048',        # Max tokens
    '--temp', '0.3',               # Temperature (lower is more stable)
]
```

### Score Only Specific Outputs

Modify line 98 of `prepare_judge.py`:

```python
# Original code: score all 8 outputs
for output_idx in range(8):

# Modified: score only specific outputs
for output_idx in [0, 2, 4, 6]:  # Score only even indices
```

## 🔍 FAQ

### Q1: High failure rate in scoring

**Causes:**
- Judge model failed to follow output format.
- Prompt is not clear enough.
- Temperature parameter is too high.

**Solutions:**
1. Check `judge_failed.jsonl` for failure cases.
2. Lower the temperature parameter (e.g., to 0.1-0.3).
3. Adjust prompt format.
4. Use a stronger judge model.

### Q2: Slow scoring speed

**Causes:**
- Insufficient parallelism.
- GPU resources not fully utilized.

**Solutions:**
1. Increase `dp_size` (requires more GPUs).
2. Increase `max_concurrency`.
3. Decrease `tp_size` (if the model is not too large).

### Q3: Out of VRAM

**Causes:**
- Model is too large.
- Parallelism is too high.

**Solutions:**
1. Decrease `tp_size` and `dp_size`.
2. Decrease `max_concurrency`.
3. Use FP8 quantized models.

### Q4: Unreasonable scoring results

**Causes:**
- Judge model capability is insufficient.
- Scoring criteria are not clear.
- Reference answer quality issues.

**Solutions:**
1. Use a stronger judge model (30B+ recommended).
2. Optimize scoring criteria descriptions.
3. Check reference answer quality.
4. Adjust weights of each dimension.

## 💡 Advanced Usage

### Custom Scoring Dimensions

Add new dimensions in `prepare_judge.py`, and update the validation function in `extract_scores.py`.

### Batch Processing Multiple Files

```bash
for file in /data/input/*.jsonl; do
    basename=$(basename "$file" .jsonl)
    python run_judge_pipeline.py \
        --input "$file" \
        --output-dir "/data/output/${basename}"
done
```

### Integration with Data Filtering Pipeline

```python
import json

# Keep only high-score outputs
threshold = 80
with open('judge_scores.jsonl') as f_in, \
     open('high_quality.jsonl', 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        for output_name, scores in data['scores'].items():
            if scores['total_score'] >= threshold:
                # Save high-quality output
                ...
```

## 📈 Performance Reference

Based on tests with Qwen3-30B-A3B-Thinking model:

- **Processing Speed**: Approx. 100-200 samples/min (8 outputs/sample, 8-card parallel).
- **VRAM Usage**: Approx. 15-20GB/card (FP8 quantization).
- **Scoring Success Rate**: 95%+ (Temperature=0.3).
- **Scoring Consistency**: Error < 5 points for repeated scoring of the same sample.

## 🛠️ Testing the System

Using provided test data:

```bash
python run_judge_pipeline.py \
    --input test_data_example.jsonl \
    --output-dir test_output \
    --judge-model /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8

# View results
python analyze_scores.py --scores test_output/judge_scores.jsonl
```

## 📚 More Resources

- **Main Documentation**: See `README.md` (Chinese).
- **Quick Start**: See `QUICKSTART.md`.
- **Test Data**: See `test_data_example.jsonl`.
- **Example Script**: See `run_example.sh`.

## 🤝 Contribution

If you have questions or suggestions, feel free to give feedback!

## 📄 License

Same as the main project.
