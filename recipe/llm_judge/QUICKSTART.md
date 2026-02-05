# Quick Start Guide

## 5-Minute Setup

### Step 1: Prepare Input File

Ensure your input JSONL file is formatted as follows (one JSON object per line):

```json
{
    "conversations": [
        {"from": "system", "value": "System prompt"},
        {"from": "human", "value": "Question content"},
        {"from": "gpt", "value": "Reference answer"}
    ],
    "JoyAI_output_0": "Model output 1",
    "JoyAI_output_1": "Model output 2",
    ...
    "JoyAI_output_7": "Model output 8"
}
```

### Step 2: Run Scoring

**Method 1: Use Example Script (Easiest)**

Edit the paths in `run_example.sh`, then run:

```bash
./run_example.sh
```

**Method 2: Use Python Command**

```bash
python run_judge_pipeline.py \
    --input /path/to/your/input.jsonl \
    --output-dir /path/to/output \
    --judge-model /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8
```

### Step 3: Check Results

After scoring, the following files will be generated in the output directory:

- `judge_scores.jsonl` - Scoring results (Main file)
- `judge_failed.jsonl` - Records of failed scoring attempts
- `judge_prepared.jsonl` - Intermediate file from the preparation stage
- `judge_inference.jsonl` - Intermediate file from the inference stage

### Step 4: Analyze Results

Use the analysis tool to generate a statistical report:

```bash
python analyze_scores.py --scores /path/to/output/judge_scores.jsonl
```

Export to CSV format:

```bash
python analyze_scores.py \
    --scores /path/to/output/judge_scores.jsonl \
    --export-csv /path/to/output.csv
```

## Complete Example

Assuming your data is at `/data/my_data.jsonl`:

```bash
# 1. Run scoring
python run_judge_pipeline.py \
    --input /data/my_data.jsonl \
    --output-dir /data/judge_results

# 2. View results
cat /data/judge_results/judge_scores.jsonl | head -n 1 | jq .

# 3. Generate analysis report
python analyze_scores.py --scores /data/judge_results/judge_scores.jsonl

# 4. Export CSV
python analyze_scores.py \
    --scores /data/judge_results/judge_scores.jsonl \
    --export-csv /data/judge_results/scores.csv
```

## Output Example

Scoring result example:

```json
{
    "original_idx": 0,
    "question": "Calculate the optical band gap...",
    "reference_answer": "E_g = 3.65 eV",
    "scores": {
        "JoyAI_output_0": {
            "correctness": 48,
            "logic": 23,
            "clarity": 14,
            "completeness": 9,
            "total_score": 94,
            "brief_comment": "Answer is correct, reasoning is clear, used appropriate formulas and unit conversions."
        },
        "JoyAI_output_1": {
            "correctness": 35,
            "logic": 18,
            "clarity": 11,
            "completeness": 7,
            "total_score": 71,
            "brief_comment": "Basic idea is correct, but there are minor errors in the calculation process."
        }
    }
}
```

Analysis report example:

```
================================================================================
LLM Scoring Results Analysis Report
================================================================================

Total samples: 100
Total evaluations: 800
Avg evaluations per sample: 8.00

--------------------------------------------------------------------------------
Scoring Statistics per Model Output
--------------------------------------------------------------------------------

📊 JoyAI_output_0
   Sample count: 100

   Total Score (0-100):
      Mean:   85.30
      Median: 87.00
      Std:    8.45
      Max:    98
      Min:    62

   Dimension Scores:
      Correctness (0-50):  42.50 ± 4.20
      Logic (0-25):        21.30 ± 2.15
      Clarity (0-15):      13.20 ± 1.50
      Completeness (0-10): 8.30 ± 1.20

...

--------------------------------------------------------------------------------
Model Output Ranking (by Mean Total Score)
--------------------------------------------------------------------------------
 1. JoyAI_output_2      - Mean Score:  87.50
 2. JoyAI_output_0      - Mean Score:  85.30
 3. JoyAI_output_5      - Mean Score:  83.20
...
```

## FAQ

### Q: How to adjust scoring criteria?

Edit the `JUDGE_SYSTEM_PROMPT` variable in `prepare_judge.py` to modify scoring dimensions and weights.

### Q: What if the failure rate is high?

1. Check the `judge_failed.jsonl` file.
2. Try lowering the inference temperature (modify in `run_judge_pipeline.py`).
3. Ensure the judge model is loaded correctly.
4. Adjust the prompt format.

### Q: How to speed up scoring?

In the `step2_inference` method of `run_judge_pipeline.py`:

- Increase `dp_size` (requires more GPUs).
- Increase `max_concurrency` (requires more VRAM).
- Decrease `tp_size` (if the model is small).

### Q: How to score only specific outputs?

Modify the loop in `prepare_judge.py`:

```python
# Original code
for output_idx in range(8):

# Change to score only outputs 0, 2, 4
for output_idx in [0, 2, 4]:
```
