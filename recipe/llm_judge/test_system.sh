#!/bin/bash
# LLM Judge System Test Script

set -e  # Exit immediately if a command exits with a non-zero status

echo "=================================="
echo "LLM Judge System Test"
echo "=================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_INPUT="${SCRIPT_DIR}/test_data_example.jsonl"
TEST_OUTPUT="${SCRIPT_DIR}/test_output"
JUDGE_MODEL="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8"

echo "Test Configuration:"
echo "  Input File:     ${TEST_INPUT}"
echo "  Output Directory: ${TEST_OUTPUT}"
echo "  Judge Model:    ${JUDGE_MODEL}"
echo ""

# Check if input file exists
if [ ! -f "${TEST_INPUT}" ]; then
    echo "❌ ERROR: Test data file not found: ${TEST_INPUT}"
    exit 1
fi

# Check if model directory exists
if [ ! -d "${JUDGE_MODEL}" ]; then
    echo "⚠️  WARNING: Judge model path not found: ${JUDGE_MODEL}"
    echo "Please modify the JUDGE_MODEL variable in the script to the correct path."
    exit 1
fi

# Clean previous test output
if [ -d "${TEST_OUTPUT}" ]; then
    echo "Cleaning previous test output..."
    rm -rf "${TEST_OUTPUT}"
fi

echo "=================================="
echo "Starting Test Run..."
echo "=================================="
echo ""

# Run the scoring pipeline
cd "${SCRIPT_DIR}"
python run_judge_pipeline.py \
    --input "${TEST_INPUT}" \
    --output-dir "${TEST_OUTPUT}" \
    --judge-model "${JUDGE_MODEL}"

# Verify output files
echo ""
echo "=================================="
echo "Verifying Output Files..."
echo "=================================="

required_files=(
    "judge_prepared.jsonl"
    "judge_inference.jsonl"
    "judge_scores.jsonl"
    "judge_failed.jsonl"
)

all_exist=true
for file in "${required_files[@]}"; do
    if [ -f "${TEST_OUTPUT}/${file}" ]; then
        line_count=$(wc -l < "${TEST_OUTPUT}/${file}" || echo "0")
        echo "✅ ${file} (${line_count} lines)"
    else
        echo "❌ ${file} (Missing)"
        all_exist=false
    fi
done

if [ "${all_exist}" = false ]; then
    echo ""
    echo "❌ Test Failed: Some output files are missing."
    exit 1
fi

# Run analysis
echo ""
echo "=================================="
echo "Generating Analysis Report..."
echo "=================================="
echo ""

python analyze_scores.py --scores "${TEST_OUTPUT}/judge_scores.jsonl"

# Export to CSV
echo ""
echo "Exporting to CSV..."
python analyze_scores.py \
    --scores "${TEST_OUTPUT}/judge_scores.jsonl" \
    --export-csv "${TEST_OUTPUT}/scores.csv"

# Check CSV file
if [ -f "${TEST_OUTPUT}/scores.csv" ]; then
    line_count=$(wc -l < "${TEST_OUTPUT}/scores.csv")
    echo "✅ CSV file generated (${line_count} lines)"
fi

# Show result example
echo ""
echo "=================================="
echo "Result Example:"
echo "=================================="
echo ""
head -n 1 "${TEST_OUTPUT}/judge_scores.jsonl" | jq '.' 2>/dev/null || \
    head -n 1 "${TEST_OUTPUT}/judge_scores.jsonl"

echo ""
echo "=================================="
echo "✅ Test Completed Successfully!"
echo "=================================="
echo ""
echo "Test output directory: ${TEST_OUTPUT}"
echo ""
echo "You can check the following files:"
echo "  - judge_scores.jsonl : Scoring results"
echo "  - judge_failed.jsonl : Failed records"
echo "  - scores.csv         : CSV format"
echo ""
echo "Run the following command for detailed analysis:"
echo "  python analyze_scores.py --scores ${TEST_OUTPUT}/judge_scores.jsonl"
echo ""
