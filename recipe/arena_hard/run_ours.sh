#!/bin/bash

OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/arena_hard"
MODEL_NAME="sft-16k-v2"  # Set your model name here
MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/sft-16k-v2"
BASELINE_MODEL="o3-mini-2025-01-31"
JUDGE_MODEL="gpt-4.1"
JUDGE_DIR="$OUTPUT_DIR/judgments/$JUDGE_MODEL"

JUDGE_PARALLEL=16
JUDGE_MAX_TOKENS=16000
JUDGE_TEMPERATURE=0
ARENA_HARD_CONFIG="/mnt/llm-train/users/explore-train/qingyu/arena-hard-auto/config/arena-hard-v2.0.yaml"

CATEGORIES=("hard_prompt" "creative_writing" "coding" "math")
API_BASE=http://gpt-proxy.jd.com/v1
CONTROL_FEATURES=""  # e.g. "length markdown" to enable style control
SCRIPT_DIR="/mnt/llm-train/users/explore-train/qingyu/slimulation"
STYLE_ARGS=""

if [ -n "$CONTROL_FEATURES" ]; then
    STYLE_ARGS="--answer-dir $OUTPUT_DIR --control-features $CONTROL_FEATURES"
fi

echo "OPENAI_API_KEY: $OPENAI_API_KEY"

function preprocess() {
    python $SCRIPT_DIR/recipe/arena_hard/preprocess.py \
        --input $OUTPUT_DIR/question.jsonl \
        --output $OUTPUT_DIR/preprocess.jsonl \
        --tokenizer $MODEL_PATH
}

function inference() {
    python $SCRIPT_DIR/recipe/arena_hard/inference.py \
        --input "$OUTPUT_DIR/preprocess.jsonl" \
        --output "$OUTPUT_DIR/results.jsonl" \
        --model "$MODEL_PATH" \
        --tp-size 1 \
        --dp-size 8 \
        --temperature 0.6 \
        --top-p 1 \
        --max-tokens 32768
}

function postprocess() {
    python $SCRIPT_DIR/recipe/arena_hard/postprocess.py \
        --input "$OUTPUT_DIR/results.jsonl" \
        --output "$OUTPUT_DIR/${MODEL_NAME}.jsonl" \
        --model-name "$MODEL_NAME" \
        --tokenizer "$MODEL_PATH"
}

function api_judge() {
    python $SCRIPT_DIR/recipe/arena_hard/api_judge.py \
        --questions "$OUTPUT_DIR/question.jsonl" \
        --model-a "$OUTPUT_DIR/${BASELINE_MODEL}.jsonl" \
        --model-b "$OUTPUT_DIR/${MODEL_NAME}.jsonl" \
        --output-dir "$JUDGE_DIR" \
        --output-prefix "$MODEL_NAME" \
        --judge-model "$JUDGE_MODEL" \
        --api-base "$API_BASE" \
        --api-key "$OPENAI_API_KEY" \
        --temperature $JUDGE_TEMPERATURE \
        --max-tokens $JUDGE_MAX_TOKENS \
        --parallel $JUDGE_PARALLEL \
        --timeout 3600 \
        --categories ${CATEGORIES[*]} \
        --score-output-dir "$OUTPUT_DIR" \
        --baseline-model "$BASELINE_MODEL" \
        --arena-hard-config "$ARENA_HARD_CONFIG" \
        $STYLE_ARGS
}

function aggregate_results() {
    python3 $SCRIPT_DIR/recipe/arena_hard/aggregate_results.py \
        --output-dir "$OUTPUT_DIR" \
        --judge-model "$JUDGE_MODEL" \
        --model-name "$MODEL_NAME" \
        --baseline-model "$BASELINE_MODEL" \
        --categories ${CATEGORIES[*]} \
        $STYLE_ARGS
}

api_judge