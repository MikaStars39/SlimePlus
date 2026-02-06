python /mnt/llm-train/users/explore-train/qingyu/slimulation/slimulation/backend/online.py \
  --api-key "sk-xxxxxxxxxxxxxxxxxxx" \
  --base-url "http://6.30.2.202:39000/v1" \
  --model "gpt-4.1" \
  --input "/mnt/llm-train/users/explore-train/qingyu/data/arena_hard/prompt.jsonl" \
  --output "/mnt/llm-train/users/explore-train/qingyu/data/arena_hard/output.jsonl" \
  --concurrency 500 \
  --temperature 1 \
  --max-tokens 16000 \
  --extra