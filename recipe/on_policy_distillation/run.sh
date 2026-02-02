python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/on_policy_distillation/prepare_40b_rollout.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/opd/40b_rollout_pre.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/ckpt/sft-s1-lr1e-5-DECAY_SAMPLES-0212_1456

head -n 16384 /mnt/llm-train/users/explore-train/qingyu/data/opd/40b_rollout_pre.jsonl > /mnt/llm-train/users/explore-train/qingyu/data/opd/40b_rollout_pre_16k.jsonl

python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/inference.py \
    --input "/mnt/llm-train/users/explore-train/qingyu/data/opd/40b_rollout_pre_16k.jsonl" \
    --output "/mnt/llm-train/users/explore-train/qingyu/data/opd/40b_rollout_infer.jsonl" \
    --model "/mnt/llm-train/users/explore-train/wangzhenfang8/hf_outputs/40b/sft-postrain-exp-v5-baselong-e2_lr5e-5_minlr5e-6/13638/" \
    --tp-size 8 \
    --dp-size 1 \
    --temperature 1 \
    --max-tokens 16384

python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/on_policy_distillation/preprocess.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/opd/dapo_raw.jsonl \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/ckpt/sft-s1-lr1e-5-DECAY_SAMPLES-0212_1456

head -n 16384 /mnt/llm-train/users/explore-train/qingyu/data/opd/dapo_raw.jsonl > /mnt/llm-train/users/explore-train/qingyu/data/opd/dapo_raw_16k.jsonl

python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/inference.py \
    --input "/mnt/llm-train/users/explore-train/qingyu/data/opd/dapo_raw_16k.jsonl" \
    --output "/mnt/llm-train/users/explore-train/qingyu/data/opd/dapo_infer.jsonl" \
    --model "/mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-V3.2/DeepSeek-V3.2" \
    --tp-size 8 \
    --dp-size 1 \
    --temperature 1 \
    --max-tokens 16384

python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/on_policy_distillation/prepare_to_slime_format.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/opd/infer.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/opd/infer_slime.jsonl

# -------------------------- DAPO --------------------------

python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/on_policy_distillation/prepare_40b_rollout.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/stage_1/DAPO-Math-17k-Processed/raw.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/opd/ours_raw.jsonl \
    --k 8 \
    --tokenizer /mnt/llm-train/users/explore-train/qingyu/ckpt/sft-s1-lr1e-5-DECAY_SAMPLES-0212_1456

python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/inference.py \
    --input "/mnt/llm-train/users/explore-train/qingyu/data/opd/ours_raw.jsonl" \
    --output "/mnt/llm-train/users/explore-train/qingyu/data/opd/ours_infer.jsonl" \
    --model "/mnt/llm-train/users/explore-train/wangzhenfang8/hf_outputs/40b/sft-postrain-exp-v5-baselong-e2_lr5e-5_minlr5e-6/13638" \
    --tp-size 8 \
    --dp-size 1 \
    --temperature 1 \
    --max-tokens 8192

python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/on_policy_distillation/prepare_to_slime_format_language.py \
    --input /mnt/llm-train/users/explore-train/qingyu/data/language/typos.jsonl \
    --output /mnt/llm-train/users/explore-train/qingyu/data/opd/infer_slime_language.jsonl