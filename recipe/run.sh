
pip install -e /mnt/llm-train/users/explore-train/qingyu/MikaEval --no-deps -i https://mirrors.jd.com/pypi/web/simple
pip install -e /mnt/llm-train/qingyu/MikaEval --no-deps -i https://mirrors.jd.com/pypi/web/simple

python /mnt/llm-train/users/qingyu/MikaEval/recipe/data_pipeline/4_instruct_llm_do_first/preprocess.py \
    --input /mnt/llm-train/users/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_relabel_post.jsonl \
    --output /mnt/llm-train/users/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_profile_pre.jsonl \
    --k 10 \
    --tokenizer /mnt/llm-train/users/qingyu/Model/DeepSeek-R1-Distill-Qwen-32B

python /mnt/llm-train/users/qingyu/MikaEval/recipe/data_pipeline/4_instruct_llm_do_first/inference.py \
    --input /mnt/llm-train/users/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_profile_pre.jsonl \
    --output /mnt/llm-train/users/qingyu/data/stage_1/INTELLECT-3-RL-Math/raw_profile_infer_distill_32b.jsonl \
    --model_path /mnt/llm-train/users/qingyu/Model/DeepSeek-R1-Distill-Qwen-32B \
    --tp_size 1 \
    --dp_size 8 \
    --temp 0.8 \
    --top_p 1 \
    --max_concurrency 1024 \
    --max_tokens 12000
    
# 4. 执行打分脚本，生成最终10题文件（带新pass_rate和num_rollout）
echo -e "\n===== 开始计算分数和pass_rate ====="
python3 miaoji/math_with_judge/score_math_data.py \
  --infer_file "/mnt/llm-train/users/qingyu/WebInstruct-verified-math/raw_profile_distill_infer_50k.jsonl" \
  --origin_file "/mnt/llm-train/users/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_post.jsonl" \
  --output_file "/mnt/llm-train/users/qingyu/data/profile/WebInstruct-verified-math/final_distill_32b.jsonl"

python3 miaoji/math_with_judge/score_math_data.py \
  --infer_file "/mnt/llm-train/users/qingyu/WebInstruct-verified-math/raw_profile_infer_50k.jsonl" \
  --origin_file "/mnt/llm-train/users/qingyu/data/stage_1/WebInstruct-verified-math/raw_relabel_post.jsonl" \
  --output_file "/mnt/llm-train/users/qingyu/data/profile/WebInstruct-verified-math/final_30binstruct.jsonl"