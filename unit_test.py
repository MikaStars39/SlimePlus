
def test_prepare_pass_at_k_jsonl():
    from mika_eval.tasks import prepare_pass_at_k_jsonl
    # The input configuration string
    config_str = "aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32,ifeval@1,mmlu_pro@1,gpqa_diamond@1,ceval@1"
    output_file = "outputs/debug/prepared_inference_data.jsonl"
    cache_dir = "/mnt/llm-train/users/explore-train/qingyu/.cache"
    
    prepare_pass_at_k_jsonl(
        config_str=config_str, 
        output_file=output_file, 
        cache_dir=cache_dir,
    )

if __name__ == "__main__":
    test_prepare_pass_at_k_jsonl()
