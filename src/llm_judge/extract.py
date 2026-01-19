import json
from pathlib import Path
from typing import Optional

PROMPT_TEMPLATE = (
    "Reference Format (Ground Truth): {label}\n\n"
    "Model Reasoning Process:\n{response}\n\n"
    "Task: Extract the final answer from the reasoning process above.\n"
    "Instructions:\n"
    "1. Follow the style/format of the Reference Format.\n"
    "2. DO NOT correct any mistakes. Extract what the model actually concluded, even if wrong.\n"
    "3. DO NOT simplify the answer. If the model concludes with an equation (e.g., 'x = 0'), extract the FULL equation: \\boxed{{x = 0}}, NOT just \\boxed{{0}}.\n"
    "4. You can do short analysis for the answer. Your final response should ONLY with \\boxed{{answer}} format."
)

def prepare_extraction_data(
    input_file: Path,
    output_file: Path,
    output_no_eval_file: Path,
):
    """
    Read inference results and prepare prompts for answer extraction.

    Args:
        input_file: Path to inference_results.jsonl
        output_file: Path to eval_input.jsonl
        output_no_eval_file: Path,
        prompt_template: Custom prompt template for extraction
    """

    no_eval_data = []

    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            # Use 'response' as the key for the model generated text
            raw_res = data.get("response", "") # Remove old response to avoid triple redundancy
            need_llm_extract = data.get("need_llm_extract", True)

            # if no need to extract, skip
            if not need_llm_extract:
                no_eval_data.append(data)
                continue
            
            data.pop("response")

            # Prepare prompt using label as reference; do not correct model's reasoning errors
            data["prompt"] = PROMPT_TEMPLATE.format(
                response=str(raw_res),
                label=data.get("label", "N/A")
            )
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    with open(output_no_eval_file, "w", encoding="utf-8") as f_out:
        for data in no_eval_data:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

# The main function below references extract_metrics_from_file
# Ensure this function is defined elsewhere in project if running as main.
if __name__ == "__main__":
    eval_output_file = Path("outputs/20260110_173129_gspo_qwen30ba3b_0000223_slime_new/eval_results.jsonl")
    results = extract_metrics_from_file(eval_output_file)
    print(results)
