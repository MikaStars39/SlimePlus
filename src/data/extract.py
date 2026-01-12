import json
from pathlib import Path
from typing import Optional

def prepare_extraction_data(
    input_file: Path,
    output_file: Path,
    prompt_template: Optional[str] = None
):
    """
    Read inference results and prepare prompts for answer extraction.

    Args:
        input_file: Path to inference_results.jsonl
        output_file: Path to eval_input.jsonl
        prompt_template: Custom prompt template for extraction
    """
    if prompt_template is None:
        prompt_template = (
            "Please extract the final numerical or concise answer from the following reasoning process. "
            "Respond only with the answer wrapped in \\boxed{}.\n\n"
            "Reasoning Process:\n{response}\n\n"
            "You should ONLY output \\boxed{{answer}} format. Do not output anything else."
        )

    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            # Use 'response' as the key for the model generated text
            raw_res = data.pop("response", "") # Remove old response to avoid triple redundancy
            # Use a more robust way to handle the prompt template
            data["prompt"] = prompt_template.replace("{response}", str(raw_res))
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

# The main function below references extract_metrics_from_file
# Ensure this function is defined elsewhere in project if running as main.
if __name__ == "__main__":
    eval_output_file = Path("outputs/20260110_173129_gspo_qwen30ba3b_0000223_slime_new/eval_results.jsonl")
    results = extract_metrics_from_file(eval_output_file)
    print(results)
