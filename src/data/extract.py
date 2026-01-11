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
            "Extract the final concise answer from the following response. "
            "If the answer is a number or a simple expression, provide it directly. \n\n"
            "Response: {raw_res}\n\n"
            "Final Answer:"
        )

    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            # Use 'response' as the key for the model generated text
            raw_res = data.get("response", "")
            data["raw_res"] = raw_res  # Keep original response for reference
            data["prompt"] = prompt_template.format(raw_res=raw_res)
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

