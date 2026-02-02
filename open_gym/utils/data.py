import json
import logging
from tqdm import tqdm
from transformers import AutoTokenizer

from .template import SYSTEM_PROMPT_TEMPLATES

def apply_template_to_jsonl(
    input_file: str, 
    output_file: str, 
    model_path: str, 
    user_template: str,
    system_prompt: str = None,
):
    """
    Reads a JSONL file and wraps the 'prompt' field into a model-specific chat template.
    """
    
    # ------------------------------ 1. Load the tokenizer ------------------------------ 
    logging.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ------------------------------ 2. Process the file ------------------------------ 
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        logging.info(f"Applying template...")
        
        for line in tqdm(f_in):
            if not line.strip():
                continue
                
            data = json.loads(line)
            raw_question = data.get("prompt", "")
            data_source = data.get("source", "")

            user_template_type = data.get("user_template_type", "blank")
            user_template = PROMPT_TEMPLATES[user_template_type]
            system_prompt_type = data.get("system_template_type", None)
            system_prompt = SYSTEM_PROMPT_TEMPLATES[system_prompt_type] if system_prompt_type is not None else None

            formatted_user_content = user_template.replace("{problem}", raw_question)

            # ------------------------------ 4. Create the chat message structure ------------------------------ 
            messages = [
                {"role": "user", "content": formatted_user_content},
            ] if system_prompt is None else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_content},
            ]

            # ------------------------------ 5. Apply the official chat template ------------------------------ 
            try:
                final_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback for tokenizers/models without chat_template support.
                # Keep the pipeline running by using the formatted user content directly.
                logging.info(f"[template] apply_chat_template failed; fallback to raw prompt. Error: {e}")
                final_prompt = formatted_user_content

            # ------------------------------ 6. Save the record ------------------------------ 
            data["prompt"] = final_prompt
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    logging.info(f"-"*100)
    logging.info(f"Using template: {current_template}")
    logging.info(f"-"*100)

    if system_prompt is not None:
        logging.info(f"Using system prompt: {PROMPT_TEMPLATES[system_prompt]}")
    logging.info(f"Done! Formatted file saved to: {output_file}")

if __name__ == "__main__":
    input_file = "outputs/debug/prepared_inference_data.jsonl"
    output_file = "outputs/debug/formatted_chat_data.jsonl"
    model_path = "/mnt/llm-train/users/explore-train/qingyu/MikaEval/.cache/Qwen3-4B-Instruct-2507" 
    
    system_prompt = "You are an expert mathematician. Provide detailed step-by-step solutions."

    # Now this will work because we switched to .replace()!
    # No need to double the braces for \boxed{}
    user_template = (
        "Problem: {prompt}\n\n"
        "Please reason carefully and provide the final answer in \\boxed{}."
    )

    apply_template_to_jsonl(
        input_file=input_file, 
        output_file=output_file, 
        model_path=model_path, 
        system_prompt=system_prompt,
        user_template=user_template
    )