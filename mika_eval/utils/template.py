import json
import logging
from tqdm import tqdm
from transformers import AutoTokenizer

PROMPT_TEMPLATES = {
    #lighteval
    "lighteval": """{problem} Please reason step by step, and put your final answer within \\boxed{{}}.""",
    
    # open-r1
    "open-r1": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),

    # for llm extraction
    "extraction": """
Please extract the final answer from the following response. The answer should be put inside \\boxed{{}}. 

Response:
{response}
""".strip(),

    # slime default training format
    "slime": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),

    # GPQA inferece
    "GPQA-Diamond": """
{problem}
Choose an answer in A,B,C,D. Answer with \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}.
""".strip(),

    "blank": """
{problem}
""".strip(),

    "jd_thinking": """
    You are JoyAI, a large language model trained by JD (京东). For every response, please provide a step-by-step reasoning process enclosed in <think> and </think> tags. After the thinking, you need to output the final answer.
""".strip(),

}


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

            # ------------------------------ 3. Format the user prompt ------------------------------  
            # We use .replace() instead of .format() to avoid IndexError with LaTeX braces like \boxed{}.
            # Support both "{problem}" and legacy "{prompt}" placeholders.

            if user_template == "auto": # this is only for inference (for questions)
                if "gpqa_diamond" in data_source:
                    current_template = PROMPT_TEMPLATES["GPQA-Diamond"]
                elif "ifeval" in data_source:
                    current_template = PROMPT_TEMPLATES["blank"]
                else:
                    current_template = PROMPT_TEMPLATES["slime"]
            elif user_template in PROMPT_TEMPLATES:
                current_template = PROMPT_TEMPLATES[user_template]
            else:
                current_template = user_template

            formatted_user_content = (
                current_template.replace("{problem}", raw_question).replace("{prompt}", raw_question)
            )

            # ------------------------------ 4. Create the chat message structure ------------------------------ 
            messages = [
                {"role": "user", "content": formatted_user_content},
            ] if system_prompt is None else [
                {"role": "system", "content": PROMPT_TEMPLATES[system_prompt]},
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