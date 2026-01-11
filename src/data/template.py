import json
from tqdm import tqdm
from transformers import AutoTokenizer

PROMPT_TEMPLATES = {
    "lighteval": """{problem} Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "open-r1": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),

    "extraction": """
Please extract the final answer from the following response. The answer should be put inside \\boxed{{}}. 

Response:
{response}
""".strip(),
    "slime": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),
}


def apply_template_to_jsonl(
    input_file: str, 
    output_file: str, 
    model_path: str, 
    system_prompt: str = "You are a helpful assistant.",
    user_template: str = "{prompt}"
):
    """
    Reads a JSONL file and wraps the 'prompt' field into a model-specific chat template.
    """
    
    # 1. Load the tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. Process the file
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        print(f"Applying template...")
        
        for line in tqdm(f_in):
            if not line.strip():
                continue
                
            data = json.loads(line)
            raw_question = data["prompt"]

            # 3. Format the user prompt 
            # We use .replace() instead of .format() to avoid IndexError with LaTeX braces like \boxed{}
            formatted_user_content = user_template.replace("{prompt}", raw_question)

            # 4. Create the chat message structure
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_content},
            ]

            # 5. Apply the official chat template
            final_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 6. Save the record
            data["prompt"] = final_prompt
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Done! Formatted file saved to: {output_file}")

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