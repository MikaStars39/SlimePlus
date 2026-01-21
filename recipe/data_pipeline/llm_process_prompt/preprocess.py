import json
import argparse
import os

from transformers import AutoTokenizer

# ------ Configuration --------
# We use <result> tags to wrap the output for precise regex/string extraction
WASH_SYSTEM_PROMPT = (
    "You are a professional data judging assistant. Your task is to extract the core question from the given text.\n"
    "Instructions:\n"
    "1. Remove question numbers, unrelated text, or noise.\n"
    "2. The extracted question must include all necessary conditions to make the problem solvable. "
    "Do NOT only extract the final question or the last sentence---you must keep all information, definitions, and problem conditions so that the question is complete. "
    "You do NOT need to check if the question is actually solvable, but you must preserve all the provided information needed to solve it.\n"
    "3. If the text needs cleaning, wrap the cleaned question in <result> cleaned_question </result>.\n"
    "4. If NO cleaning is needed, wrap the word 'UNCHANGED' in <result> UNCHANGED </result>.\n"
    "Here is an example:\n"
    "Input: The following is a problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):\n\n【Problem]\n\n1) Suppose\n\\[\\frac{1}{x^3 - 3x^2 - 13x + 15} = \\frac{A}{x+3} + \\frac{B}{x-1} + \\frac{C}{(x-1)^2}\\]\nwhere $A$, $B$, and $C$ are real constants. What is $A$?\n Output the answer inside the \\boxed{{}}"
    "You should Output: <result>Suppose\n\\[\\frac{1}{x^3 - 3x^2 - 13x + 15} = \\frac{A}{x+3} + \\frac{B}{x-1} + \\frac{C}{(x-1)^2}\\]\nwhere $A$, $B$, and $C$ are real constants. What is $A$?</result>\n"
    "Below is the real text to clean. You can think first, then extract: \n\n"
)

# ------ Logic --------
def prepare_data(input_file, output_file, tokenizer_name):
    """
    Wraps the 'prompt' field and adds explicit formatting instructions.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Formulating the message with the new instructions
            data['prompt'] = [
                {"role": "user", "content": WASH_SYSTEM_PROMPT + data.get('prompt', '')}
            ]

            data['prompt'] = tokenizer.apply_chat_template(
                data['prompt'],
                tokenize=False,
                add_generation_prompt=True
            )

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process: Wrap prompts with tag instructions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.tokenizer)
    print(f"[Pre-process] Done. Inference file ready: {args.output}")