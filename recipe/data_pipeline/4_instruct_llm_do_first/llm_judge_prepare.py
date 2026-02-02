import json
import argparse
import os

from transformers import AutoTokenizer

# ------ Configuration --------
# We use <result> tags to wrap the output for precise regex/string extraction
WASH_SYSTEM_PROMPT = """
### Role
You are a professional judger. You will judge if the model's answer of a question is correct.

### Input
The model's answer, the ground truth answer.

### Output Format
You must output the result strictly in this format:
<result>CORRECT/INCORRECT</result>

You can reason first, then answer. Your answer must be either CORRECT or INCORRECT.

### Task
Question:
{question}

Model answer:
{model_answer}

Ground truth:
{ground_truth}
"""

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
                {"role": "user", "content": WASH_SYSTEM_PROMPT.format(
                    question=data.get('prompt'), 
                    model_answer=data.get('response'), 
                    ground_truth=data.get('label')
                )}
            ]

            data['prompt'] = tokenizer.apply_chat_template(
                data['prompt'],
                tokenize=False,
                add_generation_prompt=True
            )

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare: Prepare data for LLM judge.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    prepare_data(args.input, args.output, args.tokenizer)
    print(f"[Pre-process] Done. Inference file ready: {args.output}")