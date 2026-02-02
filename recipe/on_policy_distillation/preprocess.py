import json
import argparse
import os

from transformers import AutoTokenizer

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
                {
                    "role": "user", 
                    "content": "Here is a question:" + data.get('prompt', '') + "\n\n and **Reference Answer:**\n" + data.get('label', '') + "\n\nYou need to write a tutorial about how to solve this problem."
                }
            ]

            data['prompt'] = tokenizer.apply_chat_template(
                data['prompt'],
                tokenize=False,
                add_generation_prompt=True
            )

            # replace <think> to </think>
            data['prompt'] = data['prompt'].replace('<think>', '</think>')

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