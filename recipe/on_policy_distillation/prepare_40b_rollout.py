import json
import argparse
import os

from transformers import AutoTokenizer

# ------ Logic --------
def prepare_data(input_file, output_file, k, tokenizer_name):
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
                    "role": "system",
                    "content": "You are JoyAI, a large language model trained by JD (京东). Answer as concisely as possible."
                },
                {
                    "role": "user", 
                    "content": data.get('prompt', '')
                }
            ]

            data['prompt'] = tokenizer.apply_chat_template(
                data['prompt'],
                tokenize=False,
                add_generation_prompt=True
            )

            # replace <think> to </think>
            data['prompt'] = data['prompt'].replace('<think>', '</think>')

            data_id = data["id"][:-1]

            for index in range(int(k)):
                data["id"] = f"{data_id}{str(index)}"
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process: Wrap prompts with tag instructions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.k, args.tokenizer)
    print(f"[Prepare 40B Rollout] Done. Rollout file ready: {args.output}")