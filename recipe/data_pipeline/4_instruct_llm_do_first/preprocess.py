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
                {"role": "user", "content": data.get('prompt', '')}
            ]

            data['prompt'] = tokenizer.apply_chat_template(
                data['prompt'],
                tokenize=False,
                add_generation_prompt=True
            )

            data_id = data["id"][:-1]

            for index in range(int(args.k)):
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

    prepare_data(args.input, args.output, args.tokenizer)
    print(f"[Pre-process] Done. Inference file ready: {args.output}")