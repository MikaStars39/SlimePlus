import json
import argparse
import re

def prepare_data(input_file, output_file):
    """
    Extracts the plain text prompt from the chat template in 'data["prompt"]',
    and saves it as the new 'prompt' in the output JSON.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            
            pure_prompt = data["prompt"][48:-119]

            new_data = {}
            new_data["prompt"] = [ 
                { "content": "You are JoyAI, a large language model trained by JD (京东). Answer as concisely as possible.", "role": "system" },
                { "content": pure_prompt, "role": "user" }
             ]
            new_data["label"] = data["label"]
            data["reference"] = data["response"]
            new_data["metadata"] = data

            f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract plain text prompt by removing chat template tags.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    prepare_data(args.input, args.output)
    print(f"[Pre-process] Done. Inference file ready: {args.output}")