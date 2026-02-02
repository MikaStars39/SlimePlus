import json
import argparse
import os

from transformers import AutoTokenizer

# ------ Configuration --------
# We use <result> tags to wrap the output for precise regex/string extraction
WASH_SYSTEM_PROMPT = """
### Role
You are a professional data taxonomist. Your task is to classify the input prompt into a primary and secondary category.

### ALLOWED CATEGORY LIST
You MUST select the primary and secondary categories ONLY from this list. If a topic is not explicitly listed, you MUST use the "others" category of the most relevant primary domain.

**Primary Categories:**
["math", "science", "humanities", "general", "logic"]

**Secondary Categories (mapped to Primary):**
- **math**: ["arithmetic", "algebra", "geometry", "number_theory", "combinatorics", "probability_stats", "calculus", "discrete_math", "others"]
- **science**: ["physics", "chemistry", "biology", "earth_space", "engineering", "medicine_health", "computer_science", "finance_accounting", "economics", "psychology", "materials_science", "public_health", "agriculture", "environmental_science", "others"]
- **humanities**: ["political_science_sociology", "history_archaeology", "law", "philosophy_ethics", "literature_linguistics", "arts_design", "others"]
- **general**: ["instruction_following", "commonsense", "creative_writing", "general_factoid", "safety", "others"]
- **logic**: ["logic"]

### STRICT INSTRUCTIONS
1. **NO OUT-OF-VOCABULARY LABELS**: Do not invent new labels like "complex_numbers", "trigonometry", or "mechanics". These must be mapped to their parents (e.g., trigonometry -> geometry, complex_numbers -> algebra).
2. **NO DESCRIPTIONS**: Use the exact strings provided in the lists above.
3. **MAPPING RULES**:
   - Complex numbers, Trigonometry, Logarithms -> algebra or geometry.
   - Mechanics, Thermodynamics -> physics.
   - Algorithms (theory) -> computer_science.
   - If unsure, use "others" under the correct primary category.

### Output Format
You must output the result strictly in this format:
<result>primary_category,secondary_category</result>
The primary category MUST be selected from the list of primary categories.
The secondary category MUST be selected from the list of secondary categories that are mapped to the primary category.

geometry is not a primary category.
linear_algebra is math,algebra.
group_theory is math,algebra.

You can first think shortly, then do the category classification task.

### Input Prompt to Classify:
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