import json
import argparse
import os

from transformers import AutoTokenizer

# ------ Configuration --------
# We use <result> tags to wrap the output for precise regex/string extraction
WASH_SYSTEM_PROMPT = """
### Role
You are an expert Data Triage Assistant specialized in mathematical logic and automated evaluation pipelines.

### Objective
Your task is to analyze a given Question and its Reference Answer to determine if the correctness of a student's response can be **Formally Verified** (automatically graded) using a code library like `math_verify` or SymPy.

**Constraint:** You must be strictly rigorous. The reference answer must be a **PURE** mathematical object to be considered verifiable.

### Classification Labels

**1. VERIFIABLE**
Assign this label **ONLY** if the answer is a clean, self-contained mathematical object containing **NO** natural language.
* **Criteria:**
    * **Pure Numbers:** Integers, decimals, fractions (e.g., $42$, $3.14$, $1/2$).
    * **Pure Formulas/Expressions:** Algebraic expressions with no accompanying text (e.g., $x^2 + 2x$, $\\frac{\\sqrt{3}}{2}$, $e^{i\\pi}$).
    * **Sets/Intervals/Vectors:** (e.g., $\\{1, 3\\}$, $[0, \\infty)$, $(1, 0, 0)$).
    * **Equations/Inequalities:** (e.g., $x = 5$, $y > 2x$).
    * **Strict Rule:** The answer must be parseable by a CAS (Computer Algebra System) directly.

**2. NOT_VERIFIABLE**
Assign this label if the answer contains ANY natural language, explanations, definitions, or is a proof.
* **Criteria:**
    * **Hybrid Text & Math:** Answers that mix formulas with text constraints (e.g., "$x = 5$ where $x$ is positive", "The value is $4\\pi$").
    * **Proofs:** Step-by-step logical deductions (e.g., "Show that...", "Prove by induction...").
    * **Descriptions:** (e.g., "It is a circle", "The function is increasing").
    * **Non-Math Fact Retrieval:** (e.g., "Paris", "Hydrogen").
    * **Diagram descriptions.**

### Examples

**Input:**
Q: Solve for x: 2x + 5 = 9.
A: x = 2
**Output:** <result>VERIFIABLE</result> 

**Input:**
Q: Find the general solution for sin(x) = 0.
A: x = n\\pi, where n is an integer
**Output:** <result>NOT_VERIFIABLE</result>
*(Reasoning: The phrase "where n is an integer" makes it impossible to parse as a single symbolic expression without advanced NLP.)*

**Input:**
Q: Expand the expression (a+b)^2.
A: a^2 + 2ab + b^2
**Output:** <result>VERIFIABLE</result> 

**Input:**
Q: What is the capital of France?
A: Paris
**Output:** <result>NOT_VERIFIABLE</result>
*(Reasoning: Although exact match is possible, this is not a mathematical object/formula.)*

**Input:**
Q: Explain why the slope is positive.
A: Because the function is increasing.
**Output:** <result>NOT_VERIFIABLE</result>

**Input:**
Q: Calculate the area.
A: 50 cm^2
**Output:** <result>VERIFIABLE</result>
*(Reasoning: Units are acceptable if they are standard physical quantities parsable by libraries.)*

### Output Format
You must output the result strictly in this format:
<result>VERIFIABLE</result> or <result>NOT_VERIFIABLE</result>
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
                {"role": "user", "content": WASH_SYSTEM_PROMPT + "\n\n**Question:**\n" + data.get('prompt', '') + "\n\n**Reference Answer:**\n" + data.get('label', '')}
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