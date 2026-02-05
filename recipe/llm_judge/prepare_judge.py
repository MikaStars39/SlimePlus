import json
import argparse
import os
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# ------ Scoring Rubric --------
JUDGE_SYSTEM_PROMPT = """### Role
You are a professional evaluator. You need to assess the quality of a model's response to math/science problems.

### Input
- Question description
- Reference answer (may include the model's reasoning process)
- Model response to be evaluated

### Scoring Criteria
Please score from the following four dimensions (total 100 points):

1. **Correctness (50 points)** - The most important dimension
   - Is the final answer correct?
   - Are the key steps correct?
   - Are the calculation results accurate?

2. **Logic (25 points)**
   - Is the reasoning process coherent?
   - Is each step of the derivation reasonable?
   - Are there any logical errors or jumps?

3. **Clarity (15 points)**
   - Is the expression clear and concise?
   - Are the steps easy to understand?
   - Is the format standardized (e.g., using LaTeX formulas)?

4. **Completeness (10 points)**
   - Does it answer all parts of the question?
   - Are any necessary steps or explanations missing?
   - Is the final answer explicit?

### Output Format
You must output the result strictly in the following JSON format (do not add any other text):

<result>
{{
  "correctness": <integer from 0-50>,
  "logic": <integer from 0-25>,
  "clarity": <integer from 0-15>,
  "completeness": <integer from 0-10>,
  "total_score": <total score, integer from 0-100>,
  "brief_comment": "<brief comment, no more than 100 words>"
}}
</result>

### Evaluation Task

**Question:**
{question}

**Reference Answer:**
{reference_answer}

**Model Response:**
{model_answer}

Please score strictly according to the above format.
"""

# Global tokenizer for worker processes
_tokenizer = None

def init_worker(tokenizer_name):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def process_line(line_data):
    line_idx, line = line_data
    if not line.strip():
        return None
    
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    # Extract question and reference answer
    conversations = data.get('conversations', [])
    if not conversations:
        return None
    
    # Build complete context: system + human question
    system_prompt = ""
    question = ""
    reference_answer = ""
    
    for conv in conversations:
        if conv.get('from') == 'system':
            system_prompt = conv.get('value', '')
        elif conv.get('from') == 'human':
            question = conv.get('value', '')
        elif conv.get('from') == 'gpt':
            reference_answer = conv.get('value', '')
    
    # Combine system prompt and question for complete context
    if system_prompt and question:
        full_question = f"[System Context]\n{system_prompt}\n\n[Question]\n{question}"
    elif question:
        full_question = question
    else:
        return None
    
    results = []
    # Process each model output (JoyAI_output_0 to JoyAI_output_3, only keep 4 outputs)
    for output_idx in range(4):
        output_key = f"JoyAI_output_{output_idx}"
        model_answer = data.get(output_key, "")
        
        if not model_answer:
            continue
        
        # Build evaluation request
        judge_request = JUDGE_SYSTEM_PROMPT.format(
            question=full_question,
            reference_answer=reference_answer,
            model_answer=model_answer
        )
        
        # Build conversation format
        messages = [
            {"role": "user", "content": judge_request}
        ]
        
        # Apply chat template
        prompt = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Create output entry
        output_entry = {
            'original_idx': line_idx,
            'output_idx': output_idx,
            'question': full_question,
            'reference_answer': reference_answer,
            'model_answer': model_answer,
            'prompt': prompt
        }
        results.append(json.dumps(output_entry, ensure_ascii=False))
    
    return results

# ------ Data Processing Logic --------
def prepare_judge_data(input_file, output_file, tokenizer_name, num_workers=None):
    """
    Converts data containing multiple model outputs into the format to be evaluated using multiprocessing.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    print(f"🚀 Starting multiprocessing preparation with {num_workers} workers...")
    
    # Count lines for progress bar using a faster method
    print("Estimating file size...")
    import subprocess
    try:
        total_lines = int(subprocess.check_output(['wc', '-l', input_file]).split()[0])
    except Exception:
        # Fallback to a relatively fast python way if wc fails
        with open(input_file, 'rb') as f:
            total_lines = sum(1 for _ in f)
    
    total_count = 0
    output_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # Create a pool of workers
        with Pool(processes=num_workers, initializer=init_worker, initargs=(tokenizer_name,)) as pool:
            # Enumerate lines for processing
            line_iterator = enumerate(f_in)
            
            # Use imap for memory efficiency with large files
            for results in tqdm(pool.imap(process_line, line_iterator, chunksize=10), total=total_lines, desc="Preparing data"):
                if results:
                    total_count += 1
                    for entry_json in results:
                        f_out.write(entry_json + '\n')
                        output_count += 1
    
    print("\n" + "=" * 60)
    print("Data Preparation Statistics")
    print("=" * 60)
    print(f"Input data count:     {total_count}")
    print(f"Evaluation tasks:     {output_count}")
    print(f"Avg tasks per data:   {output_count / total_count if total_count > 0 else 0:.1f} outputs")
    print(f"Output file:          {output_file}")
    print("=" * 60)

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LLM scoring data with multiprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Input JSONL file (containing conversations and JoyAI_output_0-7)")
    parser.add_argument("--output", required=True, help="Output JSONL file (for LLM scoring)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: cpu_count - 2)")
    
    args = parser.parse_args()
    
    prepare_judge_data(args.input, args.output, args.tokenizer, args.workers)
    print(f"\n✅ [Preparation] Done! Evaluation data saved to: {args.output}")
