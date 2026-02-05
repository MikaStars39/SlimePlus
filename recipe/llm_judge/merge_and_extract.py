#!/usr/bin/env python3
"""
Merge response shards and extract scores efficiently.
Process all response files in parallel and extract scores.
"""
import json
import argparse
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict

def extract_score_json(text):
    """Extract JSON content within <result> tags from text."""
    match = re.search(r'<result>\s*(\{.*?\})\s*</result>', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def validate_scores(score_data):
    """Validate if scores are within reasonable ranges."""
    required_fields = ['correctness', 'logic', 'clarity', 'completeness', 'total_score']
    
    for field in required_fields:
        if field not in score_data:
            return False, f"Missing field: {field}"
    
    ranges = {
        'correctness': (0, 50),
        'logic': (0, 25),
        'clarity': (0, 15),
        'completeness': (0, 10),
        'total_score': (0, 100)
    }
    
    for field, (min_val, max_val) in ranges.items():
        score = score_data[field]
        if not isinstance(score, (int, float)):
            return False, f"{field} is not a number"
        if not (min_val <= score <= max_val):
            return False, f"{field} out of range"
    
    return True, "OK"

def process_response_file(response_file):
    """Process one response file and return extracted scores."""
    results = []
    failed = []
    
    with open(response_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                failed.append({'line': line_num, 'file': str(response_file), 'error': 'JSON parse error'})
                continue
            
            original_idx = data.get('original_idx')
            output_idx = data.get('output_idx')
            raw_response = data.get('response', '')
            
            # Extract score
            score_data = extract_score_json(raw_response)
            if score_data is None:
                failed.append({
                    'line': line_num,
                    'file': str(response_file),
                    'original_idx': original_idx,
                    'output_idx': output_idx,
                    'error': 'Failed to extract score JSON'
                })
                continue
            
            # Validate score
            is_valid, msg = validate_scores(score_data)
            if not is_valid:
                failed.append({
                    'line': line_num,
                    'file': str(response_file),
                    'original_idx': original_idx,
                    'output_idx': output_idx,
                    'error': msg
                })
                continue
            
            # Store result
            results.append({
                'original_idx': original_idx,
                'output_idx': output_idx,
                'question': data.get('question', ''),
                'reference_answer': data.get('reference_answer', ''),
                'model_answer': data.get('model_answer', ''),
                'scores': score_data
            })
    
    return results, failed

def merge_and_extract(response_dir, output_file, failed_file, num_workers=None):
    """
    Process all response files in parallel and extract scores.
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 24)
    
    response_dir = Path(response_dir)
    response_files = sorted(response_dir.glob('response_*.jsonl'))
    
    if not response_files:
        print(f"❌ No response files found in {response_dir}")
        return
    
    print(f"Found {len(response_files)} response files")
    print(f"Processing with {num_workers} workers...")
    
    all_results = []
    all_failed = []
    
    # Process files in parallel
    with Pool(processes=num_workers) as pool:
        for results, failed in tqdm(pool.imap(process_response_file, response_files), 
                                     total=len(response_files),
                                     desc="Processing response files"):
            all_results.extend(results)
            all_failed.extend(failed)
    
    # Write extracted scores
    print(f"\nWriting {len(all_results)} extracted scores to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Write failed cases
    if all_failed:
        print(f"Writing {len(all_failed)} failed cases to {failed_file}...")
        with open(failed_file, 'w', encoding='utf-8') as f:
            for fail in all_failed:
                f.write(json.dumps(fail, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 60)
    print("Extraction Statistics")
    print("=" * 60)
    print(f"Total extracted:      {len(all_results)}")
    print(f"Failed extractions:   {len(all_failed)}")
    print(f"Success rate:         {len(all_results)/(len(all_results)+len(all_failed))*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and extract scores from response shards")
    parser.add_argument("--response-dir", required=True, help="Directory containing response_*.jsonl files")
    parser.add_argument("--output", required=True, help="Output file for extracted scores")
    parser.add_argument("--failed", required=True, help="Output file for failed extractions")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    
    args = parser.parse_args()
    merge_and_extract(args.response_dir, args.output, args.failed, args.workers)
    print(f"\n✅ Done! Extracted scores saved to: {args.output}")
