import json
import argparse
import re
import os

# ------ Extraction Logic --------
def extract_score_json(text):
    """
    Extracts JSON content within <result> tags from text.
    Returns: dict or None
    """
    # First try to extract <result> tags
    match = re.search(r'<result>\s*(\{.*?\})\s*</result>', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            score_data = json.loads(json_str)
            return score_data
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            print(f"Extracted content: {json_str[:200]}")
            return None
    
    # If no tags found, try to find JSON object directly
    match = re.search(r'\{[^{}]*"correctness"[^{}]*"total_score"[^{}]*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            score_data = json.loads(json_str)
            return score_data
        except json.JSONDecodeError:
            return None
    
    return None

def validate_scores(score_data):
    """
    Validates if scores are within a reasonable range.
    """
    required_fields = ['correctness', 'logic', 'clarity', 'completeness', 'total_score']
    
    for field in required_fields:
        if field not in score_data:
            return False, f"Missing field: {field}"
    
    # Check score ranges
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
            return False, f"{field} is not a number: {score}"
        if not (min_val <= score <= max_val):
            return False, f"{field} out of range [{min_val}, {max_val}]: {score}"
    
    return True, "OK"

# ------ Core Processing --------
def extract_and_aggregate_scores(prepared_file, response_file, output_file, failed_file):
    """
    Extracts scores from LLM responses and aggregates them by original data.
    """
    success_count = 0
    failed_count = 0
    total_count = 0
    
    # Used to aggregate multiple outputs for the same original data
    aggregated_data = {}
    
    with open(prepared_file, 'r', encoding='utf-8') as f_prep, \
         open(response_file, 'r', encoding='utf-8') as f_resp, \
         open(failed_file, 'w', encoding='utf-8') as f_fail:
        
        for line_prep, line_resp in zip(f_prep, f_resp):
            if not line_prep.strip():
                continue
            
            prep_data = json.loads(line_prep)
            resp_data = json.loads(line_resp)
            total_count += 1
            
            original_idx = prep_data['original_idx']
            output_idx = prep_data['output_idx']
            
            # Extract LLM scoring
            raw_response = resp_data.get('response', '')
            score_data = extract_score_json(raw_response)
            
            if score_data is None:
                # Extraction failed
                failed_count += 1
                fail_entry = prep_data.copy()
                fail_entry['raw_response'] = raw_response
                fail_entry['error'] = "Unable to extract score JSON"
                f_fail.write(json.dumps(fail_entry, ensure_ascii=False) + '\n')
                continue
            
            # Validate scores
            is_valid, msg = validate_scores(score_data)
            if not is_valid:
                failed_count += 1
                fail_entry = prep_data.copy()
                fail_entry['raw_response'] = raw_response
                fail_entry['extracted_scores'] = score_data
                fail_entry['error'] = msg
                f_fail.write(json.dumps(fail_entry, ensure_ascii=False) + '\n')
                continue
            
            success_count += 1
            
            # Aggregate data
            if original_idx not in aggregated_data:
                aggregated_data[original_idx] = {
                    'original_idx': original_idx,
                    'question': prep_data['question'],
                    'reference_answer': prep_data['reference_answer'],
                    'scores': {}
                }
            
            # Save scores for this output
            aggregated_data[original_idx]['scores'][f'JoyAI_output_{output_idx}'] = {
                'correctness': score_data['correctness'],
                'logic': score_data['logic'],
                'clarity': score_data['clarity'],
                'completeness': score_data['completeness'],
                'total_score': score_data['total_score'],
                'brief_comment': score_data.get('brief_comment', '')
            }
    
    # Write aggregated results
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Sort output by original_idx
        for idx in sorted(aggregated_data.keys()):
            f_out.write(json.dumps(aggregated_data[idx], ensure_ascii=False) + '\n')
    
    # Statistics
    print("\n" + "=" * 60)
    print("Score Extraction Statistics")
    print("=" * 60)
    print(f"Total evaluation tasks:   {total_count}")
    print(f"Successfully extracted:   {success_count} ({success_count/total_count*100:.1f}%)")
    print(f"Extraction failed:        {failed_count} ({failed_count/total_count*100:.1f}%)")
    print(f"Aggregated data count:    {len(aggregated_data)}")
    print(f"Output file:              {output_file}")
    print(f"Failed records:           {failed_file}")
    print("=" * 60)

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract LLM scoring results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prepared", required=True, help="Output file from preparation stage")
    parser.add_argument("--response", required=True, help="Response file from LLM inference")
    parser.add_argument("--output", required=True, help="Aggregated scoring results file")
    parser.add_argument("--failed", required=True, help="Failed records file")
    
    args = parser.parse_args()
    
    extract_and_aggregate_scores(args.prepared, args.response, args.output, args.failed)
    print(f"\n✅ [Extraction] Done! Scoring results saved to: {args.output}")
