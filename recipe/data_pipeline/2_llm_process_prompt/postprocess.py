import json
import argparse
import re
import os

# ------ Extraction Logic --------
def extract_result(text):
    """
    Extracts content between <result> and </result> tags.
    Returns: extracted_string or None if not found.
    """
    match = re.search(r'<result>(.*?)</result>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# ------ Core Processing --------
def finalize_results(original_file, response_file, output_file, failed_file):
    """
    Parses LLM responses using tags and updates the original dataset.
    """
    cleaned_count = 0
    failed_extraction_count = 0
    total_count = 0

    with open(original_file, 'r', encoding='utf-8') as f_orig, \
         open(response_file, 'r', encoding='utf-8') as f_resp, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
         open(failed_file, 'w', encoding='utf-8') as f_fail:
        
        # Using zip for synchronized processing (Linux: row-by-row stream)
        for line_orig, line_resp in zip(f_orig, f_resp):
            if not line_orig.strip(): continue
            
            orig_data = json.loads(line_orig)
            resp_data = json.loads(line_resp)
            total_count += 1
            
            raw_llm_output = resp_data.get('response', '')
            extracted_content = extract_result(raw_llm_output)
            
            if extracted_content is None:
                # If model failed to follow tag format, save to failed file and skip output
                failed_extraction_count += 1
                
                # Save failed case to a separate file for inspection
                fail_item = orig_data.copy()
                fail_item['raw_response'] = raw_llm_output
                f_fail.write(json.dumps(fail_item, ensure_ascii=False) + '\n')
                continue  # Skip writing to f_out
            
            if extracted_content == "UNCHANGED":
                # Model explicitly said no cleaning needed
                final_prompt = orig_data.get('prompt')
            else:
                # Cleaning performed
                final_prompt = extracted_content
                if final_prompt != orig_data.get('prompt'):
                    cleaned_count += 1
            
            # Finalize the JSON object and save to output
            orig_data['prompt'] = final_prompt
            if 'messages' in orig_data: del orig_data['messages']
            
            f_out.write(json.dumps(orig_data, ensure_ascii=False) + '\n')

    # ------ Statistics --------
    print("-" * 40)
    print(f"Post-processing Report for: {os.path.basename(original_file)}")
    print(f"  - Total processed:     {total_count}")
    print(f"  - Successfully cleaned: {cleaned_count}")
    print(f"  - Tag parse failures:  {failed_extraction_count}")
    print(f"  - Final output saved:  {output_file}")
    print(f"  - Failed cases saved:   {failed_file}")
    print("-" * 40)

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process: Tag-based extraction.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed", required=True, help="File to save records where extraction failed.")
    args = parser.parse_args()
    
    finalize_results(args.original, args.response, args.output, args.failed)