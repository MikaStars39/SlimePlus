import json
import argparse
import re
import os

# ------ Extraction Logic --------
def extract_result(text):
    """
    Extract content between <result> and </result> tags.
    Returns the extracted string, or None if not found.
    """
    match = re.search(r'<result>(.*?)</result>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# ------ Core Processing --------
def finalize_results(original_file, response_file, output_file, failed_file):
    """
    Parses LLM responses using tags and updates the original dataset.
    Also counts the number of verifiable and not-verifiable cases.
    """
    failed_extraction_count = 0
    total_count = 0
    verifiable_count = 0
    not_verifiable_count = 0

    with open(original_file, 'r', encoding='utf-8') as f_orig, \
         open(response_file, 'r', encoding='utf-8') as f_resp, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
         open(failed_file, 'w', encoding='utf-8') as f_fail:

        # Line-by-line synchronized processing
        for line_orig, line_resp in zip(f_orig, f_resp):
            if not line_orig.strip():
                continue

            orig_data = json.loads(line_orig)
            resp_data = json.loads(line_resp)
            total_count += 1

            raw_llm_output = resp_data.get('response', '')
            extracted_content = extract_result(raw_llm_output)

            if extracted_content is None:
                # Model did not output the expected tag.
                failed_extraction_count += 1
                fail_item = orig_data.copy()
                fail_item['raw_response'] = raw_llm_output
                f_fail.write(json.dumps(fail_item, ensure_ascii=False) + '\n')
                continue  # Skip writing to successful output

            if extracted_content == "VERIFIABLE":
                orig_data['verifiable'] = True
                verifiable_count += 1
            elif extracted_content == "NOT_VERIFIABLE":
                orig_data['verifiable'] = False
                not_verifiable_count += 1
            else:
                # Unexpected content, treat as failed extraction
                failed_extraction_count += 1
                fail_item = orig_data.copy()
                fail_item['raw_response'] = raw_llm_output
                fail_item['extracted_content'] = extracted_content
                f_fail.write(json.dumps(fail_item, ensure_ascii=False) + '\n')
                continue

            f_out.write(json.dumps(orig_data, ensure_ascii=False) + '\n')

    # ------ Statistics --------
    print("-" * 40)
    print(f"Post-processing Report for: {os.path.basename(original_file)}")
    print(f"  - Total processed:         {total_count}")
    print(f"  - VERIFIABLE cases:        {verifiable_count}")
    print(f"  - NOT_VERIFIABLE cases:    {not_verifiable_count}")
    print(f"  - Tag parse failures:      {failed_extraction_count}")
    print(f"  - Output saved:            {output_file}")
    print(f"  - Failed cases saved:      {failed_file}")
    print("-" * 40)

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process LLM verifiability tagging results.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed", required=True, help="File to save records where extraction failed.")
    args = parser.parse_args()

    finalize_results(args.original, args.response, args.output, args.failed)