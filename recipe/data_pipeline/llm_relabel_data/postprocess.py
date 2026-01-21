import json
import argparse
import re
import os
from collections import Counter, defaultdict

# ------ Taxonomy Whitelist --------
# Only categories in this dictionary are allowed.
VALID_TAXONOMY = {
    "math": ["arithmetic", "algebra", "geometry", "number_theory", "combinatorics", "probability_stats", "calculus", "discrete_math", "others"],
    "science": ["physics", "chemistry", "biology", "earth_space", "engineering", "medicine_health", "computer_science", "finance_accounting", "economics", "psychology", "others"],
    "humanities": ["political_science_sociology", "history_archaeology", "law", "philosophy_ethics", "literature_linguistics", "arts_design", "others"],
    "general": ["instruction_following", "commonsense", "creative_writing", "general_factoid", "safety", "others"],
    "logic": ["logic"]
}

# ------ Extraction Logic --------
def extract_result(text):
    """
    Extracts content between <result> and </result> tags.
    """
    match = re.search(r'<result>(.*?)</result>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# ------ Core Processing --------
def finalize_results(original_file, response_file, output_file):
    """
    Parses LLM responses, validates against whitelist, and reports distribution.
    Implements correction: if math,logic then logic,logic. Also keeps previous linear_algebra/geometery handling.
    """
    total_count = 0
    format_failure_count = 0
    invalid_hallucination_count = 0

    # Statistics counters
    primary_dist = Counter()
    secondary_dist = defaultdict(Counter)
    hallucinated_labels = Counter() # To track what the model is inventing

    with open(original_file, 'r', encoding='utf-8') as f_orig, \
         open(response_file, 'r', encoding='utf-8') as f_resp, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_orig, line_resp in zip(f_orig, f_resp):
            if not line_orig.strip(): continue

            orig_data = json.loads(line_orig)
            resp_data = json.loads(line_resp)
            total_count += 1

            raw_llm_output = resp_data.get('response', '')
            extracted_content = extract_result(raw_llm_output)

            primary_cat = "unknown"
            secondary_cat = "unknown"
            is_valid = False

            if extracted_content:
                parts = [p.strip().lower() for p in extracted_content.split(',')]
                if len(parts) == 2:
                    p_candidate, s_candidate = parts

                    # ==== REWRITE for math,logic -> logic,logic ====
                    # If secondary is linear_algebra, replace with algebra.
                    if s_candidate == "linear_algebra":
                        s_candidate = "algebra"
                    # If primary is geometry, replace with math.
                    if p_candidate == "geometry":
                        p_candidate = "math"
                    # If it's math,logic re-categorize as logic,logic
                    if p_candidate == "math" and s_candidate == "logic":
                        p_candidate = "logic"
                        s_candidate = "logic"
                    # ==== END REWRITE ====

                    # Validation Check
                    if p_candidate in VALID_TAXONOMY and s_candidate in VALID_TAXONOMY[p_candidate]:
                        primary_cat = p_candidate
                        secondary_cat = s_candidate
                        is_valid = True
                    else:
                        invalid_hallucination_count += 1
                        hallucinated_labels[f"{p_candidate},{s_candidate}"] += 1
                else:
                    format_failure_count += 1
            else:
                format_failure_count += 1

            # Update original data
            orig_data['category'] = {
                "primary": primary_cat,
                "secondary": secondary_cat,
                "is_valid": is_valid # Add a flag for easy filtering later
            }

            # Update stats
            primary_dist[primary_cat] += 1
            secondary_dist[primary_cat][secondary_cat] += 1

            # Clean up and save
            if 'messages' in orig_data: del orig_data['messages']
            f_out.write(json.dumps(orig_data, ensure_ascii=False) + '\n')

    # ------ Detailed Distribution Report --------
    print("\n" + "="*60)
    print(f" FINAL TAXONOMY REPORT: {os.path.basename(original_file)}")
    print("="*60)
    print(f"Total Processed       : {total_count}")
    print(f"Format/Tag Failures   : {format_failure_count}")
    print(f"Invalid Hallucinations: {invalid_hallucination_count}")
    print("-" * 60)

    if hallucinated_labels:
        print("TOP HALLUCINATED LABELS (Discarded):")
        for label, count in hallucinated_labels.most_common(5):
            print(f"  - {label:<30} : {count} times")
        print("-" * 60)

    print(f"{'PRIMARY CATEGORY':<25} | {'COUNT':<10} | {'PERCENT'}")
    print("-" * 60)

    for pri, pri_count in primary_dist.most_common():
        percentage = (pri_count / total_count) * 100
        print(f"{pri:<25} | {pri_count:<10} | {percentage:>6.2f}%")

        for sec, sec_count in secondary_dist[pri].most_common():
            sub_percentage = (sec_count / pri_count) * 100
            print(f"  - {sec:<21} : {sec_count:<10} ({sub_percentage:>5.1f}% of {pri})")
        print("-" * 60)

    print(f"Process complete. Cleaned data saved to: {output_file}\n")

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify dataset with strict whitelist validation.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    finalize_results(args.original, args.response, args.output)