import json
import argparse
import re
import os
from collections import Counter, defaultdict

# ------ Taxonomy Whitelist (Source of Truth) --------
VALID_TAXONOMY = {
    "math": ["arithmetic", "algebra", "geometry", "number_theory", "combinatorics", "probability_stats", "calculus", "discrete_math", "others"],
    "science": ["physics", "chemistry", "biology", "earth_space", "engineering", "medicine_health", "computer_science", "finance_accounting", "economics", "psychology", "materials_science", "public_health", "agriculture", "environmental_science", "others"],
    "humanities": ["political_science_sociology", "history_archaeology", "law", "philosophy_ethics", "literature_linguistics", "arts_design", "others"],
    "general": ["instruction_following", "commonsense", "creative_writing", "general_factoid", "safety", "others"],
    "logic": ["logic"]
}

def extract_result(text):
    match = re.search(r'<result>(.*?)</result>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def remap_logic(p, s):
    """
    Final Polish Remapping Logic (V4).
    针对: Logic, Anatomy, Pharmacy, Security, Forensics
    """
    # 1. Normalize
    p = p.lower().strip().replace(" ", "_")
    s = s.lower().strip().replace(" ", "_")
    
    # ==================================================
    # [优先] 绝对规则 (Absolute Overrides)
    # ==================================================
    
    # [Logic] 逻辑学是顶级分类 (161 times fix)
    if s == "logic":
        return "logic", "logic"

    # [CS] 暴力合并算法/编程 (之前的逻辑保持)
    cs_force_keywords = ["algorithm", "program", "software", "coding", "concurrency", "distributed", "compiler"]
    if any(k in s for k in cs_force_keywords):
        return "science", "computer_science"

    # ==================================================
    # [新增] 针对最新报错的特定修复
    # ==================================================

    # 1. [Medicine] 解剖、生理、药学、法医 -> medicine_health (200+ times fix)
    # 这里回答了你的问题：不新加分类，全部归入 medicine_health
    med_extended_keywords = [
        "anatomy", "physiology", "pharmacy", "pharmacology", 
        "forensics", "medical_health", "dentistry", "pathology", "clinical"
    ]
    if s in med_extended_keywords:
        return "science", "medicine_health"

    # 2. [Security] 安全的分流 (90+ times fix)
    if s == "security":
        # 如果模型认为是 general，归入 safety (符合 whitelist)
        if p == "general":
            return "general", "safety"
        # 如果模型认为是 science，归入 CS (通常指 cybersecurity)
        else:
            return "science", "computer_science"

    # 3. [Biology] 修正 Humanities 下的生物 (30 times fix)
    if s == "biology":
        return "science", "biology"

    # ==================================================
    # [常规] 基础同义词与父类纠正
    # ==================================================
    if s in ["other", "miscellaneous", "general"]: s = "others"
    if s in ["stat", "statistics", "statistical_analysis"]: s = "probability_stats"
    if s == "mathematics": p = "math"; s = "others"
    
    # 强制父类纠正
    if s in ["medicine_health", "public_health", "medicine", "health"]:
        p = "science"
        if s in ["medicine", "health"]: s = "medicine_health"
    
    if s in ["probability_stats", "combinatorics", "calculus", "algebra", "geometry", "number_theory", "discrete_math"]:
        p = "math"

    if s == "economics": p = "science"
    if s == "psychology": p = "science"
    if s in ["finance", "accounting", "finance_accounting"]:
        p = "science"; s = "finance_accounting"

    # ==================================================
    # [常规] 细粒度映射 (Mapping)
    # ==================================================
    
    # Math Optimization
    if s == "optimization":
        p = "math"; s = "discrete_math"

    # Public Policy
    if s in ["public_policy", "policy", "international_relations"]:
        p = "humanities"; s = "political_science_sociology"

    # Physics Sub-fields
    physics_keywords = {"thermodynamics", "fluid_mechanics", "mechanics", "optics", "quantum", "astrophysics"}
    if s in physics_keywords:
        p = "science"; s = "physics"

    # Biology Sub-fields (剩余的)
    bio_keywords = {"neuroscience", "ecology", "cell_biology", "genetics", "zoology", "botany"}
    if s in bio_keywords:
        p = "science"; s = "biology"

    # Earth Space
    if s in ["cosmology", "astronomy", "geology", "climate"]:
        p = "science"; s = "earth_space"
        
    # Engineering
    if "engineering" in s and s != "software_engineering" and s != "materials_science":
        p = "science"; s = "engineering"

    # ==================================================
    # 兜底与 Math 微调
    # ==================================================
    for primary, secondaries in VALID_TAXONOMY.items():
        if p in secondaries and p not in ["others", "logic"]:
            p = primary
            break

    if p == "math":
        if s in ["analysis", "differential_equations"]: s = "calculus"
        if s in ["graph_theory", "game_theory"]: s = "discrete_math"
        if s in ["matrix", "vector", "linear_algebra"]: s = "algebra"

    return p, s

def finalize_results(original_file, response_file, output_file, failed_file):
    total_count = 0
    invalid_hallucination_count = 0
    failed_extraction_count = 0
    valid_count = 0
    
    primary_dist = Counter()
    secondary_dist = defaultdict(Counter)
    hallucinated_labels = Counter()

    with open(original_file, 'r', encoding='utf-8') as f_orig, \
         open(response_file, 'r', encoding='utf-8') as f_resp, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
         open(failed_file, 'w', encoding='utf-8') as f_fail:
        
        for line_orig, line_resp in zip(f_orig, f_resp):
            if not line_orig.strip(): continue
            
            orig_data = json.loads(line_orig)
            resp_data = json.loads(line_resp)
            total_count += 1
            
            raw_llm_output = resp_data.get('response', '')
            extracted_content = extract_result(raw_llm_output)
            
            p_final, s_final = "unknown", "unknown"
            is_valid = False
            fail_reason = ""
            
            if extracted_content is None:
                # Failed to extract <result> tag
                failed_extraction_count += 1
                fail_reason = "NO_RESULT_TAG"
            elif extracted_content:
                parts = [p.strip() for p in extracted_content.split(',')]
                if len(parts) == 2:
                    # Apply the Remapping Logic
                    p_mapped, s_mapped = remap_logic(parts[0], parts[1])
                    
                    # Check Whitelist
                    if p_mapped in VALID_TAXONOMY and s_mapped in VALID_TAXONOMY[p_mapped]:
                        p_final, s_final = p_mapped, s_mapped
                        is_valid = True
                    else:
                        invalid_hallucination_count += 1
                        hallucinated_labels[f"{p_mapped},{s_mapped}"] += 1
                        fail_reason = f"INVALID_LABEL:{p_mapped},{s_mapped}"
                else:
                    fail_reason = "INVALID_FORMAT"
            
            # If invalid, save to failed file and skip output
            if not is_valid:
                fail_item = orig_data.copy()
                fail_item['raw_response'] = raw_llm_output
                fail_item['fail_reason'] = fail_reason
                fail_item['extracted_content'] = extracted_content
                f_fail.write(json.dumps(fail_item, ensure_ascii=False) + '\n')
                continue
            
            # Valid case: save to output
            valid_count += 1
            orig_data['category'] = {"primary": p_final, "secondary": s_final}
            primary_dist[p_final] += 1
            secondary_dist[p_final][s_final] += 1
            
            if 'messages' in orig_data: del orig_data['messages']
            f_out.write(json.dumps(orig_data, ensure_ascii=False) + '\n')

    # ------ Report --------
    print("\n" + "="*60)
    print(f" FINAL TAXONOMY REPORT: {os.path.basename(original_file)}")
    print("="*60)
    print(f"Total Processed           : {total_count}")
    print(f"Valid (saved to output)   : {valid_count}")
    print(f"Failed (saved separately) : {total_count - valid_count}")
    print(f"  - No result tag         : {failed_extraction_count}")
    print(f"  - Invalid labels        : {invalid_hallucination_count}")
    print("-" * 60)
    
    if hallucinated_labels:
        print("TOP UNRECOGNIZED LABELS (After Remapping):")
        for label, count in hallucinated_labels.most_common(10):
            print(f"  - {label:<35} : {count} times")
        print("-" * 60)

    if valid_count > 0:
        for pri, pri_count in primary_dist.most_common():
            percentage = (pri_count / valid_count) * 100
            print(f"{pri:<25} | {pri_count:<10} | {percentage:>6.2f}%")
            for sec, sec_count in secondary_dist[pri].most_common():
                sub_percentage = (sec_count / pri_count) * 100
                print(f"  - {sec:<21} : {sec_count:<10} ({sub_percentage:>5.1f}% of {pri})")
            print("-" * 60)
    
    print(f"Valid dataset saved to  : {output_file}")
    print(f"Failed cases saved to   : {failed_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed", required=True, help="File to save records where extraction failed or labels are invalid.")
    args = parser.parse_args()
    finalize_results(args.original, args.response, args.output, args.failed)