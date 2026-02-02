import json
import argparse
from collections import defaultdict


def calculate_pass_rate(rollout_file, reference_file, output_file):
    """
    Calculate the average pass rate (pass rate) for each question.
    
    Args:
        rollout_file: A jsonl file with all rollouts. Each rollout has id, question_id, pass_4b fields.
        reference_file: Reference jsonl file, one record per question.
        output_file: Output file. Adds the pass_rate field to each entry in the reference file.
    """
    # Step 1: Read all rollouts, group by question_id and count
    question_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    
    print(f"Reading rollout file: {rollout_file}")
    with open(rollout_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                question_id = data.get('question_id')
                pass_4b = data.get('4b_pass', False)
                
                if question_id is None:
                    print(f"Warning: Line {line_num} missing question_id, skipping")
                    continue
                
                question_stats[question_id]["total"] += 1
                if pass_4b:
                    question_stats[question_id]["passed"] += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} JSON decode failed: {e}")
                continue
    
    # Step 2: Calculate pass rate for each question
    question_pass_rates = {}
    for question_id, stats in question_stats.items():
        if stats["total"] > 0:
            pass_rate = stats["passed"] / stats["total"]
            question_pass_rates[question_id] = {
                "pass_rate": pass_rate,
                "passed_count": stats["passed"],
                "total_count": stats["total"]
            }
    
    print(f"Statistics collection done: {len(question_pass_rates)} questions found")
    
    # Step 3: Read reference file, add pass rate info
    print(f"Processing reference file: {reference_file}")
    processed_count = 0
    missing_count = 0
    
    with open(reference_file, 'r', encoding='utf-8') as f_ref, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_ref, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                question_id = data.get('question_id')
                
                if question_id is None:
                    print(f"Warning: Reference file line {line_num} missing question_id")
                    missing_count += 1
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    continue
                
                # Add pass rate information
                if question_id in question_pass_rates:
                    stats = question_pass_rates[question_id]
                    data['pass_rate'] = stats['pass_rate']
                    data['passed_count'] = stats['passed_count']
                    data['total_rollouts'] = stats['total_count']
                else:
                    print(f"Warning: question_id '{question_id}' not found in rollout file")
                    missing_count += 1
                    data['pass_rate'] = 0.0
                    data['passed_count'] = 0
                    data['total_rollouts'] = 0
                
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Reference file line {line_num} JSON decode failed: {e}")
                continue
    
    # Step 4: Print statistics
    print("-" * 60)
    print("Processing complete!")
    print(f"  - Number of questions processed: {processed_count}")
    print(f"  - Questions missing statistics: {missing_count}")
    
    if question_pass_rates:
        avg_pass_rate = sum(stats['pass_rate'] for stats in question_pass_rates.values()) / len(question_pass_rates)
        print(f"  - Average pass rate:          {avg_pass_rate:.4f} ({avg_pass_rate*100:.2f}%)")
    
    print(f"  - Output file:                {output_file}")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the average pass rate for each question"
    )
    parser.add_argument(
        "--rollout",
        required=True,
        help="Path to the jsonl file containing all rollouts"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference jsonl file (one record per question)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output jsonl file"
    )
    
    args = parser.parse_args()
    
    calculate_pass_rate(args.rollout, args.reference, args.output)