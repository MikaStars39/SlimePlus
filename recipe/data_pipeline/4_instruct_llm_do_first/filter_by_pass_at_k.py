import json
import argparse


def filter_by_pass_rate(input_file, threshold, output_high, output_low):
    """
    Filter data by pass_rate threshold and split into two files.
    
    Args:
        input_file: Input jsonl file with pass_rate field
        threshold: Pass rate threshold (0.0 to 1.0)
        output_high: Output file for records with pass_rate >= threshold
        output_low: Output file for records with pass_rate < threshold
    """
    high_count = 0
    low_count = 0
    total_count = 0
    missing_pass_rate_count = 0
    
    print(f"Reading input file: {input_file}")
    print(f"Using threshold: {threshold}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_high, 'w', encoding='utf-8') as f_high, \
         open(output_low, 'w', encoding='utf-8') as f_low:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # Check if pass_rate field exists
                if 'pass_rate' not in data:
                    print(f"Warning: Line {line_num} missing pass_rate field, skipping")
                    missing_pass_rate_count += 1
                    continue
                
                pass_rate = data['pass_rate']
                
                # Filter by threshold
                if pass_rate >= threshold:
                    f_high.write(json.dumps(data, ensure_ascii=False) + '\n')
                    high_count += 1
                else:
                    f_low.write(json.dumps(data, ensure_ascii=False) + '\n')
                    low_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} JSON decode failed: {e}")
                continue
            except (TypeError, ValueError) as e:
                print(f"Warning: Line {line_num} invalid pass_rate value: {e}")
                continue
    
    # Print statistics
    print("-" * 60)
    print("Filtering complete!")
    print(f"  - Total records processed:      {total_count}")
    print(f"  - Records missing pass_rate:    {missing_pass_rate_count}")
    print(f"  - Records with pass_rate >= {threshold}: {high_count} ({high_count/(total_count-missing_pass_rate_count)*100:.2f}%)" if total_count > missing_pass_rate_count else f"  - Records with pass_rate >= {threshold}: {high_count}")
    print(f"  - Records with pass_rate <  {threshold}: {low_count} ({low_count/(total_count-missing_pass_rate_count)*100:.2f}%)" if total_count > missing_pass_rate_count else f"  - Records with pass_rate <  {threshold}: {low_count}")
    print(f"  - High pass rate file:          {output_high}")
    print(f"  - Low pass rate file:           {output_low}")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter data by pass_rate threshold and split into two files"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input jsonl file with pass_rate field"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Pass rate threshold (0.0 to 1.0). Records >= threshold go to high file, < threshold go to low file"
    )
    parser.add_argument(
        "--output-high",
        required=True,
        help="Output file for records with pass_rate >= threshold"
    )
    parser.add_argument(
        "--output-low",
        required=True,
        help="Output file for records with pass_rate < threshold"
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")
    
    filter_by_pass_rate(args.input, args.threshold, args.output_high, args.output_low)