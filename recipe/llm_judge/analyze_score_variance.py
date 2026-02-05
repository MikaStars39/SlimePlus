#!/usr/bin/env python3
"""
Analyze the score variance (max - min) distribution across all questions.
This helps understand model response consistency.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def analyze_score_variance(extracted_file, output_file=None):
    """
    Calculate score variance (max - min) for each question and analyze distribution.
    """
    print(f"Loading extracted scores from {extracted_file}...")
    
    # Group by original_idx
    grouped = defaultdict(list)
    
    with open(extracted_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading scores"):
            if not line.strip():
                continue
            
            data = json.loads(line)
            original_idx = data['original_idx']
            total_score = data['scores']['total_score']
            grouped[original_idx].append(total_score)
    
    print(f"\nFound {len(grouped)} unique questions")
    print("Calculating score variance...")
    
    variances = []
    variance_details = []
    
    for original_idx in sorted(grouped.keys()):
        scores = grouped[original_idx]
        
        if len(scores) < 2:
            continue
        
        max_score = max(scores)
        min_score = min(scores)
        variance = max_score - min_score
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        variances.append(variance)
        variance_details.append({
            'original_idx': original_idx,
            'num_answers': len(scores),
            'max_score': max_score,
            'min_score': min_score,
            'variance': variance,
            'mean_score': mean_score,
            'std_score': std_score,
            'all_scores': sorted(scores, reverse=True)
        })
    
    # Calculate statistics
    variances = np.array(variances)
    stats = {
        'total_questions': len(variances),
        'mean_variance': float(np.mean(variances)),
        'median_variance': float(np.median(variances)),
        'std_variance': float(np.std(variances)),
        'min_variance': float(np.min(variances)),
        'max_variance': float(np.max(variances)),
        'percentiles': {
            '25th': float(np.percentile(variances, 25)),
            '50th': float(np.percentile(variances, 50)),
            '75th': float(np.percentile(variances, 75)),
            '90th': float(np.percentile(variances, 90)),
            '95th': float(np.percentile(variances, 95)),
            '99th': float(np.percentile(variances, 99))
        }
    }
    
    # Distribution by bins
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    distribution = {}
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        count = np.sum((variances >= low) & (variances < high))
        if i == len(bins) - 2:  # Last bin includes upper bound
            count = np.sum((variances >= low) & (variances <= high))
        distribution[f'{low}-{high}'] = int(count)
    
    # Print results
    print("\n" + "=" * 70)
    print("Score Variance Analysis (Max - Min per Question)")
    print("=" * 70)
    print(f"\nTotal questions analyzed:  {stats['total_questions']}")
    print(f"\nVariance Statistics:")
    print(f"  Mean:                    {stats['mean_variance']:.2f}")
    print(f"  Median:                  {stats['median_variance']:.2f}")
    print(f"  Std Dev:                 {stats['std_variance']:.2f}")
    print(f"  Min:                     {stats['min_variance']:.2f}")
    print(f"  Max:                     {stats['max_variance']:.2f}")
    
    print(f"\nPercentiles:")
    for pct, val in stats['percentiles'].items():
        print(f"  {pct:5s}:                   {val:.2f}")
    
    print(f"\nScore Variance Distribution:")
    print(f"{'Range':>12s}  {'Count':>8s}  {'Percentage':>10s}  {'Bar':s}")
    print("-" * 70)
    
    for bin_range in [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]:
        count = distribution[bin_range]
        percentage = count / stats['total_questions'] * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{bin_range:>12s}  {count:8d}  {percentage:9.2f}%  {bar}")
    
    print("=" * 70)
    
    # Interpretation
    print("\n📊 Interpretation:")
    if stats['mean_variance'] < 10:
        print("  ✓ Low variance: Model responses are highly consistent")
    elif stats['mean_variance'] < 20:
        print("  ⚠ Moderate variance: Some inconsistency in model responses")
    else:
        print("  ⚠ High variance: Significant inconsistency in model responses")
    
    if stats['percentiles']['75th'] < 15:
        print("  ✓ 75% of questions have variance < 15 points")
    
    # Find high variance questions
    high_variance_threshold = stats['percentiles']['90th']
    high_variance_questions = [d for d in variance_details if d['variance'] >= high_variance_threshold]
    high_variance_questions.sort(key=lambda x: x['variance'], reverse=True)
    
    print(f"\n🔍 Top 10 highest variance questions (≥{high_variance_threshold:.1f}):")
    for i, q in enumerate(high_variance_questions[:10], 1):
        print(f"  {i:2d}. Question {q['original_idx']:6d}: "
              f"variance={q['variance']:5.1f} (scores: {q['all_scores']})")
    
    # Save detailed results if output file specified
    if output_file:
        output_data = {
            'statistics': stats,
            'distribution': distribution,
            'variance_details': variance_details
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Detailed results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze score variance (max-min) distribution"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Extracted scores JSONL file (from merge_and_extract.py)"
    )
    parser.add_argument(
        "--output", 
        help="Optional: output JSON file with detailed variance analysis"
    )
    
    args = parser.parse_args()
    analyze_score_variance(args.input, args.output)
    print("\n✅ Analysis complete!")
