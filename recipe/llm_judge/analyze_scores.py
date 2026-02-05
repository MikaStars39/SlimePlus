#!/usr/bin/env python3
"""
Analyzes LLM scoring results and generates statistical reports.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_scores(scores_file):
    """Analyzes scoring results and generates a report."""
    
    # Collect scores for each output
    scores_by_output = defaultdict(lambda: {
        'correctness': [],
        'logic': [],
        'clarity': [],
        'completeness': [],
        'total_score': []
    })
    
    total_samples = 0
    total_evaluations = 0
    
    with open(scores_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            total_samples += 1
            
            for output_name, score_data in data.get('scores', {}).items():
                total_evaluations += 1
                scores_by_output[output_name]['correctness'].append(score_data['correctness'])
                scores_by_output[output_name]['logic'].append(score_data['logic'])
                scores_by_output[output_name]['clarity'].append(score_data['clarity'])
                scores_by_output[output_name]['completeness'].append(score_data['completeness'])
                scores_by_output[output_name]['total_score'].append(score_data['total_score'])
    
    # Generate report
    print("\n" + "=" * 80)
    print("LLM Scoring Results Analysis Report")
    print("=" * 80)
    print(f"\nTotal samples: {total_samples}")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Avg evaluations per sample: {total_evaluations / total_samples if total_samples > 0 else 0:.2f}")
    
    print("\n" + "-" * 80)
    print("Scoring Statistics per Model Output")
    print("-" * 80)
    
    # Sort by output name
    output_names = sorted(scores_by_output.keys())
    
    for output_name in output_names:
        scores = scores_by_output[output_name]
        
        print(f"\n📊 {output_name}")
        print(f"   Sample count: {len(scores['total_score'])}")
        print(f"\n   Total Score (0-100):")
        print(f"      Mean:   {np.mean(scores['total_score']):.2f}")
        print(f"      Median: {np.median(scores['total_score']):.2f}")
        print(f"      Std:    {np.std(scores['total_score']):.2f}")
        print(f"      Max:    {np.max(scores['total_score']):.0f}")
        print(f"      Min:    {np.min(scores['total_score']):.0f}")
        
        print(f"\n   Dimension Scores:")
        print(f"      Correctness (0-50):  {np.mean(scores['correctness']):.2f} ± {np.std(scores['correctness']):.2f}")
        print(f"      Logic (0-25):        {np.mean(scores['logic']):.2f} ± {np.std(scores['logic']):.2f}")
        print(f"      Clarity (0-15):      {np.mean(scores['clarity']):.2f} ± {np.std(scores['clarity']):.2f}")
        print(f"      Completeness (0-10): {np.mean(scores['completeness']):.2f} ± {np.std(scores['completeness']):.2f}")
    
    # Ranking
    print("\n" + "-" * 80)
    print("Model Output Ranking (by Mean Total Score)")
    print("-" * 80)
    
    ranking = []
    for output_name in output_names:
        avg_score = np.mean(scores_by_output[output_name]['total_score'])
        ranking.append((output_name, avg_score))
    
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (output_name, avg_score) in enumerate(ranking, 1):
        print(f"{rank:2d}. {output_name:20s} - Mean Score: {avg_score:6.2f}")
    
    # Best performance per dimension
    print("\n" + "-" * 80)
    print("Best Performance per Dimension")
    print("-" * 80)
    
    dimensions = [
        ('correctness', 'Correctness', 50),
        ('logic', 'Logic', 25),
        ('clarity', 'Clarity', 15),
        ('completeness', 'Completeness', 10)
    ]
    
    for dim_key, dim_name, max_score in dimensions:
        best_output = max(output_names, 
                         key=lambda x: np.mean(scores_by_output[x][dim_key]))
        best_score = np.mean(scores_by_output[best_output][dim_key])
        print(f"{dim_name} (0-{max_score}): {best_output} - {best_score:.2f}")
    
    # Score distribution
    print("\n" + "-" * 80)
    print("Total Score Distribution Statistics")
    print("-" * 80)
    
    all_scores = []
    for scores in scores_by_output.values():
        all_scores.extend(scores['total_score'])
    
    if all_scores:
        bins = [0, 50, 60, 70, 80, 90, 100]
        labels = ['Fail (<50)', 'Pass (50-59)', 'Good (60-69)', 'Good+ (70-79)', 'Excellent (80-89)', 'Excellent+ (90-100)']
        
        for i in range(len(bins) - 1):
            count = sum(1 for s in all_scores if bins[i] <= s < bins[i+1])
            if i == len(bins) - 2:  # Last interval includes upper bound
                count = sum(1 for s in all_scores if bins[i] <= s <= bins[i+1])
            percentage = count / len(all_scores) * 100 if all_scores else 0
            bar = '█' * int(percentage / 2)
            print(f"{labels[i]:18s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    print("\n" + "=" * 80)


def compare_with_reference(scores_file, reference_file):
    """
    Comparative analysis with reference answers (if ground truth scores are available).
    """
    # TODO: Implement comparative analysis with reference data
    pass


def export_to_csv(scores_file, output_csv):
    """Exports to CSV format for further analysis."""
    import csv
    
    with open(scores_file, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', encoding='utf-8', newline='') as f_out:
        
        writer = csv.writer(f_out)
        # Write header
        writer.writerow([
            'sample_idx', 'output_name', 
            'correctness', 'logic', 'clarity', 'completeness', 'total_score',
            'brief_comment'
        ])
        
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            sample_idx = data['original_idx']
            
            for output_name, score_data in data.get('scores', {}).items():
                writer.writerow([
                    sample_idx,
                    output_name,
                    score_data['correctness'],
                    score_data['logic'],
                    score_data['clarity'],
                    score_data['completeness'],
                    score_data['total_score'],
                    score_data.get('brief_comment', '')
                ])
    
    print(f"\n✅ CSV file exported to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM scoring results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--scores',
        required=True,
        help='Scoring results JSONL file'
    )
    parser.add_argument(
        '--export-csv',
        help='Export CSV file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.scores).exists():
        print(f"❌ ERROR: File not found: {args.scores}")
        return
    
    # Analyze scoring results
    analyze_scores(args.scores)
    
    # Export to CSV (if requested)
    if args.export_csv:
        export_to_csv(args.scores, args.export_csv)


if __name__ == '__main__':
    main()
