#!/usr/bin/env python3
"""
Analyze extracted scores to find best and worst answers for each question.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def analyze_best_worst(extracted_file, output_file, stats_file=None):
    """
    Group scores by original_idx and find best/worst answers.
    """
    print(f"Loading extracted scores from {extracted_file}...")
    
    # Group by original_idx
    grouped = defaultdict(list)
    
    with open(extracted_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Grouping scores"):
            if not line.strip():
                continue
            
            data = json.loads(line)
            original_idx = data['original_idx']
            grouped[original_idx].append(data)
    
    print(f"\nFound {len(grouped)} unique questions")
    print("Analyzing best and worst answers...")
    
    results = []
    score_stats = {
        'total_questions': 0,
        'avg_best_score': 0,
        'avg_worst_score': 0,
        'avg_score_range': 0,
        'score_distribution': defaultdict(int)
    }
    
    for original_idx in tqdm(sorted(grouped.keys()), desc="Analyzing"):
        answers = grouped[original_idx]
        
        if not answers:
            continue
        
        # Sort by total_score
        answers_sorted = sorted(answers, key=lambda x: x['scores']['total_score'], reverse=True)
        
        best_answer = answers_sorted[0]
        worst_answer = answers_sorted[-1]
        
        best_score = best_answer['scores']['total_score']
        worst_score = worst_answer['scores']['total_score']
        score_range = best_score - worst_score
        
        # Update statistics
        score_stats['total_questions'] += 1
        score_stats['avg_best_score'] += best_score
        score_stats['avg_worst_score'] += worst_score
        score_stats['avg_score_range'] += score_range
        
        # Score distribution (by 10-point bins)
        for ans in answers:
            bin_val = (ans['scores']['total_score'] // 10) * 10
            score_stats['score_distribution'][bin_val] += 1
        
        result = {
            'original_idx': original_idx,
            'question': best_answer['question'],
            'reference_answer': best_answer['reference_answer'],
            'num_answers': len(answers),
            'best_answer': {
                'output_idx': best_answer['output_idx'],
                'answer': best_answer['model_answer'],
                'scores': best_answer['scores']
            },
            'worst_answer': {
                'output_idx': worst_answer['output_idx'],
                'answer': worst_answer['model_answer'],
                'scores': worst_answer['scores']
            },
            'score_range': score_range,
            'all_scores': [ans['scores']['total_score'] for ans in answers_sorted]
        }
        
        results.append(result)
    
    # Calculate averages
    n = score_stats['total_questions']
    if n > 0:
        score_stats['avg_best_score'] /= n
        score_stats['avg_worst_score'] /= n
        score_stats['avg_score_range'] /= n
    
    # Write results
    print(f"\nWriting {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Analysis Statistics")
    print("=" * 60)
    print(f"Total questions:        {score_stats['total_questions']}")
    print(f"Avg best score:         {score_stats['avg_best_score']:.2f}")
    print(f"Avg worst score:        {score_stats['avg_worst_score']:.2f}")
    print(f"Avg score range:        {score_stats['avg_score_range']:.2f}")
    print("\nScore Distribution:")
    for score_bin in sorted(score_stats['score_distribution'].keys()):
        count = score_stats['score_distribution'][score_bin]
        pct = count / (score_stats['total_questions'] * 4) * 100  # 4 answers per question
        bar = '█' * int(pct / 2)
        print(f"  {score_bin:3d}-{score_bin+9:3d}: {count:6d} ({pct:5.1f}%) {bar}")
    print("=" * 60)
    
    # Write statistics if requested
    if stats_file:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(score_stats, f, ensure_ascii=False, indent=2)
        print(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze best and worst answers per question")
    parser.add_argument("--input", required=True, help="Extracted scores JSONL file")
    parser.add_argument("--output", required=True, help="Output file with best/worst analysis")
    parser.add_argument("--stats", help="Optional: output file for statistics JSON")
    
    args = parser.parse_args()
    analyze_best_worst(args.input, args.output, args.stats)
    print(f"\n✅ Done! Analysis saved to: {args.output}")
