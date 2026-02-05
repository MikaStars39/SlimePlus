#!/usr/bin/env python3
"""
Enrich all_scores.jsonl with complete conversations from the original file.
This adds the full conversation context based on original_idx.
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import subprocess

def load_original_conversations(original_file):
    """
    Load conversations from original file, indexed by line number (0-based).
    Returns a dict: {line_idx: conversations}
    """
    print(f"Loading conversations from {original_file}...")
    
    # Count lines for progress bar
    try:
        total_lines = int(subprocess.check_output(['wc', '-l', original_file]).split()[0])
    except:
        print("Warning: Could not count lines, progress bar may be inaccurate")
        total_lines = None
    
    conversations_map = {}
    
    with open(original_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Loading original data")):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                conversations = data.get('conversations', [])
                
                if conversations:
                    conversations_map[line_idx] = conversations
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_idx}: {e}")
                continue
    
    print(f"Loaded {len(conversations_map)} conversations")
    return conversations_map

def enrich_scores_with_conversations(scores_file, conversations_map, output_file):
    """
    Add full conversations to each score entry based on original_idx.
    """
    print(f"\nEnriching scores from {scores_file}...")
    
    # Count lines for progress bar
    try:
        total_lines = int(subprocess.check_output(['wc', '-l', scores_file]).split()[0])
    except:
        total_lines = None
    
    enriched_count = 0
    missing_count = 0
    
    with open(scores_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Enriching scores"):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                original_idx = data.get('original_idx')
                
                if original_idx is None:
                    print(f"Warning: Entry without original_idx: {data.keys()}")
                    f_out.write(line)
                    continue
                
                # Look up conversations
                if original_idx in conversations_map:
                    data['conversations'] = conversations_map[original_idx]
                    enriched_count += 1
                else:
                    print(f"Warning: No conversations found for original_idx={original_idx}")
                    missing_count += 1
                
                # Write enriched data
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse score entry: {e}")
                continue
    
    print("\n" + "=" * 60)
    print("Enrichment Statistics")
    print("=" * 60)
    print(f"Total enriched:       {enriched_count}")
    print(f"Missing conversations: {missing_count}")
    print(f"Success rate:         {enriched_count/(enriched_count+missing_count)*100:.2f}%")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Enrich scores with full conversations from original file"
    )
    parser.add_argument(
        "--scores", 
        required=True, 
        help="Input scores JSONL file (e.g., all_scores.jsonl)"
    )
    parser.add_argument(
        "--original", 
        required=True, 
        help="Original JSONL file with conversations"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output enriched JSONL file"
    )
    
    args = parser.parse_args()
    
    # Load conversations map
    conversations_map = load_original_conversations(args.original)
    
    # Enrich scores
    enrich_scores_with_conversations(args.scores, conversations_map, args.output)
    
    print(f"\n✅ Done! Enriched data saved to: {args.output}")

if __name__ == "__main__":
    main()
