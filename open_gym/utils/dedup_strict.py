import sys
import json
import os
import argparse
from typing import Set, Dict, List

# ------ Similarity Logic --------
def get_word_set(text: str) -> Set[str]:
    """
    Standardize text into a set of words.
    Linux principle: Keep it simple and focused.
    """
    if not text:
        return set()
    return set(text.split())

def is_similar(target_words: Set[str], ref_words_list: List[Set[str]], threshold: float) -> bool:
    """
    Check if target word set overlaps significantly with any in the list.
    """
    if not target_words:
        return False
    
    target_len = len(target_words)
    for ref_words in ref_words_list:
        intersection_len = len(target_words.intersection(ref_words))
        if intersection_len / target_len > threshold:
            return True
    return False

# ------ Knowledge Base --------
class GlobalKnowledgeBase:
    """
    Stateful engine to track unique entries across multiple files.
    SOLID Principle: Single Responsibility.
    """
    def __init__(self, threshold: float):
        self.seen_prompts = set()
        self.label_map: Dict[str, List[Set[str]]] = {}
        self.threshold = threshold

    def is_duplicate(self, prompt: str, label: str) -> bool:
        """
        Check against the global state and update if unique.
        """
        # 1. Exact match check
        if prompt in self.seen_prompts:
            return True
        
        # 2. Label + Similarity check
        current_words = get_word_set(prompt)
        if label and label in self.label_map:
            if is_similar(current_words, self.label_map[label], self.threshold):
                return True

        # If it's unique, register it
        self.seen_prompts.add(prompt)
        if label:
            if label not in self.label_map:
                self.label_map[label] = []
            self.label_map[label].append(current_words)
        
        return False

# ------ Core File Processing --------
def process_file(file_path: str, kb: GlobalKnowledgeBase):
    """
    Handles IO for a single file, saving output to the same directory.
    """
    # Generate output paths in the same directory as the source
    file_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    out_path = os.path.join(file_dir, f"{base_name}_deduped.jsonl")
    rem_path = os.path.join(file_dir, f"{base_name}_removed.jsonl")

    print(f"[*] Processing: {file_path}")
    
    dup_count = 0
    total_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out, \
         open(rem_path, 'w', encoding='utf-8') as f_rem:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # Deduplication logic
                if kb.is_duplicate(data.get('prompt', ''), data.get('label')):
                    f_rem.write(line + '\n')
                    dup_count += 1
                else:
                    f_out.write(line + '\n')
                    
            except json.JSONDecodeError:
                continue

    print(f"    - Finished: {total_count} lines processed.")
    print(f"    - Saved: {out_path} (Keep: {total_count - dup_count})")
    print(f"    - Saved: {rem_path} (Removed: {dup_count})")

# ------ CLI Entry Point --------
def main():
    parser = argparse.ArgumentParser(description="Multi-file mutual deduplication.")
    parser.add_argument("files", nargs="+", help="Input .jsonl files.")
    parser.add_argument("-t", "--threshold", type=float, default=0.95, 
                        help="Similarity threshold for word overlap")
    
    args = parser.parse_args()
    
    # Initialize global state
    kb = GlobalKnowledgeBase(args.threshold)

    # Process files sequentially to maintain order priority
    for f in args.files:
        if os.path.exists(f):
            process_file(f, kb)
        else:
            print(f"[!] File not found: {f}")

    print("\n[✔] All tasks completed.")

if __name__ == "__main__":
    main()