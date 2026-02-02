import re
import math

# ==========================================
# Shared Helper Functions
# ==========================================

def levenshtein_distance(A: str, B: str) -> int:
    """
    Calculates the Levenshtein distance between two sequences A and B using Dynamic Programming.
    Used to replace difflib for similarity checks.
    """
    N, M = len(A), len(B)
    # Create an array of size (N+1)x(M+1)
    dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

    # Base Case: When N = 0
    for j in range(M + 1):
        dp[0][j] = j
    # Base Case: When M = 0
    for i in range(N + 1):
        dp[i][0] = i
    
    # Transitions
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],   # Insertion
                    dp[i][j-1],   # Deletion
                    dp[i-1][j-1]  # Replacement
                )

    return dp[N][M]

def extract_answer(llm_answer: str) -> str:
    """
    Extracts the answer part from a string following the pattern '... --- answer --- ...'.
    """
    pattern = r'.* --- (.*?) --- .*'
    match = re.search(pattern, llm_answer)
    return match.group(1) if match else llm_answer

# ==========================================
# Evaluator 1: Plot Unscrambling
# ==========================================

def extract_plot_summary(text: str) -> str:
    pattern = r'<PLOT_SUMMARY>(.*)</PLOT_SUMMARY>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        pattern = r'<PLOT_SUMMARY>(.*)'
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text

def plot_unscrambling_process_results(ground_truth: str, llm_answer: str, debug=False) -> float:
    """
    Evaluates how well the LLM ordered sentences compared to the ground truth.
    Uses Levenshtein distance on the sentence indices.
    """
    # Extract relevant text
    llm_answer = extract_plot_summary(llm_answer)

    # Split into sentences
    gt_sentences = [s.strip() for s in ground_truth.split('.')]
    ans_sentences = [
        s.strip() for s in llm_answer.split('.') 
        if s.strip() != '</PLOT_SUMMARY>' and s.strip() != '**End of Plot Summary**'
    ]

    # Filter empty sentences
    gt_sentences = [s for s in gt_sentences if s]
    ans_sentences = [s for s in ans_sentences if s]

    ans_ordering = []
    
    # Map ground truth sentences to the answer sentences
    for x in gt_sentences:
        if not ans_sentences:
            break
            
        # Replacement for difflib.get_close_matches:
        # Find the sentence in 'ans_sentences' with the smallest Levenshtein distance to 'x'
        best_match = None
        min_dist = float('inf')
        
        for candidate in ans_sentences:
            dist = levenshtein_distance(x, candidate)
            # Find the closest match (simulating cutoff=0.0 logic)
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
        
        if best_match:
            try:
                ans_ordering.append(ans_sentences.index(best_match))
            except ValueError:
                pass

    n_sentences_gt = len(gt_sentences)
    if n_sentences_gt == 0:
        return 0.0

    # Calculate edit distance between the expected index order (0, 1, 2...) and actual found order
    raw_distance = levenshtein_distance(list(range(len(gt_sentences))), ans_ordering)
    score = 1 - (raw_distance / n_sentences_gt)

    if debug and score < 1:
        print(f'[DEBUG-PLOT] INCORRECT Score: {score}')
        print(f'[DEBUG-PLOT] GT Sentences: {gt_sentences}')
        print(f'[DEBUG-PLOT] Ans Sentences: {ans_sentences}')

    return score

# ==========================================
# Evaluator 2: Connections Puzzle
# ==========================================

def last_boxed_only_string(string: str):
    """Parses LaTeX style \\boxed{content}."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    return retval

def remove_boxed(s: str):
    left = "\\boxed{"
    try:
        if s[: len(left)] == left and s[-1] == "}":
            return s[len(left) : -1]
        return None
    except Exception:
        return None

def group_words(words: list):
    """Groups a list of words into sets of 4."""
    groups = [set()]
    words = [w.strip().lower() for w in words]
    for word in words:
        if len(groups[-1]) == 4:
            groups.append(set())
        groups[-1].add(word)
    return groups

def connections_process_results_old(ground_truth: str, llm_answer: str, debug=False) -> float:
    """Evaluator for older puzzles (looks for bold text)."""
    bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer.replace('\n', ''))

    if not bold_words:
        if debug:
            print('[DEBUG-CONN-OLD] No bold words found.')
        return 0
    
    bold_words = [words.split(',') for words in bold_words]
    ground_truth_groups = group_words(ground_truth.split(','))
    
    max_score = 0
    # Check similarity against extracted bold groups
    for output_groups in list(map(group_words, bold_words)):
        correct_groups = 0
        for ground_truth_group in ground_truth_groups:
            for output_group in output_groups:
                # Check if all words in GT group exist in Output group
                if all([word in output_group for word in ground_truth_group]):
                    correct_groups += 1
                    break
        
        if len(ground_truth_groups) > 0:
            max_score = max(max_score, correct_groups / len(ground_truth_groups))
            
    if debug and max_score < 1:
        print(f'[DEBUG-CONN-OLD] Incorrect. Score: {max_score}')
    return max_score

def connections_process_results(ground_truth: str, llm_answer: str, debug=False) -> float:
    """Evaluator for newer puzzles (looks for <solution> tags or boxed text)."""
    
    # Try to find content inside <solution> tags
    solution_matches = re.findall(r'<solution>(.*?)<\/solution>', llm_answer)
    if not solution_matches:
        solution_matches = re.findall(r'<solution>(.*?)<\/solution>', llm_answer.replace('\n', ''))
    if not solution_matches:
        # Check for malformed closing tags scenarios
        solution_matches = re.findall(r'</solution>(.*?)<\/solution>', llm_answer)

    ground_truth_words = ground_truth.split(',')

    # Fallback to \boxed format if no xml tags found
    if len(solution_matches) == 0 and '\\boxed' in llm_answer:
        boxed = last_boxed_only_string(llm_answer)
        if boxed:
            no_box = remove_boxed(boxed)
            if no_box:
                # Clean up latex syntax
                clean_text = no_box.replace('\\text{', '').replace('}', '').replace('\\', '')
                solution_matches = [clean_text]

    # Clean newlines from matches
    solution_matches = [match.replace('\n', '') for match in solution_matches]

    if len(solution_matches) == 0:
        if debug:
            print('[DEBUG-CONN] No solution text found.')
        return 0
    
    # Handle multiple matches or single match
    if len(solution_matches) > 1:
        if debug:
            print('[DEBUG-CONN] Multiple solution texts found. Combining from last.')
        all_words = []
        num_words = len(ground_truth_words)
        for match in solution_matches:
            all_words.extend(match.split(','))
        solution_words = all_words[-num_words:]
    else:
        solution_words = solution_matches[-1].split(',')

    # Compare Groups
    llm_groups = group_words(solution_words)
    ground_truth_groups = group_words(ground_truth_words)

    correct_groups = 0
    for llm_group in llm_groups:
        if llm_group in ground_truth_groups:
            correct_groups += 1

    if len(ground_truth_groups) == 0:
        return 0
        
    score = correct_groups / len(ground_truth_groups)

    if debug and score < 1:
        print(f'[DEBUG-CONN] Incorrect. Score: {score}')
        print(f'GT Groups: {sorted([sorted(list(g)) for g in ground_truth_groups])}')
        print(f'LLM Groups: {sorted([sorted(list(g)) for g in llm_groups])}')

    return score

def get_connections_puzzle_evaluator(release_date: str):
    """Factory function to get the correct evaluator based on date."""
    if release_date < '2024-11-25':
        return connections_process_results_old
    return connections_process_results

# ==========================================
# Evaluator 3: Typos / Exact Match
# ==========================================

def typos_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    """
    Checks if the ground truth is present in the LLM answer.
    """
    parsed_answer = None

    # Priority 1: Extract from <solution> tags
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    if len(solution_matches) > 0:
        match = solution_matches[-1]
        parsed_answer = match
    else:
        # Priority 2: Clean tags and use separator pattern extraction
        parsed_answer = llm_answer.replace('<solution>', '').replace('</solution>', '')
        parsed_answer = extract_answer(parsed_answer)

    # Clean up whitespace/newlines
    parsed_answer = ' '.join(list(filter(None, parsed_answer.strip().split('\n'))))

    # Core Logic: Check for substring inclusion
    if int(ground_truth in parsed_answer):
        return 1

    # Simplified Debug Logic (No difflib)
    score = 0
    if debug and score == 0:
        print('[DEBUG-TYPO] INCORRECT')
        print(f'GT  : {ground_truth}')
        print(f'PRED: {parsed_answer}')

    return score


# ==========================================
# Test Cases / Usage Examples
# ==========================================

def language_judge(
    ground_truth: str, 
    llm_answer: str, 
    task_type: str
) -> float:
    """
    Judges the language of the LLM answer.
    """
    if task_type == "typos":
        return typos_process_results(ground_truth, llm_answer, debug=False)
    elif task_type == "connections":
        return connections_process_results(ground_truth, llm_answer, debug=False)
    elif task_type == "unscrambling":
        return plot_unscrambling_process_results(ground_truth, llm_answer, debug=False)
    else:
        raise ValueError(f"Invalid task type: {task_type}")

if __name__ == "__main__":
    print("=== TEST 1: Typos Evaluator ===")
    gt_typo = "We investigate simplified models of computer data networks and examine how the introduction of additional random links influences the performance of these net works. In general, the impact of additional random links on the performance of the network strongly depends on the routing algorithm used in the network. Significant performance gains can be achieved if the routing is based on \"geometrical distance\" or shortest path reduced table routing. With shortest path full table routing degradation of performance is observed."
    # Case: Correct answer inside text
    ans_typo_correct = "We investigate simplified models of computer data networks and examine how the introduction of additional random links influences the performance of these net works. In general, the impact of additional random links on the performance of the network strongly depends on the routing algorithm used in the network. Significant performance gains can be achieved if the routing is based on \"geometrical distance\" or shortest path reduced table routing. With shortest path full table routing degradation of performance is observed."
    # Case: Incorrect answer
    ans_typo_wrong = "We investigateee simplified models of computer data networks and examine how the introduction of additional random links influences the performance of these net works. In general, the impact of additional random links on the performance of the network strongly depends on the routing algorithm used in the network. Significant performance gains can be achieved if the routing is based on \"geometrical distance\" or shortest path reduced table routing. With shortest path full table routing degradation of performance is observed."
    
    print(f"Correct Case Score: {typos_process_results(ans_typo_correct, ans_typo_correct)}")
    print(f"Wrong Case Score:   {typos_process_results(ans_typo_correct, ans_typo_wrong, debug=True)}")
    print("-" * 30)

    print("\n=== TEST 2: Connections Evaluator ===")
    # 8 words total (2 groups of 4)
    gt_conn = "Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow"
    
    # Case: Perfect match inside tags
    ans_conn_correct = "<solution>Apple, Banana, Pear, Grape, Red, Blue, Green, Yellow</solution>"
    # Case: One group wrong (Orange instead of Grape)
    ans_conn_partial = "<solution>Apple, Banana, Pear, Orange, Red, Blue, Green, Yellow</solution>"

    evaluator = get_connections_puzzle_evaluator('2025-01-01') # Use new version
    print(f"Perfect Score: {evaluator(gt_conn, ans_conn_correct)}")
    print(f"Partial Score: {evaluator(gt_conn, ans_conn_partial, debug=True)}")
    print("-" * 30)

    print("\n=== TEST 3: Plot Unscrambling Evaluator ===")
    # GT: Sentence A. Sentence B. Sentence C.
    gt_plot = "The hero wakes up. He fights the dragon. He wins the gold."
    
    # Case: Perfect order
    ans_plot_perfect = "<PLOT_SUMMARY>The hero wakes up. He fights the dragon. He wins the gold.</PLOT_SUMMARY>"
    # Case: Swapped last two sentences
    ans_plot_swapped = "<PLOT_SUMMARY>The hero wakes up. He wins the gold. He fights the dragon.</PLOT_SUMMARY>"
    
    print(f"Perfect Order Score: {plot_unscrambling_process_results(gt_plot, ans_plot_perfect)}")
    print(f"Swapped Order Score: {plot_unscrambling_process_results(gt_plot, ans_plot_swapped, debug=True)}") 
    # Note: Score calculation logic -> Distance between [0,1,2] and [0,2,1] is 2 edits? 
    # Actually Levenshtein([0,1,2], [0,2,1]) cost is usually 2 (delete 1, insert 1) or 1 (swap if supported, this DP supports replace).
    # This specific DP: 0==0 match. 1!=2 replace. 2!=1 replace. Cost 2. Score = 1 - (2/3) = 0.33.