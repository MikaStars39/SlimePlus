import re
from slime.rollout.rm_hub.f1 import f1_score
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.math_dapo_utils import compute_score

def extract_answer(text: str) -> str:
    """Extract answer from model response using regex (boxed or last value)."""
    if not text:
        return ""
    
    # 1. Try to find content inside \boxed{}
    boxed_match = re.findall(r'\\boxed\{(.*?)\}', text)
    if boxed_match:
        return boxed_match[-1].strip()
    
    # 2. Try to find "Final Answer: <val>"
    final_match = re.findall(r'[Ff]inal [Aa]nswer:\s*(.*)', text)
    if final_match:
        return final_match[-1].strip()
    
    # 3. Fallback: just return the stripped text (last line if multiple)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()

def get_reward(
    response: str, 
    label: str, 
    reward_type: str,
) -> float:
    """Evaluate response against label using specified reward type."""
    # Pre-clean labels and responses
    if not label or not str(label).strip():
        return 0.0
    
    # Clean up response (some models might still include "Answer: 123")
    response = response.replace("Answer:", "").strip()

    res = 0.0
    if reward_type == "f1":
        res = f1_score(response, label)
    elif reward_type == "deepscaler":
        # Deepscaler expects a specific format
        formatted_res = f"<think></think>\\boxed{{{response}}}"
        res = get_deepscaler_rule_based_reward(formatted_res, label)
    elif reward_type == "dapo":
        # Dapo expects "Answer:xxx"
        res = compute_score(f"Answer:{response}", label)
    else:
        raise ValueError(f"Invalid reward type: {reward_type}")

    # If the reward function returns a dict (common in slime), extract the score
    if isinstance(res, dict):
        score = float(res.get("score", res.get("reward", 0.0)))
    else:
        score = float(res)

    # For math datasets (dapo/deepscaler), we usually want binary scores (0 or 1)
    # This prevents small partial credits from inflating Pass@K
    if reward_type in ["dapo", "deepscaler"]:
        return 1.0 if score >= 1.0 else 0.0
    
    return score

if __name__ == "__main__":
    test_res = "The answer is \\boxed{204}"
    print(f"Extracted: {extract_answer(test_res)}")
    print(f"Reward: {get_reward('204', '204', 'dapo')}")
