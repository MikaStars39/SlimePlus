from slime.rollout.rm_hub.f1 import f1_score
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.math_dapo_utils import compute_score
# from slime.rollout.rm_hub.ifbench import compute_ifbench_reward

def get_reward(
    response: str, 
    label: str, 
    reward_type: str,
):
    if reward_type == "f1":
        return f1_score(response, label)
    elif reward_type == "deepscaler":
        response = f"<think></think>\\boxed{{{response}}}"
        return get_deepscaler_rule_based_reward(response, label)
    elif reward_type == "dapo":
        response = f"Answer:{response}"
        return compute_score(response, label)
    else:
        raise ValueError(f"Invalid reward type: {reward_type}")

if __name__ == "__main__":
    response = "100"
    label = "100"
    print(get_reward(response, label, reward_type="dapo"))