import re
import json
from typing import Dict, List
from pathlib import Path
from slime.rollout.rm_hub.f1 import f1_score
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.math_dapo_utils import compute_score

from src.reward.math_verify_reward import grade_answer

def _calculate_matrics(
    updated_items: List[Dict]
) -> Dict[str, Dict[str, float]]:
    # Calculate final metrics
    raw_data = {}
    for item in updated_items:
        ds_name = item.get("source", "unknown")
        q_id = item.get("question_id", "unknown")
        score_tuple = item.get("score", (0.0, 0.0))
        # Extract the actual score (first element of tuple)
        score = float(score_tuple[0]) if isinstance(score_tuple, (list, tuple)) and len(score_tuple) > 0 else float(score_tuple)

        raw_data.setdefault(ds_name, {}).setdefault(q_id, []).append(score)

    final_results = {}
    for ds_name, q_map in raw_data.items():
        all_scores = []
        pass_at_k_scores = []

        for q_id, scores in q_map.items():
            all_scores.extend(scores)
            pass_at_k_scores.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

        final_results[ds_name] = {
            "avg_k": sum(all_scores) / len(all_scores) if all_scores else 0,
            "pass_k": sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0
        }

    # Overall metrics across all datasets
    overall_all_scores = []
    overall_pass_at_k_scores = []
    for ds_name, q_map in raw_data.items():
        for q_id, scores in q_map.items():
            overall_all_scores.extend(scores)
            overall_pass_at_k_scores.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

    final_results["overall"] = {
        "avg_k": sum(overall_all_scores) / len(overall_all_scores) if overall_all_scores else 0,
        "pass_k": sum(overall_pass_at_k_scores) / len(overall_pass_at_k_scores) if overall_pass_at_k_scores else 0,
    }

    return final_results

def extract_answer(text: str) -> str:
    """Extract answer from model response using regex (boxed or last value)."""
    if not text:
        return ""

    # NOTE:
    # - 一些 jsonl 里可能错误地写成 "\boxed{...}"（单反斜杠）。
    #   json.loads 会把 "\b" 解析成退格符 \x08，导致后续正则匹配不到。
    #   这里把退格符还原为字面量 "\b"（两字符：反斜杠 + b）。
    if "\x08" in text:
        text = text.replace("\x08", "\\b")

    # 1) 优先提取 \boxed{...}
    # 不能用简单正则去找 "第一个 }" 结束，因为 boxed 内容里常见嵌套花括号：
    #   \boxed{9.0 \times 10^{11}}
    # 这里用括号配对解析，确保提取完整 boxed 内容；若有多个，取最后一个。
    results = []
    for m in re.finditer(r"\\boxed\b", text):
        i = m.end()
        # 跳过 \boxed 后面的空白，找到第一个 '{'
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            continue

        i += 1  # skip '{'
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1

        if depth == 0:
            # i 已经指向匹配到的 '}' 之后
            results.append(text[start : i - 1].strip())

    if results:
        return results[-1]
    else:
        return None

def extract_metrics_from_file(eval_output_file: Path) -> Dict[str, Dict[str, float]]:
    updated_lines = []
    updated_items = []

    failed_items = []

    with open(eval_output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
           
            # ------------------ load the item and extract answer ------------------ 
            item = json.loads(line)
            label = item.get("label", "")
            raw_eval_res = item.get("response", "") 
            pred_ans = extract_answer(raw_eval_res)

            if pred_ans is None:
                failed_items.append(item)
                continue

            # ------------------ grade the answer ------------------ 
            score = grade_answer(f"${pred_ans}$", f"${label}$")
            
            # ------------------ update the item ------------------ 
            item["pred"] = pred_ans
            item["score"] = score
            updated_items.append(item)
            updated_lines.append(json.dumps(item, ensure_ascii=False))

    # ------------------ write the updated items to the file ------------------ 
    new_eval_output_file = eval_output_file.with_suffix(f".scored{eval_output_file.suffix}")
    with open(new_eval_output_file, "w", encoding="utf-8") as f:
        for line in updated_lines:
            item = json.loads(line)
            cleaned_item = {k: v for k, v in item.items() if k not in ["prompt", "response"]}
            f.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
    # ------------------ write the failed items to the file ------------------ 
    with open(eval_output_file.with_suffix(f".failed{eval_output_file.suffix}"), "w", encoding="utf-8") as f:
        for item in failed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ------------------ calculate the metrics and return ------------------ 
    return _calculate_matrics(updated_items)

if __name__ == "__main__":
    test_res = "The answer is \\boxed{m/frac{2}{3}}"
    print(f"Extracted: {extract_answer(test_res)}")
    print(f"Reward: {get_reward(extract_answer(test_res), 'm/frac{{2}}{{3}}', 'dapo')}")
