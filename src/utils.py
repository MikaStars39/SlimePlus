import logging
import sys
import time
import json
import re
import shutil
from typing import Any, Dict, List
from pathlib import Path

def merge_two_jsonl_file(file1_path: Path, file2_path: Path, output_path: Path) -> None:
    """
    Merge two JSONL files into one, writing the combined lines to output_path.
    If a key "id" is present in each JSON object, de-duplicate by "id",
    keeping the object from file2 if duplicates exist.

    Args:
        file1_path (Path): Path to the first JSONL file.
        file2_path (Path): Path to the second JSONL file.
        output_path (Path): Path to save the merged JSONL output.
    """
    seen = {}

    # if file1 does not exist, save file2 to output_path
    if not file1_path.exists():
        shutil.copy(file2_path, output_path)
        logging.info(f"File {file1_path} does not exist, copying {file2_path} to {output_path}")
        return
    elif not file2_path.exists():
        shutil.copy(file1_path, output_path)
        logging.info(f"File {file2_path} does not exist, copying {file1_path} to {output_path}")
        return

    # First read file1
    with file1_path.open("r", encoding="utf-8") as f1:
        for line in f1:
            obj = json.loads(line)
            obj_id = obj.get("id")
            if obj_id is not None:
                seen[obj_id] = obj
            else:
                # If no id, use str(line) as key to avoid simple duplicates
                seen[str(line).strip()] = obj

    # Then read file2, overwrite if duplicate id/line
    with file2_path.open("r", encoding="utf-8") as f2:
        for line in f2:
            obj = json.loads(line)
            obj_id = obj.get("id")
            if obj_id is not None:
                seen[obj_id] = obj
            else:
                seen[str(line).strip()] = obj

    # Write combined data to output_path
    with output_path.open("w", encoding="utf-8") as out:
        for obj in seen.values():
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

def setup_logging(result_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "eval.log"
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("eval")

def display_metrics_report(metrics_data: Dict[str, Dict[str, float]]):
    if not metrics_data:
        print("No metrics data to display.")
        return

    print("\n" + "="*60)
    print(f"{'Dataset':<25} | {'Avg@K':<12} | {'Pass@K':<12}")
    print("-" * 60)

    # 优先打印 overall（如果存在）
    if "overall" in metrics_data:
        stats = metrics_data["overall"]
        avg_k = stats["avg_k"]
        pass_k = stats["pass_k"]
        print(f"{'overall':<25} | {avg_k:>11.2%} | {pass_k:>11.2%}")
        print("-" * 60)

    for ds_name, stats in metrics_data.items():
        if ds_name == "overall":
            continue
        avg_k = stats["avg_k"]
        pass_k = stats["pass_k"]
        print(f"{ds_name:<25} | {avg_k:>11.2%} | {pass_k:>11.2%}")
        
    print("="*60 + "\n")

def calculate_and_print_metrics(eval_output_file: Path, cache_dir: str = None):
    from src.reward.reward import extract_metrics_from_file
    results = extract_metrics_from_file(eval_output_file)
    display_metrics_report(results)

def _extract_answer(text: str) -> str:
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
