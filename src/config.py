"""
Centralized CLI argument parsers and path configurations.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path


# ----------------------- Path Configuration -----------------------

@dataclass
class TaskPaths:
    """File paths for evaluation pipeline stages."""
    data_file: Path
    formatted_input_file: Path
    infer_output_file: Path
    eval_input_file: Path
    eval_chat_input_file: Path
    eval_output_file: Path
    no_eval_output_file: Path
    final_eval_output_file: Path
    score_output_file: Path

    @classmethod
    def from_result_dir(cls, result_dir: Path) -> "TaskPaths":
        """Create TaskPaths from a result directory."""
        return cls(
            data_file=result_dir / "data.jsonl",
            formatted_input_file=result_dir / "data.chat.jsonl",
            infer_output_file=result_dir / "inference_results.jsonl",
            eval_input_file=result_dir / "eval_input.jsonl",
            eval_chat_input_file=result_dir / "eval_input.chat.jsonl",
            eval_output_file=result_dir / "eval_results.jsonl",
            no_eval_output_file=result_dir / "no_eval_results.jsonl",
            final_eval_output_file=result_dir / "final.jsonl",
            score_output_file=result_dir / "score_results.jsonl",
        )


# ----------------------- Eval CLI -----------------------

def parse_eval_args() -> argparse.Namespace:
    """Parse CLI arguments for offline evaluation pipeline."""
    parser = argparse.ArgumentParser(description="MikaEval: Offline Inference and Evaluation")
    
    # Execution Mode
    parser.add_argument("--mode", choices=["prepare", "infer", "llm-eval", "metrics", "all"], default="infer")
    
    # Paths
    parser.add_argument("--result-dir", required=True, help="Directory for output results.")
    parser.add_argument("--model", required=True, help="Base model path for inference.")
    parser.add_argument("--eval-model", default=None, help="Model path for LLM-based answer extraction. Defaults to --model.")
    parser.add_argument("--dataset", default="aime2024", help="Dataset name/abbreviation.")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for datasets.")
    
    # SGLang Engine Config
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization.")
    
    # Sampling Params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)

    # LLM Extraction Sampling Params (deterministic + short)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    
    # Template
    parser.add_argument("--prompt-format", default="slime", help="Prompt template to use.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-concurrency", type=int, default=2000, help="Max concurrency for inference.")

    return parser.parse_args()


# ----------------------- RM Server CLI -----------------------

def parse_rm_args() -> argparse.Namespace:
    """Parse CLI arguments for Reward Model Server."""
    parser = argparse.ArgumentParser(description="Reward Model Server")
    
    # Model settings
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Path to the model")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--dp-size", type=int, default=1,
                        help="Data parallel size")
    parser.add_argument("--mem-fraction-static", type=float, default=0.90,
                        help="Static memory fraction for SGLang")
    
    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    
    # Server settings
    parser.add_argument("--host", default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--max-concurrent", type=int, default=16,
                        help="Maximum concurrent requests")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Request timeout in seconds")
    
    # Logging
    parser.add_argument("--output-dir", default="rm_logs",
                        help="Directory for log output")
    
    return parser.parse_args()
