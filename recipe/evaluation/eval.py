from pathlib import Path
from open_gym.utils import setup_logging

import argparse
import asyncio
import logging
from pathlib import Path

from open_gym.backend import BatchInferenceEngine
from open_gym.config import TaskPaths
from open_gym.llm_judge.llm_judge import llm_judge
from open_gym.utils import setup_logging

from dataclasses import dataclass

# --------------------------------------------
# 1. prepare data
# 2. inference
# 3. llm extract the answer
# 4. judge
# 5. llm re-judge and genrm
# 6. summarize the metrics
# --------------------------------------------

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


class TaskManager:
    def __init__(self, args: argparse.Namespace, result_dir: Path) -> None:
        self.args = args
        self.result_dir = Path(result_dir)
        self.paths = TaskPaths.from_result_dir(self.result_dir)

    @property
    def eval_model_path(self) -> str:
        return str(self.args.eval_model or self.args.model)

    def setup(self) -> None:
        setup_logging(self.result_dir)

    def prepare_data(self) -> Path:
        """
        Step 1: Prepare dataset jsonl + apply prompt/chat template for inference.

        Returns:
            Path to the formatted jsonl file used as inference input.
        """
        from open_gym.tasks import prepare_pass_at_k_jsonl
        from open_gym.utils import apply_template_to_jsonl

        logging.info(f"Preparing data for {self.args.dataset}...")

        dataset_configs = []
        for item in self.args.dataset.split(","):
            name, k_val = item.split("@")
            dataset_configs.append((name.strip(), int(k_val.strip())))

        out_dir = os.path.dirname(str(self.paths.data_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(str(self.paths.data_file), "w", encoding="utf-8") as f_out:
            for ds_name, k in dataset_configs:

                logging.info(f"Processing {ds_name} (repeat {k} times)...")
                
                loader_name = f"load_{ds_name.replace('-', '_')}"
                loader = getattr(tasks, loader_name, None)
                if loader is None:
                    raise ValueError(
                        f"Could not find loader '{loader_name}' for dataset '{ds_name}'. "
                        f"Please implement '{loader_name}(dataset_name, cache_dir, k, f_out)'."
                    )
                loader(ds_name, self.args.cache_dir, k, f_out)

        logging.info(f"Successfully generated {str(self.paths.data_file)}.")

        logging.info(
            f"Applying prompt/chat template for inference (format={self.args.prompt_format})..."
        )

    def apply_template_to_jsonl(
        input_file: str,
        output_file: str,
        model_path: str,
    ):
        
        from open_gym.utils import apply_template_to_jsonl
        apply_template_to_jsonl(
            input_file=input_file,
            output_file=output_file,
            model_path=model_path,
        )

    def inference(self) -> Path:
        """
        Step 2: Run offline inference for the main model.

        Returns:
            Path to inference_results.jsonl
        """
        resume = self.paths.infer_output_file.exists()
        if resume:
            logging.info(f"Resuming inference from {self.paths.infer_output_file}")

        async def _run():
            engine_args = {
                "model_path": self.args.model,
                "dp_size": self.args.dp_size,
                "tp_size": self.args.tp_size,
                "max_inflight": self.args.max_concurrency,
                "mem_fraction_static": self.args.gpu_memory_utilization,
            }
            async with BatchInferenceEngine(**engine_args) as engine:
                await engine.run(
                    input_file=str(self.paths.formatted_input_file),
                    output_file=str(self.paths.infer_output_file),
                    sampling_params={
                        "temperature": self.args.temperature,
                        "top_p": self.args.top_p,
                        "max_new_tokens": self.args.max_new_tokens,
                    },
                    resume=resume,
                )

        asyncio.run(_run())
        return self.paths.infer_output_file

    def llm_evaluation(self) -> Path:
        """
        Step 3: LLM extraction (answer extraction) to produce eval_results.jsonl.

        Returns:
            Path to eval_results.jsonl (merged)
        """
        return llm_judge(
            eval_output_file=self.paths.eval_output_file,
            infer_output_file=self.paths.infer_output_file,
            eval_input_file=self.paths.eval_input_file,
            no_eval_output_file=self.paths.no_eval_output_file,
            eval_model_path=self.eval_model_path,
            eval_chat_input_file=self.paths.eval_chat_input_file,
            dp_size=self.args.dp_size,
            tp_size=self.args.tp_size,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            eval_temperature=self.args.eval_temperature,
            eval_top_p=self.args.eval_top_p,
            eval_max_new_tokens=self.args.eval_max_new_tokens,
            max_concurrency=self.args.max_concurrency,
        )

    def calculate_metrics(self) -> Path:
        """
        Step 4: Calculate accuracy/metrics and write final.jsonl

        Input:
            eval_output_file: merged from noth eval and non-eval

        Returns:
            Patn to score_output_file
            Path to final.jsonl (only metrics)
        """
        from open_gym.reward import eval_results

        eval_results(
            eval_output_file=self.paths.eval_output_file,
            score_output_file=self.paths.score_output_file,
            final_eval_output_file=self.paths.final_eval_output_file,
        )
        
        return self.paths.final_eval_output_file



def main() -> None:
    args = parse_eval_args()
    result_dir = Path(args.result_dir)
    setup_logging(result_dir)
    task_manager = TaskManager(args=args, result_dir=result_dir)

    # ------------------------------ 1. Prepare ------------------------------
    if args.mode in ["all", "prepare"]:
        task_manager.prepare_data()
    
    # ------------------------------ 2. Inference ------------------------------
    if args.mode in ["all", "infer"]:
        task_manager.inference()

    # ------------------------------ 3. LLM Eval ------------------------------
    if args.mode in ["all", "llm-eval"]:
        task_manager.llm_evaluation()

    # ------------------------------ 4. Calculate Metrics ------------------------------
    if args.mode in ["all", "metrics"]:
        task_manager.calculate_metrics()

if __name__ == "__main__":
    main()
