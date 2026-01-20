import argparse
import asyncio
import logging
from pathlib import Path

from mika_eval.backend import BatchInferenceEngine
from mika_eval.config import TaskPaths
from mika_eval.llm_judge.llm_judge import llm_judge
from mika_eval.utils import setup_logging

# --------------------------------------------
# 1. prepare data
# 2. inference
# 3. llm extract the answer
# 4. judge
# 5. llm re-judge and genrm
# 6. summarize the metrics
# --------------------------------------------


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
        from mika_eval.tasks import prepare_pass_at_k_jsonl
        from mika_eval.utils import apply_template_to_jsonl

        logging.info(f"Preparing data for {self.args.dataset}...")
        prepare_pass_at_k_jsonl(
            config_str=self.args.dataset,
            output_file=str(self.paths.data_file),
            cache_dir=self.args.cache_dir,
        )

        logging.info(
            f"Applying prompt/chat template for inference (format={self.args.prompt_format})..."
        )
        apply_template_to_jsonl(
            input_file=str(self.paths.data_file),
            output_file=str(self.paths.formatted_input_file),
            model_path=str(self.args.model),
            user_template=self.args.prompt_format,
            system_prompt=self.args.system_prompt,
        )
        return self.paths.formatted_input_file

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
        from mika_eval.reward import eval_results

        eval_results(
            eval_output_file=self.paths.eval_output_file,
            score_output_file=self.paths.score_output_file,
            final_eval_output_file=self.paths.final_eval_output_file,
        )
        
        return self.paths.final_eval_output_file
