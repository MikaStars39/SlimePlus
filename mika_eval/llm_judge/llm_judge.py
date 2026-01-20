import asyncio
import logging
import os
from pathlib import Path

from mika_eval.backend.offline import BatchInferenceEngine
from mika_eval.llm_judge.extract import prepare_extraction_data
from mika_eval.utils import apply_template_to_jsonl, merge_two_jsonl_file


def llm_judge(
    eval_output_file: Path,
    infer_output_file: Path,
    eval_input_file: Path,
    no_eval_output_file: Path,
    eval_model_path: str,
    eval_chat_input_file: Path,
    dp_size: int,
    tp_size: int,
    gpu_memory_utilization: float,
    eval_temperature: float,
    eval_top_p: float,
    eval_max_new_tokens: int,
    max_concurrency: int,
) -> Path:

    # --------------------------------- check if need to skip ---------------------------------
    logging.info(f"Using model {eval_model_path} for extraction.")
    if eval_output_file.exists():
        logging.info(f"Eval results exist at {eval_output_file}, skipping.")
        return eval_output_file

    # --------------------------------- prepare extraction data from the model output ---------------------------------
    logging.info("Extracting answers using LLM...")
    prepare_extraction_data(
        input_file=infer_output_file,
        output_file=eval_input_file,
        output_no_eval_file=no_eval_output_file,
    )

    if os.path.getsize(eval_input_file) == 0:
        logging.info(
            f"Input file {eval_input_file} is empty, skipping LLM extraction inference."
        )
    else:
        logging.info("Applying chat template to eval_input.jsonl for LLM extraction...")
        apply_template_to_jsonl(
            input_file=str(eval_input_file),
            output_file=str(eval_chat_input_file),
            model_path=eval_model_path,
            user_template="blank",  # does not need to apply any other user template
        )

        async def _run():
            engine_args = {
                "model_path": eval_model_path,
                "dp_size": dp_size,
                "tp_size": tp_size,
                "max_inflight": max_concurrency,
                "mem_fraction_static": gpu_memory_utilization,
            }
            async with BatchInferenceEngine(**engine_args) as engine:
                await engine.run(
                    input_file=str(eval_chat_input_file),
                    output_file=str(eval_output_file),
                    sampling_params={
                        "temperature": eval_temperature,
                        "top_p": eval_top_p,
                        "max_new_tokens": eval_max_new_tokens,
                    },
                )

        asyncio.run(_run())
        logging.info(f"Inference completed for {eval_chat_input_file}")

    merge_two_jsonl_file(
        file1_path=eval_output_file,
        file2_path=no_eval_output_file,
        output_path=eval_output_file,
    )

    return eval_output_file
