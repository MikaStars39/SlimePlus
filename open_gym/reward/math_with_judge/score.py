# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone scoring function extracted from app.py
# Removes: FastAPI server, async operations, LLM judge
# Keeps: Library-based verification using math_verify

import contextlib
import logging
from io import StringIO
from typing import Optional

from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


class MathVerifier:
    """Math answer verifier using math_verify library."""

    def __init__(self):
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self._library_verifier = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    @staticmethod
    @contextlib.contextmanager
    def _mute_output():
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    def verify_answer(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        """
        Verify the correctness of a generated answer using math_verify library.

        This functionality is migrated from Nemo RL.
        https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py

        Args:
            expected_answer: Ground truth answer (without \\boxed{})
            generated_answer: Model-generated answer text (should contain \\boxed{answer})

        Returns:
            (reward, extracted_answer): reward is 0.0 or 1.0, extracted_answer is the parsed answer
        """
        try:
            ground_truth_parsable = "\\boxed{" + expected_answer + "}"
            with self._mute_output():
                ret_score, extracted_answer = self._library_verifier([ground_truth_parsable], [generated_answer])

            reward = float(ret_score)

            if extracted_answer is not None:
                # Make sure the extracted answer has two elements.
                assert len(extracted_answer) == 2

                extracted_gold, extracted_prediction = extracted_answer

                # Get the extracted answer.
                for pred in extracted_prediction:
                    if any(grader.verify(gold, pred) for gold in extracted_gold):
                        extracted_answer = pred
                        break
                else:
                    # If no match is found, that means all the answers are
                    # incorrect. The first prediction is used as the extracted
                    # answer.
                    extracted_answer = extracted_prediction[0]

            return reward, extracted_answer

        # It's possible to emit a TimeoutException and that wouldn't be caught since
        # it actually subclasses from BaseException and math-verify itself does not
        # catch it.
        except (Exception, TimeoutException):
            return 0.0, None


# Global singleton verifier
_global_verifier = None


def _get_verifier() -> MathVerifier:
    global _global_verifier
    if _global_verifier is None:
        _global_verifier = MathVerifier()
    return _global_verifier


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Standalone math scoring function for verl reward manager.

    Args:
        model_output: Model-generated answer text (should contain \\boxed{answer})
        extra_info: Dictionary from parquet containing at least:
            - expected_answer: Ground truth answer

    Returns:
        float: 1.0 (correct) or 0.0 (incorrect)
    """
    expected_answer = extra_info.get("expected_answer", "")
    if not expected_answer:
        return 0.0

    verifier = _get_verifier()
    reward, _ = verifier.verify_answer(str(expected_answer), model_output)
    return reward
