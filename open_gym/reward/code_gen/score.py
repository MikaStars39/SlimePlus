# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone scoring function extracted from app.py
# Removes: FastAPI server, async operations, ray.remote
# Keeps: Code extraction and unit test verification logic

import multiprocessing
from typing import Any, Dict, List, Optional

from mjnemogym.code_gen.lcb_integration.compute_code_generation_metrics import check_correctness
from mjnemogym.code_gen.lcb_integration.extraction_utils import LMStyle, extract_code
from pydantic import BaseModel


class UnitTests(BaseModel):
    """LiveCodeBench format for unit tests."""
    inputs: List[str]
    outputs: List[str]
    fn_name: Optional[str] = None


def verify_code(
    model_output: str,
    unit_tests: Dict[str, Any],
    timeout_secs: int = 10,
    debug: bool = False,
) -> tuple[float, Optional[str], Optional[List[bool]], Optional[Dict[str, Any]]]:
    """
    Verify code generation - core logic from app.py CompCodingResourcesServer.verify()

    Args:
        model_output: Model-generated code text
        unit_tests: Dict with "inputs", "outputs", and optional "fn_name"
        timeout_secs: Timeout for unit test execution
        debug: Enable debug output

    Returns:
        (reward, extracted_code, result, metadata):
            - reward: 1.0 if all tests pass, 0.0 otherwise
            - extracted_code: The extracted code from model output
            - result: List of test results
            - metadata: Additional metadata from execution
    """
    if not model_output or not model_output.strip():
        return 0.0, None, None, None

    # Validate unit tests
    tests = UnitTests.model_validate(unit_tests)

    # Extract code (code fence or raw)
    code = extract_code(model_output, LMStyle.OpenAIChat)
    if not code:
        return 0.0, None, None, None

    # Run unit tests (synchronously, no ray)
    sample = {"input_output": tests.model_dump_json()}

    try:
        result, metadata = check_correctness(
            sample,
            code,
            timeout_secs,
            debug,
        )
        reward = 1.0 if all(r == True for r in result) else 0.0
        return reward, code, result, metadata
    except Exception as e:
        if debug:
            print(f"Error during code verification: {e}")
        return 0.0, code, None, None


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Standalone code generation scoring function for verl reward manager.

    Args:
        model_output: Model-generated code text
        extra_info: Dictionary from parquet containing:
            - verifier_metadata.unit_tests: Unit tests with inputs, outputs, fn_name
            - timeout_secs: Optional timeout (default: 10)
            - debug: Optional debug flag (default: False)

    Returns:
        float: 1.0 (all tests pass) or 0.0 (any test fails)
    """
    # Extract unit tests from verifier_metadata
    verifier_metadata = extra_info.get("verifier_metadata", {})
    unit_tests = verifier_metadata.get("unit_tests")

    if not unit_tests:
        return 0.0

    timeout_secs = extra_info.get("timeout_secs", 10)
    debug = extra_info.get("debug", False)

    reward, _, _, _ = verify_code(
        model_output=model_output,
        unit_tests=unit_tests,
        timeout_secs=timeout_secs,
        debug=debug,
    )
    return reward
