# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone scoring function extracted from app.py
# Removes: FastAPI server, async operations, session management
# Keeps: Tool call state comparison logic

import json
from typing import Any, Dict, List

from mjnemogym.workplace_assistant.utils import execute_actions_and_reset_state


def convert_strs_to_lowercase(df):
    """Convert string columns to lowercase, except for specific fields."""
    # For some fields the case matters, so we don't convert them to lowercase
    fields_not_to_convert = ["status", "list_name", "board"]
    for col in df.columns:
        if col not in fields_not_to_convert:
            df[col] = df[col].str.lower()
    return df


def is_correct(predicted_actions: List[Dict[str, str]], ground_truth_actions: List[Dict[str, str]], error: str = None) -> bool:
    """
    Checks if the prediction is correct by comparing the state change after executing the actions.

    Args:
        predicted_actions: List of predicted actions (dicts with "name" and "arguments")
        ground_truth_actions: List of ground truth actions
        error: Optional error message from the prediction

    Returns:
        bool: True if predicted actions result in the same state as ground truth
    """
    if error:
        return False

    predict_env = execute_actions_and_reset_state(predicted_actions)
    ground_truth_env = execute_actions_and_reset_state(ground_truth_actions)

    # We allow for case-insensitive comparison of strings for most fields
    predicted_calendar_state = convert_strs_to_lowercase(predict_env["containers"]["calendar"]._calendar_events)
    predicted_email_state = convert_strs_to_lowercase(predict_env["containers"]["email"]._emails)
    predicted_analytics_state = convert_strs_to_lowercase(predict_env["containers"]["analytics"]._plots_data)
    predicted_project_management_state = convert_strs_to_lowercase(
        predict_env["containers"]["project_management"]._project_tasks
    )
    predicted_customer_relationship_manager_state = convert_strs_to_lowercase(
        predict_env["containers"]["customer_relationship_manager"]._crm_data
    )

    ground_truth_calendar_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["calendar"]._calendar_events
    )
    ground_truth_email_state = convert_strs_to_lowercase(ground_truth_env["containers"]["email"]._emails)
    ground_truth_analytics_state = convert_strs_to_lowercase(ground_truth_env["containers"]["analytics"]._plots_data)
    ground_truth_project_management_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["project_management"]._project_tasks
    )
    ground_truth_customer_relationship_manager_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["customer_relationship_manager"]._crm_data
    )

    return (
        predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )


def extract_function_calls(response_output: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract function calls from model response output.

    Args:
        response_output: List of response output items (messages, function calls, etc.)

    Returns:
        List of function call dicts with "name" and "arguments"
    """
    function_calls = []
    for message in response_output:
        # Handle dict format (from parquet)
        if isinstance(message, dict):
            if message.get("type") == "function_call":
                function_calls.append({
                    "name": message.get("name", ""),
                    "arguments": message.get("arguments", "{}"),
                })
        # Handle object format (from pydantic models)
        elif hasattr(message, "type") and message.type == "function_call":
            function_calls.append(message.model_dump())
    return function_calls


def verify_workplace_assistant(
    response_output: List[Dict[str, Any]],
    ground_truth: List[Dict[str, str]],
) -> float:
    """
    Verify workplace assistant - core logic from app.py WorkbenchResourcesServer.verify()

    Args:
        response_output: Model response output (list of messages/function calls)
        ground_truth: Ground truth actions list

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    # Extract predicted function calls
    predicted_function_calls = extract_function_calls(response_output)

    # Compare state after executing actions
    if is_correct(predicted_function_calls, ground_truth, None):
        return 1.0
    return 0.0


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Standalone workplace assistant scoring function for verl reward manager.

    Note: Unlike other domains, workplace_assistant uses structured function calls,
    not raw text. The model_output is expected to be a JSON string of the response
    output list, or the extra_info should contain "response_output" directly.

    Args:
        model_output: JSON string of response output (if provided)
        extra_info: Dictionary from parquet containing:
            - ground_truth: List of ground truth actions
            - response_output: Optional list of response output items

    Returns:
        float: 1.0 (correct) or 0.0 (incorrect)
    """
    ground_truth = extra_info.get("ground_truth", [])

    if not ground_truth:
        return 0.0

    # Get response output from extra_info or parse from model_output
    response_output = extra_info.get("response_output", [])

    if not response_output and model_output:
        try:
            response_output = json.loads(model_output)
        except (json.JSONDecodeError, TypeError):
            # If model_output is not valid JSON, try to extract function calls
            response_output = []

    if not response_output:
        return 0.0

    return verify_workplace_assistant(
        response_output=response_output,
        ground_truth=ground_truth,
    )
