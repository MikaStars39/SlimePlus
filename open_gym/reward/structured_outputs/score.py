# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone scoring function extracted from app.py
# Removes: FastAPI server, async operations
# Keeps: JSON schema validation logic using openapi_schema_validator

import json
from enum import Enum
from typing import Any, Dict

from openapi_schema_validator import validate as validate_against_schema_openapi


class SchemaType(str, Enum):
    JSON = "json"


def strictify_schema_json(schema: Dict[str, Any]) -> None:
    """Make a schema strict as per OpenAPI guidelines.

    Also handles parquet data bug where "required": true (boolean) appears
    at property level instead of standard JSON Schema format.
    """
    if isinstance(schema, Dict):
        # Fix parquet data bug: remove invalid "required": true (boolean)
        # Standard JSON Schema expects "required" to be array of property names at object level
        if "required" in schema and isinstance(schema["required"], bool):
            del schema["required"]
        # Original logic: make all object properties required
        if "properties" in schema:
            schema["required"] = list(schema["properties"])
            schema["additionalProperties"] = False
        for k, v in schema.items():
            strictify_schema_json(v)


def evaluate_structured_output_json(schema_str: str, response_text: str) -> float:
    """
    Evaluate if response_text is valid JSON matching the schema.

    Args:
        schema_str: JSON schema as string
        response_text: Model-generated JSON response

    Returns:
        float: 1.0 if valid, 0.0 if invalid
    """
    try:
        schema = json.loads(schema_str)
        strictify_schema_json(schema)
        response_obj = json.loads(response_text)
        validate_against_schema_openapi(response_obj, schema)
        return 1.0
    except Exception:
        return 0.0


def verify_structured_output(
    model_output: str,
    schema_str: str,
    schema_type: str = "json",
) -> float:
    """
    Verify structured output - core logic from app.py StructuredOutputsResourcesServer.verify()

    Args:
        model_output: Model-generated text response
        schema_str: Schema string for validation
        schema_type: Type of schema ("json")

    Returns:
        float: 1.0 if valid, 0.0 if invalid
    """
    schema_type_enum = SchemaType(schema_type)

    match schema_type_enum:
        case SchemaType.JSON:
            return evaluate_structured_output_json(schema_str, model_output)
        case _:
            raise NotImplementedError(f"SchemaType must be one of {list(SchemaType)}, got {schema_type}!")


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Standalone structured output scoring function for verl reward manager.

    Args:
        model_output: Model-generated JSON response
        extra_info: Dictionary from parquet containing:
            - schema_str: JSON schema as string
            - schema_type: Type of schema (default: "json")

    Returns:
        float: 1.0 (valid) or 0.0 (invalid)
    """
    schema_str = extra_info.get("schema_str", "")
    schema_type = extra_info.get("schema_type", "json")

    if not schema_str:
        return 0.0

    return verify_structured_output(
        model_output=model_output,
        schema_str=schema_str,
        schema_type=schema_type,
    )
