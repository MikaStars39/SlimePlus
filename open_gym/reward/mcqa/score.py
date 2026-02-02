# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone scoring function extracted from app.py
# Removes: FastAPI server, async operations, Pydantic models
# Keeps: All regex patterns and parsing logic

import re
from typing import Any, Literal, Optional


# Regex patterns (directly from app.py)
CHOICE_LETTER_PATTERN = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")
STRICT_BOXED_PATTERN = re.compile(r"\\boxed\{\s*[^A-Za-z]*([A-Z])[^A-Za-z]*\s*\}")
ANSWER_COLON_PATTERN = re.compile(r"(?i)answer\s*:\s*(.+)")
BOXED_CONTENT_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)
LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)


def _strip_latex_wrappers(s: str) -> str:
    """Remove successive \\text{...} wrappers from a LaTeX string."""
    while True:
        m = LATEX_TEXT_WRAP_PATTERN.fullmatch(s)
        if not m:
            break
        s = m.group(1)
    return s


def _normalize_for_match(s: str) -> str:
    """Lowercase and collapse whitespace for robust substring/equality checks."""
    return " ".join(s.lower().split())


def _get_allowed_letters_from_options(options: Optional[list[dict[str, str]]]) -> set[str]:
    """Collect uppercase option letters from list of single-key dicts."""
    letters: set[str] = set()
    if options:
        for entry in options:
            for k in entry.keys():
                if isinstance(k, str) and len(k) == 1 and k.isalpha():
                    letters.add(k.upper())
                break
    return letters


def _parse_answer_letter_strict_boxed(text: str, allowed_letters: set[str]) -> tuple[Optional[str], str, bool]:
    parsed_text = text
    m = STRICT_BOXED_PATTERN.search(text)
    if not m:
        return None, parsed_text, True
    letter = m.group(1).upper()
    if letter not in allowed_letters:
        return None, parsed_text, True
    return letter, parsed_text, False


def _match_option_text(text: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    """Match boxed content against option texts and return the option letter."""
    boxed = BOXED_CONTENT_PATTERN.search(text)
    if not boxed:
        return None
    inner = boxed.group(1)
    candidate_texts = [inner, _strip_latex_wrappers(inner)]
    normalized_candidates = [_normalize_for_match(t) for t in candidate_texts]

    normalized_options: list[tuple[str, str]] = []
    for entry in options or []:
        for k, v in entry.items():
            if isinstance(k, str) and len(k) == 1 and k.upper() in allowed_letters:
                normalized_options.append((k.upper(), _normalize_for_match(v)))
                break

    matched_letters: set[str] = set()
    for cand in normalized_candidates:
        for letter, opt_norm in normalized_options:
            if opt_norm and opt_norm in cand:
                matched_letters.add(letter)
    if len(matched_letters) == 1:
        return next(iter(matched_letters))
    return None


def _parse_answer_with_custom_regex(
    text: str, regex_pattern: str, allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
    """Parse answer using custom regex from template_metadata."""
    try:
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        if not matches:
            return None

        captured = matches[-1].strip().upper()

        if len(captured) == 1 and captured.isalpha():
            if allowed_letters and captured in allowed_letters:
                return captured
            elif not allowed_letters:
                return captured
            else:
                return captured

        normalized_captured = _normalize_for_match(captured)
        for entry in options or []:
            for k, v in entry.items():
                if k.upper() in allowed_letters and _normalize_for_match(v) == normalized_captured:
                    return k.upper()

        return None
    except re.error:
        return None


def verify_mcqa(
    model_output: str,
    expected_answer: Optional[str],
    options: Optional[list[dict[str, str]]] = None,
    grading_mode: Literal["strict_single_letter_boxed", "lenient_boxed", "lenient_answer_colon"] = "strict_single_letter_boxed",
    template_metadata: Optional[dict[str, Any]] = None,
) -> tuple[float, Optional[str]]:
    """
    Verify MCQA answer - core logic from app.py MCQAResourcesServer.verify()

    Args:
        model_output: Model-generated answer text
        expected_answer: Ground truth answer letter
        options: List of option dicts, e.g. [{"A": "text"}, {"B": "text"}]
        grading_mode: Grading mode selector
        template_metadata: Optional dict with "output_regex" for custom parsing

    Returns:
        (reward, extracted_answer): reward is 0.0 or 1.0
    """
    text = model_output
    allowed_letters = _get_allowed_letters_from_options(options)

    pred: Optional[str] = None

    # Check for template_metadata first (highest priority)
    if template_metadata and "output_regex" in template_metadata:
        regex_pattern = template_metadata["output_regex"]
        pred = _parse_answer_with_custom_regex(text, regex_pattern, allowed_letters, options)

    # Fallback to existing grading_mode logic
    if pred is None:
        if grading_mode == "strict_single_letter_boxed":
            pred, _, _ = _parse_answer_letter_strict_boxed(text, allowed_letters)
        elif grading_mode == "lenient_boxed":
            pred, _, _ = _parse_answer_letter_strict_boxed(text, allowed_letters)
            if pred is None:
                letter_from_text = _match_option_text(text, options, allowed_letters)
                if letter_from_text is not None:
                    pred = letter_from_text
        elif grading_mode == "lenient_answer_colon":
            m = ANSWER_COLON_PATTERN.search(text)
            if m:
                candidate = _strip_latex_wrappers(m.group(1)).strip()
                if len(candidate) == 1 and candidate.isalpha():
                    letter_up = candidate.upper()
                    if letter_up in allowed_letters:
                        pred = letter_up
                if pred is None:
                    cand_norm = _normalize_for_match(candidate)
                    for entry in options or []:
                        for k, v in entry.items():
                            k_up = k.upper()
                            if k_up in allowed_letters and _normalize_for_match(v) == cand_norm:
                                pred = k_up
                                break
                        if pred is not None:
                            break

    gold = (expected_answer or "").strip().upper()
    is_correct = (pred == gold) if (pred is not None and gold) else False
    reward = 1.0 if is_correct else 0.0

    return reward, pred


def score_fn(model_output: str, extra_info: dict) -> float:
    """
    Standalone MCQA scoring function for verl reward manager.

    Args:
        model_output: Model-generated answer text
        extra_info: Dictionary from parquet containing:
            - expected_answer: Ground truth answer letter
            - options: List of option dicts (optional)
            - grading_mode: Grading mode (optional, default: strict_single_letter_boxed)
            - template_metadata: Custom regex metadata (optional)

    Returns:
        float: 1.0 (correct) or 0.0 (incorrect)
    """
    expected_answer = extra_info.get("expected_answer")
    options = extra_info.get("options")
    grading_mode = extra_info.get("grading_mode", "strict_single_letter_boxed")
    template_metadata = extra_info.get("template_metadata")

    reward, _ = verify_mcqa(
        model_output=model_output,
        expected_answer=expected_answer,
        options=options,
        grading_mode=grading_mode,
        template_metadata=template_metadata,
    )
    return reward
