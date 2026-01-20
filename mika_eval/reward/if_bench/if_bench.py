# Copyright 2025 Allen Institute for AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic evaluation function for IFBench instruction following."""

from typing import Dict, List, Any, Optional, Union

from .instructions_registry import INSTRUCTION_DICT


def ifbench_judge(
    response: str,
    instruction_id_list: List[str],
    kwargs: List[Dict[str, Any]],
    prompt: Optional[str] = None,
    **extra_kwargs
) -> Dict[str, Union[int, bool, List[bool]]]:
    """
    Evaluate if a response follows the given instructions.
    
    This is an atomic function that checks whether a single response follows
    a list of IFBench instructions.
    
    Args:
        response: The model's response text to evaluate.
        instruction_id_list: List of instruction IDs to check (e.g., ["count:keywords_multiple"]).
        kwargs: List of keyword argument dicts for each instruction. Each dict contains
                the parameters needed for the corresponding instruction checker.
        prompt: Optional original prompt, needed for some instruction types.
        **extra_kwargs: Additional keyword arguments (ignored, for compatibility).
    
    Returns:
        A dictionary containing:
            - 'instruction_count': Total number of instructions checked.
            - 'instruction_pass_cnt': Number of instructions passed.
            - 'pass': Boolean indicating if ALL instructions were followed.
            - 'follow_instruction_list': List of booleans for each instruction.
    
    Example:
        >>> result = ifbench_judge(
        ...     response="This is a test with kaleidoscope...",
        ...     instruction_id_list=["count:keywords_multiple"],
        ...     kwargs=[{"keyword1": "kaleidoscope", "keyword2": "nebula", ...}]
        ... )
        >>> print(result)
        {'instruction_count': 1, 'instruction_pass_cnt': 0, 'pass': False, 'follow_instruction_list': [False]}
    """
    if not response or not response.strip():
        # Empty response fails all instructions
        return {
            'instruction_count': len(instruction_id_list),
            'instruction_pass_cnt': 0,
            'pass': False,
            'follow_instruction_list': [False] * len(instruction_id_list)
        }
    
    prompt_level_pass_flag = True
    instruction_pass_cnt = 0
    follow_instruction_list = []
    
    for index, instruction_id in enumerate(instruction_id_list):
        # Get the instruction class from registry
        instruction_cls = INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            raise ValueError(f"Unknown instruction ID: {instruction_id}")
        
        # Create instruction instance
        instruction = instruction_cls(instruction_id)
        
        # Filter kwargs: remove None values and only keep supported keys
        instruction_kwargs = kwargs[index] if index < len(kwargs) else {}
        supported_keys = instruction.get_instruction_args_keys()
        filtered_kwargs = {
            k: v for k, v in instruction_kwargs.items() 
            if v is not None and k in supported_keys
        }
        
        # Build instruction description with filtered kwargs
        instruction.build_description(**filtered_kwargs)
        
        # Some instructions need the original prompt
        args = instruction.get_instruction_args()
        if args and "prompt" in args and prompt:
            instruction.build_description(prompt=prompt)
        
        # Check if the response follows this instruction
        passed = instruction.check_following(response)
        
        follow_instruction_list.append(passed)
        if passed:
            instruction_pass_cnt += 1
        else:
            prompt_level_pass_flag = False
    
    return {
        'instruction_count': len(instruction_id_list),
        'instruction_pass_cnt': instruction_pass_cnt,
        'pass': prompt_level_pass_flag,
        'follow_instruction_list': follow_instruction_list
    }


def ifbench_judge_loose(
    response: str,
    instruction_id_list: List[str],
    kwargs: List[Dict[str, Any]],
    prompt: Optional[str] = None,
    **extra_kwargs
) -> Dict[str, Union[int, bool, List[bool]]]:
    """
    Evaluate if a response follows the given instructions (loose mode).
    
    This version tries multiple variations of the response (removing first/last lines,
    removing markdown asterisks) to give an upper bound for instruction following.
    
    Args:
        response: The model's response text to evaluate.
        instruction_id_list: List of instruction IDs to check.
        kwargs: List of keyword argument dicts for each instruction.
        prompt: Optional original prompt, needed for some instruction types.
        **extra_kwargs: Additional keyword arguments (ignored, for compatibility).
    
    Returns:
        Same format as ifbench_judge.
    """
    if not response or not response.strip():
        return {
            'instruction_count': len(instruction_id_list),
            'instruction_pass_cnt': 0,
            'pass': False,
            'follow_instruction_list': [False] * len(instruction_id_list)
        }
    
    # Generate response variations
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    
    prompt_level_pass_flag = True
    instruction_pass_cnt = 0
    follow_instruction_list = []
    
    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            raise ValueError(f"Unknown instruction ID: {instruction_id}")
        
        instruction = instruction_cls(instruction_id)
        
        instruction_kwargs = kwargs[index] if index < len(kwargs) else {}
        supported_keys = instruction.get_instruction_args_keys()
        filtered_kwargs = {
            k: v for k, v in instruction_kwargs.items() 
            if v is not None and k in supported_keys
        }
        
        instruction.build_description(**filtered_kwargs)
        
        args = instruction.get_instruction_args()
        if args and "prompt" in args and prompt:
            instruction.build_description(prompt=prompt)
        
        # Try all response variations
        passed = False
        for resp_variant in all_responses:
            if resp_variant.strip() and instruction.check_following(resp_variant):
                passed = True
                break
        
        follow_instruction_list.append(passed)
        if passed:
            instruction_pass_cnt += 1
        else:
            prompt_level_pass_flag = False
    
    return {
        'instruction_count': len(instruction_id_list),
        'instruction_pass_cnt': instruction_pass_cnt,
        'pass': prompt_level_pass_flag,
        'follow_instruction_list': follow_instruction_list
    }


def calculate_scores(results: List[Dict[str, Any]]) -> Dict[str, Union[float, int]]:
    """
    Calculate aggregate scores from a list of evaluation results.
    
    Args:
        results: List of result dicts from ifbench_judge calls.
    
    Returns:
        A dictionary containing:
            - 'prompt_level_score': Fraction of prompts where all instructions were followed.
            - 'instruct_level_score': Fraction of individual instructions followed.
            - 'prompt_level_passed': Number of prompts fully passed.
            - 'total_prompts': Total number of prompts.
            - 'instruction_level_passed': Number of instructions passed.
            - 'total_instructions': Total number of instructions.
    """
    total_prompts = len(results)
    prompt_level_passed = sum(1 for r in results if r['pass'])
    
    total_instructions = sum(r['instruction_count'] for r in results)
    instruction_level_passed = sum(r['instruction_pass_cnt'] for r in results)
    
    prompt_level_score = prompt_level_passed / total_prompts if total_prompts > 0 else 0
    instruct_level_score = instruction_level_passed / total_instructions if total_instructions > 0 else 0
    
    return {
        'prompt_level_score': prompt_level_score,
        'instruct_level_score': instruct_level_score,
        'prompt_level_passed': prompt_level_passed,
        'total_prompts': total_prompts,
        'instruction_level_passed': instruction_level_passed,
        'total_instructions': total_instructions
    }


if __name__ == '__main__':
    # Example usage with the sample from the user's query
    sample_kwargs = [{
        "N": None, "capital_frequency": None, "capital_relation": None,
        "end_phrase": None, "first_word": None, "forbidden_words": None,
        "frequency": None, "keyword": None, "keyword1": "kaleidoscope",
        "keyword2": "nebula", "keyword3": "whisper", "keyword4": "labyrinth",
        "keyword5": "paradox", "keywords": None, "language": None,
        "let_frequency": None, "let_relation": None, "letter": None,
        "m": None, "max_words": None, "min_words": None, "n": None,
        "n_end": None, "n_start": None, "nth_paragraph": None,
        "num_bullets": None, "num_highlights": None, "num_paragraphs": None,
        "num_placeholders": None, "num_sections": None, "num_sentences": None,
        "num_words": None, "options": None, "percentage": None,
        "postscript_marker": None, "prompt_to_repeat": None,
        "reference_text": None, "relation": None, "section_spliter": None,
        "sep": None, "small_n": None, "word": None
    }]
    
    # Test with a sample response
    test_response = """
    This is a kaleidoscope of ideas. The nebula shines bright in the sky.
    Another nebula appears. I whisper to you, whisper again, and whisper once more.
    The labyrinth is complex. Enter the labyrinth carefully. The labyrinth has many paths.
    Navigate the labyrinth with care. Exit the labyrinth successfully.
    This is a paradox. What a paradox! The paradox continues. Another paradox emerges.
    The final paradox awaits. Paradox upon paradox. The ultimate paradox.
    """
    
    result = ifbench_judge(
        response=test_response,
        instruction_id_list=["count:keywords_multiple"],
        kwargs=sample_kwargs
    )
    
    print("Evaluation Result:")
    print(f"  Instructions: {result['instruction_count']}")
    print(f"  Passed: {result['instruction_pass_cnt']}")
    print(f"  All Passed: {result['pass']}")
    print(f"  Details: {result['follow_instruction_list']}")
