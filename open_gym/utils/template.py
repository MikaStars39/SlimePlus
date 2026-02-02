PROMPT_TEMPLATES = {
    #lighteval
    "lighteval": """{problem} Please reason step by step, and put your final answer within \\boxed{{}}.""",
    
    # open-r1
    "open-r1": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),

    # for llm extraction
    "extraction": """
Please extract the final answer from the following response. The answer should be put inside \\boxed{{}}. 

Response:
{response}
""".strip(),

    # slime default training format
    "slime": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),

    # GPQA inferece
    "GPQA-Diamond": """
{problem}
Choose an answer in A,B,C,D. Answer with \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}.
""".strip(),

    "blank": """
{problem}
""".strip(),

}

SYSTEM_PROMPT_TEMPLATES = {
    "jd_thinking": """
    You are JoyAI, a large language model trained by JD (京东). For every response, please provide a step-by-step reasoning process enclosed in <think> and </think> tags. After the thinking, you need to output the final answer.
""".strip(),
}