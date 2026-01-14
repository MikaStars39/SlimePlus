
import json
from src.reward.if_eval.instructions_registry import INSTRUCTION_DICT

def if_judge(instance):
    instructions = instance['instruction_id_list']
    kwargs_list = instance['kwargs']
    response = instance['response']

    prompt_level_pass_flag = True
    instruction_pass_cnt = 0
    
    for instruction_id, kwargs in zip(instructions, kwargs_list):
        instruct = INSTRUCTION_DICT[instruction_id](instruction_id)
        
        supported_keys = instruct.get_instruction_args_keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_keys}
        
        instruct.build_description(**filtered_kwargs)
        passed = instruct.check_following(response)
        
        if passed:
            instruction_pass_cnt += 1
        else:
            prompt_level_pass_flag = False
    
    return {
        **instance,
        'instruction_count': len(instructions),
        'instruction_pass_cnt': instruction_pass_cnt,
        'pass': prompt_level_pass_flag
    }


def calculate_scores(results):
    """计算prompt level和instruct level的分数"""
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
    pass
    # with open('./ifeval_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    # print("详细结果已保存到 ifeval_results.json")