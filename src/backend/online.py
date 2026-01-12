import sglang as sgl
import pandas as pd
import numpy as np
import torch
import argparse
import logging
import multiprocessing
import os
import json
import time
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - process-%(processName)s - %(message)s')

def get_available_gpus(count: int = None) -> List[int]:
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        if count is None or count > total_gpus:
            count = total_gpus
        return list(range(count))
    return []

def worker_process(
    gpu_id: int, 
    data_shard: List[Dict[str, Any]], 
    model_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> List[Dict[str, Any]]:
    """
    单个 GPU 的工作进程：初始化 Engine 并批量生成
    """
    # 1. 显卡隔离：这一步至关重要，让当前进程只看得到一张卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        logging.info(f"GPU {gpu_id}: 正在加载模型 (Engine模式)...")
        
        # === 核心：初始化离线引擎 ===
        # 这里的参数完全对标 vLLM 的 LLM 类
        engine = sgl.Engine(
            model_path=model_path,
            tp_size=1, # DP 模式，每张卡 TP=1
            trust_remote_code=True,
            mem_fraction_static=0.85, # 稍微保守一点，留给 PyTorch
            context_length=32768,     # 强制指定上下文长度支持 30k+
            disable_radix_cache=True, # 离线批量任务，禁用 Radix 缓存以节省显存
            chunked_prefill_size=4096,    # 开启分块 Prefill，防止超长 Prompt OOM
            schedule_policy="fcfs",       # 先来后到，不搞复杂的缓存匹配
        )

        # 2. 准备 Prompt
        prompts = []
        for item in data_shard:
            p = item.get('prompt')
            if not p and 'question' in item:
                p = item['question']
                # 如果有 background，拼一下 (根据你的数据格式调整)
                # if 'background' in item and item['background']:
                #     p = item['background'] + "\n" + p
            prompts.append(p)

        logging.info(f"GPU {gpu_id}: 模型加载完毕，开始处理 {len(prompts)} 条数据...")

        # 3. 批量生成
        # SGLang Engine 的 generate 接口支持 list 输入，内部自动处理 Batching
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        
        # 计时
        start_t = time.time()
        
        # 直接调用 engine 生成
        # 这里的 batch 调度由 Engine 内部全权负责，它会根据显存动态塞尽可能多的请求
        outputs = engine.generate(prompts, sampling_params)
        
        # 4. 收集结果
        results = []
        for item, output in zip(data_shard, outputs):
            item['responses'] = [output["text"]]
            # item['meta_info'] = output["meta_info"] # 如果需要调试信息
            results.append(item)
            
        logging.info(f"GPU {gpu_id}: 完成任务! 耗时 {time.time()-start_t:.2f}s")
        
        # 显式关闭 engine 释放资源
        engine.shutdown()
        
        return results

    except Exception as e:
        logging.error(f"GPU {gpu_id} 发生严重错误: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    # workers_per_gpu 参数在这里失效了，因为 Engine 会自动管理并发
    # 我们只需要管好 num_gpus 即可
    parser.add_argument("--max_tokens", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    # 兼容脚本里的多余参数
    parser.add_argument("--workers_per_gpu", type=int, default=16) 
    args = parser.parse_args()

    # 1. 准备数据
    logging.info(f"Reading {args.input_file}...")
    df = pd.read_json(args.input_file, lines=True)
    all_data = df.to_dict('records')
    
    # 2. 准备 GPU
    gpu_ids = get_available_gpus(args.num_gpus)
    logging.info(f"Detected GPUs: {gpu_ids}")
    
    # 3. 数据切分
    data_shards = np.array_split(all_data, len(gpu_ids))
    data_shards = [shard.tolist() for shard in data_shards]

    # 4. 多进程启动 (Offline Engine 必须在 spawn 模式下运行)
    multiprocessing.set_start_method('spawn', force=True)
    
    with multiprocessing.Pool(processes=len(gpu_ids)) as pool:
        tasks = []
        for i, gpu_id in enumerate(gpu_ids):
            if len(data_shards[i]) == 0: continue
            
            tasks.append(pool.apply_async(
                worker_process,
                args=(
                    gpu_id, 
                    data_shards[i], 
                    args.model_name, 
                    args.max_tokens, 
                    args.temperature, 
                    args.top_p
                )
            ))
        
        final_results = []
        for task in tasks:
            try:
                # 这里的 get 会等待进程结束，如果那是超长任务，这里会挂起很久，是正常的
                res = task.get()
                final_results.extend(res)
            except Exception as e:
                logging.error(f"Task failed: {e}")

    # 5. 保存
    if final_results:
        out_df = pd.DataFrame(final_results)
        output_dir = os.path.dirname(args.output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        out_df.to_json(args.output_file, orient='records', lines=True)
        logging.info(f"Saved to {args.output_file}")
    else:
        logging.error("No results generated!")

if __name__ == "__main__":
    main()