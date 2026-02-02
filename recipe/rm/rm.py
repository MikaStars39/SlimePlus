"""
Reward Model Server with LLM-based answer extraction.

Flow:
1. Receive response from LLM
2. Extract answer using OnlineServingEngine
3. Judge extracted answer using rule-based judge_router
4. Return score
"""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from open_gym.backend import OnlineServingEngine
from open_gym.llm_judge import extract_answer_online
from open_gym.reward import judge_router
from open_gym.server import RewardRequest, save_request_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("RM")

# ----------------------- RM Server CLI -----------------------

def parse_rm_args() -> argparse.Namespace:
    """Parse CLI arguments for Reward Model Server."""
    parser = argparse.ArgumentParser(description="Reward Model Server")
    
    # Model settings
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Path to the model")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--dp-size", type=int, default=1,
                        help="Data parallel size")
    parser.add_argument("--mem-fraction-static", type=float, default=0.90,
                        help="Static memory fraction for SGLang")
    
    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    
    # Server settings
    parser.add_argument("--host", default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--max-concurrent", type=int, default=16,
                        help="Maximum concurrent requests")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Request timeout in seconds")
    
    # Logging
    parser.add_argument("--output-dir", default="rm_logs",
                        help="Directory for log output")
    
    return parser.parse_args()



# ------------ Global State ------------

args = None
engine: OnlineServingEngine = None
semaphore: asyncio.Semaphore = None


# ------------ FastAPI App ------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage OnlineServingEngine lifecycle."""
    global engine
    async with OnlineServingEngine(
        model_path=args.model_path,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        mem_fraction_static=args.mem_fraction_static,
        enable_radix_cache=True,
    ) as eng:
        engine = eng
        logger.info("OnlineServingEngine initialized")
        yield


app = FastAPI(title="RM Server", lifespan=lifespan)


@app.post("/reward")
async def reward_endpoint(req: RewardRequest):
    """Calculate reward: extract answer with LLM, then judge with rules."""
    async with semaphore:
        try:
            # Step 1: Extract answer using LLM
            sampling_params = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            }
            extracted = await asyncio.wait_for(
                extract_answer_online(engine, req.response, req.label, sampling_params),
                timeout=args.timeout
            )
            
            # Step 2: Judge using rule-based router
            result = judge_router(
                response=extracted,
                label=req.label,
                source=req.source,
                **(req.metadata or {})
            )
            result["extracted"] = extracted
            
            # Step 3: Return score
            score = 1.0 if result.get("pass", False) else 0.0
            
            # Log
            preview = req.label[:30] if req.label else ""
            logger.info(f"[{req.source}] GT: {preview}... | Pred: {result.get('pred', 'N/A')} | Score: {score}")
            asyncio.create_task(save_request_log(args.output_dir, req, result, score))
            
            return score
            
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return 0.0
        except Exception as e:
            logger.error(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ------------ Entry Point ------------

def main():
    """Main entry point."""
    global args, semaphore
    
    args = parse_rm_args()
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    print(f"🔥 Starting RM Server on {args.host}:{args.port}")
    print(f"   Model: {args.model_path}")
    print(f"   TP={args.tp_size}, DP={args.dp_size}")
    print(f"   MaxConcurrent={args.max_concurrent}, Timeout={args.timeout}s")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
