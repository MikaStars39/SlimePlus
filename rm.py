"""
Reward Model Server with LLM-based answer extraction.

Flow:
1. Receive response from LLM
2. Wrap response in extraction prompt template
3. Convert to chat format and apply chat template
4. Send to OnlineServingEngine to extract the answer
5. Use judge_router to evaluate the extracted answer
6. Return score
"""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from src.backend import OnlineServingEngine
from src.config import parse_rm_args
from src.llm_judge.extract import PROMPT_TEMPLATE
from src.reward import judge_router
from src.server import RewardRequest, save_request_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("RM")


# ------------ Global State ------------

args = None
engine: OnlineServingEngine = None
semaphore: asyncio.Semaphore = None


# ------------ Answer Extraction ------------

async def extract_answer(response: str, label: str) -> str:
    """
    Use LLM to extract the final answer from model response.
    
    Args:
        response: The model's reasoning response
        label: Ground truth label (used as reference format)
    
    Returns:
        Extracted answer string from LLM
    """
    # 1. Build extraction prompt using template
    extraction_prompt = PROMPT_TEMPLATE.format(response=response, label=label)
    
    # 2. Convert to chat message format
    messages = [{"role": "user", "content": extraction_prompt}]
    
    # 3. Build sampling params from args
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    
    # 4. Send to engine for extraction
    extracted = await engine.chat(messages, sampling_params)
    return extracted


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
        logger.info("OnlineServingEngine shutting down")


app = FastAPI(title="RM Server", lifespan=lifespan)


@app.post("/reward")
async def reward_endpoint(req: RewardRequest):
    """
    Calculate reward for a given response.
    
    Steps:
    1. Extract answer using LLM
    2. Judge extracted answer using rule-based judge_router
    3. Return score and result
    """
    async with semaphore:
        try:
            # Step 1: Extract answer using LLM
            extracted_answer = await asyncio.wait_for(
                extract_answer(req.response, req.label),
                timeout=args.timeout
            )
            
            # Step 2: Judge using rule-based router
            result = judge_router(
                response=extracted_answer,
                label=req.label,
                source=req.source,
                **(req.metadata or {})
            )
            
            # Add extracted answer to result
            result["extracted"] = extracted_answer
            
            # Step 3: Calculate score
            score = 1.0 if result.get("pass", False) else 0.0
            
            # Log
            preview = req.label[:30] if req.label else ""
            pred = result.get("pred", "N/A")
            logger.info(f"[{req.source}] GT: {preview}... | Pred: {pred} | Score: {score}")
            asyncio.create_task(save_request_log(args.output_dir, req, result, score))
            
            return {"score": score, "result": result}
            
        except asyncio.TimeoutError:
            logger.error("Request timed out during answer extraction")
            return {"score": 0.0, "result": {"error": "timeout"}}
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
