"""
Async logging utilities for Reward Model Server.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import anyio

logger = logging.getLogger("RM")


async def save_request_log(
    output_dir: str,
    req,
    result: dict,
    score: float
):
    """Async save request log to file."""
    if not output_dir:
        return
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = output_path / f"{timestamp}_{uuid4().hex[:8]}.json"
        
        payload = {
            "timestamp": timestamp,
            "source": req.source,
            "prompt": req.prompt,
            "response": req.response,
            "label": req.label,
            "metadata": req.metadata,
            "judge_result": result,
            "score": score,
        }
        content = json.dumps(payload, ensure_ascii=False)
        await anyio.to_thread.run_sync(log_file.write_text, content, "utf-8")
    except Exception as e:
        logger.error(f"Failed to save log: {e}")
