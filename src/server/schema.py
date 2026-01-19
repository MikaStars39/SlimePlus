"""
Request/Response schemas for Reward Model Server.
"""
from pydantic import BaseModel


class RewardRequest(BaseModel):
    """Request schema for reward calculation."""
    prompt: str
    response: str
    label: str
    source: str = "math"  # Task source for judge routing (e.g., "math", "ifeval")
    metadata: dict | None = None
