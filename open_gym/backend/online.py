import logging
from typing import List, Dict, Any
from .base import BaseSGLangEngine

logger = logging.getLogger("OnlineServing")

class OnlineServingEngine(BaseSGLangEngine):
    """
    A lightweight wrapper for direct, online request serving.
    Enables RadixCache by default for multi-turn chat acceleration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------- Interface ------------------------

    async def generate(self, prompt: str, sampling_params: Dict[str, Any]) -> str:
        """Direct raw prompt generation."""
        output = await self._generate_safe(prompt, sampling_params)
        return output["text"]

    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        sampling_params: Dict[str, Any]
    ) -> str:
        """Apply chat template and generate response."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for raw text or simple models
            prompt = messages[-1]["content"]

        return await self.generate(prompt, sampling_params)