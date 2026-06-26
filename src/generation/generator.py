import time
from typing import List, Optional
from loguru import logger
from src.generation.client import GroqClient
from src.generation.prompts import PromptManager
from src.schemas.chunk import RetrievedChunk
from src.schemas.answer import GeneratedAnswer

class Generator:
    """
    Centralized component for generating grounded answers.
    Combines the PromptManager for templates and GroqClient for inference.
    """
    def __init__(self, client: Optional[GroqClient] = None, prompt_manager: Optional[PromptManager] = None):
        self.client = client or GroqClient()
        self.prompt_manager = prompt_manager or PromptManager()

    def estimate_tokens(self, text: str) -> int:
        """
        TODO: Decision-005. Move to src/utils/token_utils.py during Phase 2.5.
        Rough heuristic for token estimation (approx 4 chars per token).
        """
        return len(text) // 4

    def generate_answer(self, query: str, context: List[RetrievedChunk], rewrite_required: bool = False, mode: str = "grounded_extraction") -> GeneratedAnswer:
        """
        Generates an answer based solely on the provided context chunks.
        Tracks latency and estimated tokens.
        """
        if rewrite_required:
            prompt = self.prompt_manager.repair_qa(query, context)
        elif mode == "qa":
            prompt = self.prompt_manager.scientific_qa(query, context)
        elif mode == "strict_extraction":
            prompt = self.prompt_manager.strict_extraction(query, context)
        elif mode == "grounded_extraction":
            prompt = self.prompt_manager.grounded_extraction(query, context)
        else:
            prompt = self.prompt_manager.grounded_extraction(query, context)
        
        logger.info("Generating answer...")
        start_time = time.perf_counter()
        
        raw_answer = self.client.generate(prompt)
        
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # milliseconds
        
        estimated_tokens = self.estimate_tokens(prompt) + self.estimate_tokens(raw_answer)
        
        logger.info(f"Answer generated in {latency:.2f}ms. Estimated tokens: {estimated_tokens}")
        
        return GeneratedAnswer(
            answer=raw_answer,
            model=self.client.default_model,
            latency=latency,
            estimated_tokens=estimated_tokens,
            prompt_version=self.prompt_manager.get_version()
        )
