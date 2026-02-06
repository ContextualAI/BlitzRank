from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

import tiktoken

from .client import LitellmClient


@dataclass
class CompareResultBase:
    """Common metrics for all comparers."""
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    raw_prompt: Any
    raw_response: str


_tokenizer = tiktoken.get_encoding("o200k_base")


class BaseComparer(ABC):
    def __init__(self, model: str, max_doc_tokens: int, max_query_tokens: int = 1024):
        self.model = self._normalize_model(model)
        self.max_doc_tokens = max_doc_tokens
        self.max_query_tokens = max_query_tokens
        self.client = LitellmClient()

    def _normalize_model(self, model: str) -> str:
        if "gpt" in model and not model.startswith("openai"):
            return f"openai/{model}"
        if "claude" in model and not model.startswith("anthropic"):
            return f"anthropic/{model}"
        return model

    def _trim_to_tokens(self, text: str, max_tokens: int) -> str:
        encoded = _tokenizer.encode(text)
        if len(encoded) <= max_tokens:
            return text
        return _tokenizer.decode(encoded[:max_tokens])

    def _prepare_doc(self, content: str) -> str:
        return self._trim_to_tokens(
            content.replace("Title: Content: ", "").strip(),
            self.max_doc_tokens
        )

    def _prepare_query(self, query: str) -> str:
        return self._trim_to_tokens(query.strip(), self.max_query_tokens)

    @abstractmethod
    async def compare(self, query: str, docs: List[str]) -> CompareResultBase:
        pass
