"""
Setwise comparer: pick one winner from N docs.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from .base import BaseComparer, CompareResultBase
from ..utils.retry_utils import async_retry


CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]
SYSTEM_PROMPT = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."


@dataclass
class SetwiseResult(CompareResultBase):
    winner_index: int


class SetwiseComparer(BaseComparer):
    """
    Comparer that selects the most relevant document from a pool of documents.
    Returns the 0-based index of the winner.
    """

    def __init__(self, model: str, max_doc_tokens: int, max_query_tokens: int = 1024, temperature: Optional[float] = None):
        super().__init__(model, max_doc_tokens, max_query_tokens)
        self.temperature = temperature

    def _build_user_prompt(self, query: str, docs: List[str]) -> str:
        """
        Build user prompt matching OpenAiSetwiseLlmRanker format.
        """
        passages = "\n\n".join(
            [f'Passage {CHARACTERS[i]}: "{doc}"' for i, doc in enumerate(docs)]
        )
        return (
            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
            + passages
            + "\n\nOutput only the passage label of the most relevant passage."
        )

    def _build_messages(self, query: str, docs: List[str]) -> List[Dict[str, str]]:
        """
        Build messages matching OpenAiSetwiseLlmRanker format.
        """
        user_prompt = self._build_user_prompt(query, docs)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, response: str, num_docs: int) -> int:
        """
        Parse the LLM response to extract the winner index.
        Handles formats like: "A", "Passage A", "passage a", etc.
        Matching OpenAiSetwiseLlmRanker parsing logic.
        """
        response = response.strip().upper()
        # Try to find "Passage X" pattern first
        matches = re.findall(r"Passage ([A-Z])", response, re.IGNORECASE)
        if matches:
            char = matches[0].upper()
            if char in CHARACTERS[:num_docs]:
                return CHARACTERS.index(char)

        # Try single character
        if len(response) == 1 and response in CHARACTERS[:num_docs]:
            return CHARACTERS.index(response)

        # Default to first document if parsing fails
        return 0

    @async_retry()
    async def compare(self, query: str, docs: List[str]) -> SetwiseResult:
        """
        Compare multiple docs and return the result with winner index.
        """
        prepared_query = self._prepare_query(query)
        prepared_docs = [self._prepare_doc(doc) for doc in docs]
        messages = self._build_messages(prepared_query, prepared_docs)
        raw_response, latency_ms, usage = await self.client.get_response(self.model, messages, self.temperature)
        winner_index = self._parse_response(raw_response, len(docs))

        return SetwiseResult(
            winner_index=winner_index,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            latency_ms=latency_ms,
            model=self.model,
            raw_prompt=messages,
            raw_response=raw_response,
        )
