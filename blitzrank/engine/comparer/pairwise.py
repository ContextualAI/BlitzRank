"""
Pairwise comparer: compare two docs, return which is more relevant.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from .base import BaseComparer, CompareResultBase
from ..utils.retry_utils import async_retry


SYSTEM_PROMPT = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pair of passages based on their relevance to the query."
PROMPT_TEMPLATE = """Given a query "{query}", which of the following two passages is more relevant to the query?
        
Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:"""
CHARACTERS = ["A", "B"]


@dataclass
class PairwiseResult(CompareResultBase):
    winner_index: int  # 0 or 1


class PairwiseComparer(BaseComparer):
    """
    Comparer that selects the more relevant document from a pair.
    Returns 0 if first doc wins, 1 if second doc wins.

    For swap verification, the orchestrator should call compare() twice
    with docs in both orders and reconcile the results.
    """

    def __init__(self, model: str, max_doc_tokens: int, max_query_tokens: int = 1024, temperature: Optional[float] = None):
        super().__init__(model, max_doc_tokens, max_query_tokens)
        self.temperature = temperature

    def _build_messages(self, query: str, doc1: str, doc2: str) -> List[Dict[str, str]]:
        user_prompt = PROMPT_TEMPLATE.format(query=query, doc1=doc1, doc2=doc2)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, response: str) -> int:
        """Parse response to extract winner index (0 for A, 1 for B)."""
        response = response.strip().upper()
        # Try to find "Passage X" pattern
        matches = re.findall(r"Passage ([A-B])", response, re.IGNORECASE)
        if matches:
            return 0 if matches[0].upper() == "A" else 1
        # Try single character
        if response in CHARACTERS:
            return 0 if response == "A" else 1
        # Default to first doc
        return 0

    @async_retry()
    async def compare(self, query: str, docs: List[str]) -> PairwiseResult:
        """
        Compare two docs and return the winner index.
        docs[0] is presented as Passage A, docs[1] as Passage B.
        Returns 0 if A wins, 1 if B wins.
        """
        if len(docs) != 2:
            raise ValueError(f"Pairwise comparer requires exactly 2 docs, got {len(docs)}")

        prepared_query = self._prepare_query(query)
        doc1, doc2 = self._prepare_doc(docs[0]), self._prepare_doc(docs[1])
        messages = self._build_messages(prepared_query, doc1, doc2)
        raw_response, latency_ms, usage = await self.client.get_response(self.model, messages, self.temperature)
        winner_index = self._parse_response(raw_response)

        return PairwiseResult(
            winner_index=winner_index,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            latency_ms=latency_ms,
            model=self.model,
            raw_prompt=messages,
            raw_response=raw_response,
        )
