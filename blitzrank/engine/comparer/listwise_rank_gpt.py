"""
RankGPT comparer: full permutation ranking.

Wraps the existing LlmRanker logic for rank_gpt style comparisons.
"""
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

from .base import BaseComparer, CompareResultBase
from ..utils.retry_utils import async_retry


@dataclass
class ListwiseRankGptResult(CompareResultBase):
    permutation: List[int]
    missing_indices: List[int]
    duplicate_indices: List[int]
    thought_tokens: int
    num_trimmed_docs: int


def get_prefix_prompt(query: str, num: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
        },
        {
            "role": "user",
            "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]


def get_post_prompt(query: str, num: int) -> str:
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def create_permutation_instruction(query: str, docs: List[str]) -> List[Dict[str, str]]:
    messages = get_prefix_prompt(query, len(docs))
    for rank, content in enumerate(docs, 1):
        messages.append({"role": "user", "content": f"[{rank}] {content}"})
        messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
    messages.append({"role": "user", "content": get_post_prompt(query, len(docs))})
    return messages


def parse_permutation(raw_response: str, num_docs: int) -> Tuple[List[int], List[int], List[int]]:
    cleaned = "".join(" " if not c.isdigit() else c for c in raw_response).strip()
    parsed = [int(x) - 1 for x in cleaned.split() if x]

    seen, duplicates, unique = set(), [], []
    for idx in parsed:
        if idx in seen:
            duplicates.append(idx)
        else:
            seen.add(idx)
            unique.append(idx)

    valid_range = set(range(num_docs))
    filtered = [idx for idx in unique if idx in valid_range]
    missing = [idx for idx in valid_range if idx not in filtered]
    return filtered + missing, missing, duplicates


def _get_thought_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    details = None
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
    else:
        details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
    else:
        value = getattr(details, "reasoning_tokens", None)
    return value if isinstance(value, int) else 0


class ListwiseRankGptComparer(BaseComparer):
    """
    Comparer that produces a full permutation ranking of documents.
    Uses the RankGPT conversational prompt format.
    """

    def __init__(self, model: str, max_doc_tokens: int, max_query_tokens: int = 1024, temperature: Optional[float] = None):
        super().__init__(model, max_doc_tokens, max_query_tokens)
        self.num_trimmed_docs = 0
        self.temperature = temperature

    def _prepare_doc(self, content: str) -> str:
        """Override to track trimmed docs."""
        from .base import _tokenizer
        text = content.replace("Title: Content: ", "").strip()
        encoded = _tokenizer.encode(text)
        if len(encoded) <= self.max_doc_tokens:
            return text
        self.num_trimmed_docs += 1
        return _tokenizer.decode(encoded[:self.max_doc_tokens])

    @async_retry()
    async def compare(self, query: str, docs: List[str]) -> ListwiseRankGptResult:
        """
        Rank all docs and return permutation result.
        """
        self.num_trimmed_docs = 0
        prepared_query = self._prepare_query(query)
        prepared_docs = [self._prepare_doc(doc) for doc in docs]
        messages = create_permutation_instruction(prepared_query, prepared_docs)

        raw_response, latency_ms, usage = await self.client.get_response(self.model, messages, self.temperature)
        permutation, missing, duplicates = parse_permutation(raw_response, len(docs))
        thought_tokens = _get_thought_tokens(usage)

        return ListwiseRankGptResult(
            permutation=permutation,
            missing_indices=missing,
            duplicate_indices=duplicates,
            thought_tokens=thought_tokens,
            num_trimmed_docs=self.num_trimmed_docs,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            latency_ms=latency_ms,
            model=self.model,
            raw_prompt=messages,
            raw_response=raw_response,
        )
