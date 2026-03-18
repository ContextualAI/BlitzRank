"""
Listwise CoT comparer: chain-of-thought reasoning before ranking.

Uses structured output to enforce a reasoning field (capped) followed by
a ranking field (list of integer IDs). The model is prompted to identify
the essential problem, reason about each passage's relevance, then rank.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

from pydantic import BaseModel, Field

from .base import BaseComparer, CompareResultBase
from ..utils.retry_utils import async_retry
from ..utils.logging_utils import logger


MAX_PARSE_RETRIES = 3


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class RankingResponse(BaseModel):
    """Structured output: reasoning followed by ranking."""
    thinking: str = Field(
        description=(
            "First identify the essential problem in the query. "
            "Then briefly reason about why each passage is relevant or irrelevant."
        ),
    )
    ranking: List[int] = Field(
        description=(
            "All passage identifiers ranked from most to least relevant. "
            "Must be a permutation of [1, 2, ..., N] with no missing or duplicate values."
        ),
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ListwiseCotResult(CompareResultBase):
    permutation: List[int]
    missing_indices: List[int]
    duplicate_indices: List[int]
    thinking: str
    thought_tokens: int
    num_trimmed_docs: int


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert relevance ranker. Given a query and passages, "
    "you carefully reason about each passage before ranking."
)

DEFAULT_RELEVANCE_INSTRUCTION = (
    "First identify the essential problem in the query. "
    "Then think step by step about why each passage is relevant or irrelevant."
)


def _build_messages(query: str, docs: List[str],
                    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                    relevance_instruction: str = DEFAULT_RELEVANCE_INSTRUCTION,
                    ) -> List[Dict[str, str]]:
    num = len(docs)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"I will provide you with {num} passages, each indicated by a "
                f"number identifier [].\n"
                f"Rank the passages based on their relevance to query: {query}."
            ),
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]
    for rank, content in enumerate(docs, 1):
        messages.append({"role": "user", "content": f"[{rank}] {content}"})
        messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
    messages.append({
        "role": "user",
        "content": (
            f"The following passages are related to query: {query}\n\n"
            f"{relevance_instruction}\n\n"
            f"Finally, rank all {num} passages from most to least relevant."
        ),
    })
    return messages


# ---------------------------------------------------------------------------
# Parse / validate the ranking list
# ---------------------------------------------------------------------------

def _validate_ranking(ranking: List[int], num_docs: int) -> Tuple[List[int], List[int], List[int]]:
    """Validate and fix a ranking list. Returns (permutation, missing, duplicates)."""
    # Convert from 1-indexed to 0-indexed
    parsed = [x - 1 for x in ranking]

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


# ---------------------------------------------------------------------------
# Comparer
# ---------------------------------------------------------------------------

class ListwiseCotComparer(BaseComparer):
    """
    Listwise comparer with chain-of-thought reasoning via structured output.

    The model outputs a JSON object with `thinking` (reasoning) and `ranking`
    (list of passage IDs). Structured output guarantees valid JSON; we still
    validate the ranking for missing/duplicate IDs and retry if needed.
    """

    def __init__(self, model: str, max_doc_tokens: int, max_query_tokens: int = 1024,
                 temperature: Optional[float] = None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 relevance_instruction: str = DEFAULT_RELEVANCE_INSTRUCTION):
        super().__init__(model, max_doc_tokens, max_query_tokens)
        self.num_trimmed_docs = 0
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.relevance_instruction = relevance_instruction

    def _prepare_doc(self, content: str) -> str:
        from .base import _tokenizer
        text = content.replace("Title: Content: ", "").strip()
        encoded = _tokenizer.encode(text)
        if len(encoded) <= self.max_doc_tokens:
            return text
        self.num_trimmed_docs += 1
        return _tokenizer.decode(encoded[:self.max_doc_tokens])

    @async_retry()
    async def compare(self, query: str, docs: List[str]) -> ListwiseCotResult:
        self.num_trimmed_docs = 0
        prepared_query = self._prepare_query(query)
        prepared_docs = [self._prepare_doc(doc) for doc in docs]
        messages = _build_messages(prepared_query, prepared_docs,
                                   system_prompt=self.system_prompt,
                                   relevance_instruction=self.relevance_instruction)

        total_input = 0
        total_output = 0
        total_latency = 0.0
        thinking_text = ""
        raw_response = ""
        # _validate_ranking always produces a valid permutation (appending
        # missing indices in identity order), so we seed with an empty ranking.
        permutation, missing, duplicates = _validate_ranking([], len(docs))

        for attempt in range(1 + MAX_PARSE_RETRIES):
            raw_response, latency_ms, usage = await self.client.get_response(
                self.model, messages, self.temperature,
                response_format=RankingResponse,
            )

            total_input += getattr(usage, "prompt_tokens", 0)
            total_output += getattr(usage, "completion_tokens", 0)
            total_latency += latency_ms

            # Parse structured JSON response
            ranking = None
            try:
                parsed = RankingResponse.model_validate_json(raw_response)
                thinking_text = parsed.thinking
                ranking = parsed.ranking
            except Exception:
                try:
                    data = json.loads(raw_response)
                    parsed = RankingResponse(**data)
                    thinking_text = parsed.thinking
                    ranking = parsed.ranking
                except Exception:
                    # Extract whatever integers we can from the raw response
                    nums = [int(x) for x in re.findall(r'\b(\d+)\b', raw_response)
                            if 1 <= int(x) <= len(docs)]
                    if nums:
                        ranking = nums
                    logger.warning(
                        f"CoT parse attempt {attempt+1}/{1+MAX_PARSE_RETRIES}: "
                        f"failed to parse JSON"
                        f"{f', extracted {len(nums)} indices from raw text' if nums else ''}"
                    )

            # Validate: dedup, range-check, append missing in identity order
            if ranking is not None:
                permutation, missing, duplicates = _validate_ranking(ranking, len(docs))

            if not missing and not duplicates:
                break

            # Build retry prompt
            if attempt < MAX_PARSE_RETRIES:
                issues = []
                if missing:
                    issues.append("missing " + ", ".join(f"[{i+1}]" for i in missing))
                if duplicates:
                    issues.append("duplicate " + ", ".join(f"[{i+1}]" for i in duplicates))
                if issues:
                    logger.warning(
                        f"CoT retry {attempt+1}/{MAX_PARSE_RETRIES}: {' and '.join(issues)}"
                    )
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content":
                    f"Your ranking has {' and '.join(issues)}. "
                    f"Rank ALL {len(docs)} passages exactly once. "
                    f"Respond with the same JSON format."
                    if issues else
                    "Please respond with valid JSON containing 'thinking' (string) "
                    "and 'ranking' (list of integers)."
                })

        return ListwiseCotResult(
            permutation=permutation,
            missing_indices=missing,
            duplicate_indices=duplicates,
            thinking=thinking_text,
            thought_tokens=0,
            num_trimmed_docs=self.num_trimmed_docs,
            input_tokens=total_input,
            output_tokens=total_output,
            latency_ms=total_latency,
            model=self.model,
            raw_prompt=messages,
            raw_response=raw_response,
        )
