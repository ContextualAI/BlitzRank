"""
Factory function for creating comparers.
"""
from typing import TYPE_CHECKING

from .base import BaseComparer
from .setwise import SetwiseComparer
from .pairwise import PairwiseComparer
from .listwise_rank_gpt import ListwiseRankGptComparer
from .listwise_cot import (
    DEFAULT_RELEVANCE_INSTRUCTION,
    DEFAULT_SYSTEM_PROMPT,
    ListwiseCotComparer,
)
from .ctxl_api import CtxlApiComparer

if TYPE_CHECKING:
    from ..config import ComparerConfig


def create_comparer(config: "ComparerConfig") -> BaseComparer:
    """
    Create a comparer based on config type.
    """
    from ..config import ComparerType

    if config.type == ComparerType.SETWISE:
        return SetwiseComparer(
            model=config.model,
            max_doc_tokens=config.max_doc_tokens,
            max_query_tokens=config.max_query_tokens,
            temperature=config.temperature,
        )
    elif config.type == ComparerType.PAIRWISE:
        return PairwiseComparer(
            model=config.model,
            max_doc_tokens=config.max_doc_tokens,
            max_query_tokens=config.max_query_tokens,
            temperature=config.temperature,
        )
    elif config.type == ComparerType.LISTWISE_RANK_GPT:
        return ListwiseRankGptComparer(
            model=config.model,
            max_doc_tokens=config.max_doc_tokens,
            max_query_tokens=config.max_query_tokens,
            temperature=config.temperature,
        )
    elif config.type == ComparerType.LISTWISE_COT:
        return ListwiseCotComparer(
            model=config.model,
            max_doc_tokens=config.max_doc_tokens,
            max_query_tokens=config.max_query_tokens,
            temperature=config.temperature,
            system_prompt=config.system_prompt or DEFAULT_SYSTEM_PROMPT,
            relevance_instruction=(
                config.relevance_instruction or DEFAULT_RELEVANCE_INSTRUCTION
            ),
        )
    elif config.type == ComparerType.CTXL_API:
        return CtxlApiComparer(
            model=config.model,
            max_doc_tokens=config.max_doc_tokens,
            max_query_tokens=config.max_query_tokens,
        )
    else:
        raise ValueError(f"Unknown comparer type: {config.type}")
