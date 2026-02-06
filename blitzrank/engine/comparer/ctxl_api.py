"""
CTXL API comparer: uses Contextual AI's reranking API.
"""
import os
import time
from dataclasses import dataclass
from typing import List

import httpx

from .base import BaseComparer, CompareResultBase
from ..utils.retry_utils import async_retry

API_TIMEOUT = 60.0


@dataclass
class CtxlApiResult(CompareResultBase):
    permutation: List[int]


class CtxlApiComparer(BaseComparer):
    """
    Comparer that uses Contextual AI's reranking API.
    Returns a full permutation ranking.
    """

    @async_retry()
    async def compare(self, query: str, docs: List[str]) -> CtxlApiResult:
        ctxl_api_url = os.getenv("CTXL_API_URL")
        api_key = os.getenv("CTXL_API_KEY")
        if ctxl_api_url is None or api_key is None:
            raise ValueError("Please set CTXL_API_URL and CTXL_API_KEY")

        prepared_docs = [self._prepare_doc(doc) for doc in docs]
        request_body = {
            "query": query,
            "documents": prepared_docs,
            "top_n": len(prepared_docs),
            "model": self.model,
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(
                ctxl_api_url,
                json=request_body,
                headers={
                    "accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        if "results" not in data:
            raise ValueError(f"Unexpected API response: {data}")

        permutation = [r["index"] for r in data["results"]]

        return CtxlApiResult(
            permutation=permutation,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            model=self.model,
            raw_prompt=request_body,
            raw_response=str(data),
        )
