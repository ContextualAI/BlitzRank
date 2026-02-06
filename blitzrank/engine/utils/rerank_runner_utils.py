import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List


class CheckpointWriter:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.queue: asyncio.Queue = asyncio.Queue()
        self.state: Dict[int, dict] = {}
        self._task: asyncio.Task | None = None

    def load(self) -> Dict[int, dict]:
        if self.filepath.exists():
            try:
                with open(self.filepath) as f:
                    self.state = {int(k): v for k, v in json.load(f).items()}
            except (json.JSONDecodeError, ValueError):
                self.filepath.unlink()
                self.state = {}
        return self.state

    async def start(self):
        self._task = asyncio.create_task(self._worker())

    async def _worker(self):
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            query_idx, checkpoint = item
            self.state[query_idx] = checkpoint
            with open(self.filepath, "w") as f:
                json.dump(self.state, f)
            self.queue.task_done()

    async def update(self, query_idx: int, checkpoint: dict):
        await self.queue.put((query_idx, checkpoint))

    async def close(self):
        await self.queue.put(None)
        await self.queue.join()
        if self._task:
            await self._task


def _extract_numeric(entries: List[dict], key: str) -> List[float]:
    return [e[key] for e in entries if isinstance(e.get(key), (int, float))]


def _extract_list_lengths(entries: List[dict], key: str) -> List[int]:
    return [len(e[key]) for e in entries if isinstance(e.get(key), list)]


def _add_aggregates(summary: dict, prefix: str, values: List[float]) -> None:
    if not values:
        return
    summary[f"{prefix}_total"] = sum(values)
    summary[f"{prefix}_avg"] = summary[f"{prefix}_total"] / len(values)
    summary[f"{prefix}_max"] = max(values)
    summary[f"{prefix}_min"] = min(values)


def build_task_summary(
    query_idx: int, latency_ms: float, item_logs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a summary from item_logs dict.

    Expected format:
        {
            # Top-level metrics (optional, copied directly)
            "num_rounds": int,
            "num_oracle_calls": int,
            # Per-LLM-call logs (required)
            "iterations": [{"input_tokens": ..., "output_tokens": ..., ...}, ...]
        }
    """
    summary = {"query_idx": query_idx, "latency_ms": latency_ms}

    for k, v in item_logs.items():
        if k != "iterations" and isinstance(v, (int, float)):
            summary[k] = v

    iterations = item_logs.get("iterations", [])
    if not iterations:
        return summary

    summary["num_llm_calls"] = len(iterations)

    # Aggregate numeric metrics
    for source_key, prefix in [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("thought_tokens", "thought_tokens"),
        ("latency_ms", "llm_latency_ms"),
        ("num_trimmed_docs", "num_trimmed_docs"),
    ]:
        _add_aggregates(summary, prefix, _extract_numeric(iterations, source_key))

    # Combined token stats
    input_total = summary.get("input_tokens_total", 0)
    output_total = summary.get("output_tokens_total", 0)
    if input_total or output_total:
        summary["tokens_total"] = input_total + output_total
        summary["tokens_avg"] = summary["tokens_total"] / len(iterations)

    # Aggregate list-length metrics
    for source_key, prefix in [
        ("missing_indices", "missing_indices"),
        ("duplicate_indices", "duplicate_indices"),
        ("indices", "num_docs"),
    ]:
        lengths = _extract_list_lengths(iterations, source_key)
        if lengths:
            summary[f"{prefix}_total"] = sum(lengths)
            summary[f"{prefix}_max"] = max(lengths)

    return summary
