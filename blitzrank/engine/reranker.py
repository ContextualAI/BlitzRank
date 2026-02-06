import copy
from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable, Awaitable

from .config import RerankingConfig, ComparerType
from .rerank_runner import run_with_checkpoints
from .comparer import create_comparer, BaseComparer


class RerankerType(Enum):
    SINGLE_PASS = "single_pass"
    SLIDING_WINDOW = "sliding_window"
    TOURNAMENT = "tournament"
    TOURNAMENT_GRAPH = "tournament_graph"


class Selector(ABC):
    """Controls which items to rank and tracks ranking state."""

    hits: List[Dict]

    async def run(
        self,
        query: str,
        comparer: BaseComparer,
        on_iteration: Callable[[dict, List[dict]], Awaitable[None]] | None = None,
        initial_logs: List[dict] | None = None,
    ) -> Tuple[List[int], List[Dict]]:
        """Run the selection process. Returns (final_ranking, logs)."""
        logs = list(initial_logs) if initial_logs else []
        while (indices := self.next_indices()) is not None:
            docs = [self.hits[i]["content"] for i in indices]
            result = await comparer.compare(query, docs)
            self.update(result.permutation)
            logs.append({"indices": indices, **asdict(result)})
            if on_iteration:
                await on_iteration(self.to_state(), logs)
        return self.get_final_ranking(), logs

    def next_indices(self) -> List[int] | None:
        """Return indices to rank next, or None if done."""
        pass

    def update(self, permutation: List[int]) -> None:
        """Apply ranking result. permutation[i] = which input position should be at rank i."""
        pass

    @abstractmethod
    def get_final_ranking(self) -> List[int]:
        """Return final ordering as list of original indices."""
        pass

    @abstractmethod
    def to_state(self) -> dict:
        """Serialize selector state for checkpointing."""
        pass

    @abstractmethod
    def from_state(self, state: dict) -> None:
        """Restore selector state from checkpoint."""
        pass


class SinglePassSelector(Selector):
    def __init__(self, hits: List[Dict], rank_end: int):
        self.hits = hits
        self.indices = list(range(min(len(hits), rank_end)))
        self.done = False
        self.final_ranking = None

    def next_indices(self) -> List[int] | None:
        if self.done:
            return None
        self.done = True
        return self.indices

    def update(self, permutation: List[int]) -> None:
        self.final_ranking = [self.indices[p] for p in permutation]

    def get_final_ranking(self) -> List[int]:
        return self.final_ranking if self.final_ranking else self.indices

    def to_state(self) -> dict:
        return {"done": self.done, "final_ranking": self.final_ranking}

    def from_state(self, state: dict) -> None:
        self.done = state["done"]
        self.final_ranking = state["final_ranking"]


class SlidingWindowSelector(Selector):
    def __init__(self, hits: List[Dict], rank_end: int, window_size: int, step: int, num_rounds: int = 1):
        self.hits = hits
        self.window_size = window_size
        self.step = step
        self.num_rounds = num_rounds
        effective_end = min(len(hits), rank_end)
        self.ranking = list(range(effective_end))
        self.end_pos = effective_end
        self.current_round = 0

    def next_indices(self) -> List[int] | None:
        start = self.end_pos - self.window_size
        if start < 0:
            self.current_round += 1
            if self.current_round >= self.num_rounds:
                return None
            self.end_pos = len(self.ranking)
            start = self.end_pos - self.window_size
            if start < 0:
                return None
        self.current_start = start
        self.current_end = self.end_pos
        indices = [self.ranking[i] for i in range(start, self.end_pos)]

        if self.end_pos == self.window_size:
            self.end_pos = 0
        elif self.end_pos - self.step < self.window_size:
            self.end_pos = self.window_size
        else:
            self.end_pos -= self.step
        return indices

    def update(self, permutation: List[int]) -> None:
        window_slice = [
            self.ranking[i] for i in range(self.current_start, self.current_end)
        ]
        for new_pos, old_pos in enumerate(permutation):
            self.ranking[self.current_start + new_pos] = window_slice[old_pos]

    def get_final_ranking(self) -> List[int]:
        return self.ranking

    def to_state(self) -> dict:
        return {"ranking": self.ranking, "end_pos": self.end_pos, "current_round": self.current_round}

    def from_state(self, state: dict) -> None:
        self.ranking = state["ranking"]
        self.end_pos = state["end_pos"]
        self.current_round = state.get("current_round", 0)


def build_reranked_item_from_final_indices(
    query: str, hits: List[Dict], final_indices: List[int]
) -> Dict[str, Any]:
    """
    Build a reranked result item using original hits and final ordering.

    Required evaluation format (see `evaluate_results` in `run.py`):
    - Output must be a dict with keys: "query" (str) and "hits" (list of dicts).
    - Each hit must include "qid", "docid", and "score"; list order defines rank.
    - `EvalFunction.write_file` writes `qid`, `docid`, `score` and uses list order
      to assign rank in the TREC output, so the returned list must already be in
      the desired ranking order.
    """
    reranked_hits = []
    for new_rank, orig_idx in enumerate(final_indices):
        hit = copy.deepcopy(hits[orig_idx])
        hit["rank"] = new_rank + 1
        hit["score"] = len(final_indices) - new_rank
        reranked_hits.append(hit)
    return {"query": query, "hits": reranked_hits}


async def rerank_item(
    item: Dict, selector: Selector, comparer, on_iteration=None, initial_logs=None
) -> Tuple[Dict, List[Dict]]:
    final_indices, logs = await selector.run(
        item["query"], comparer, on_iteration, initial_logs
    )
    reranked_item = build_reranked_item_from_final_indices(
        item["query"], item["hits"], final_indices
    )
    return reranked_item, logs


async def rerank_results(
    dataset: Dict[str, Any], config: RerankingConfig, logger, output_dir: Path
) -> Dict[str, Any]:
    results = dataset["results"]

    def create_selector(hits: List[Dict]) -> Selector:
        if config.selector.type == "single_pass":
            return SinglePassSelector(hits, config.selector.rank_end)
        elif config.selector.type == "sliding_window":
            return SlidingWindowSelector(
                hits,
                config.selector.rank_end,
                config.selector.window_size,
                config.selector.step,
                config.selector.num_rounds,
            )
        elif config.selector.type == "tournament":
            raise NotImplementedError("Tournament selector is not implemented")
        else:
            raise ValueError(f"Unknown selector type: {config.selector.type}")

    if config.comparer.type not in [ComparerType.LISTWISE_RANK_GPT, ComparerType.CTXL_API]:
        raise ValueError(f"Selector only supports listwise_rank_gpt and ctxl_api comparers, got: {config.comparer.type.value}")
    comparer = create_comparer(config.comparer)

    logger.info(
        f"Reranking with model: {config.comparer.model}, selector: {config.selector.type}, comparer: {config.comparer.type.value}"
    )

    async def process_item(item, idx, checkpoint_entry, checkpoint_writer):
        selector = create_selector(item["hits"])
        initial_logs = None
        if checkpoint_entry:
            selector.from_state(checkpoint_entry["selector_state"])
            initial_logs = checkpoint_entry["logs"]

        async def on_iteration(state, logs):
            await checkpoint_writer.update(
                idx, {"complete": False, "selector_state": state, "logs": logs}
            )

        reranked_item, logs = await rerank_item(
            item, selector, comparer, on_iteration, initial_logs
        )
        return reranked_item, {"iterations": logs}

    reranked, logs = await run_with_checkpoints(
        results,
        config.max_parallel_requests,
        output_dir,
        logger,
        process_item,
        description="Reranking",
    )
    iteration_logs = [
        {"query_idx": idx, **item_logs} for idx, item_logs in enumerate(logs)
    ]
    return {
        "results": reranked,
        "qrels": dataset["qrels"],
        "iteration_logs": iteration_logs,
    }
