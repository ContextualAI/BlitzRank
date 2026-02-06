from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ...comparer import create_comparer
from ...comparer.listwise_rank_gpt import ListwiseRankGptResult
from ...config import ComparerConfig, ComparerType, RerankingConfig
from ...logger import ExperimentLogger
from ...rerank_runner import run_with_checkpoints
from ...reranker import build_reranked_item_from_final_indices
from .compare_oracle import (
    CompareOracle,
    Item,
    OracleResult,
    make_edges_from_linear_order,
)
from .tournament_graph_sort import (
    TournamentGraphSortConfig,
    TournamentGraphSortResult,
    tournament_graph_sort_async,
)


@dataclass
class SingleContent:
    content: str
    qid: int
    docid: str
    rank: int
    score: float
    orig_idx: int


@dataclass
class RerankTask:
    """
    Given a query and a list of contents (docs/chunks/info related to the query),
    we want to output a list of indices of the contents that are the top m contents relevant to the query.
    """

    query: str
    contents: list[SingleContent]
    hits: list[Dict[str, Any]]


class ContentItem(Item):
    def __init__(self, content: SingleContent, query: str):
        super().__init__(content.docid)
        self.content: SingleContent = content
        self.query: str = query


class LLMCompareOracle(CompareOracle):
    def __init__(self, k: int, comparer_config: ComparerConfig):
        super().__init__(k)
        self.comparer_config = comparer_config
        self.comparer = create_comparer(comparer_config)
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self.thought_tokens_total = 0
        self.llm_latency_ms_total = 0.0
        self.num_trimmed_docs_total = 0
        self.call_logs: list[dict] = []

    async def _compare_k_async(self, items: List[ContentItem]) -> OracleResult:
        results: ListwiseRankGptResult = await self.comparer.compare(
            query=items[0].query,
            docs=[item.content.content for item in items],
        )
        self.input_tokens_total += results.input_tokens
        self.output_tokens_total += results.output_tokens
        self.thought_tokens_total += results.thought_tokens
        self.llm_latency_ms_total += results.latency_ms
        self.num_trimmed_docs_total += results.num_trimmed_docs

        call_log = {
            "input_tokens": results.input_tokens,
            "output_tokens": results.output_tokens,
            "thought_tokens": results.thought_tokens,
            "latency_ms": results.latency_ms,
            "missing_indices": results.missing_indices,
            "duplicate_indices": results.duplicate_indices,
            "num_trimmed_docs": results.num_trimmed_docs,
        }
        self.call_logs.append(call_log)

        ordered_items = [items[i] for i in results.permutation]
        return OracleResult(
            edges=make_edges_from_linear_order(ordered_items),
            metadata=call_log,
        )


@dataclass
class TournamentGraphRerankConfig:
    """
    Config for the tournament graph reranker.
    """

    top_m: int
    oracle: LLMCompareOracle
    max_concurrent_tasks: int = 1


async def tournament_graph_rerank_dataset_from_config(
    dataset_raw: Dict[str, Any],
    reranking_config: RerankingConfig,
    logger,
    output_dir: Path,
) -> Dict[str, Any]:
    if reranking_config.comparer.type != ComparerType.LISTWISE_RANK_GPT:
        raise ValueError(
            f"Tournament graph reranking only supports comparer.type == {ComparerType.LISTWISE_RANK_GPT}"
        )
    return await tournament_graph_rerank_dataset(
        dataset_raw, reranking_config, logger, output_dir
    )


async def tournament_graph_rerank_dataset(
    dataset_raw: Dict[str, Any],
    reranking_config: RerankingConfig,
    logger: ExperimentLogger,
    output_dir: Path,
) -> Dict[str, Any]:
    list_of_tasks: list[RerankTask] = [
        RerankTask(
            query=task["query"],
            contents=[
                SingleContent(**hit, orig_idx=idx)
                for idx, hit in enumerate(task["hits"])
            ],
            hits=task["hits"],
        )
        for task in dataset_raw["results"]
    ]

    async def process_item(task, idx, _checkpoint_entry, _checkpoint_writer):
        oracle = LLMCompareOracle(
            reranking_config.selector.window_size, reranking_config.comparer
        )
        task_config = TournamentGraphRerankConfig(
            top_m=reranking_config.selector.top_m,
            oracle=oracle,
            max_concurrent_tasks=1,
        )

        sort_config = TournamentGraphSortConfig(
            on_round_complete=lambda round_log: logger.log_round(idx, round_log)
        )
        result = await rerank_with_tournament_graph(task, task_config, sort_config)
        final_indices = [item.content.orig_idx for item in result.results]
        reranked_item = build_reranked_item_from_final_indices(
            task.query, task.hits, final_indices
        )
        scc_stats = result.get_scc_stats()
        item_logs = {
            "num_rounds": result.num_rounds,
            "num_oracle_calls": result.num_oracle_calls,
            # Global SCC stats
            "num_3_cycles": scc_stats.num_3_cycles,
            "num_sccs": scc_stats.num_sccs,
            "num_non_trivial_sccs": scc_stats.num_non_trivial_sccs,
            "total_nodes_in_non_trivial_sccs": scc_stats.total_nodes_in_non_trivial_sccs,
            "max_scc_size": scc_stats.max_scc_size,
            "max_scc_size_rank": scc_stats.max_scc_size_rank,
            # Top-m SCC stats
            "max_scc_size_top_m": scc_stats.max_scc_size_top_m,
            "avg_scc_size_top_m": scc_stats.avg_scc_size_top_m,
            "num_sccs_top_m": scc_stats.num_sccs_top_m,
            "iterations": oracle.call_logs,
        }
        return reranked_item, item_logs

    reranked, logs = await run_with_checkpoints(
        list_of_tasks,
        reranking_config.max_parallel_requests,
        output_dir,
        logger,
        process_item,
        description="tasks processed",
    )
    iteration_logs = [
        {"query_idx": idx, **item_logs} for idx, item_logs in enumerate(logs)
    ]
    return {
        "results": reranked,
        "qrels": dataset_raw["qrels"],
        "iteration_logs": iteration_logs,
    }


async def rerank_with_tournament_graph(
    task: RerankTask,
    config: TournamentGraphRerankConfig,
    tournament_graph_sort_config: TournamentGraphSortConfig | None = None,
) -> TournamentGraphSortResult:
    """
    Given a RerankTask, output a list of indices (from RerankTask.contents) corresponding to the top m contents.
    """
    result = await tournament_graph_sort_async(
        [ContentItem(content, task.query) for content in task.contents],
        config.oracle,
        config.top_m,
        tournament_graph_sort_config,
    )
    return result
