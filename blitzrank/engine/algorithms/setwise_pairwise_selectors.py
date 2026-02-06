"""
Setwise and Pairwise ranking selectors using bubble sort, heap sort, and all-pair algorithms.
Ports logic from SetwiseLlmRanker and PairwiseLlmRanker.
"""
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

from ..config import RerankingConfig, ComparerType
from ..comparer import create_comparer
from ..comparer.setwise import SetwiseComparer
from ..comparer.pairwise import PairwiseComparer
from ..reranker import build_reranked_item_from_final_indices
from ..rerank_runner import run_with_checkpoints
from ..logger import ExperimentLogger
from .acurank_selectors import SingleContent, RerankTask


@dataclass
class SetwiseConfig:
    # top_m: Number of top documents to rank.
    #   - bubblesort: runs top_m outer iterations to bubble up top positions
    #   - heapsort: extracts top_m elements from heap then stops
    top_m: int
    num_child: int
    method: str  # "bubblesort" or "heapsort"


@dataclass
class PairwiseConfig:
    # top_m: Number of top documents to rank.
    #   - bubblesort: runs top_m outer iterations to bubble up top positions
    #   - heapsort: extracts top_m elements from heap then stops
    #   - allpair: compares all pairs, but only top_m get explicit scores in output
    top_m: int
    method: str  # "bubblesort", "heapsort", or "allpair"


# ============== Setwise Implementations ==============


async def setwise_bubblesort(
    task: RerankTask,
    comparer: SetwiseComparer,
    config: SetwiseConfig,
) -> tuple[List[SingleContent], Dict[str, Any]]:
    """
    Setwise bubble sort: bubble up the best document to each position.
    Ported from SetwiseLlmRanker.rerank() bubblesort method.
    """
    contents = list(task.contents)
    n = len(contents)
    k = min(config.top_m, n)
    num_child = config.num_child
    call_logs = []
    num_oracle_calls = 0

    if n < 2:
        return contents, {"num_oracle_calls": 0, "iterations": []}

    last_start = n - (num_child + 1)

    for i in range(k):
        start_ind = last_start
        end_ind = last_start + (num_child + 1)
        is_change = False

        while True:
            if start_ind < i:
                start_ind = i
            window = contents[start_ind:end_ind]
            docs = [c.content for c in window]
            result = await comparer.compare(task.query, docs)
            num_oracle_calls += 1
            call_logs.append({
                "phase": "bubblesort",
                "iteration": i,
                "start_ind": start_ind,
                "end_ind": end_ind,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
            })

            best_ind = result.winner_index
            if best_ind != 0:
                contents[start_ind], contents[start_ind + best_ind] = (
                    contents[start_ind + best_ind],
                    contents[start_ind],
                )
                if not is_change:
                    is_change = True
                    if (last_start != n - (num_child + 1)
                            and best_ind == len(window) - 1):
                        last_start += len(window) - 1

            if start_ind == i:
                break

            if not is_change:
                last_start -= num_child

            start_ind -= num_child
            end_ind -= num_child

    return contents, {"num_oracle_calls": num_oracle_calls, "iterations": call_logs}


async def _setwise_heapify(
    contents: List[SingleContent],
    n: int,
    i: int,
    query: str,
    comparer: SetwiseComparer,
    num_child: int,
    call_logs: List[Dict],
) -> int:
    """
    Heapify subtree rooted at index i. Returns number of oracle calls made.
    Ported from SetwiseLlmRanker.heapify().
    """
    num_calls = 0
    stack = [i]

    while stack:
        current = stack.pop()
        first_child = num_child * current + 1
        if first_child >= n:
            continue

        last_child = min(num_child * (current + 1) + 1, n)
        children_indices = list(range(first_child, last_child))
        all_indices = [current] + children_indices
        docs = [contents[idx].content for idx in all_indices]

        result = await comparer.compare(query, docs)
        num_calls += 1
        call_logs.append({
            "phase": "heapsort_heapify",
            "root": current,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
        })

        best_ind = result.winner_index
        if best_ind != 0:
            largest = all_indices[best_ind]
            contents[current], contents[largest] = contents[largest], contents[current]
            stack.append(largest)

    return num_calls


async def setwise_heapsort(
    task: RerankTask,
    comparer: SetwiseComparer,
    config: SetwiseConfig,
) -> tuple[List[SingleContent], Dict[str, Any]]:
    """
    Setwise heap sort: build max-heap then extract k times.
    Ported from SetwiseLlmRanker.heapSort().
    """
    contents = list(task.contents)
    n = len(contents)
    k = min(config.top_m, n)
    num_child = config.num_child
    call_logs = []
    num_oracle_calls = 0

    if n < 2:
        return contents, {"num_oracle_calls": 0, "iterations": []}

    # Build max heap
    for i in range(n // num_child, -1, -1):
        calls = await _setwise_heapify(
            contents, n, i, task.query, comparer, num_child, call_logs
        )
        num_oracle_calls += calls

    # Extract k elements
    ranked = 0
    for i in range(n - 1, 0, -1):
        contents[i], contents[0] = contents[0], contents[i]
        ranked += 1
        if ranked == k:
            break
        calls = await _setwise_heapify(
            contents, i, 0, task.query, comparer, num_child, call_logs
        )
        num_oracle_calls += calls

    contents = list(reversed(contents))
    return contents, {"num_oracle_calls": num_oracle_calls, "iterations": call_logs}


# ============== Pairwise Implementations ==============


async def _pairwise_compare_with_verification(
    query: str,
    doc1: SingleContent,
    doc2: SingleContent,
    comparer: PairwiseComparer,
    call_logs: List[Dict],
) -> tuple[bool, int]:
    """
    Compare two docs with swap verification (both orders).
    Returns (doc1_wins, num_calls).
    doc1 wins only if it wins in BOTH comparison orders.
    """
    # Forward: doc1 as A, doc2 as B
    result1 = await comparer.compare(query, [doc1.content, doc2.content])
    call_logs.append({
        "phase": "pairwise_compare",
        "comparison_type": "forward",
        "input_tokens": result1.input_tokens,
        "output_tokens": result1.output_tokens,
        "latency_ms": result1.latency_ms,
    })

    # Reverse: doc2 as A, doc1 as B
    result2 = await comparer.compare(query, [doc2.content, doc1.content])
    call_logs.append({
        "phase": "pairwise_compare",
        "comparison_type": "reverse",
        "input_tokens": result2.input_tokens,
        "output_tokens": result2.output_tokens,
        "latency_ms": result2.latency_ms,
    })

    # doc1 wins if: forward says A wins (0) AND reverse says B wins (1)
    doc1_wins = (result1.winner_index == 0 and result2.winner_index == 1)
    return doc1_wins, 2


async def pairwise_bubblesort(
    task: RerankTask,
    comparer: PairwiseComparer,
    config: PairwiseConfig,
) -> tuple[List[SingleContent], Dict[str, Any]]:
    """
    Pairwise bubble sort with swap verification.
    Ported from PairwiseLlmRanker.rerank() bubblesort method.
    """
    contents = list(task.contents)
    n = len(contents)
    k = min(config.top_m, n)
    call_logs = []
    num_oracle_calls = 0

    if n < 2:
        return contents, {"num_oracle_calls": 0, "iterations": []}

    last_end = n - 1
    for i in range(k):
        current_ind = last_end
        is_change = False

        while current_ind > i:
            doc1 = contents[current_ind]
            doc2 = contents[current_ind - 1]
            doc1_wins, calls = await _pairwise_compare_with_verification(
                task.query, doc1, doc2, comparer, call_logs
            )
            num_oracle_calls += calls

            if doc1_wins:
                contents[current_ind - 1], contents[current_ind] = (
                    contents[current_ind],
                    contents[current_ind - 1],
                )
                if not is_change:
                    is_change = True
                    if last_end != n - 1:
                        last_end += 1

            if not is_change:
                last_end -= 1

            current_ind -= 1

    return contents, {"num_oracle_calls": num_oracle_calls, "iterations": call_logs}


async def pairwise_heapsort(
    task: RerankTask,
    comparer: PairwiseComparer,
    config: PairwiseConfig,
) -> tuple[List[SingleContent], Dict[str, Any]]:
    """
    Pairwise heap sort: binary heap with pairwise comparisons.
    Ported from PairwiseLlmRanker.heapSort().
    """
    contents = list(task.contents)
    n = len(contents)
    k = min(config.top_m, n)
    call_logs = []
    num_oracle_calls = 0

    if n < 2:
        return contents, {"num_oracle_calls": 0, "iterations": []}

    async def heapify(arr: List[SingleContent], heap_size: int, i: int) -> int:
        """Heapify subtree rooted at index i. Returns number of oracle calls."""
        nonlocal call_logs
        calls = 0
        stack = [(i, False)]  # (index, needs_recheck)

        while stack:
            current, _ = stack.pop()
            largest = current
            left = 2 * current + 1
            right = 2 * current + 2

            # Compare with left child
            if left < heap_size:
                left_wins, c = await _pairwise_compare_with_verification(
                    task.query, arr[left], arr[largest], comparer, call_logs
                )
                calls += c
                if left_wins:
                    largest = left

            # Compare with right child
            if right < heap_size:
                right_wins, c = await _pairwise_compare_with_verification(
                    task.query, arr[right], arr[largest], comparer, call_logs
                )
                calls += c
                if right_wins:
                    largest = right

            if largest != current:
                arr[current], arr[largest] = arr[largest], arr[current]
                stack.append((largest, False))

        return calls

    # Build max heap
    for i in range(n // 2, -1, -1):
        calls = await heapify(contents, n, i)
        num_oracle_calls += calls

    # Extract k elements
    ranked = 0
    for i in range(n - 1, 0, -1):
        contents[i], contents[0] = contents[0], contents[i]
        ranked += 1
        if ranked == k:
            break
        calls = await heapify(contents, i, 0)
        num_oracle_calls += calls

    contents = list(reversed(contents))
    return contents, {"num_oracle_calls": num_oracle_calls, "iterations": call_logs}


async def pairwise_allpair(
    task: RerankTask,
    comparer: PairwiseComparer,
    config: PairwiseConfig,
) -> tuple[List[SingleContent], Dict[str, Any]]:
    """
    Pairwise all-pairs: compare every pair with swap verification.
    Ported from PairwiseLlmRanker.rerank() allpair method.
    """
    contents = list(task.contents)
    n = len(contents)
    call_logs = []
    num_oracle_calls = 0

    if n < 2:
        return contents, {"num_oracle_calls": 0, "iterations": []}

    scores = {c.docid: 0.0 for c in contents}
    doc_pairs = list(combinations(range(n), 2))

    for idx1, idx2 in doc_pairs:
        doc1, doc2 = contents[idx1], contents[idx2]

        # Forward comparison
        result1 = await comparer.compare(task.query, [doc1.content, doc2.content])
        call_logs.append({
            "phase": "allpair",
            "comparison_type": "forward",
            "doc1_idx": idx1,
            "doc2_idx": idx2,
            "input_tokens": result1.input_tokens,
            "output_tokens": result1.output_tokens,
            "latency_ms": result1.latency_ms,
        })

        # Reverse comparison
        result2 = await comparer.compare(task.query, [doc2.content, doc1.content])
        call_logs.append({
            "phase": "allpair",
            "comparison_type": "reverse",
            "doc1_idx": idx2,
            "doc2_idx": idx1,
            "input_tokens": result2.input_tokens,
            "output_tokens": result2.output_tokens,
            "latency_ms": result2.latency_ms,
        })
        num_oracle_calls += 2

        # Score based on consistency
        if result1.winner_index == 0 and result2.winner_index == 1:
            # doc1 wins both
            scores[doc1.docid] += 1
        elif result1.winner_index == 1 and result2.winner_index == 0:
            # doc2 wins both
            scores[doc2.docid] += 1
        else:
            # Conflict: tie
            scores[doc1.docid] += 0.5
            scores[doc2.docid] += 0.5

    # Sort by scores descending
    contents.sort(key=lambda c: scores[c.docid], reverse=True)
    return contents, {"num_oracle_calls": num_oracle_calls, "iterations": call_logs}


# ============== Entry Points ==============


async def setwise_rerank_dataset_from_config(
    dataset_raw: Dict[str, Any],
    reranking_config: RerankingConfig,
    logger: ExperimentLogger,
    output_dir: Path,
) -> Dict[str, Any]:
    """Entry point for setwise reranking."""
    if reranking_config.comparer.type != ComparerType.SETWISE:
        raise ValueError(
            f"Setwise selector requires setwise comparer, got: {reranking_config.comparer.type.value}"
        )

    list_of_tasks = [
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

    setwise_config = SetwiseConfig(
        top_m=reranking_config.selector.top_m,
        num_child=reranking_config.selector.num_child,
        method=reranking_config.selector.sorting_method,
    )

    async def process_item(task, idx, _checkpoint_entry, _checkpoint_writer):
        comparer = create_comparer(reranking_config.comparer)

        if setwise_config.method == "bubblesort":
            result, item_logs = await setwise_bubblesort(task, comparer, setwise_config)
        elif setwise_config.method == "heapsort":
            result, item_logs = await setwise_heapsort(task, comparer, setwise_config)
        else:
            raise ValueError(f"Unknown setwise method: {setwise_config.method}")

        final_indices = [c.orig_idx for c in result]
        reranked_item = build_reranked_item_from_final_indices(
            task.query, task.hits, final_indices
        )
        return reranked_item, item_logs

    reranked, logs = await run_with_checkpoints(
        list_of_tasks,
        reranking_config.max_parallel_requests,
        output_dir,
        logger,
        process_item,
        description="Setwise reranking",
    )
    iteration_logs = [{"query_idx": idx, **item_logs} for idx, item_logs in enumerate(logs)]
    return {
        "results": reranked,
        "qrels": dataset_raw["qrels"],
        "iteration_logs": iteration_logs,
    }


async def pairwise_rerank_dataset_from_config(
    dataset_raw: Dict[str, Any],
    reranking_config: RerankingConfig,
    logger: ExperimentLogger,
    output_dir: Path,
) -> Dict[str, Any]:
    """Entry point for pairwise reranking."""
    if reranking_config.comparer.type != ComparerType.PAIRWISE:
        raise ValueError(
            f"Pairwise selector requires pairwise comparer, got: {reranking_config.comparer.type.value}"
        )

    list_of_tasks = [
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

    pairwise_config = PairwiseConfig(
        top_m=reranking_config.selector.top_m,
        method=reranking_config.selector.sorting_method,
    )

    async def process_item(task, idx, _checkpoint_entry, _checkpoint_writer):
        comparer = create_comparer(reranking_config.comparer)

        if pairwise_config.method == "bubblesort":
            result, item_logs = await pairwise_bubblesort(task, comparer, pairwise_config)
        elif pairwise_config.method == "heapsort":
            result, item_logs = await pairwise_heapsort(task, comparer, pairwise_config)
        elif pairwise_config.method == "allpair":
            result, item_logs = await pairwise_allpair(task, comparer, pairwise_config)
        else:
            raise ValueError(f"Unknown pairwise method: {pairwise_config.method}")

        final_indices = [c.orig_idx for c in result]
        reranked_item = build_reranked_item_from_final_indices(
            task.query, task.hits, final_indices
        )
        return reranked_item, item_logs

    reranked, logs = await run_with_checkpoints(
        list_of_tasks,
        reranking_config.max_parallel_requests,
        output_dir,
        logger,
        process_item,
        description="Pairwise reranking",
    )
    iteration_logs = [{"query_idx": idx, **item_logs} for idx, item_logs in enumerate(logs)]
    return {
        "results": reranked,
        "qrels": dataset_raw["qrels"],
        "iteration_logs": iteration_logs,
    }
