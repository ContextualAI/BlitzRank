import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import trueskill
from scipy.stats import norm
from trueskill import Rating

from ..config import RerankingConfig
from ..comparer import create_comparer
from ..comparer.base import BaseComparer
from ..reranker import build_reranked_item_from_final_indices
from ..rerank_runner import run_with_checkpoints
from ..logger import ExperimentLogger
from .acurank_adaptive_utils import _find_threshold_binary_search


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
    query: str
    contents: List[SingleContent]
    hits: List[Dict[str, Any]]


@dataclass
class AcuRankConfig:
    window_size: int
    tol: float
    hard_constraint: int
    uncertain_U: int
    R: int
    break_mode: str


@dataclass
class TourRankConfig:
    num_rounds: int
    window_size: int


async def acurank_rerank_dataset_from_config(
    dataset_raw: Dict[str, Any],
    reranking_config: RerankingConfig,
    logger: ExperimentLogger,
    output_dir: Path,
) -> Dict[str, Any]:
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

    acurank_config = AcuRankConfig(
        window_size=reranking_config.selector.window_size,
        tol=reranking_config.selector.tol,
        hard_constraint=reranking_config.selector.hard_constraint,
        uncertain_U=reranking_config.selector.uncertain_U,
        R=reranking_config.selector.R,
        break_mode=reranking_config.selector.break_mode,
    )

    async def process_item(task, idx, _checkpoint_entry, _checkpoint_writer):
        comparer = create_comparer(reranking_config.comparer)
        result, item_logs = await acurank_rerank_single(
            task, comparer, acurank_config, logger, idx
        )
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
        description="AcuRank reranking",
    )
    iteration_logs = [{"query_idx": idx, **item_logs} for idx, item_logs in enumerate(logs)]
    return {
        "results": reranked,
        "qrels": dataset_raw["qrels"],
        "iteration_logs": iteration_logs,
    }


async def acurank_rerank_single(
    task: RerankTask,
    comparer: BaseComparer,
    config: AcuRankConfig,
    logger: ExperimentLogger,
    query_idx: int,
) -> tuple[List[SingleContent], Dict[str, Any]]:
    candidates = [
        {"content": c, "rating": Rating()} for c in task.contents
    ]
    num_candidates = len(candidates)
    call_logs = []
    num_oracle_calls = 0

    if num_candidates < 2:
        return [c["content"] for c in candidates], {"num_oracle_calls": 0, "iterations": []}

    # Initial pass: window-based comparison
    for start in range(0, num_candidates, config.window_size):
        end = min(start + config.window_size, num_candidates)
        window_candidates = candidates[start:end]
        if len(window_candidates) < 2:
            continue

        docs = [c["content"].content for c in window_candidates]
        result = await comparer.compare(task.query, docs)
        num_oracle_calls += 1
        call_logs.append({
            "phase": "initial",
            "start": start,
            "end": end,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
        })

        window_candidates = _update_ratings(result.permutation, window_candidates)
        window_candidates.sort(key=lambda c: c["rating"].mu, reverse=True)
        candidates[start:end] = window_candidates

    # Global sort after initial pass
    candidates.sort(key=lambda c: c["rating"].mu, reverse=True)

    # Adaptive phase
    if config.break_mode == "reduce_uncertain":
        candidates, adaptive_logs, adaptive_calls = await _adaptive_prob_rank_reduce_uncertain(
            task.query, candidates, comparer, config
        )
    else:
        candidates, adaptive_logs, adaptive_calls = await _adaptive_prob_rank_top10_nochange(
            task.query, candidates, comparer, config
        )
    call_logs.extend(adaptive_logs)
    num_oracle_calls += adaptive_calls

    final_contents = [c["content"] for c in candidates]
    item_logs = {
        "num_oracle_calls": num_oracle_calls,
        "iterations": call_logs,
    }
    return final_contents, item_logs


def _update_ratings(permutation: List[int], window_candidates: List[Dict]) -> List[Dict]:
    if len(permutation) <= 1:
        return window_candidates

    sorted_candidates = [window_candidates[i] for i in permutation]
    updated_ratings = trueskill.rate(
        [[c["rating"]] for c in sorted_candidates],
        ranks=list(range(len(sorted_candidates)))
    )
    for i, c in enumerate(sorted_candidates):
        c["rating"] = updated_ratings[i][0]

    # Restore to original order
    inverse_order = [0] * len(permutation)
    for new_idx, old_idx in enumerate(permutation):
        inverse_order[old_idx] = new_idx
    return [sorted_candidates[inverse_order[i]] for i in range(len(permutation))]


def _compute_probs_above_t(mus, sigmas, t):
    probs = []
    for mu_i, sigma_i in zip(mus, sigmas):
        z = (t - mu_i) / sigma_i
        probs.append(float(1.0 - norm.cdf(z)))
    return probs


async def _adaptive_prob_rank_reduce_uncertain(
    query: str,
    candidates: List[Dict],
    comparer: BaseComparer,
    config: AcuRankConfig,
) -> tuple[List[Dict], List[Dict], int]:
    num_candidates = len(candidates)
    call_logs = []
    num_calls = 0
    iteration = 0

    if num_candidates < 2:
        return candidates, call_logs, num_calls

    should_break = False
    while True:
        mus = torch.tensor([c["rating"].mu for c in candidates])
        sigmas = torch.tensor([c["rating"].sigma for c in candidates])
        threshold_sumto = config.R / num_candidates
        t = _find_threshold_binary_search(threshold_sumto, mus, sigmas)
        probs_above_t = np.array(_compute_probs_above_t(mus, sigmas, t))

        # Re-order candidates by probs_above_t
        sorted_pairs = sorted(zip(candidates, probs_above_t), key=lambda x: x[1], reverse=True)
        candidates = [c for c, _ in sorted_pairs]
        sorted_probs = np.array([p for _, p in sorted_pairs])

        mask = (sorted_probs > config.tol) & (sorted_probs < (1.0 - config.tol))
        chosen_indices = np.where(mask)[0]

        if len(chosen_indices) < config.uncertain_U:
            chosen_indices = np.where(sorted_probs > config.tol)[0]
            should_break = True

        subset = [copy.deepcopy(candidates[i]) for i in chosen_indices]
        grouped = _group_candidates(subset, config.window_size)

        if iteration + len(grouped) >= config.hard_constraint:
            grouped = grouped[:config.hard_constraint - iteration]

        for chunk_candidates, chunk_indices in grouped:
            if len(chunk_candidates) < 2:
                continue

            docs = [c["content"].content for c in chunk_candidates]
            result = await comparer.compare(query, docs)
            num_calls += 1
            call_logs.append({
                "phase": "adaptive",
                "iteration": iteration,
                "num_uncertain": len(chosen_indices),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
            })

            updated = _update_ratings(result.permutation, chunk_candidates)
            for i, u in enumerate(updated):
                orig_idx = chosen_indices[chunk_indices[i]]
                candidates[orig_idx] = u

        iteration += len(grouped)

        if iteration >= config.hard_constraint or should_break:
            candidates.sort(key=lambda c: c["rating"].mu, reverse=True)
            break

    return candidates, call_logs, num_calls


async def _adaptive_prob_rank_top10_nochange(
    query: str,
    candidates: List[Dict],
    comparer: BaseComparer,
    config: AcuRankConfig,
) -> tuple[List[Dict], List[Dict], int]:
    num_candidates = len(candidates)
    call_logs = []
    num_calls = 0
    iteration = 0
    nochange_cnt = 0

    if num_candidates < 2:
        return candidates, call_logs, num_calls

    def get_top_r_pids(cands):
        sorted_cands = sorted(cands, key=lambda c: c["rating"].mu, reverse=True)[:config.R]
        return [c["content"].docid for c in sorted_cands]

    prev_top_pids = get_top_r_pids(candidates)

    while True:
        mus = torch.tensor([c["rating"].mu for c in candidates])
        sigmas = torch.tensor([c["rating"].sigma for c in candidates])
        threshold_sumto = config.R / num_candidates
        t = _find_threshold_binary_search(threshold_sumto, mus, sigmas)
        probs_above_t = np.array(_compute_probs_above_t(mus, sigmas, t))

        sorted_pairs = sorted(zip(candidates, probs_above_t), key=lambda x: x[1], reverse=True)
        candidates = [c for c, _ in sorted_pairs]
        sorted_probs = np.array([p for _, p in sorted_pairs])

        mask = (sorted_probs > config.tol) & (sorted_probs < (1.0 - config.tol))
        chosen_indices = np.where(mask)[0]

        subset = [copy.deepcopy(candidates[i]) for i in chosen_indices]
        grouped = _group_candidates(subset, config.window_size)

        for chunk_candidates, chunk_indices in grouped:
            if len(chunk_candidates) < 2:
                continue

            docs = [c["content"].content for c in chunk_candidates]
            result = await comparer.compare(query, docs)
            num_calls += 1
            call_logs.append({
                "phase": "adaptive",
                "iteration": iteration,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
            })

            updated = _update_ratings(result.permutation, chunk_candidates)
            for i, u in enumerate(updated):
                orig_idx = chosen_indices[chunk_indices[i]]
                candidates[orig_idx] = u

        iteration += len(grouped)
        cur_top_pids = get_top_r_pids(candidates)

        if prev_top_pids == cur_top_pids:
            nochange_cnt += 1

        if iteration >= config.hard_constraint or nochange_cnt >= 1:
            candidates.sort(key=lambda c: c["rating"].mu, reverse=True)
            break

        prev_top_pids = cur_top_pids

    return candidates, call_logs, num_calls


def _group_candidates(candidates: List[Dict], window_size: int) -> List[tuple]:
    indices = list(range(len(candidates)))
    groups = []
    for i in range(0, len(candidates), window_size):
        chunk_indices = indices[i:i + window_size]
        chunk_candidates = [candidates[j] for j in chunk_indices]
        groups.append((chunk_candidates, chunk_indices))
    return groups


# ============== TourRank Implementation ==============


async def tourrank_rerank_dataset_from_config(
    dataset_raw: Dict[str, Any],
    reranking_config: RerankingConfig,
    logger: ExperimentLogger,
    output_dir: Path,
) -> Dict[str, Any]:
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

    tourrank_config = TourRankConfig(
        num_rounds=reranking_config.selector.num_rounds,
        window_size=reranking_config.selector.window_size,
    )

    async def process_item(task, idx, _checkpoint_entry, _checkpoint_writer):
        comparer = create_comparer(reranking_config.comparer)
        result, item_logs = await tourrank_rerank_single(
            task, comparer, tourrank_config
        )
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
        description="TourRank reranking",
    )
    iteration_logs = [{"query_idx": idx, **item_logs} for idx, item_logs in enumerate(logs)]
    return {
        "results": reranked,
        "qrels": dataset_raw["qrels"],
        "iteration_logs": iteration_logs,
    }


async def tourrank_rerank_single(
    task: RerankTask,
    comparer: BaseComparer,
    config: TourRankConfig,
) -> tuple[List[SingleContent], Dict[str, Any]]:
    contents = list(task.contents)
    doc_ids = [c.docid for c in contents]
    content_map = {c.docid: c for c in contents}
    call_logs = []
    num_oracle_calls = 0

    score_dict = {doc_id: 0 for doc_id in doc_ids}

    for _ in range(config.num_rounds):
        round_scores, round_logs, round_calls = await _filter_processing(
            task.query, doc_ids, content_map, comparer
        )
        call_logs.extend(round_logs)
        num_oracle_calls += round_calls
        for doc_id, score in round_scores.items():
            score_dict[doc_id] += score

    ranked_ids = _sort_docs_by_relevance(list(score_dict.keys()), list(score_dict.values()))
    result = [content_map[doc_id] for doc_id in ranked_ids]

    item_logs = {
        "num_oracle_calls": num_oracle_calls,
        "iterations": call_logs,
    }
    return result, item_logs


def _sort_docs_by_relevance(doc_ids: List[str], scores: List[float]) -> List[str]:
    combined = list(zip(doc_ids, scores))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_combined]


def _get_groups_skip(doc_ids: List[str], to_n_groups: int, m_docs_per_group: int) -> List[List[str]]:
    groups = []
    for i in range(to_n_groups):
        group = []
        for j in range(m_docs_per_group):
            idx = j * to_n_groups + i
            if idx < len(doc_ids):
                group.append(doc_ids[idx])
        groups.append(group)
    return groups


async def _group_processing(
    query: str,
    group: List[str],
    content_map: Dict[str, SingleContent],
    comparer: BaseComparer,
    N: int,
    M: int,
) -> tuple[Dict[str, int], Dict]:
    random.shuffle(group)
    docs = [content_map[doc_id].content for doc_id in group]

    result = await comparer.compare(query, docs)
    top_m_indices = result.permutation[:M]
    top_m_ids = [group[i] for i in top_m_indices]

    score_dict = {doc_id: 1 for doc_id in top_m_ids}
    log = {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
    }
    return score_dict, log


async def _filter_processing(
    query: str,
    doc_ids: List[str],
    content_map: Dict[str, SingleContent],
    comparer: BaseComparer,
) -> tuple[Dict[str, int], List[Dict], int]:
    score_dict = {doc_id: 0 for doc_id in doc_ids}
    call_logs = []
    num_calls = 0

    # Stage 1: 100 -> 50 (5 groups of 20, keep top 10 from each)
    if len(doc_ids) > 50:
        N, M = 20, 10
        groups = _get_groups_skip(doc_ids, to_n_groups=5, m_docs_per_group=N)
        for group in groups:
            if len(group) < 2:
                continue
            group_scores, log = await _group_processing(query, group, content_map, comparer, N, M)
            num_calls += 1
            log["stage"] = 1
            call_logs.append(log)
            for doc_id, s in group_scores.items():
                score_dict[doc_id] += s
        ranked_list = _sort_docs_by_relevance(list(score_dict.keys()), list(score_dict.values()))
    else:
        ranked_list = doc_ids

    # Stage 2: 50 -> 20 (5 groups of 10, keep top 4 from each)
    if len(doc_ids) > 20:
        N, M = 10, 4
        stage2_docs = ranked_list[:50]
        groups = _get_groups_skip(stage2_docs, to_n_groups=5, m_docs_per_group=N)
        for group in groups:
            if len(group) < 2:
                continue
            group_scores, log = await _group_processing(query, group, content_map, comparer, N, M)
            num_calls += 1
            log["stage"] = 2
            call_logs.append(log)
            for doc_id, s in group_scores.items():
                score_dict[doc_id] += s
        ranked_list = _sort_docs_by_relevance(list(score_dict.keys()), list(score_dict.values()))
    else:
        ranked_list = doc_ids

    # Stage 3: 20 -> 10
    if len(doc_ids) > 10:
        N, M = 20, 10
        stage3_docs = ranked_list[:20]
        groups = _get_groups_skip(stage3_docs, to_n_groups=1, m_docs_per_group=N)
        for group in groups:
            if len(group) < 2:
                continue
            group_scores, log = await _group_processing(query, group, content_map, comparer, N, M)
            num_calls += 1
            log["stage"] = 3
            call_logs.append(log)
            for doc_id, s in group_scores.items():
                score_dict[doc_id] += s
        ranked_list = _sort_docs_by_relevance(list(score_dict.keys()), list(score_dict.values()))
    else:
        ranked_list = doc_ids

    # Stage 4: 10 -> 5
    if len(doc_ids) > 5:
        N, M = 10, 5
        stage4_docs = ranked_list[:10]
        groups = _get_groups_skip(stage4_docs, to_n_groups=1, m_docs_per_group=N)
        for group in groups:
            if len(group) < 2:
                continue
            group_scores, log = await _group_processing(query, group, content_map, comparer, N, M)
            num_calls += 1
            log["stage"] = 4
            call_logs.append(log)
            for doc_id, s in group_scores.items():
                score_dict[doc_id] += s
        ranked_list = _sort_docs_by_relevance(list(score_dict.keys()), list(score_dict.values()))
    else:
        ranked_list = doc_ids

    # Stage 5: 5 -> 2
    if len(doc_ids) > 2:
        N, M = 5, 2
        stage5_docs = ranked_list[:5]
        groups = _get_groups_skip(stage5_docs, to_n_groups=1, m_docs_per_group=N)
        for group in groups:
            if len(group) < 2:
                continue
            group_scores, log = await _group_processing(query, group, content_map, comparer, N, M)
            num_calls += 1
            log["stage"] = 5
            call_logs.append(log)
            for doc_id, s in group_scores.items():
                score_dict[doc_id] += s

    return score_dict, call_logs, num_calls
