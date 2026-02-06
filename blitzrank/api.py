import asyncio
from datetime import datetime

from .engine.config import DatasetConfig, LoggingConfig, EvaluationConfig
from .engine.dataset import load_dataset
from .engine.evaluator import evaluate_results
from .engine.logger import ExperimentLogger
from .engine.reranker import build_reranked_item_from_final_indices


def rank(ranker, model, query, docs, topk=10):
    return asyncio.run(ranker(query, docs, topk, model))


def evaluate(ranker, dataset, model):
    return asyncio.run(_evaluate_async(ranker, dataset, model))


def _parse_dataset_string(s):
    parts = s.split("/")
    collection, split = parts[0], parts[1]
    if collection == "msmarco":
        name = "dl_hard-passage" if split == "dlhard" else f"{split}-passage"
        return DatasetConfig(type="online_bm25", name=name, index="msmarco-v1-passage")
    if collection == "beir":
        return DatasetConfig(type="online_bm25", name=f"beir-v1.0.0-{split}-test", index=f"beir-v1.0.0-{split}.flat")
    if collection == "bright":
        return DatasetConfig(type="bright", name=split)
    raise ValueError(f"Unknown dataset collection: {collection}")


def _convert_custom_dataset(items):
    results, qrels = [], {}
    for i, item in enumerate(items):
        qid = str(i)
        hits = [
            {"content": content, "qid": qid, "docid": docid, "rank": r, "score": len(item["docs"]) - r + 1}
            for r, (docid, content) in enumerate(item["docs"].items(), 1)
        ]
        results.append({"query": item["query"], "hits": hits})
        if "qrels" in item:
            qrels[qid] = {str(k): int(v) for k, v in item["qrels"].items()}
    return {"results": results, "qrels": qrels}


def _get_rerank_fn(selector_type):
    if selector_type == "tournament_graph":
        from .engine.algorithms.tournament_graph.experimental_interface import tournament_graph_rerank_dataset_from_config
        return tournament_graph_rerank_dataset_from_config
    if selector_type == "sliding_window":
        from .engine.reranker import rerank_results
        return rerank_results
    if selector_type == "setwise":
        from .engine.algorithms.setwise_pairwise_selectors import setwise_rerank_dataset_from_config
        return setwise_rerank_dataset_from_config
    if selector_type == "pairwise":
        from .engine.algorithms.setwise_pairwise_selectors import pairwise_rerank_dataset_from_config
        return pairwise_rerank_dataset_from_config
    if selector_type == "tourrank":
        from .engine.algorithms.acurank_selectors import tourrank_rerank_dataset_from_config
        return tourrank_rerank_dataset_from_config
    if selector_type == "acurank":
        from .engine.algorithms.acurank_selectors import acurank_rerank_dataset_from_config
        return acurank_rerank_dataset_from_config
    raise ValueError(f"Unknown selector type: {selector_type}")


async def _evaluate_async(ranker, dataset, model):
    dataset_config = _parse_dataset_string(dataset) if isinstance(dataset, str) else DatasetConfig(type="custom", name="custom")
    has_config = hasattr(ranker, "_build_reranking_config")
    reranking_config = ranker._build_reranking_config(model) if has_config else None

    logging_config = LoggingConfig(
        output_dir="blitzrank_outputs",
        experiment_name=f"blitzrank_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        enable_wandb=False,
    )
    config_dict = {
        "dataset": {"type": dataset_config.type, "name": dataset_config.name},
        "evaluation": {"type": "trec_eval", "metrics": ["ndcg_cut_10", "map"]},
        "logging": {"output_dir": logging_config.output_dir, "experiment_name": logging_config.experiment_name, "enable_wandb": False},
    }
    logger = ExperimentLogger(logging_config, dataset_config, reranking_config, config_dict)

    try:
        dataset_raw = load_dataset(dataset_config, logger) if isinstance(dataset, str) else _convert_custom_dataset(dataset)

        if has_config:
            rerank_fn = _get_rerank_fn(reranking_config.selector.type)
            reranked = await rerank_fn(dataset_raw, reranking_config, logger, logger.get_output_dir())
        else:
            reranked = await _evaluate_per_query(ranker, dataset_raw, model)

        eval_config = EvaluationConfig(type="trec_eval", metrics=["ndcg_cut_10", "map"])
        metrics = evaluate_results(reranked, eval_config, logger.get_output_dir(), logger)

        rankings = [
            {"query": item["query"], "ranking": [hit["docid"] for hit in item["hits"]]}
            for item in reranked["results"]
        ]
        return rankings, metrics
    finally:
        logger.close()


async def _evaluate_per_query(ranker, dataset_raw, model):
    reranked_results = []
    for item in dataset_raw["results"]:
        docs = [hit["content"] for hit in item["hits"]]
        indices = await ranker(item["query"], docs, len(docs), model)
        reranked_results.append(build_reranked_item_from_final_indices(item["query"], item["hits"], indices))
    return {"results": reranked_results, "qrels": dataset_raw["qrels"]}
