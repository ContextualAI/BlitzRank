from typing import Dict, Any

import hashlib
import numpy as np
import requests
import datasets
from tqdm import tqdm
import json
from .config import DatasetConfig
from pathlib import Path
from datetime import datetime
import orjson
import os

BRIGHT_TASKS_SHORT = [
    "aops",
    "biology",
    "earth_science",
    "economics",
    "leetcode",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "theoremqa_questions",
    "theoremqa_theorems",
]
BRIGHT_TASKS_LONG = [
    "biology",
    "earth_science",
    "economics",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
]


def _get_cache_path(config: DatasetConfig) -> Path:
    cache_base = os.getenv("DATASET_CACHE_DIR", ".cache")
    cache_dir = Path(cache_base) / "datasets" / config.name
    cache_file = f"{config.index}_bm25flat_k{config.k}.json"
    return cache_dir / cache_file


def _load_from_cache(config: DatasetConfig, logger) -> Dict[str, Any] | None:
    cache_path = _get_cache_path(config)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cached = orjson.loads(f.read())

        metadata = cached["metadata"]
        if (
            metadata["type"] == config.type
            and metadata["name"] == config.name
            and metadata["index"] == config.index
            and metadata["k"] == config.k
        ):
            logger.info(f"Loaded dataset from cache: {cache_path}")
            return cached["data"]
        return None
    except (OSError, KeyError, orjson.JSONDecodeError):
        return None


def _save_to_cache(config: DatasetConfig, data: Dict[str, Any], logger):
    cache_path = _get_cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "metadata": {
                        "type": config.type,
                        "name": config.name,
                        "index": config.index,
                        "k": config.k,
                        "cached_at": datetime.now().isoformat(),
                        "version": "1.0",
                    },
                    "data": data,
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
            )
        )
    tmp_path.replace(cache_path)
    logger.info(f"Cached dataset to: {cache_path}")


def _load_bright_dataset(config: DatasetConfig, logger) -> Dict[str, Any]:
    task, long_context, rewrite_queries = (
        config.name,
        config.long_context,
        config.rewrite_queries,
    )
    valid_tasks = BRIGHT_TASKS_LONG if long_context else BRIGHT_TASKS_SHORT
    if task not in valid_tasks:
        raise ValueError(
            f"Task '{task}' not valid for long_context={long_context}. Valid: {valid_tasks}"
        )

    cache_base = os.getenv("DATASET_CACHE_DIR", ".cache")
    cache_dir = Path(cache_base) / "datasets" / "bright"
    cache_file = (
        cache_dir
        / f"{task}_long{long_context}_rewrite{rewrite_queries}_k{config.k}.json"
    )

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = orjson.loads(f.read())
        logger.info(f"Loaded BRIGHT dataset from cache: {cache_file}")
        return data

    logger.info(
        f"Loading BRIGHT dataset: {task} (long_context={long_context}, rewrite_queries={rewrite_queries})"
    )

    prefix = "INF-X-Retriever" if rewrite_queries else "inf-retriever-v1-pro"
    suffix = "_long" if long_context else ""
    scores_url = f"https://raw.githubusercontent.com/yaoyichen/INF-X-Retriever/refs/heads/main/output/{prefix}{suffix}/{task}_inf_long_{long_context}/score.json"
    logger.info(f"Loading scores from: {scores_url}")
    scores = requests.get(scores_url).json()

    if rewrite_queries:
        queries = requests.get(
            f"https://raw.githubusercontent.com/yaoyichen/INF-X-Retriever/3b47f4e70dd3a764a2999b84a0e0a6f19f7d28dd/rewrite_data/{task}_queries.json"
        ).json()
        logger.info(
            f"Loading queries from: {f'https://raw.githubusercontent.com/yaoyichen/INF-X-Retriever/3b47f4e70dd3a764a2999b84a0e0a6f19f7d28dd/rewrite_data/{task}_queries.json'}"
        )
    else:
        queries = list(datasets.load_dataset("xlangai/BRIGHT", "examples")[task])
        logger.info("Loading queries from HF hub: xlangai/BRIGHT:examples")
    docs_split = "long_documents" if long_context else "documents"
    docs_list = list(datasets.load_dataset("xlangai/BRIGHT", docs_split)[task])
    logger.info(f"Loading docs from HF hub: xlangai/BRIGHT:{docs_split}")
    docs_by_id = {d["id"]: d["content"] for d in docs_list}

    results, qrels = [], {}
    gold_key = "gold_ids_long" if long_context else "gold_ids"
    for q in tqdm(queries, desc="Building results"):
        qid = q["id"]
        query_scores = scores.get(qid, {})
        excluded = set(q.get("excluded_ids", []))
        scored_docs = [
            (did, sc)
            for did, sc in query_scores.items()
            if did not in excluded and did in docs_by_id
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        scored_docs = scored_docs[: config.k]

        hits = []
        for rank, (docid, score) in enumerate(scored_docs, 1):
            hits.append(
                {
                    "content": docs_by_id[docid],
                    "qid": qid,
                    "docid": docid,
                    "rank": rank,
                    "score": score,
                }
            )
        results.append({"query": q["query"], "hits": hits})

        qrels[qid] = {gid: 1 for gid in q.get(gold_key, []) if gid in docs_by_id}

    data = {"results": results, "qrels": qrels}

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(
            orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS)
        )
    tmp.replace(cache_file)
    logger.info(f"Cached BRIGHT dataset to: {cache_file}")

    return data


def _load_dlhard_results_qrels():
    url = "https://raw.githubusercontent.com/soyoung97/AcuRank/refs/heads/main/data/bm25/dl-hard.jsonl"
    response = requests.get(url)
    results = []
    qrels = {}
    for line in response.text.strip().split("\n"):
        item = json.loads(line)
        qid = item["qid"]
        hits = []
        for rank, doc in enumerate(item["bm25_results"], 1):
            content = doc["text"]
            if doc.get("title"):
                content = "Title: " + doc["title"] + " Content: " + content
            content = " ".join(content.split())
            hits.append({
                "content": content,
                "qid": qid,
                "docid": doc["pid"],
                "rank": rank,
                "score": doc["bm25_score"],
            })
        results.append({"query": item["q_text"], "hits": hits})
        qrels[qid] = {str(did): rel for did, rel in item["qrels"].items()}
    return results, qrels


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({"query": topics, "hits": []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if "title" in content:
                content = (
                    "Title: " + content["title"] + " " + "Content: " + content["text"]
                )
            else:
                content = content["contents"]
            content = " ".join(content.split())
            ranks[-1]["hits"].append(
                {
                    "content": content,
                    "qid": qid,
                    "docid": hit.docid,
                    "rank": rank,
                    "score": hit.score,
                }
            )
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]["title"]
            ranks.append({"query": query, "hits": []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if "title" in content:
                    content = (
                        "Title: "
                        + content["title"]
                        + " "
                        + "Content: "
                        + content["text"]
                    )
                elif "contents" in content:
                    content = content["contents"]
                elif "passage" in content:
                    content = content["passage"]
                else:
                    raise ValueError(f"Unknown content format: {content.keys()}")
                content = " ".join(content.split())
                ranks[-1]["hits"].append(
                    {
                        "content": content,
                        "qid": qid,
                        "docid": hit.docid,
                        "rank": rank,
                        "score": hit.score,
                    }
                )
    return ranks


"""
dataset = {'results': RESULTS, 'qrels': QRELS}
RESULTS = [{'query'(str) : query, 'hits': [HIT]}]
HIT = {'content'(str): content, 'qid'(int): qid, 'docid'(str): docid,
       'rank'(int, 1..N): rank, 'score'(float): score}
QRELS = {qid(int): {docid(int): rel(str)}}
"""


def perturb_dataset(data: Dict[str, Any], perturb_dataset: str) -> Dict[str, Any]:
    for result in data["results"]:
        hits = result["hits"]
        if perturb_dataset == "invert":
            for hit in hits:
                hit["score"] = -hit["score"]
            hits.sort(key=lambda h: h["score"], reverse=True)
            for i, hit in enumerate(hits):
                hit["rank"] = i + 1
        elif perturb_dataset == "random":
            scores = np.random.uniform(0, 1, len(hits))
            for hit, score in zip(hits, scores):
                hit["score"] = float(score)
            hits.sort(key=lambda h: h["score"], reverse=True)
            for i, hit in enumerate(hits):
                hit["rank"] = i + 1
    return data


def load_dataset(config: DatasetConfig, logger) -> Dict[str, Any]:
    def get_shuffled_cache_path():
        cache_base = os.getenv("DATASET_CACHE_DIR", ".cache")
        cache_dir = Path(cache_base) / "datasets" / config.name
        config_dict = {
            k: v for k, v in sorted(vars(config).items()) if k != "shuffle_seed"
        }
        config_hash = hashlib.md5(str(config_dict).encode()).hexdigest()[:8]
        return cache_dir / f"{config_hash}_shuffle{config.shuffle_seed}.json"

    def shuffle_dataset(data):
        if config.shuffle_seed is None:
            return data
        for result in data["results"]:
            hits = result["hits"]
            hits.sort(key=lambda h: hash((config.shuffle_seed, h["docid"])))
            for i, hit in enumerate(hits):
                hit["rank"] = i + 1
        return data

    def save_shuffled_cache(data, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(
                orjson.dumps(
                    {"metadata": {"shuffle_seed": config.shuffle_seed}, "data": data},
                    option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
                )
            )
        tmp.replace(path)
        logger.info(f"Cached shuffled dataset to: {path}")

    def apply_subset(data):
        if config.subset_size is None or config.subset_size >= len(data["results"]):
            return data
        rng = np.random.RandomState(config.subset_seed)
        sampled_indices = sorted(
            rng.choice(len(data["results"]), config.subset_size, replace=False)
        )
        data["results"] = [data["results"][i] for i in sampled_indices]
        sampled_qids = {str(r["hits"][0]["qid"]) for r in data["results"]}
        data["qrels"] = {qid: data["qrels"][qid] for qid in sampled_qids}
        logger.info(
            f"Sampled {config.subset_size} queries with seed {config.subset_seed}"
        )
        return data

    shuffled_path = get_shuffled_cache_path() if config.shuffle_seed else None

    if shuffled_path and shuffled_path.exists():
        try:
            with open(shuffled_path, "rb") as f:
                data = orjson.loads(f.read())["data"]
            data["qrels"] = {str(qid): docs for qid, docs in data["qrels"].items()}
            logger.info(f"Loaded shuffled dataset from cache: {shuffled_path}")
            return apply_subset(data)
        except (OSError, KeyError, orjson.JSONDecodeError):
            pass

    if config.type == "bright":
        data = _load_bright_dataset(config, logger)
    elif (cached := _load_from_cache(config, logger)) is not None:
        data = cached
    elif config.type == "online_bm25":
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.search import get_topics, get_qrels

        logger.info(f"Loading dataset: {config.name} with index: {config.index}")
        searcher = LuceneSearcher.from_prebuilt_index(config.index)
        if config.name == "dl_hard-passage":
            results, qrels = _load_dlhard_results_qrels()
        else:
            if config.name in ["dl21-passage", "dl22-passage", "dl23-passage"]:
                topics = get_topics(config.name.split("-")[0])
            else:
                topics = get_topics(config.name)
            qrels = get_qrels(config.name)
            results = run_retriever(topics, searcher, qrels, k=config.k)
        logger.info(f"Loaded {len(results)} queries")

        data = {"results": results, "qrels": qrels}
        _save_to_cache(config, data, logger)
    else:
        raise ValueError(f"Unknown dataset type: {config.type}")

    data = shuffle_dataset(data)
    if shuffled_path:
        save_shuffled_cache(data, shuffled_path)

    data["qrels"] = {str(qid): docs for qid, docs in data["qrels"].items()}
    data = apply_subset(data)

    if config.perturb_dataset:
        data = perturb_dataset(data, config.perturb_dataset)

    return data
