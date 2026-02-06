# BlitzRank

**Principled Zero-shot Ranking Agents with Tournament Graphs**

[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2506.XXXXX)
[![Website](https://img.shields.io/badge/Website-blitzrank.github.io-blue)](https://blitzrank.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

BlitzRank uses tournament graphs to extract maximal information from each LLM call, a principled framework achieving Pareto optimality across 14 benchmarks and 5 LLMs with 25–40% fewer queries.

<p align="center">
  <img src="assets/images/tournament_graph.png" alt="Tournament Graph Framework" width="600"/>
</p>

## Installation

```bash
uv pip install blitzrank
```

## Quick Start

```python
from blitzrank import BlitzRank, rank

ranker = BlitzRank()

query = "capital of France"
docs = [
    "Berlin is the capital of Germany.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan.",
]

indices = rank(ranker, model="openai/gpt-4.1", query=query, docs=docs, topk=2)  # [1, 0]
top_docs = [docs[i] for i in indices]
```

Any [LiteLLM](https://github.com/BerriAI/litellm)-compatible model works — just change the `model` string:

```python
rank(ranker, model="openai/gpt-4.1", query=query, docs=docs, topk=2)
rank(ranker, model="vertex_ai/gemini-3-flash-preview", query=query, docs=docs, topk=2)
rank(ranker, model="openrouter/deepseek/deepseek-v3.2", query=query, docs=docs, topk=2)
rank(ranker, model="openrouter/qwen/qwen3-235b-a22b-2507", query=query, docs=docs, topk=2)
rank(ranker, model="openrouter/z-ai/glm-4.7", query=query, docs=docs, topk=2)
```

## Evaluate on a Benchmark

```python
from blitzrank import BlitzRank, evaluate

ranker = BlitzRank()
rankings, metrics = evaluate(ranker, dataset="msmarco/dl19/bm25", model="openai/gpt-4.1")

print(metrics)   # {"ndcg@10": 0.72, "map@10": 0.51}
print(rankings)  # [{"query": "...", "ranking": [3, 0, 7, ...]}, ...]
```

Dataset names follow the format `collection/split/retriever`.

| Category | Datasets |
|----------|----------|
| **MSMARCO** | `msmarco/dl19/bm25`, `msmarco/dl20/bm25`, `msmarco/dl21/bm25`, `msmarco/dl22/bm25`, `msmarco/dl23/bm25`, `msmarco/dlhard/bm25` |
| **BEIR** | `beir/nfcorpus/bm25`, `beir/fiqa/bm25`, `beir/trec-covid/bm25`, `beir/nq/bm25`, `beir/hotpotqa/bm25`, `beir/scifact/bm25`, `beir/arguana/bm25`, `beir/quora/bm25`, `beir/scidocs/bm25`, `beir/fever/bm25`, `beir/climate-fever/bm25`, `beir/dbpedia-entity/bm25`, `beir/robust04/bm25`, `beir/signal1m/bm25`, `beir/trec-news/bm25`, `beir/webis-touche2020/bm25` |
| **BRIGHT** | `bright/aops/infx`, `bright/biology/infx`, `bright/leetcode/infx`, `bright/stackoverflow/infx`, ... |

## Other Methods

All methods share the same interface. Create a ranker (with optional parameters), pass the model to `rank`/`evaluate`.

```python
from blitzrank import BlitzRank, SlidingWindow, SetWise, PairWise, TourRank, AcuRank, rank

query = "capital of France"
docs = ["Berlin is in Germany", "Paris is in France", "Tokyo is in Japan"]

for Method in [BlitzRank, SlidingWindow, SetWise, PairWise, TourRank, AcuRank]:
    indices = rank(Method(), model="openai/gpt-4.1", query=query, docs=docs, topk=2)
```

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `BlitzRank` | Tournament graphs with transitive closure (ours) | `window_size`, `top_m` |
| `SlidingWindow` | RankGPT-style sliding window | `window_size`, `step`, `num_rounds` |
| `SetWise` | Pick-the-winner with heapsort/bubblesort | `num_child`, `sorting_method` |
| `PairWise` | Pairwise comparisons with heapsort/bubblesort | `sorting_method` |
| `TourRank` | Multi-round tournament filtering | `num_rounds`, `window_size` |
| `AcuRank` | Adaptive uncertainty-based ranking | `tol`, `hard_constraint` |

## Reproducing Paper Results

```python
from blitzrank import BlitzRank, SlidingWindow, SetWise, PairWise, evaluate

datasets = ["msmarco/dl19/bm25", "msmarco/dl20/bm25", "beir/nfcorpus/bm25", "beir/trec-covid/bm25", "beir/fiqa/bm25"]
models = ["openai/gpt-4.1", "vertex_ai/gemini-3-flash-preview",
          "openrouter/deepseek/deepseek-v3.2", "openrouter/qwen/qwen3-235b-a22b-2507",
          "openrouter/z-ai/glm-4.7"]
rankers = {
    "blitzrank": BlitzRank(),
    "sliding_window": SlidingWindow(),
    "setwise": SetWise(),
    "pairwise": PairWise(),
}

for dataset in datasets:
    for model in models:
        for name, ranker in rankers.items():
            rankings, metrics = evaluate(ranker, dataset=dataset, model=model)
            print(f"{dataset}/{model}/{name}: {metrics}")
```

## Custom Dataset

Pass any dataset as a list of dicts with optional relevance judgments:

```python
from blitzrank import BlitzRank, evaluate

dataset = [
    {
        "query": "capital of France",
        "docs": {"d1": "Paris is...", "d2": "Berlin is..."},
        "qrels": {"d1": 1, "d2": 0},  # optional, needed for metrics
    },
    {
        "query": "largest ocean",
        "docs": {"d3": "The Pacific...", "d4": "The Atlantic..."},
        "qrels": {"d3": 1},
    },
]

rankings, metrics = evaluate(BlitzRank(), dataset=dataset, model="openai/gpt-4.1")
print(metrics)   # {"ndcg@10": ..., "map@10": ...}
print(rankings)  # per-query rankings
```

## Custom Method

Implement `__call__` returning indices sorted by relevance:

```python
from blitzrank import Ranker, evaluate

class MyRanker(Ranker):
    def __call__(self, query: str, docs: list[str], topk: int, model: str) -> list[int]:
        """Return topk indices from docs, most relevant first."""
        # your ranking logic
        return sorted(range(len(docs)), key=lambda i: score(query, docs[i]), reverse=True)[:topk]

rankings, metrics = evaluate(MyRanker(), dataset="msmarco/dl19/bm25", model="openai/gpt-4.1")
```

## Acknowledgements

- [RankGPT](https://github.com/sunnweiwei/RankGPT) — Sliding window reranking
- [SetWise](https://github.com/ielab/llm-rankers) — SetWise and PairWise rankers
- [AcuRank](https://github.com/soyoung97/AcuRank) — Adaptive uncertainty ranking
- [Pyserini](https://github.com/castorini/pyserini) — BM25 retrieval
- [LiteLLM](https://github.com/BerriAI/litellm) — Unified LLM API

## Citation

```bibtex
@article{blitzrank2026,
  title={BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs},
  author={Agrawal, Sheshansh and Nguyen, Thien Hang and Kiela, Douwe},
  journal={arXiv preprint arXiv:2602.05448},
  year={2026}
}
```

## License

MIT
