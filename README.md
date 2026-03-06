<h1 align="center">BlitzRank</h1>

<p align="center"><b>Principled Zero-shot Ranking Agents with Tournament Graphs</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.05448"><img src="https://img.shields.io/badge/arXiv-2602.05448-b31b1b.svg" alt="arXiv"></a>
  <a href="https://blitzrank.ai"><img src="https://img.shields.io/badge/Website-blitzrank.ai-blue" alt="Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

BlitzRank uses tournament graphs to extract maximal information from each LLM call, a principled framework achieving Pareto optimality across 14 benchmarks and 5 LLMs with 25–40% fewer queries.

<p align="center">
  <img src="assets/images/blitzrank_demo.gif" alt="BlitzRank vs Sliding Window animation" width="720"/>
  <br>
  <em><strong>Algorithm visualization</strong> on the <a href="https://books.google.com/books?id=RosxmAYFFosC">25 horses puzzle</a>: find the 3 fastest horses from 25, racing 5 at a time.<br>BlitzRank converges in <strong>7 rounds</strong> vs Sliding Window's <strong>11 rounds</strong>.</em>
</p>

## Installation

```bash
uv pip install blitzrank
```

From source:

```bash
git clone https://github.com/ContextualAI/BlitzRank.git
cd BlitzRank
uv pip install -e .
```

Install with additional dependencies for baselines (AcuRank, TourRank):

```bash
uv pip install "blitzrank[all]"
# or from source:
uv pip install -e ".[all]"
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

# Any LiteLLM-compatible model works — just set the appropriate API keys as env variables
indices = rank(ranker, model="openai/gpt-4.1", query=query, docs=docs, topk=2)  # [1, 0]
top_docs = [docs[i] for i in indices]

# Get token usage statistics
indices, stats = rank(ranker, model="openai/gpt-4.1", query=query, docs=docs, topk=2, return_stats=True)
print(stats)  # {"num_llm_calls": 1, "input_tokens": 245, "output_tokens": 12, ...}
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
| **BRIGHT** | `bright/aops/infx`, `bright/biology/infx`, `bright/earth_science/infx`, `bright/economics/infx`, `bright/leetcode/infx`, `bright/pony/infx`, `bright/psychology/infx`, `bright/robotics/infx`, `bright/stackoverflow/infx`, `bright/sustainable_living/infx`, `bright/theoremqa_questions/infx`, `bright/theoremqa_theorems/infx` |

## Baselines

All methods share the same interface. Create a ranker (with optional parameters), pass the model to `rank`/`evaluate`.

```python
from blitzrank import BlitzRank, SlidingWindow, SetWise, PairWise, TourRank, AcuRank, rank

query = "capital of France"
docs = ["Berlin is in Germany", "Paris is in France", "Tokyo is in Japan"]

for Method in [BlitzRank, SlidingWindow, SetWise, PairWise, TourRank, AcuRank]:
    indices = rank(Method(), model="openai/gpt-4.1", query=query, docs=docs, topk=2)
```

Available methods: `BlitzRank`, `SlidingWindow`, `SetWise`, `PairWise`, `TourRank`, `AcuRank`

📖 [Full parameter reference →](docs/parameters.md)

## Token Usage Tracking

<details>
<summary>Get detailed token consumption statistics by setting <code>return_stats=True</code></summary>

```python
from blitzrank import BlitzRank, rank

ranker = BlitzRank()
indices, stats = rank(
    ranker, 
    model="openai/gpt-4.1", 
    query="capital of France", 
    docs=["Berlin is in Germany", "Paris is in France", "Tokyo is in Japan"],
    topk=2,
    return_stats=True
)

print(stats)
# {
#     "num_llm_calls": 1,      # Number of LLM API calls made
#     "input_tokens": 245,     # Total prompt tokens
#     "output_tokens": 12,     # Total completion tokens
#     "thought_tokens": 0,     # Reasoning tokens (for models that support it)
#     "total_tokens": 257,     # input_tokens + output_tokens
#     "latency_ms": 1523.4     # Total API latency in milliseconds
# }
```

This works with all ranker types and is useful for cost tracking and performance monitoring.

</details>

## Reproducing Paper Results

Run all methods across all 14 datasets and 5 LLMs from the paper (Table 3):

```python
from blitzrank import BlitzRank, SlidingWindow, SetWise, PairWise, TourRank, AcuRank, evaluate

# 6 TREC-DL + 8 BEIR = 14 benchmarks
DATASETS = [
    # TREC-DL
    "msmarco/dl19/bm25", "msmarco/dl20/bm25", "msmarco/dl21/bm25",
    "msmarco/dl22/bm25", "msmarco/dl23/bm25", "msmarco/dlhard/bm25",
    # BEIR
    "beir/trec-covid/bm25", "beir/nfcorpus/bm25", "beir/signal1m/bm25",
    "beir/trec-news/bm25", "beir/robust04/bm25", "beir/webis-touche2020/bm25",
    "beir/dbpedia-entity/bm25", "beir/scifact/bm25",
]
MODELS = [
    "openai/gpt-4.1",
    "vertex_ai/gemini-3-flash-preview",
    "openrouter/deepseek/deepseek-v3.2",
    "openrouter/qwen/qwen3-235b-a22b-2507",
    "openrouter/z-ai/glm-4.7",
]
RANKERS = {
    "Blitz-k20": BlitzRank(window_size=20),
    "Blitz-k10": BlitzRank(window_size=10),
    "SW": SlidingWindow(),
    "SW-R2": SlidingWindow(num_rounds=2),
    "Setwise": SetWise(),
    "Pairwise": PairWise(),
    "TourRank": TourRank(),
    "TourRank-R2": TourRank(num_rounds=2),
    "AcuRank": AcuRank(),
    "AcuRank-H": AcuRank(tol=1e-4),
}

for dataset in DATASETS:
    for model in MODELS:
        for name, ranker in RANKERS.items():
            rankings, metrics = evaluate(ranker, dataset=dataset, model=model)
            print(f"{name:>12} | {dataset:<28} | {model:<40} | nDCG@10={metrics['ndcg@10']:.3f}")
```

📖 [Custom datasets and methods →](docs/extending.md)

## Acknowledgements

This project builds upon the following open-source repositories: [RankGPT](https://github.com/sunnweiwei/RankGPT), [LLM-Rankers](https://github.com/ielab/llm-rankers), [AcuRank](https://github.com/soyoung97/AcuRank), [Pyserini](https://github.com/castorini/pyserini), and [LiteLLM](https://github.com/BerriAI/litellm).

## Citation

```bibtex
@article{blitzrank2026,
  title={BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs},
  author={Agrawal, Sheshansh and Nguyen, Thien Hang and Kiela, Douwe},
  journal={arXiv preprint arXiv:2602.05448},
  year={2026}
}
```
