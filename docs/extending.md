# Extending BlitzRank

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
