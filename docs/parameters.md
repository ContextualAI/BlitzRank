# Ranker Parameters

All rankers share the same interface:

```python
from blitzrank import rank, evaluate

ranker = Method(**params)
indices = rank(ranker, model="openai/gpt-4.1", query=query, docs=docs, topk=10)
rankings, metrics = evaluate(ranker, dataset="msmarco/dl19/bm25", model="openai/gpt-4.1")
```

---

## BlitzRank

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `20` | Maximum number of documents per LLM call |
| `top_m` | `int` | `10` | Number of top documents to identify |

---

## SlidingWindow

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `20` | Maximum number of documents per LLM call |
| `step` | `int` | `10` | Step size between consecutive windows |
| `num_rounds` | `int` | `1` | Number of full passes over the document list |
| `rank_end` | `int` | `100` | Rank position at which to stop reranking |

---

## SetWise

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sorting_method` | `str` | `"heapsort"` | Sorting algorithm: `"heapsort"` or `"bubblesort"` |
| `num_child` | `int` | `2` | Number of child nodes in the sorting heap; each LLM call compares `num_child + 1` documents |
| `top_m` | `int` | `10` | Number of top documents to identify |

---

## PairWise

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sorting_method` | `str` | `"heapsort"` | Sorting algorithm: `"heapsort"`, `"bubblesort"`, or `"allpair"` |
| `top_m` | `int` | `10` | Number of top documents to identify |

---

## TourRank

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | `int` | `1` | Number of parallel tournament rounds |
| `window_size` | `int` | `20` | Maximum number of documents per LLM call |

---

## AcuRank

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `20` | Maximum number of documents per LLM call (reranker capacity) |
| `tol` | `float` | `1e-2` | Uncertainty threshold ε; documents with rank probability in (ε, 1−ε) are considered uncertain |
| `hard_constraint` | `int` | `100` | Maximum number of reranker calls (budget) |
| `uncertain_U` | `int` | `10` | Stopping threshold τ; terminate when fewer than this many uncertain documents remain |
| `R` | `int` | `10` | Number of top-k positions to rank |
| `break_mode` | `str` | `"reduce_uncertain"` | Stopping strategy |
