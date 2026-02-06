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

Tournament graphs with transitive closure (ours).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `20` | Number of documents per LLM comparison call |
| `top_m` | `int` | `10` | Number of top documents to identify |

---

## SlidingWindow

RankGPT-style sliding window reranking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `20` | Number of documents per window |
| `step` | `int` | `10` | Step size between consecutive windows |
| `num_rounds` | `int` | `1` | Number of full passes over the document list |
| `rank_end` | `int` | `100` | Rank position at which to stop reranking |

---

## SetWise

Pick-the-winner comparisons with sorting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sorting_method` | `str` | `"heapsort"` | Sorting algorithm: `"heapsort"` or `"bubblesort"` |
| `num_child` | `int` | `2` | Number of candidates per comparison |
| `top_m` | `int` | `10` | Number of top documents to identify |

---

## PairWise

Pairwise comparisons with sorting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sorting_method` | `str` | `"heapsort"` | Sorting algorithm: `"heapsort"`, `"bubblesort"`, or `"allpair"` |
| `top_m` | `int` | `10` | Number of top documents to identify |

---

## TourRank

Multi-round tournament filtering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | `int` | `1` | Number of tournament rounds |
| `window_size` | `int` | `20` | Number of documents per comparison window |

---

## AcuRank

Adaptive uncertainty-based ranking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | `20` | Number of documents per comparison window |
| `tol` | `float` | `1e-2` | Convergence tolerance for rank stability |
| `hard_constraint` | `int` | `100` | Maximum number of LLM calls allowed |
| `uncertain_U` | `int` | `10` | Number of uncertain documents to re-examine per iteration |
| `R` | `int` | `10` | Number of top ranks to stabilize |
| `break_mode` | `str` | `"reduce_uncertain"` | Early-stopping strategy: `"reduce_uncertain"` |
