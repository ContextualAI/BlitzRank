# Improving Non-Transitive Tournament Graph Tests

## Background

The `ToyNonTransitiveBucketOracle` in `toy_setup.py` models a bucketed preference structure
designed to exercise the SCC-detection and non-transitive handling paths of the algorithm:

- **Cross-bucket**: lower bucket always beats higher bucket (strictly transitive)
- **Within-bucket**: items whose within-indices share a residue class mod 3 form a directed
  3-cycle. Specifically, residue 1 beats residue 0, residue 0 beats residue 2, residue 2 beats
  residue 1. Items with the same residue are tie-broken by within-index (lower wins).

Example with 3 items in a single bucket (indices 0.0, 0.1, 0.2):

```
0.1 → 0.0 → 0.2 → 0.1   (directed 3-cycle)
```

### How the oracle constructs its output

Each oracle call receives a subset of items. Within each bucket it builds a
Hamiltonian path by insertion sort under `_bucket_beats`, then returns only the
**adjacent-pair edges** of that path (not all pairs).

- A call on {0.0, 0.1, 0.2} yields edges [(0.2, 0.1), (0.1, 0.0)] — a 2-edge chain, not a cycle.
- The cycle-closing edge (0.0→0.2) would only appear if 0.0 and 0.2 are scheduled in the
  same match **without** 0.1 present.

**Key implication**: the algorithm terminates as soon as `known_relationships >= n-1`, which
can be satisfied via a transitive chain (DAG) without ever closing the cycle. In practice,
sort-level tests should NOT assert `num_3_cycles >= 1` — cycles only appear in the graph if
the scheduling happens to produce the cycle-closing comparison.

To reliably test `num_3_cycles` and SCC formation, inject edges directly into
`TournamentGraph.process_round` rather than running the sort loop.

Because each directed edge only appears once (no parallel edges), `enforce_tournament=True`
(the default) remains valid even in the non-transitive case.

---

## What "correct" means in the non-transitive case

There is no total order, so we cannot assert exact output sequences. Instead we
assert on **bucket membership**:

1. All top-m results must come from the expected buckets in expected order.
   - If m ≤ |bucket 0|: all results from bucket 0.
   - If m > |bucket 0|: fill bucket 0 first, then bucket 1, etc.
2. Within a bucket, any ordering of the bucket's items is acceptable (since cycles
   make all items in the bucket "equivalent" once they form one SCC).

---

## Properties that tests should exercise

| Property                          | Why it matters                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------ |
| Termination                       | Cycles could in theory block the finalization criterion; the algorithm must still converge |
| Correct bucket ordering           | Cross-bucket transitivity must be preserved despite within-bucket cycles                   |
| SCC formation within a bucket     | Items that participate in a cycle should end up in the same SCC                            |
| `num_3_cycles > 0`                | The graph must actually detect cycles (not silently ignore them)                           |
| Scaling across bucket/item counts | The algorithm must work for heterogeneous bucket sizes and various (k, m) combos           |

---

## Test catalogue

### Helpers

```python
def _get_bucket(item: Item) -> int:
    return int(item.id.split(".")[0])

def _verify_results_from_buckets(result, expected_buckets: list[int]) -> None:
    result_buckets = [_get_bucket(item) for item in result.results]
    assert result_buckets == expected_buckets, (
        f"Expected buckets {expected_buckets}, got {result_buckets}"
    )
```

---

### 1. `test_single_3_cycle_terminates`

**Setup**: 3 items in 1 bucket `[0, 0, 0]`, k=3, m=3

**What it tests**: The simplest possible non-transitive case. The three items form
a directed 3-cycle. The algorithm must detect this, collapse them into one SCC, and
finalize all three.

**Assertions**:

- Returns exactly 3 results
- All results from bucket 0
- Final graph has `_count_3_cycles() >= 1`

---

### 2. `test_single_bucket_9_items_all_finalized`

**Setup**: 9 items in 1 bucket `[0]*9`, k=3, m=9

**What it tests**: A larger single-bucket case where multiple 3-cycles exist (three
residue-class triples: {0,3,6}, {1,4,7}, {2,5,8} each forming a sub-cycle, plus
cross-residue cycles). All items should eventually end up in the same SCC.

**Assertions**:

- Returns all 9 items
- All from bucket 0
- `_count_3_cycles() >= 1`

---

### 3. `test_two_buckets_top_m_equals_bucket_size`

**Setup**: `[0, 0, 0, 1, 1, 1]`, k=3, m=3

**What it tests**: With two equal-size buckets and m equal to the size of bucket 0,
the top-3 results must all be from bucket 0 — even though within bucket 0 there are
cycles. Cross-bucket transitivity must dominate.

**Assertions**:

- All 3 results from bucket 0

---

### 4. `test_two_buckets_top_1`

**Setup**: `[0, 0, 0, 1, 1, 1]`, k=3, m=1

**What it tests**: The tightest possible correctness check — a single top result must
always come from bucket 0.

**Assertions**:

- The single result is from bucket 0

---

### 5. `test_three_buckets_boundary_crossing`

**Setup**: `[0, 0, 1, 1, 1, 2, 2, 2]`, k=3, m=4

**What it tests**: m crosses the bucket-0 boundary. The top-4 should be the 2 items
from bucket 0 followed by 2 items from bucket 1.

**Assertions**:

- `result_buckets == [0, 0, 1, 1]`

---

### 6. `test_scc_forms_within_bucket`

**Setup**: `[0, 0, 0, 1, 1, 1, 2, 2, 2]`, k=3, m=9

**What it tests**: After a full sort, items within the same bucket must form a single
SCC in the final tournament graph. This directly validates the SCC-detection logic.

**How**: Inspect `result.graph.get_graph()`, run `nx.condensation()`, and verify each
bucket's items appear in the same condensation node.

**Assertions**:

- Each bucket's items belong to the same SCC
- `_count_3_cycles() >= 1` (at least one cycle was detected)

---

### 7. `test_correctness_parameterized`

**Setup**: Parametrize over `(bucket_labels, k, m, expected_bucket_seq)`:

| bucket_labels           | k   | m   | expected_bucket_seq |
| ----------------------- | --- | --- | ------------------- |
| `[0,0,1,1]`             | 3   | 2   | `[0,0]`             |
| `[0,0,0,1,1,1]`         | 3   | 6   | `[0,0,0,1,1,1]`     |
| `[0,1,1,2,2,2]`         | 3   | 3   | `[0,1,1]`           |
| `[0,0,1,1,1,2,2,2,3,3]` | 4   | 5   | `[0,0,1,1,1]`       |
| `[0,0,0,0,1,1,1,1]`     | 4   | 4   | `[0,0,0,0]`         |

**What it tests**: Robustness across heterogeneous bucket sizes and (k, m) values.

---

### 8. `test_stress_multi_bucket_random_seeds`

**Setup**: `[0,0,0,0, 1,1,1,1, 2,2,2,2]` (3 buckets × 4 items = 12 items), k=4, m=6
Parametrize over seeds 0–4.

**What it tests**: Different shuffle orders should always produce the same bucket-ordered
top results.

**Assertions**:

- `result_buckets == [0, 0, 0, 0, 1, 1]`

---

### 9. `test_large_non_transitive`

**Setup**: 5 buckets × 4 items = 20 items, k=4, m=8

**What it tests**: Larger-scale non-transitive sort. Verifies termination within a
generous `max_num_rounds` budget.

**Assertions**:

- Returns 8 results
- First 4 from bucket 0, next 4 from bucket 1

---

### 10. `test_single_3_cycle_graph_structure` (unit-level, no sort)

**Setup**: Directly build a `TournamentGraph` with 3 nodes and add the known 3-cycle edges.

**What it tests**: Unit-tests `TournamentGraph._count_3_cycles()` and SCC computation
in isolation, decoupled from the sort loop.

**Assertions**:

- `num_3_cycles == 1`
- `num_sccs == 1` (all three nodes in one SCC)
- `known_relationships` for each node == 2 (knows both others)
