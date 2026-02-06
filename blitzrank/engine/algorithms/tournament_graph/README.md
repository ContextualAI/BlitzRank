# Tournament Graph Sorting Algorithm

This document provides a formal introduction to the tournament graph sorting algorithm, which finds the top-$m$ items from a set of $n$ items using a $k$-wise comparison oracle.

## 1. Preliminaries: Notation and Background

### 1.1 Problem Setup

We consider an oracle-based comparison model:

- **Universe**: $V$ of $n$ items with an unknown underlying tournament $T^* = (V, E^*)$
- **Oracle**: $\mathcal{O}(S)$ compares $|S| \leq k$ items and returns edges consistent with $T^*$
- **Goal**: Find the top-$m$ items (those that lose to the fewest others) using minimal oracle calls

**Notation:**

| Symbol           | Definition                                         |
| ---------------- | -------------------------------------------------- |
| $n = \|V\|$      | Total number of items                              |
| $n \ge k \ge 2$  | Oracle capacity (max items per query)              |
| $n \ge m \ge 1$  | Number of top items to output                      |
| $T^* = (V, E^*)$ | Underlying true tournament (unknown)               |
| $G = (V, E)$     | Observed subgraph (edges from oracle calls so far) |

### 1.2 Tournament Graph Definitions

- **Tournament**: A directed graph where for every $u \neq v$, exactly one of $(u,v)$ or $(v,u) \in E$
- **SCC (Strongly Connected Component)**: A maximal vertex set where every vertex can reach every other vertex
- **Transitive Closure**: For a DAG, the reachability relation (computed via BFS/DFS from each node)

### 1.3 Equivalence Relation and Condensation Graph

Define the **mutual reachability** relation $\sim$ on $V$:

$$u \sim v \iff u \rightsquigarrow v \text{ and } v \rightsquigarrow u$$

where $\rightsquigarrow$ denotes reachability (existence of a directed path).

**Claim:** $\sim$ is an equivalence relation.

- *Reflexivity*: $u \rightsquigarrow u$ (trivial path of length 0)
- *Symmetry*: By definition of $\sim$
- *Transitivity*: If $u \sim v$ and $v \sim w$, then $u \rightsquigarrow v \rightsquigarrow w$ and $w \rightsquigarrow v \rightsquigarrow u$, so $u \sim w$

The equivalence classes $[u] = \{v \in V : u \sim v\}$ are precisely the **strongly connected components** (SCCs). We denote the SCC containing $u$ as $B(u)$.

**Condensation Graph (Quotient Graph):** The condensation $G^c = (V^c, E^c)$ is the quotient graph $G/{\sim}$:

- **Vertices**: $V^c = V/{\sim} = \{B(u) : u \in V\}$ (one vertex per equivalence class)
- **Edges**: $(B(u), B(v)) \in E^c \iff \exists\, u' \in B(u), v' \in B(v)$ such that $(u', v') \in E$ and $B(u) \neq B(v)$

**Key Properties:**

1. $G^c$ is a **DAG**: If $G^c$ had a cycle $B_1 \to B_2 \to \cdots \to B_1$, all nodes in these SCCs would be mutually reachable, contradicting maximality of SCCs.
2. **Order preservation**: $B(u) \rightsquigarrow B(v)$ in $G^c$ implies every node in $B(u)$ can reach every node in $B(v)$ in $G$.
3. **Information compression**: Nodes within an SCC are "equivalent" for ranking purposes—they form cycles and cannot be strictly ordered relative to each other.

The condensation collapses cyclic dependencies into single super-nodes, leaving a DAG that captures the partial order among equivalence classes.

### 1.4 Reach Metrics

For node $u$ with SCC membership $B(u)$:

$$\verb|out_reach|(u) = |\{v : B(u) \rightsquigarrow B(v) \text{ in } G^c,\, B(u) \neq B(v)\}|$$

$$\verb|in_reach|(u) = |\{v : B(v) \rightsquigarrow B(u) \text{ in } G^c,\, B(u) \neq B(v)\}|$$

$$\verb|out_reach_inclusive|(u) = \verb|out_reach|(u) + |B(u)| - 1$$

$$\verb|in_reach_inclusive|(u) = \verb|in_reach|(u) + |B(u)| - 1$$

where $\rightsquigarrow$ denotes reachability in the condensation DAG.

### 1.5 Finalization Criterion

A node $u$ is **finalized** iff:

$$\verb|in_reach_inclusive|(u) + \verb|out_reach_inclusive|(u) \geq n - 1$$

**Intuition (Information Sufficiency):** A node is finalized when we have gathered enough edge information to determine its position relative to all other $n-1$ nodes. In a complete tournament, every node has exactly $n-1$ neighbors (either predecessors or successors). The finalization criterion checks whether we can account for all $n-1$ relationships:

- Directly observed edges, or
- Transitively inferred via the condensation DAG (if $u$ reaches $v$ through a path in $G^c$, then $u$ beats $v$)

The sum $\verb|in_reach_inclusive| + \verb|out_reach_inclusive|$ counts the total number of nodes whose relationship to $u$ is determined. When this reaches $n-1$, no further oracle queries involving $u$ are needed.

### 1.6 Transitive Case

When oracle responses are globally transitive (no 3-cycles in $T^*$):

- All SCCs are singletons: $B(u) = \{u\}$ for all $u$
- $G^c \cong G$ (condensation is isomorphic to observed graph)
- Reach metrics reduce to simple reachability counts in $G$
- The finalization criterion simplifies: node $u$ is finalized when $\verb|in_degree|(u) + \verb|out_degree|(u) = n-1$ in the transitive closure

This is the simpler case and serves as the basis for correctness proofs.

### 1.7 Non-Transitive Case

When oracle responses contain cycles (e.g., $a \to b \to c \to a$):

- SCCs can have $|B(u)| > 1$ (multiple nodes in the same equivalence class)
- Inclusive reach metrics account for SCC members
- Same finalization criterion applies, but interpretation changes: nodes within an SCC are "tied" and finalize together

### 1.8 Scheduling Strategy

**Goal:** Select the next $k$ nodes for oracle query to maximize progress toward finalizing the top-$m$ nodes.

**Strategy:** Sort all nodes by $(\verb|in_reach_inclusive|(u), \verb|out_reach_inclusive|(u))$ ascending, then select one representative from each of the top $k$ non-finalized SCCs.

**Why SCC-based scheduling guarantees progress:**

- Within an SCC, all pairwise edges already exist (by definition of mutual reachability)
- Between different SCCs, at least some edges are missing (otherwise they would merge)
- Querying representatives from $k$ different SCCs adds new cross-SCC edges
- New edges either establish ordering in the condensation DAG or merge SCCs

**Intuition (Greedy Information Gain):**

- Nodes with low $\verb|in_reach_inclusive|$ are candidate top nodes (few known predecessors $\Rightarrow$ likely high-ranked)
- Among those, nodes with low $\verb|out_reach_inclusive|$ have the least determined relationships (need more edges)
- By querying one representative per SCC, we:
  1. Gather edges among potential top candidates (resolving their relative order)
  2. Avoid redundant queries within the same SCC
  3. Leverage transitive closure: a single edge $(u,v)$ may transitively determine many relationships via the condensation DAG

If fewer than $k$ non-finalized SCCs exist, we use all available representatives.

### 1.9 Main Algorithm

```
Algorithm: TournamentGraphSort(V, O, k, m)
Input:  V (items), O (oracle), k (capacity), m (top count)
Output: Top-m items sorted by ascending loss count

1. G ← (V, ∅)           // Initialize observed graph with no edges
2. S ← V[0:k]           // Initial match set (first k items)
3. F ← []               // Finalized nodes

4. while |F| < m:
     // Query oracle and update graph
     E_new ← O(S)
     G ← G ∪ E_new
     
     // Compute condensation and reach metrics
     G^c ← Condense(G)
     for u ∈ V: compute reach_inclusive(u)
     
     // Sort by (in_reach_inclusive, out_reach_inclusive) ascending
     sorted_V ← Sort(V, key=(in_reach_inclusive, out_reach_inclusive))
     
     // Finalize eligible nodes, schedule next match (one rep per SCC)
     F ← [u ∈ sorted_V : finalized(u)]
     S ← first k non-finalized SCCs from sorted_V, one representative each
     
5. return F[0:m]
```

---

## 2. Implementation

### 2.1 Core Classes

| Formal Concept                   | Implementation                                    |
| -------------------------------- | ------------------------------------------------- |
| Graph $G$ and condensation $G^c$ | [`TournamentGraph`](tournament_graph.py)          |
| Oracle $\mathcal{O}$             | [`CompareOracle`](compare_oracle.py)              |
| Main algorithm                   | [`TournamentGraphSort`](tournament_graph_sort.py) |

### 2.2 Key Methods Mapping

| Formal Concept                | Code Location                                                                             |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| Oracle query $\mathcal{O}(S)$ | [`CompareOracle.compare_k()`](compare_oracle.py#L45-L47)                                  |
| Graph update $G \cup E$       | [`TournamentGraph.process_round()`](tournament_graph.py#L95-L174)                         |
| Condensation $G^c$            | Uses `nx.condensation()` in `process_round()`                                             |
| Reach metrics                 | [`RoundOutput.{in,out}_reach_inclusive`](tournament_graph.py#L33-L34)                     |
| Finalization criterion        | [`node_satisfies_finalization_criterion()`](tournament_graph_sort.py#L242-L250)           |
| Scheduling                    | [`get_tournament_progress_and_schedule_next_match()`](tournament_graph_sort.py#L252-L281) |
| Main loop                     | [`sort()`](tournament_graph_sort.py#L137-L180)                                            |

### 2.3 Transitive Case Tests

Reference: [`test_transitive_cases.py`](../../../tests/algorithms/tournament_graph/test_transitive_cases.py)

- `ToyCompareOracle` provides globally consistent comparisons (sorts by item id)
- Verifies correctness for various $(n, k, m)$ configurations
- Classic example: 25-horses puzzle solved in 7 oracle calls

### 2.4 Non-Transitive Case Tests

Reference: [`test_non_transitive_cases.py`](../../../tests/algorithms/tournament_graph/test_non_transitive_cases.py)

- `ToyNonTransitiveBucketOracle` creates cyclic relations within buckets (using mod-3 cycles)
- Buckets form SCCs; cross-bucket relations are transitive
- Tests verify that items from lower-numbered buckets are ranked higher
