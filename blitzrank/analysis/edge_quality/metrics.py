"""
Edge-quality, graph-level, and task-level metrics for the validation study.
"""
import math
import random
import statistics
from typing import Dict, List, Set, Tuple

import numpy as np

Edge = Tuple[str, str]


def pairwise_accuracy(
    predicted: Set[Edge], ground_truth: Set[Edge], ambiguous: Set[Tuple[str, str]]
) -> Dict[str, float]:
    """Edge accuracy excluding ambiguous (tied) pairs from the denominator."""
    gt_set = set(ground_truth)
    evaluable = {
        e for e in predicted
        if e in gt_set or (e[1], e[0]) in gt_set
    }
    if not evaluable:
        return {"accuracy": float("nan"), "n_evaluable": 0, "n_correct": 0}
    correct = sum(1 for e in evaluable if e in gt_set)
    return {
        "accuracy": correct / len(evaluable),
        "n_evaluable": len(evaluable),
        "n_correct": correct,
    }


def weighted_accuracy_by_gap(
    predicted: Set[Edge],
    qrels: Dict[str, int],
) -> Dict[str, float]:
    """Weight each edge by |rel(u) - rel(v)| so high-gap errors count more."""
    total_weight = 0.0
    weighted_correct = 0.0
    for u, v in predicted:
        rel_u = int(qrels.get(u, 0))
        rel_v = int(qrels.get(v, 0))
        gap = abs(rel_u - rel_v)
        if gap == 0:
            continue
        total_weight += gap
        if rel_u > rel_v:
            weighted_correct += gap
    if total_weight == 0:
        return {"weighted_accuracy": float("nan")}
    return {"weighted_accuracy": weighted_correct / total_weight}


def swap_flip_rate(
    edges_original: Set[Edge], edges_swapped: Set[Edge]
) -> Dict[str, float]:
    """Fraction of edges that flip direction when doc order is reversed."""
    common_pairs = set()
    for u, v in edges_original:
        if (u, v) in edges_swapped or (v, u) in edges_swapped:
            common_pairs.add((u, v))
    if not common_pairs:
        return {"flip_rate": float("nan"), "n_compared": 0}
    flips = sum(1 for u, v in common_pairs if (v, u) in edges_swapped)
    return {"flip_rate": flips / len(common_pairs), "n_compared": len(common_pairs)}


def self_consistency(runs: List[Set[Edge]]) -> Dict[str, float]:
    """Agreement rate across repeated runs on the same input."""
    if len(runs) < 2:
        return {"self_consistency": float("nan")}
    all_pairs: Set[Tuple[str, str]] = set()
    for r in runs:
        for u, v in r:
            all_pairs.add((min(u, v), max(u, v)))
    agreements = 0
    total = 0
    for u, v in all_pairs:
        directions = []
        for r in runs:
            if (u, v) in r:
                directions.append(1)
            elif (v, u) in r:
                directions.append(-1)
        if len(directions) < 2:
            continue
        total += 1
        if len(set(directions)) == 1:
            agreements += 1
    if total == 0:
        return {"self_consistency": float("nan")}
    return {"self_consistency": agreements / total, "n_pairs_checked": total}


def transitivity_violations(edges: Set[Edge]) -> Dict[str, int]:
    """Count directed 3-cycles."""
    adj: Dict[str, Set[str]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
    cycles = 0
    for u, successors_u in adj.items():
        for v in successors_u:
            for w in adj.get(v, ()):
                if u in adj.get(w, ()):
                    cycles += 1
    return {"three_cycles": cycles // 3}


def scc_profile(edges: Set[Edge], nodes: List[str]) -> Dict[str, float]:
    """SCC size statistics using Tarjan-like approach via networkx."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    sccs = list(nx.strongly_connected_components(G))
    sizes = sorted([len(s) for s in sccs], reverse=True)
    non_trivial = [s for s in sizes if s > 1]
    return {
        "num_sccs": len(sizes),
        "num_non_trivial_sccs": len(non_trivial),
        "max_scc_size": sizes[0] if sizes else 0,
        "mean_scc_size": statistics.mean(sizes) if sizes else 0,
    }


def bootstrap_ci(
    values: List[float], n_bootstrap: int = 2000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.RandomState(42)
    means = []
    arr = np.array(values)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    means.sort()
    lo = means[int(n_bootstrap * alpha / 2)]
    hi = means[int(n_bootstrap * (1 - alpha / 2))]
    return float(np.mean(arr)), lo, hi


def paired_bootstrap_test(
    values_a: List[float], values_b: List[float], n_bootstrap: int = 5000
) -> Dict[str, float]:
    """Two-sided paired bootstrap test. Returns p-value and mean delta."""
    assert len(values_a) == len(values_b)
    n = len(values_a)
    if n == 0:
        return {"p_value": float("nan"), "mean_delta": float("nan")}
    deltas = [a - b for a, b in zip(values_a, values_b)]
    observed = statistics.mean(deltas)
    rng = random.Random(42)
    count = 0
    for _ in range(n_bootstrap):
        sample = [d * rng.choice([-1, 1]) for d in deltas]
        if abs(statistics.mean(sample)) >= abs(observed):
            count += 1
    return {"p_value": count / n_bootstrap, "mean_delta": observed}


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return list of booleans: True if hypothesis rejected after correction."""
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    m = len(p_values)
    rejected = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            rejected[orig_idx] = True
        else:
            break
    return rejected
