"""
Noise-injection calibration: build edge_error -> NDCG@10 sensitivity curve.

Uses qrels-derived perfect edges as a starting oracle, then progressively
corrupts random edges and measures how BlitzRank's NDCG@10 degrades.
"""
import random
from typing import Dict, List, Set, Tuple

import pytrec_eval

Edge = Tuple[str, str]


def _build_ranking_from_edges(
    edges: Set[Edge], doc_ids: List[str], top_m: int = 10
) -> List[str]:
    """Simple win-count ranking from edges."""
    wins: Dict[str, int] = {d: 0 for d in doc_ids}
    for u, v in edges:
        if u in wins:
            wins[u] = wins.get(u, 0) + 1
    ranked = sorted(doc_ids, key=lambda d: wins.get(d, 0), reverse=True)
    return ranked[:top_m]


def _ndcg_at_k(
    ranking: List[str],
    qrels_for_query: Dict[str, int],
    k: int = 10,
) -> float:
    qid = "0"
    qrels = {qid: {str(d): int(r) for d, r in qrels_for_query.items()}}
    results = {qid: {d: float(len(ranking) - i) for i, d in enumerate(ranking[:k])}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    scores = evaluator.evaluate(results)
    return scores.get(qid, {}).get(f"ndcg_cut_{k}", 0.0)


def calibration_curve(
    perfect_edges: Set[Edge],
    doc_ids: List[str],
    qrels_for_query: Dict[str, int],
    noise_levels: List[float] = None,
    seed: int = 42,
    top_m: int = 10,
) -> List[Dict[str, float]]:
    """Return list of {noise_fraction, ndcg_at_10} points.

    noise_fraction: fraction of edges whose direction is flipped.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    rng = random.Random(seed)
    edge_list = list(perfect_edges)
    points = []
    for noise_frac in noise_levels:
        n_flip = int(len(edge_list) * noise_frac)
        corrupted = set(edge_list)
        to_flip = rng.sample(range(len(edge_list)), min(n_flip, len(edge_list)))
        for idx in to_flip:
            u, v = edge_list[idx]
            corrupted.discard((u, v))
            corrupted.add((v, u))
        ranking = _build_ranking_from_edges(corrupted, doc_ids, top_m)
        ndcg = _ndcg_at_k(ranking, qrels_for_query)
        points.append({"noise_fraction": noise_frac, "ndcg_at_10": ndcg})
    return points


def effective_noise_level(
    observed_ndcg: float, curve: List[Dict[str, float]]
) -> float:
    """Interpolate the calibration curve to find the noise level matching observed NDCG."""
    sorted_curve = sorted(curve, key=lambda p: p["noise_fraction"])
    for i in range(len(sorted_curve) - 1):
        n1, v1 = sorted_curve[i]["noise_fraction"], sorted_curve[i]["ndcg_at_10"]
        n2, v2 = sorted_curve[i + 1]["noise_fraction"], sorted_curve[i + 1]["ndcg_at_10"]
        if v2 <= observed_ndcg <= v1 or v1 <= observed_ndcg <= v2:
            if abs(v1 - v2) < 1e-9:
                return (n1 + n2) / 2
            t = (observed_ndcg - v1) / (v2 - v1)
            return n1 + t * (n2 - n1)
    return sorted_curve[-1]["noise_fraction"]
