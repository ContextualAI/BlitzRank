"""
Construct pairwise ground-truth edges from relevance labels and strong-LLM consensus.
"""
from typing import Dict, List, Set, Tuple

Edge = Tuple[str, str]


def edges_from_qrels(
    qrels_for_query: Dict[str, int], doc_ids: List[str]
) -> Tuple[Set[Edge], Set[Tuple[str, str]]]:
    """Return (strict_edges, ambiguous_pairs).

    strict_edges: (u, v) where rel(u) > rel(v)
    ambiguous_pairs: {u, v} where rel(u) == rel(v) (ties)
    """
    strict: Set[Edge] = set()
    ambiguous: Set[Tuple[str, str]] = set()
    for i, u in enumerate(doc_ids):
        rel_u = int(qrels_for_query.get(u, 0))
        for j in range(i + 1, len(doc_ids)):
            v = doc_ids[j]
            rel_v = int(qrels_for_query.get(v, 0))
            if rel_u > rel_v:
                strict.add((u, v))
            elif rel_v > rel_u:
                strict.add((v, u))
            else:
                ambiguous.add((min(u, v), max(u, v)))
    return strict, ambiguous


def consensus_edges(
    edges_a: Set[Edge], edges_b: Set[Edge], all_pairs: Set[Tuple[str, str]]
) -> Tuple[Set[Edge], Set[Edge], Set[Tuple[str, str]]]:
    """Partition into agree / disagree / unknown.

    edges_a, edges_b: directed edges from two strong-LLM judges.
    all_pairs: set of unordered pairs to evaluate.

    Returns (agree_edges, disagree_pairs_as_edges_from_a, unknown_pairs).
    """
    agree: Set[Edge] = set()
    disagree: Set[Edge] = set()
    unknown: Set[Tuple[str, str]] = set()

    for u, v in all_pairs:
        a_uv = (u, v) in edges_a
        a_vu = (v, u) in edges_a
        b_uv = (u, v) in edges_b
        b_vu = (v, u) in edges_b

        if a_uv and b_uv:
            agree.add((u, v))
        elif a_vu and b_vu:
            agree.add((v, u))
        elif (a_uv or a_vu) and (b_uv or b_vu):
            edge = (u, v) if a_uv else (v, u)
            disagree.add(edge)
        else:
            unknown.add((u, v))

    return agree, disagree, unknown


def all_unordered_pairs(doc_ids: List[str]) -> Set[Tuple[str, str]]:
    return {
        (doc_ids[i], doc_ids[j])
        for i in range(len(doc_ids))
        for j in range(i + 1, len(doc_ids))
    }
