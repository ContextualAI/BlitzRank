"""
Extract pairwise edges from LLM ranking outputs.

Supports two edge encodings:
- adjacent_edges: (item[i], item[i+1]) for consecutive pairs (what BlitzRank uses)
- pair_complete_edges: all (item[i], item[j]) for i < j (full transitive closure of permutation)
"""
from typing import List, Tuple


Edge = Tuple[str, str]


def edges_from_permutation_adjacent(
    doc_ids: List[str], permutation: List[int]
) -> List[Edge]:
    ordered = [doc_ids[p] for p in permutation]
    return [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]


def edges_from_permutation_complete(
    doc_ids: List[str], permutation: List[int]
) -> List[Edge]:
    ordered = [doc_ids[p] for p in permutation]
    return [
        (ordered[i], ordered[j])
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    ]


def edges_from_pairwise_winner(
    doc_id_a: str, doc_id_b: str, winner_index: int
) -> List[Edge]:
    if winner_index == 0:
        return [(doc_id_a, doc_id_b)]
    return [(doc_id_b, doc_id_a)]
