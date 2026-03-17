from .compare_oracle import (
    CompareOracle,
    Item,
    OracleResult,
    make_edges_from_linear_order,
)
from typing import List, Dict, Tuple, Union
import random


#  -------------------------- Transitive Setup --------------------------
class ToyCompareOracle(CompareOracle):
    def _compare_k(self, items: List[Item]) -> OracleResult:
        if len(items) > self.k:
            raise ValueError(
                f"compare_k expects at most k={self.k} items, got {len(items)}"
            )
        ordered = sorted(items, key=lambda x: x.id)
        return OracleResult(edges=make_edges_from_linear_order(ordered))

    async def _compare_k_async(self, items: List[Item]) -> OracleResult:
        return self._compare_k(items)


def make_shuffled_items(n: int) -> List[Item]:
    items = [Item(i) for i in range(n)]
    random.shuffle(items)
    return items


#  -------------------------- Non-transitive Setup --------------------------


class ToyNonTransitiveFullPairwiseOracle(CompareOracle):
    """
    Like ToyNonTransitiveBucketOracle but emits all C(k,2) pairwise edges rather
    than only the adjacent pairs of the Hamiltonian path.

    This guarantees cycle-closing edges appear in the graph whenever cycle members
    are scheduled in the same match, making SCC formation deterministic in tests
    rather than dependent on scheduling luck.
    """

    def _compare_k(self, items: List[Item]) -> OracleResult:
        if len(items) > self.k:
            raise ValueError(
                f"compare_k expects at most k={self.k} items, got {len(items)}"
            )
        edges = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                edges.append((a, b) if _bucket_beats(a, b) else (b, a))
        return OracleResult(edges=edges)

    async def _compare_k_async(self, items: List[Item]) -> OracleResult:
        return self._compare_k(items)


class ToyNonTransitiveBucketOracle(CompareOracle):
    def _compare_k(self, items: List[Item]) -> OracleResult:
        """
        Bucketed non-transitive oracle (Option A):
        - Cross-bucket: bucket 0 beats bucket 1 beats bucket 2 ...
        - Within a bucket: define a cyclic tournament relation to enable SCCs (>= 3).

        We encode ids as strings "{bucket}.{within_bucket}", e.g. "2.1".
        Given a subset of items, we return a Hamiltonian path under the "beats" relation
        (so adjacent pairs are consistent with that relation).
        """
        if len(items) > self.k:
            raise ValueError(
                f"compare_k expects at most k={self.k} items, got {len(items)}"
            )
        items_by_bucket: Dict[int, List[Tuple[int, Item]]] = {}
        for item in items:
            bucket, within = _parse_bucketed_item_id(item.id)
            items_by_bucket.setdefault(bucket, []).append((within, item))

        ordered: List[Item] = []
        for bucket in sorted(items_by_bucket.keys()):
            within_and_items = sorted(items_by_bucket[bucket], key=lambda x: x[0])
            bucket_items = [it for _, it in within_and_items]
            ordered.extend(_hamiltonian_path(bucket_items, _bucket_beats))

        return OracleResult(edges=make_edges_from_linear_order(ordered))

    async def _compare_k_async(self, items: List[Item]) -> OracleResult:
        return self._compare_k(items)


def make_shuffled_bucketed_items(bucket_labels: List[int]) -> List[Item]:
    """
    Example:
    - bucket_labels = [0, 0, 1, 1, 1, 2, 2]
    Construct ground truth:
    - items = [Item('0.0'), Item('0.1'), Item('1.0'), Item('1.1'), Item('1.2'), Item('2.0'), Item('2.1')]
    Then return a shuffled version of items.
    Now there is no longer a total order on the items.
    While [0.0, 0.1, 1.0, 1.1, 1.2, 2.0, 2.1] is a valid order,
    we also accept [0.1, 0.0, 1.2, 1.0, 1.1, 2.0, 2.1] as a valid order.
    We only sort the first position.
    """
    counts_by_bucket: Dict[int, int] = {}
    items: List[Item] = []
    for bucket in bucket_labels:
        within = counts_by_bucket.get(bucket, 0)
        counts_by_bucket[bucket] = within + 1
        # Note: Item is annotated with id: int, but this toy setup intentionally uses
        # string ids to encode (bucket, within_bucket) in a readable way.
        items.append(Item(f"{bucket}.{within}"))  # type: ignore[arg-type]

    random.shuffle(items)
    return items


def _parse_bucketed_item_id(item_id: Union[int, str]) -> Tuple[int, int]:
    if not isinstance(item_id, str):
        raise TypeError(
            f"Bucketed setup expects string item ids like '2.1', got {type(item_id)}"
        )
    parts = item_id.split(".")
    if len(parts) != 2:
        raise ValueError(f"Expected id like 'bucket.within', got {item_id!r}")
    bucket_str, within_str = parts
    try:
        bucket = int(bucket_str)
        within = int(within_str)
    except ValueError as e:
        raise ValueError(f"Expected numeric 'bucket.within', got {item_id!r}") from e
    return bucket, within


def _bucket_beats(a: Item, b: Item) -> bool:
    """
    Tournament-style "beats" relation:
    - Lower bucket always beats higher bucket.
    - Within a bucket, use a 3-cycle on residue classes mod 3 to induce non-transitivity.
      (0 beats 2, 1 beats 0, 2 beats 1). Ties are broken deterministically by within index.
    """
    a_bucket, a_within = _parse_bucketed_item_id(a.id)
    b_bucket, b_within = _parse_bucketed_item_id(b.id)

    if a_bucket != b_bucket:
        return a_bucket < b_bucket

    a_r = a_within % 3
    b_r = b_within % 3
    if a_r != b_r:
        return (a_r - b_r) % 3 == 1

    return a_within < b_within


def _hamiltonian_path(items: List[Item], beats) -> List[Item]:
    """
    Construct an ordering where each adjacent pair (u, v) satisfies beats(u, v).
    Standard insertion algorithm for tournaments.
    """
    path: List[Item] = []
    for v in items:
        if not path:
            path.append(v)
            continue

        inserted = False
        for i, u in enumerate(path):
            if beats(v, u):
                path.insert(i, v)
                inserted = True
                break
        if not inserted:
            path.append(v)
    return path
