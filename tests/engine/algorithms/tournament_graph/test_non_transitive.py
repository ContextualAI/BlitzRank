from blitzrank.engine.algorithms.tournament_graph.tournament_graph import TournamentGraph
from blitzrank.engine.algorithms.tournament_graph.tournament_graph_sort import (
    tournament_graph_sort,
    TournamentGraphSortConfig,
    TournamentGraphSortResult,
)
from blitzrank.engine.algorithms.tournament_graph.toy_setup import (
    make_shuffled_bucketed_items,
    ToyNonTransitiveBucketOracle,
    ToyNonTransitiveFullPairwiseOracle,
    Item,
)
import networkx as nx
import pytest
import random


def _get_bucket(item: Item) -> int:
    return int(str(item.id).split(".")[0])


def _verify_results_from_buckets(
    result: TournamentGraphSortResult, expected_buckets: list
) -> None:
    result_buckets = [_get_bucket(item) for item in result.results]
    assert result_buckets == expected_buckets, (
        f"Expected bucket sequence {expected_buckets}, got {result_buckets}\n"
        f"Items: {[item.id for item in result.results]}"
    )


# --------------------------------------------------------------------------- #
# Unit-level: TournamentGraph with a known 3-cycle
# --------------------------------------------------------------------------- #


def test_single_3_cycle_graph_structure() -> None:
    """
    Directly construct a TournamentGraph with a 3-cycle and verify:
    - num_3_cycles == 1
    - All three nodes end up in the same SCC (num_sccs == 1)
    - Each node's known_relationships == 2 (knows both peers via the SCC)
    """
    a, b, c = Item("0.0"), Item("0.1"), Item("0.2")
    g = TournamentGraph([a, b, c], enforce_tournament=True)
    # 3-cycle: b→a, a→c, c→b
    round_output = g.process_round([(b, a), (a, c), (c, b)])

    assert round_output.num_3_cycles == 1, (
        f"Expected 1 three-cycle, got {round_output.num_3_cycles}"
    )
    assert round_output.num_sccs == 1, (
        f"Expected single SCC (all tied), got {round_output.num_sccs}"
    )
    for node in [a, b, c]:
        assert round_output.known_relationships[node] == 2, (
            f"Node {node.id} should know 2 others, got {round_output.known_relationships[node]}"
        )


# --------------------------------------------------------------------------- #
# Single-bucket: purely non-transitive
# --------------------------------------------------------------------------- #


def test_single_3_cycle_terminates() -> None:
    """
    Minimum non-transitive case: 3 items in one bucket form a directed 3-cycle.
    The algorithm must detect it, form an SCC, finalize all items, and report cycles.
    """
    items = make_shuffled_bucketed_items([0, 0, 0])
    oracle = ToyNonTransitiveBucketOracle(k=3)
    result = tournament_graph_sort(items, oracle, num_top_nodes_to_output=3)

    assert len(result.results) == 3
    _verify_results_from_buckets(result, [0, 0, 0])


def test_single_bucket_9_items_all_finalized() -> None:
    """
    Larger single-bucket case (9 items). Multiple 3-cycles exist across residue classes.
    All items should end up in the same SCC once the algorithm has run enough rounds.
    """
    items = make_shuffled_bucketed_items([0] * 9)
    oracle = ToyNonTransitiveBucketOracle(k=3)
    result = tournament_graph_sort(
        items,
        oracle,
        num_top_nodes_to_output=9,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=200),
    )

    assert len(result.results) == 9
    _verify_results_from_buckets(result, [0] * 9)


# --------------------------------------------------------------------------- #
# Multi-bucket: cross-bucket transitivity must survive within-bucket cycles
# --------------------------------------------------------------------------- #


def test_single_item_bucket() -> None:
    """
    Bucket 0 has exactly 1 item (no cycle possible); bucket 1 has 3 items forming a cycle.
    Top-1 must still be from bucket 0 — the size-1 bucket must not confuse the algorithm.
    """
    items = make_shuffled_bucketed_items([0, 1, 1, 1])
    oracle = ToyNonTransitiveBucketOracle(k=3)
    result = tournament_graph_sort(items, oracle, num_top_nodes_to_output=1)

    _verify_results_from_buckets(result, [0])


def test_two_buckets_top_m_equals_bucket_size() -> None:
    """
    Two equal-size buckets, m = size of bucket 0.
    All top results must come from bucket 0 despite within-bucket cycles.
    """
    items = make_shuffled_bucketed_items([0, 0, 0, 1, 1, 1])
    oracle = ToyNonTransitiveBucketOracle(k=3)
    result = tournament_graph_sort(items, oracle, num_top_nodes_to_output=3)

    _verify_results_from_buckets(result, [0, 0, 0])


def test_two_buckets_top_1() -> None:
    """
    Tightest correctness check: top-1 result must always be from bucket 0.
    """
    items = make_shuffled_bucketed_items([0, 0, 0, 1, 1, 1])
    oracle = ToyNonTransitiveBucketOracle(k=3)
    result = tournament_graph_sort(items, oracle, num_top_nodes_to_output=1)

    _verify_results_from_buckets(result, [0])


def test_three_buckets_boundary_crossing() -> None:
    """
    m crosses the bucket-0 boundary: top-4 from [0,0,1,1,1,2,2,2] should be
    the 2 bucket-0 items followed by 2 bucket-1 items.
    """
    items = make_shuffled_bucketed_items([0, 0, 1, 1, 1, 2, 2, 2])
    oracle = ToyNonTransitiveBucketOracle(k=3)
    result = tournament_graph_sort(items, oracle, num_top_nodes_to_output=4)

    _verify_results_from_buckets(result, [0, 0, 1, 1])


# --------------------------------------------------------------------------- #
# Full-pairwise oracle: guaranteed cycle exposure
# --------------------------------------------------------------------------- #


def test_full_pairwise_oracle_exposes_3_cycle() -> None:
    """
    With the full-pairwise oracle, all C(k,2) edges are emitted in a single match.
    For 3 items in one bucket this guarantees the cycle-closing edge is present,
    so num_3_cycles must be >= 1 after the sort.
    """
    items = make_shuffled_bucketed_items([0, 0, 0])
    oracle = ToyNonTransitiveFullPairwiseOracle(k=3)
    result = tournament_graph_sort(items, oracle, num_top_nodes_to_output=3)

    _verify_results_from_buckets(result, [0, 0, 0])
    assert result.get_scc_stats().num_3_cycles >= 1, (
        "Full-pairwise oracle must expose the 3-cycle on the first match"
    )


def test_scc_forms_within_bucket() -> None:
    """
    With k == bucket_size, all bucket items appear in round 1 and the full-pairwise
    oracle closes the cycle immediately. Verify via nx.condensation that all 3 items
    share a single SCC in the final graph — a graph-level check decoupled from the
    sort-level bucket-ordering assertions in other tests.

    Note: multi-bucket setups with k=3 won't exhibit within-bucket SCC formation
    because the SCC-based scheduler picks one rep per SCC and always selects
    cross-bucket representatives when there are 3+ distinct-bucket SCCs.
    """
    items = make_shuffled_bucketed_items([0, 0, 0])
    oracle = ToyNonTransitiveFullPairwiseOracle(k=3)
    result = tournament_graph_sort(items, oracle, num_top_nodes_to_output=3)

    condensation = nx.condensation(result.graph.get_graph())
    scc_membership = condensation.graph["mapping"]  # node -> scc_idx

    scc_ids = {scc_membership[item] for item in items}
    assert len(scc_ids) == 1, (
        f"All 3 same-bucket items should share 1 SCC, but span {len(scc_ids)} SCCs"
    )
    assert result.get_scc_stats().num_3_cycles >= 1, (
        "nx.condensation must reflect the detected 3-cycle"
    )


# --------------------------------------------------------------------------- #
# SCC structure validation
# --------------------------------------------------------------------------- #


def test_tournament_graph_multi_scc_cycle_structure() -> None:
    """
    Directly inject a rich non-transitive edge set into TournamentGraph and verify
    that SCC detection and 3-cycle counting are correct.

    Setup: 3 buckets × 3 items = 9 items.
    - Within each bucket: a directed 3-cycle (1→0, 0→2, 2→1).
    - Across buckets: bucket 0 beats all of bucket 1 and 2; bucket 1 beats all of bucket 2.

    Expected:
    - 3 SCCs (one per bucket — within-bucket cycles merge each bucket into one SCC).
    - 3 three-cycles detected.
    - Cross-bucket reach: bucket-0 items have out_reach=6, bucket-2 items have in_reach=6.
    """
    buckets = [[Item(f"{b}.{w}") for w in range(3)] for b in range(3)]
    all_items = [item for bucket in buckets for item in bucket]
    g = TournamentGraph(all_items, enforce_tournament=True)

    edges = []
    # Within-bucket 3-cycles: 1→0, 0→2, 2→1
    for bucket_items in buckets:
        a, b, c = bucket_items  # within indices 0, 1, 2
        edges += [(b, a), (a, c), (c, b)]

    # Cross-bucket edges (bucket 0 > bucket 1 > bucket 2)
    for b0 in buckets[0]:
        for b1 in buckets[1]:
            edges.append((b0, b1))
        for b2 in buckets[2]:
            edges.append((b0, b2))
    for b1 in buckets[1]:
        for b2 in buckets[2]:
            edges.append((b1, b2))

    round_output = g.process_round(edges)

    assert round_output.num_sccs == 3, (
        f"Expected 3 SCCs (one per bucket), got {round_output.num_sccs}"
    )
    assert round_output.num_3_cycles == 3, (
        f"Expected 3 three-cycles (one per bucket), got {round_output.num_3_cycles}"
    )

    for i, bucket_items in enumerate(buckets):
        scc_ids = {round_output.scc_membership[item] for item in bucket_items}
        assert len(scc_ids) == 1, (
            f"Bucket {i} items span {len(scc_ids)} SCCs, expected 1"
        )

    # Bucket 0 beats all 6 items in buckets 1 and 2
    b0_rep = buckets[0][0]
    assert round_output.out_reach[b0_rep] == 6, (
        f"Bucket-0 representative should have out_reach=6, got {round_output.out_reach[b0_rep]}"
    )
    # Bucket 2 is beaten by all 6 items in buckets 0 and 1
    b2_rep = buckets[2][0]
    assert round_output.in_reach[b2_rep] == 6, (
        f"Bucket-2 representative should have in_reach=6, got {round_output.in_reach[b2_rep]}"
    )


# --------------------------------------------------------------------------- #
# Parameterized correctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bucket_labels,k,m,expected_bucket_seq",
    [
        ([0, 0, 1, 1], 3, 2, [0, 0]),
        ([0, 0, 0, 1, 1, 1], 3, 6, [0, 0, 0, 1, 1, 1]),
        ([0, 1, 1, 2, 2, 2], 3, 3, [0, 1, 1]),
        ([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], 4, 5, [0, 0, 1, 1, 1]),
        ([0, 0, 0, 0, 1, 1, 1, 1], 4, 4, [0, 0, 0, 0]),
    ],
)
def test_correctness_parameterized(
    bucket_labels: list, k: int, m: int, expected_bucket_seq: list
) -> None:
    items = make_shuffled_bucketed_items(bucket_labels)
    oracle = ToyNonTransitiveBucketOracle(k=k)
    result = tournament_graph_sort(
        items,
        oracle,
        num_top_nodes_to_output=m,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=200),
    )
    _verify_results_from_buckets(result, expected_bucket_seq)


# --------------------------------------------------------------------------- #
# Stress tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_stress_multi_bucket_random_seeds(seed: int) -> None:
    """
    3 buckets × 4 items, top-6 should always be: 4 from bucket 0, 2 from bucket 1.
    Different shuffle orders must not affect the bucket-level correctness.
    """
    random.seed(seed)
    items = make_shuffled_bucketed_items([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    oracle = ToyNonTransitiveBucketOracle(k=4)
    result = tournament_graph_sort(
        items,
        oracle,
        num_top_nodes_to_output=6,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=200),
    )
    _verify_results_from_buckets(result, [0, 0, 0, 0, 1, 1])


def test_large_non_transitive() -> None:
    """
    5 buckets × 4 items = 20 items total, k=4, m=8.
    Top-8 must be: all 4 from bucket 0 followed by all 4 from bucket 1.
    """
    random.seed(42)
    items = make_shuffled_bucketed_items([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    oracle = ToyNonTransitiveBucketOracle(k=4)
    result = tournament_graph_sort(
        items,
        oracle,
        num_top_nodes_to_output=8,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=300),
    )
    _verify_results_from_buckets(result, [0, 0, 0, 0, 1, 1, 1, 1])
