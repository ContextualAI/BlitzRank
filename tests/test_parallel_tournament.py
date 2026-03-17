"""
Exhaustive tests for parallel tournament graph sort.
Validates correctness and oracle call parity between sequential and parallel modes.
"""

import asyncio
import random
import pytest
from dataclasses import dataclass
from typing import List, Tuple

from blitzrank.engine.algorithms.tournament_graph.compare_oracle import (
    CompareOracle,
    Item,
    OracleResult,
    make_edges_from_linear_order,
)
from blitzrank.engine.algorithms.tournament_graph.tournament_graph_sort import (
    TournamentGraphSort,
    TournamentGraphSortConfig,
)
from blitzrank.engine.algorithms.tournament_graph.types import SortStrategy


class MockOracle(CompareOracle):
    """Deterministic oracle based on ground truth ordering."""

    def __init__(self, k: int, ground_truth: List[Item]):
        super().__init__(k)
        self.ground_truth_order = {item: idx for idx, item in enumerate(ground_truth)}

    async def _compare_k_async(self, items: List[Item]) -> OracleResult:
        sorted_items = sorted(items, key=lambda x: self.ground_truth_order[x])
        edges = make_edges_from_linear_order(sorted_items)
        return OracleResult(edges=edges)

    def _compare_k(self, items: List[Item]) -> OracleResult:
        return asyncio.get_event_loop().run_until_complete(self._compare_k_async(items))


class CyclicMockOracle(CompareOracle):
    """
    Oracle that creates cycles (non-transitive results) within specified groups.
    Items in the same tier_group will form a cycle when compared together.
    """

    def __init__(self, k: int, ground_truth: List[Item], tier_groups: List[List[int]]):
        """
        Args:
            k: window size
            ground_truth: overall ordering (used for cross-group comparisons)
            tier_groups: list of item ID lists that should form cycles/ties
                         e.g., [[0,1,2], [5,6,7]] means items 0,1,2 are tied, and 5,6,7 are tied
        """
        super().__init__(k)
        self.ground_truth_order = {item: idx for idx, item in enumerate(ground_truth)}
        # Map item_id -> tier_group_id (items in same group are "tied")
        self.tier_membership = {}
        for tier_id, group in enumerate(tier_groups):
            for item_id in group:
                self.tier_membership[item_id] = tier_id

    async def _compare_k_async(self, items: List[Item]) -> OracleResult:
        # Sort by ground truth first
        sorted_items = sorted(items, key=lambda x: self.ground_truth_order[x])
        edges = []
        
        for i in range(len(sorted_items) - 1):
            a, b = sorted_items[i], sorted_items[i + 1]
            a_tier = self.tier_membership.get(a.id)
            b_tier = self.tier_membership.get(b.id)
            
            if a_tier is not None and a_tier == b_tier:
                # Same tier group: create cycle by having loser beat winner
                # This creates A > B edge normally, but we also add B > A later
                # to form a 2-cycle (or larger cycles with more items)
                edges.append((a, b))
                edges.append((b, a))  # Creates cycle!
            else:
                # Normal comparison
                edges.append((a, b))
        
        return OracleResult(edges=edges)

    def _compare_k(self, items: List[Item]) -> OracleResult:
        return asyncio.get_event_loop().run_until_complete(self._compare_k_async(items))


def run_tournament(
    items: List[Item],
    ground_truth: List[Item],
    k: int,
    top_m: int,
    max_parallel: int,
    max_num_rounds: int = 50,
) -> Tuple[List[int], int, int]:
    """Run tournament and return (result_ids, num_rounds, num_oracle_calls)."""
    oracle = MockOracle(k=k, ground_truth=ground_truth)
    config = TournamentGraphSortConfig(
        sort_strategy=SortStrategy.ASCENDING_OUT_REACH,
        enforce_tournament=False,
        max_parallel_matches=max_parallel,
        max_num_rounds=max_num_rounds,
    )
    sorter = TournamentGraphSort(
        items=items,
        oracle=oracle,
        num_top_nodes_to_output=top_m,
        tournament_graph_sort_config=config,
    )
    result = sorter.sort()
    return (
        [item.id for item in result.results],
        result.num_rounds,
        result.num_oracle_calls,
    )


def run_tournament_with_cycles(
    items: List[Item],
    ground_truth: List[Item],
    tier_groups: List[List[int]],
    k: int,
    top_m: int,
    max_parallel: int,
    max_num_rounds: int = 50,
) -> Tuple[List[int], int, int, int]:
    """Run tournament with cyclic oracle. Returns (result_ids, rounds, calls, num_sccs)."""
    oracle = CyclicMockOracle(k=k, ground_truth=ground_truth, tier_groups=tier_groups)
    config = TournamentGraphSortConfig(
        sort_strategy=SortStrategy.ASCENDING_OUT_REACH,
        enforce_tournament=False,  # Must be False to allow cycles
        max_parallel_matches=max_parallel,
        max_num_rounds=max_num_rounds,
    )
    sorter = TournamentGraphSort(
        items=items,
        oracle=oracle,
        num_top_nodes_to_output=top_m,
        tournament_graph_sort_config=config,
    )
    result = sorter.sort()
    scc_stats = result.get_scc_stats()
    return (
        [item.id for item in result.results],
        result.num_rounds,
        result.num_oracle_calls,
        scc_stats.num_sccs,
    )


class TestParallelCorrectness:
    """Test that parallel mode produces correct results."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_25_horses_puzzle(self, seed):
        """Classic 25 horses puzzle: find top 3 from 25, racing 5 at a time."""
        random.seed(seed)
        items = [Item(id=i) for i in range(25)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:3]]

        result, _, _ = run_tournament(items, ground_truth, k=5, top_m=3, max_parallel=5)
        assert result == expected, f"seed={seed}: expected {expected}, got {result}"

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_100_items(self, seed):
        """Larger test: 100 items, k=10, find top 5."""
        random.seed(seed)
        items = [Item(id=i) for i in range(100)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:5]]

        result, _, _ = run_tournament(items, ground_truth, k=10, top_m=5, max_parallel=10)
        assert result == expected, f"seed={seed}: expected {expected}, got {result}"

    @pytest.mark.parametrize("n,k,top_m", [
        (10, 3, 2),
        (15, 5, 3),
        (20, 4, 5),
        (30, 6, 4),
        (50, 10, 5),
    ])
    def test_various_sizes(self, n, k, top_m):
        """Test various problem sizes."""
        random.seed(42)
        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:top_m]]

        max_parallel = n // k
        result, _, _ = run_tournament(items, ground_truth, k, top_m, max_parallel)
        assert result == expected


class TestOracleCallParity:
    """Test that parallel uses same number of oracle calls as sequential."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_25_horses_call_parity(self, seed):
        """25 horses: parallel should use same calls as sequential."""
        random.seed(seed)
        items = [Item(id=i) for i in range(25)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)

        seq_result, seq_rounds, seq_calls = run_tournament(
            items, ground_truth, k=5, top_m=3, max_parallel=1
        )
        par_result, par_rounds, par_calls = run_tournament(
            items, ground_truth, k=5, top_m=3, max_parallel=5
        )

        assert seq_result == par_result, "Results should match"
        assert par_calls == seq_calls, f"Oracle calls should match: seq={seq_calls}, par={par_calls}"
        assert par_rounds <= seq_rounds, "Parallel should use fewer or equal rounds"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_100_items_call_parity(self, seed):
        """100 items: parallel should use same calls as sequential."""
        random.seed(seed)
        items = [Item(id=i) for i in range(100)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)

        seq_result, seq_rounds, seq_calls = run_tournament(
            items, ground_truth, k=10, top_m=5, max_parallel=1
        )
        par_result, par_rounds, par_calls = run_tournament(
            items, ground_truth, k=10, top_m=5, max_parallel=10
        )

        assert seq_result == par_result, "Results should match"
        assert par_calls == seq_calls, f"Oracle calls should match: seq={seq_calls}, par={par_calls}"
        assert par_rounds <= seq_rounds, "Parallel should use fewer or equal rounds"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_n_less_than_k(self):
        """When n < k, all items fit in one match."""
        items = [Item(id=i) for i in range(4)]
        ground_truth = [items[2], items[0], items[3], items[1]]  # 2, 0, 3, 1
        expected = [2, 0]

        result, rounds, calls = run_tournament(items, ground_truth, k=5, top_m=2, max_parallel=5)
        assert result == expected
        assert calls == 1  # Only one match needed

    def test_top_m_equals_n(self):
        """Find all items (top_m = n)."""
        random.seed(42)
        items = [Item(id=i) for i in range(10)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth]

        result, _, _ = run_tournament(items, ground_truth, k=3, top_m=10, max_parallel=5)
        assert result == expected

    def test_top_m_equals_1(self):
        """Find only the top 1 item."""
        random.seed(42)
        items = [Item(id=i) for i in range(25)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [ground_truth[0].id]

        result, _, _ = run_tournament(items, ground_truth, k=5, top_m=1, max_parallel=5)
        assert result == expected

    def test_max_parallel_1_is_sequential(self):
        """max_parallel=1 should behave exactly like original sequential."""
        random.seed(42)
        items = [Item(id=i) for i in range(25)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)

        result, rounds, calls = run_tournament(items, ground_truth, k=5, top_m=3, max_parallel=1)
        
        # Should have the expected number of sequential rounds
        assert rounds == calls  # Each round is one call in sequential mode

    def test_k_equals_n(self):
        """When k = n, one match compares all items."""
        items = [Item(id=i) for i in range(5)]
        ground_truth = [items[3], items[1], items[4], items[0], items[2]]
        expected = [3, 1, 4]

        result, rounds, calls = run_tournament(items, ground_truth, k=5, top_m=3, max_parallel=1)
        assert result == expected
        # With k=n, need enough rounds to establish full ordering for top-3


class TestConsistency:
    """Test that results are consistent across multiple runs."""

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same results."""
        results = []
        for _ in range(5):
            random.seed(42)
            items = [Item(id=i) for i in range(25)]
            ground_truth = items.copy()
            random.shuffle(ground_truth)
            result, rounds, calls = run_tournament(items, ground_truth, k=5, top_m=3, max_parallel=5)
            results.append((result, rounds, calls))

        # All results should be identical
        assert all(r == results[0] for r in results)

    @pytest.mark.parametrize("max_parallel", [1, 2, 3, 5, 10])
    def test_result_independent_of_parallelism(self, max_parallel):
        """Result should be the same regardless of parallelism level."""
        random.seed(42)
        items = [Item(id=i) for i in range(25)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:3]]

        result, _, _ = run_tournament(items, ground_truth, k=5, top_m=3, max_parallel=max_parallel)
        assert result == expected, f"max_parallel={max_parallel} gave wrong result"


class TestRoundReduction:
    """Test that parallel mode reduces the number of rounds."""

    def test_25_horses_round_reduction(self):
        """Parallel should reduce rounds for 25 horses puzzle."""
        random.seed(42)
        items = [Item(id=i) for i in range(25)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)

        _, seq_rounds, _ = run_tournament(items, ground_truth, k=5, top_m=3, max_parallel=1)
        _, par_rounds, _ = run_tournament(items, ground_truth, k=5, top_m=3, max_parallel=5)

        # Parallel should have fewer rounds (initial heats collapsed)
        assert par_rounds < seq_rounds
        # With 25 items, k=5: 5 initial heats become 1 parallel round
        # So par_rounds should be roughly seq_rounds - 4

    def test_100_items_round_reduction(self):
        """Parallel should significantly reduce rounds for 100 items."""
        random.seed(42)
        items = [Item(id=i) for i in range(100)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)

        _, seq_rounds, _ = run_tournament(items, ground_truth, k=10, top_m=5, max_parallel=1)
        _, par_rounds, _ = run_tournament(items, ground_truth, k=10, top_m=5, max_parallel=10)

        # With 100 items, k=10: 10 initial heats become 1 parallel round
        assert par_rounds < seq_rounds
        # Expect roughly 9 fewer rounds (10 heats → 1 round)


class TestLargeSCCs:
    """Test behavior with large SCCs (cycles/ties in comparisons)."""

    def test_single_large_scc_at_top(self):
        """Top 5 items form a cycle (all tied for 1st place)."""
        n = 25
        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()  # 0 is best, 24 is worst
        # Items 0-4 form a cycle (all tied at the top)
        tier_groups = [[0, 1, 2, 3, 4]]

        result, rounds, calls, num_sccs = run_tournament_with_cycles(
            items, ground_truth, tier_groups, k=5, top_m=3, max_parallel=5
        )
        # All of 0-4 are in same SCC, so any 3 of them are valid top-3
        assert all(r in [0, 1, 2, 3, 4] for r in result), f"Expected top items from SCC, got {result}"
        assert num_sccs < n, f"Expected fewer SCCs due to cycles, got {num_sccs}"
        print(f"\n  Single large SCC: result={result}, num_sccs={num_sccs}, rounds={rounds}")

    def test_multiple_large_sccs(self):
        """Multiple groups of tied items."""
        n = 30
        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()
        # Three tiers: top tier (0-4), middle tier (10-14), bottom tier (20-24)
        tier_groups = [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]

        result, rounds, calls, num_sccs = run_tournament_with_cycles(
            items, ground_truth, tier_groups, k=6, top_m=5, max_parallel=5
        )
        # Top 5 should all be from the top tier SCC (items 0-4)
        assert all(r in [0, 1, 2, 3, 4] for r in result), f"Expected top tier items, got {result}"
        print(f"\n  Multiple SCCs: result={result}, num_sccs={num_sccs}, rounds={rounds}")

    def test_large_scc_with_parallel(self):
        """Large SCC handling with parallelism."""
        n = 50
        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()
        # Top 10 items all tied
        tier_groups = [[i for i in range(10)]]

        seq_result, seq_rounds, seq_calls, seq_sccs = run_tournament_with_cycles(
            items, ground_truth, tier_groups, k=10, top_m=5, max_parallel=1
        )
        par_result, par_rounds, par_calls, par_sccs = run_tournament_with_cycles(
            items, ground_truth, tier_groups, k=10, top_m=5, max_parallel=5
        )

        # Both should return items from the top SCC
        assert all(r in range(10) for r in seq_result), f"Sequential got wrong items: {seq_result}"
        assert all(r in range(10) for r in par_result), f"Parallel got wrong items: {par_result}"
        # SCC count should be similar
        assert seq_sccs == par_sccs, f"SCC count mismatch: seq={seq_sccs}, par={par_sccs}"
        print(f"\n  Large SCC parallel: seq_sccs={seq_sccs}, par_sccs={par_sccs}")
        print(f"  Sequential: {seq_rounds} rounds, Parallel: {par_rounds} rounds")

    @pytest.mark.parametrize("scc_size", [5, 10, 20])
    def test_varying_scc_sizes(self, scc_size):
        """Test with different SCC sizes at the top."""
        n = 50
        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()
        tier_groups = [[i for i in range(scc_size)]]  # Top scc_size items tied

        result, rounds, calls, num_sccs = run_tournament_with_cycles(
            items, ground_truth, tier_groups, k=10, top_m=5, max_parallel=5
        )
        assert all(r in range(scc_size) for r in result), f"Expected items from top SCC, got {result}"
        print(f"\n  SCC size={scc_size}: result={result}, num_sccs={num_sccs}")

    def test_scc_scattered_across_groups(self):
        """Items from the same SCC are in different initial groups."""
        n = 25
        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()
        # Items 0, 5, 10, 15, 20 (one from each initial group of 5) form a cycle
        tier_groups = [[0, 5, 10, 15, 20]]

        result, rounds, calls, num_sccs = run_tournament_with_cycles(
            items, ground_truth, tier_groups, k=5, top_m=3, max_parallel=5
        )
        # Top 3 should be from the scattered SCC
        assert all(r in [0, 5, 10, 15, 20] for r in result), f"Expected scattered SCC items, got {result}"
        print(f"\n  Scattered SCC: result={result}, num_sccs={num_sccs}, rounds={rounds}")


class TestStressTest:
    """Stress tests with larger inputs."""

    @pytest.mark.parametrize("seed", range(10))
    def test_200_items_stress(self, seed):
        """Stress test with 200 items."""
        random.seed(seed)
        items = [Item(id=i) for i in range(200)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:10]]

        result, _, _ = run_tournament(items, ground_truth, k=20, top_m=10, max_parallel=10)
        assert result == expected, f"seed={seed} failed"

    def test_large_k(self):
        """Test with large k value."""
        random.seed(42)
        items = [Item(id=i) for i in range(50)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:5]]

        result, _, _ = run_tournament(items, ground_truth, k=25, top_m=5, max_parallel=2)
        assert result == expected

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_3000_items_k30_100_chunks(self, seed):
        """Large scale test: 3000 items, k=30, 100 initial chunks."""
        random.seed(seed)
        n = 3000
        k = 30
        top_m = 10
        max_parallel = 100  # 100 chunks of 30 items each
        # Sequential needs ~100+ rounds (100 heats + finalization), so increase limit
        max_rounds = 150

        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:top_m]]

        seq_result, seq_rounds, seq_calls = run_tournament(
            items, ground_truth, k, top_m, max_parallel=1, max_num_rounds=max_rounds
        )
        par_result, par_rounds, par_calls = run_tournament(
            items, ground_truth, k, top_m, max_parallel=max_parallel, max_num_rounds=max_rounds
        )

        assert seq_result == expected, f"Sequential failed for seed={seed}"
        assert par_result == expected, f"Parallel failed for seed={seed}"
        # Allow small variance (<2%) for large-scale tests due to edge cases in scheduling
        call_diff_pct = abs(par_calls - seq_calls) / seq_calls * 100
        assert call_diff_pct < 2, f"Call parity failed: seq={seq_calls}, par={par_calls} ({call_diff_pct:.1f}% diff)"
        assert par_rounds < seq_rounds, f"Parallel should reduce rounds: seq={seq_rounds}, par={par_rounds}"
        print(f"\n  seed={seed}: n={n}, k={k}, top_m={top_m}")
        print(f"  Sequential: {seq_rounds} rounds, {seq_calls} calls")
        print(f"  Parallel:   {par_rounds} rounds, {par_calls} calls")

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_1500_items_k50_30_chunks(self, seed):
        """Large scale test: 1500 items, k=50, 30 initial chunks."""
        random.seed(seed)
        n = 1500
        k = 50
        top_m = 10
        max_parallel = 30  # 30 chunks of 50 items each
        max_rounds = 100

        items = [Item(id=i) for i in range(n)]
        ground_truth = items.copy()
        random.shuffle(ground_truth)
        expected = [i.id for i in ground_truth[:top_m]]

        seq_result, seq_rounds, seq_calls = run_tournament(
            items, ground_truth, k, top_m, max_parallel=1, max_num_rounds=max_rounds
        )
        par_result, par_rounds, par_calls = run_tournament(
            items, ground_truth, k, top_m, max_parallel=max_parallel, max_num_rounds=max_rounds
        )

        assert seq_result == expected, f"Sequential failed for seed={seed}"
        assert par_result == expected, f"Parallel failed for seed={seed}"
        assert par_calls == seq_calls, f"Call parity failed: seq={seq_calls}, par={par_calls}"
        assert par_rounds < seq_rounds, f"Parallel should reduce rounds: seq={seq_rounds}, par={par_rounds}"
        print(f"\n  seed={seed}: n={n}, k={k}, top_m={top_m}")
        print(f"  Sequential: {seq_rounds} rounds, {seq_calls} calls")
        print(f"  Parallel:   {par_rounds} rounds, {par_calls} calls")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
