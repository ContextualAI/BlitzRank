from blitzrank.engine.algorithms.tournament_graph.tournament_graph_sort import (
    tournament_graph_sort,
    TournamentGraphSortResult,
    TournamentGraphSortConfig,
)
from blitzrank.engine.algorithms.tournament_graph.toy_setup import (
    make_shuffled_items,
    Item,
    ToyCompareOracle,
)
from typing import List
import pytest
import random


def _verify_top_m(
    tournament_graph_sort_result: TournamentGraphSortResult, m: int
) -> None:
    """Verify results contain exactly the top-m items (ids 0..m-1) in correct order."""
    results = tournament_graph_sort_result.results
    assert len(results) == m, f"Expected {m} results, got {len(results)}"
    result_ids = [item.id for item in results]
    assert result_ids == list(range(m)), f"Expected {list(range(m))}, got {result_ids}"


def test_25_horses_puzzle_7_oracle_calls():
    n = 25
    k = 5
    m = 3
    oracle = ToyCompareOracle(k)
    items: List[Item] = make_shuffled_items(n)

    result = tournament_graph_sort(items, oracle, m)
    assert len(result.results) >= m
    for i in range(3):
        assert result.results[i].id == i

    assert result.num_oracle_calls == 7


def test_n_equals_k_single_match() -> None:
    """When n=k, a single match should suffice for any m."""
    n, k, m = 5, 5, 3
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(make_shuffled_items(n), oracle, m)
    _verify_top_m(results, m)
    assert oracle.get_num_calls() == 1


@pytest.mark.parametrize(
    "n,k,m",
    [
        (10, 5, 1),
        (10, 5, 5),
        (10, 5, 10),
        (20, 5, 3),
        (20, 5, 10),
        (20, 5, 20),
        (50, 5, 5),
        (50, 5, 25),
        (50, 10, 5),
        (50, 10, 25),
        (100, 5, 10),
        (100, 10, 10),
        (100, 20, 10),
        # non-divisible cases
        (11, 5, 3),
        (17, 5, 7),
        (23, 7, 5),
        (31, 10, 15),
    ],
)
def test_correctness_parameterized(n: int, k: int, m: int) -> None:
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(make_shuffled_items(n), oracle, m)
    _verify_top_m(results, m)


@pytest.mark.parametrize("seed", range(5))
def test_stress_n100_k5_m10(seed: int) -> None:
    """Stress test with multiple random seeds."""
    random.seed(seed)
    n, k, m = 100, 5, 10
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(make_shuffled_items(n), oracle, m)
    _verify_top_m(results, m)


@pytest.mark.parametrize("seed", range(5))
def test_stress_n200_k10_m20(seed: int) -> None:
    random.seed(seed)
    n, k, m = 200, 10, 20
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(
        make_shuffled_items(n),
        oracle,
        m,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=100),
    )
    _verify_top_m(results, m)


@pytest.mark.parametrize("k", [3, 5, 10, 20, 50])
def test_stress_vary_k_fixed_n_m(k: int) -> None:
    """Test algorithm behavior across different k values."""
    random.seed(42)
    n, m = 100, 15
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(
        make_shuffled_items(n),
        oracle,
        m,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=100),
    )
    _verify_top_m(results, m)


@pytest.mark.parametrize("m", [1, 5, 10, 25, 50, 100])
def test_stress_vary_m_fixed_n_k(m: int) -> None:
    """Test algorithm behavior across different m values."""
    random.seed(42)
    n, k = 100, 10
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(
        make_shuffled_items(n),
        oracle,
        m,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=100),
    )
    _verify_top_m(results, m)


def test_stress_full_sort_large() -> None:
    """Full sort of a large list."""
    random.seed(42)
    n, k, m = 200, 10, 200
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(
        make_shuffled_items(n),
        oracle,
        m,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=100),
    )
    _verify_top_m(results, m)


@pytest.mark.parametrize("seed", range(5))
def test_stress_n100_k5_full_sort(seed: int) -> None:
    """Stress test with multiple random seeds."""
    random.seed(seed)
    n, k = 100, 5
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(
        make_shuffled_items(n),
        oracle,
        n,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=None),
    )
    _verify_top_m(results, n)


# --------------------------------------------------------------------------- #
# Random Configuration Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(10))
def test_random_configurations(seed: int) -> None:
    """Generate random valid (n, k, m) configurations and verify correctness."""
    rng = random.Random(seed)
    n = rng.randint(10, 200)
    k = rng.randint(2, min(n, 30))
    m = rng.randint(1, n)

    random.seed(seed)
    oracle = ToyCompareOracle(k)
    results = tournament_graph_sort(
        make_shuffled_items(n),
        oracle,
        m,
        tournament_graph_sort_config=TournamentGraphSortConfig(max_num_rounds=100),
    )
    _verify_top_m(results, m)
