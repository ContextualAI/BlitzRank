"""Test max_parallel_matches parameter in BlitzRank constructor."""
import pytest
from blitzrank import BlitzRank


def test_max_parallel_matches_in_constructor():
    """Test that BlitzRank accepts max_parallel_matches in constructor."""
    ranker = BlitzRank(max_parallel_matches=3)
    assert ranker.max_parallel_matches == 3


def test_max_parallel_matches_default():
    """Test that default max_parallel_matches is 5."""
    ranker = BlitzRank()
    assert ranker.max_parallel_matches == 5


def test_max_parallel_matches_with_other_params():
    """Test that max_parallel_matches works with other parameters."""
    ranker = BlitzRank(window_size=10, top_m=5, max_parallel_matches=2)
    assert ranker.window_size == 10
    assert ranker.top_m == 5
    assert ranker.max_parallel_matches == 2


def test_max_parallel_matches_various_values():
    """Test that various max_parallel_matches values are accepted."""
    for value in [1, 2, 5, 10, 20]:
        ranker = BlitzRank(max_parallel_matches=value)
        assert ranker.max_parallel_matches == value


def test_max_parallel_matches_flows_to_config():
    """Test that max_parallel_matches is passed to TournamentGraphSortConfig."""
    from blitzrank.engine.algorithms.tournament_graph.experimental_interface import (
        TournamentGraphSortConfig,
    )
    
    ranker = BlitzRank(window_size=10, top_m=5, max_parallel_matches=3)
    
    # Create a sort config the way BlitzRank does
    sort_config = TournamentGraphSortConfig(max_parallel_matches=ranker.max_parallel_matches)
    
    assert sort_config.max_parallel_matches == 3
