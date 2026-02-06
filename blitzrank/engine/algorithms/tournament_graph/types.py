from dataclasses import dataclass
from typing import Awaitable, Callable, List, Tuple

from .tournament_graph import TournamentGraph
from .compare_oracle import Item
from enum import Enum


@dataclass
class SCCStats:
    """SCC statistics for the final tournament graph state."""

    # Global stats
    num_3_cycles: int  # direct measure of non-transitivity (0 = perfectly transitive)
    num_sccs: int  # total SCCs (n SCCs = fully transitive with n singletons)
    num_non_trivial_sccs: int  # SCCs with size > 1
    total_nodes_in_non_trivial_sccs: int  # nodes "tied" with at least one other
    max_scc_size: int
    max_scc_size_rank: int  # rank (0-indexed) of the first node in the largest SCC

    # Top-m stats
    max_scc_size_top_m: int
    avg_scc_size_top_m: float
    num_sccs_top_m: int  # distinct SCCs among top m


@dataclass
class TournamentGraphSortResult:
    results: List[Item]
    num_rounds: int
    num_oracle_calls: int
    graph: TournamentGraph  # the tournament graph that was built during the sorting process

    def get_scc_stats(self) -> SCCStats:
        """Compute SCC statistics from the final graph state."""
        # Get current graph state by calling process_round with no new edges
        round_output = self.graph.process_round([])

        # Global stats
        scc_sizes_list = round_output.scc_sizes  # sorted descending
        num_3_cycles = round_output.num_3_cycles
        num_sccs = round_output.num_sccs
        num_non_trivial_sccs = sum(1 for size in scc_sizes_list if size > 1)
        total_nodes_in_non_trivial_sccs = sum(s for s in scc_sizes_list if s > 1)
        max_scc_size = scc_sizes_list[0] if scc_sizes_list else 0

        # Find rank of first node in largest SCC
        # Sort nodes by (in_reach, out_reach) to match scheduling order
        nodes = list(self.graph.G.nodes())
        sorted_nodes = sorted(
            nodes,
            key=lambda n: (
                round_output.in_reach[n],
                round_output.out_reach[n],
            ),
        )

        # Find the largest SCC(s)
        largest_scc_indices = {
            scc_idx
            for scc_idx, members in round_output.scc_members.items()
            if len(members) == max_scc_size
        }
        max_scc_size_rank = next(
            (
                rank
                for rank, node in enumerate(sorted_nodes)
                if round_output.scc_membership[node] in largest_scc_indices
            ),
            -1,
        )

        # Top-m stats: analyze SCCs of the finalized results
        top_m = len(self.results)
        if top_m > 0:
            top_m_scc_sizes = [
                len(round_output.scc_members[round_output.scc_membership[item]])
                for item in self.results
            ]
            top_m_sccs = {round_output.scc_membership[item] for item in self.results}
            max_scc_size_top_m = max(top_m_scc_sizes)
            avg_scc_size_top_m = sum(top_m_scc_sizes) / top_m
            num_sccs_top_m = len(top_m_sccs)
        else:
            max_scc_size_top_m = 0
            avg_scc_size_top_m = 0.0
            num_sccs_top_m = 0

        return SCCStats(
            num_3_cycles=num_3_cycles,
            num_sccs=num_sccs,
            num_non_trivial_sccs=num_non_trivial_sccs,
            total_nodes_in_non_trivial_sccs=total_nodes_in_non_trivial_sccs,
            max_scc_size=max_scc_size,
            max_scc_size_rank=max_scc_size_rank,
            max_scc_size_top_m=max_scc_size_top_m,
            avg_scc_size_top_m=avg_scc_size_top_m,
            num_sccs_top_m=num_sccs_top_m,
        )


class SortStrategy(Enum):
    """
    Sort strategy for selecting the next set of players to play a match.
    """

    # ascending/descending out-reach: the number of nodes that can reach the node
    # note: this doesn't work with non-transitive setups.
    ASCENDING_OUT_REACH = "ascending_out_reach"
    DESCENDING_OUT_REACH = "descending_out_reach"


DEFAULT_SORT_STRATEGY = SortStrategy.ASCENDING_OUT_REACH
DEFAULT_MAX_NUM_ROUNDS = 50


@dataclass
class TournamentGraphSortConfig:
    sort_strategy: SortStrategy = DEFAULT_SORT_STRATEGY
    enforce_tournament: bool = False  # whether to enforce that the graph is a tournament -- i.e., no parallel edges.
    on_round_complete: Callable[["RoundLog"], Awaitable[None] | None] | None = None

    max_num_rounds: int = DEFAULT_MAX_NUM_ROUNDS


@dataclass
class TournamentProgress:
    nodes_for_match: List[Item]
    finalized_nodes: List[Item]
    sorted_node_infos: List["NodeInfo"]


@dataclass
class NodeInfo:
    node: Item

    # inclusive reach: the number of nodes that can reach the node + other nodes in the same SCC.
    # this reduces to in_reach and out_reach for transitive setups.
    in_reach: int
    out_reach: int
    known_relationships: int

    scc_group: set[Item]  # the set of nodes that belong to the same SCC as the node.


@dataclass
class RoundLog:
    round_idx: int
    sorted_node_infos: List[dict]
    finalized_nodes: List[str]
    round_output: dict
    oracle_metadata: dict | None
    nodes_for_match: List[str]
    edges_added: List[Tuple[str, str]]
