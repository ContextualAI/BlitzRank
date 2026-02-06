"""
Use NetworkX for streaming tournament graph construction and query processing.
"""

import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any


@dataclass
class RoundOutput:
    """
    Output after processing a round of edges.
    This is used by the algorithm to determine whether we are done and can output the top m nodes
    or whether to schedule additional matches to add edges to the tournament graph.
    """

    # Original graph degrees
    in_degree: Dict[Any, int]  # node -> in-degree in G
    out_degree: Dict[Any, int]  # node -> out-degree in G

    # Condensation info
    scc_membership: Dict[Any, int]  # B(u): node -> SCC index in G*
    scc_members: Dict[int, Set[Any]]  # SCC index -> set of nodes in that SCC
    scc_sizes: List[int]  # List of SCC sizes (sorted descending)
    num_sccs: int  # Number of SCCs

    # Transitive closure of condensation degrees
    out_reach: Dict[Any, int]  # nodes in SCCs reachable FROM u's SCC
    in_reach: Dict[Any, int]  # nodes in SCCs that can REACH u's SCC

    known_relationships: Dict[
        Any, int
    ]  # nodes that beat $u$, nodes that $u$ beats, and the $|B(u)| - 1$ other nodes tied with $u$ in the same SCC.

    num_3_cycles: int  # Count of 3-cycles (0 = perfectly transitive)


class TournamentGraph:
    """
    Processes a tournament graph built incrementally across rounds.

    At each round:
    1. Add new edges to the graph
    2. Compute condensation G* and transitive closure of G*
    3. Return degree information for original and condensation graphs
    """

    def __init__(self, nodes: List[Any] = None, enforce_tournament: bool = True):
        """
        Initialize with optional list of nodes.
        Nodes can also be added implicitly when edges are added.
        enforce_tournament: whether to enforce that the graph is a tournament -- i.e., no parallel edges.
        """
        self.enforce_tournament = enforce_tournament
        self.G = nx.DiGraph()
        if nodes:
            self.G.add_nodes_from(nodes)

    def _add_edges_tournament(self, edges: List[Tuple[Any, Any]]) -> None:
        for u, v in edges:
            if u == v:
                raise ValueError(
                    f"Self-loop edge ({u}, {v}) is not allowed in a tournament."
                )
            if self.G.has_edge(v, u) and self.enforce_tournament:
                raise ValueError(
                    f"Contradictory edge ({u}, {v}): reverse edge ({v}, {u}) already exists."
                )
            self.G.add_edge(u, v)

    def _count_3_cycles(self) -> int:
        """
        Count directed 3-cycles in the current graph.

        Returns:
            num_3_cycles
        """
        n = self.G.number_of_nodes()

        if n < 3:
            return 0

        adj = {u: set(self.G.successors(u)) for u in self.G.nodes()}
        cycles = 0
        for u, u_neighbors in adj.items():
            for v in u_neighbors:
                for w in adj.get(v, ()):
                    if u in adj.get(w, ()):
                        cycles += 1

        num_3_cycles = cycles // 3
        return num_3_cycles

    def process_round(self, edges: List[Tuple[Any, Any]]) -> RoundOutput:
        """
        Add edges from this round and compute all required outputs.

        Args:
            edges: List of (u, v) meaning u beats v (edge from u to v)

        Returns:
            RoundOutput with all degree information
        """
        # Step 1: Add edges to graph
        self._add_edges_tournament(edges)

        # Step 3: Compute condensation G*
        condensation = nx.condensation(self.G)

        # B(u): node -> SCC index
        scc_membership = condensation.graph["mapping"]

        # SCC index -> set of member nodes
        scc_members = {
            scc_idx: condensation.nodes[scc_idx]["members"]
            for scc_idx in condensation.nodes()
        }

        # SCC sizes (sorted descending)
        scc_sizes = sorted(
            [len(members) for members in scc_members.values()], reverse=True
        )
        num_sccs = len(scc_sizes)

        # Step 2: Compute original graph degrees and reach metrics
        in_degree = dict(self.G.in_degree())
        out_degree = dict(self.G.out_degree())
        out_reach = {}
        in_reach = {}
        known_relationships = {}
        for node in self.G.nodes():
            n_other_nodes_in_scc = len(scc_members[scc_membership[node]]) - 1
            out_reach[node] = len(
                nx.descendants(self.G, node)
            ) - n_other_nodes_in_scc  # all nodes reachable from node
            in_reach[node] = len(
                nx.ancestors(self.G, node)
            ) - n_other_nodes_in_scc  # all nodes that can reach node
            known_relationships[node] = (
                out_reach[node] + in_reach[node] + n_other_nodes_in_scc
            )  # to account for double-counting the scc relationships

        num_3_cycles = self._count_3_cycles()

        return RoundOutput(
            in_degree=in_degree,
            out_degree=out_degree,
            scc_membership=scc_membership,
            scc_members=scc_members,
            scc_sizes=scc_sizes,
            num_sccs=num_sccs,
            out_reach=out_reach,
            in_reach=in_reach,
            known_relationships=known_relationships,
            num_3_cycles=num_3_cycles,
        )

    def get_graph(self) -> nx.DiGraph:
        """Return the current graph for inspection."""
        return self.G


def main():
    # test code
    pass
