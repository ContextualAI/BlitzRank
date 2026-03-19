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

    # R⁺_G(u) and R⁻_G(u): reachability in G (includes same-SCC members)
    out_reach: Dict[Any, int]  # |R⁺_G(u)|: nodes reachable FROM u in G
    in_reach: Dict[Any, int]  # |R⁻_G(u)|: nodes that can REACH u in G

    known_relationships: Dict[Any, int]  # κ_G(u) = |R⁻_G(u) ∪ R⁺_G(u)|

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
        if self.G.number_of_nodes() < 3:
            return 0
        adj = {u: set(self.G.successors(u)) for u in self.G.nodes()}
        cycles = sum(
            1
            for u, u_neighbors in adj.items()
            for v in u_neighbors
            for w in adj.get(v, ())
            if u in adj.get(w, ())
        )
        return cycles // 3

    def process_round(self, edges: List[Tuple[Any, Any]]) -> RoundOutput:
        """
        Add edges from this round and compute all required outputs.

        Uses graph algorithms for efficient computation:
        1. Tarjan's/Kosaraju's algorithm via nx.condensation() for SCC decomposition
        2. Transitive closure on the condensation DAG for batch reachability
        3. Map SCC-level reach back to nodes (nodes in same SCC share reach values)

        Args:
            edges: List of (u, v) meaning u beats v (edge from u to v)

        Returns:
            RoundOutput with all degree information
        """
        self._add_edges_tournament(edges)

        # SCC decomposition via Tarjan's algorithm (O(V+E))
        condensation = nx.condensation(self.G)
        scc_membership = condensation.graph["mapping"]
        scc_members = {
            scc_idx: condensation.nodes[scc_idx]["members"]
            for scc_idx in condensation.nodes()
        }
        scc_sizes = sorted(
            [len(members) for members in scc_members.values()], reverse=True
        )

        # Transitive closure of condensation DAG for O(S^2) reachability
        # where S = number of SCCs, typically << V
        condensation_tc = nx.transitive_closure(condensation)

        # Precompute per-SCC reachability (excludes self-SCC)
        scc_out_reach = {}
        scc_in_reach = {}
        for scc_idx in condensation.nodes():
            successor_sccs = set(condensation_tc.successors(scc_idx))
            scc_out_reach[scc_idx] = sum(len(scc_members[s]) for s in successor_sccs)
            predecessor_sccs = set(condensation_tc.predecessors(scc_idx))
            scc_in_reach[scc_idx] = sum(len(scc_members[s]) for s in predecessor_sccs)

        # Map SCC-level reach back to original nodes
        # Within an SCC, all nodes are mutually reachable, so add (scc_size - 1)
        out_reach = {}
        in_reach = {}
        known_relationships = {}
        for node in self.G.nodes():
            scc = scc_membership[node]
            scc_size = len(scc_members[scc])
            out_reach[node] = scc_out_reach[scc] + (scc_size - 1)
            in_reach[node] = scc_in_reach[scc] + (scc_size - 1)
            # known = predecessor nodes + successor nodes + other SCC members (counted once)
            known_relationships[node] = scc_out_reach[scc] + scc_in_reach[scc] + (scc_size - 1)

        return RoundOutput(
            in_degree=dict(self.G.in_degree()),
            out_degree=dict(self.G.out_degree()),
            scc_membership=scc_membership,
            scc_members=scc_members,
            scc_sizes=scc_sizes,
            num_sccs=len(scc_sizes),
            out_reach=out_reach,
            in_reach=in_reach,
            known_relationships=known_relationships,
            num_3_cycles=self._count_3_cycles(),
        )

    def get_graph(self) -> nx.DiGraph:
        """Return the current graph for inspection."""
        return self.G
