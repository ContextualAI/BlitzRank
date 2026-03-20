import asyncio
from typing import Dict, List, Optional, Tuple, Any

from .tournament_graph import TournamentGraph, RoundOutput
from .compare_oracle import CompareOracle, OracleResult, Item
from .types import (
    TournamentGraphSortConfig,
    TournamentGraphSortResult,
    TournamentProgress,
    NodeInfo,
    SortStrategy,
)


class TournamentGraphSort:
    def __init__(
        self,
        items: List[Item],
        oracle: CompareOracle,
        num_top_nodes_to_output: Optional[int] = None,
        tournament_graph_sort_config: Optional[TournamentGraphSortConfig] = None,
    ):
        self.items = items
        self.n = len(items)
        self.oracle = oracle
        self.tournament_graph_sort_config = (
            tournament_graph_sort_config or TournamentGraphSortConfig()
        )
        self.tournament_graph = TournamentGraph(
            items,
            enforce_tournament=self.tournament_graph_sort_config.enforce_tournament,
        )
        self.k = oracle.get_k()
        self.num_top_nodes_to_output = min(num_top_nodes_to_output or self.n, self.n)
        self.sort_strategy = self.tournament_graph_sort_config.sort_strategy
        self.on_round_complete = self.tournament_graph_sort_config.on_round_complete

        self.max_num_rounds = self.tournament_graph_sort_config.max_num_rounds
        self.max_parallel_matches = self.tournament_graph_sort_config.max_parallel_matches

    async def sort_async(self) -> TournamentGraphSortResult:
        finalized_nodes = []
        num_rounds = 0
        nodes_for_next_round = self.items[: self.k * self.max_parallel_matches]
        while len(finalized_nodes) < self.num_top_nodes_to_output:
            num_rounds += 1
            if self.max_num_rounds is not None and num_rounds > self.max_num_rounds:
                raise RuntimeError(
                    f"Max number of rounds reached: {self.max_num_rounds}. "
                    f"Finalized nodes: {finalized_nodes}"
                )
            matches = self._chunk_into_matches(nodes_for_next_round)
            (
                round_output,
                edges_added,
                oracle_results,
            ) = await self._run_parallel_matches_async(matches)
            tournament_progress = self.get_tournament_progress_for_round(round_output)
            await self._emit_round_log_async(
                self._build_round_log(
                    round_idx=num_rounds,
                    nodes_for_match=nodes_for_next_round,
                    finalized_nodes=tournament_progress.finalized_nodes,
                    sorted_node_infos=tournament_progress.sorted_node_infos,
                    round_output=round_output,
                    oracle_result=oracle_results[0] if len(oracle_results) == 1 else OracleResult(edges=edges_added),
                    edges_added=edges_added,
                )
            )

            self.check_for_infinite_loop(
                nodes_for_next_round, tournament_progress, num_rounds
            )
            nodes_for_next_round = tournament_progress.nodes_for_match
            finalized_nodes = tournament_progress.finalized_nodes
        return self.build_sort_result(finalized_nodes, num_rounds)

    def _chunk_into_matches(self, nodes: List[Item]) -> List[List[Item]]:
        """Split nodes into matches of size k."""
        return [nodes[i:i + self.k] for i in range(0, len(nodes), self.k)]

    async def _run_parallel_matches_async(
        self, matches: List[List[Item]]
    ) -> Tuple[RoundOutput, List[Tuple[Item, Item]], List[OracleResult]]:
        """Run multiple matches in parallel, collect edges, update graph once."""
        oracle_results = await asyncio.gather(
            *[self.oracle.compare_k_async(match) for match in matches]
        )
        all_edges = [edge for result in oracle_results for edge in result.edges]
        round_output = self.tournament_graph.process_round(all_edges)
        return round_output, all_edges, oracle_results

    def sort(self) -> TournamentGraphSortResult:
        """
        Main sort method.
        We keep playing matches and finalizing nodes until we have enough top m nodes.
        Returns:
        - TournamentGraphSortResult: the result of the sort
        """
        return asyncio.run(self.sort_async())

    def check_for_infinite_loop(
        self,
        nodes_for_prev_round: List[Item],
        tournament_progress: TournamentProgress,
        num_rounds: int,
    ) -> None:
        """
        Raise error if the same first match is scheduled consecutively, indicating a bug.
        With SCC-based scheduling, this should never happen: each match includes nodes
        from different SCCs, guaranteeing new cross-SCC edges are added.
        """
        prev_first_match = set(nodes_for_prev_round[: self.k])
        next_first_match = set(tournament_progress.nodes_for_match[: self.k])
        if prev_first_match == next_first_match:
            raise RuntimeError(
                f"[Round {num_rounds}] Infinite loop detected: same match scheduled consecutively. "
                f"This indicates a bug in the scheduling algorithm. Item0: {self.items[0]}"
            )

    def get_tournament_progress_for_round(
        self, round_output: RoundOutput
    ) -> TournamentProgress:
        node_infos = [
            NodeInfo(
                node=item,
                in_reach=round_output.in_reach[item],
                out_reach=round_output.out_reach[item],
                known_relationships=round_output.known_relationships[item],
                scc_group=round_output.scc_members[round_output.scc_membership[item]],
            )
            for item in self.items
        ]
        return self.get_tournament_progress_and_schedule_next_match(node_infos)

    def build_sort_result(
        self, finalized_nodes: List[Item], num_rounds: int
    ) -> TournamentGraphSortResult:
        return TournamentGraphSortResult(
            results=finalized_nodes[: self.num_top_nodes_to_output],
            num_rounds=num_rounds,
            num_oracle_calls=self.oracle.get_num_calls(),
            graph=self.tournament_graph,
        )

    def node_satisfies_finalization_criterion(self, node_info: NodeInfo) -> bool:
        """A node is finalized when its relationship to all other nodes is known."""
        return node_info.known_relationships >= self.n - 1

    def get_tournament_progress_and_schedule_next_match(
        self, node_infos: List[NodeInfo]
    ) -> TournamentProgress:
        """
        Check termination and schedule the next match (Algorithm 1).

        Termination (Algorithm 1, lines 10-14):
            T ← top-m nodes by ascending in_reach
            F ← resolved nodes (known_relationships == n-1)
            if T ⊆ F: return T

        Scheduling (Algorithm 1, lines 17-19):
            Pick one representative per unresolved SCC, ordered by ascending in_reach
            in the condensation, up to k representatives total.

        With max_parallel_matches > 1, we collect up to k * max_parallel_matches
        representatives to run multiple non-overlapping matches in parallel.
        Parallelism is capped to avoid scattering comparisons across non-competitive items.
        """
        sorted_node_infos = self.get_sorted_node_infos(node_infos)
        top_m = sorted_node_infos[: self.num_top_nodes_to_output]

        # Terminate iff every node in the current top-m is resolved
        if all(self.node_satisfies_finalization_criterion(ni) for ni in top_m):
            return TournamentProgress([], [ni.node for ni in top_m], sorted_node_infos)

        # Schedule: one representative per unresolved SCC, by ascending in_reach
        seen_sccs: set[frozenset[Item]] = set()
        representatives: List[Item] = []
        for node_info in sorted_node_infos:
            if self.node_satisfies_finalization_criterion(node_info):
                continue
            scc_key = frozenset(node_info.scc_group)
            if scc_key not in seen_sccs:
                seen_sccs.add(scc_key)
                representatives.append(node_info.node)

        # Determine how many parallel matches to run
        top_item_compared = sorted_node_infos[0].known_relationships > 0 if sorted_node_infos else False
        num_competitive = sum(1 for ni in sorted_node_infos if ni.in_reach == 0)

        if not top_item_compared:
            max_reps = min(len(representatives), self.k * self.max_parallel_matches)
        elif num_competitive > self.k:
            num_full_matches = min(num_competitive // self.k, self.max_parallel_matches)
            max_reps = num_full_matches * self.k
        else:
            max_reps = self.k

        return TournamentProgress(representatives[:max_reps], [], sorted_node_infos)

    def get_sorted_node_infos(self, node_infos: List[NodeInfo]) -> List[NodeInfo]:
        if self.sort_strategy == SortStrategy.ASCENDING_OUT_REACH:
            return sorted(node_infos, key=lambda x: (x.in_reach, x.out_reach))
        elif self.sort_strategy == SortStrategy.DESCENDING_OUT_REACH:
            return sorted(node_infos, key=lambda x: (x.in_reach, -x.out_reach))
        else:
            raise ValueError(f"Invalid sort strategy: {self.sort_strategy}")

    def _build_round_log(
        self,
        round_idx: int,
        nodes_for_match: List[Item],
        finalized_nodes: List[Item],
        edges_added: List[Tuple[Item, Item]],
        sorted_node_infos: List[NodeInfo],
        round_output: RoundOutput,
        oracle_result: OracleResult,
    ) -> Dict[str, Any]:
        return dict(
            round_idx=round_idx,
            nodes_for_match=[str(item.id) for item in nodes_for_match],
            finalized_nodes=[str(item.id) for item in finalized_nodes],
            edges_added=[(str(u.id), str(v.id)) for u, v in edges_added],
            sorted_node_infos=[
                {
                    "node_id": str(node_info.node.id),
                    "in_reach": node_info.in_reach,
                    "out_reach": node_info.out_reach,
                    "known_relationships": node_info.known_relationships,
                    "scc_group_size": len(node_info.scc_group),
                    "scc_group": [str(item.id) for item in node_info.scc_group],
                }
                for node_info in sorted_node_infos
            ],
            round_output={
                "num_sccs": round_output.num_sccs,
                "scc_sizes": round_output.scc_sizes,
                "num_3_cycles": round_output.num_3_cycles,
            },
            oracle_metadata=oracle_result.metadata,
        )

    async def _emit_round_log_async(self, round_log: Dict[str, Any]) -> None:
        if self.on_round_complete is None:
            return
        result = self.on_round_complete(round_log)
        if asyncio.iscoroutine(result):
            await result


def tournament_graph_sort(
    items: List[Item],
    oracle: CompareOracle,
    num_top_nodes_to_output: Optional[int] = None,
    tournament_graph_sort_config: Optional[TournamentGraphSortConfig] = None,
) -> TournamentGraphSortResult:
    tournament_graph_sort = TournamentGraphSort(
        items, oracle, num_top_nodes_to_output, tournament_graph_sort_config
    )
    return tournament_graph_sort.sort()


async def tournament_graph_sort_async(
    items: List[Item],
    oracle: CompareOracle,
    num_top_nodes_to_output: Optional[int] = None,
    tournament_graph_sort_config: Optional[TournamentGraphSortConfig] = None,
) -> TournamentGraphSortResult:
    tournament_graph_sort = TournamentGraphSort(
        items, oracle, num_top_nodes_to_output, tournament_graph_sort_config
    )
    return await tournament_graph_sort.sort_async()
