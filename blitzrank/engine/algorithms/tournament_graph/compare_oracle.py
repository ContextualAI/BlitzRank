from dataclasses import dataclass
from typing import List, Tuple, Union
from abc import ABC


class Item:
    def __init__(self, id: Union[int, str]):
        self.id = id

    def __hash__(self) -> int:
        return hash(str(self.id))

    def __str__(self) -> str:
        return f"Item(id={self.id})"

    def __repr__(self) -> str:
        return f"Item(id={self.id})"


@dataclass
class OracleResult:
    edges: List[Tuple[Item, Item]]
    metadata: dict | None = None


def make_edges_from_linear_order(ordered_items: List[Item]) -> List[Tuple[Item, Item]]:
    """Convert a linear ordering of items to adjacent-pair edges."""
    return [(ordered_items[i], ordered_items[i + 1]) for i in range(len(ordered_items) - 1)]


class CompareOracle(ABC):
    """
    Comparator oracle: compare at most k items at a time.
    Returns edges representing pairwise comparisons between items.
    """

    def __init__(self, k: int):
        self.k = k
        self.calls = 0

    async def compare_k_async(self, items: List[Item]) -> OracleResult:
        self.calls += 1
        return await self._compare_k_async(items)

    def compare_k(self, items: List[Item]) -> OracleResult:
        self.calls += 1
        return self._compare_k(items)

    def _compare_k(self, items: List[Item]) -> OracleResult:
        raise NotImplementedError(
            "Subclasses must implement _compare_k to compare k items."
        )

    def _compare_k_async(self, items: List[Item]) -> OracleResult:
        raise NotImplementedError(
            "Subclasses must implement _compare_k_async to compare k items asynchronously."
        )

    def reset_calls(self) -> None:
        self.calls = 0

    def get_num_calls(self) -> int:
        return self.calls

    def get_k(self) -> int:
        return self.k
