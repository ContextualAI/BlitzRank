from abc import ABC, abstractmethod

from .engine.config import ComparerConfig, ComparerType, RerankingConfig, SelectorConfig
from .engine.comparer import create_comparer


class Ranker(ABC):
    @abstractmethod
    async def __call__(self, query: str, docs: list[str], topk: int, model: str) -> list[int]:
        """Return topk doc indices sorted by relevance (most relevant first)."""
        ...


def _make_hits(docs):
    return [
        {"content": d, "qid": "0", "docid": str(i), "rank": i + 1, "score": len(docs) - i}
        for i, d in enumerate(docs)
    ]


def _make_task(query, docs):
    from .engine.algorithms.acurank_selectors import RerankTask, SingleContent
    contents = [
        SingleContent(content=d, qid="0", docid=str(i), rank=i + 1, score=len(docs) - i, orig_idx=i)
        for i, d in enumerate(docs)
    ]
    return RerankTask(query=query, contents=contents, hits=_make_hits(docs))


class BlitzRank(Ranker):
    def __init__(self, window_size: int = 20, top_m: int = 10):
        self.window_size = window_size
        self.top_m = top_m

    async def __call__(self, query, docs, topk, model):
        from .engine.algorithms.tournament_graph.experimental_interface import (
            rerank_with_tournament_graph, RerankTask, SingleContent,
            LLMCompareOracle, TournamentGraphRerankConfig,
        )
        ws = min(self.window_size, len(docs))
        oracle = LLMCompareOracle(ws, ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model))
        hits = _make_hits(docs)
        contents = [
            SingleContent(content=d, qid="0", docid=str(i), rank=i + 1, score=len(docs) - i, orig_idx=i)
            for i, d in enumerate(docs)
        ]
        task = RerankTask(query=query, contents=contents, hits=hits)
        result = await rerank_with_tournament_graph(
            task, TournamentGraphRerankConfig(top_m=min(topk, len(docs)), oracle=oracle)
        )
        return [item.content.orig_idx for item in result.results][:topk]

    def _build_reranking_config(self, model):
        return RerankingConfig(
            selector=SelectorConfig(type="tournament_graph", window_size=self.window_size, top_m=self.top_m),
            comparer=ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model),
        )


class SlidingWindow(Ranker):
    def __init__(self, window_size: int = 20, step: int = 10, num_rounds: int = 1, rank_end: int = 100):
        self.window_size = window_size
        self.step = step
        self.num_rounds = num_rounds
        self.rank_end = rank_end

    async def __call__(self, query, docs, topk, model):
        from .engine.reranker import SlidingWindowSelector
        comparer = create_comparer(ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model))
        ws = min(self.window_size, len(docs))
        selector = SlidingWindowSelector(
            _make_hits(docs), self.rank_end, ws, self.step, self.num_rounds
        )
        final_indices, _ = await selector.run(query, comparer)
        return final_indices[:topk]

    def _build_reranking_config(self, model):
        return RerankingConfig(
            selector=SelectorConfig(
                type="sliding_window", window_size=self.window_size,
                step=self.step, num_rounds=self.num_rounds, rank_end=self.rank_end,
            ),
            comparer=ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model),
        )


class SetWise(Ranker):
    def __init__(self, sorting_method: str = "heapsort", num_child: int = 2, top_m: int = 10):
        self.sorting_method = sorting_method
        self.num_child = num_child
        self.top_m = top_m

    async def __call__(self, query, docs, topk, model):
        from .engine.algorithms.setwise_pairwise_selectors import (
            setwise_heapsort, setwise_bubblesort, SetwiseConfig,
        )
        comparer = create_comparer(ComparerConfig(type=ComparerType.SETWISE, model=model))
        config = SetwiseConfig(top_m=min(topk, len(docs)), num_child=self.num_child, method=self.sorting_method)
        fn = setwise_heapsort if self.sorting_method == "heapsort" else setwise_bubblesort
        result, _ = await fn(_make_task(query, docs), comparer, config)
        return [c.orig_idx for c in result][:topk]

    def _build_reranking_config(self, model):
        return RerankingConfig(
            selector=SelectorConfig(
                type="setwise", sorting_method=self.sorting_method,
                num_child=self.num_child, top_m=self.top_m,
            ),
            comparer=ComparerConfig(type=ComparerType.SETWISE, model=model),
        )


class PairWise(Ranker):
    def __init__(self, sorting_method: str = "heapsort", top_m: int = 10):
        self.sorting_method = sorting_method
        self.top_m = top_m

    async def __call__(self, query, docs, topk, model):
        from .engine.algorithms.setwise_pairwise_selectors import (
            pairwise_heapsort, pairwise_bubblesort, pairwise_allpair, PairwiseConfig,
        )
        comparer = create_comparer(ComparerConfig(type=ComparerType.PAIRWISE, model=model))
        config = PairwiseConfig(top_m=min(topk, len(docs)), method=self.sorting_method)
        fn = {"heapsort": pairwise_heapsort, "bubblesort": pairwise_bubblesort, "allpair": pairwise_allpair}[self.sorting_method]
        result, _ = await fn(_make_task(query, docs), comparer, config)
        return [c.orig_idx for c in result][:topk]

    def _build_reranking_config(self, model):
        return RerankingConfig(
            selector=SelectorConfig(
                type="pairwise", sorting_method=self.sorting_method, top_m=self.top_m,
            ),
            comparer=ComparerConfig(type=ComparerType.PAIRWISE, model=model),
        )


class TourRank(Ranker):
    def __init__(self, num_rounds: int = 1, window_size: int = 20):
        self.num_rounds = num_rounds
        self.window_size = window_size

    async def __call__(self, query, docs, topk, model):
        from .engine.algorithms.acurank_selectors import tourrank_rerank_single, TourRankConfig
        comparer = create_comparer(ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model))
        ws = min(self.window_size, len(docs))
        config = TourRankConfig(num_rounds=self.num_rounds, window_size=ws)
        result, _ = await tourrank_rerank_single(_make_task(query, docs), comparer, config)
        return [c.orig_idx for c in result][:topk]

    def _build_reranking_config(self, model):
        return RerankingConfig(
            selector=SelectorConfig(
                type="tourrank", window_size=self.window_size, num_rounds=self.num_rounds,
            ),
            comparer=ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model),
        )


class AcuRank(Ranker):
    def __init__(self, window_size: int = 20, tol: float = 1e-2, hard_constraint: int = 100,
                 uncertain_U: int = 10, R: int = 10, break_mode: str = "reduce_uncertain"):
        self.window_size = window_size
        self.tol = tol
        self.hard_constraint = hard_constraint
        self.uncertain_U = uncertain_U
        self.R = R
        self.break_mode = break_mode

    async def __call__(self, query, docs, topk, model):
        from .engine.algorithms.acurank_selectors import acurank_rerank_single, AcuRankConfig
        ws = min(self.window_size, len(docs))
        comparer = create_comparer(ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model))
        config = AcuRankConfig(
            window_size=ws, tol=self.tol, hard_constraint=self.hard_constraint,
            uncertain_U=self.uncertain_U, R=self.R, break_mode=self.break_mode,
        )
        result, _ = await acurank_rerank_single(_make_task(query, docs), comparer, config, None, 0)
        return [c.orig_idx for c in result][:topk]

    def _build_reranking_config(self, model):
        return RerankingConfig(
            selector=SelectorConfig(
                type="acurank", window_size=self.window_size, tol=self.tol,
                hard_constraint=self.hard_constraint, uncertain_U=self.uncertain_U,
                R=self.R, break_mode=self.break_mode,
            ),
            comparer=ComparerConfig(type=ComparerType.LISTWISE_RANK_GPT, model=model),
        )
