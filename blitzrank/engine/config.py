from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import yaml
from enum import Enum


class ComparerType(Enum):
    LISTWISE_RANK_GPT = "listwise_rank_gpt"
    SETWISE = "setwise"
    PAIRWISE = "pairwise"
    CTXL_API = "ctxl_api"


@dataclass
class DatasetConfig:
    type: str
    name: str
    index: str = ""
    k: int = 100
    subset_size: Optional[int] = None
    subset_seed: int = 42
    shuffle_seed: Optional[int] = None
    long_context: bool = False
    rewrite_queries: bool = False
    perturb_dataset: str = ""

    @property
    def short_name(self) -> str:
        parts = self.name.replace(".", "-").split("-")
        name = parts[-2] if len(parts) >= 2 else self.name
        components = [name, f"k{self.k}"]
        if self.subset_size:
            components.append(f"sub{self.subset_size}_seed{self.subset_seed}")
        if self.shuffle_seed is not None:
            components.append(f"shuf{self.shuffle_seed}")
        if self.long_context:
            components.append("longctx")
        if self.rewrite_queries:
            components.append("rewrite")
        if self.perturb_dataset:
            components.append(f"pert_{self.perturb_dataset}")
        return "_".join(components)


@dataclass
class SelectorConfig:
    type: str
    rank_end: int = 100
    window_size: int = 20
    step: int = 10
    num_child: int = 2  # For setwise: compare num_child+1 docs at a time
    sorting_method: str = "heapsort"  # "heapsort", "bubblesort", or "allpair" (pairwise only)
    # AcuRank parameters
    tol: float = 1e-2
    hard_constraint: int = 100
    uncertain_U: int = 10
    R: int = 10
    break_mode: str = "reduce_uncertain"  # or "top10_nochange"
    # TourRank parameters
    num_rounds: int = 1
    # Number of top documents to rank. Used by tournament_graph, setwise, and pairwise selectors:
    #   - bubblesort: runs top_m outer iterations to bubble up top positions
    #   - heapsort: extracts top_m elements from heap then stops
    #   - allpair: compares all pairs, but only top_m get explicit scores in output
    top_m: int = 10

    @property
    def short_name(self) -> str:
        if self.type == "single_pass":
            return f"singlepass_top{self.rank_end}"
        elif self.type == "sliding_window":
            if self.num_rounds == 1:
                return f"sliding_w{self.window_size}_s{self.step}_end{self.rank_end}"
            else:
                return f"sliding_w{self.window_size}_s{self.step}_end{self.rank_end}_r{self.num_rounds}"
        elif self.type == "tournament":
            return f"tournament_w{self.window_size}_top{self.top_m}"
        elif self.type == "tournament_graph":
            return f"tournament_graph_k{self.window_size}_top{self.top_m}"
        elif self.type == "acurank":
            return f"acurank_w{self.window_size}_tol{self.tol}_hc{self.hard_constraint}"
        elif self.type == "tourrank":
            return f"tourrank_w{self.window_size}_r{self.num_rounds}"
        elif self.type == "setwise":
            return f"setwise_{self.sorting_method}_k{self.top_m}_nc{self.num_child}"
        elif self.type == "pairwise":
            return f"pairwise_{self.sorting_method}_k{self.top_m}"
        return self.type


@dataclass
class ComparerConfig:
    type: ComparerType
    model: str
    max_doc_tokens: int = 1024
    max_query_tokens: int = 1024
    max_concurrent_requests: int = 3
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    relevance_instruction: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = ComparerType(self.type)

    @property
    def short_name(self) -> str:
        model_short = self.model.split("/")[-1].lower()
        return f"{self.type.value}_{model_short}_tok{self.max_doc_tokens}"


@dataclass
class RerankingConfig:
    selector: SelectorConfig
    comparer: ComparerConfig
    max_parallel_requests: int = 5

    @property
    def short_name(self) -> str:
        return f"{self.selector.short_name}_{self.comparer.short_name}"


@dataclass
class EvaluationConfig:
    type: str
    metrics: List[str]


@dataclass
class LoggingConfig:
    output_dir: str
    experiment_name: str

    # a summary file contains summary metrics for all experiments. useful for comparing different configs.
    summary_metrics_path: str = "summary.jsonl"

    enable_wandb: bool = True
    wandb_project: str = "rerank-listwise"

    # Whether to build and log the per-round `rounds_table` to W&B at all.
    # If you see `wandb.finish()` taking a long time due to artifact uploads (e.g. `...-rounds_table`),
    # set this to False to disable the table entirely while keeping scalar metrics.
    log_round_table_to_wandb: bool = True

    # Whether to log the `rounds_table` on every round (old behavior). This can be very slow because
    # W&B persists tables via artifacts and repeated updates create many upload tasks.
    # Default False: the table is logged once at the end of the run.
    log_round_table_every_round: bool = False
    sync_to_remote: bool = False


@dataclass
class Config:
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    reranking: Optional[RerankingConfig] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Config":
        dataset = DatasetConfig(**config["dataset"])
        evaluation = EvaluationConfig(**config["evaluation"])
        logging_config = LoggingConfig(**config["logging"])

        reranking = None
        if "reranking" in config:
            reranking_config = config["reranking"]
            selector = SelectorConfig(**reranking_config["selector"])
            comparer = ComparerConfig(**reranking_config["comparer"])
            reranking = RerankingConfig(
                selector=selector,
                comparer=comparer,
                max_parallel_requests=reranking_config.get("max_parallel_requests", 5),
            )

            if reranking.selector.type not in [
                "single_pass",
                "sliding_window",
                "tournament",
                "tournament_graph",
                "acurank",
                "tourrank",
                "setwise",
                "pairwise",
            ]:
                raise ValueError(f"Unknown selector type: {reranking.selector.type}")
            if reranking.comparer.type not in [ComparerType.LISTWISE_RANK_GPT, ComparerType.SETWISE, ComparerType.PAIRWISE, ComparerType.CTXL_API]:
                raise ValueError(f"Unknown comparer type: {reranking.comparer.type}")
            if (
                not isinstance(reranking.max_parallel_requests, int)
                or reranking.max_parallel_requests < 1
            ):
                raise ValueError("max_parallel_requests must be a positive integer")

        if dataset.type not in ["online_bm25", "bright"]:
            raise ValueError(f"Unknown dataset type: {dataset.type}")
        if evaluation.type not in ["none", "trec_eval"]:
            raise ValueError(f"Unknown evaluation type: {evaluation.type}")
        if dataset.subset_size is not None and dataset.subset_size < 1:
            raise ValueError("subset_size must be a positive integer")

        return cls(
            dataset=dataset,
            reranking=reranking,
            evaluation=evaluation,
            logging=logging_config,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
