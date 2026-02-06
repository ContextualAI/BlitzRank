import asyncio
import argparse
import time
import yaml
from datetime import datetime
from dotenv import load_dotenv

from .config import Config
from .logger import ExperimentLogger
from .dataset import load_dataset
from .reranker import rerank_results
from .evaluator import evaluate_results
from .algorithms.tournament_graph.experimental_interface import (
    tournament_graph_rerank_dataset_from_config,
)
from .algorithms.acurank_selectors import (
    acurank_rerank_dataset_from_config,
    tourrank_rerank_dataset_from_config,
)
from .algorithms.setwise_pairwise_selectors import (
    setwise_rerank_dataset_from_config,
    pairwise_rerank_dataset_from_config,
)
from .utils.metadata_utils import build_reranking_metrics
import warnings

# Ignore Pydantic serializer warnings due to LiteLLM's contains provider-specific classes (OpenAI/Anthropic/Gemini SDK objects) that don't perfectly match the wrapper's Pydantic types
warnings.filterwarnings(
    "ignore",
    message=r"^Pydantic serializer warnings:.*",
    category=UserWarning,
    module=r"pydantic\..*",
)


def _evaluate_and_log_metrics(
    dataset: dict,
    config: Config,
    logger: ExperimentLogger,
    extra_metrics: dict | None = None,
) -> None:
    metrics = evaluate_results(
        dataset, config.evaluation, logger.get_output_dir(), logger
    )
    if metrics:
        if extra_metrics:
            metrics.update(extra_metrics)
        logger.log_metrics(metrics)


def results_for_run_is_available(config: Config, logger: ExperimentLogger) -> bool:
    """
    Returns True if the results.trec file exists and the metrics are already logged.
    If the results.trec file exists but the metrics are not logged, it will recompute the metrics and log the summary.
    If the results.trec file does not exist, it will return False.
    """
    output_path = logger.get_output_dir()
    trec_file = output_path / "results.trec"
    if not trec_file.exists():
        return False

    if logger.summary_metrics_already_logged():
        logger.warning(
            f"Skipping: results.trec already exists in {output_path} and metrics are already logged"
        )
        return True

    logger.warning(
        f"Existing results.trec found in {output_path}; recomputing metrics and logging summary"
    )
    dataset_raw = load_dataset(config.dataset, logger)
    _evaluate_and_log_metrics(dataset_raw, config, logger, None)
    return True


def parse_value(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def set_nested_value(d: dict, key: str, value: str):
    keys = key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = parse_value(value)


async def main(config_path: str, config_overrides: list = None):
    print("Start listwise/run.py")
    load_dotenv()

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # override config values
    if config_overrides:
        for override in config_overrides:
            key, value = override.split("=", 1)
            set_nested_value(config_data, key, value)

    # set experiment name
    if "experiment_name" not in config_data.get("logging", {}):
        if "logging" not in config_data:
            config_data["logging"] = {}
        config_data["logging"]["experiment_name"] = (
            f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # create config object and logger
    config = Config.from_dict(config_data)
    logger = ExperimentLogger(
        config.logging, config.dataset, config.reranking, config_data
    )

    try:
        # if results.trec already exists (which means the experiment was already run), skip. this helps with resuming failed experiments.
        if results_for_run_is_available(config, logger):
            return

        logger.info(f"Starting experiment: {config.logging.experiment_name}")
        logger.info(f"Config loaded from: {config_path}")

        dataset_raw = load_dataset(config.dataset, logger)

        reranking_metrics = None
        if config.reranking:
            reranking_start = time.perf_counter()
            if config.reranking.selector.type == "tournament_graph":
                dataset = await tournament_graph_rerank_dataset_from_config(
                    dataset_raw, config.reranking, logger, logger.get_output_dir()
                )
            elif config.reranking.selector.type == "acurank":
                dataset = await acurank_rerank_dataset_from_config(
                    dataset_raw, config.reranking, logger, logger.get_output_dir()
                )
            elif config.reranking.selector.type == "tourrank":
                dataset = await tourrank_rerank_dataset_from_config(
                    dataset_raw, config.reranking, logger, logger.get_output_dir()
                )
            elif config.reranking.selector.type == "setwise":
                dataset = await setwise_rerank_dataset_from_config(
                    dataset_raw, config.reranking, logger, logger.get_output_dir()
                )
            elif config.reranking.selector.type == "pairwise":
                dataset = await pairwise_rerank_dataset_from_config(
                    dataset_raw, config.reranking, logger, logger.get_output_dir()
                )
            else:
                dataset = await rerank_results(
                    dataset_raw, config.reranking, logger, logger.get_output_dir()
                )
            reranking_latency_s = time.perf_counter() - reranking_start
            if "iteration_logs" in dataset:
                logger.log_iteration_results(dataset["iteration_logs"])
            reranking_metrics = build_reranking_metrics(
                dataset.get("iteration_logs", []),
                config.reranking.comparer.model,
                reranking_latency_s,
            )
        else:
            dataset = dataset_raw

        _evaluate_and_log_metrics(dataset, config, logger, reranking_metrics)

        logger.info(f"Experiment complete. Outputs saved to: {logger.get_output_dir()}")

    except Exception as e:
        logger.exception(f"Experiment failed with error: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reranking experiments")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument(
        "--set",
        action="append",
        help="Override config values (e.g., --set reranking.model=gpt-4)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.config, args.set))
