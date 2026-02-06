import json
import os
import re
import sys
import yaml
from pathlib import Path
import hashlib
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger as loguru_logger
from .config import LoggingConfig, DatasetConfig, RerankingConfig
from .utils.logging_utils import get_git_info


class ExperimentLogger:
    def __init__(
        self,
        logging_config: LoggingConfig,
        dataset_config: DatasetConfig,
        reranking_config: RerankingConfig | None,
        experiment_config: dict,
    ):
        self.logging_config = logging_config
        self.experiment_config = experiment_config

        self.start_time = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.output_dir = Path(
            logging_config.output_dir.format(timestamp=self.start_time)
        )
        self.experiment_name = logging_config.experiment_name.format(
            timestamp=self.start_time
        )
        dataset_dir = dataset_config.short_name
        rerank_dir = reranking_config.short_name if reranking_config else "no_rerank"
        self.exp_dir = self.output_dir / dataset_dir / self.experiment_name / rerank_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.summary_metrics_file_path = (
            self.output_dir / logging_config.summary_metrics_path
        )

        self._wandb = None
        self._wandb_run = None
        self._rounds_table = None
        self._rounds_table_columns: List[str] = ["query_idx", "round_idx"]
        self._rounds_table_rows: List[Dict[str, Any]] = []
        self._wandb_rounds_logged = 0

        self.logger = loguru_logger.bind(experiment_name=self.experiment_name)
        self.logger.remove()
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        file_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
            "{name}:{function}:{line} - {message}"
        )
        log_level = os.environ.get("LISTWISE_LOG_LEVEL", "DEBUG")
        self.logger.add(
            sys.stderr,
            level=log_level,
            format=fmt,
            colorize=True,
        )
        self.logger.add(
            self.exp_dir / "run.log",
            level=log_level,
            format=file_fmt,
        )

        self.info(
            f"Initializing experiment logger with config: {json.dumps(self.experiment_config, indent=2)}."
            f"Output directory: {self.exp_dir}."
        )
        self.dump_repro_info(self.experiment_config)
        if self.logging_config.enable_wandb:
            self._init_wandb_run()

    def info(self, msg: str):
        self.logger.opt(depth=1).info(msg)

    def warning(self, msg: str):
        self.logger.opt(depth=1).warning(msg)

    def debug(self, msg: str):
        self.logger.opt(depth=1).debug(msg)

    def error(self, msg: str):
        self.logger.opt(depth=1).error(msg)

    def exception(self, msg: str):
        self.logger.opt(depth=1).exception(msg)

    def log_metrics(self, metrics: Dict[str, float]):
        """
        At the end of the experiment, log the metrics to the log file as well as to a summary csv file.
        """
        self.info("Metrics:")
        for metric, value in metrics.items():
            self.info(f"  {metric}: {value}")

        # log summary metrics to the summary jsonl file
        self._log_summary_metrics_to_proj_dir(metrics)
        # if wandb is enabled, log the metrics to wandb
        if self.logging_config.enable_wandb:
            self._log_summary_metrics_to_wandb(metrics)

    def log_task_summary(self, payload: Dict[str, Any]) -> None:
        """Log per-task metrics to tasks.jsonl and W&B."""
        output_file = self.exp_dir / "tasks.jsonl"
        to_log = dict(payload)
        if "timestamp" not in to_log:
            to_log["timestamp"] = datetime.now().isoformat(timespec="seconds")
        with open(output_file, "a") as f:
            f.write(json.dumps(to_log) + "\n")

        if self.logging_config.enable_wandb and self._wandb_run is not None:
            query_idx = to_log.get("query_idx")
            wandb_payload = {"task/query_idx": query_idx}
            for k, v in to_log.items():
                if k in ("timestamp", "query_idx", "qid"):
                    continue
                if isinstance(v, (int, float)):
                    wandb_payload[f"task/{self._sanitize_metric_key(k)}"] = v
            self._wandb_run.log(wandb_payload)

    def log_round(self, query_idx: int, round_log: Dict[str, Any]) -> None:
        """
        Mainly used by TournamentGraphSort to log rounds information.
        """
        rounds_dir = self.exp_dir / "rounds"
        rounds_dir.mkdir(exist_ok=True)
        output_file = rounds_dir / f"query_{query_idx}.jsonl"
        to_log = dict(round_log)
        to_log.setdefault("query_idx", query_idx)
        with open(output_file, "a") as f:
            f.write(json.dumps(to_log) + "\n")
        self._log_round_to_wandb(query_idx, to_log)

    def _wandb_table_value(self, v: Any) -> Any:
        if v is None or isinstance(v, (int, float, str, bool)):
            return v
        try:
            return json.dumps(
                v, ensure_ascii=False, default=str, sort_keys=True, indent=2
            )
        except TypeError:
            return str(v)

    def _log_round_to_wandb(self, query_idx: int, to_log: Dict[str, Any]) -> None:
        if not self.logging_config.enable_wandb or self._wandb_run is None:
            return

        rounds_table = None
        if self.logging_config.log_round_table_to_wandb:
            row = dict(to_log)
            row["query_idx"] = query_idx
            row = {k: self._wandb_table_value(v) for k, v in row.items()}
            new_keys = [k for k in row.keys() if k not in self._rounds_table_columns]
            if new_keys:
                self._rounds_table_columns.extend(sorted(new_keys))
                self._rounds_table_rows.append(row)
                self._rounds_table = self._wandb.Table(
                    columns=self._rounds_table_columns, log_mode="MUTABLE"
                )
                for r in self._rounds_table_rows:
                    self._rounds_table.add_data(
                        *[r.get(c) for c in self._rounds_table_columns]
                    )
            else:
                self._rounds_table_rows.append(row)
                if self._rounds_table is None:
                    self._rounds_table = self._wandb.Table(
                        columns=self._rounds_table_columns, log_mode="MUTABLE"
                    )
                self._rounds_table.add_data(
                    *[row.get(c) for c in self._rounds_table_columns]
                )
            if self.logging_config.log_round_table_every_round:
                rounds_table = self._rounds_table

        self._wandb_rounds_logged += 1
        payload = {
            "round/query_idx": query_idx,
            "round/round_idx": to_log.get("round_idx"),
            "round/rounds_logged": self._wandb_rounds_logged,
        }
        if rounds_table is not None:
            payload["rounds_table"] = rounds_table
        self._wandb_run.log(payload)

    def summary_metrics_already_logged(self) -> bool:
        if not self.summary_metrics_file_path.exists():
            return False

        with open(self.summary_metrics_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("experiment_config") == self.experiment_config:
                    return True

        return False

    def _log_summary_metrics_to_proj_dir(self, metrics: Dict[str, float]):
        to_log = {
            "experiment_name": self.experiment_name,
            "experiment_config": self.experiment_config,
            "metrics": metrics,
            "git": get_git_info(),
            "start_time": self.start_time,
            "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        with open(self.summary_metrics_file_path, "a") as f:
            f.write(json.dumps(to_log) + "\n")
        self.info(f"Summary metrics saved to: {self.summary_metrics_file_path}")

    def _log_summary_metrics_to_wandb(self, metrics: Dict[str, float]):
        if self._wandb_run is None:
            raise RuntimeError("W&B is enabled but no run was initialized")

        to_log = {
            f"metrics/{self._sanitize_metric_key(k)}": v for k, v in metrics.items()
        }
        self._wandb_run.log(to_log)
        self._wandb_run.summary.update(to_log)

    def close(self) -> None:
        if self.logging_config.sync_to_remote:
            self._sync_to_remote()
        else:
            self.info("Syncing to remote is disabled")
        if self._wandb is None or self._wandb_run is None:
            return
        try:
            if (
                self.logging_config.log_round_table_to_wandb
                and not self.logging_config.log_round_table_every_round
                and self._rounds_table is not None
            ):
                self._wandb_run.log({"rounds_table": self._rounds_table})
            self._wandb.finish()
        except Exception as e:
            self.warning(f"Failed to finish W&B run: {e}")

    def _sync_to_remote(self) -> None:
        dest = os.environ.get("LISTWISE_RSYNC_DEST")
        if not dest:
            return
        from sh import rsync

        relative_path = self.exp_dir.relative_to(self.output_dir)
        src = f"./{relative_path}/"
        self.info(f"Syncing {relative_path} to {dest}")
        rsync("-avzR", src, dest.rstrip("/") + "/", _cwd=str(self.output_dir))

    def _init_wandb_run(self) -> None:
        import wandb  # type: ignore[import-not-found]

        self._wandb = wandb
        run_id = self._wandb_run_id()
        config = {
            "experiment_name": self.experiment_name,
            "experiment_config": self.experiment_config,
            "git": get_git_info(),
            "start_time": self.start_time,
        }

        # Extract tags from experiment config
        tags = self._extract_wandb_tags()

        self._wandb_run = wandb.init(
            project=self.logging_config.wandb_project,
            name=self.get_full_experiment_name(),
            id=run_id,
            resume="allow",
            dir=str(self.exp_dir),
            config=config,
            tags=tags,
        )
        self.info(
            f"W&B enabled: project={self.logging_config.wandb_project} run_id={run_id} tags={tags}"
        )

    def _extract_wandb_tags(self) -> List[str]:
        """Extract tags for W&B run based on experiment configuration."""
        tags = []

        # Always add dataset name
        dataset_name = self.experiment_config.get("dataset", {}).get("name")
        if dataset_name:
            tags.append(dataset_name)

        # Check if reranking is configured
        reranking_config = self.experiment_config.get("reranking")
        if not reranking_config:
            # No reranking - baseline
            tags.append("no_rerank")
        else:
            # Reranking is configured
            ranker_type = reranking_config.get("ranker", {}).get("type")
            if ranker_type == "ctxl_api":
                # Pointwise reranker
                tags.append("ptwise reranker")
            else:
                # Specific model and selector for LLM-based reranking
                model = reranking_config.get("ranker", {}).get("model")
                if model:
                    # Extract model name (remove provider prefix if present)
                    model_name = model.split("/")[-1] if "/" in model else model
                    tags.append(model_name)

                selector_type = reranking_config.get("selector", {}).get("type")
                if selector_type:
                    tags.append(selector_type)

        return tags

    def _wandb_run_id(self) -> str:
        stable = f"{self.logging_config.wandb_project}:{str(self.exp_dir)}"
        return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]

    def _sanitize_metric_key(self, key: str) -> str:
        key = key.replace("@", "_")
        return re.sub(r"[^0-9A-Za-z_.-]", "_", key)

    def get_output_dir(self) -> Path:
        return self.exp_dir

    def log_iteration_results(self, iteration_logs: List[Dict[str, Any]]) -> None:
        output_file = self.exp_dir / "iteration_results.jsonl"
        with open(output_file, "w") as f:
            for entry in iteration_logs:
                f.write(json.dumps(entry) + "\n")
        self.info(f"Iteration results saved to: {output_file}")

    def dump_repro_info(self, config_data: dict):
        """
        Dump the repro info to the output directory.
        """
        repro_dir = self.get_output_dir() / "repro"
        repro_dir.mkdir(exist_ok=True)
        git_info = get_git_info()
        (repro_dir / "commit_id.txt").write_text(git_info["commit"])
        (repro_dir / "git.json").write_text(json.dumps(git_info, indent=2))
        with open(repro_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def get_full_experiment_name(self) -> str:
        return self.exp_dir.name
