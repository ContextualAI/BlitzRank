import asyncio
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from tqdm.asyncio import tqdm as tqdm_async
from .utils.rerank_runner_utils import CheckpointWriter, build_task_summary
from .logger import ExperimentLogger


async def run_with_checkpoints(
    items: List[Any],
    max_parallel_requests: int,
    output_dir: Path,
    logger: ExperimentLogger,
    process_item: Callable[
        [Any, int, Dict[str, Any] | None, CheckpointWriter],
        Awaitable[Tuple[Any, Any]],
    ],
    description: str = "Reranking",
    checkpoint_filename: str = "checkpoint.json",
) -> Tuple[List[Any], List[Any]]:
    results: List[Any] = [None] * len(items)
    logs: List[Any] = [None] * len(items)
    semaphore = asyncio.Semaphore(max_parallel_requests)

    checkpoint_writer = CheckpointWriter(output_dir / checkpoint_filename)
    checkpoint = checkpoint_writer.load()
    await checkpoint_writer.start()

    num_complete = sum(1 for c in checkpoint.values() if c.get("complete"))
    if num_complete > 0 and logger is not None:
        logger.info(f"Resuming: {num_complete}/{len(items)} queries already complete")

    async def run_item(item: Any, idx: int):
        if idx in checkpoint and checkpoint[idx].get("complete"):
            return checkpoint[idx]["result"], checkpoint[idx]["logs"], idx

        async with semaphore:
            try:
                start_time = time.perf_counter()
                result, item_logs = await process_item(
                    item, idx, checkpoint.get(idx), checkpoint_writer
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                if logger is not None and hasattr(logger, "log_task_summary"):
                    logger.log_task_summary(
                        build_task_summary(idx, latency_ms, item_logs)
                    )

                await checkpoint_writer.update(
                    idx, {"complete": True, "result": result, "logs": item_logs}
                )
                return result, item_logs, idx
            except Exception as e:
                logger.exception(f"Error processing item query_{idx}: {e}")
                raise

    try:
        tasks = [run_item(item, idx) for idx, item in enumerate(items)]
        task_results = await tqdm_async.gather(*tasks, desc=description)
        for result, item_logs, idx in task_results:
            results[idx] = result
            logs[idx] = item_logs
        return results, logs
    finally:
        await checkpoint_writer.close()
