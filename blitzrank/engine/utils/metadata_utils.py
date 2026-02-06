import statistics
from collections import Counter, defaultdict
from typing import Iterable

from .model_costs import estimate_cost_usd


# Field mappings: (step_key, totals_key, total_entry_key)
FIELD_MAPPINGS = [
    ("input_tokens", "input_tokens", "input_tokens_total"),
    ("output_tokens", "output_tokens", "output_tokens_total"),
    ("thought_tokens", "thought_tokens", "thought_tokens_total"),
    ("latency_ms", "llm_latency_ms", "llm_latency_ms_total"),
    ("num_trimmed_docs", "num_trimmed_docs", "num_trimmed_docs"),
]


def collect_reranking_stats(iteration_logs: Iterable[dict]) -> dict:
    """Collect both totals and individual values for computing statistics."""
    totals: Counter = Counter(
        {
            totals_key: 0 if totals_key != "llm_latency_ms" else 0.0
            for _, totals_key, _ in FIELD_MAPPINGS
        }
    )
    values = defaultdict(list)

    for entry in iteration_logs:
        if not isinstance(entry, dict):
            continue
        iterations = entry.get("iterations")
        if isinstance(iterations, list):
            for step in iterations:
                if not isinstance(step, dict):
                    continue
                _accumulate_step_stats(totals, values, step)
            continue

        # Process non-iterative entries
        for step_key, totals_key, total_entry_key in FIELD_MAPPINGS:
            _accumulate_total(totals, entry, total_entry_key, totals_key)
            value = entry.get(total_entry_key)
            if isinstance(value, (int, float)):
                values[totals_key].append(value)

    return {"totals": totals, "values": values}


def collect_reranking_totals(iteration_logs: Iterable[dict]) -> Counter:
    """Legacy function for backward compatibility."""
    return collect_reranking_stats(iteration_logs)["totals"]


def build_reranking_metrics(
    iteration_logs: Iterable[dict], model: str, latency_seconds: float
) -> dict[str, float]:
    stats = collect_reranking_stats(iteration_logs)
    totals = stats["totals"]
    values = stats["values"]

    metrics = {"reranking/latency_total_s": latency_seconds}

    # Add total metrics for each tracked field
    for _, totals_key, _ in FIELD_MAPPINGS:
        if totals_key == "num_trimmed_docs":
            metrics["total_num_trimmed_docs"] = totals[totals_key]
        else:
            metrics[f"reranking/{totals_key}_total"] = totals[totals_key]

    # Add statistical metrics for each tracked field
    for _, totals_key, _ in FIELD_MAPPINGS:
        field_values = values[totals_key]
        if field_values:
            metrics.update(_compute_field_stats(totals_key, field_values))

    cost = estimate_cost_usd(
        model, int(totals["input_tokens"]), int(totals["output_tokens"])
    )
    if cost is not None:
        metrics["reranking/cost_usd_estimate"] = cost
    return metrics


def _accumulate_step_stats(totals: Counter, values: dict, step: dict) -> None:
    for step_key, totals_key, _ in FIELD_MAPPINGS:
        _accumulate_total(totals, step, step_key, totals_key)
        value = step.get(step_key)
        if isinstance(value, (int, float)):
            values[totals_key].append(value)


def _compute_field_stats(field: str, values: list) -> dict[str, float]:
    """Compute average, min, max, and standard deviation for a field."""
    if not values:
        return {}

    metrics = {}
    metrics[f"reranking/{field}_avg"] = statistics.mean(values)
    metrics[f"reranking/{field}_min"] = min(values)
    metrics[f"reranking/{field}_max"] = max(values)

    # Only compute stdev if we have more than 1 value
    if len(values) > 1:
        metrics[f"reranking/{field}_stdev"] = statistics.stdev(values)

    return metrics


def _accumulate_total(
    totals: Counter, entry: dict, entry_key: str, total_key: str
) -> None:
    value = entry.get(entry_key)
    if isinstance(value, (int, float)):
        totals[total_key] += value
