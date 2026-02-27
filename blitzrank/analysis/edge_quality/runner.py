"""
Orchestrate edge-quality experiments: load data, run LLM comparisons across
prompt/model/temperature configs, extract edges, evaluate against ground truth,
and persist raw + aggregated results.
"""
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

from litellm import acompletion

from ...engine.comparer.listwise_rank_gpt import parse_permutation
from ...engine.config import DatasetConfig
from ...engine.dataset import load_dataset
from .calibration import calibration_curve, effective_noise_level
from .edge_extractor import (
    edges_from_permutation_adjacent,
    edges_from_permutation_complete,
)
from .ground_truth import all_unordered_pairs, edges_from_qrels
from .metrics import (
    bootstrap_ci,
    holm_bonferroni,
    paired_bootstrap_test,
    pairwise_accuracy,
    scc_profile,
    self_consistency,
    swap_flip_rate,
    transitivity_violations,
    weighted_accuracy_by_gap,
)
from .prompts import PROMPT_REGISTRY

import tiktoken

_tokenizer = tiktoken.get_encoding("o200k_base")

Edge = Tuple[str, str]

load_dotenv()


@dataclass
class TrialConfig:
    model: str
    prompt_style: str
    temperature: float
    max_doc_tokens: int
    shuffle_docs: bool
    window_size: int = 20
    repeat_id: int = 0


@dataclass
class QueryResult:
    query_idx: int
    query: str
    qid: str
    doc_ids: List[str]
    permutation: List[int]
    raw_response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    parse_failures: int


def _trim(text: str, max_tokens: int) -> str:
    enc = _tokenizer.encode(text)
    if len(enc) <= max_tokens:
        return text
    return _tokenizer.decode(enc[:max_tokens])


class SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")
    def warning(self, msg): print(f"[WARN] {msg}")


def _api_key_for_model(model: str) -> Optional[str]:
    if model.startswith("openai/"):
        return os.getenv("OPENAI_API_KEY")
    if model.startswith("anthropic/"):
        return os.getenv("ANTHROPIC_API_KEY")
    return None


async def _run_single_query(
    model: str,
    query: str,
    docs: List[str],
    doc_ids: List[str],
    prompt_fn,
    temperature: float,
    max_doc_tokens: int,
) -> QueryResult:
    prepped_docs = [_trim(d.replace("Title: Content: ", "").strip(), max_doc_tokens) for d in docs]
    prepped_query = _trim(query.strip(), 1024)
    messages = prompt_fn(prepped_query, prepped_docs)

    kwargs: Dict[str, Any] = dict(model=model, messages=messages, timeout=120)
    api_key = _api_key_for_model(model)
    if api_key:
        kwargs["api_key"] = api_key
    if temperature > 0:
        kwargs["temperature"] = temperature

    t0 = time.perf_counter()
    response = await acompletion(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    raw_response = response.choices[0].message.content or ""
    usage = response.usage

    if prompt_fn.__name__ == "structured_json_listwise":
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            arr = json.loads(cleaned)
            permutation = [int(x) - 1 for x in arr]
            valid = set(range(len(docs)))
            seen = set()
            filtered = []
            for idx in permutation:
                if idx in valid and idx not in seen:
                    filtered.append(idx)
                    seen.add(idx)
            missing = [i for i in range(len(docs)) if i not in seen]
            permutation = filtered + missing
            parse_failures = len(missing)
        except (json.JSONDecodeError, TypeError, ValueError):
            permutation, missing, _ = parse_permutation(raw_response, len(docs))
            parse_failures = len(missing) + 1
    else:
        permutation, missing, duplicates = parse_permutation(raw_response, len(docs))
        parse_failures = len(missing) + len(duplicates)

    return QueryResult(
        query_idx=-1,
        query=query,
        qid="",
        doc_ids=doc_ids,
        permutation=permutation,
        raw_response=raw_response,
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
        latency_ms=latency_ms,
        parse_failures=parse_failures,
    )


async def run_trial(
    dataset_raw: Dict[str, Any],
    config: TrialConfig,
    max_queries: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one trial (one prompt x model x temperature combo) across all queries."""
    prompt_fn = PROMPT_REGISTRY[config.prompt_style]
    results_list = dataset_raw["results"]
    qrels = dataset_raw["qrels"]

    if max_queries and max_queries < len(results_list):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(results_list)), max_queries))
    else:
        indices = list(range(len(results_list)))

    query_results: List[QueryResult] = []
    for idx in indices:
        item = results_list[idx]
        hits = item["hits"][:config.window_size]
        docs = [h["content"] for h in hits]
        doc_ids = [str(h["docid"]) for h in hits]
        qid = str(hits[0]["qid"]) if hits else str(idx)

        if config.shuffle_docs:
            combined = list(zip(docs, doc_ids))
            random.Random(seed + idx + config.repeat_id).shuffle(combined)
            docs, doc_ids = [list(t) for t in zip(*combined)]

        try:
            qr = await _run_single_query(
                config.model, item["query"], docs, doc_ids,
                prompt_fn, config.temperature, config.max_doc_tokens,
            )
            qr.query_idx = idx
            qr.qid = qid
            query_results.append(qr)
        except Exception:
            query_results.append(QueryResult(
                query_idx=idx, query=item["query"], qid=qid,
                doc_ids=doc_ids, permutation=list(range(len(docs))),
                raw_response="FAILED", input_tokens=0, output_tokens=0,
                latency_ms=0, parse_failures=len(docs),
            ))

    trial_metrics = _evaluate_trial(query_results, qrels)
    trial_metrics["config"] = {
        "model": config.model,
        "prompt_style": config.prompt_style,
        "temperature": config.temperature,
        "max_doc_tokens": config.max_doc_tokens,
        "shuffle_docs": config.shuffle_docs,
        "repeat_id": config.repeat_id,
    }
    trial_metrics["raw_results"] = [
        {
            "query_idx": qr.query_idx,
            "permutation": qr.permutation,
            "parse_failures": qr.parse_failures,
            "input_tokens": qr.input_tokens,
            "output_tokens": qr.output_tokens,
            "latency_ms": qr.latency_ms,
        }
        for qr in query_results
    ]
    return trial_metrics


def _evaluate_trial(
    query_results: List[QueryResult],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """Compute all edge-quality metrics for a completed trial."""
    per_query_acc_adj = []
    per_query_acc_comp = []
    per_query_wacc = []
    per_query_cycles = []
    all_parse_failures = 0
    total_edges_adj = 0
    total_edges_comp = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0.0
    per_query_details = []

    for qr in query_results:
        qrels_for_query = qrels.get(qr.qid, {})

        adj_edges = set(edges_from_permutation_adjacent(qr.doc_ids, qr.permutation))
        comp_edges = set(edges_from_permutation_complete(qr.doc_ids, qr.permutation))

        gt_edges, ambiguous = edges_from_qrels(qrels_for_query, qr.doc_ids)

        acc_adj = pairwise_accuracy(adj_edges, gt_edges, ambiguous)
        acc_comp = pairwise_accuracy(comp_edges, gt_edges, ambiguous)
        wacc = weighted_accuracy_by_gap(comp_edges, qrels_for_query)
        cycles = transitivity_violations(comp_edges)

        if not (acc_adj["accuracy"] != acc_adj["accuracy"]):  # not NaN
            per_query_acc_adj.append(acc_adj["accuracy"])
        if not (acc_comp["accuracy"] != acc_comp["accuracy"]):
            per_query_acc_comp.append(acc_comp["accuracy"])
        if not (wacc["weighted_accuracy"] != wacc["weighted_accuracy"]):
            per_query_wacc.append(wacc["weighted_accuracy"])
        per_query_cycles.append(cycles["three_cycles"])

        all_parse_failures += qr.parse_failures
        total_edges_adj += acc_adj.get("n_evaluable", 0)
        total_edges_comp += acc_comp.get("n_evaluable", 0)
        total_input_tokens += qr.input_tokens
        total_output_tokens += qr.output_tokens
        total_latency_ms += qr.latency_ms

        per_query_details.append({
            "query_idx": qr.query_idx,
            "acc_adjacent": acc_adj["accuracy"],
            "acc_complete": acc_comp["accuracy"],
            "weighted_acc": wacc["weighted_accuracy"],
            "three_cycles": cycles["three_cycles"],
            "parse_failures": qr.parse_failures,
        })

    mean_adj, ci_lo_adj, ci_hi_adj = bootstrap_ci(per_query_acc_adj)
    mean_comp, ci_lo_comp, ci_hi_comp = bootstrap_ci(per_query_acc_comp)
    mean_wacc, ci_lo_wacc, ci_hi_wacc = bootstrap_ci(per_query_wacc)

    return {
        "edge_accuracy_adjacent": {"mean": mean_adj, "ci_lo": ci_lo_adj, "ci_hi": ci_hi_adj},
        "edge_accuracy_complete": {"mean": mean_comp, "ci_lo": ci_lo_comp, "ci_hi": ci_hi_comp},
        "weighted_accuracy": {"mean": mean_wacc, "ci_lo": ci_lo_wacc, "ci_hi": ci_hi_wacc},
        "mean_three_cycles": sum(per_query_cycles) / max(len(per_query_cycles), 1),
        "total_parse_failures": all_parse_failures,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_latency_ms": total_latency_ms,
        "n_queries": len(query_results),
        "per_query": per_query_details,
    }


MODELS_SMALL = ["openai/gpt-4.1-mini", "anthropic/claude-3-haiku-20240307"]
MODELS_MEDIUM = ["openai/gpt-4.1-mini", "openai/gpt-4.1", "anthropic/claude-3-haiku-20240307", "anthropic/claude-sonnet-4-20250514"]


def build_small_pilot_configs() -> List[TrialConfig]:
    """Fractional grid for rapid signal detection."""
    configs = []
    prompts = ["baseline", "criteria_guided"]
    for model in MODELS_SMALL:
        for prompt in prompts:
            configs.append(TrialConfig(
                model=model, prompt_style=prompt, temperature=0.0,
                max_doc_tokens=512, shuffle_docs=False,
            ))
    for model in MODELS_SMALL:
        configs.append(TrialConfig(
            model=model, prompt_style="baseline", temperature=0.0,
            max_doc_tokens=512, shuffle_docs=True,
        ))
    return configs


def build_medium_configs() -> List[TrialConfig]:
    """Expanded grid focusing on most informative comparisons from pilot."""
    configs = []
    prompts = ["baseline", "criteria_guided", "structured_json"]
    temperatures = [0.0, 0.3]
    for model in MODELS_MEDIUM:
        for prompt in prompts:
            for temp in temperatures:
                configs.append(TrialConfig(
                    model=model, prompt_style=prompt, temperature=temp,
                    max_doc_tokens=512, shuffle_docs=False,
                ))
    for model in MODELS_MEDIUM:
        configs.append(TrialConfig(
            model=model, prompt_style="baseline", temperature=0.0,
            max_doc_tokens=512, shuffle_docs=True,
        ))
        for repeat_id in range(3):
            configs.append(TrialConfig(
                model=model, prompt_style="criteria_guided", temperature=0.3,
                max_doc_tokens=512, shuffle_docs=False, repeat_id=repeat_id,
            ))
    return configs


DATASET_SPECS = {
    "small": [
        {"type": "online_bm25", "name": "dl19-passage", "index": "msmarco-v1-passage"},
        {"type": "online_bm25", "name": "beir-v1.0.0-nfcorpus-test", "index": "beir-v1.0.0-nfcorpus.flat"},
        {"type": "online_bm25", "name": "beir-v1.0.0-scifact-test", "index": "beir-v1.0.0-scifact.flat"},
    ],
    "broad": [
        {"type": "online_bm25", "name": "dl19-passage", "index": "msmarco-v1-passage"},
        {"type": "online_bm25", "name": "dl20-passage", "index": "msmarco-v1-passage"},
        {"type": "online_bm25", "name": "beir-v1.0.0-nfcorpus-test", "index": "beir-v1.0.0-nfcorpus.flat"},
        {"type": "online_bm25", "name": "beir-v1.0.0-scifact-test", "index": "beir-v1.0.0-scifact.flat"},
        {"type": "online_bm25", "name": "beir-v1.0.0-trec-covid-test", "index": "beir-v1.0.0-trec-covid.flat"},
        {"type": "online_bm25", "name": "beir-v1.0.0-fiqa-test", "index": "beir-v1.0.0-fiqa.flat"},
    ],
}


async def run_experiment(
    stage: str = "small",
    output_dir: str = "outputs/edge_quality",
    max_queries: int = 10,
    seed: int = 42,
):
    """Main entry point for the edge-quality experiment."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger()

    configs = build_small_pilot_configs() if stage == "small" else build_medium_configs()
    dataset_specs = DATASET_SPECS.get(stage, DATASET_SPECS["small"])

    all_results = []

    for ds_spec in dataset_specs:
        ds_config = DatasetConfig(**ds_spec, k=100, subset_size=max_queries, subset_seed=seed)
        logger.info(f"Loading dataset: {ds_spec['name']}")
        dataset_raw = load_dataset(ds_config, logger)
        ds_name = ds_spec["name"]

        for trial_config in configs:
            trial_id = (
                f"{ds_name}__{trial_config.model.replace('/', '_')}"
                f"__{trial_config.prompt_style}__t{trial_config.temperature}"
                f"__tok{trial_config.max_doc_tokens}"
                f"__shuf{trial_config.shuffle_docs}__r{trial_config.repeat_id}"
            )
            result_file = out / f"{trial_id}.json"
            if result_file.exists():
                logger.info(f"Skipping existing: {trial_id}")
                with open(result_file) as f:
                    all_results.append(json.load(f))
                continue

            logger.info(f"Running trial: {trial_id}")
            try:
                trial_result = await run_trial(dataset_raw, trial_config)
                trial_result["dataset"] = ds_name
                trial_result["trial_id"] = trial_id
                with open(result_file, "w") as f:
                    json.dump(trial_result, f, indent=2, default=str)
                all_results.append(trial_result)
            except Exception as e:
                logger.warning(f"Trial {trial_id} failed: {e}")
                continue

    summary_file = out / f"summary_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {summary_file}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run edge-quality experiments")
    parser.add_argument("--stage", default="small", choices=["small", "medium", "broad"])
    parser.add_argument("--output-dir", default="outputs/edge_quality")
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run_experiment(args.stage, args.output_dir, args.max_queries, args.seed))


if __name__ == "__main__":
    main()
