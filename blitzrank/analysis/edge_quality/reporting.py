"""
Generate the final analysis report from experiment results.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .metrics import bootstrap_ci, holm_bonferroni, paired_bootstrap_test


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    path = Path(results_dir)
    results = []
    for f in sorted(path.glob("*.json")):
        if f.name.startswith("summary_"):
            continue
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def _safe_mean(values):
    return statistics.mean(values) if values else float("nan")


def aggregate_by_config(results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """Group results by (model, prompt_style, temperature) and aggregate."""
    groups = defaultdict(list)
    for r in results:
        cfg = r.get("config", {})
        key = f"{cfg.get('model', '?')} | {cfg.get('prompt_style', '?')} | t={cfg.get('temperature', '?')}"
        groups[key].append(r)

    aggregated = {}
    for key, group in groups.items():
        adj_means = [r["edge_accuracy_adjacent"]["mean"] for r in group
                      if r["edge_accuracy_adjacent"]["mean"] == r["edge_accuracy_adjacent"]["mean"]]
        comp_means = [r["edge_accuracy_complete"]["mean"] for r in group
                       if r["edge_accuracy_complete"]["mean"] == r["edge_accuracy_complete"]["mean"]]
        wacc_means = [r["weighted_accuracy"]["mean"] for r in group
                       if r["weighted_accuracy"]["mean"] == r["weighted_accuracy"]["mean"]]

        aggregated[key] = {
            "n_trials": len(group),
            "edge_accuracy_adjacent": _safe_mean(adj_means),
            "edge_accuracy_complete": _safe_mean(comp_means),
            "weighted_accuracy": _safe_mean(wacc_means),
            "mean_cycles": _safe_mean([r.get("mean_three_cycles", 0) for r in group]),
            "total_parse_failures": sum(r.get("total_parse_failures", 0) for r in group),
            "datasets": list({r.get("dataset", "?") for r in group}),
        }
    return aggregated


def test_h1(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """H1: best prompt/model > baseline prompt on edge accuracy."""
    baseline_vals = []
    best_vals = []
    for r in results:
        cfg = r.get("config", {})
        per_q = r.get("per_query", [])
        accs = [q["acc_complete"] for q in per_q
                if q["acc_complete"] == q["acc_complete"]]
        if not accs:
            continue
        if cfg.get("prompt_style") == "baseline" and not cfg.get("shuffle_docs"):
            baseline_vals.extend(accs)
        elif cfg.get("prompt_style") != "baseline" and not cfg.get("shuffle_docs"):
            best_vals.extend(accs)

    if not baseline_vals or not best_vals:
        return {"h1_result": "insufficient_data"}

    baseline_mean, bl_lo, bl_hi = bootstrap_ci(baseline_vals)
    best_mean, bt_lo, bt_hi = bootstrap_ci(best_vals)
    return {
        "h1_result": "supported" if best_mean > baseline_mean and bt_lo > bl_lo else "not_supported",
        "baseline_accuracy": {"mean": baseline_mean, "ci": [bl_lo, bl_hi]},
        "best_accuracy": {"mean": best_mean, "ci": [bt_lo, bt_hi]},
        "delta": best_mean - baseline_mean,
    }


def test_h2(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """H2: robustness under shuffle / repeated runs."""
    original_accs = []
    shuffled_accs = []
    for r in results:
        cfg = r.get("config", {})
        per_q = r.get("per_query", [])
        accs = [q["acc_complete"] for q in per_q
                if q["acc_complete"] == q["acc_complete"]]
        if not accs:
            continue
        if cfg.get("shuffle_docs"):
            shuffled_accs.extend(accs)
        elif cfg.get("prompt_style") == "baseline":
            original_accs.extend(accs)

    if not original_accs or not shuffled_accs:
        return {"h2_result": "insufficient_data"}

    orig_mean, _, _ = bootstrap_ci(original_accs)
    shuf_mean, _, _ = bootstrap_ci(shuffled_accs)
    drop = orig_mean - shuf_mean
    return {
        "h2_result": "supported" if abs(drop) < 0.05 else "not_supported",
        "original_accuracy": orig_mean,
        "shuffled_accuracy": shuf_mean,
        "order_sensitivity": drop,
    }


def test_h4(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """H4: optimized prompts reduce position/order bias."""
    baseline_drops = []
    optimized_drops = []
    by_model = defaultdict(lambda: {"baseline_orig": [], "baseline_shuf": [], "opt_orig": [], "opt_shuf": []})

    for r in results:
        cfg = r.get("config", {})
        per_q = r.get("per_query", [])
        accs = [q["acc_complete"] for q in per_q if q["acc_complete"] == q["acc_complete"]]
        if not accs:
            continue
        model = cfg.get("model", "?")
        mean_acc = _safe_mean(accs)
        if cfg.get("prompt_style") == "baseline":
            if cfg.get("shuffle_docs"):
                by_model[model]["baseline_shuf"].append(mean_acc)
            else:
                by_model[model]["baseline_orig"].append(mean_acc)
        else:
            if cfg.get("shuffle_docs"):
                by_model[model]["opt_shuf"].append(mean_acc)
            else:
                by_model[model]["opt_orig"].append(mean_acc)

    for model, buckets in by_model.items():
        if buckets["baseline_orig"] and buckets["baseline_shuf"]:
            baseline_drops.append(_safe_mean(buckets["baseline_orig"]) - _safe_mean(buckets["baseline_shuf"]))
        if buckets["opt_orig"] and buckets["opt_shuf"]:
            optimized_drops.append(_safe_mean(buckets["opt_orig"]) - _safe_mean(buckets["opt_shuf"]))

    if not baseline_drops:
        return {"h4_result": "insufficient_data"}

    return {
        "h4_result": "supported" if _safe_mean(optimized_drops) < _safe_mean(baseline_drops) else "not_supported",
        "baseline_order_sensitivity": _safe_mean(baseline_drops),
        "optimized_order_sensitivity": _safe_mean(optimized_drops) if optimized_drops else None,
    }


def generate_report(results: List[Dict[str, Any]], output_path: str) -> str:
    """Generate full markdown report."""
    agg = aggregate_by_config(results)
    h1 = test_h1(results)
    h2 = test_h2(results)
    h4 = test_h4(results)

    lines = [
        "# Edge-Quality Validation Study: Results",
        "",
        f"**Generated:** {__import__('datetime').datetime.now().isoformat()}",
        f"**Total trials:** {len(results)}",
        "",
        "## 1. Hypothesis Testing Summary",
        "",
        "### H1: Edge Validity (best prompt > baseline)",
        f"- Result: **{h1.get('h1_result', 'N/A')}**",
    ]
    if "delta" in h1:
        lines.append(f"- Baseline accuracy: {h1['baseline_accuracy']['mean']:.4f} "
                      f"[{h1['baseline_accuracy']['ci'][0]:.4f}, {h1['baseline_accuracy']['ci'][1]:.4f}]")
        lines.append(f"- Best accuracy: {h1['best_accuracy']['mean']:.4f} "
                      f"[{h1['best_accuracy']['ci'][0]:.4f}, {h1['best_accuracy']['ci'][1]:.4f}]")
        lines.append(f"- Delta: {h1['delta']:+.4f}")

    lines += [
        "",
        "### H2: Robustness Under Order Swap",
        f"- Result: **{h2.get('h2_result', 'N/A')}**",
    ]
    if "order_sensitivity" in h2:
        lines.append(f"- Original accuracy: {h2['original_accuracy']:.4f}")
        lines.append(f"- Shuffled accuracy: {h2['shuffled_accuracy']:.4f}")
        lines.append(f"- Order sensitivity (drop): {h2['order_sensitivity']:+.4f}")

    lines += [
        "",
        "### H4: Bias Reduction",
        f"- Result: **{h4.get('h4_result', 'N/A')}**",
    ]
    if "baseline_order_sensitivity" in h4:
        lines.append(f"- Baseline order sensitivity: {h4['baseline_order_sensitivity']:.4f}")
        if h4.get("optimized_order_sensitivity") is not None:
            lines.append(f"- Optimized order sensitivity: {h4['optimized_order_sensitivity']:.4f}")

    lines += [
        "",
        "## 2. Per-Configuration Results",
        "",
        "| Config | Datasets | Adj. Accuracy | Complete Accuracy | Weighted Acc | Cycles | Parse Fails |",
        "|--------|----------|---------------|-------------------|--------------|--------|-------------|",
    ]
    for key, vals in sorted(agg.items()):
        ds = ", ".join(vals["datasets"])
        adj = f"{vals['edge_accuracy_adjacent']:.4f}" if vals['edge_accuracy_adjacent'] == vals['edge_accuracy_adjacent'] else "N/A"
        comp = f"{vals['edge_accuracy_complete']:.4f}" if vals['edge_accuracy_complete'] == vals['edge_accuracy_complete'] else "N/A"
        wacc = f"{vals['weighted_accuracy']:.4f}" if vals['weighted_accuracy'] == vals['weighted_accuracy'] else "N/A"
        lines.append(f"| {key} | {ds} | {adj} | {comp} | {wacc} | {vals['mean_cycles']:.1f} | {vals['total_parse_failures']} |")

    lines += [
        "",
        "## 3. Decision Rubric",
        "",
    ]
    h1_pass = h1.get("h1_result") == "supported"
    h2_pass = h2.get("h2_result") == "supported"

    if h1_pass and h2_pass:
        decision = "**SUPPORTED**: Edge validity and robustness both pass."
    elif h1_pass:
        decision = "**PARTIALLY SUPPORTED**: Edge validity passes but robustness concerns remain."
    else:
        decision = "**NOT SUPPORTED**: Edge validity hypothesis did not pass."
    lines.append(decision)
    lines.append("")

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text)
    return report_text
