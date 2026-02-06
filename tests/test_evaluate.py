"""Test: Evaluate on a Benchmark snippet from README."""
from blitzrank import BlitzRank, evaluate

ranker = BlitzRank()
rankings, metrics = evaluate(ranker, dataset="msmarco/dl19/bm25", model="openai/gpt-4.1-nano")

print(f"Metrics: {metrics}")
print(f"Num rankings: {len(rankings)}")
print(f"First ranking: {rankings[0]}")
keys = [x.lower() for x in list(metrics.keys())]
assert "ndcg_cut_10" in keys or "ndcg@10" in keys, f"Expected ndcg metric in {keys}"
assert len(rankings) > 0, "Expected at least one ranking"
assert "query" in rankings[0] and "ranking" in rankings[0], f"Unexpected ranking format: {rankings[0]}"
print("PASSED")
