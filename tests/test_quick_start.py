"""Test: Quick Start snippet from README."""
from blitzrank import BlitzRank, rank

ranker = BlitzRank()

query = "capital of France"
docs = [
    "Berlin is the capital of Germany.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan.",
]

indices = rank(ranker, model="openai/gpt-4.1", query=query, docs=docs, topk=2)
top_docs = [docs[i] for i in indices]

print(f"Indices: {indices}")
print(f"Top docs: {top_docs}")
assert len(indices) == 2, f"Expected 2 indices, got {len(indices)}"
assert indices[0] == 1, f"Expected Paris (index 1) ranked first, got index {indices[0]}"
print("PASSED")
