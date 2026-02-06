"""Test: Baselines snippet from README (requires .[all] extras)."""
from blitzrank import BlitzRank, SlidingWindow, SetWise, PairWise, TourRank, AcuRank, rank

query = "capital of France"
docs = [
    "Berlin is the capital of Germany.",
    "Paris is the capital of France.",
    "Tokyo is the capital of Japan.",
]

for Method in [BlitzRank, SlidingWindow, SetWise, PairWise, TourRank, AcuRank]:
    name = Method.__name__
    print(f"Testing {name}...", end=" ")
    indices = rank(Method(), model="openai/gpt-4.1", query=query, docs=docs, topk=2)
    assert len(indices) == 2, f"{name}: expected 2 indices, got {len(indices)}"
    assert indices[0] == 1, f"{name}: expected Paris (index 1) ranked first, got index {indices[0]}"
    assert all(0 <= i < len(docs) for i in indices), f"{name}: indices out of range: {indices}"
    print(f"indices={indices} OK")

print("ALL PASSED")
