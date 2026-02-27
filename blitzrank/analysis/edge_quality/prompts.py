"""
Prompt variants for the edge-quality study.

Each variant returns a list of messages ready for the LiteLLM client.
The 'baseline' variant matches the existing RankGPT prompt exactly.
"""
from typing import Dict, List


def baseline_listwise(query: str, docs: List[str]) -> List[Dict[str, str]]:
    """Original RankGPT multi-turn permutation prompt (default BlitzRank behavior)."""
    messages = [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
        },
        {
            "role": "user",
            "content": f"I will provide you with {len(docs)} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]
    for rank, content in enumerate(docs, 1):
        messages.append({"role": "user", "content": f"[{rank}] {content}"})
        messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
    messages.append({
        "role": "user",
        "content": (
            f"Search Query: {query}. \nRank the {len(docs)} passages above based on their relevance "
            "to the search query. The passages should be listed in descending order using identifiers. "
            "The most relevant passages should be listed first. The output format should be [] > [], "
            "e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."
        ),
    })
    return messages


def criteria_guided_listwise(query: str, docs: List[str]) -> List[Dict[str, str]]:
    """Adds explicit relevance criteria to help the LLM judge more consistently."""
    numbered = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    return [
        {
            "role": "system",
            "content": (
                "You are a relevance judge. Given a search query and passages, rank all passages "
                "by how directly and completely they answer the query. Prefer passages that are "
                "factually specific over vaguely topical ones."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\nPassages:\n{numbered}\n\n"
                f"Rank the {len(docs)} passages from most to least relevant. "
                "Output ONLY the ranking as identifiers separated by ' > ', e.g. [1] > [2] > [3]."
            ),
        },
    ]


def structured_json_listwise(query: str, docs: List[str]) -> List[Dict[str, str]]:
    """Requests JSON array output for easier parsing and potentially higher consistency."""
    numbered = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    return [
        {
            "role": "system",
            "content": "You are a relevance ranking assistant. Output only valid JSON.",
        },
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\nPassages:\n{numbered}\n\n"
                f"Rank the {len(docs)} passages by relevance to the query. "
                'Return a JSON array of passage numbers from most to least relevant, e.g. [3, 1, 2]. '
                "No explanation."
            ),
        },
    ]


PROMPT_REGISTRY: Dict[str, callable] = {
    "baseline": baseline_listwise,
    "criteria_guided": criteria_guided_listwise,
    "structured_json": structured_json_listwise,
}
