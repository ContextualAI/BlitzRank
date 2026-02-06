MODEL_COSTS = {
    "openai/o3": {  # https://openai.com/api/pricing/
        "input_cost_per_token": 2.0 / 1e6,  # $2.00 per million tokens
        "output_cost_per_token": 8.0 / 1e6,  # $8.00 per million tokens
    },
    "vertex_ai/gemini-2.5-flash": {  # https://cloud.google.com/vertex-ai/pricing
        "input_cost_per_token": 0.3 / 1e6,  # $0.30 per million tokens
        "output_cost_per_token": 2.5 / 1e6,  # $2.50 per million tokens
    },
    "vertex_ai/gemini-2.5-pro": {  # https://cloud.google.com/vertex-ai/generative-ai/pricing
        "input_cost_per_token": 1.25 / 1e6,  # $1.25 per million tokens
        "output_cost_per_token": 10.0 / 1e6,  # $10.00 per million tokens
    },
    "vertex_ai/gemini-3-flash-preview": {  # https://cloud.google.com/vertex-ai/generative-ai/pricing
        "input_cost_per_token": 0.5 / 1e6,  # $0.50 per million tokens
        "output_cost_per_token": 3.0 / 1e6,  # $3.00 per million tokens
    },
    "vertex_ai/gemini-3-pro-preview": {  # https://cloud.google.com/vertex-ai/generative-ai/pricing
        "input_cost_per_token": 2.0 / 1e6,  # $2.00 per million tokens
        "output_cost_per_token": 12.0
        / 1e6,  # $12.00 per million tokens (note this is both thoughts and responses tokens)
    },
    "vertex_ai/gemini-2.0-flash-lite": {  # https://cloud.google.com/vertex-ai/generative-ai/pricing
        "input_cost_per_token": 0.075 / 1e6,  # $0.075 per million tokens
        "output_cost_per_token": 0.30 / 1e6,  # $0.30 per million tokens
    },
    "openai/o4-mini": {  # https://platform.openai.com/docs/pricing
        "input_cost_per_token": 1.1 / 1e6,  # $1.10 per million tokens
        "output_cost_per_token": 4.4 / 1e6,  # $4.40 per million tokens
    },
    "anthropic/claude-sonnet-4-20250514": {  # https://www.anthropic.com/pricing#api
        "input_cost_per_token": 3.0 / 1e6,  # $3.00 per million tokens
        "output_cost_per_token": 15.0 / 1e6,  # $15.00 per million tokens
    },
    "openrouter/qwen/qwen3-coder": {  # https://openrouter.ai/qwen/qwen3-coder
        "input_cost_per_token": 0.3 / 1e6,  # $0.30 per million tokens
        "output_cost_per_token": 1.2 / 1e6,  # $1.20 per million tokens
    },
    "moonshotai/kimi-k2": {  # https://openrouter.ai/moonshotai/kimi-k2
        "input_cost_per_token": 0.60 / 1e6,  # $0.60 per million tokens
        "output_cost_per_token": 2.50 / 1e6,  # $2.50 per million tokens
    },
    "openrouter/qwen/qwen3-235b-a22b-thinking-2507": {  # https://openrouter.ai/qwen/qwen3-235b-a22b-thinking-2507
        "input_cost_per_token": 0.6 / 1e6,  # $0.60 per million tokens
        "output_cost_per_token": 3.0 / 1e6,  # $3.00 per million tokens
    },
    "openai/gpt-oss-20b": {  # https://openrouter.ai/openai/gpt-oss-20b
        "input_cost_per_token": 0.05 / 1e6,  # $0.05 per million tokens
        "output_cost_per_token": 0.20 / 1e6,  # $0.20 per million tokens
    },
    "openai/gpt-oss-120b": {  # https://openrouter.ai/openai/gpt-oss-120b
        "input_cost_per_token": 0.15 / 1e6,  # $0.15 per million tokens
        "output_cost_per_token": 0.60 / 1e6,  # $0.60 per million tokens
    },
    "z-ai/glm-4.5": {  # https://openrouter.ai/z-ai/glm-4.5
        "input_cost_per_token": 0.60 / 1e6,  # $0.60 per million tokens
        "output_cost_per_token": 2.20 / 1e6,  # $2.20 per million tokens
    },
    "openai/gpt-5": {  # https://platform.openai.com/docs/models/gpt-5
        "input_cost_per_token": 1.25 / 1e6,  # $1.25 per million tokens
        "output_cost_per_token": 5.0 / 1e6,  # $5.00 per million tokens
    },
}


def normalize_model_for_costs(model: str) -> str:
    if "gpt" in model and not model.startswith("openai/"):
        return f"openai/{model}"
    if "claude" in model and not model.startswith("anthropic/"):
        return f"anthropic/{model}"
    return model


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int
) -> float | None:
    normalized = normalize_model_for_costs(model)
    cost_info = MODEL_COSTS.get(normalized)
    if cost_info is None:
        return None
    return (
        input_tokens * cost_info["input_cost_per_token"]
        + output_tokens * cost_info["output_cost_per_token"]
    )
