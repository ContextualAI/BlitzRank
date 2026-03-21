"""
LiteLLM client for making LLM API calls.
"""
import os
import time
from litellm import acompletion, BadRequestError
from ..utils.retry_utils import async_retry
from ..utils.logging_utils import logger


API_TIMEOUT = 300.0


FALLBACK_MODELS = {}

class LitellmClient:
    def set_vars(self, model_name):
        if model_name.startswith("openai/"):
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic/"):
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("vertex_ai/"):
            self.api_key = None
        elif model_name.startswith("gemini/"):
            self.api_key = os.getenv("GEMINI_API_KEY")
        elif model_name.startswith("openrouter/"):
            # self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.api_key = None
        else:
            self.api_key = os.getenv("LOCAL_VLLM_API_KEY")

        if model_name.startswith(("openai/", "anthropic/", "vertex_ai/", "gemini/", "openrouter/")):
            self.base_url = None
            return

        self.base_url = os.getenv("LOCAL_VLLM_BASE_URL")
        if self.base_url is None:
            raise Exception("Please set LOCAL_VLLM_BASE_URL")

    def get_extra_kwargs(self, model_name):
        if "glm" in model_name and model_name.startswith("openrouter/"):
            return {"reasoning": {"enabled": False}}
        if model_name.startswith(("openai/", "anthropic/", "vertex_ai/", "gemini/")):
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    @async_retry()
    async def get_response(self, model, messages, temperature, fallback_enabled=True, **kwargs):
        start_time = time.perf_counter()
        args = dict(model=model, messages=messages, **kwargs)
        if temperature is not None:
            args["temperature"] = temperature
        try:
            response = await self.chat(**args)
        except BadRequestError:
            response = None
        if response is None or len(response.choices) == 0 or response.choices[0].message.content is None:
            if fallback_enabled and model in FALLBACK_MODELS:
                logger.warning(f"No response from {model}, falling back to {FALLBACK_MODELS[model]}")
                return await self.get_response(FALLBACK_MODELS[model], messages, temperature, **kwargs)
            raise Exception("No response from model")
        latency_ms = (time.perf_counter() - start_time) * 1000
        return response.choices[0].message.content, latency_ms, response.usage

    async def chat(self, *args, **kwargs):
        if "model" not in kwargs:
            raise Exception("Model is required.")
        self.set_vars(kwargs["model"])

        if os.getenv("LANGFUSE_BASE_URL") is not None:
            kwargs.update(
                dict(
                    success_callback=["langfuse_otel"],
                    metadata={
                        "session_id": "experiment-20251225-rankgpt",
                        "tags": [
                            kwargs.get("model", "unknown-model"),
                            "dl19",
                            "rankgpt-testing",
                        ],
                    },
                )
            )

        return await acompletion(
            api_key=self.api_key,
            api_base=self.base_url,
            extra_body=self.get_extra_kwargs(kwargs["model"]),
            timeout=API_TIMEOUT,
            *args,
            **kwargs,
        )
