import httpx
import litellm
from .logging_utils import logger
import asyncio
from functools import wraps


def _is_retryable_error(e):
    if isinstance(
        e,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            litellm.APIError,
        ),
    ) or str(e).startswith("No response"):
        return True

    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code == 429 or e.response.status_code >= 500

    error_name = type(e).__name__
    if (
        "Timeout" in error_name
        or "RateLimit" in error_name
        or "APIConnectionError" in error_name
    ):
        return True

    if hasattr(e, "status_code"):
        return e.status_code == 429 or e.status_code >= 500

    error_msg = str(e).lower()
    if any(
        keyword in error_msg
        for keyword in ["timeout", "connection", "rate limit", "overload"]
    ):
        return True

    return False


MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
BACKOFF_FACTOR = 2.0


def async_retry(
    max_retries=MAX_RETRIES,
    initial_delay=INITIAL_RETRY_DELAY,
    backoff_factor=BACKOFF_FACTOR,
):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if _is_retryable_error(e) and attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} retry {attempt + 1}/{max_retries} after {type(e).__name__}: {str(e)}"
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        if not _is_retryable_error(e):
                            logger.error(
                                f"{func.__name__} failed with non-retryable error: {type(e).__name__}: {str(e)}"
                            )
                        raise

        return wrapper

    return decorator
