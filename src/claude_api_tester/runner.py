"""Claude API model discovery and endpoint tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from anthropic import APIError, Anthropic

from .models import list_models as list_models


@dataclass
class EndpointResult:
    status: str
    duration_seconds: float
    details: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ModelTestResult:
    model_id: str
    token_count: EndpointResult
    message: EndpointResult

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _call_endpoint(call: Any) -> EndpointResult:
    started = perf_counter()
    try:
        details = call()
    except APIError as exc:
        return EndpointResult(
            status="failed",
            duration_seconds=round(perf_counter() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
    return EndpointResult(
        status="passed",
        duration_seconds=round(perf_counter() - started, 3),
        details=details,
    )


def exercise_model(
    client: Anthropic,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> ModelTestResult:
    """Test token counting and a short Messages API response for one model."""
    messages = [{"role": "user", "content": prompt}]

    token_count = _call_endpoint(
        lambda: {
            "input_tokens": client.messages.count_tokens(
                model=model_id,
                messages=messages,
            ).input_tokens
        }
    )

    def create_message() -> dict[str, Any]:
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=messages,
        )
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        )
        return {
            "response_text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
        }

    message = _call_endpoint(create_message)
    return ModelTestResult(
        model_id=model_id,
        token_count=token_count,
        message=message,
    )


def exercise_models(
    client: Anthropic,
    model_ids: list[str],
    prompt: str,
    max_tokens: int,
    concurrency: int = 2,
) -> list[ModelTestResult]:
    """Test model endpoints concurrently and return results in input order."""
    results: dict[str, ModelTestResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(exercise_model, client, model_id, prompt, max_tokens): model_id
            for model_id in model_ids
        }
        for future in as_completed(futures):
            model_id = futures[future]
            results[model_id] = future.result()
    return [results[model_id] for model_id in model_ids]
