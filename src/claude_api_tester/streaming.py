"""Helpers for streaming text from the Claude Messages API."""

from collections.abc import Iterator

from anthropic import Anthropic


def stream_response(
    client: Anthropic,
    model: str,
    prompt: str,
    max_tokens: int = 256,
) -> Iterator[str]:
    """Yield response text as the model generates it."""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        yield from stream.text_stream
