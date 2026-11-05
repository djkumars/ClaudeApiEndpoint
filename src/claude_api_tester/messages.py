"""Helpers for the Claude Messages API."""

from anthropic import Anthropic


def create_message(
    client: Anthropic,
    model: str,
    prompt: str,
    max_tokens: int = 128,
):
    """Create one text message and return the SDK response."""
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
