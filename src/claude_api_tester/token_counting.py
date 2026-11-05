"""Helpers for counting Claude message input tokens."""

from anthropic import Anthropic


def count_tokens(
    client: Anthropic,
    model: str,
    prompt: str,
):
    """Count input tokens without generating a response."""
    return client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
