"""Helpers for creating and collecting Claude Message Batches."""

from anthropic import Anthropic


def create_message_batch(
    client: Anthropic,
    model: str,
    prompts: list[str],
    max_tokens: int = 128,
):
    """Submit one Messages API request per prompt as a batch."""
    if not prompts:
        raise ValueError("At least one prompt is required.")

    return client.messages.batches.create(
        requests=[
            {
                "custom_id": f"request_{index}",
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            for index, prompt in enumerate(prompts, start=1)
        ]
    )


def retrieve_batch(client: Anthropic, batch_id: str):
    """Retrieve batch processing status."""
    return client.messages.batches.retrieve(batch_id)


def iter_batch_results(client: Anthropic, batch_id: str):
    """Iterate over the completed batch's result stream."""
    return client.messages.batches.results(batch_id)
