"""Helpers for Claude model discovery and metadata."""

from anthropic import Anthropic


def list_models(client: Anthropic) -> list[str]:
    """Fetch every model returned by the Models API, following its cursors."""
    model_ids: list[str] = []
    after_id: str | None = None

    while True:
        page = client.models.list(limit=1000, after_id=after_id)
        model_ids.extend(model.id for model in page.data)
        if not page.has_more or not page.data:
            break
        after_id = page.data[-1].id

    return model_ids


def retrieve_model(client: Anthropic, model_id: str):
    """Retrieve metadata for one model."""
    return client.models.retrieve(model_id)
