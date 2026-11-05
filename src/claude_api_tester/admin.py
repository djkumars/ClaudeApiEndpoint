"""Minimal read-only Anthropic Admin API example.

Use an Admin API key, not a regular Anthropic API key.
"""

from typing import Any

from anthropic import Anthropic


def list_workspaces(admin_client: Anthropic) -> dict[str, Any]:
    """List organization workspaces using an Admin API client."""
    return admin_client.get(
        "/v1/organizations/workspaces",
        cast_to=dict,
    )
