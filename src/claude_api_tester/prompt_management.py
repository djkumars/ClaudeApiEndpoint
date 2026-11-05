"""Local prompt-template management and Messages API execution.

The Anthropic API does not provide a hosted CRUD API for prompt templates.
This module stores templates locally; prompt caching is configured separately
on Messages API requests.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from anthropic import Anthropic


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    prompt: str
    system: str | None = None

    def render(self, **variables: str) -> str:
        """Format the user prompt with the supplied named variables."""
        return self.prompt.format(**variables)


def save_prompt(path: str | Path, template: PromptTemplate) -> Path:
    """Save a prompt template as JSON at the requested path."""
    target = Path(path)
    target.write_text(json.dumps(asdict(template), indent=2) + "\n", encoding="utf-8")
    return target


def load_prompt(path: str | Path) -> PromptTemplate:
    """Load a prompt template from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PromptTemplate(**payload)


def run_prompt(
    client: Anthropic,
    model: str,
    template: PromptTemplate,
    max_tokens: int = 256,
    cache_system_prompt: bool = False,
    **variables: str,
):
    """Render and send a local template, optionally caching its system prompt."""
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": template.render(**variables)}],
    }
    if template.system is not None:
        system = template.system.format(**variables)
        request["system"] = (
            [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if cache_system_prompt
            else system
        )
    elif cache_system_prompt:
        raise ValueError("A system prompt is required to enable prompt caching.")
    return client.messages.create(**request)
