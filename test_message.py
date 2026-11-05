"""Make a single Claude Messages API request using the configured API key."""

import os
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

from claude_api_tester.messages import create_message
from claude_api_tester.models import list_models


def main() -> None:
    project_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=project_env)
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit(
            "ANTHROPIC_API_KEY is missing. Copy .env.example to .env and "
            "replace the placeholder with your Anthropic API key."
        )

    client = Anthropic()

    available_models = list_models(client)
    if not available_models:
        raise RuntimeError("No models are available for this API key.")

    model = available_models[0]
    response = create_message(
        client=client,
        model=model,
        prompt="Reply with exactly: Message API works.",
        max_tokens=32,
    )

    print(f"Model: {model}")
    for block in response.content:
        if block.type == "text":
            print(block.text)
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()
