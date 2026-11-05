"""List models available to the configured Anthropic API key."""

import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from .models import list_models


def main() -> int:
    """Print available model IDs."""
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Add it to .env or export it in your shell.",
            file=sys.stderr,
        )
        return 2

    model_ids = list_models(Anthropic())
    for model_id in model_ids:
        print(model_id)
    print(f"\n{len(model_ids)} models available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
