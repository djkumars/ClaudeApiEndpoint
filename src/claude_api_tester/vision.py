"""Helpers for image input through the Claude Messages API."""

import base64
import mimetypes
from pathlib import Path

from anthropic import Anthropic

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def analyze_image(
    client: Anthropic,
    model: str,
    image_path: str | Path,
    prompt: str = "Describe this image.",
    max_tokens: int = 256,
):
    """Send a local image and prompt to a vision-capable model."""
    path = Path(image_path)
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image type: {media_type or path.suffix}")

    image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
