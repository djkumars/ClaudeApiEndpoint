"""A safe, deterministic client-tool/function-calling example."""

from anthropic import Anthropic

WEATHER_TOOL = {
    "name": "get_demo_weather",
    "description": "Get fixed demo weather data for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


def run_tool_example(
    client: Anthropic,
    model: str,
    prompt: str = "What is the demo weather in Seattle?",
    max_tokens: int = 256,
):
    """Request a tool call, return a fixed demo result, and continue the turn."""
    messages = [{"role": "user", "content": prompt}]
    first_response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=[WEATHER_TOOL],
        tool_choice={"type": "tool", "name": WEATHER_TOOL["name"]},
        messages=messages,
    )
    tool_uses = [
        block for block in first_response.content if block.type == "tool_use"
    ]
    if not tool_uses:
        return first_response

    messages.extend(
        [
            {"role": "assistant", "content": first_response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": (
                            "Demo fixture: clear skies, 20 C in "
                            f"{tool_use.input['location']}."
                        ),
                    }
                    for tool_use in tool_uses
                ],
            },
        ]
    )
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=[WEATHER_TOOL],
        messages=messages,
    )
