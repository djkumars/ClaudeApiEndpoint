from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from claude_api_tester.runner import exercise_model, list_models


class RunnerTests(unittest.TestCase):
    def test_list_models_follows_cursors(self):
        client = Mock()
        client.models.list.side_effect = [
            SimpleNamespace(
                data=[SimpleNamespace(id="claude-one")],
                has_more=True,
            ),
            SimpleNamespace(
                data=[SimpleNamespace(id="claude-two")],
                has_more=False,
            ),
        ]

        self.assertEqual(list_models(client), ["claude-one", "claude-two"])
        self.assertEqual(
            client.models.list.call_args_list[1].kwargs["after_id"],
            "claude-one",
        )

    def test_exercise_model_checks_token_count_and_message(self):
        client = Mock()
        client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=8)
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello from Claude")],
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
            stop_reason="end_turn",
        )

        result = exercise_model(client, "claude-one", "Say hello", 16)

        self.assertEqual(result.token_count.status, "passed")
        self.assertEqual(result.token_count.details, {"input_tokens": 8})
        self.assertEqual(result.message.status, "passed")
        self.assertEqual(
            result.message.details["response_text"],
            "Hello from Claude",
        )
        client.messages.create.assert_called_once()
