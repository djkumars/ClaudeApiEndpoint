import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from claude_api_tester import (
    admin,
    batch_processing,
    files,
    messages,
    models,
    prompt_management,
    streaming,
    token_counting,
    tool_function_calling,
    vision,
)


class ApiModuleTests(unittest.TestCase):
    def test_messages_and_token_counting_use_requested_model_and_prompt(self):
        client = Mock()

        messages.create_message(client, "claude-test", "Hello", max_tokens=20)
        token_counting.count_tokens(client, "claude-test", "Hello")

        self.assertEqual(
            client.messages.create.call_args.kwargs["messages"],
            [{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(
            client.messages.count_tokens.call_args.kwargs["model"],
            "claude-test",
        )

    def test_vision_encodes_image_and_rejects_unsupported_extensions(self):
        client = Mock()
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            image_path.write_bytes(b"image-bytes")

            vision.analyze_image(client, "claude-test", image_path)
            image_block = client.messages.create.call_args.kwargs["messages"][0][
                "content"
            ][0]
            self.assertEqual(image_block["source"]["media_type"], "image/png")
            self.assertEqual(
                base64.b64decode(image_block["source"]["data"]),
                b"image-bytes",
            )

            unsupported_path = Path(directory) / "document.pdf"
            unsupported_path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                vision.analyze_image(client, "claude-test", unsupported_path)

    def test_batch_processing_creates_requests_and_requires_prompts(self):
        client = Mock()

        batch_processing.create_message_batch(
            client,
            "claude-test",
            ["First", "Second"],
        )

        requests = client.messages.batches.create.call_args.kwargs["requests"]
        self.assertEqual([item["custom_id"] for item in requests], ["request_1", "request_2"])
        self.assertEqual(requests[1]["params"]["messages"][0]["content"], "Second")
        with self.assertRaises(ValueError):
            batch_processing.create_message_batch(client, "claude-test", [])

    def test_streaming_yields_text_chunks(self):
        client = Mock()
        stream = Mock()
        stream_manager = MagicMock()
        stream.text_stream = iter(["Hello", " Claude"])
        stream_manager.__enter__.return_value = stream
        client.messages.stream.return_value = stream_manager

        self.assertEqual(
            list(streaming.stream_response(client, "claude-test", "Say hello")),
            ["Hello", " Claude"],
        )

    def test_file_module_uses_files_beta(self):
        client = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.txt"
            path.write_text("test file", encoding="utf-8")
            files.upload_file(client, path)

        client.beta.files.upload.assert_called_once()
        self.assertEqual(
            client.beta.files.upload.call_args.kwargs["betas"],
            files.FILES_API_BETA,
        )

    def test_models_delegate_to_sdk_and_follow_pagination(self):
        client = Mock()
        client.models.list.side_effect = [
            SimpleNamespace(data=[SimpleNamespace(id="one")], has_more=True),
            SimpleNamespace(data=[SimpleNamespace(id="two")], has_more=False),
        ]

        self.assertEqual(models.list_models(client), ["one", "two"])
        client.models.retrieve.assert_not_called()
        models.retrieve_model(client, "one")
        client.models.retrieve.assert_called_once_with("one")

    def test_prompt_template_round_trip_and_execution(self):
        client = Mock()
        template = prompt_management.PromptTemplate(
            name="greeting",
            prompt="Say hello to {name}.",
            system="Be concise.",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "greeting.json"
            prompt_management.save_prompt(path, template)
            saved_payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = prompt_management.load_prompt(path)

        self.assertEqual(loaded, template)
        self.assertEqual(saved_payload["name"], "greeting")
        prompt_management.run_prompt(client, "claude-test", loaded, name="Taylor")
        call = client.messages.create.call_args.kwargs
        self.assertEqual(call["messages"][0]["content"], "Say hello to Taylor.")
        self.assertEqual(call["system"], "Be concise.")
        prompt_management.run_prompt(
            client,
            "claude-test",
            loaded,
            cache_system_prompt=True,
            name="Taylor",
        )
        cached_system = client.messages.create.call_args.kwargs["system"][0]
        self.assertEqual(cached_system["cache_control"]["type"], "ephemeral")

    def test_admin_helper_uses_workspace_read_endpoint(self):
        client = Mock()

        admin.list_workspaces(client)

        client.get.assert_called_once_with(
            "/v1/organizations/workspaces",
            cast_to=dict,
        )

    def test_tool_example_returns_tool_result_to_messages_api(self):
        client = Mock()
        tool_use = SimpleNamespace(
            type="tool_use",
            id="tool_123",
            input={"location": "Seattle"},
        )
        client.messages.create.side_effect = [
            SimpleNamespace(content=[tool_use]),
            SimpleNamespace(content=[]),
        ]

        tool_function_calling.run_tool_example(client, "claude-test")

        self.assertEqual(client.messages.create.call_count, 2)
        final_messages = client.messages.create.call_args.kwargs["messages"]
        self.assertEqual(final_messages[-1]["content"][0]["tool_use_id"], "tool_123")
        self.assertIn("Seattle", final_messages[-1]["content"][0]["content"])


if __name__ == "__main__":
    unittest.main()
