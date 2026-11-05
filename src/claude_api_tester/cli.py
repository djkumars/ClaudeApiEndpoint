"""Command-line entry point for Claude API tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from anthropic import Anthropic, APIError

from .runner import ModelTestResult, exercise_models, list_models

DEFAULT_PROMPT = "Reply with exactly: Claude API test passed."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-api-tester",
        description="Discover Claude models or test their token-count and Messages APIs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-models", help="List model IDs available to your API key.")

    test_parser = commands.add_parser(
        "test-models",
        help="Test token counting and generate a short response for each model.",
    )
    test_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    test_parser.add_argument("--max-tokens", type=int, default=32)
    test_parser.add_argument("--concurrency", type=int, default=2)
    test_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON report. The parent directory must exist.",
    )
    return parser


def _summary(results: list[ModelTestResult]) -> dict[str, int]:
    endpoint_results = [
        endpoint
        for result in results
        for endpoint in (result.token_count, result.message)
    ]
    return {
        "models_tested": len(results),
        "endpoints_passed": sum(item.status == "passed" for item in endpoint_results),
        "endpoints_failed": sum(item.status == "failed" for item in endpoint_results),
    }


def _report(results: list[ModelTestResult]) -> dict[str, Any]:
    return {
        "summary": _summary(results),
        "results": [result.to_dict() for result in results],
    }


def _print_results(results: list[ModelTestResult]) -> None:
    print(f"{'MODEL':<48} {'TOKEN COUNT':<14} {'MESSAGE':<14}")
    print("-" * 78)
    for result in results:
        print(
            f"{result.model_id:<48} "
            f"{result.token_count.status:<14} {result.message.status:<14}"
        )
    summary = _summary(results)
    print(
        f"\nTested {summary['models_tested']} models: "
        f"{summary['endpoints_passed']} endpoint checks passed, "
        f"{summary['endpoints_failed']} failed."
    )


def main() -> int:
    load_dotenv()
    args = _parser().parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Add it to .env or export it in your shell.",
            file=sys.stderr,
        )
        return 2
    if args.command == "test-models":
        if args.max_tokens < 1:
            print("--max-tokens must be at least 1.", file=sys.stderr)
            return 2
        if args.concurrency < 1:
            print("--concurrency must be at least 1.", file=sys.stderr)
            return 2
        if not args.prompt.strip():
            print("--prompt must not be empty.", file=sys.stderr)
            return 2

    client = Anthropic()
    try:
        model_ids = list_models(client)
        if args.command == "list-models":
            for model_id in model_ids:
                print(model_id)
            print(f"\n{len(model_ids)} models available.")
            return 0

        print(
            f"Testing {len(model_ids)} models. Messages API calls may incur charges."
        )
        results = exercise_models(
            client=client,
            model_ids=model_ids,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
    except APIError as exc:
        print(f"Claude API request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_results(results)
    report = _report(results)
    if args.output:
        try:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Could not write report: {exc}", file=sys.stderr)
            return 1
        print(f"JSON report written to {args.output}")

    return 1 if report["summary"]["endpoints_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
