#!/usr/bin/env python3
"""LearnForge support assistant: retrieval, grounded answering, and escalation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from learnforge import Assistant, Conversation, KnowledgeBase, OpenAICompatibleGenerator

ROOT = Path(__file__).parent
DEFAULT_DATA = Path(os.environ.get(
    "KNOWLEDGE_BASE_DIR",
    "/home/hussain/Downloads/learnforge-knowledge-base-data/learnforge-knowledge-base",
))


def build_assistant(data_dir: Path) -> Assistant:
    kb = KnowledgeBase.from_markdown_dir(data_dir)
    generator = OpenAICompatibleGenerator.from_environment()
    return Assistant(kb, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LearnForge support RAG prototype.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--question", help="Answer one question and exit.")
    args = parser.parse_args()
    assistant = build_assistant(args.data_dir)
    conversation = Conversation()

    def reply(question: str) -> None:
        result = assistant.answer(question, conversation)
        print(f"\n{result.answer}\n")
        print(f"Decision: {result.decision} | confidence: {result.confidence:.0%}")
        print("Sources: " + ", ".join(result.sources))

    if args.question:
        reply(args.question)
        return
    print("LearnForge assistant. Type 'quit' to exit.")
    while (question := input("\nYou: ").strip().lower()) not in {"quit", "exit"}:
        if question:
            reply(question)


if __name__ == "__main__":
    main()
