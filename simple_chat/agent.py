"""
Simple Chat Agent (State A) - Basic LLM chat loop.

This agent uses a deliberately suboptimal system prompt to demonstrate
how FutureAGI evaluation can identify issues and optimization can improve it.

Flaws in current prompt:
- Vague identity (no personality or guardrails)
- No conciseness guidance (agent will ramble)
- No tone direction (responses feel robotic)
- No follow-up behavior (just answers, doesn't guide)

Run: python simple_chat/agent.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_llm_client, get_model_name

# Deliberately suboptimal prompt - evaluation will catch these issues
SYSTEM_PROMPT = "You are an AI assistant. Answer user questions. Provide information when asked."


def chat():
    client = get_llm_client()
    model = get_model_name()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Simple Chat Agent")
    print(f"Model: {model}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        assistant_msg = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_msg})

        print(f"\nAssistant: {assistant_msg}\n")


if __name__ == "__main__":
    chat()
