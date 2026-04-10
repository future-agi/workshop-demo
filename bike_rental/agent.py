"""
Bike Rental Agent (State A) - LangChain agent with tool calling.

This agent uses a deliberately suboptimal system prompt to demonstrate
how FutureAGI evaluation can identify issues and optimization can improve it.

Flaws in current prompt:
- No guidance on WHEN to use tools vs just chatting
- No confirmation before booking
- No formatting guidance for prices
- No handling of unavailable items
- No upsell or recommendation behavior

Run: python bike_rental/agent.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from config import get_langchain_llm
from bike_rental.tools import ALL_TOOLS

# Deliberately suboptimal prompt
SYSTEM_PROMPT = (
    "You are a bike rental assistant. Help customers with bike rentals. "
    "You have access to tools for checking prices, availability, and making bookings."
)


def create_agent():
    llm = get_langchain_llm(temperature=0.7)
    agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)
    return agent


def chat():
    agent = create_agent()

    print("Bike Rental Agent")
    print("Type 'quit' or 'exit' to stop.\n")

    messages = []

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

        messages.append(HumanMessage(content=user_input))

        result = agent.invoke({"messages": messages})

        # Get the last AI message
        ai_messages = [m for m in result["messages"] if hasattr(m, 'content') and m.type == "ai" and m.content]
        if ai_messages:
            response = ai_messages[-1].content
            messages = result["messages"]
            print(f"\nAssistant: {response}\n")
        else:
            print("\nAssistant: I'm processing your request...\n")


if __name__ == "__main__":
    chat()
