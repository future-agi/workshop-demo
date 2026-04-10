"""
Simple Chat Agent (State B) - With FutureAGI Simulate SDK.

Wraps the same chat agent in a Simulate SDK AgentWrapper, runs simulated
conversations with diverse personas, and prints transcripts.

Run: python simple_chat/agent_with_simulate.py
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_llm_client, get_model_name
from fi.simulate import (
    AgentWrapper,
    AgentInput,
    AgentResponse,
    Scenario,
    Persona,
)
from simulate_utils import run_local_simulation

# Same suboptimal prompt from agent.py
SYSTEM_PROMPT = "You are an AI assistant. Answer user questions. Provide information when asked."


class ChatAgentWrapper(AgentWrapper):
    """Wraps our simple chat agent for the Simulate SDK."""

    def __init__(self):
        self.client = get_llm_client()
        self.model = get_model_name()

    async def call(self, input: AgentInput) -> AgentResponse:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + input.messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return AgentResponse(
            content=response.choices[0].message.content,
            metadata={"model": self.model},
        )


# Test personas designed to expose prompt weaknesses
scenario = Scenario(
    name="simple-chat-test",
    description="Test the chat agent with diverse user personas",
    dataset=[
        Persona(
            persona={"name": "Priya", "age": 20, "role": "college student", "mood": "curious"},
            situation="She is studying quantum physics and wants a clear, concise explanation of quantum entanglement for her exam prep.",
            outcome="Receive a helpful, concise explanation that aids understanding without unnecessary jargon.",
        ),
        Persona(
            persona={"name": "Marcus", "age": 35, "role": "software engineer", "mood": "frustrated"},
            situation="His laptop keeps crashing during important meetings and he needs troubleshooting help urgently.",
            outcome="Get empathetic, step-by-step troubleshooting guidance that resolves or identifies the issue.",
        ),
        Persona(
            persona={"name": "Dorothy", "age": 72, "role": "retired teacher", "mood": "confused"},
            situation="She just got her first smartphone and doesn't understand how to send photos to her grandchildren.",
            outcome="Receive patient, simple instructions without technical jargon.",
        ),
        Persona(
            persona={"name": "Alex", "age": 28, "role": "startup CEO", "mood": "rushed"},
            situation="He needs a quick summary of the pros and cons of microservices vs monolith architecture for a board meeting in 10 minutes.",
            outcome="Get a rapid, bullet-point response without fluff or preamble.",
        ),
    ],
)


async def main():
    print("Running Simple Chat Agent simulation...")
    print(f"Model: {get_model_name()}")
    print(f"Personas: {len(scenario.dataset)}\n")

    wrapper = ChatAgentWrapper()

    report = await run_local_simulation(
        agent=wrapper,
        scenario=scenario,
        max_turns=4,
    )

    # Print results
    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    for result in report.results:
        persona = result.persona
        print(f"\nPersona: {persona.persona['name']} ({persona.persona['role']}, {persona.persona['mood']})")
        print(f"Situation: {persona.situation}")
        print(f"\nTranscript:\n{result.transcript}")
        print("-" * 60)

    return report


if __name__ == "__main__":
    asyncio.run(main())
