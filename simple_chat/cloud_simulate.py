"""
Simple Chat Agent - Cloud Simulation via FutureAGI Platform.

Uses the FutureAGI TestRunner in cloud mode: scenarios and personas are
managed on the platform, and the agent callback runs locally while the
orchestration happens in the cloud.

Prerequisites:
    1. Set FI_API_KEY and FI_SECRET_KEY in .env
    2. Create a test run named "simple-chat-cloud" on the FutureAGI platform

Run: python simple_chat/cloud_simulate.py
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
    TestRunner,
)

SYSTEM_PROMPT = "You are an AI assistant. Answer user questions. Provide information when asked."

TEST_NAME = "simple-chat-cloud"


class ChatAgentWrapper(AgentWrapper):
    """Wraps the simple chat agent for cloud simulation."""

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


async def main():
    print(f"Running cloud simulation: {TEST_NAME}")
    print(f"Model: {get_model_name()}")
    print("Connecting to FutureAGI platform...\n")

    runner = TestRunner()
    wrapper = ChatAgentWrapper()

    report = await runner.run_test(
        run_test_name=TEST_NAME,
        agent_callback=wrapper.call,
    )

    print("=" * 60)
    print("CLOUD SIMULATION RESULTS")
    print("=" * 60)

    for result in report.results:
        persona = result.persona
        name = persona.persona.get("name", "Unknown") if persona else "Cloud Persona"
        print(f"\nPersona: {name}")
        print(f"\nTranscript:\n{result.transcript}")
        print("-" * 60)

    return report


if __name__ == "__main__":
    asyncio.run(main())
