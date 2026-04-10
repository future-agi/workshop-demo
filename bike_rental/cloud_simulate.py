"""
Bike Rental Agent - Cloud Simulation via FutureAGI Platform.

Uses the FutureAGI TestRunner in cloud mode: scenarios and personas are
managed on the platform, and the agent callback runs locally while the
orchestration happens in the cloud.

Prerequisites:
    1. Set FI_API_KEY and FI_SECRET_KEY in .env
    2. Create a test run named "bike-rental-cloud" on the FutureAGI platform

Run: python bike_rental/cloud_simulate.py
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

from config import get_langchain_llm
from bike_rental.tools import ALL_TOOLS
from fi.simulate import (
    AgentWrapper,
    AgentInput,
    AgentResponse,
    TestRunner,
)

SYSTEM_PROMPT = (
    "You are a bike rental assistant. Help customers with bike rentals. "
    "You have access to tools for checking prices, availability, and making bookings."
)

TEST_NAME = "bike-rental-test"


class BikeRentalAgentWrapper(AgentWrapper):
    """Wraps the LangChain bike rental agent for cloud simulation."""

    def __init__(self):
        llm = get_langchain_llm(temperature=0.7)
        self.agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)

    async def call(self, input: AgentInput) -> AgentResponse:
        lc_messages = []
        for msg in input.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        result = self.agent.invoke({"messages": lc_messages})

        ai_messages = [
            m
            for m in result["messages"]
            if hasattr(m, "type") and m.type == "ai" and m.content
        ]
        content = (
            ai_messages[-1].content if ai_messages else "I'm processing your request."
        )

        return AgentResponse(content=content)


async def main():
    print(f"Running cloud simulation: {TEST_NAME}")
    print("Connecting to FutureAGI platform...\n")

    runner = TestRunner()
    wrapper = BikeRentalAgentWrapper()

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
