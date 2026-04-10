"""
Bike Rental Agent (State B) - With FutureAGI Simulate SDK.

Wraps the LangChain bike rental agent with a custom AgentWrapper and runs
simulated conversations with diverse customer personas.

Run: python bike_rental/agent_with_simulate.py
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
    Scenario,
    Persona,
)
from simulate_utils import run_local_simulation

# Same suboptimal prompt from agent.py
SYSTEM_PROMPT = (
    "You are a bike rental assistant. Help customers with bike rentals. "
    "You have access to tools for checking prices, availability, and making bookings."
)


class BikeRentalAgentWrapper(AgentWrapper):
    """Wraps the LangChain bike rental agent for the Simulate SDK."""

    def __init__(self):
        llm = get_langchain_llm(temperature=0.7)
        self.agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)

    async def call(self, input: AgentInput) -> AgentResponse:
        # Convert messages to LangChain format
        lc_messages = []
        for msg in input.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        result = self.agent.invoke({"messages": lc_messages})

        # Get last AI message with content
        ai_messages = [m for m in result["messages"] if hasattr(m, 'type') and m.type == "ai" and m.content]
        content = ai_messages[-1].content if ai_messages else "I'm processing your request."

        return AgentResponse(content=content)


# Test personas that expose prompt weaknesses
scenario = Scenario(
    name="bike-rental-test",
    description="Test the bike rental agent with diverse customer personas",
    dataset=[
        Persona(
            persona={"name": "Sofia", "age": 24, "role": "tourist", "mood": "budget-conscious"},
            situation="She wants the cheapest bike available for 3 hours tomorrow to explore the city.",
            outcome="Agent uses tools to find cheapest option, shows price clearly, and completes booking after confirming details.",
        ),
        Persona(
            persona={"name": "James", "age": 42, "role": "family dad", "mood": "planning"},
            situation="He needs 4 bikes for his family for a full day trip. His kids are ages 8 and 12.",
            outcome="Agent recommends appropriate bikes, calculates total cost for 4, and handles the multi-bike booking.",
        ),
        Persona(
            persona={"name": "Mika", "age": 30, "role": "cyclist", "mood": "enthusiastic"},
            situation="She wants the best road bike for a 50km ride this weekend. She cares about specs and performance.",
            outcome="Agent recommends the Road Bike Elite, shows detailed specs, and confirms it meets her needs before booking.",
        ),
        Persona(
            persona={"name": "Tom", "age": 55, "role": "first-timer", "mood": "confused"},
            situation="He has never rented a bike before and doesn't know what type he needs. He just wants to ride around the park for a couple hours.",
            outcome="Agent guides him through options, recommends the City Cruiser for casual riding, and explains the rental process.",
        ),
    ],
)


async def main():
    print("Running Bike Rental Agent simulation...")
    print(f"Personas: {len(scenario.dataset)}\n")

    wrapper = BikeRentalAgentWrapper()

    report = await run_local_simulation(
        agent=wrapper,
        scenario=scenario,
        max_turns=5,
    )

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
