"""
Hotel Voice Agent (State B) - With FutureAGI Simulate SDK.

Uses the Simulate SDK's LiveKit engine to run simulated voice conversations
with the deployed hotel voice agent. The simulator plays the role of different
caller personas.

Prerequisites:
- The voice agent must be running: python hotel_voice/agent.py dev
- LiveKit credentials must be configured in .env

Run: python hotel_voice/agent_with_simulate.py
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from fi.simulate import (
    AgentDefinition,
    SimulatorAgentDefinition,
    TestRunner,
    Scenario,
    Persona,
)

# Same suboptimal prompt from agent.py
SYSTEM_PROMPT = """You are a hotel receptionist at The Grand Hotel. Help guests with:
- Room availability and pricing
- Making reservations
- Cancellations and modifications
- Hotel amenities information

Use the available tools to look up information. Be professional and helpful.
Always provide complete details including:
* Room type options
* Pricing breakdowns
* Available dates
* Amenity lists"""

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://localhost:7880")
TEST_ROOM = "hotel-test-room"

# Define the deployed voice agent
agent_definition = AgentDefinition(
    name="Hotel Receptionist",
    url=LIVEKIT_URL,
    room_name=TEST_ROOM,
    system_prompt=SYSTEM_PROMPT,
)

# Configure the simulated caller
simulator = SimulatorAgentDefinition(
    name="sim-hotel-caller",
    instructions=(
        "You are a realistic hotel caller. Keep your responses natural and conversational. "
        "Ask clarifying questions when needed. When satisfied, say 'Thank you, that's all I need.'"
    ),
    llm={"model": "gpt-4o-mini", "temperature": 0.6},
    tts={"model": "tts-1", "voice": "echo"},
    stt={"language": "en"},
    vad={"provider": "silero"},
    allow_interruptions=True,
    min_endpointing_delay=0.3,
    max_endpointing_delay=4.0,
)

# Test personas that expose voice-specific prompt weaknesses
scenario = Scenario(
    name="hotel-voice-test",
    description="Test the hotel voice agent with diverse caller personas",
    dataset=[
        Persona(
            persona={"name": "David Chen", "age": 45, "role": "business traveler", "mood": "efficient"},
            situation="He needs a suite for next Thursday through Sunday for a business trip. Wants to know about business center and airport shuttle.",
            outcome="Quick booking with minimal back-and-forth. Agent speaks concisely and confirms details.",
        ),
        Persona(
            persona={"name": "Maria Santos", "age": 32, "role": "planning anniversary", "mood": "excited"},
            situation="She's planning a surprise anniversary weekend and wants the best room with romantic amenities. Budget isn't a huge concern.",
            outcome="Agent recommends Presidential or Suite, mentions spa and restaurant. Warm, celebratory tone.",
        ),
        Persona(
            persona={"name": "Ryan O'Brien", "age": 22, "role": "budget traveler", "mood": "price-sensitive"},
            situation="He wants the cheapest room available for two nights. Asks about any discounts or deals.",
            outcome="Agent quotes Standard Room price clearly, no overwhelming details. Concise for voice.",
        ),
        Persona(
            persona={"name": "Dr. Priya Patel", "age": 50, "role": "conference organizer", "mood": "detail-oriented"},
            situation="She needs to book 5 rooms for a medical conference next month. Wants group info and meeting facilities.",
            outcome="Agent handles multi-room request, clarifies room types and dates, provides clear pricing summary.",
        ),
    ],
)


async def main():
    print("Running Hotel Voice Agent simulation...")
    print(f"LiveKit URL: {LIVEKIT_URL}")
    print(f"Test Room: {TEST_ROOM}")
    print(f"Personas: {len(scenario.dataset)}\n")
    print("NOTE: Make sure the voice agent is running: python hotel_voice/agent.py dev\n")

    runner = TestRunner()

    report = await runner.run_test(
        agent_definition=agent_definition,
        scenario=scenario,
        simulator=simulator,
        record_audio=True,
        recorder_sample_rate=8000,
    )

    print("=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    for result in report.results:
        persona = result.persona
        print(f"\nPersona: {persona.persona['name']} ({persona.persona['role']})")
        print(f"Situation: {persona.situation}")
        print(f"\nTranscript:\n{result.transcript}")

        if result.audio_combined_path:
            print(f"Audio: {result.audio_combined_path}")
        print("-" * 60)

    return report


if __name__ == "__main__":
    asyncio.run(main())
