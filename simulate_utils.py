"""
Local simulation utility.

Runs simulated multi-turn conversations without requiring the FutureAGI
platform. Uses an LLM to play the persona role and the agent wrapper to
respond.

For production use, use the FutureAGI platform's cloud mode (run_test_name).
This utility is for local development and workshop demos.
"""

import asyncio
from typing import Optional

from openai import OpenAI

from fi.simulate import (
    AgentWrapper,
    AgentInput,
    AgentResponse,
    Scenario,
    Persona,
    TestReport,
    TestCaseResult,
)


async def run_local_simulation(
    agent: AgentWrapper,
    scenario: Scenario,
    max_turns: int = 6,
    simulator_model: str = "gpt-4o-mini",
) -> TestReport:
    """
    Run a local simulation: an LLM plays each persona, the agent responds.

    Args:
        agent: Your AgentWrapper implementation
        scenario: Scenario with personas to test
        max_turns: Max conversation turns per persona
        simulator_model: Model for the simulated user

    Returns:
        TestReport with transcripts for each persona
    """
    client = OpenAI()
    results = []

    for persona in scenario.dataset:
        print(f"  Simulating: {persona.persona.get('name', 'Unknown')}...")

        # Build simulator system prompt from persona
        sim_prompt = (
            f"You are role-playing as a customer. Stay in character.\n"
            f"Your persona: {persona.persona}\n"
            f"Your situation: {persona.situation}\n"
            f"Your desired outcome: {persona.outcome}\n\n"
            f"Rules:\n"
            f"- Be natural and conversational\n"
            f"- Stay in character based on your mood and role\n"
            f"- After 4-5 exchanges, wrap up the conversation naturally\n"
            f"- Keep messages concise (1-3 sentences)\n"
            f"- Start by describing your need"
        )

        messages = []  # Conversation history for the agent
        sim_messages = [{"role": "system", "content": sim_prompt}]
        transcript_lines = []

        for turn in range(max_turns):
            # Simulator generates user message
            sim_response = client.chat.completions.create(
                model=simulator_model,
                messages=sim_messages,
                max_tokens=200,
            )
            user_msg = sim_response.choices[0].message.content

            # Check if conversation is ending
            if any(phrase in user_msg.lower() for phrase in ["goodbye", "bye", "that's all", "thanks, that"]):
                transcript_lines.append(f"User: {user_msg}")
                break

            messages.append({"role": "user", "content": user_msg})
            transcript_lines.append(f"User: {user_msg}")

            # Agent responds
            agent_input = AgentInput(
                thread_id=f"sim-{persona.persona.get('name', 'unknown')}",
                messages=messages,
                new_message={"role": "user", "content": user_msg},
            )

            response = await agent.call(agent_input)

            if isinstance(response, str):
                agent_msg = response
            else:
                agent_msg = response.content

            messages.append({"role": "assistant", "content": agent_msg})
            transcript_lines.append(f"Agent: {agent_msg}")

            # Update simulator context
            sim_messages.append({"role": "assistant", "content": user_msg})
            sim_messages.append({"role": "user", "content": f"The agent replied: {agent_msg}"})

        transcript = "\n\n".join(transcript_lines)
        results.append(TestCaseResult(persona=persona, transcript=transcript))

    return TestReport(results=results)
