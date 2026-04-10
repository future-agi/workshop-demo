"""
Hotel Voice Agent (State A) - LiveKit voice agent for hotel booking.

This agent uses a deliberately suboptimal system prompt to demonstrate
how FutureAGI evaluation can identify issues and optimization can improve it.

Flaws in current prompt:
- Uses bullet points/markdown formatting (bad for voice/speech)
- Encourages verbose "complete details" dumps
- No conciseness guidance for voice
- No clarification behavior for names/dates
- No interruption handling guidance

Run: python hotel_voice/agent.py dev
"""

import sys
import os
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from livekit import agents
from livekit.agents import llm, function_tool
from livekit.plugins import openai as lk_openai, silero, deepgram

from hotel_voice.tools import (
    check_room_availability,
    get_room_prices,
    get_hotel_amenities,
    book_room,
    cancel_reservation,
    get_reservation_details,
)
from config import get_livekit_llm_plugin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deliberately suboptimal prompt - uses text formatting, too verbose for voice
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


# =============================================================================
# LiveKit Function Tools (wrap the standalone functions from tools.py)
# =============================================================================


@function_tool()
async def tool_check_room_availability(room_type: str, check_in: str, check_out: str) -> str:
    """Check if a room type is available for given dates. Room types: standard, deluxe, suite, presidential."""
    return check_room_availability(room_type, check_in, check_out)


@function_tool()
async def tool_get_room_prices(room_type: Optional[str] = None) -> str:
    """Get pricing for a specific room type or all rooms. Room types: standard, deluxe, suite, presidential."""
    return get_room_prices(room_type)


@function_tool()
async def tool_get_hotel_amenities() -> str:
    """Get the list of hotel amenities and facilities."""
    return get_hotel_amenities()


@function_tool()
async def tool_book_room(
    guest_name: str,
    room_type: str,
    check_in: str,
    check_out: str,
    num_guests: int = 1,
) -> str:
    """Book a hotel room. Room types: standard, deluxe, suite, presidential. Dates in YYYY-MM-DD format."""
    return book_room(guest_name, room_type, check_in, check_out, num_guests)


@function_tool()
async def tool_cancel_reservation(reservation_id: str) -> str:
    """Cancel an existing reservation by its ID (e.g., RES-ABC123)."""
    return cancel_reservation(reservation_id)


@function_tool()
async def tool_get_reservation_details(reservation_id: str) -> str:
    """Look up an existing reservation by its ID."""
    return get_reservation_details(reservation_id)


TOOLS = [
    tool_check_room_availability,
    tool_get_room_prices,
    tool_get_hotel_amenities,
    tool_book_room,
    tool_cancel_reservation,
    tool_get_reservation_details,
]


# =============================================================================
# LiveKit Agent Setup
# =============================================================================

server = agents.AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Agent session starting in room: {ctx.room.name}")

    await ctx.connect()

    # Build chat context with system prompt
    initial_ctx = llm.ChatContext(
        messages=[
            llm.ChatMessage(role="system", content=SYSTEM_PROMPT),
        ]
    )

    # Configure voice pipeline
    agent_llm = get_livekit_llm_plugin()
    agent_stt = deepgram.STT(model="nova-3")
    agent_tts = lk_openai.TTS(voice="alloy")
    agent_vad = silero.VAD.load()

    # Create voice assistant
    assistant = agents.VoiceAssistant(
        vad=agent_vad,
        stt=agent_stt,
        llm=agent_llm,
        tts=agent_tts,
        chat_ctx=initial_ctx,
    )

    # Register tools
    assistant.set_tools(TOOLS)

    # Start the agent
    assistant.start(ctx)
    logger.info("Hotel voice agent is ready.")

    # Greet the caller
    assistant.say("Welcome to The Grand Hotel! How may I help you today?")

    await assistant.wait()


if __name__ == "__main__":
    agents.cli.run_app(server)
