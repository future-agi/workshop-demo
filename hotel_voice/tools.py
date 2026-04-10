"""
Hotel Voice Agent - Tool Definitions.

Mock hotel database and standalone functions for room booking.
These are wrapped with LiveKit's @function_tool decorator in agent.py.
"""

import uuid
from typing import Optional

# =============================================================================
# Mock Database
# =============================================================================

ROOMS = {
    "standard": {
        "type": "Standard Room",
        "price_per_night": 120,
        "description": "Comfortable room with queen bed, city view, and work desk.",
        "capacity": 2,
        "amenities": ["Wi-Fi", "TV", "mini fridge", "coffee maker"],
        "available_count": 15,
    },
    "deluxe": {
        "type": "Deluxe Room",
        "price_per_night": 200,
        "description": "Spacious room with king bed, premium bedding, and panoramic view.",
        "capacity": 2,
        "amenities": ["Wi-Fi", "TV", "mini bar", "coffee maker", "bathrobe", "room service"],
        "available_count": 8,
    },
    "suite": {
        "type": "Suite",
        "price_per_night": 350,
        "description": "Luxury suite with separate living area, king bed, and premium amenities.",
        "capacity": 3,
        "amenities": ["Wi-Fi", "TV", "mini bar", "espresso machine", "bathrobe", "room service", "lounge access"],
        "available_count": 4,
    },
    "presidential": {
        "type": "Presidential Suite",
        "price_per_night": 600,
        "description": "Our finest accommodation with two bedrooms, dining area, and butler service.",
        "capacity": 4,
        "amenities": ["Wi-Fi", "TV", "full bar", "espresso machine", "bathrobe", "butler service", "lounge access", "spa access"],
        "available_count": 1,
    },
}

HOTEL_AMENITIES = [
    "Outdoor swimming pool (open 7am-10pm)",
    "Fitness center (24/7)",
    "Full-service spa",
    "Restaurant - The Grand Dining (breakfast, lunch, dinner)",
    "Rooftop bar - Skyline Lounge (5pm-midnight)",
    "Business center",
    "Concierge service",
    "Valet parking ($30/night)",
    "Airport shuttle ($25 each way)",
    "Free Wi-Fi throughout the hotel",
]

_reservations = {}


# =============================================================================
# Tool Functions (plain Python - wrapped with @function_tool in agent.py)
# =============================================================================


def check_room_availability(room_type: str, check_in: str, check_out: str) -> str:
    """Check if a room type is available for the given dates."""
    room = ROOMS.get(room_type.lower())
    if not room:
        types = ", ".join(ROOMS.keys())
        return f"Room type '{room_type}' not found. Available types: {types}"

    if room["available_count"] > 0:
        return (
            f"{room['type']} is available from {check_in} to {check_out}. "
            f"We have {room['available_count']} rooms of this type. "
            f"Rate: ${room['price_per_night']} per night."
        )
    return f"Sorry, {room['type']} is fully booked. Consider upgrading to a higher room type."


def get_room_prices(room_type: Optional[str] = None) -> str:
    """Get pricing for rooms. If no room_type specified, returns all prices."""
    if room_type:
        room = ROOMS.get(room_type.lower())
        if not room:
            return f"Room type '{room_type}' not found."
        return f"{room['type']}: ${room['price_per_night']} per night. Capacity: {room['capacity']} guests."

    lines = ["Room rates at The Grand Hotel:"]
    for room in ROOMS.values():
        lines.append(f"  {room['type']}: ${room['price_per_night']} per night (up to {room['capacity']} guests)")
    return "\n".join(lines)


def get_hotel_amenities() -> str:
    """Get the list of hotel amenities and facilities."""
    return "The Grand Hotel amenities:\n" + "\n".join(f"  - {a}" for a in HOTEL_AMENITIES)


def book_room(
    guest_name: str,
    room_type: str,
    check_in: str,
    check_out: str,
    num_guests: int = 1,
) -> str:
    """Book a room for a guest."""
    room = ROOMS.get(room_type.lower())
    if not room:
        return f"Room type '{room_type}' not found."
    if room["available_count"] <= 0:
        return f"Sorry, {room['type']} is fully booked."
    if num_guests > room["capacity"]:
        return f"{room['type']} fits up to {room['capacity']} guests. You need {num_guests}. Consider a larger room."

    reservation_id = f"RES-{uuid.uuid4().hex[:6].upper()}"
    _reservations[reservation_id] = {
        "id": reservation_id,
        "guest": guest_name,
        "room_type": room_type,
        "room_name": room["type"],
        "check_in": check_in,
        "check_out": check_out,
        "num_guests": num_guests,
        "price_per_night": room["price_per_night"],
    }
    ROOMS[room_type.lower()]["available_count"] -= 1

    return (
        f"Reservation confirmed!\n"
        f"  Reservation ID: {reservation_id}\n"
        f"  Guest: {guest_name}\n"
        f"  Room: {room['type']}\n"
        f"  Check-in: {check_in}\n"
        f"  Check-out: {check_out}\n"
        f"  Guests: {num_guests}\n"
        f"  Rate: ${room['price_per_night']} per night"
    )


def cancel_reservation(reservation_id: str) -> str:
    """Cancel an existing reservation."""
    res = _reservations.get(reservation_id.upper())
    if not res:
        return f"Reservation '{reservation_id}' not found."
    ROOMS[res["room_type"]]["available_count"] += 1
    del _reservations[reservation_id.upper()]
    return f"Reservation {reservation_id} for {res['guest']} has been cancelled. No charges applied."


def get_reservation_details(reservation_id: str) -> str:
    """Look up an existing reservation by its ID."""
    res = _reservations.get(reservation_id.upper())
    if not res:
        return f"Reservation '{reservation_id}' not found."
    return (
        f"Reservation {res['id']}:\n"
        f"  Guest: {res['guest']}\n"
        f"  Room: {res['room_name']}\n"
        f"  Check-in: {res['check_in']}\n"
        f"  Check-out: {res['check_out']}\n"
        f"  Guests: {res['num_guests']}\n"
        f"  Rate: ${res['price_per_night']}/night"
    )
