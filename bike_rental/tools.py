"""
Bike Rental Agent - Tool Definitions.

Mock database and LangChain tools for a bike rental shop.
"""

import uuid
from typing import Optional
from langchain_core.tools import tool

# =============================================================================
# Mock Database
# =============================================================================

VEHICLES = {
    "MTB-001": {
        "id": "MTB-001",
        "name": "Mountain Bike Pro",
        "type": "mountain_bike",
        "price_per_hour": 8,
        "price_per_day": 45,
        "description": "Rugged mountain bike with front suspension, 21-speed gears. Great for trails.",
        "available": True,
        "specs": {"frame": "aluminum", "wheels": "26 inch", "gears": 21, "weight": "13 kg"},
    },
    "CTY-002": {
        "id": "CTY-002",
        "name": "City Cruiser",
        "type": "city_bike",
        "price_per_hour": 6,
        "price_per_day": 35,
        "description": "Comfortable city bike with basket and bell. Perfect for sightseeing.",
        "available": True,
        "specs": {"frame": "steel", "wheels": "28 inch", "gears": 7, "weight": "15 kg"},
    },
    "ESC-003": {
        "id": "ESC-003",
        "name": "Electric Scooter X",
        "type": "electric_scooter",
        "price_per_hour": 10,
        "price_per_day": 55,
        "description": "Fast electric scooter with 30km range. Max speed 25 km/h.",
        "available": True,
        "specs": {"range": "30 km", "max_speed": "25 km/h", "charge_time": "3 hours", "weight": "12 kg"},
    },
    "RDB-004": {
        "id": "RDB-004",
        "name": "Road Bike Elite",
        "type": "road_bike",
        "price_per_hour": 12,
        "price_per_day": 65,
        "description": "Lightweight carbon road bike for serious cyclists. 22-speed Shimano groupset.",
        "available": True,
        "specs": {"frame": "carbon", "wheels": "700c", "gears": 22, "weight": "8 kg"},
    },
    "EBK-005": {
        "id": "EBK-005",
        "name": "E-Bike Comfort",
        "type": "e_bike",
        "price_per_hour": 15,
        "price_per_day": 80,
        "description": "Pedal-assist e-bike with 60km range. Effortless riding for all fitness levels.",
        "available": False,
        "specs": {"range": "60 km", "motor": "250W", "gears": 9, "weight": "22 kg"},
    },
    "TND-006": {
        "id": "TND-006",
        "name": "Tandem Fun Ride",
        "type": "tandem_bike",
        "price_per_hour": 18,
        "price_per_day": 95,
        "description": "Two-person tandem bike. Great for couples and friends. Smooth ride.",
        "available": True,
        "specs": {"frame": "steel", "wheels": "26 inch", "gears": 21, "weight": "20 kg", "seats": 2},
    },
}

_bookings = {}


# =============================================================================
# LangChain Tools
# =============================================================================


@tool
def get_vehicle_list() -> str:
    """Get the list of all available vehicles with basic info and pricing."""
    lines = []
    for v in VEHICLES.values():
        status = "Available" if v["available"] else "Currently Unavailable"
        lines.append(
            f"- {v['name']} ({v['id']}): ${v['price_per_hour']}/hr, "
            f"${v['price_per_day']}/day - {status}"
        )
    return "Our vehicles:\n" + "\n".join(lines)


@tool
def get_vehicle_price(vehicle_id: str) -> str:
    """Get detailed pricing for a specific vehicle by its ID (e.g., 'MTB-001')."""
    vehicle = VEHICLES.get(vehicle_id.upper())
    if not vehicle:
        return f"Vehicle '{vehicle_id}' not found. Use get_vehicle_list to see available IDs."
    return (
        f"{vehicle['name']} ({vehicle['id']}) pricing:\n"
        f"  Hourly: ${vehicle['price_per_hour']}/hour\n"
        f"  Daily: ${vehicle['price_per_day']}/day\n"
        f"  Note: Daily rate applies for 6+ hours."
    )


@tool
def check_availability(vehicle_id: str, date: str) -> str:
    """Check if a vehicle is available on a specific date. Date format: YYYY-MM-DD."""
    vehicle = VEHICLES.get(vehicle_id.upper())
    if not vehicle:
        return f"Vehicle '{vehicle_id}' not found."
    if not vehicle["available"]:
        return f"{vehicle['name']} is currently unavailable for maintenance. Try another vehicle."
    return f"{vehicle['name']} is available on {date}."


@tool
def get_vehicle_details(vehicle_id: str) -> str:
    """Get full specifications and details for a vehicle by its ID."""
    vehicle = VEHICLES.get(vehicle_id.upper())
    if not vehicle:
        return f"Vehicle '{vehicle_id}' not found."
    specs = ", ".join(f"{k}: {v}" for k, v in vehicle["specs"].items())
    return (
        f"{vehicle['name']} ({vehicle['type']})\n"
        f"  Description: {vehicle['description']}\n"
        f"  Specs: {specs}\n"
        f"  Price: ${vehicle['price_per_hour']}/hr, ${vehicle['price_per_day']}/day\n"
        f"  Status: {'Available' if vehicle['available'] else 'Unavailable'}"
    )


@tool
def book_vehicle(vehicle_id: str, customer_name: str, date: str, duration_hours: int) -> str:
    """
    Book a vehicle for a customer.

    Args:
        vehicle_id: The vehicle ID (e.g., 'MTB-001')
        customer_name: Customer's full name
        date: Rental date (YYYY-MM-DD)
        duration_hours: How many hours to rent
    """
    vehicle = VEHICLES.get(vehicle_id.upper())
    if not vehicle:
        return f"Vehicle '{vehicle_id}' not found."
    if not vehicle["available"]:
        return f"{vehicle['name']} is currently unavailable."

    # Calculate price
    if duration_hours >= 6:
        days = duration_hours // 8 or 1
        total = days * vehicle["price_per_day"]
        pricing = f"{days} day(s) x ${vehicle['price_per_day']}/day"
    else:
        total = duration_hours * vehicle["price_per_hour"]
        pricing = f"{duration_hours} hr(s) x ${vehicle['price_per_hour']}/hr"

    booking_id = f"BK-{uuid.uuid4().hex[:6].upper()}"
    _bookings[booking_id] = {
        "id": booking_id,
        "vehicle_id": vehicle_id,
        "vehicle_name": vehicle["name"],
        "customer": customer_name,
        "date": date,
        "duration_hours": duration_hours,
        "total": total,
    }

    return (
        f"Booking confirmed!\n"
        f"  Booking ID: {booking_id}\n"
        f"  Vehicle: {vehicle['name']}\n"
        f"  Customer: {customer_name}\n"
        f"  Date: {date}\n"
        f"  Duration: {duration_hours} hours\n"
        f"  Pricing: {pricing}\n"
        f"  Total: ${total}"
    )


@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking by its booking ID (e.g., 'BK-ABC123')."""
    booking = _bookings.get(booking_id.upper())
    if not booking:
        return f"Booking '{booking_id}' not found."
    del _bookings[booking_id.upper()]
    return f"Booking {booking_id} for {booking['vehicle_name']} on {booking['date']} has been cancelled. Refund of ${booking['total']} will be processed."


ALL_TOOLS = [
    get_vehicle_list,
    get_vehicle_price,
    check_availability,
    get_vehicle_details,
    book_vehicle,
    cancel_booking,
]
