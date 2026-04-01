from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool

from .spacex_client import SpaceXClient


client = SpaceXClient()


def _as_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def _launch_summary(launch: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": launch.get("id"),
        "name": launch.get("name"),
        "date_utc": launch.get("date_utc"),
        "success": launch.get("success"),
        "upcoming": launch.get("upcoming"),
        "rocket_id": launch.get("rocket"),
        "launchpad_id": launch.get("launchpad"),
        "details": launch.get("details"),
    }


@tool
def get_latest_launch() -> str:
    """Get the latest SpaceX launch with IDs for related entities."""
    return _as_json(_launch_summary(client.latest_launch()))


@tool
def get_next_launch() -> str:
    """Get the next scheduled SpaceX launch with IDs for related entities."""
    return _as_json(_launch_summary(client.next_launch()))


@tool
def search_launches_by_name(name_query: str, limit: int = 10) -> str:
    """Search launches by partial mission name. Useful when the user names a mission."""
    launches = client.launches_query({"name": {"$regex": name_query, "$options": "i"}}, limit=limit)
    return _as_json([_launch_summary(launch) for launch in launches])


@tool
def get_launches_in_year(year: int, limit: int = 200) -> str:
    """Get launches in a calendar year. Use when user asks for counts or year-based summaries."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    launches = client.launches_query(
        {
            "date_utc": {
                "$gte": start,
                "$lt": end,
            }
        },
        limit=limit,
    )
    return _as_json([_launch_summary(launch) for launch in launches])


@tool
def get_successful_launches_by_rocket(rocket_name: str, limit: int = 50) -> str:
    """Find successful launches for a rocket name like Falcon 9 or Falcon Heavy."""
    rockets = client.rockets_query({"name": {"$regex": rocket_name, "$options": "i"}}, limit=5)
    if not rockets:
        return _as_json({"error": f"No rocket found for query: {rocket_name}"})

    rocket_id = rockets[0].get("id")
    launches = client.launches_query({"rocket": rocket_id, "success": True}, limit=limit)
    return _as_json(
        {
            "rocket": {"id": rocket_id, "name": rockets[0].get("name")},
            "launches": [_launch_summary(launch) for launch in launches],
        }
    )


@tool
def get_rocket_by_id(rocket_id: str) -> str:
    """Resolve a rocket ID to rocket details. Use after a launch tool returns rocket_id."""
    rocket = client.rocket_by_id(rocket_id)
    result = {
        "id": rocket.get("id"),
        "name": rocket.get("name"),
        "type": rocket.get("type"),
        "active": rocket.get("active"),
        "description": rocket.get("description"),
        "success_rate_pct": rocket.get("success_rate_pct"),
        "first_flight": rocket.get("first_flight"),
    }
    return _as_json(result)


@tool
def get_launchpad_by_id(launchpad_id: str) -> str:
    """Resolve a launchpad ID to location details. Use after a launch tool returns launchpad_id."""
    launchpad = client.launchpad_by_id(launchpad_id)
    result = {
        "id": launchpad.get("id"),
        "name": launchpad.get("name"),
        "full_name": launchpad.get("full_name"),
        "locality": launchpad.get("locality"),
        "region": launchpad.get("region"),
        "timezone": launchpad.get("timezone"),
        "status": launchpad.get("status"),
    }
    return _as_json(result)


@tool
def get_recent_launches_from_location(location_query: str, limit: int = 10) -> str:
    """Get recent launches from a location such as Vandenberg."""
    launchpads = client.launchpads_query(
        {
            "$or": [
                {"name": {"$regex": location_query, "$options": "i"}},
                {"full_name": {"$regex": location_query, "$options": "i"}},
                {"locality": {"$regex": location_query, "$options": "i"}},
                {"region": {"$regex": location_query, "$options": "i"}},
            ]
        },
        limit=5,
    )
    if not launchpads:
        return _as_json({"error": f"No launchpads found for query: {location_query}"})

    launches: list[dict[str, Any]] = []
    for pad in launchpads:
        pad_launches = client.launches_query({"launchpad": pad.get("id")}, limit=limit)
        launches.extend(pad_launches)

    launches = sorted(launches, key=lambda item: item.get("date_utc") or "", reverse=True)
    return _as_json(
        {
            "matched_launchpads": [
                {
                    "id": pad.get("id"),
                    "name": pad.get("name"),
                    "full_name": pad.get("full_name"),
                }
                for pad in launchpads
            ],
            "launches": [_launch_summary(launch) for launch in launches[:limit]],
        }
    )


SPACEX_TOOLS = [
    get_latest_launch,
    get_next_launch,
    search_launches_by_name,
    get_launches_in_year,
    get_successful_launches_by_rocket,
    get_rocket_by_id,
    get_launchpad_by_id,
    get_recent_launches_from_location,
]
