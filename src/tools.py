from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool
import requests

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


def _latest_spacex_launch_from_ll2() -> dict[str, Any]:
    base_url = "https://ll.thespacedevs.com/2.2.0"
    candidate_paths = [
        "/launch/?limit=1&ordering=-net&lsp__name=SpaceX",
        "/launch/?limit=5&ordering=-net&search=SpaceX",
        "/launch/previous/?limit=30",
        "/launch/?limit=30&ordering=-net",
    ]

    last_error: str | None = None
    now_utc = datetime.now(timezone.utc)

    for path in candidate_paths:
        try:
            response = requests.get(f"{base_url}{path}", timeout=20)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                results = payload.get("results", [])
            elif isinstance(payload, dict):
                results = [payload]
            else:
                results = []

            if not results:
                continue

            selected: dict[str, Any] | None = None
            for item in results:
                provider = (item.get("launch_service_provider") or {}).get("name", "")
                net = item.get("net")
                launch_time: datetime | None = None
                if isinstance(net, str):
                    try:
                        launch_time = datetime.fromisoformat(net.replace("Z", "+00:00"))
                    except ValueError:
                        launch_time = None

                if (
                    isinstance(provider, str)
                    and "spacex" in provider.lower()
                    and launch_time is not None
                    and launch_time <= now_utc
                ):
                    selected = item
                    break
            if not selected:
                continue
            if selected is None:
                continue
            selected_item: dict[str, Any] = selected

            return {
                "source": "launch_library_2",
                "id": selected_item.get("id"),
                "name": selected_item.get("name"),
                "date_utc": selected_item.get("net"),
                "status": (selected_item.get("status") or {}).get("name"),
                "provider": (selected_item.get("launch_service_provider") or {}).get("name"),
                "pad": (selected_item.get("pad") or {}).get("name"),
                "location": ((selected_item.get("pad") or {}).get("location") or {}).get("name"),
                "url": selected_item.get("url"),
            }
        except requests.RequestException as exc:
            last_error = str(exc)

    return {
        "source": "launch_library_2",
        "error": "Unable to fetch a recent SpaceX launch from Launch Library 2",
        "details": last_error,
    }


def _next_spacex_launch_from_ll2() -> dict[str, Any]:
    base_url = "https://ll.thespacedevs.com/2.2.0"
    candidate_paths = [
        "/launch/upcoming/?limit=30",
        "/launch/?limit=30&ordering=net",
        "/launch/?limit=30&search=SpaceX&ordering=net",
    ]

    now_utc = datetime.now(timezone.utc)
    last_error: str | None = None

    for path in candidate_paths:
        try:
            response = requests.get(f"{base_url}{path}", timeout=20)
            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                results = payload.get("results", [])
            elif isinstance(payload, dict):
                results = [payload]
            else:
                results = []

            if not results:
                continue

            selected: dict[str, Any] | None = None
            for item in results:
                provider = (item.get("launch_service_provider") or {}).get("name", "")
                net = item.get("net")
                launch_time: datetime | None = None
                if isinstance(net, str):
                    try:
                        launch_time = datetime.fromisoformat(net.replace("Z", "+00:00"))
                    except ValueError:
                        launch_time = None

                if (
                    isinstance(provider, str)
                    and "spacex" in provider.lower()
                    and launch_time is not None
                    and launch_time >= now_utc
                ):
                    selected = item
                    break

            if selected is None:
                continue

            selected_item: dict[str, Any] = selected
            return {
                "source": "launch_library_2",
                "id": selected_item.get("id"),
                "name": selected_item.get("name"),
                "date_utc": selected_item.get("net"),
                "status": (selected_item.get("status") or {}).get("name"),
                "provider": (selected_item.get("launch_service_provider") or {}).get("name"),
                "pad": (selected_item.get("pad") or {}).get("name"),
                "location": ((selected_item.get("pad") or {}).get("location") or {}).get("name"),
                "url": selected_item.get("url"),
            }
        except requests.RequestException as exc:
            last_error = str(exc)

    return {
        "source": "launch_library_2",
        "error": "Unable to fetch upcoming SpaceX launch from Launch Library 2",
        "details": last_error,
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
def get_latest_launch_external() -> str:
    """Get latest SpaceX launch from external cross-check source (Launch Library 2)."""
    return _as_json(_latest_spacex_launch_from_ll2())


@tool
def get_next_launch_external() -> str:
    """Get next upcoming SpaceX launch from external cross-check source (Launch Library 2)."""
    return _as_json(_next_spacex_launch_from_ll2())


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
    get_latest_launch_external,
    get_next_launch_external,
    search_launches_by_name,
    get_launches_in_year,
    get_successful_launches_by_rocket,
    get_rocket_by_id,
    get_launchpad_by_id,
    get_recent_launches_from_location,
]
