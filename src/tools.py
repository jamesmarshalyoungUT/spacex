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


def _parse_spacex_website_launch_datetime(launch_date: Any, launch_time: Any) -> datetime | None:
    if not (isinstance(launch_date, str) and launch_date.strip()):
        return None

    date_formats = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"]
    parsed_date: datetime | None = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(launch_date.strip(), fmt)
            break
        except ValueError:
            continue

    if parsed_date is None:
        return None

    if not isinstance(launch_time, str) or not launch_time.strip():
        return parsed_date.replace(tzinfo=timezone.utc)

    normalized_time = launch_time.strip().upper().replace(" UTC", "")
    if normalized_time in {"TBD", "TBA"}:
        return parsed_date.replace(tzinfo=timezone.utc)

    time_formats = ["%I:%M %p", "%I %p", "%H:%M", "%H"]
    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(normalized_time, fmt)
            return parsed_date.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue

    return parsed_date.replace(tzinfo=timezone.utc)


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


def _latest_spacex_launch_from_spacex_website() -> dict[str, Any]:
    tiles_url = "https://content.spacex.com/api/spacex-website/launches-page-tiles"
    now_utc = datetime.now(timezone.utc)

    try:
        response = requests.get(tiles_url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        tiles = payload if isinstance(payload, list) else []
        if not tiles:
            return {
                "source": "spacex_website",
                "error": "SpaceX website launches endpoint returned no results",
                "details": None,
            }

        latest_tile: dict[str, Any] | None = None
        latest_dt: datetime | None = None

        for item in tiles:
            if not isinstance(item, dict):
                continue

            mission_status = str(item.get("missionStatus") or "").lower()
            if mission_status not in {"final", "in-progress"}:
                continue

            dt = _parse_spacex_website_launch_datetime(item.get("launchDate"), item.get("launchTime"))
            if dt is None or dt > now_utc:
                continue

            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
                latest_tile = item

        if latest_tile is None:
            return {
                "source": "spacex_website",
                "error": "No recent SpaceX launch found on SpaceX website source",
                "details": None,
            }

        pad = latest_tile.get("launchSite")
        mission_id = latest_tile.get("correlationId") or latest_tile.get("link")
        mission_link = latest_tile.get("link")
        return {
            "source": "spacex_website",
            "id": mission_id,
            "name": latest_tile.get("title") or latest_tile.get("name"),
            "date_utc": latest_dt.isoformat().replace("+00:00", "Z") if latest_dt else None,
            "status": latest_tile.get("missionStatus"),
            "provider": "SpaceX",
            "pad": pad,
            "location": None,
            "url": f"https://www.spacex.com/launches/{mission_link}" if mission_link else None,
        }
    except requests.RequestException as exc:
        return {
            "source": "spacex_website",
            "error": "Unable to fetch recent SpaceX launch from SpaceX website source",
            "details": str(exc),
        }


def _latest_spacex_launch_from_rocketlaunch_live() -> dict[str, Any]:
    url = "https://fdo.rocketlaunch.live/json/launches/next/50"
    now_utc = datetime.now(timezone.utc)

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("result", []) if isinstance(payload, dict) else []
        if not isinstance(results, list) or not results:
            return {
                "source": "rocketlaunch_live",
                "error": "RocketLaunch.Live returned no results",
                "details": None,
            }

        latest_item: dict[str, Any] | None = None
        latest_dt: datetime | None = None

        for item in results:
            provider = (item.get("provider") or {}).get("name", "")
            if not (isinstance(provider, str) and "spacex" in provider.lower()):
                continue

            raw_date = item.get("t0") or item.get("win_open")
            if not isinstance(raw_date, str):
                continue
            try:
                dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except ValueError:
                continue

            # For latest-launch, we only accept launches that are not in the future.
            if dt > now_utc:
                continue

            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
                latest_item = item

        if not latest_item:
            return {
                "source": "rocketlaunch_live",
                "error": "No recent SpaceX launch found in RocketLaunch.Live results",
                "details": None,
            }

        raw_result = latest_item.get("result")
        status_map = {
            -1: "Go for Launch",
            1: "Launch Successful",
            0: "To Be Determined",
            3: "Launch Failure",
            4: "Launch Partial Failure",
        }
        if isinstance(raw_result, int):
            status_label = status_map.get(raw_result, str(raw_result))
        else:
            status_label = str(raw_result)

        pad = latest_item.get("pad") or {}
        location = (pad.get("location") or {}) if isinstance(pad, dict) else {}

        return {
            "source": "rocketlaunch_live",
            "id": latest_item.get("id"),
            "name": latest_item.get("name"),
            "date_utc": latest_item.get("t0") or latest_item.get("win_open"),
            "status": status_label,
            "provider": (latest_item.get("provider") or {}).get("name"),
            "pad": pad.get("name") if isinstance(pad, dict) else None,
            "location": location.get("name") if isinstance(location, dict) else None,
            "url": f"https://rocketlaunch.live/launch/{latest_item.get('slug')}" if latest_item.get("slug") else None,
        }
    except requests.RequestException as exc:
        return {
            "source": "rocketlaunch_live",
            "error": "Unable to fetch recent SpaceX launch from RocketLaunch.Live",
            "details": str(exc),
        }


def _latest_spacex_launch_external() -> dict[str, Any]:
    ll2 = _latest_spacex_launch_from_ll2()
    if not ll2.get("error"):
        ll2["fallback_chain"] = ["launch_library_2"]
        return ll2

    rll = _latest_spacex_launch_from_rocketlaunch_live()
    if not rll.get("error"):
        rll["fallback_chain"] = ["launch_library_2", "rocketlaunch_live"]
        rll["previous_error"] = ll2.get("error")
        return rll

    return {
        "source": "multi_source_fallback",
        "error": "All latest-launch secondary sources failed",
        "details": {
            "launch_library_2": {"error": ll2.get("error"), "details": ll2.get("details")},
            "rocketlaunch_live": {"error": rll.get("error"), "details": rll.get("details")},
        },
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


def _next_spacex_launch_from_spacex_website() -> dict[str, Any]:
    upcoming_tiles_url = "https://content.spacex.com/api/spacex-website/launches-page-tiles/upcoming"
    timings_url = "https://sxcontent9668.azureedge.us/cms-assets/future_missions.json"
    now_utc = datetime.now(timezone.utc)

    try:
        tiles_resp = requests.get(upcoming_tiles_url, timeout=20)
        tiles_resp.raise_for_status()
        payload = tiles_resp.json()
        tiles = payload if isinstance(payload, list) else []
        if not tiles:
            return {
                "source": "spacex_website",
                "error": "SpaceX website upcoming endpoint returned no results",
                "details": None,
            }

        timings_resp = requests.get(timings_url, timeout=20)
        timings_resp.raise_for_status()
        timings_payload = timings_resp.json()
        timings = timings_payload if isinstance(timings_payload, dict) else {}

        best_tile: dict[str, Any] | None = None
        best_dt: datetime | None = None

        for item in tiles:
            if not isinstance(item, dict):
                continue
            correlation_id = item.get("correlationId")
            timing = timings.get(correlation_id) if isinstance(correlation_id, str) else None
            if not isinstance(timing, dict):
                continue

            t0 = timing.get("TZeroLaunchDate")
            primary = timing.get("PrimaryLaunchDate")
            seconds = t0.get("Seconds") if isinstance(t0, dict) else None
            if not isinstance(seconds, int):
                seconds = primary.get("Seconds") if isinstance(primary, dict) else None
            if not isinstance(seconds, int):
                continue

            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
            if dt < now_utc:
                continue

            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_tile = item

        if best_tile is None:
            return {
                "source": "spacex_website",
                "error": "No upcoming SpaceX launch found on SpaceX website source",
                "details": None,
            }

        mission_id = best_tile.get("correlationId") or best_tile.get("link")
        mission_link = best_tile.get("link")
        return {
            "source": "spacex_website",
            "id": mission_id,
            "name": best_tile.get("title") or best_tile.get("name"),
            "date_utc": best_dt.isoformat().replace("+00:00", "Z") if best_dt else best_tile.get("dateUtc"),
            "status": best_tile.get("missionStatus"),
            "provider": "SpaceX",
            "pad": best_tile.get("launchSite"),
            "location": None,
            "url": f"https://www.spacex.com/launches/{mission_link}" if mission_link else None,
        }
    except requests.RequestException as exc:
        return {
            "source": "spacex_website",
            "error": "Unable to fetch upcoming SpaceX launch from SpaceX website source",
            "details": str(exc),
        }


def _next_spacex_launch_from_rocketlaunch_live() -> dict[str, Any]:
    url = "https://fdo.rocketlaunch.live/json/launches/next/20"
    now_utc = datetime.now(timezone.utc)

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("result", []) if isinstance(payload, dict) else []
        if not isinstance(results, list) or not results:
            return {
                "source": "rocketlaunch_live",
                "error": "RocketLaunch.Live returned no results",
                "details": None,
            }

        for item in results:
            provider = (item.get("provider") or {}).get("name", "")
            if not (isinstance(provider, str) and "spacex" in provider.lower()):
                continue

            raw_date = item.get("t0") or item.get("win_open")
            launch_time: datetime | None = None
            if isinstance(raw_date, str):
                try:
                    launch_time = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                except ValueError:
                    launch_time = None

            if launch_time is None or launch_time < now_utc:
                continue

            pad = item.get("pad") or {}
            location = (pad.get("location") or {}) if isinstance(pad, dict) else {}
            raw_result = item.get("result")
            status_map = {
                -1: "Go for Launch",
                1: "Launch Successful",
                0: "To Be Determined",
                3: "Launch Failure",
                4: "Launch Partial Failure",
            }
            status_label = status_map.get(raw_result, str(raw_result))

            return {
                "source": "rocketlaunch_live",
                "id": item.get("id"),
                "name": item.get("name"),
                "date_utc": raw_date,
                "status": status_label,
                "provider": provider,
                "pad": pad.get("name") if isinstance(pad, dict) else None,
                "location": location.get("name") if isinstance(location, dict) else None,
                "url": f"https://rocketlaunch.live/launch/{item.get('slug')}" if item.get("slug") else None,
            }

        return {
            "source": "rocketlaunch_live",
            "error": "No upcoming SpaceX launch found in RocketLaunch.Live results",
            "details": None,
        }
    except requests.RequestException as exc:
        return {
            "source": "rocketlaunch_live",
            "error": "Unable to fetch upcoming SpaceX launch from RocketLaunch.Live",
            "details": str(exc),
        }


def _next_spacex_launch_external() -> dict[str, Any]:
    ll2 = _next_spacex_launch_from_ll2()
    if not ll2.get("error"):
        ll2["fallback_chain"] = ["launch_library_2"]
        return ll2

    rll = _next_spacex_launch_from_rocketlaunch_live()
    if not rll.get("error"):
        rll["fallback_chain"] = ["launch_library_2", "rocketlaunch_live"]
        rll["previous_error"] = ll2.get("error")
        return rll

    return {
        "source": "multi_source_fallback",
        "error": "All upcoming-launch secondary sources failed",
        "details": {
            "launch_library_2": {"error": ll2.get("error"), "details": ll2.get("details")},
            "rocketlaunch_live": {"error": rll.get("error"), "details": rll.get("details")},
        },
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
    """Get latest SpaceX launch from external cross-check sources (LL2, then RocketLaunch.Live)."""
    return _as_json(_latest_spacex_launch_external())


@tool
def get_next_launch_external() -> str:
    """Get next upcoming SpaceX launch from external cross-check sources (LL2, then RocketLaunch.Live)."""
    return _as_json(_next_spacex_launch_external())


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
