from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests


@dataclass
class SpaceXClient:
    base_url: str = os.getenv("SPACEX_API_BASE_URL", "https://api.spacexdata.com/v5")
    timeout_seconds: int = 20

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method=method, url=url, timeout=self.timeout_seconds, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else "<no response body>"
            raise RuntimeError(f"SpaceX API HTTP error for {path}: {body}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"SpaceX API request failed for {path}: {exc}") from exc

    def latest_launch(self) -> dict[str, Any]:
        return self._request("GET", "/launches/latest")

    def next_launch(self) -> dict[str, Any]:
        return self._request("GET", "/launches/next")

    def launch_by_id(self, launch_id: str) -> dict[str, Any]:
        return self._request("GET", f"/launches/{launch_id}")

    def rocket_by_id(self, rocket_id: str) -> dict[str, Any]:
        return self._request("GET", f"/rockets/{rocket_id}")

    def launchpad_by_id(self, launchpad_id: str) -> dict[str, Any]:
        return self._request("GET", f"/launchpads/{launchpad_id}")

    def launches_query(self, query: dict[str, Any], limit: int = 25) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "options": {
                "pagination": False,
                "limit": limit,
                "sort": {"date_utc": "desc"},
            },
        }
        data = self._request("POST", "/launches/query", json=payload)
        docs = data.get("docs", [])
        if not isinstance(docs, list):
            raise RuntimeError("Unexpected launches query response format from SpaceX API")
        return docs

    def rockets_query(self, query: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "options": {
                "pagination": False,
                "limit": limit,
            },
        }
        data = self._request("POST", "/rockets/query", json=payload)
        docs = data.get("docs", [])
        if not isinstance(docs, list):
            raise RuntimeError("Unexpected rockets query response format from SpaceX API")
        return docs

    def launchpads_query(self, query: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "options": {
                "pagination": False,
                "limit": limit,
            },
        }
        data = self._request("POST", "/launchpads/query", json=payload)
        docs = data.get("docs", [])
        if not isinstance(docs, list):
            raise RuntimeError("Unexpected launchpads query response format from SpaceX API")
        return docs
