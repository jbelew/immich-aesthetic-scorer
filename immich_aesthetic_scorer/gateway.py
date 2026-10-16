"""Connection-pooled HTTP gateway adapter for the Immich REST API.

Encapsulates session connection pooling, server version checks, metadata search
pagination, preview thumbnail retrieval, album resolution and 100-item batching,
and native star rating updates behind a single ports-and-adapters boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import requests
from requests.adapters import HTTPAdapter


@dataclass(frozen=True)
class ServerVersion:
    """Semantic version tuple reported by an Immich server.

    Attributes:
        major: Major version integer (defaults to 1 if omitted by server).
        minor: Minor version integer (defaults to 0 if omitted by server).
        patch: Patch version integer (defaults to 0 if omitted by server).
    """

    major: int
    minor: int
    patch: int = 0


class ImmichGatewayError(RuntimeError):
    """Domain error raised when an Immich HTTP request or response fails.

    Attributes:
        status_code: HTTP status code if a response was received, otherwise None.
        endpoint: Immich API path or URL associated with the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint


class ImmichGateway:
    """Persistent, connection-pooled HTTP gateway for Immich REST operations.

    Maintains a reusable `requests.Session` mounted with `HTTPAdapter` connection
    pools for both `http://` and `https://` to avoid repeated TCP/TLS handshakes
    across multi-threaded scoring and rating synchronization.
    """

    def __init__(
        self,
        immich_url: str,
        api_key: str,
        session: Any | None = None,
        pool_maxsize: int = 20,
    ) -> None:
        """Initializes the Immich gateway with a pooled HTTP session.

        Args:
            immich_url: Base URL of the Immich server (trailing slashes are stripped).
            api_key: Immich API key sent via the `x-api-key` request header.
            session: Optional pre-configured or mock session for testing.
            pool_maxsize: Maximum number of pooled keep-alive connections per host
                (default: 20, sized to exceed typical worker thread concurrency).
        """
        self.immich_url = immich_url.rstrip("/")
        self.api_key = api_key
        if session is not None:
            self.session = session
            if hasattr(self.session, "headers") and isinstance(self.session.headers, dict):
                self.session.headers.setdefault("x-api-key", api_key)
        else:
            pooled_session = requests.Session()
            pooled_session.headers.update({"x-api-key": api_key})
            # Size connection pools to support parallel worker threads without socket churn
            adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
            pooled_session.mount("http://", adapter)
            pooled_session.mount("https://", adapter)
            self.session = pooled_session

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        timeout: float = 20,
        **kwargs: Any,
    ) -> Any:
        """Executes an HTTP call on the pooled session and translates errors."""
        url = f"{self.immich_url}{endpoint}"
        try:
            method_fn = getattr(self.session, method.lower())
            response = method_fn(url, timeout=timeout, **kwargs)
        except Exception as exc:
            raise ImmichGatewayError(
                f"Immich request to {endpoint} failed: {exc}",
                status_code=None,
                endpoint=endpoint,
            ) from exc

        status_code = getattr(response, "status_code", 200)
        if status_code >= 400:
            body_preview = getattr(response, "text", "")
            raise ImmichGatewayError(
                f"Immich returned HTTP {status_code} for {endpoint}: {body_preview}",
                status_code=status_code,
                endpoint=endpoint,
            )
        return response

    def verify_connection(self) -> ServerVersion:
        """Verifies connectivity and retrieves the Immich server version.

        Returns:
            ServerVersion with major, minor, and patch numbers.

        Raises:
            ImmichGatewayError: If the server is unreachable, unauthorized, or
                returns a malformed payload.
        """
        endpoint = "/api/server/version"
        response = self._request("get", endpoint, timeout=10)
        try:
            data = response.json()
            return ServerVersion(
                major=int(data.get("major", 1)),
                minor=int(data.get("minor", 0)),
                patch=int(data.get("patch", 0)),
            )
        except Exception as exc:
            raise ImmichGatewayError(
                f"Invalid version payload from {endpoint}: {exc}",
                status_code=getattr(response, "status_code", None),
                endpoint=endpoint,
            ) from exc

    def fetch_image_assets(
        self,
        *,
        person_id: str | None = None,
        album_id: str | None = None,
        page_size: int = 250,
    ) -> list[dict[str, Any]]:
        """Paginates through `/api/search/metadata` to retrieve all `IMAGE` assets.

        Args:
            person_id: Optional target person UUID (`personIds` filter).
            album_id: Optional target album UUID (`albumIds` filter).
            page_size: Number of asset items requested per page (default: 250).

        Returns:
            List of Immich asset metadata dictionaries where `type == 'IMAGE'`.

        Raises:
            ValueError: If neither `person_id` nor `album_id` is provided, or if
                `page_size` is not positive.
            ImmichGatewayError: If any paginated HTTP request fails.
        """
        if person_id:
            payload_key = "personIds"
            payload_val = [person_id]
        elif album_id:
            payload_key = "albumIds"
            payload_val = [album_id]
        else:
            raise ValueError("Either person_id or album_id must be provided")

        if page_size <= 0:
            raise ValueError("page_size must be a positive integer")

        endpoint = "/api/search/metadata"
        assets: list[dict[str, Any]] = []
        page = 1

        while True:
            payload: dict[str, Any] = {
                payload_key: payload_val,
                "page": page,
                "size": page_size,
            }
            response = self._request("post", endpoint, json=payload, timeout=30)
            data = response.json()
            items = data.get("assets", {}).get("items", []) if isinstance(data, dict) else []
            if not items:
                break

            images = [
                item for item in items if isinstance(item, dict) and item.get("type") == "IMAGE"
            ]
            assets.extend(images)

            # Check raw un-filtered page count against page_size so pages containing
            # mixed IMAGE and VIDEO items do not terminate pagination prematurely.
            if len(items) < page_size:
                break

            page += 1

        return assets

    def download_thumbnail(self, asset_id: str) -> bytes:
        """Downloads the raw uncropped preview thumbnail for an asset.

        Client-side image downscaling is intentionally excluded from the gateway;
        evaluators requiring specific dimensions handle resizing at their own seam.

        Args:
            asset_id: UUID of the target Immich asset.

        Returns:
            Raw binary image payload returned by `/api/assets/{asset_id}/thumbnail?size=preview`.

        Raises:
            ImmichGatewayError: If the thumbnail request fails or returns a non-200 status.
        """
        endpoint = f"/api/assets/{asset_id}/thumbnail?size=preview"
        response = self._request("get", endpoint, timeout=20)
        return bytes(response.content)

    def ensure_album_with_assets(
        self,
        album_name: str,
        asset_ids: Sequence[str],
        chunk_size: int = 100,
    ) -> str:
        """Resolves an existing album or creates a new one, then adds assets in 100-ID batches.

        Deduplicates `asset_ids` while preserving first-seen order and slices `PUT`
        requests into batches of at most `chunk_size` (default 100) to respect
        Immich server payload size limits.

        Args:
            album_name: Target album display name to find or create.
            asset_ids: Sequence of asset UUID strings to register in the album.
            chunk_size: Maximum number of asset IDs per `PUT` request (default: 100).

        Returns:
            UUID string of the matched or newly created album.

        Raises:
            ValueError: If `album_name` is empty or `chunk_size <= 0`.
            ImmichGatewayError: If album lookup, album creation, or any batch update fails.
        """
        if not album_name or not album_name.strip():
            raise ValueError("album_name must be a non-empty string")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        albums_endpoint = "/api/albums"
        album_id: str | None = None
        for album in self.list_albums():
            if album.get("albumName") == album_name:
                album_id = str(album["id"])
                break

        if album_id is None:
            create_resp = self._request(
                "post",
                albums_endpoint,
                json={"albumName": album_name},
                timeout=20,
            )
            created_data = create_resp.json()
            album_id = str(created_data["id"])

        # Deduplicate across the full input before chunking so an asset ID
        # repeated across 100-item boundaries is not sent in multiple PUT calls.
        seen: set[str] = set()
        unique_ids: list[str] = []
        for aid in asset_ids:
            if aid not in seen:
                seen.add(aid)
                unique_ids.append(aid)

        assets_endpoint = f"/api/albums/{album_id}/assets"
        for i in range(0, len(unique_ids), chunk_size):
            batch = unique_ids[i : i + chunk_size]
            self._request("put", assets_endpoint, json={"ids": batch}, timeout=30)

        return album_id

    def update_asset_rating(self, asset_id: str, rating: int) -> None:
        """Updates the native star rating (`1–5`) on a single Immich asset.

        Args:
            asset_id: UUID of the target Immich asset.
            rating: Star rating integer in the range `1..5`.

        Raises:
            ValueError: If `rating` is outside `1..5` (or is a boolean).
            ImmichGatewayError: If the `PUT /api/assets` request fails.
        """
        if isinstance(rating, bool) or not isinstance(rating, int) or not (1 <= rating <= 5):
            raise ValueError(f"rating must be an integer between 1 and 5, got {rating!r}")

        endpoint = "/api/assets"
        self._request(
            "put",
            endpoint,
            json={"ids": [asset_id], "rating": rating},
            timeout=10,
        )

    @staticmethod
    def _extract_dict_list(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def get_person_details(self, person_id: str) -> dict[str, Any]:
        """Retrieves metadata for a specific person from `/api/people/{person_id}`.

        Args:
            person_id: UUID of the target person.

        Returns:
            Dictionary of person metadata returned by Immich.

        Raises:
            ImmichGatewayError: If the request fails or returns a non-200 status.
        """
        endpoint = f"/api/people/{person_id}"
        response = self._request("get", endpoint, timeout=10)
        data = response.json()
        return dict(data) if isinstance(data, dict) else {}

    def search_people(self, query: str) -> list[dict[str, Any]]:
        """Searches for people by name via `/api/search/person`.

        Args:
            query: Name substring to search for.

        Returns:
            List of matching person dictionaries.

        Raises:
            ImmichGatewayError: If the search request fails.
        """
        endpoint = "/api/search/person"
        response = self._request("get", endpoint, params={"name": query}, timeout=10)
        return self._extract_dict_list(response.json())

    def list_people(self, size: int = 100) -> list[dict[str, Any]]:
        """Lists people from `/api/people`.

        Handles both flat list responses and paginated `{"people": [...]}` payloads.

        Args:
            size: Maximum number of people to fetch (default: 100).

        Returns:
            List of person dictionaries.

        Raises:
            ImmichGatewayError: If the request fails.
        """
        endpoint = "/api/people"
        response = self._request("get", endpoint, params={"size": size}, timeout=10)
        data = response.json()
        if isinstance(data, dict):
            return self._extract_dict_list(data.get("people"))
        return self._extract_dict_list(data)

    def get_album_details(self, album_id: str) -> dict[str, Any]:
        """Retrieves metadata for a specific album from `/api/albums/{album_id}`.

        Args:
            album_id: UUID of the target album.

        Returns:
            Dictionary of album metadata returned by Immich.

        Raises:
            ImmichGatewayError: If the request fails or returns a non-200 status.
        """
        endpoint = f"/api/albums/{album_id}"
        response = self._request("get", endpoint, timeout=10)
        data = response.json()
        return dict(data) if isinstance(data, dict) else {}

    def list_albums(self) -> list[dict[str, Any]]:
        """Lists all albums from `/api/albums`.

        Returns:
            List of album metadata dictionaries.

        Raises:
            ImmichGatewayError: If the request fails.
        """
        endpoint = "/api/albums"
        response = self._request("get", endpoint, timeout=20)
        return self._extract_dict_list(response.json())
