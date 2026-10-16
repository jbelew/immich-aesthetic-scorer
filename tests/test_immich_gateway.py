"""Unit tests for the connection-pooled Immich HTTP gateway adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from requests.adapters import HTTPAdapter

from immich_aesthetic_scorer.gateway import (
    ImmichGateway,
    ImmichGatewayError,
    ServerVersion,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        content: bytes = b"",
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.text = text

    def json(self) -> Any:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class TestConnectionAndSessionPooling:
    def test_configures_persistent_session_with_connection_pooling_and_api_key(self) -> None:
        gateway = ImmichGateway("http://immich.local:2283/", "secret-key", pool_maxsize=25)

        assert gateway.immich_url == "http://immich.local:2283"
        assert isinstance(gateway.session, requests.Session)
        assert gateway.session.headers.get("x-api-key") == "secret-key"

        http_adapter = gateway.session.get_adapter("http://immich.local:2283")
        https_adapter = gateway.session.get_adapter("https://immich.local:2283")
        assert isinstance(http_adapter, HTTPAdapter)
        assert isinstance(https_adapter, HTTPAdapter)
        assert http_adapter._pool_maxsize == 25
        assert https_adapter._pool_maxsize == 25

    def test_verify_connection_returns_server_version_using_injected_session(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.get.return_value = FakeResponse(
            status_code=200,
            json_data={"major": 1, "minor": 118, "patch": 2},
        )

        gateway = ImmichGateway("http://immich.local:2283/", "secret-key", session=session)
        version = gateway.verify_connection()

        assert version == ServerVersion(major=1, minor=118, patch=2)
        session.get.assert_called_once_with(
            "http://immich.local:2283/api/server/version",
            timeout=10,
        )

    def test_verify_connection_translates_http_and_network_errors(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.get.return_value = FakeResponse(status_code=401, text="Unauthorized")

        gateway = ImmichGateway("http://immich.local:2283", "bad-key", session=session)
        with pytest.raises(ImmichGatewayError) as exc_info:
            gateway.verify_connection()

        assert exc_info.value.status_code == 401
        assert "/api/server/version" in (exc_info.value.endpoint or "")

        session.get.side_effect = requests.ConnectionError("Connection refused")
        with pytest.raises(ImmichGatewayError) as net_exc:
            gateway.verify_connection()

        assert "Connection refused" in str(net_exc.value)


class TestMetadataSearchPagination:
    def test_paginates_250_item_pages_and_filters_for_image_assets(self) -> None:
        # Page 1 has 250 items (249 images + 1 video -> full page triggers page 2)
        page1_items = [
            {"id": f"img-{i}", "type": "IMAGE"} if i < 249 else {"id": "vid-1", "type": "VIDEO"}
            for i in range(250)
        ]
        # Page 2 has 2 items (< 250 -> terminal page)
        page2_items = [
            {"id": "img-250", "type": "IMAGE"},
            {"id": "vid-2", "type": "VIDEO"},
        ]

        session = MagicMock()
        session.headers = {}
        session.post.side_effect = [
            FakeResponse(status_code=200, json_data={"assets": {"items": page1_items}}),
            FakeResponse(status_code=200, json_data={"assets": {"items": page2_items}}),
        ]

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        assets = gateway.fetch_image_assets(person_id="person-123")

        # 249 from page 1 + 1 from page 2 = 250 total IMAGE assets
        assert len(assets) == 250
        assert assets[0]["id"] == "img-0"
        assert assets[-1]["id"] == "img-250"
        assert all(a["type"] == "IMAGE" for a in assets)

        assert session.post.call_count == 2
        first_call = session.post.call_args_list[0]
        second_call = session.post.call_args_list[1]
        assert first_call.args[0] == "http://immich.local:2283/api/search/metadata"
        assert first_call.kwargs["json"] == {"personIds": ["person-123"], "page": 1, "size": 250}
        assert second_call.kwargs["json"] == {"personIds": ["person-123"], "page": 2, "size": 250}

    def test_stops_when_next_page_returns_empty_items_for_album(self) -> None:
        # Exactly 250 items on page 1, followed by empty page 2
        page1_items = [{"id": f"img-{i}", "type": "IMAGE"} for i in range(250)]

        session = MagicMock()
        session.headers = {}
        session.post.side_effect = [
            FakeResponse(status_code=200, json_data={"assets": {"items": page1_items}}),
            FakeResponse(status_code=200, json_data={"assets": {"items": []}}),
        ]

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        assets = gateway.fetch_image_assets(album_id="album-456")

        assert len(assets) == 250
        assert session.post.call_count == 2
        assert session.post.call_args_list[0].kwargs["json"] == {
            "albumIds": ["album-456"],
            "page": 1,
            "size": 250,
        }

    def test_rejects_missing_target_and_translates_http_errors(self) -> None:
        session = MagicMock()
        session.headers = {}
        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)

        with pytest.raises(ValueError, match="person_id or album_id"):
            gateway.fetch_image_assets()

        session.post.return_value = FakeResponse(status_code=500, text="Internal Server Error")
        with pytest.raises(ImmichGatewayError) as exc_info:
            gateway.fetch_image_assets(person_id="person-123")

        assert exc_info.value.status_code == 500


class TestPreviewThumbnailDownload:
    def test_downloads_raw_preview_thumbnail_bytes_without_resizing(self) -> None:
        raw_preview_bytes = b"\xff\xd8\xff\xe0unresized-preview-image-payload"
        session = MagicMock()
        session.headers = {}
        session.get.return_value = FakeResponse(status_code=200, content=raw_preview_bytes)

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        result = gateway.download_thumbnail("asset-uuid-789")

        assert result == raw_preview_bytes
        session.get.assert_called_once_with(
            "http://immich.local:2283/api/assets/asset-uuid-789/thumbnail?size=preview",
            timeout=20,
        )

    def test_download_thumbnail_translates_http_errors(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.get.return_value = FakeResponse(status_code=404, text="Asset not found")

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        with pytest.raises(ImmichGatewayError) as exc_info:
            gateway.download_thumbnail("missing-asset")

        assert exc_info.value.status_code == 404
        assert "missing-asset" in (exc_info.value.endpoint or "")


class TestEnsureAlbumWithAssets:
    def test_resolves_existing_album_deduplicates_ids_and_slices_into_100_item_batches(
        self,
    ) -> None:
        # 205 unique IDs with duplicates sprinkled in -> 3 batches (100, 100, 5)
        raw_ids = [f"asset-{i}" for i in range(205)]
        raw_ids.insert(10, "asset-0")
        raw_ids.insert(150, "asset-120")

        session = MagicMock()
        session.headers = {}
        session.get.return_value = FakeResponse(
            status_code=200,
            json_data=[
                {"id": "other-album", "albumName": "Vacation"},
                {"id": "existing-album-id", "albumName": "Best of Madeline"},
            ],
        )
        session.put.return_value = FakeResponse(status_code=200, json_data=[])

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        album_id = gateway.ensure_album_with_assets("Best of Madeline", raw_ids)

        assert album_id == "existing-album-id"
        session.post.assert_not_called()
        assert session.put.call_count == 3

        batch_1 = session.put.call_args_list[0].kwargs["json"]["ids"]
        batch_2 = session.put.call_args_list[1].kwargs["json"]["ids"]
        batch_3 = session.put.call_args_list[2].kwargs["json"]["ids"]

        assert len(batch_1) == 100
        assert len(batch_2) == 100
        assert len(batch_3) == 5
        assert batch_1 + batch_2 + batch_3 == [f"asset-{i}" for i in range(205)]
        assert (
            session.put.call_args_list[0].args[0]
            == "http://immich.local:2283/api/albums/existing-album-id/assets"
        )

    def test_creates_new_album_when_missing_and_adds_assets(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.get.return_value = FakeResponse(
            status_code=200,
            json_data=[{"id": "other-album", "albumName": "Vacation"}],
        )
        session.post.return_value = FakeResponse(
            status_code=201,
            json_data={"id": "new-album-id", "albumName": "Best of 2026"},
        )
        session.put.return_value = FakeResponse(status_code=200, json_data=[])

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        album_id = gateway.ensure_album_with_assets("Best of 2026", ["a1", "a2", "a1"])

        assert album_id == "new-album-id"
        session.post.assert_called_once_with(
            "http://immich.local:2283/api/albums",
            json={"albumName": "Best of 2026"},
            timeout=20,
        )
        session.put.assert_called_once_with(
            "http://immich.local:2283/api/albums/new-album-id/assets",
            json={"ids": ["a1", "a2"]},
            timeout=30,
        )


class TestStarRatingAndMetadataLookups:
    def test_updates_native_star_rating_and_validates_range(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.put.return_value = FakeResponse(status_code=204)

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        gateway.update_asset_rating("asset-42", 5)

        session.put.assert_called_once_with(
            "http://immich.local:2283/api/assets",
            json={"ids": ["asset-42"], "rating": 5},
            timeout=10,
        )

        with pytest.raises(ValueError, match="between 1 and 5"):
            gateway.update_asset_rating("asset-42", 0)

        with pytest.raises(ValueError, match="between 1 and 5"):
            gateway.update_asset_rating("asset-42", 6)

    def test_update_asset_rating_translates_http_errors(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.put.return_value = FakeResponse(status_code=400, text="Bad Request")

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        with pytest.raises(ImmichGatewayError) as exc_info:
            gateway.update_asset_rating("asset-42", 4)

        assert exc_info.value.status_code == 400

    def test_update_assets_rating_batches_in_100_item_chunks_and_deduplicates(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.put.return_value = FakeResponse(status_code=204)

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)
        # 150 unique IDs + 1 duplicate across boundary -> 2 PUT batches (100 + 50)
        ids = [f"id-{i}" for i in range(150)] + ["id-0"]
        gateway.update_assets_rating(ids, rating=4)

        assert session.put.call_count == 2
        first_call = session.put.call_args_list[0]
        second_call = session.put.call_args_list[1]
        assert first_call.kwargs["json"] == {"ids": [f"id-{i}" for i in range(100)], "rating": 4}
        assert second_call.kwargs["json"] == {
            "ids": [f"id-{i}" for i in range(100, 150)],
            "rating": 4,
        }

    def test_retrieves_person_and_album_metadata(self) -> None:
        session = MagicMock()
        session.headers = {}
        session.get.side_effect = [
            FakeResponse(status_code=200, json_data={"id": "p1", "name": "Madeline"}),
            FakeResponse(status_code=200, json_data=[{"id": "p1", "name": "Madeline"}]),
            FakeResponse(
                status_code=200,
                json_data={"people": [{"id": "p1", "name": "Madeline"}]},
            ),
            FakeResponse(status_code=200, json_data={"id": "alb1", "albumName": "Champs"}),
            FakeResponse(status_code=200, json_data=[{"id": "alb1", "albumName": "Champs"}]),
        ]

        gateway = ImmichGateway("http://immich.local:2283", "secret-key", session=session)

        assert gateway.get_person_details("p1") == {"id": "p1", "name": "Madeline"}
        assert gateway.search_people("Mad") == [{"id": "p1", "name": "Madeline"}]
        assert gateway.list_people() == [{"id": "p1", "name": "Madeline"}]
        assert gateway.get_album_details("alb1") == {"id": "alb1", "albumName": "Champs"}
        assert gateway.list_albums() == [{"id": "alb1", "albumName": "Champs"}]
