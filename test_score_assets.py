import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Import functions from score_assets
from score_assets import (
    check_immich_connection,
    fetch_all_image_assets,
    get_album_details,
    get_person_details,
    load_cache,
    save_cache,
)


class TestImmichScorer(unittest.TestCase):
    def setUp(self):
        self.immich_url = "http://fake-immich:2283"
        self.api_key = "fake-api-key"

    @patch("score_assets.requests.get")
    def test_check_immich_connection_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"major": 3, "minor": 0, "patch": 3}
        mock_get.return_value = mock_resp

        success, major, minor = check_immich_connection(self.immich_url, self.api_key)
        self.assertTrue(success)
        self.assertEqual(major, 3)
        self.assertEqual(minor, 0)

    @patch("score_assets.requests.get")
    def test_check_immich_connection_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        success, err, _ = check_immich_connection(self.immich_url, self.api_key)
        self.assertFalse(success)
        self.assertIn("Status code 401", err)

    @patch("score_assets.requests.get")
    def test_get_person_details(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "person-uuid", "name": "Madeline"}
        mock_get.return_value = mock_resp

        details = get_person_details(self.immich_url, self.api_key, "person-uuid")
        self.assertIsNotNone(details)
        assert details is not None
        self.assertEqual(details["name"], "Madeline")

    @patch("score_assets.requests.get")
    def test_get_album_details(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "album-uuid", "albumName": "Cheezy Champs"}
        mock_get.return_value = mock_resp

        details = get_album_details(self.immich_url, self.api_key, "album-uuid")
        self.assertIsNotNone(details)
        assert details is not None
        self.assertEqual(details["albumName"], "Cheezy Champs")

    @patch("score_assets.requests.post")
    def test_fetch_all_image_assets_person(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "assets": {
                "items": [
                    {"id": "asset-1", "type": "IMAGE", "originalFileName": "pic1.jpg"},
                    {"id": "asset-2", "type": "VIDEO", "originalFileName": "vid1.mp4"},
                    {"id": "asset-3", "type": "IMAGE", "originalFileName": "pic2.jpg"},
                ]
            }
        }
        mock_post.return_value = mock_resp

        assets = fetch_all_image_assets(self.immich_url, self.api_key, person_id="person-uuid")
        # Video should be filtered out, leaving 2 image assets
        self.assertEqual(len(assets), 2)
        self.assertEqual(assets[0]["id"], "asset-1")
        self.assertEqual(assets[1]["id"], "asset-3")

    def test_cache_load_save(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_cache_path = f.name

        try:
            test_data = {
                "asset-1": {
                    "score": 85,
                    "raw_score_stage1": 7.5,
                    "model_id_stage1": "rsinema/aesthetic-scorer",
                    "updatedAt": "2026-07-23T12:00:00Z",
                }
            }
            save_cache(test_data, temp_cache_path)

            loaded = load_cache(temp_cache_path)
            self.assertEqual(loaded["asset-1"]["score"], 85)
            self.assertEqual(loaded["asset-1"]["model_id_stage1"], "rsinema/aesthetic-scorer")
        finally:
            if os.path.exists(temp_cache_path):
                os.remove(temp_cache_path)

    def test_sigmoid_math(self):
        # z = 0 should yield exactly 50
        z = 0.0
        s_norm = 100.0 / (1.0 + math.exp(-1.5 * z))
        self.assertAlmostEqual(s_norm, 50.0)

        # positive z should yield > 50
        z = 1.0
        s_norm = 100.0 / (1.0 + math.exp(-1.5 * z))
        self.assertTrue(s_norm > 50.0)

        # negative z should yield < 50
        z = -1.0
        s_norm = 100.0 / (1.0 + math.exp(-1.5 * z))
        self.assertTrue(s_norm < 50.0)

    def test_stage2_thumbnail_dimensions_logic(self):
        # Local model logic helper (replicates logic in score_assets.py)
        def get_max_dim_for_stage2_model(stage2_model):
            is_local_s2 = not (
                "gemini" in stage2_model.lower()
                or "openai" in stage2_model.lower()
                or "gpt" in stage2_model.lower()
            )
            return None if is_local_s2 else 512

        # Assert local model uses None (full-res preview)
        self.assertIsNone(get_max_dim_for_stage2_model("musiq-spaq"))
        self.assertIsNone(get_max_dim_for_stage2_model("other-local-model"))

        # Assert remote models use 512
        self.assertEqual(get_max_dim_for_stage2_model("gemini-2.5-flash"), 512)
        self.assertEqual(get_max_dim_for_stage2_model("gemini"), 512)
        self.assertEqual(get_max_dim_for_stage2_model("openai/gpt-4o-mini"), 512)
        self.assertEqual(get_max_dim_for_stage2_model("gpt-4o"), 512)

    @patch("score_assets.requests.get")
    def test_download_thumbnail_scaling_logic(self, mock_get):
        import io

        from PIL import Image

        # Create a mock 1000x1000 JPEG image in memory
        img = Image.new("RGB", (1000, 1000), color="red")
        img_bytes_io = io.BytesIO()
        img.save(img_bytes_io, format="JPEG")
        img_data = img_bytes_io.getvalue()

        # Mock requests.get content
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = img_data
        mock_get.return_value = mock_resp

        # Import download_thumbnail
        from score_assets import download_thumbnail

        # Case 1: max_dim is None (no downscaling)
        res_bytes_none = download_thumbnail(
            "http://fake-immich", "fake-api", "asset-id", max_dim=None
        )
        # Should return exactly the original unscaled image bytes
        self.assertEqual(res_bytes_none, img_data)

        # Case 2: max_dim is 512 (downscaled)
        res_bytes_512 = download_thumbnail(
            "http://fake-immich", "fake-api", "asset-id", max_dim=512
        )
        # Should be scaled to 512x512
        scaled_img = Image.open(io.BytesIO(res_bytes_512))
        self.assertEqual(scaled_img.size, (512, 512))

    def test_score_to_stars(self):
        from score_assets import score_to_stars

        self.assertEqual(score_to_stars(95), 5)
        self.assertEqual(score_to_stars(90), 5)
        self.assertEqual(score_to_stars(89), 4)
        self.assertEqual(score_to_stars(75), 4)
        self.assertEqual(score_to_stars(74), 3)
        self.assertEqual(score_to_stars(50), 3)
        self.assertEqual(score_to_stars(49), 2)
        self.assertEqual(score_to_stars(20), 2)
        self.assertEqual(score_to_stars(19), 1)
        self.assertEqual(score_to_stars(0), 1)

    def test_extract_raw_score_from_reason(self):
        from score_assets import extract_raw_score_from_reason

        self.assertEqual(extract_raw_score_from_reason("Local CLIP score: 7.55/10.0"), 7.55)
        self.assertEqual(
            extract_raw_score_from_reason("Local CLIP score: 8.0/10.0 (z-score: 1.2)"), 8.0
        )
        self.assertIsNone(extract_raw_score_from_reason("Some other reason"))
        self.assertIsNone(extract_raw_score_from_reason(None))

    def test_parse_asset_time(self):
        import datetime

        from score_assets import parse_asset_time

        # Test localDateTime
        self.assertEqual(
            parse_asset_time({"asset": {"localDateTime": "2026-07-24T10:00:00"}}),
            datetime.datetime(2026, 7, 24, 10, 0, 0),
        )
        # Test with milliseconds and Z suffix
        self.assertEqual(
            parse_asset_time({"asset": {"fileCreatedAt": "2026-07-24T10:00:00.123456Z"}}),
            datetime.datetime(2026, 7, 24, 10, 0, 0, 123456),
        )
        # Test missing / invalid fallback
        self.assertEqual(
            parse_asset_time({"asset": {"createdAt": "invalid-date-string"}}), datetime.datetime.min
        )
        self.assertEqual(parse_asset_time({}), datetime.datetime.min)

    def test_deduplicate_bursts(self):
        from score_assets import deduplicate_bursts

        scored_assets = [
            {"id": "a1", "score": 80, "asset": {"localDateTime": "2026-07-24T10:00:00"}},
            {
                "id": "a2",
                "score": 90,
                "asset": {"localDateTime": "2026-07-24T10:00:05"},
            },  # Within 10s of a1, higher score
            {
                "id": "a3",
                "score": 85,
                "asset": {"localDateTime": "2026-07-24T10:00:08"},
            },  # Within 10s of a1, lower score
            {
                "id": "a4",
                "score": 70,
                "asset": {"localDateTime": "2026-07-24T10:00:25"},
            },  # New group, 20s later
        ]

        # Case 1: Deduplication disabled (dedup_window <= 0)
        self.assertEqual(len(deduplicate_bursts(scored_assets, 0)), 4)

        # Case 2: Deduplication with 10s window
        # Group 1: a1 (10:00:00), a2 (10:00:05), a3 (10:00:08) -> a2 has highest score (90)
        # Group 2: a4 (10:00:25) -> kept
        deduped = deduplicate_bursts(scored_assets, 10)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["id"], "a2")
        self.assertEqual(deduped[1]["id"], "a4")


if __name__ == "__main__":
    unittest.main()
