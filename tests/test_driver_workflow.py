"""Unit and integration tests for the CLI driver and workflow orchestration seam."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from score_assets import ScorerConfig, resolve_config  # noqa: E402


class TestConfigResolution:
    def test_resolves_precedence_cli_over_env_over_config_over_defaults(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "immich_url": "http://config-immich:2283/",
                    "immich_api_key": "config-key",
                    "person_id": "config-person",
                    "scorer_type": "local",
                    "limit": 50,
                    "write_ratings": True,
                }
            ),
            encoding="utf-8",
        )

        env = {
            "IMMICH_URL": "http://env-immich:2283",
            "GEMINI_MODEL": "gemini-env-model",
            "DEDUP_WINDOW": "12",
        }

        cfg = resolve_config(
            argv=[
                "--config",
                str(config_file),
                "--immich-url",
                "http://cli-immich:2283/",
                "--album-id",
                "cli-album",
            ],
            env=env,
        )

        assert isinstance(cfg, ScorerConfig)
        # CLI overrides env and config
        assert cfg.immich_url == "http://cli-immich:2283"
        # --album-id on CLI clears person_id from config.json
        assert cfg.album_id == "cli-album"
        assert cfg.person_id is None
        # Env overrides config and defaults
        assert cfg.gemini_model == "gemini-env-model"
        assert cfg.dedup_window == 12
        # Config JSON overrides defaults when not set in CLI or env
        assert cfg.api_key == "config-key"
        assert cfg.scorer_type == "local"
        assert cfg.limit == 50
        assert cfg.write_ratings is True
        # Local scorer defaults delay to 0.0 when not specified
        assert cfg.delay == 0.0
        # Default fallback values
        assert cfg.stage2_top_pct == 15.0
        assert cfg.stage2_weight == 0.5
        assert cfg.two_stage is False
        assert cfg.dry_run is False

    def test_boolean_cli_flags_can_negate_config_json_values(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "write_ratings": True,
                    "two_stage": True,
                    "dry_run": True,
                    "force_score": True,
                    "use_cache_only": True,
                }
            ),
            encoding="utf-8",
        )

        cfg = resolve_config(
            argv=[
                "--config",
                str(config_file),
                "--no-write-ratings",
                "--no-two-stage",
                "--no-dry-run",
                "--no-force-score",
                "--no-use-cache-only",
            ],
            env={},
        )

        assert cfg.write_ratings is False
        assert cfg.two_stage is False
        assert cfg.dry_run is False
        assert cfg.force_score is False
        assert cfg.use_cache_only is False


def _make_jpeg_bytes(width: int = 800, height: int = 600, color: str = "red") -> bytes:
    import io

    from PIL import Image

    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestSingleStageWorkflow:
    def test_single_stage_workflow_evaluates_uncached_syncs_ratings_deduplicates_and_creates_album(
        self,
    ) -> None:
        import io
        from unittest.mock import MagicMock

        from PIL import Image

        from immich_aesthetic_scorer.evaluators import FakeEvaluator
        from immich_aesthetic_scorer.pipeline import ScorePipeline
        from immich_aesthetic_scorer.store import EvaluationStore
        from score_assets import run_scoring_workflow

        # 3 assets: a1 is cached, a2 and a3 are within a 5s burst window (a2 has higher score than a3)
        assets = [
            {
                "id": "a1",
                "type": "IMAGE",
                "originalFileName": "one.jpg",
                "fileModifiedAt": "2026-07-24T10:00:00Z",
                "localDateTime": "2026-07-24T10:00:00",
            },
            {
                "id": "a2",
                "type": "IMAGE",
                "originalFileName": "two.jpg",
                "fileModifiedAt": "2026-07-24T10:05:00Z",
                "localDateTime": "2026-07-24T10:05:00",
            },
            {
                "id": "a3",
                "type": "IMAGE",
                "originalFileName": "three.jpg",
                "fileModifiedAt": "2026-07-24T10:05:03Z",
                "localDateTime": "2026-07-24T10:05:03",
            },
        ]

        gateway = MagicMock()
        gateway.fetch_image_assets.return_value = assets
        raw_800x600 = _make_jpeg_bytes(800, 600, color="blue")
        gateway.download_thumbnail.return_value = raw_800x600
        gateway.ensure_album_with_assets.return_value = "highlights-album-id"

        # Pre-populate a1 in store so only a2 and a3 are downloaded/evaluated
        store = EvaluationStore.in_memory()
        store.record_stage1(assets[0], raw_score=5.0, model_id="fake-s1")

        # Give a2 raw score 9.0 and a3 raw score 7.0
        call_scores = [9.0, 7.0]
        s1_eval = FakeEvaluator(
            scores=lambda _b: call_scores.pop(0),
            model_id="fake-s1",
            required_max_dimension=512,
        )

        cfg = ScorerConfig(
            immich_url="http://immich.local:2283",
            api_key="secret",
            person_id="person-1",
            target_album_name="Best of Person",
            scorer_type="local",
            local_model_id="fake-s1",
            concurrency=1,
            write_ratings=True,
            dedup_window=10,
            limit=5,
            two_stage=False,
        )

        top_assets = run_scoring_workflow(
            cfg,
            gateway=gateway,
            store=store,
            stage1_evaluator=s1_eval,
            pipeline=ScorePipeline(dedup_window=10.0),
        )

        # Only uncached assets (a2, a3) should have thumbnails downloaded & evaluated
        assert gateway.download_thumbnail.call_count == 2
        assert len(s1_eval.evaluated_images) == 2

        # Evaluators with required_max_dimension=512 receive downscaled thumbnails (<= 512px)
        evaluated_img = Image.open(io.BytesIO(s1_eval.evaluated_images[0]))
        assert max(evaluated_img.size) == 512

        # All 3 scored assets have their calibrated star ratings synced in batches by star bucket
        synced_ids = [
            aid for call in gateway.update_assets_rating.call_args_list for aid in call.args[0]
        ]
        assert sorted(synced_ids) == ["a1", "a2", "a3"]

        # Burst deduplication within 10s keeps a2 (score > a3) and a1 -> 2 assets in album
        assert [item["id"] for item in top_assets] == ["a2", "a1"]
        gateway.ensure_album_with_assets.assert_called_once_with("Best of Person", ["a2", "a1"])

        # Final calibrated scores are persisted in the store
        entry_a2 = store.get_entry("a2")
        assert entry_a2 is not None
        assert entry_a2["score"] > 50
        assert "Local CLIP score:" in entry_a2["reason"]


class TestTwoStageWorkflow:
    def test_two_stage_workflow_promotes_top_candidates_fuses_scores_and_respects_dry_run(
        self,
    ) -> None:
        import io
        from unittest.mock import MagicMock

        from PIL import Image

        from immich_aesthetic_scorer.evaluators import EvaluationResult, FakeEvaluator
        from immich_aesthetic_scorer.pipeline import ScorePipeline
        from immich_aesthetic_scorer.store import EvaluationStore
        from score_assets import run_scoring_workflow

        # 4 assets: top 50% (2 candidates) should be promoted to Stage 2
        assets = [
            {
                "id": f"img-{i}",
                "type": "IMAGE",
                "originalFileName": f"img-{i}.jpg",
                "fileModifiedAt": "2026-07-24T10:00:00Z",
                "fileCreatedAt": f"2026-07-24T10:0{i}:00Z",
            }
            for i in range(1, 5)
        ]

        gateway = MagicMock()
        gateway.fetch_image_assets.return_value = assets
        raw_1000x800 = _make_jpeg_bytes(1000, 800, color="green")
        gateway.download_thumbnail.return_value = raw_1000x800

        store = EvaluationStore.in_memory()
        # Pre-cache Stage 1 for all 4 assets: img-1 (9.0), img-2 (8.0), img-3 (5.0), img-4 (4.0)
        for asset, raw_s1 in zip(assets, [9.0, 8.0, 5.0, 4.0]):
            store.record_stage1(asset, raw_score=raw_s1, model_id="fake-s1")

        # Also pre-cache Stage 2 for img-1 so only img-2 needs Stage 2 download & evaluation
        store.record_stage2(
            assets[0],
            raw_score=60.0,
            model_id="musiq-spaq",
            reason="Cached S2 reason",
        )

        s1_eval = FakeEvaluator(model_id="fake-s1", required_max_dimension=512)
        # Stage 2 evaluator with required_max_dimension=None (full-res preview, e.g. musiq-spaq)
        # Gives img-2 a higher Stage 2 raw score (90.0) than img-1 (60.0)
        s2_eval = FakeEvaluator(
            scores=lambda _b: EvaluationResult(raw_score=90.0, reason="Sharp focus"),
            model_id="musiq-spaq",
            required_max_dimension=None,
        )

        cfg = ScorerConfig(
            immich_url="http://immich.local:2283",
            api_key="secret",
            album_id="source-album",
            target_album_name="Best of Source",
            scorer_type="local",
            local_model_id="fake-s1",
            two_stage=True,
            stage2_top_pct=50.0,
            stage2_model="musiq-spaq",
            stage2_weight=0.5,
            write_ratings=True,
            dry_run=True,
            limit=2,
            concurrency=1,
        )

        top_assets = run_scoring_workflow(
            cfg,
            gateway=gateway,
            store=store,
            stage1_evaluator=s1_eval,
            stage2_evaluator=s2_eval,
            pipeline=ScorePipeline(stage2_weight=0.5),
        )

        # Stage 1 was 100% cached; Stage 2 had 2 candidates (img-1 cached, img-2 uncached)
        assert len(s1_eval.evaluated_images) == 0
        assert gateway.download_thumbnail.call_count == 1
        assert len(s2_eval.evaluated_images) == 1

        # Stage 2 evaluator with required_max_dimension=None receives unscaled 1000x800 preview
        s2_img = Image.open(io.BytesIO(s2_eval.evaluated_images[0]))
        assert s2_img.size == (1000, 800)

        # Dry run skips both star rating sync and album creation/addition
        gateway.update_assets_rating.assert_not_called()
        gateway.update_asset_rating.assert_not_called()
        gateway.ensure_album_with_assets.assert_not_called()

        # Candidate entries have Two-stage reason and persisted Stage 2 metadata;
        # Non-candidates have Stage 1 only fallback reason
        entry_1 = store.get_entry("img-1")
        entry_2 = store.get_entry("img-2")
        entry_3 = store.get_entry("img-3")
        assert entry_1 is not None and "Two-stage:" in entry_1["reason"]
        assert entry_1["reason_stage2"] == "Cached S2 reason"
        assert entry_2 is not None and "Two-stage:" in entry_2["reason"]
        assert entry_2["reason_stage2"] == "Sharp focus"
        assert entry_3 is not None and "Stage 1 only:" in entry_3["reason"]
        assert len(top_assets) == 2

    def test_main_coordinates_end_to_end_workflow_with_cache_only(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        from unittest.mock import MagicMock

        import score_assets
        from immich_aesthetic_scorer.gateway import ServerVersion
        from immich_aesthetic_scorer.store import save_cache

        cache_path = tmp_path / "cache.json"
        save_cache(
            {
                "cached-1": {
                    "raw_score_stage1": 8.8,
                    "model_id_stage1": "somepago/AestheticSigLIP",
                    "updatedAt": "2026-07-24T10:00:00Z",
                }
            },
            str(cache_path),
        )

        fake_gateway = MagicMock()
        fake_gateway.verify_connection.return_value = ServerVersion(major=1, minor=120, patch=0)
        fake_gateway.get_person_details.return_value = {"id": "p-1", "name": "Madeline"}
        fake_gateway.fetch_image_assets.return_value = [
            {
                "id": "cached-1",
                "type": "IMAGE",
                "originalFileName": "cached.jpg",
                "fileModifiedAt": "2026-07-24T10:00:00Z",
                "fileCreatedAt": "2026-07-24T10:00:00Z",
            },
            {
                "id": "uncached-2",
                "type": "IMAGE",
                "originalFileName": "uncached.jpg",
                "fileModifiedAt": "2026-07-24T10:01:00Z",
                "fileCreatedAt": "2026-07-24T10:01:00Z",
            },
        ]
        fake_gateway.ensure_album_with_assets.return_value = "album-out"

        monkeypatch.setattr(score_assets, "ImmichGateway", lambda *a, **kw: fake_gateway)  # type: ignore[attr-defined]

        score_assets.main(
            argv=[
                "--config",
                str(tmp_path / "nonexistent.json"),
                "--immich-url",
                "http://immich.local:2283",
                "--api-key",
                "secret-key",
                "--person-id",
                "p-1",
                "--scorer-type",
                "local",
                "--cache-file",
                str(cache_path),
                "--use-cache-only",
            ]
        )

        # Verify main() used ImmichGateway and compiled "Best of Madeline" from cached-1 only
        fake_gateway.verify_connection.assert_called_once()
        fake_gateway.download_thumbnail.assert_not_called()
        fake_gateway.ensure_album_with_assets.assert_called_once_with(
            "Best of Madeline", ["cached-1"]
        )
