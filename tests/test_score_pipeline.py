import io

import pytest
from PIL import Image

import immich_aesthetic_scorer.pipeline as pipeline_mod
from immich_aesthetic_scorer.evaluators import EvaluationResult, FakeEvaluator
from immich_aesthetic_scorer.pipeline import (
    PipelineResult,
    PopulationStats,
    ScorePipeline,
    score_to_stars,
)
from immich_aesthetic_scorer.store import EvaluationStore


def _make_jpeg_bytes(width: int = 800, height: int = 600, color: str = "blue") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_public_module_interface_is_narrowed():
    assert set(pipeline_mod.__all__) == {
        "PipelineResult",
        "PopulationStats",
        "ScorePipeline",
        "score_to_stars",
    }
    for legacy_helper in (
        "calculate_population_stats",
        "normalize_z_score",
        "select_candidates",
        "fuse_scores",
        "parse_asset_time",
        "deduplicate_bursts",
    ):
        assert not hasattr(pipeline_mod, legacy_helper)


def test_score_to_stars_thresholds():
    # 90+ -> 5
    assert score_to_stars(100.0) == 5
    assert score_to_stars(95) == 5
    assert score_to_stars(90.0) == 5

    # 75 to <90 -> 4
    assert score_to_stars(89.9) == 4
    assert score_to_stars(75.0) == 4

    # 50 to <75 -> 3
    assert score_to_stars(74.9) == 3
    assert score_to_stars(50.0) == 3

    # 20 to <50 -> 2
    assert score_to_stars(49.9) == 2
    assert score_to_stars(20.0) == 2

    # <20 -> 1
    assert score_to_stars(19.9) == 1
    assert score_to_stars(0.0) == 1
    assert score_to_stars(-10.0) == 1


def test_run_single_stage_evaluates_downscales_calibrates_and_persists_to_store():
    # Mean: (2+4+4+4+5+5+7+9)/8 = 40/8 = 5.0
    # Population variance: ((2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2) / 8 = 32 / 8 = 4.0
    # Population std: sqrt(4.0) = 2.0
    raw_scores = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assets = [
        {
            "id": f"a{i}",
            "originalFileName": f"photo_{i}.jpg",
            "fileModifiedAt": "2026-07-25T10:00:00Z",
            "fileCreatedAt": f"2026-07-25T10:0{i}:00Z",
        }
        for i in range(len(raw_scores))
    ]
    score_by_id = {f"a{i}": score for i, score in enumerate(raw_scores)}
    raw_800x600 = _make_jpeg_bytes(800, 600)

    loaded_ids: list[str] = []

    def thumbnail_loader(asset_id: str) -> bytes:
        loaded_ids.append(asset_id)
        return raw_800x600

    eval_iter = iter(raw_scores)
    s1_eval = FakeEvaluator(
        scores=lambda _b: next(eval_iter),
        model_id="fake-s1",
        required_max_dimension=512,
    )
    store = EvaluationStore.in_memory()
    pipeline = ScorePipeline(min_std_s1=1.0)

    result = pipeline.run(
        assets=assets,
        store=store,
        stage1_evaluator=s1_eval,
        thumbnail_loader=thumbnail_loader,
        concurrency=1,
        scorer_type="local",
    )

    assert isinstance(result, PipelineResult)
    assert isinstance(result.stage1_stats, PopulationStats)
    assert result.stage1_stats.mean == pytest.approx(5.0)
    assert result.stage1_stats.std == pytest.approx(2.0)
    assert result.stage2_stats is None

    # All 8 thumbnails loaded and downscaled to max_dim=512
    assert len(loaded_ids) == 8
    assert len(s1_eval.evaluated_images) == 8
    first_img = Image.open(io.BytesIO(s1_eval.evaluated_images[0]))
    assert max(first_img.size) == 512

    by_id = {item["id"]: item for item in result.scored_assets}
    # a4 has raw=5.0 == mean -> z=0 -> s1_norm=50.0 -> score=50, rating=3
    assert by_id["a4"]["s1_norm"] == pytest.approx(50.0)
    assert by_id["a4"]["score"] == 50
    assert by_id["a4"]["rating"] == 3

    # a7 has raw=9.0 -> z=(9-5)/2 = 2.0 -> 100 / (1 + exp(-3.0)) ~ 95.2574 -> score=95, rating=5
    expected_pos = 100.0 / (1.0 + 0.049787068)
    assert by_id["a7"]["s1_norm"] == pytest.approx(expected_pos, rel=1e-4)
    assert by_id["a7"]["score"] == 95
    assert by_id["a7"]["rating"] == 5

    # deduplicated_assets is sorted descending by calibrated score
    dedup_scores = [item["score"] for item in result.deduplicated_assets]
    assert dedup_scores == sorted(dedup_scores, reverse=True)
    assert result.deduplicated_assets[0]["id"] == "a7"

    # Store persisted final entries for every asset
    for aid, expected_raw in score_by_id.items():
        entry = store.get_entry(aid)
        assert entry is not None
        assert entry["raw_score_stage1"] == pytest.approx(expected_raw)
        assert entry["model_id_stage1"] == "fake-s1"
        assert entry["score"] == by_id[aid]["score"]
        assert "Local CLIP score:" in entry["reason"]


def test_run_enforces_stage1_and_stage2_variance_floor_guards():
    assets = [{"id": f"u{i}", "fileModifiedAt": "2026-07-25T10:00:00Z"} for i in range(4)]
    # Narrow Stage 1 scores: [5.0, 5.2, 4.8, 5.0] -> std < 1.0 -> floor 1.0 enforced
    s1_scores = iter([5.0, 5.2, 4.8, 5.0])
    s1_eval = FakeEvaluator(
        scores=lambda _b: next(s1_scores),
        model_id="fake-s1",
        required_max_dimension=None,
    )
    # Uniform Stage 2 scores: [70.0, 70.0] -> std == 0.0 -> floor 0.01 enforced
    s2_eval = FakeEvaluator(
        scores=lambda _b: 70.0,
        model_id="musiq-spaq",
        required_max_dimension=None,
    )
    store = EvaluationStore.in_memory()
    pipeline = ScorePipeline(min_std_s1=1.0, min_std_s2=0.01, stage2_weight=0.5)

    result = pipeline.run(
        assets=assets,
        store=store,
        stage1_evaluator=s1_eval,
        stage2_evaluator=s2_eval,
        thumbnail_loader=lambda _aid: b"img",
        stage2_top_pct=50.0,
        concurrency=1,
    )

    assert result.stage1_stats.mean == pytest.approx(5.0)
    assert result.stage1_stats.std == 1.0
    assert result.stage2_stats is not None
    assert result.stage2_stats.mean == pytest.approx(70.0)
    assert result.stage2_stats.std == 0.01

    # Custom floor guard via constructor
    custom_pipeline = ScorePipeline(min_std_s1=2.0)
    uniform_s1 = FakeEvaluator(
        scores=lambda _b: 7.0, model_id="fake-uniform", required_max_dimension=None
    )
    custom_res = custom_pipeline.run(
        assets=assets[:3],
        store=EvaluationStore.in_memory(),
        stage1_evaluator=uniform_s1,
        thumbnail_loader=lambda _aid: b"img",
        concurrency=1,
    )
    assert custom_res.stage1_stats.mean == pytest.approx(7.0)
    assert custom_res.stage1_stats.std == 2.0
    for item in custom_res.scored_assets:
        assert item["score"] == 50
        assert item["rating"] == 3


def test_run_empty_and_single_asset_population_fallbacks():
    pipeline = ScorePipeline()
    s1_eval = FakeEvaluator(default_score=7.5, model_id="fake-s1", required_max_dimension=None)
    s2_eval = FakeEvaluator(default_score=80.0, model_id="musiq-spaq", required_max_dimension=None)

    # Empty collection
    empty_res = pipeline.run(
        assets=[],
        store=EvaluationStore.in_memory(),
        stage1_evaluator=s1_eval,
        stage2_evaluator=s2_eval,
        thumbnail_loader=lambda _aid: b"img",
    )
    assert empty_res.scored_assets == []
    assert empty_res.deduplicated_assets == []
    assert empty_res.stage1_stats == PopulationStats(mean=6.0, std=1.0)
    assert empty_res.stage2_stats == PopulationStats(mean=50.0, std=10.0)

    # Single asset collection uses default_mean=6.0, default_std=1.0 for S1
    # and guarantees at least 1 candidate promoted to Stage 2 even with small top_pct
    single_res = pipeline.run(
        assets=[{"id": "only", "fileModifiedAt": "2026-07-25T10:00:00Z"}],
        store=EvaluationStore.in_memory(),
        stage1_evaluator=s1_eval,
        stage2_evaluator=s2_eval,
        thumbnail_loader=lambda _aid: b"img",
        stage2_top_pct=5.0,
        concurrency=1,
    )
    assert single_res.stage1_stats == PopulationStats(mean=6.0, std=1.0)
    assert single_res.stage2_stats == PopulationStats(mean=50.0, std=10.0)
    assert len(single_res.scored_assets) == 1
    assert "s2_norm" in single_res.scored_assets[0]


def test_run_sigmoid_extreme_bounds():
    # Single-asset population uses fallback mean=6.0, std=1.0, so extreme raw scores
    # produce |z| >> 700 and clamp cleanly to 0.0 and 100.0 without OverflowError
    pipeline = ScorePipeline(min_std_s1=1.0)

    low_res = pipeline.run(
        assets=[{"id": "low", "fileModifiedAt": "2026-07-25T10:00:00Z"}],
        store=EvaluationStore.in_memory(),
        stage1_evaluator=FakeEvaluator(
            default_score=-10000.0, model_id="fake-s1", required_max_dimension=None
        ),
        thumbnail_loader=lambda _aid: b"low",
        concurrency=1,
    )
    assert low_res.scored_assets[0]["s1_norm"] == 0.0
    assert low_res.scored_assets[0]["score"] == 0

    high_res = pipeline.run(
        assets=[{"id": "high", "fileModifiedAt": "2026-07-25T10:00:00Z"}],
        store=EvaluationStore.in_memory(),
        stage1_evaluator=FakeEvaluator(
            default_score=10000.0, model_id="fake-s1", required_max_dimension=None
        ),
        thumbnail_loader=lambda _aid: b"high",
        concurrency=1,
    )
    assert high_res.scored_assets[0]["s1_norm"] == 100.0
    assert high_res.scored_assets[0]["score"] == 100


def test_run_two_stage_candidate_promotion_and_continuous_50_fallback_fusion():
    # 10 items with Stage 1 raw scores 1.0..10.0 -> top 20% promotes 2 candidates (img10, img9)
    assets = [{"id": f"img{i}", "fileModifiedAt": "2026-07-25T10:00:00Z"} for i in range(1, 11)]
    s1_map = {f"img{i}".encode(): float(i) for i in range(1, 11)}
    # img10 gets S2=80.0, img9 gets S2=60.0 -> S2 mean=70.0, std=10.0
    s2_map = {
        b"img10": EvaluationResult(raw_score=80.0, reason="Crisp details"),
        b"img9": EvaluationResult(raw_score=60.0, reason="Soft edges"),
    }

    s1_eval = FakeEvaluator(scores=s1_map, model_id="fake-s1", required_max_dimension=None)
    s2_eval = FakeEvaluator(scores=s2_map, model_id="musiq-spaq", required_max_dimension=None)
    store = EvaluationStore.in_memory()
    pipeline = ScorePipeline(stage2_weight=0.4)

    res = pipeline.run(
        assets=assets,
        store=store,
        stage1_evaluator=s1_eval,
        stage2_evaluator=s2_eval,
        thumbnail_loader=lambda aid: aid.encode(),
        stage2_top_pct=20.0,
        concurrency=1,
    )

    # Exactly 2 candidates evaluated in Stage 2
    assert set(s2_eval.evaluated_images) == {b"img10", b"img9"}
    assert res.stage2_stats == PopulationStats(mean=70.0, std=10.0)

    by_id = {item["id"]: item for item in res.scored_assets}
    assert "s2_norm" in by_id["img10"]
    assert "s2_norm" in by_id["img9"]
    assert "Two-stage:" in by_id["img10"]["reason"]

    # Non-candidates receive continuous fallback S2 = 50.0:
    # combined = (1 - 0.4) * s1_norm + 0.4 * 50.0
    for i in range(1, 9):
        nc = by_id[f"img{i}"]
        assert "s2_norm" not in nc
        expected_combined = 0.6 * nc["s1_norm"] + 0.4 * 50.0
        assert nc["score"] == int(round(expected_combined))
        assert "Stage 1 only:" in nc["reason"]

    # Continuity invariant: a candidate whose S2 raw equals the S2 population mean (s2_norm == 50.0)
    # receives the exact same fused score as a non-candidate with the same s1_norm
    cand_at_mean_s2 = {
        b"c1": EvaluationResult(raw_score=80.0),
        b"c2": EvaluationResult(raw_score=70.0),  # exactly at mean 70.0 -> s2_norm = 50.0
        b"c3": EvaluationResult(raw_score=60.0),
    }
    cont_assets = [
        {"id": "c1", "fileModifiedAt": "2026-07-25T10:00:00Z"},
        {"id": "c2", "fileModifiedAt": "2026-07-25T10:00:00Z"},
        {"id": "c3", "fileModifiedAt": "2026-07-25T10:00:00Z"},
        {"id": "nc", "fileModifiedAt": "2026-07-25T10:00:00Z"},
    ]
    # Give c2 and nc the exact same Stage 1 score (7.0), while c1=9.0, c3=8.0
    # Top 75% (3 of 4) promotes c1, c3, and c2 (or nc); pre-seed store so c1, c2, c3 are the top 3
    cont_s1 = {b"c1": 9.0, b"c3": 8.0, b"c2": 7.0, b"nc": 6.9999}
    cont_res = pipeline.run(
        assets=cont_assets,
        store=EvaluationStore.in_memory(),
        stage1_evaluator=FakeEvaluator(scores=cont_s1, model_id="s1", required_max_dimension=None),
        stage2_evaluator=FakeEvaluator(
            scores=cand_at_mean_s2, model_id="s2", required_max_dimension=None
        ),
        thumbnail_loader=lambda aid: aid.encode(),
        stage2_top_pct=75.0,
        concurrency=1,
    )
    cont_by_id = {item["id"]: item for item in cont_res.scored_assets}
    assert cont_by_id["c2"]["s2_norm"] == pytest.approx(50.0)
    assert cont_by_id["c2"]["score"] == cont_by_id["nc"]["score"]


def test_run_respects_cache_only_and_force_score():
    assets = [
        {"id": "cached", "fileModifiedAt": "2026-07-25T10:00:00Z"},
        {"id": "uncached", "fileModifiedAt": "2026-07-25T10:00:00Z"},
    ]
    store = EvaluationStore.in_memory()
    store.record_stage1(assets[0], raw_score=8.0, model_id="fake-s1")
    store.record_stage2(assets[0], raw_score=85.0, model_id="musiq-spaq", reason="Cached S2")

    s1_eval = FakeEvaluator(default_score=4.0, model_id="fake-s1", required_max_dimension=None)
    s2_eval = FakeEvaluator(default_score=40.0, model_id="musiq-spaq", required_max_dimension=None)
    pipeline = ScorePipeline()

    # 1. use_cache_only=True skips uncached assets without invoking evaluators or thumbnail_loader
    loader_calls: list[str] = []
    cache_only_res = pipeline.run(
        assets=assets,
        store=store,
        stage1_evaluator=s1_eval,
        stage2_evaluator=s2_eval,
        thumbnail_loader=lambda aid: (loader_calls.append(aid), b"img")[1],
        use_cache_only=True,
        stage2_top_pct=100.0,
    )
    assert loader_calls == []
    assert len(s1_eval.evaluated_images) == 0
    assert len(s2_eval.evaluated_images) == 0
    assert [item["id"] for item in cache_only_res.scored_assets] == ["cached"]

    # 2. force_score=True re-evaluates even cached assets
    force_res = pipeline.run(
        assets=[assets[0]],
        store=store,
        stage1_evaluator=s1_eval,
        stage2_evaluator=s2_eval,
        thumbnail_loader=lambda aid: (loader_calls.append(aid), b"img")[1],
        force_score=True,
        stage2_top_pct=100.0,
        concurrency=1,
    )
    assert len(s1_eval.evaluated_images) == 1
    assert len(s2_eval.evaluated_images) == 1
    assert force_res.scored_assets[0]["raw_score_stage1"] == pytest.approx(4.0)
    assert force_res.scored_assets[0]["raw_score_stage2"] == pytest.approx(40.0)


def test_run_burst_deduplication_clusters_chronologically_and_preserves_all_scored_assets():
    # Out-of-order burst of 3 photos within 2s window (img2 has highest score),
    # 1 photo 10s later (img4), and 1 photo with unparseable timestamp (broken)
    assets = [
        {"id": "img3", "fileCreatedAt": "2026-07-25T12:00:02Z"},
        {"id": "img1", "fileCreatedAt": "2026-07-25T12:00:00.000Z"},
        {"id": "img2", "localDateTime": "2026-07-25T12:00:01"},
        {"id": "img4", "createdAt": "2026-07-25T12:00:12+00:00"},
        {"id": "broken", "createdAt": "invalid-date"},
    ]
    raw_map = {
        b"img1": 6.0,
        b"img2": 9.5,
        b"img3": 7.5,
        b"img4": 5.0,
        b"broken": 4.0,
    }
    s1_eval = FakeEvaluator(scores=raw_map, model_id="fake-s1", required_max_dimension=None)
    pipeline = ScorePipeline(dedup_window=2.0)

    res = pipeline.run(
        assets=assets,
        store=EvaluationStore.in_memory(),
        stage1_evaluator=s1_eval,
        thumbnail_loader=lambda aid: aid.encode(),
        concurrency=1,
    )

    # scored_assets retains all 5 evaluated assets (for star rating sync)
    assert len(res.scored_assets) == 5

    # deduplicated_assets keeps img2 (best of img1/img2/img3 burst), img4, and broken, sorted descending by score
    assert [item["id"] for item in res.deduplicated_assets] == ["img2", "img4", "broken"]
