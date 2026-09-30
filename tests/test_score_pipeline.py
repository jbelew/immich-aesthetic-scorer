import pytest

from immich_aesthetic_scorer.pipeline import (
    PopulationStats,
    ScorePipeline,
    calculate_population_stats,
    deduplicate_bursts,
    fuse_scores,
    normalize_z_score,
    parse_asset_time,
    score_to_stars,
    select_candidates,
)


def test_calculate_population_stats_standard():
    # Mean: (2+4+4+4+5+5+7+9)/8 = 40/8 = 5.0
    # Population variance: ((2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2) / 8
    # = (9 + 3 + 0 + 4 + 16) / 8 = 32 / 8 = 4.0
    # Population std: sqrt(4.0) = 2.0
    scores = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stats = calculate_population_stats(scores, min_std=1.0)
    assert isinstance(stats, PopulationStats)
    assert stats.mean == pytest.approx(5.0)
    assert stats.std == pytest.approx(2.0)


def test_calculate_population_stats_enforces_floor_guard():
    # Uniform scores: variance = 0.0, std = 0.0 < floor 1.0
    uniform_scores = [5.0, 5.0, 5.0]
    stats_s1 = calculate_population_stats(uniform_scores, min_std=1.0)
    assert stats_s1.mean == pytest.approx(5.0)
    assert stats_s1.std == 1.0

    # Narrow scores: std < 1.0
    narrow_scores = [5.0, 5.2, 4.8]
    stats_narrow = calculate_population_stats(narrow_scores, min_std=1.0)
    assert stats_narrow.mean == pytest.approx(5.0)
    assert stats_narrow.std == 1.0

    # Stage 2 floor guard: min_std=0.01
    stats_s2 = calculate_population_stats(uniform_scores, min_std=0.01)
    assert stats_s2.mean == pytest.approx(5.0)
    assert stats_s2.std == 0.01

    # Custom floor guard: min_std=2.0
    stats_custom = calculate_population_stats(uniform_scores, min_std=2.0)
    assert stats_custom.mean == pytest.approx(5.0)
    assert stats_custom.std == 2.0


def test_calculate_population_stats_empty_and_single_fallbacks():
    empty_stats = calculate_population_stats([], min_std=1.0, default_mean=6.0, default_std=1.0)
    assert empty_stats.mean == 6.0
    assert empty_stats.std == 1.0

    single_stats = calculate_population_stats([7.5], min_std=1.0, default_mean=6.0, default_std=1.0)
    assert single_stats.mean == 6.0
    assert single_stats.std == 1.0


def test_normalize_z_score_exact_mean_is_50():
    # When raw == mean, z = 0, sigmoid = 100 / (1 + 1) = 50.0
    assert normalize_z_score(5.0, mean=5.0, std=1.0) == pytest.approx(50.0)


def test_normalize_z_score_formula_and_bounds():
    # z = (7.0 - 5.0) / 1.0 = 2.0
    # S = 100 / (1 + exp(-1.5 * 2.0)) = 100 / (1 + exp(-3.0)) ~ 95.2574
    expected_pos = 100.0 / (1.0 + 0.049787068)
    assert normalize_z_score(7.0, mean=5.0, std=1.0) == pytest.approx(expected_pos, rel=1e-4)

    # z = (3.0 - 5.0) / 1.0 = -2.0
    # S = 100 / (1 + exp(-1.5 * -2.0)) = 100 / (1 + exp(3.0)) ~ 4.74258
    expected_neg = 100.0 / (1.0 + 20.0855369)
    assert normalize_z_score(3.0, mean=5.0, std=1.0) == pytest.approx(expected_neg, rel=1e-4)

    # Extreme values asymptote and clamp cleanly to [0.0, 100.0]
    assert normalize_z_score(1000.0, mean=5.0, std=1.0) == 100.0
    assert normalize_z_score(-1000.0, mean=5.0, std=1.0) == 0.0

    # Zero or invalid std dev gracefully defaults to std=1.0
    assert normalize_z_score(5.0, mean=5.0, std=0.0) == 50.0


def test_select_candidates_percentile_split():
    # 10 items with s1_norm scores from 10 to 100
    items = [{"id": f"img{i}", "s1_norm": float(i * 10)} for i in range(1, 11)]

    # Top 20% -> 2 items (img10 with 100.0, img9 with 90.0)
    candidates, non_candidates = select_candidates(items, top_pct=20.0)
    assert len(candidates) == 2
    assert len(non_candidates) == 8
    assert [c["id"] for c in candidates] == ["img10", "img9"]
    assert [nc["id"] for nc in non_candidates] == [f"img{i}" for i in range(8, 0, -1)]


def test_select_candidates_edge_cases():
    # Empty items
    empty_cand, empty_non_cand = select_candidates([], top_pct=20.0)
    assert empty_cand == []
    assert empty_non_cand == []

    # Single item with small top_pct still guarantees at least 1 candidate
    single_item = [{"id": "only", "s1_norm": 42.0}]
    c, nc = select_candidates(single_item, top_pct=5.0)
    assert len(c) == 1
    assert c[0]["id"] == "only"
    assert nc == []

    # 100% selects all items
    items = [{"id": f"img{i}", "s1_norm": float(i)} for i in range(3)]
    c, nc = select_candidates(items, top_pct=100.0)
    assert len(c) == 3
    assert len(nc) == 0


def test_fuse_scores_candidate():
    # S1 = 80.0, S2 = 60.0, weight = 0.5 -> 0.5 * 80 + 0.5 * 60 = 70.0
    assert fuse_scores(80.0, s2_norm=60.0, stage2_weight=0.5) == pytest.approx(70.0)

    # Custom weight: weight = 0.25 -> 0.75 * 80 + 0.25 * 60 = 60 + 15 = 75.0
    assert fuse_scores(80.0, s2_norm=60.0, stage2_weight=0.25) == pytest.approx(75.0)

    # Weight extremes: 0.0 uses S1 only, 1.0 uses S2 only
    assert fuse_scores(80.0, s2_norm=60.0, stage2_weight=0.0) == pytest.approx(80.0)
    assert fuse_scores(80.0, s2_norm=60.0, stage2_weight=1.0) == pytest.approx(60.0)


def test_fuse_scores_non_candidate_fallback():
    # Non-candidate has s2_norm=None, receives continuous fallback S2 = 50.0
    # S1 = 80.0, weight = 0.5 -> 0.5 * 80 + 0.5 * 50 = 40 + 25 = 65.0
    assert fuse_scores(80.0, s2_norm=None, stage2_weight=0.5) == pytest.approx(65.0)

    # Continuity test: A candidate with S2 = 50.0 gets the exact same score as a non-candidate
    cand_score = fuse_scores(72.5, s2_norm=50.0, stage2_weight=0.4)
    non_cand_score = fuse_scores(72.5, s2_norm=None, stage2_weight=0.4)
    assert cand_score == pytest.approx(non_cand_score)


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


def test_parse_asset_time():
    from datetime import datetime

    # ISO format with Z
    item1 = {"asset": {"fileCreatedAt": "2026-07-25T14:30:00.000Z"}}
    dt1 = parse_asset_time(item1)
    assert dt1 == datetime(2026, 7, 25, 14, 30, 0)
    assert dt1.tzinfo is None  # Must be timezone-naive

    # localDateTime field
    item2 = {"asset": {"localDateTime": "2026-07-25T15:00:00"}}
    assert parse_asset_time(item2) == datetime(2026, 7, 25, 15, 0, 0)

    # Flat dict support
    item3 = {"createdAt": "2026-07-25T16:00:00+00:00"}
    assert parse_asset_time(item3) == datetime(2026, 7, 25, 16, 0, 0)

    # Missing or invalid timestamp returns datetime.min
    assert parse_asset_time({}) == datetime.min
    assert parse_asset_time({"asset": {"createdAt": "invalid-date"}}) == datetime.min


def test_deduplicate_bursts_clustering():
    # 3 photos in a 2-second burst window (t=0s, t=1s, t=2s), highest score is img2 (score 92)
    assets = [
        {"id": "img1", "score": 80, "asset": {"fileCreatedAt": "2026-07-25T12:00:00Z"}},
        {"id": "img2", "score": 92, "asset": {"fileCreatedAt": "2026-07-25T12:00:01Z"}},
        {"id": "img3", "score": 85, "asset": {"fileCreatedAt": "2026-07-25T12:00:02Z"}},
        # Separate photo 10 seconds later
        {"id": "img4", "score": 70, "asset": {"fileCreatedAt": "2026-07-25T12:00:12Z"}},
    ]

    deduped = deduplicate_bursts(assets, dedup_window=2.0)
    assert len(deduped) == 2
    assert deduped[0]["id"] == "img2"
    assert deduped[1]["id"] == "img4"


def test_deduplicate_bursts_edge_cases():
    # dedup_window <= 0 disables deduplication
    assets = [
        {"id": "img1", "score": 80, "asset": {"fileCreatedAt": "2026-07-25T12:00:00Z"}},
        {"id": "img2", "score": 90, "asset": {"fileCreatedAt": "2026-07-25T12:00:01Z"}},
    ]
    assert len(deduplicate_bursts(assets, dedup_window=0)) == 2
    assert len(deduplicate_bursts(assets, dedup_window=-1.0)) == 2

    # Empty list
    assert deduplicate_bursts([], dedup_window=5.0) == []

    # Out-of-order input is sorted chronologically before clustering
    unordered = [
        {"id": "img2", "score": 95, "asset": {"fileCreatedAt": "2026-07-25T12:00:01Z"}},
        {"id": "img1", "score": 80, "asset": {"fileCreatedAt": "2026-07-25T12:00:00Z"}},
    ]
    deduped = deduplicate_bursts(unordered, dedup_window=2.0)
    assert len(deduped) == 1
    assert deduped[0]["id"] == "img2"

    # Photos with unparseable timestamps are preserved individually without error
    unparseable = [
        {"id": "broken1", "score": 50, "asset": {}},
        {"id": "valid1", "score": 90, "asset": {"fileCreatedAt": "2026-07-25T12:00:00Z"}},
        {"id": "valid2", "score": 80, "asset": {"fileCreatedAt": "2026-07-25T12:00:01Z"}},
    ]
    res = deduplicate_bursts(unparseable, dedup_window=2.0)
    assert len(res) == 2
    ids = [x["id"] for x in res]
    assert "broken1" in ids
    assert "valid1" in ids
    assert "valid2" not in ids


def test_score_pipeline_single_stage():
    pipeline = ScorePipeline(min_std_s1=1.0)
    assets = [
        {"id": "a1", "raw_score_stage1": 8.0},
        {"id": "a2", "raw_score_stage1": 6.0},
        {"id": "a3", "raw_score_stage1": 4.0},
    ]

    stats = pipeline.calibrate_stage1(assets)
    assert stats.mean == pytest.approx(6.0)
    assert "s1_norm" in assets[0]
    assert assets[0]["s1_norm"] > 50.0
    assert assets[1]["s1_norm"] == pytest.approx(50.0)
    assert assets[2]["s1_norm"] < 50.0

    finalized = pipeline.single_stage_finalize(assets)
    assert finalized[0]["score"] > 50
    assert finalized[1]["score"] == 50
    assert finalized[1]["rating"] == 3
    assert "reason" in finalized[0]


def test_score_pipeline_two_stage():
    pipeline = ScorePipeline(stage2_weight=0.5)
    assets = [
        {"id": "a1", "raw_score_stage1": 9.0},
        {"id": "a2", "raw_score_stage1": 8.0},
        {"id": "a3", "raw_score_stage1": 5.0},
        {"id": "a4", "raw_score_stage1": 4.0},
    ]

    pipeline.calibrate_stage1(assets)
    candidates, non_candidates = pipeline.split_candidates(assets, top_pct=50.0)
    assert len(candidates) == 2
    assert [c["id"] for c in candidates] == ["a1", "a2"]

    # Provide stage 2 raw scores for candidates
    candidates[0]["raw_score_stage2"] = 80.0
    candidates[1]["raw_score_stage2"] = 60.0

    fused = pipeline.fuse_stage2(assets)
    by_id = {item["id"]: item for item in fused}

    # Candidates should have fused scores
    assert "s2_norm" in by_id["a1"]
    assert "s2_norm" in by_id["a2"]
    assert by_id["a1"]["score"] >= by_id["a2"]["score"]
    assert "Two-stage:" in by_id["a1"]["reason"]

    # Non-candidates should have fallback scores
    assert "s2_norm" not in by_id["a3"]
    assert "Stage 1 only:" in by_id["a3"]["reason"]
    assert 1 <= by_id["a3"]["rating"] <= 5


def test_score_pipeline_uniform_scores_zero_variance():
    pipeline = ScorePipeline(min_std_s1=1.0)
    assets = [
        {"id": "u1", "raw_score_stage1": 7.0},
        {"id": "u2", "raw_score_stage1": 7.0},
        {"id": "u3", "raw_score_stage1": 7.0},
    ]

    stats = pipeline.calibrate_stage1(assets)
    assert stats.mean == 7.0
    assert stats.std == 1.0  # Floor enforced

    finalized = pipeline.single_stage_finalize(assets)
    for item in finalized:
        assert item["score"] == 50
        assert item["rating"] == 3


def test_score_pipeline_empty_collection():
    pipeline = ScorePipeline()
    stats = pipeline.calibrate_stage1([])
    assert stats.mean == 6.0
    assert stats.std == 1.0
    assert pipeline.single_stage_finalize([]) == []
    c, nc = pipeline.split_candidates([], top_pct=20.0)
    assert c == [] and nc == []
    assert pipeline.fuse_stage2([]) == []
    assert pipeline.deduplicate([]) == []
