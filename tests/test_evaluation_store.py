from unittest.mock import patch

import pytest

from immich_aesthetic_scorer.store import (
    EvaluationStore,
    extract_raw_score_from_reason,
    get_asset_mod_time,
    load_cache,
    save_cache,
)


def test_in_memory_record_and_get_stage1():
    store = EvaluationStore.in_memory()
    asset = {"id": "asset-1", "fileModifiedAt": "2026-07-25T12:00:00Z"}

    assert store.get_stage1(asset, model_id="somepago/AestheticSigLIP") is None

    store.record_stage1(asset, raw_score=7.42, model_id="somepago/AestheticSigLIP")

    score = store.get_stage1(asset, model_id="somepago/AestheticSigLIP")
    assert score == pytest.approx(7.42)


def test_stage_isolated_model_mismatch_invalidation():
    store = EvaluationStore.in_memory()
    asset = {"id": "asset-1", "fileModifiedAt": "2026-07-25T12:00:00Z"}

    store.record_stage1(asset, raw_score=7.5, model_id="somepago/AestheticSigLIP")
    store.record_stage2(
        asset,
        raw_score=84.0,
        model_id="gemini-2.5-flash",
        reason="Great lighting and expression.",
    )

    # Matching models return cached scores for both stages
    assert store.get_stage1(asset, model_id="somepago/AestheticSigLIP") == pytest.approx(7.5)
    assert store.get_stage2(asset, model_id="gemini-2.5-flash") == pytest.approx(84.0)
    assert (
        store.get_stage2_reason(asset, model_id="gemini-2.5-flash")
        == "Great lighting and expression."
    )

    # Changing Stage 1 model invalidates Stage 1 only; Stage 2 remains valid
    assert store.get_stage1(asset, model_id="rsinema/aesthetic-scorer") is None
    assert store.get_stage2(asset, model_id="gemini-2.5-flash") == pytest.approx(84.0)

    # Changing Stage 2 model invalidates Stage 2 only; Stage 1 remains valid
    assert store.get_stage1(asset, model_id="somepago/AestheticSigLIP") == pytest.approx(7.5)
    assert store.get_stage2(asset, model_id="musiq-spaq") is None
    assert store.get_stage2_reason(asset, model_id="musiq-spaq") is None

    # force=True bypasses cache for the requested stage
    assert store.get_stage1(asset, model_id="somepago/AestheticSigLIP", force=True) is None
    assert store.get_stage2(asset, model_id="gemini-2.5-flash", force=True) is None


def test_asset_timestamp_drift_invalidation_and_mod_time_resolution():
    # MS-DOS epoch (1980-01-01) falls back to fileCreatedAt or createdAt
    assert (
        get_asset_mod_time(
            {
                "id": "a1",
                "fileModifiedAt": "1980-01-01T00:00:00.000Z",
                "fileCreatedAt": "2026-07-20T08:00:00.000Z",
            }
        )
        == "2026-07-20T08:00:00.000Z"
    )
    assert (
        get_asset_mod_time({"id": "a2", "createdAt": "2026-07-21T09:00:00.000Z"})
        == "2026-07-21T09:00:00.000Z"
    )

    store = EvaluationStore.in_memory()
    original_asset = {"id": "asset-1", "fileModifiedAt": "2026-07-25T12:00:00Z"}
    store.record_stage1(original_asset, raw_score=6.8, model_id="somepago/AestheticSigLIP")
    store.record_stage2(original_asset, raw_score=78.0, model_id="gemini-2.5-flash")

    assert store.get_stage1(original_asset, model_id="somepago/AestheticSigLIP") == pytest.approx(
        6.8
    )
    assert store.get_stage2(original_asset, model_id="gemini-2.5-flash") == pytest.approx(78.0)

    # Asset is modified with a newer timestamp -> both Stage 1 and Stage 2 are invalidated
    modified_asset = {"id": "asset-1", "fileModifiedAt": "2026-07-26T15:30:00Z"}
    assert store.get_stage1(modified_asset, model_id="somepago/AestheticSigLIP") is None
    assert store.get_stage2(modified_asset, model_id="gemini-2.5-flash") is None

    # Re-recording Stage 1 on modified asset updates timestamp and clears stale Stage 2 score
    store.record_stage1(modified_asset, raw_score=8.1, model_id="somepago/AestheticSigLIP")
    assert store.get_stage1(modified_asset, model_id="somepago/AestheticSigLIP") == pytest.approx(
        8.1
    )
    assert store.get_stage2("asset-1", model_id="gemini-2.5-flash") is None


def test_legacy_cache_reason_extraction_and_fallback():
    assert extract_raw_score_from_reason("Local CLIP score: 7.55/10.0") == pytest.approx(7.55)
    assert extract_raw_score_from_reason(
        "Local CLIP score: 8.20/10.0 (z-score: 1.35)"
    ) == pytest.approx(8.2)
    assert extract_raw_score_from_reason("Unrelated reason string") is None
    assert extract_raw_score_from_reason(None) is None

    legacy_data = {
        "legacy-reason-only": {
            "score": 76,
            "reason": "Local CLIP score: 7.55/10.0 (z-score: 1.10)",
        },
        "legacy-raw-score": {
            "score": 64,
            "raw_score": 6.4,
        },
        "unparseable-legacy": {
            "score": 80,
            "reason": "Gemini aesthetic evaluation",
        },
    }
    store = EvaluationStore.in_memory(initial_data=legacy_data)

    assert store.get_stage1(
        {"id": "legacy-reason-only", "fileModifiedAt": "2026-07-25T12:00:00Z"},
        model_id="somepago/AestheticSigLIP",
    ) == pytest.approx(7.55)
    assert store.get_stage1(
        {"id": "legacy-raw-score", "fileModifiedAt": "2026-07-25T12:00:00Z"},
        model_id="somepago/AestheticSigLIP",
    ) == pytest.approx(6.4)
    assert (
        store.get_stage1(
            {"id": "unparseable-legacy", "fileModifiedAt": "2026-07-25T12:00:00Z"},
            model_id="somepago/AestheticSigLIP",
        )
        is None
    )


def test_dirty_tracking_and_atomic_disk_persistence(tmp_path):
    cache_file = tmp_path / "cache.json"
    store = EvaluationStore.from_disk(str(cache_file))

    assert len(store) == 0
    assert store.is_dirty is False

    # Flushing an clean store skips disk write entirely
    assert store.flush() is False
    assert not cache_file.exists()

    asset = {"id": "asset-1", "fileModifiedAt": "2026-07-25T12:00:00Z"}
    store.record_stage1(asset, raw_score=7.5, model_id="somepago/AestheticSigLIP")
    assert store.is_dirty is True

    # Verify atomic replacement via temporary file (.tmp -> target)
    with patch(
        "immich_aesthetic_scorer.store.os.replace", wraps=__import__("os").replace
    ) as mock_replace:
        wrote = store.flush()
        assert wrote is True
        mock_replace.assert_called_once_with(f"{cache_file}.tmp", str(cache_file))

    assert store.is_dirty is False
    assert cache_file.exists()
    assert not (tmp_path / "cache.json.tmp").exists()

    # Subsequent flush without mutations skips disk write
    with patch("immich_aesthetic_scorer.store.os.replace") as mock_replace_skip:
        assert store.flush() is False
        mock_replace_skip.assert_not_called()

    # Recording identical Stage 1 data does not mark dirty
    store.record_stage1(asset, raw_score=7.5, model_id="somepago/AestheticSigLIP")
    assert store.is_dirty is False

    # Recording final calibrated score marks dirty and persists
    store.record_final(
        asset_id="asset-1",
        score=82,
        reason="Local CLIP score: 7.50/10.0 (z-score: 1.00)",
        raw_score_stage1=7.5,
        model_id_stage1="somepago/AestheticSigLIP",
    )
    assert store.is_dirty is True
    assert store.flush() is True

    # Reload from disk and verify round-trip + backward-compatible helpers
    reloaded = EvaluationStore.from_disk(str(cache_file))
    assert len(reloaded) == 1
    assert reloaded.is_dirty is False
    assert reloaded.get_stage1(asset, model_id="somepago/AestheticSigLIP") == pytest.approx(7.5)
    raw_dict = load_cache(str(cache_file))
    assert raw_dict["asset-1"]["score"] == 82

    # Corrupted cache file degrades gracefully to empty store
    corrupted_file = tmp_path / "corrupted.json"
    corrupted_file.write_text("{broken json", encoding="utf-8")
    recovered = EvaluationStore.from_disk(str(corrupted_file))
    assert len(recovered) == 0
    assert recovered.is_dirty is False

    # save_cache helper writes atomically as well
    helper_file = tmp_path / "helper.json"
    save_cache({"asset-2": {"score": 90}}, str(helper_file))
    assert load_cache(str(helper_file))["asset-2"]["score"] == 90


def test_thread_safe_concurrent_mutations_and_interrupt_safety(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    cache_file = tmp_path / "concurrent_cache.json"
    store = EvaluationStore.from_disk(str(cache_file))

    total_assets = 100

    def worker(i: int) -> None:
        asset = {"id": f"asset-{i}", "fileModifiedAt": "2026-07-25T12:00:00Z"}
        store.record_stage1(asset, raw_score=float(i % 10), model_id="somepago/AestheticSigLIP")
        store.record_stage2(
            asset,
            raw_score=float(i),
            model_id="gemini-2.5-flash",
            reason=f"Reason {i}",
        )
        assert store.get_stage1(asset, model_id="somepago/AestheticSigLIP") == pytest.approx(
            float(i % 10)
        )
        assert store.get_stage2(asset, model_id="gemini-2.5-flash") == pytest.approx(float(i))

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(worker, i) for i in range(total_assets)]
        for f in futures:
            f.result()

    assert len(store) == total_assets
    assert store.flush() is True

    # Mutate again, then simulate a write interrupt during json.dump before os.replace
    store.record_stage1(
        {"id": "asset-extra", "fileModifiedAt": "2026-07-25T12:00:00Z"},
        raw_score=9.5,
        model_id="somepago/AestheticSigLIP",
    )
    assert store.is_dirty is True

    with patch("immich_aesthetic_scorer.store.json.dump", side_effect=OSError("Disk full")):
        assert store.flush() is False

    # Dirty flag remains True because flush failed, and existing disk file is uncorrupted
    assert store.is_dirty is True
    persisted = EvaluationStore.from_disk(str(cache_file))
    assert len(persisted) == total_assets
    assert "asset-extra" not in persisted.to_dict()
