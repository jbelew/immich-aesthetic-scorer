"""Thread-safe evaluation cache store module with in-memory and atomic disk backends."""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

_LEGACY_CLIP_SCORE_RE = re.compile(r"Local CLIP score:\s*([-+]?\d*\.?\d+)\s*/\s*10\.0")
cache_lock = threading.RLock()


def extract_raw_score_from_reason(reason: str | None) -> float | None:
    """Parses a local aesthetic CLIP score out of a cached reason explanation string.

    Enables reading legacy aesthetic scores saved in older cache versions
    without repeating local model evaluation.
    """
    if not reason or not isinstance(reason, str):
        return None
    match = _LEGACY_CLIP_SCORE_RE.search(reason)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (ValueError, TypeError):
        return None


def get_asset_mod_time(asset: dict[str, Any] | Any) -> str | None:
    """Resolves a sensible modification/creation timestamp for cache invalidation.

    Falls back to fileCreatedAt or createdAt if fileModifiedAt is missing or
    defaulted to the MS-DOS epoch (1980-01-01).
    """
    if not isinstance(asset, dict):
        return None
    nested = asset.get("asset")
    target = nested if isinstance(nested, dict) else asset
    mod_time = target.get("fileModifiedAt")
    if not mod_time or "1980-01-01" in str(mod_time):
        mod_time = target.get("fileCreatedAt") or target.get("createdAt")
    return str(mod_time) if mod_time is not None else None


def load_cache(cache_path: str) -> dict[str, dict[str, Any]]:
    """Loads the local JSON cache file from disk.

    Returns an empty dict if the file does not exist or is corrupted.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}. Starting fresh.")
    return {}


def save_cache(cache: dict[str, Any], cache_path: str) -> bool:
    """Saves cache entries atomically and thread-safely to disk via a temporary file.

    Returns True if the atomic write succeeded, False otherwise.
    """
    with cache_lock:
        tmp_path = f"{cache_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            os.replace(tmp_path, cache_path)
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False


class EvaluationStore:
    """Encapsulated store for Stage 1, Stage 2, and final aesthetic evaluation records."""

    def __init__(
        self,
        cache_path: str | None = None,
        initial_data: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._cache_path = cache_path
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, Any]] = (
            {k: dict(v) for k, v in initial_data.items() if isinstance(v, dict)}
            if initial_data
            else {}
        )
        self._dirty = False

    @classmethod
    def in_memory(cls, initial_data: dict[str, dict[str, Any]] | None = None) -> EvaluationStore:
        """Creates an in-memory EvaluationStore instance for fast zero-disk usage."""
        return cls(cache_path=None, initial_data=initial_data)

    @classmethod
    def from_disk(cls, cache_path: str) -> EvaluationStore:
        """Creates an EvaluationStore backed by a JSON file on disk."""
        data = load_cache(cache_path)
        return cls(cache_path=cache_path, initial_data=data)

    @property
    def is_dirty(self) -> bool:
        """Returns True if any cache entries have been modified since initialization or last flush."""
        with self._lock:
            return self._dirty

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Returns a deep snapshot copy of all stored cache entries."""
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}

    def get_entry(self, asset_id: str) -> dict[str, Any] | None:
        """Returns a copy of the raw entry for asset_id, or None if not present."""
        with self._lock:
            entry = self._data.get(asset_id)
            return dict(entry) if entry is not None else None

    @staticmethod
    def _resolve_asset_id(asset: dict[str, Any] | str) -> str:
        if isinstance(asset, str):
            return asset
        if "id" in asset:
            return str(asset["id"])
        nested = asset.get("asset")
        if isinstance(nested, dict) and "id" in nested:
            return str(nested["id"])
        raise KeyError("Asset dict must contain an 'id' key")

    @staticmethod
    def _is_timestamp_drifted(asset: dict[str, Any] | str, entry: dict[str, Any]) -> bool:
        if isinstance(asset, str):
            return False
        mod_time = get_asset_mod_time(asset)
        cached_mod = entry.get("updatedAt")
        if cached_mod and mod_time and cached_mod != mod_time:
            return True
        return False

    def _get_validated_entry(
        self,
        asset: dict[str, Any] | str,
        model_key: str,
        model_id: str,
        force: bool,
    ) -> dict[str, Any] | None:
        if force:
            return None
        asset_id = self._resolve_asset_id(asset)
        entry = self._data.get(asset_id)
        if not entry:
            return None
        if self._is_timestamp_drifted(asset, entry):
            return None
        cached_model = entry.get(model_key)
        if cached_model is not None and cached_model != model_id:
            return None
        return entry

    def get_stage1(
        self,
        asset: dict[str, Any] | str,
        model_id: str,
        force: bool = False,
    ) -> float | None:
        """Retrieves a cached Stage 1 raw score if valid, including legacy cache fallbacks."""
        with self._lock:
            entry = self._get_validated_entry(asset, "model_id_stage1", model_id, force)
            if not entry:
                return None
            raw = entry.get("raw_score_stage1")
            if raw is None:
                raw = entry.get("raw_score")
            if raw is None:
                raw = extract_raw_score_from_reason(entry.get("reason"))
            return float(raw) if raw is not None else None

    def _get_stage_reason(
        self,
        asset: dict[str, Any] | str,
        *,
        model_key: str,
        score_key: str,
        reason_key: str,
        model_id: str,
        force: bool,
    ) -> str | None:
        with self._lock:
            entry = self._get_validated_entry(asset, model_key, model_id, force)
            if not entry or entry.get(score_key) is None:
                return None
            reason = entry.get(reason_key)
            return str(reason) if reason is not None else None

    def get_stage1_reason(
        self,
        asset: dict[str, Any] | str,
        model_id: str,
        force: bool = False,
    ) -> str | None:
        """Retrieves the cached Stage 1 reason if the Stage 1 entry is valid."""
        return self._get_stage_reason(
            asset,
            model_key="model_id_stage1",
            score_key="raw_score_stage1",
            reason_key="reason_stage1",
            model_id=model_id,
            force=force,
        )

    def record_stage1(
        self,
        asset: dict[str, Any] | str,
        raw_score: float,
        model_id: str,
        updated_at: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Records a Stage 1 evaluation score and optional reason."""
        asset_id = self._resolve_asset_id(asset)
        mod_time = updated_at if updated_at is not None else get_asset_mod_time(asset)
        with self._lock:
            existing = self._data.get(asset_id)
            if (
                existing
                and existing.get("updatedAt")
                and mod_time
                and existing.get("updatedAt") != mod_time
            ):
                # Asset was modified since previous evaluation; clear stale Stage 2 data
                entry: dict[str, Any] = {}
            else:
                entry = dict(existing) if existing else {}

            entry["raw_score_stage1"] = float(raw_score)
            entry["model_id_stage1"] = model_id
            entry["reason_stage1"] = reason
            if mod_time is not None or "updatedAt" not in entry:
                entry["updatedAt"] = mod_time

            if entry != existing:
                self._data[asset_id] = entry
                self._dirty = True

    def get_stage2(
        self,
        asset: dict[str, Any] | str,
        model_id: str,
        force: bool = False,
    ) -> float | None:
        """Retrieves a cached Stage 2 raw score if valid."""
        with self._lock:
            entry = self._get_validated_entry(asset, "model_id_stage2", model_id, force)
            if not entry:
                return None
            raw = entry.get("raw_score_stage2")
            return float(raw) if raw is not None else None

    def get_stage2_reason(
        self,
        asset: dict[str, Any] | str,
        model_id: str,
        force: bool = False,
    ) -> str | None:
        """Retrieves the cached Stage 2 reason if Stage 2 entry is valid."""
        return self._get_stage_reason(
            asset,
            model_key="model_id_stage2",
            score_key="raw_score_stage2",
            reason_key="reason_stage2",
            model_id=model_id,
            force=force,
        )

    def record_stage2(
        self,
        asset: dict[str, Any] | str,
        raw_score: float,
        model_id: str,
        reason: str | None = None,
    ) -> None:
        """Records a Stage 2 evaluation score and optional reason."""
        asset_id = self._resolve_asset_id(asset)
        with self._lock:
            existing = self._data.get(asset_id)
            entry = dict(existing) if existing else {}
            entry["raw_score_stage2"] = float(raw_score)
            entry["model_id_stage2"] = model_id
            entry["reason_stage2"] = reason
            if entry != existing:
                self._data[asset_id] = entry
                self._dirty = True

    def record_final(
        self,
        asset_id: str,
        score: int,
        reason: str,
        raw_score_stage1: float,
        model_id_stage1: str,
        raw_score_stage2: float | None = None,
        model_id_stage2: str | None = None,
        reason_stage2: str | None = None,
        updated_at: str | None = None,
        reason_stage1: str | None = None,
    ) -> None:
        """Records final calibrated score and metadata for an asset."""
        with self._lock:
            existing = self._data.get(asset_id)
            entry = dict(existing) if existing else {}
            prev_model_s1 = entry.get("model_id_stage1")
            prev_updated_at = entry.get("updatedAt")
            s1_invalidated = (prev_model_s1 is not None and prev_model_s1 != model_id_stage1) or (
                prev_updated_at is not None
                and updated_at is not None
                and prev_updated_at != updated_at
            )

            entry["score"] = int(score)
            entry["raw_score_stage1"] = float(raw_score_stage1)
            entry["model_id_stage1"] = model_id_stage1
            entry["reason"] = reason
            if reason_stage1 is not None:
                entry["reason_stage1"] = reason_stage1
            elif s1_invalidated:
                entry["reason_stage1"] = None
            elif "reason_stage1" in entry:
                entry["reason_stage1"] = entry.get("reason_stage1")
            if updated_at is not None:
                entry["updatedAt"] = updated_at
            if raw_score_stage2 is not None or "raw_score_stage2" in entry:
                entry["raw_score_stage2"] = (
                    float(raw_score_stage2) if raw_score_stage2 is not None else None
                )
                entry["model_id_stage2"] = model_id_stage2 if raw_score_stage2 is not None else None
                entry["reason_stage2"] = (
                    reason_stage2
                    if reason_stage2 is not None
                    else (entry.get("reason_stage2") if raw_score_stage2 is not None else None)
                )
            if entry != existing:
                self._data[asset_id] = entry
                self._dirty = True

    def flush(self) -> bool:
        """Atomically writes cache entries to disk if dirty.

        Returns True if a disk write was performed, False if skipped or failed.
        """
        with self._lock:
            if not self._dirty or not self._cache_path:
                return False
            snapshot = {k: dict(v) for k, v in self._data.items()}
            if save_cache(snapshot, self._cache_path):
                self._dirty = False
                return True
            return False
