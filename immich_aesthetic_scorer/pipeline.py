"""Pure computational scoring pipeline and statistical calibration module.

Provides statistical calibration, logistic sigmoid normalization, candidate selection,
continuous fallback score fusion, 1-5 star mapping, and sliding-window burst deduplication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class PopulationStats:
    """Statistical summary of a population of scores."""

    mean: float
    std: float


def calculate_population_stats(
    scores: Sequence[float],
    min_std: float = 1.0,
    default_mean: float = 6.0,
    default_std: float = 1.0,
) -> PopulationStats:
    """Calculates population mean and standard deviation with floor guard.

    Args:
        scores: Sequence of raw numerical scores.
        min_std: Minimum allowed standard deviation. If calculated std is below this,
            a floor is enforced (defaults to 1.0 for Stage 1, or 0.01 floor check).
        default_mean: Fallback mean when score count is <= 1.
        default_std: Fallback standard deviation when score count is <= 1.

    Returns:
        PopulationStats containing population mean and standard deviation.
    """
    valid_scores = [float(s) for s in scores if s is not None and not math.isnan(s)]
    if len(valid_scores) <= 1:
        return PopulationStats(mean=default_mean, std=default_std)

    mean = sum(valid_scores) / len(valid_scores)
    variance = sum((x - mean) ** 2 for x in valid_scores) / len(valid_scores)
    std = math.sqrt(variance)

    if std < min_std:
        std = min_std

    return PopulationStats(mean=mean, std=std)


def normalize_z_score(
    raw_score: float,
    mean: float,
    std: float,
    steepness: float = 1.5,
) -> float:
    """Normalizes a raw score to 0–100 using standardized z-score and logistic sigmoid.

    Formula: S_norm = 100.0 / (1.0 + exp(-steepness * z))
    where z = (raw_score - mean) / std.

    Args:
        raw_score: The raw aesthetic quality score.
        mean: Population mean.
        std: Population standard deviation.
        steepness: Sigmoid scaling factor (default: 1.5).

    Returns:
        Calibrated score bounded strictly within [0.0, 100.0].
    """
    if std <= 0.0 or math.isnan(std):
        std = 1.0

    z = (raw_score - mean) / std
    try:
        exponent = -steepness * z
        if exponent > 700:
            val = 0.0
        elif exponent < -700:
            val = 100.0
        else:
            val = 100.0 / (1.0 + math.exp(exponent))
    except OverflowError:
        val = 0.0 if (-steepness * z) > 0 else 100.0

    return min(100.0, max(0.0, val))


def _get_score(item: Any, score_key: str | Callable[[Any], float]) -> float:
    if callable(score_key):
        return float(score_key(item))
    if isinstance(item, dict):
        return float(item.get(score_key, 0.0))
    return float(getattr(item, score_key, 0.0))


def select_candidates(
    items: Sequence[T],
    top_pct: float = 20.0,
    score_key: str | Callable[[T], float] = "s1_norm",
) -> tuple[list[T], list[T]]:
    """Splits scored assets into top N percent candidates and non-candidates.

    Args:
        items: Sequence of items to split.
        top_pct: Percentage of items to select as candidates (e.g. 20.0 for top 20%).
        score_key: Key name or callable extractor to retrieve numerical score.

    Returns:
        Tuple of (candidates, non_candidates), sorted descending by score.
    """
    if not items:
        return [], []

    sorted_items = sorted(
        items,
        key=lambda x: _get_score(x, score_key),
        reverse=True,
    )

    top_n = max(1, int(len(sorted_items) * (top_pct / 100.0)))
    top_n = min(len(sorted_items), top_n)

    candidates = sorted_items[:top_n]
    non_candidates = sorted_items[top_n:]
    return candidates, non_candidates


def fuse_scores(
    s1_norm: float,
    s2_norm: float | None = None,
    stage2_weight: float = 0.5,
    fallback_s2: float = 50.0,
) -> float:
    """Fuses normalized Stage 1 and Stage 2 scores, applying fallback for non-candidates.

    Args:
        s1_norm: Normalized Stage 1 score (0–100).
        s2_norm: Normalized Stage 2 score (0–100), or None for non-candidates.
        stage2_weight: Weight given to Stage 2 (0.0 to 1.0, default: 0.5).
        fallback_s2: Fallback score for unevaluated Stage 2 (default: 50.0).

    Returns:
        Combined score bounded strictly within [0.0, 100.0].
    """
    effective_s2 = s2_norm if s2_norm is not None else fallback_s2
    combined = (1.0 - stage2_weight) * s1_norm + stage2_weight * effective_s2
    return min(100.0, max(0.0, combined))


def score_to_stars(score: float) -> int:
    """Maps a standardized 0–100 composite aesthetic score into a 1–5 star rating.

    Args:
        score: Composite quality score.

    Returns:
        Star rating value (1, 2, 3, 4, or 5).
    """
    if score >= 90:
        return 5
    elif score >= 75:
        return 4
    elif score >= 50:
        return 3
    elif score >= 20:
        return 2
    else:
        return 1


def parse_asset_time(asset_item: Any) -> datetime:
    """Parses chronological datetime from asset metadata.

    Supports Immich nested {"asset": {"fileCreatedAt": ...}} format,
    flat dicts with localDateTime/fileCreatedAt/createdAt, and direct datetime values.
    Returns timezone-naive datetime.min if parsing fails.
    """
    if isinstance(asset_item, datetime):
        return asset_item.replace(tzinfo=None) if asset_item.tzinfo is not None else asset_item

    if not isinstance(asset_item, dict):
        return datetime.min

    asset_info = asset_item.get("asset")
    if not isinstance(asset_info, dict):
        asset_info = asset_item

    time_val = (
        asset_info.get("localDateTime")
        or asset_info.get("fileCreatedAt")
        or asset_info.get("createdAt")
        or asset_item.get("localDateTime")
        or asset_item.get("fileCreatedAt")
        or asset_item.get("createdAt")
    )
    if not time_val:
        return datetime.min

    if isinstance(time_val, datetime):
        return time_val.replace(tzinfo=None) if time_val.tzinfo is not None else time_val

    try:
        cleaned_str = str(time_val).replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return datetime.min


def deduplicate_bursts(
    scored_assets: Sequence[T],
    dedup_window: float,
    time_getter: Callable[[T], datetime] | None = None,
    score_getter: Callable[[T], float] | None = None,
) -> list[T]:
    """Filters out burst photos captured within a specific time window.

    Sorts the assets chronologically, groups them into time-based clusters using a
    sliding/rolling window, and preserves only the highest scoring asset from each cluster.

    Args:
        scored_assets: Sequence of scored asset items.
        dedup_window: Deduplication window in seconds. If <= 0, returns original items.
        time_getter: Optional extractor function to retrieve datetime.
        score_getter: Optional extractor function to retrieve numeric score.

    Returns:
        Deduplicated list of assets.
    """
    if dedup_window <= 0 or not scored_assets:
        return list(scored_assets)

    get_time = time_getter or parse_asset_time
    get_score = score_getter or (lambda item: _get_score(item, "score"))

    chrono_assets: list[tuple[datetime, T]] = []
    for item in scored_assets:
        chrono_assets.append((get_time(item), item))
    chrono_assets.sort(key=lambda x: x[0])

    deduped_assets: list[T] = []
    current_group: list[tuple[datetime, T]] = []

    for dt, item in chrono_assets:
        if dt == datetime.min:
            # If timestamp parsing fails, keep it individually
            deduped_assets.append(item)
            continue

        if not current_group:
            current_group.append((dt, item))
        else:
            diff = (dt - current_group[-1][0]).total_seconds()
            if diff <= dedup_window:
                current_group.append((dt, item))
            else:
                best_item = max(current_group, key=lambda x: get_score(x[1]))[1]
                deduped_assets.append(best_item)
                current_group = [(dt, item)]

    if current_group:
        best_item = max(current_group, key=lambda x: get_score(x[1]))[1]
        deduped_assets.append(best_item)

    return deduped_assets


class ScorePipeline:
    """Coordinating engine for statistical calibration, candidate splitting, fusion, and burst dedup."""

    def __init__(
        self,
        min_std_s1: float = 1.0,
        min_std_s2: float = 0.01,
        stage2_weight: float = 0.5,
        dedup_window: float = 0.0,
        steepness: float = 1.5,
    ) -> None:
        self.min_std_s1 = min_std_s1
        self.min_std_s2 = min_std_s2
        self.stage2_weight = stage2_weight
        self.dedup_window = dedup_window
        self.steepness = steepness
        self.s1_stats: PopulationStats | None = None
        self.s2_stats: PopulationStats | None = None

    def calibrate_stage1(
        self,
        assets: list[dict[str, Any]],
        raw_key: str = "raw_score_stage1",
        norm_key: str = "s1_norm",
    ) -> PopulationStats:
        """Calibrates Stage 1 raw scores and attaches normalized s1_norm to each asset."""
        raw_scores = [item[raw_key] for item in assets if item.get(raw_key) is not None]
        self.s1_stats = calculate_population_stats(
            raw_scores,
            min_std=self.min_std_s1,
            default_mean=6.0,
            default_std=1.0,
        )
        for item in assets:
            raw = item.get(raw_key)
            if raw is not None:
                norm = normalize_z_score(
                    float(raw),
                    self.s1_stats.mean,
                    self.s1_stats.std,
                    steepness=self.steepness,
                )
                item[norm_key] = norm
            else:
                item[norm_key] = 50.0
        return self.s1_stats

    def single_stage_finalize(
        self,
        assets: list[dict[str, Any]],
        raw_key: str = "raw_score_stage1",
        norm_key: str = "s1_norm",
    ) -> list[dict[str, Any]]:
        """Finalizes single-stage scores and star ratings."""
        mean = self.s1_stats.mean if self.s1_stats else 6.0
        std = self.s1_stats.std if self.s1_stats else 1.0

        for item in assets:
            raw = item.get(raw_key)
            s1_norm = item.get(norm_key, 50.0)
            score = int(round(s1_norm))
            item["score"] = score
            item["rating"] = score_to_stars(score)
            if raw is not None:
                z = (float(raw) - mean) / std
                item["reason"] = f"Local CLIP score: {float(raw):.2f}/10.0 (z-score: {z:.2f})"
            elif "reason" not in item:
                item["reason"] = f"Aesthetic score: {s1_norm:.1f}"
        return assets

    def split_candidates(
        self,
        assets: list[dict[str, Any]],
        top_pct: float = 20.0,
        score_key: str = "s1_norm",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Splits assets into top N percent candidates and non-candidates."""
        return select_candidates(assets, top_pct=top_pct, score_key=score_key)

    def fuse_stage2(
        self,
        assets: list[dict[str, Any]],
        raw_s1_key: str = "raw_score_stage1",
        norm_s1_key: str = "s1_norm",
        raw_s2_key: str = "raw_score_stage2",
        norm_s2_key: str = "s2_norm",
    ) -> list[dict[str, Any]]:
        """Calibrates Stage 2 candidates and fuses scores with continuous fallback."""
        raw_s2_scores = [item[raw_s2_key] for item in assets if item.get(raw_s2_key) is not None]
        self.s2_stats = calculate_population_stats(
            raw_s2_scores,
            min_std=self.min_std_s2,
            default_mean=50.0,
            default_std=10.0,
        )

        mean_s1 = self.s1_stats.mean if self.s1_stats else 6.0
        std_s1 = self.s1_stats.std if self.s1_stats else 1.0

        for item in assets:
            s1_norm = item.get(norm_s1_key, 50.0)
            raw_s1 = item.get(raw_s1_key)

            if raw_s2_key in item and item[raw_s2_key] is not None:
                raw_s2 = float(item[raw_s2_key])
                s2_norm = normalize_z_score(
                    raw_s2,
                    self.s2_stats.mean,
                    self.s2_stats.std,
                    steepness=self.steepness,
                )
                item[norm_s2_key] = s2_norm

                combined = fuse_scores(
                    s1_norm,
                    s2_norm,
                    stage2_weight=self.stage2_weight,
                )
                z1 = (float(raw_s1) - mean_s1) / std_s1 if raw_s1 is not None else 0.0
                z2 = (raw_s2 - self.s2_stats.mean) / self.s2_stats.std
                item["reason"] = (
                    f"Two-stage: S1={s1_norm:.1f} (raw: {float(raw_s1 or 0.0):.2f}, z: {z1:.2f}), "
                    f"S2={s2_norm:.1f} (raw: {raw_s2:.2f}, z: {z2:.2f})"
                )
            else:
                combined = fuse_scores(
                    s1_norm,
                    s2_norm=None,
                    stage2_weight=self.stage2_weight,
                    fallback_s2=50.0,
                )
                item["reason"] = (
                    f"Stage 1 only: Aesthetics={s1_norm:.1f} (raw: {float(raw_s1 or 0.0):.2f}, assumed average S2)"
                )

            score = int(round(combined))
            item["score"] = score
            item["rating"] = score_to_stars(score)

        return assets

    def deduplicate(
        self,
        assets: list[dict[str, Any]],
        dedup_window: float | None = None,
    ) -> list[dict[str, Any]]:
        """Clusters assets chronologically and removes near-duplicates."""
        window = self.dedup_window if dedup_window is None else dedup_window
        return deduplicate_bursts(assets, dedup_window=window)
