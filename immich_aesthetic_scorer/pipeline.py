"""Score pipeline orchestration and statistical calibration module.

Coordinates Stage 1 and Stage 2 image evaluations, population z-score sigmoid
calibration, top-percentile candidate splitting, continuous fallback score fusion,
1-5 star mapping, evaluation store persistence, and sliding-window burst deduplication.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence, TypeVar

from tqdm import tqdm

from immich_aesthetic_scorer.evaluators import ImageEvaluator, resize_image_bytes
from immich_aesthetic_scorer.store import EvaluationStore, get_asset_mod_time

__all__ = [
    "PipelineResult",
    "PopulationStats",
    "ScorePipeline",
    "score_to_stars",
]

T = TypeVar("T")

_COLOR_RESET = "\033[0m"
_COLOR_BOLD = "\033[1m"
_COLOR_GREEN = "\033[32m"
_COLOR_BLUE = "\033[34m"
_COLOR_YELLOW = "\033[33m"
_COLOR_MAGENTA = "\033[35m"


@dataclass(frozen=True)
class PopulationStats:
    """Statistical summary of a population of scores."""

    mean: float
    std: float


@dataclass
class PipelineResult:
    """End-to-end output of a `ScorePipeline.run(...)` execution.

    Attributes:
        scored_assets: All evaluated and calibrated asset records (each populated with
            `score`, `rating`, and `reason`), used for star rating synchronization.
        deduplicated_assets: Asset records after burst deduplication, sorted descending
            by calibrated `score`, used for Top 10 reporting and highlights album selection.
        stage1_stats: `PopulationStats` computed during Stage 1 z-score sigmoid calibration.
        stage2_stats: `PopulationStats` computed during Stage 2 calibration, or `None` in
            single-stage mode.
    """

    scored_assets: list[dict[str, Any]]
    deduplicated_assets: list[dict[str, Any]]
    stage1_stats: PopulationStats
    stage2_stats: PopulationStats | None = None


def _calculate_population_stats(
    scores: Sequence[float],
    min_std: float = 1.0,
    default_mean: float = 6.0,
    default_std: float = 1.0,
) -> PopulationStats:
    """Calculates population mean and standard deviation with floor guard."""
    valid_scores = [float(s) for s in scores if s is not None and not math.isnan(s)]
    if len(valid_scores) <= 1:
        return PopulationStats(mean=default_mean, std=default_std)

    mean = sum(valid_scores) / len(valid_scores)
    variance = sum((x - mean) ** 2 for x in valid_scores) / len(valid_scores)
    std = math.sqrt(variance)

    if std < min_std:
        std = min_std

    return PopulationStats(mean=mean, std=std)


def _normalize_z_score(
    raw_score: float,
    mean: float,
    std: float,
    steepness: float = 1.5,
) -> float:
    """Normalizes a raw score to 0–100 using standardized z-score and logistic sigmoid."""
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


def _select_candidates(
    items: Sequence[T],
    top_pct: float = 20.0,
    score_key: str | Callable[[T], float] = "s1_norm",
) -> tuple[list[T], list[T]]:
    """Splits scored assets into top N percent candidates and non-candidates."""
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


def _fuse_scores(
    s1_norm: float,
    s2_norm: float | None = None,
    stage2_weight: float = 0.5,
    fallback_s2: float = 50.0,
) -> float:
    """Fuses normalized Stage 1 and Stage 2 scores, applying fallback for non-candidates."""
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


def _parse_asset_time(asset_item: Any) -> datetime:
    """Parses chronological datetime from asset metadata."""
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


def _deduplicate_bursts(
    scored_assets: Sequence[T],
    dedup_window: float,
) -> list[T]:
    """Filters out burst photos captured within a specific time window."""
    if dedup_window <= 0 or not scored_assets:
        return list(scored_assets)

    chrono_assets: list[tuple[datetime, T]] = []
    for item in scored_assets:
        chrono_assets.append((_parse_asset_time(item), item))
    chrono_assets.sort(key=lambda x: x[0])

    deduped_assets: list[T] = []
    current_group: list[tuple[datetime, T]] = []

    for dt, item in chrono_assets:
        if dt == datetime.min:
            deduped_assets.append(item)
            continue

        if not current_group:
            current_group.append((dt, item))
        else:
            diff = (dt - current_group[-1][0]).total_seconds()
            if diff <= dedup_window:
                current_group.append((dt, item))
            else:
                best_item = max(current_group, key=lambda x: _get_score(x[1], "score"))[1]
                deduped_assets.append(best_item)
                current_group = [(dt, item)]

    if current_group:
        best_item = max(current_group, key=lambda x: _get_score(x[1], "score"))[1]
        deduped_assets.append(best_item)

    return deduped_assets


def _classify_stage2_provider(model_id: str) -> str:
    lowered = model_id.lower()
    if "gemini" in lowered:
        return "gemini"
    if "openai" in lowered or "gpt" in lowered:
        return "openai"
    return "local"


def _run_parallel_stage(
    *,
    stage_num: int,
    items_to_score: Sequence[dict[str, Any]],
    concurrency: int,
    store: EvaluationStore,
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    on_success: Callable[[dict[str, Any], Any], None],
) -> None:
    failed_count = 0
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(worker_fn, item): item for item in items_to_score}
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Stage {stage_num} Scoring",
            )
            for future in pbar:
                res = future.result()
                if res["status"] == "success":
                    on_success(res, pbar)
                else:
                    failed_count += 1
                    pbar.write(
                        f"Warning: Failed to score asset {res['id']} in Stage {stage_num}: {res['error']}"
                    )
    finally:
        store.flush()

    if failed_count > 0:
        print(f"\nCompleted Stage {stage_num} with {failed_count} failures.")


class ScorePipeline:
    """Coordinating engine for evaluation, statistical calibration, fusion, persistence, and burst dedup."""

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

    def _evaluate_stage1_batch(
        self,
        assets: Sequence[dict[str, Any]],
        store: EvaluationStore,
        evaluator: ImageEvaluator,
        thumbnail_loader: Callable[[str], bytes],
        *,
        concurrency: int,
        force_score: bool,
        use_cache_only: bool,
        delay: float,
        scorer_type: str,
    ) -> list[dict[str, Any]]:
        stage1_results: list[dict[str, Any]] = []
        assets_to_score: list[dict[str, Any]] = []
        s1_model_name = evaluator.model_id

        for asset in assets:
            raw_s1 = store.get_stage1(asset, model_id=s1_model_name, force=force_score)
            if raw_s1 is not None:
                reason_s1 = store.get_stage1_reason(
                    asset, model_id=s1_model_name, force=force_score
                )
                entry_item: dict[str, Any] = {
                    "id": asset["id"],
                    "asset": asset,
                    "raw_score_stage1": float(raw_s1),
                }
                if reason_s1 is not None:
                    entry_item["reason_stage1"] = reason_s1
                stage1_results.append(entry_item)
            else:
                assets_to_score.append(asset)

        print(f"Stage 1 cached: {len(stage1_results)}")
        if use_cache_only:
            if assets_to_score:
                print(
                    "Running in --use-cache-only mode. Skipping Stage 1 scoring of uncached assets."
                )
            return stage1_results

        print(f"Stage 1 needing scoring: {len(assets_to_score)}")
        if not assets_to_score:
            return stage1_results

        # Remote vision APIs enforce strict RPM limits; serialize worker threads for non-local scorers
        concurrency_s1 = 1 if scorer_type not in ("local", "fake") else max(1, concurrency)
        print(
            f"Starting parallel Stage 1 scoring (using '{s1_model_name}') with {concurrency_s1} worker threads."
        )
        if scorer_type not in ("local", "fake"):
            print(
                f"Injecting a {delay}s delay between requests per thread to respect API rate limits."
            )

        def _score_one(asset_item: dict[str, Any]) -> dict[str, Any]:
            asset_id = str(asset_item["id"])
            try:
                raw_bytes = thumbnail_loader(asset_id)
                img_bytes = resize_image_bytes(raw_bytes, max_dim=evaluator.required_max_dimension)
                eval_res = evaluator.evaluate(img_bytes)
                store.record_stage1(
                    asset_item,
                    eval_res.raw_score,
                    s1_model_name,
                    reason=eval_res.reason,
                )
                if scorer_type == "local" and delay > 0:
                    time.sleep(delay)
                return {
                    "id": asset_id,
                    "raw_score_stage1": eval_res.raw_score,
                    "reason_stage1": eval_res.reason,
                    "asset": asset_item,
                    "status": "success",
                }
            except Exception as exc:
                return {"id": asset_id, "status": "error", "error": str(exc), "asset": asset_item}

        def _on_s1_success(item_res: dict[str, Any], pbar: Any) -> None:
            stage1_results.append(item_res)
            filename = item_res["asset"].get("originalFileName") or item_res["id"]
            pbar.write(
                f"Stage 1 Scored {_COLOR_BLUE}{filename}{_COLOR_RESET} using "
                f"'{_COLOR_BOLD}{s1_model_name}{_COLOR_RESET}': "
                f"Raw Score {_COLOR_GREEN}{item_res['raw_score_stage1']:.2f}{_COLOR_RESET}"
            )

        _run_parallel_stage(
            stage_num=1,
            items_to_score=assets_to_score,
            concurrency=concurrency_s1,
            store=store,
            worker_fn=_score_one,
            on_success=_on_s1_success,
        )
        return stage1_results

    def _evaluate_stage2_batch(
        self,
        candidates: Sequence[dict[str, Any]],
        store: EvaluationStore,
        evaluator: ImageEvaluator,
        thumbnail_loader: Callable[[str], bytes],
        *,
        stage2_top_pct: float,
        concurrency: int,
        force_score: bool,
        use_cache_only: bool,
    ) -> None:
        s2_model_name = evaluator.model_id
        is_local_s2 = _classify_stage2_provider(s2_model_name) == "local"
        stage2_type_str = "Technical Quality Evaluation" if is_local_s2 else "Aesthetics Evaluation"
        print(
            f"\n--- {_COLOR_MAGENTA}Stage 2: {stage2_type_str}{_COLOR_RESET} "
            f"(selecting top {len(candidates)} candidates ({stage2_top_pct}%) using model '{_COLOR_BOLD}{s2_model_name}{_COLOR_RESET}') ---"
        )

        cached_count = 0
        assets_to_score: list[dict[str, Any]] = []
        for item in candidates:
            asset_obj = item.get("asset") or item["id"]
            raw_s2 = store.get_stage2(asset_obj, model_id=s2_model_name, force=force_score)
            if raw_s2 is not None:
                item["raw_score_stage2"] = float(raw_s2)
                reason_s2 = store.get_stage2_reason(
                    asset_obj, model_id=s2_model_name, force=force_score
                )
                if reason_s2 is not None:
                    item["reason_stage2"] = reason_s2
                cached_count += 1
            else:
                assets_to_score.append(item)

        print(f"Stage 2 cached: {cached_count}")
        if use_cache_only:
            if assets_to_score:
                print(
                    "Running in --use-cache-only mode. Skipping Stage 2 scoring of uncached assets."
                )
            return

        print(f"Stage 2 needing scoring: {len(assets_to_score)}")
        if not assets_to_score:
            return

        concurrency_s2 = 1 if not is_local_s2 else max(1, concurrency)
        print(
            f"Starting parallel Stage 2 scoring (using model '{s2_model_name}') with {concurrency_s2} worker threads."
        )

        def _score_one_s2(cand_item: dict[str, Any]) -> dict[str, Any]:
            asset_id = str(cand_item["id"])
            try:
                raw_bytes = thumbnail_loader(asset_id)
                img_bytes = resize_image_bytes(raw_bytes, max_dim=evaluator.required_max_dimension)
                eval_res = evaluator.evaluate(img_bytes)
                store.record_stage2(
                    cand_item.get("asset") or asset_id,
                    eval_res.raw_score,
                    s2_model_name,
                    reason=eval_res.reason,
                )
                return {
                    "id": asset_id,
                    "item": cand_item,
                    "raw_score_stage2": eval_res.raw_score,
                    "reason_stage2": eval_res.reason,
                    "status": "success",
                }
            except Exception as exc:
                return {"id": asset_id, "item": cand_item, "status": "error", "error": str(exc)}

        def _on_s2_success(res: dict[str, Any], pbar: Any) -> None:
            cand = res["item"]
            cand["raw_score_stage2"] = res["raw_score_stage2"]
            cand["reason_stage2"] = res.get("reason_stage2")
            filename = cand["asset"].get("originalFileName") or cand["id"]
            pbar.write(
                f"Stage 2 Scored {_COLOR_BLUE}{filename}{_COLOR_RESET} using "
                f"'{_COLOR_BOLD}{s2_model_name}{_COLOR_RESET}': "
                f"Raw Score {_COLOR_GREEN}{res['raw_score_stage2']:.2f}{_COLOR_RESET} | "
                f"Stage 1 Raw: {_COLOR_YELLOW}{cand['raw_score_stage1']:.2f}{_COLOR_RESET}"
            )

        _run_parallel_stage(
            stage_num=2,
            items_to_score=assets_to_score,
            concurrency=concurrency_s2,
            store=store,
            worker_fn=_score_one_s2,
            on_success=_on_s2_success,
        )

    def _calibrate_stage1(self, assets: list[dict[str, Any]]) -> PopulationStats:
        """Calibrates Stage 1 raw scores and attaches normalized s1_norm to each asset."""
        raw_scores = [
            item["raw_score_stage1"] for item in assets if item.get("raw_score_stage1") is not None
        ]
        self.s1_stats = _calculate_population_stats(
            raw_scores,
            min_std=self.min_std_s1,
            default_mean=6.0,
            default_std=1.0,
        )
        for item in assets:
            raw = item.get("raw_score_stage1")
            if raw is not None:
                item["s1_norm"] = _normalize_z_score(
                    float(raw),
                    self.s1_stats.mean,
                    self.s1_stats.std,
                    steepness=self.steepness,
                )
            else:
                item["s1_norm"] = 50.0
        return self.s1_stats

    def _single_stage_finalize(self, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Finalizes single-stage scores and star ratings."""
        mean = self.s1_stats.mean if self.s1_stats else 6.0
        std = self.s1_stats.std if self.s1_stats else 1.0

        for item in assets:
            raw = item.get("raw_score_stage1")
            s1_norm = item.get("s1_norm", 50.0)
            score = int(round(s1_norm))
            item["score"] = score
            item["rating"] = score_to_stars(score)
            if raw is not None:
                z = (float(raw) - mean) / std
                reason_s1 = item.get("reason_stage1")
                if reason_s1:
                    item["reason"] = (
                        f"{reason_s1} (Stage 1 raw: {float(raw):.2f}/10.0, z-score: {z:.2f})"
                    )
                else:
                    item["reason"] = f"Local CLIP score: {float(raw):.2f}/10.0 (z-score: {z:.2f})"
            elif "reason" not in item:
                item["reason"] = f"Aesthetic score: {s1_norm:.1f}"
        return assets

    def _fuse_stage2(self, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Calibrates Stage 2 candidates and fuses scores with continuous fallback."""
        raw_s2_scores = [
            item["raw_score_stage2"] for item in assets if item.get("raw_score_stage2") is not None
        ]
        self.s2_stats = _calculate_population_stats(
            raw_s2_scores,
            min_std=self.min_std_s2,
            default_mean=50.0,
            default_std=10.0,
        )

        mean_s1 = self.s1_stats.mean if self.s1_stats else 6.0
        std_s1 = self.s1_stats.std if self.s1_stats else 1.0

        for item in assets:
            s1_norm = item.get("s1_norm", 50.0)
            raw_s1 = item.get("raw_score_stage1")

            if item.get("raw_score_stage2") is not None:
                raw_s2 = float(item["raw_score_stage2"])
                s2_norm = _normalize_z_score(
                    raw_s2,
                    self.s2_stats.mean,
                    self.s2_stats.std,
                    steepness=self.steepness,
                )
                item["s2_norm"] = s2_norm

                combined = _fuse_scores(
                    s1_norm,
                    s2_norm,
                    stage2_weight=self.stage2_weight,
                )
                z1 = (float(raw_s1) - mean_s1) / std_s1 if raw_s1 is not None else 0.0
                z2 = (raw_s2 - self.s2_stats.mean) / self.s2_stats.std
                base_reason = (
                    f"Two-stage: S1={s1_norm:.1f} (raw: {float(raw_s1 or 0.0):.2f}, z: {z1:.2f}), "
                    f"S2={s2_norm:.1f} (raw: {raw_s2:.2f}, z: {z2:.2f})"
                )
                reason_s2 = item.get("reason_stage2")
                if reason_s2:
                    item["reason"] = f"{base_reason} — {reason_s2}"
                else:
                    item["reason"] = base_reason
            else:
                combined = _fuse_scores(
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

    @staticmethod
    def _persist_final_scores(
        scored_assets: Sequence[dict[str, Any]],
        store: EvaluationStore,
        model_id_stage1: str,
        model_id_stage2: str | None = None,
    ) -> None:
        for item in scored_assets:
            raw_s2_val = item.get("raw_score_stage2") if model_id_stage2 is not None else None
            store.record_final(
                asset_id=str(item["id"]),
                score=int(item["score"]),
                reason=str(item["reason"]),
                raw_score_stage1=float(item["raw_score_stage1"]),
                model_id_stage1=model_id_stage1,
                raw_score_stage2=float(raw_s2_val) if raw_s2_val is not None else None,
                model_id_stage2=model_id_stage2 if raw_s2_val is not None else None,
                reason_stage2=item.get("reason_stage2") if raw_s2_val is not None else None,
                updated_at=get_asset_mod_time(item["asset"]),
                reason_stage1=item.get("reason_stage1"),
            )
        store.flush()

    def run(
        self,
        assets: Sequence[dict[str, Any]],
        store: EvaluationStore,
        stage1_evaluator: ImageEvaluator,
        thumbnail_loader: Callable[[str], bytes],
        *,
        stage2_evaluator: ImageEvaluator | None = None,
        stage2_top_pct: float = 15.0,
        concurrency: int = 5,
        force_score: bool = False,
        use_cache_only: bool = False,
        delay: float = 0.0,
        scorer_type: str = "local",
    ) -> PipelineResult:
        """Orchestrates end-to-end evaluation, calibration, fusion, store persistence, and burst deduplication.

        Args:
            assets: Sequence of target asset dictionaries to evaluate.
            store: `EvaluationStore` instance for intermediate and final score persistence.
            stage1_evaluator: `ImageEvaluator` adapter for Stage 1 evaluation.
            thumbnail_loader: Callback `(asset_id: str) -> bytes` fetching raw thumbnail bytes.
            stage2_evaluator: Optional `ImageEvaluator` adapter for Stage 2 evaluation.
            stage2_top_pct: Top percentage of Stage 1 candidates promoted to Stage 2.
            concurrency: Worker thread pool size for local batch evaluation.
            force_score: When `True`, bypasses cached Stage 1 and Stage 2 entries.
            use_cache_only: When `True`, skips evaluating uncached assets.
            delay: Additional per-request delay in seconds.
            scorer_type: Configured Stage 1 scorer type (`"local"`, `"gemini"`, `"openai"`, or `"fake"`).

        Returns:
            `PipelineResult` containing all calibrated `scored_assets`, `deduplicated_assets`
            sorted descending by score, `stage1_stats`, and `stage2_stats`.
        """
        stage1_results = self._evaluate_stage1_batch(
            assets,
            store,
            stage1_evaluator,
            thumbnail_loader,
            concurrency=concurrency,
            force_score=force_score,
            use_cache_only=use_cache_only,
            delay=delay,
            scorer_type=scorer_type,
        )
        s1_stats = self._calibrate_stage1(stage1_results)

        if stage2_evaluator is None:
            print("\nCalibrating aesthetic scores based on library distribution...")
            print("Library Statistics:")
            print(f"  - Total Scored Photos: {len(stage1_results)}")
            print(f"  - Raw Score Mean: {s1_stats.mean:.3f}")
            print(f"  - Raw Score Std Dev: {s1_stats.std:.3f}")
            print("Applying sigmoid normalization and updating cache...")

            scored_assets = self._single_stage_finalize(stage1_results)
            self._persist_final_scores(
                scored_assets,
                store,
                model_id_stage1=stage1_evaluator.model_id,
            )
            print("Cache updated successfully!")
            s2_stats = None
        else:
            candidates_for_stage2, _ = _select_candidates(
                stage1_results,
                top_pct=stage2_top_pct,
                score_key="s1_norm",
            )
            self._evaluate_stage2_batch(
                candidates_for_stage2,
                store,
                stage2_evaluator,
                thumbnail_loader,
                stage2_top_pct=stage2_top_pct,
                concurrency=concurrency,
                force_score=force_score,
                use_cache_only=use_cache_only,
            )

            print(
                f"\nCombining Stage 1 ({stage1_evaluator.model_id}) and "
                f"Stage 2 ({stage2_evaluator.model_id}) scores..."
            )
            scored_assets = self._fuse_stage2(stage1_results)
            s2_stats = self.s2_stats
            if s2_stats is not None:
                print("Stage 2 Statistics:")
                print(f"  - Total Scored Candidates: {len(scored_assets)}")
                print(f"  - Raw Score Mean: {s2_stats.mean:.3f}")
                print(f"  - Raw Score Std Dev: {s2_stats.std:.3f}")

            self._persist_final_scores(
                scored_assets,
                store,
                model_id_stage1=stage1_evaluator.model_id,
                model_id_stage2=stage2_evaluator.model_id,
            )
            print("Cache updated successfully with combined scores!")

        if self.dedup_window > 0:
            print(
                f"\nDeduplicating burst photos within a {int(self.dedup_window)}-second window..."
            )
            original_count = len(scored_assets)
            deduplicated_assets = _deduplicate_bursts(
                scored_assets, dedup_window=float(self.dedup_window)
            )
            print(
                f"Deduplicated burst photos: Kept {len(deduplicated_assets)} unique highlights out of "
                f"{original_count} scored photos (Filtered out {original_count - len(deduplicated_assets)} burst duplicates)."
            )
        else:
            deduplicated_assets = list(scored_assets)

        deduplicated_assets.sort(key=lambda x: x["score"], reverse=True)

        return PipelineResult(
            scored_assets=scored_assets,
            deduplicated_assets=deduplicated_assets,
            stage1_stats=s1_stats,
            stage2_stats=s2_stats,
        )
