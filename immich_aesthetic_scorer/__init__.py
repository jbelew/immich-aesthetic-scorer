"""Immich Aesthetic Scorer package."""

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

__all__ = [
    "PopulationStats",
    "ScorePipeline",
    "calculate_population_stats",
    "deduplicate_bursts",
    "fuse_scores",
    "normalize_z_score",
    "parse_asset_time",
    "score_to_stars",
    "select_candidates",
]
