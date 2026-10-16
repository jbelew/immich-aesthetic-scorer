#!/usr/bin/env python3
"""Immich Aesthetic Scorer & Highlight Album Creator.

CLI driver coordinating `ScorePipeline`, `EvaluationStore`, `ImageEvaluator`,
and `ImmichGateway`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from getpass import getpass
from typing import Any, Mapping, Optional, Sequence

import requests
from tqdm import tqdm

from immich_aesthetic_scorer.evaluators import (
    ImageEvaluator,
    RateLimiter,
    create_evaluator,
)
from immich_aesthetic_scorer.gateway import (
    ImmichGateway,
    ImmichGatewayError,
)
from immich_aesthetic_scorer.pipeline import (
    ScorePipeline,
    score_to_stars,
)
from immich_aesthetic_scorer.store import (
    EvaluationStore,
)

# Suppress Hugging Face hub telemetry/symlinks warnings and other log noises
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Silence optional libraries and PyTorch CUDA warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Default CLI configuration constants
DEFAULT_GEMINI_KEY = None
DEFAULT_LIMIT = 100
DEFAULT_CONCURRENCY = 5
DEFAULT_DELAY = 4.0  # 4.0s delay = max 15 RPM for free tier remote vision endpoints
DEFAULT_CACHE_FILE = ".immich_aesthetic_cache.json"

# ANSI escape codes for terminal coloring
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"
COLOR_GRAY = "\033[90m"


def _classify_stage2_provider(model_id: str) -> str:
    lowered = model_id.lower()
    if "gemini" in lowered:
        return "gemini"
    if "openai" in lowered or "gpt" in lowered:
        return "openai"
    return "local"


def get_or_prompt(
    env_name: str,
    prompt_text: str,
    secret: bool = False,
    default: Optional[str] = None,
) -> str:
    """Retrieves a configuration value from environment variables or prompts the user interactively."""
    val = os.environ.get(env_name)
    if val:
        return val

    display_prompt = f"{prompt_text} [{default}]: " if default else f"{prompt_text}: "
    raw_input = getpass(display_prompt) if secret else input(display_prompt)
    cleaned = raw_input.strip()
    if not cleaned and default:
        return default
    return cleaned


def select_person_interactive(immich_url: str, api_key: str) -> tuple[Optional[str], Optional[str]]:
    """Guides the user through an interactive terminal selection for an Immich person."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    print("\n--- Person Selection ---")
    while True:
        query = input("Search for a person by name (or press Enter to list all): ").strip()
        try:
            results = gateway.search_people(query) if query else gateway.list_people(size=100)
        except ImmichGatewayError as exc:
            print(f"Error fetching people: {exc}")
            results = []

        valid_people = [p for p in results if p.get("name")]
        if not valid_people:
            print("No matching named people found. Let's try again.")
            continue

        print("\nMatching People found:")
        for idx, person in enumerate(valid_people, 1):
            print(f" [{idx}] {person.get('name')} (ID: {person.get('id')})")
        print(f" [{len(valid_people) + 1}] Search again")
        print(f" [{len(valid_people) + 2}] Enter Person UUID manually")

        choice_str = input(f"Select option (1-{len(valid_people) + 2}): ").strip()
        try:
            choice = int(choice_str)
            if 1 <= choice <= len(valid_people):
                selected = valid_people[choice - 1]
                return str(selected["id"]), selected.get("name")
            if choice == len(valid_people) + 1:
                continue
            if choice == len(valid_people) + 2:
                manual_uuid = input("Enter Person UUID: ").strip()
                if manual_uuid:
                    return manual_uuid, None
        except ValueError:
            print("Invalid input. Please enter a number.")


def select_album_interactive(immich_url: str, api_key: str) -> tuple[Optional[str], Optional[str]]:
    """Guides the user through an interactive terminal selection for an Immich album."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    print("\n--- Album Selection ---")
    try:
        albums = gateway.list_albums()
    except ImmichGatewayError as exc:
        print(f"Error fetching albums: {exc}")
        return None, None

    if not albums:
        print("No albums found on the server.")
        return None, None

    while True:
        query = input("Search for an album by name (or press Enter to list all): ").strip().lower()
        filtered = [a for a in albums if query in str(a.get("albumName", "")).lower()]
        if not filtered:
            print("No matching albums found. Let's try again.")
            continue

        print("\nMatching Albums found:")
        for idx, album in enumerate(filtered, 1):
            print(f" [{idx}] {album.get('albumName')} (ID: {album.get('id')})")
        print(f" [{len(filtered) + 1}] Search again")
        print(f" [{len(filtered) + 2}] Enter Album UUID manually")

        choice_str = input(f"Select option (1-{len(filtered) + 2}): ").strip()
        try:
            choice = int(choice_str)
            if 1 <= choice <= len(filtered):
                selected = filtered[choice - 1]
                return str(selected["id"]), selected.get("albumName")
            if choice == len(filtered) + 1:
                continue
            if choice == len(filtered) + 2:
                manual_uuid = input("Enter Album UUID: ").strip()
                if manual_uuid:
                    return manual_uuid, None
        except ValueError:
            print("Invalid input. Please enter a number.")


@dataclass
class ScorerConfig:
    """Resolved runtime configuration for the aesthetic scoring workflow."""

    config_path: str = "config.json"
    immich_url: Optional[str] = None
    api_key: Optional[str] = None
    person_id: Optional[str] = None
    album_id: Optional[str] = None
    gemini_key: Optional[str] = None
    target_album_name: Optional[str] = None
    scorer_type: str = "gemini"
    local_model_id: str = "somepago/AestheticSigLIP"
    gemini_model: str = "gemini-2.5-flash"
    openai_key: Optional[str] = None
    openai_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    write_ratings: bool = False
    limit: int = DEFAULT_LIMIT
    concurrency: int = DEFAULT_CONCURRENCY
    delay: float = DEFAULT_DELAY
    cache_file: str = DEFAULT_CACHE_FILE
    dedup_window: int = 0
    dry_run: bool = False
    force_score: bool = False
    use_cache_only: bool = False
    two_stage: bool = False
    stage2_top_pct: float = 15.0
    stage2_model: str = "gemini-2.5-flash"
    stage2_weight: float = 0.5


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser with all supported flags and defaults."""
    parser = argparse.ArgumentParser(description="Immich Aesthetic Scorer & Album Compiler")
    parser.add_argument(
        "--config", default="config.json", help="Path to config JSON file (default: config.json)"
    )
    parser.add_argument("--immich-url", help="Immich base URL (e.g. http://192.168.1.5:2283)")
    parser.add_argument("--api-key", help="Immich API key")
    parser.add_argument("--person-id", help="Immich Person UUID")
    parser.add_argument("--album-id", help="Immich Album UUID to source photos from")
    parser.add_argument("--gemini-key", help="Gemini API Key")
    parser.add_argument(
        "--target-album-name",
        help="Target highlights album name (default: Best of <Person Name>)",
    )
    parser.add_argument(
        "--limit", type=int, help=f"Number of best images to select (default: {DEFAULT_LIMIT})"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help=f"Number of concurrent workers (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        help=f"Additional delay in seconds between calls per thread (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--cache-file", help=f"Local cache file path (default: {DEFAULT_CACHE_FILE})"
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Download and score assets but do not make changes to albums or ratings in Immich",
    )
    parser.add_argument(
        "--force-score",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force re-scoring of already cached assets",
    )
    parser.add_argument(
        "--scorer-type",
        choices=["gemini", "local", "openai"],
        help="Scoring method: gemini, local, or openai (default: gemini)",
    )
    parser.add_argument(
        "--local-model",
        help="Hugging Face model ID for local scoring (default: somepago/AestheticSigLIP)",
    )
    parser.add_argument("--gemini-model", help="Gemini model ID to use (default: gemini-2.5-flash)")
    parser.add_argument("--openai-key", help="OpenAI API Key (or for OpenAI-compatible providers)")
    parser.add_argument(
        "--openai-url",
        help="Base URL for OpenAI-compatible provider (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--openai-model",
        help="Model ID for OpenAI-compatible provider (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--write-ratings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Update the star rating (1-5) of processed assets in Immich",
    )
    parser.add_argument(
        "--dedup-window",
        type=int,
        help="Deduplicate burst photos within this time window in seconds (default: 0, disabled)",
    )
    parser.add_argument(
        "--use-cache-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Only compile the album using already-cached images, skipping scoring of uncached ones",
    )
    parser.add_argument(
        "--two-stage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable two-stage scoring using a fast model first, then a high-resolution IQA model on top candidates",
    )
    parser.add_argument(
        "--stage2-top-pct",
        type=float,
        help="Percentage of top assets from Stage 1 to evaluate in Stage 2 (default: 15.0)",
    )
    parser.add_argument(
        "--stage2-model",
        help="Model ID for Stage 2 scoring (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--stage2-weight",
        type=float,
        help="Weight of Stage 2 score in the combined score (0.0 to 1.0, default: 0.5)",
    )
    return parser


def resolve_config(
    argv: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    config_override: Optional[Mapping[str, Any]] = None,
) -> ScorerConfig:
    """Parses CLI arguments and resolves configuration settings (`CLI > Env > Config JSON > Defaults`).

    Args:
        argv: Optional sequence of CLI argument strings (defaults to `sys.argv[1:]`).
        env: Optional environment variable mapping (defaults to `os.environ`).
        config_override: Optional pre-loaded config dictionary (bypasses disk read if given).

    Returns:
        Populated `ScorerConfig` instance.
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    env_map: Mapping[str, str] = env if env is not None else os.environ

    config: dict[str, Any] = {}
    if config_override is not None:
        config = dict(config_override)
    elif os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    config = loaded
            print(f"Loaded configuration settings from '{args.config}'")
        except Exception as exc:
            print(f"Warning: Failed to load config from {args.config}: {exc}")

    def resolve(cli_val: Any, env_name: str, config_key: str, default: Any) -> Any:
        # Precedence: CLI > Env > Config JSON > Defaults
        if cli_val is not None:
            return cli_val
        env_val = env_map.get(env_name)
        if env_val is not None and env_val.strip() != "":
            return env_val
        if config and config_key in config:
            cfg_val = config[config_key]
            if cfg_val is not None and (not isinstance(cfg_val, str) or str(cfg_val).strip() != ""):
                return cfg_val
        return default

    def _to_int(val: Any, fallback: int) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return fallback

    def _to_float(val: Any, fallback: float) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return fallback

    def _resolve_bool(cli_flag: Optional[bool], config_key: str) -> bool:
        if cli_flag is not None:
            return bool(cli_flag)
        return bool(config.get(config_key, False)) if config else False

    immich_url = resolve(args.immich_url, "IMMICH_URL", "immich_url", None)
    if isinstance(immich_url, str):
        immich_url = immich_url.rstrip("/")

    api_key = resolve(args.api_key, "IMMICH_API_KEY", "immich_api_key", None)
    person_id = resolve(args.person_id, "PERSON_ID", "person_id", None)
    album_id = resolve(args.album_id, "ALBUM_ID", "album_id", None)

    # CLI target flags are mutually exclusive and override config/env targets
    if args.album_id is not None:
        person_id = None
    elif args.person_id is not None:
        album_id = None

    scorer_type = str(resolve(args.scorer_type, "SCORER_TYPE", "scorer_type", "gemini")).lower()
    default_delay = 0.0 if scorer_type == "local" else DEFAULT_DELAY

    return ScorerConfig(
        config_path=args.config,
        immich_url=immich_url,
        api_key=api_key,
        person_id=person_id,
        album_id=album_id,
        gemini_key=resolve(args.gemini_key, "GEMINI_API_KEY", "gemini_api_key", None),
        target_album_name=resolve(
            args.target_album_name, "TARGET_ALBUM_NAME", "target_album_name", None
        ),
        scorer_type=scorer_type,
        local_model_id=str(
            resolve(
                args.local_model,
                "LOCAL_MODEL_ID",
                "local_model_id",
                "somepago/AestheticSigLIP",
            )
        ),
        gemini_model=str(
            resolve(args.gemini_model, "GEMINI_MODEL", "gemini_model", "gemini-2.5-flash")
        ),
        openai_key=resolve(args.openai_key, "OPENAI_API_KEY", "openai_api_key", None),
        openai_url=str(
            resolve(
                args.openai_url,
                "OPENAI_BASE_URL",
                "openai_base_url",
                "https://api.openai.com/v1",
            )
        ),
        openai_model=str(resolve(args.openai_model, "OPENAI_MODEL", "openai_model", "gpt-4o-mini")),
        write_ratings=_resolve_bool(args.write_ratings, "write_ratings"),
        limit=_to_int(resolve(args.limit, "LIMIT", "limit", DEFAULT_LIMIT), DEFAULT_LIMIT),
        concurrency=_to_int(
            resolve(args.concurrency, "CONCURRENCY", "concurrency", DEFAULT_CONCURRENCY),
            DEFAULT_CONCURRENCY,
        ),
        delay=_to_float(resolve(args.delay, "DELAY", "delay", default_delay), default_delay),
        cache_file=str(resolve(args.cache_file, "CACHE_FILE", "cache_file", DEFAULT_CACHE_FILE)),
        dedup_window=_to_int(resolve(args.dedup_window, "DEDUP_WINDOW", "dedup_window", 0), 0),
        dry_run=_resolve_bool(args.dry_run, "dry_run"),
        force_score=_resolve_bool(args.force_score, "force_score"),
        use_cache_only=_resolve_bool(args.use_cache_only, "use_cache_only"),
        two_stage=_resolve_bool(args.two_stage, "two_stage"),
        stage2_top_pct=_to_float(
            resolve(args.stage2_top_pct, "STAGE2_TOP_PCT", "stage2_top_pct", 15.0), 15.0
        ),
        stage2_model=str(
            resolve(args.stage2_model, "STAGE2_MODEL", "stage2_model", "gemini-2.5-flash")
        ),
        stage2_weight=_to_float(
            resolve(args.stage2_weight, "STAGE2_WEIGHT", "stage2_weight", 0.5), 0.5
        ),
    )


def _sync_star_ratings(
    scored_assets: Sequence[dict[str, Any]],
    gateway: ImmichGateway,
    concurrency: int,
) -> None:
    print(
        f"Syncing calibrated star ratings (1-5) to Immich in parallel using {concurrency} threads..."
    )

    # Group assets by target star rating (1..5) so ImmichGateway.update_assets_rating
    # can send 100-item batch PUT /api/assets payloads instead of 1 request per asset.
    buckets: dict[int, list[str]] = {}
    for asset_res in scored_assets:
        stars = int(asset_res.get("rating") or score_to_stars(asset_res["score"]))
        buckets.setdefault(stars, []).append(str(asset_res["id"]))

    def _update_bucket(item: tuple[int, list[str]]) -> bool:
        stars, ids = item
        try:
            gateway.update_assets_rating(ids, stars)
            return True
        except Exception as exc:
            print(f"Warning: Failed to update {stars}-star rating batch ({len(ids)} assets): {exc}")
            return False

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(buckets) or 1))) as executor:
        futures = [executor.submit(_update_bucket, item) for item in sorted(buckets.items())]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Syncing Ratings",
        ):
            f.result()
    print("Star ratings sync completed!")


def run_scoring_workflow(
    cfg: ScorerConfig,
    *,
    gateway: Optional[ImmichGateway] = None,
    store: Optional[EvaluationStore] = None,
    stage1_evaluator: Optional[ImageEvaluator] = None,
    stage2_evaluator: Optional[ImageEvaluator] = None,
    pipeline: Optional[ScorePipeline] = None,
) -> list[dict[str, Any]]:
    """Coordinates `ImmichGateway`, `EvaluationStore`, `ImageEvaluator`, and `ScorePipeline`.

    Delegates Stage 1 (and optional Stage 2) evaluation, z-score sigmoid calibration,
    score fusion, store persistence, and burst deduplication to `pipeline.run(...)`,
    optionally synchronizes `1–5` star ratings back to Immich (`cfg.write_ratings`) across
    `result.scored_assets`, and updates the target highlights album from
    `result.deduplicated_assets[:cfg.limit]` unless `cfg.dry_run` is enabled.

    Args:
        cfg: Resolved `ScorerConfig` parameters.
        gateway: Optional pre-configured `ImmichGateway` (constructed from `cfg` if omitted).
        store: Optional `EvaluationStore` instance (loaded from `cfg.cache_file` if omitted).
        stage1_evaluator: Optional Stage 1 `ImageEvaluator` (created via factory if omitted).
        stage2_evaluator: Optional Stage 2 `ImageEvaluator` (created via factory if omitted).
        pipeline: Optional `ScorePipeline` instance (constructed from `cfg` if omitted).

    Returns:
        List of top-scoring highlight asset dictionaries up to `cfg.limit`.

    Raises:
        ValueError: If `gateway` is `None` and `cfg.immich_url` or `cfg.api_key` is missing,
            or if evaluator initialization fails due to missing API keys.
        ImmichGatewayError: If asset retrieval or target album synchronization fails.
    """
    if gateway is None:
        if not cfg.immich_url or not cfg.api_key:
            raise ValueError("immich_url and api_key are required to initialize ImmichGateway")
        gateway = ImmichGateway(
            cfg.immich_url,
            cfg.api_key,
            pool_maxsize=max(20, cfg.concurrency * 2),
        )

    if store is None:
        store = EvaluationStore.from_disk(cfg.cache_file)
    print(f"Loaded {len(store)} entries from cache file: {cfg.cache_file}")

    assets = gateway.fetch_image_assets(person_id=cfg.person_id, album_id=cfg.album_id)
    target_label = "person" if cfg.person_id else "album"
    if not assets:
        print(f"No image assets found for the specified {target_label}.")
        return []
    print(f"Found {len(assets)} total image assets for the {target_label}.")

    rate_limiter = RateLimiter(delay=cfg.delay)
    if stage1_evaluator is None:
        stage1_evaluator = create_evaluator(
            scorer_type=cfg.scorer_type,
            stage=1,
            local_model_id=cfg.local_model_id,
            gemini_key=cfg.gemini_key,
            gemini_model=cfg.gemini_model,
            openai_key=cfg.openai_key,
            openai_url=cfg.openai_url,
            openai_model=cfg.openai_model,
            delay=cfg.delay,
            rate_limiter=rate_limiter,
        )

    if cfg.two_stage and stage2_evaluator is None:
        stage2_evaluator = create_evaluator(
            scorer_type=cfg.scorer_type,
            stage=2,
            stage2_model=cfg.stage2_model,
            gemini_key=cfg.gemini_key,
            gemini_model=cfg.gemini_model,
            openai_key=cfg.openai_key,
            openai_url=cfg.openai_url,
            openai_model=cfg.openai_model,
            delay=cfg.delay,
            rate_limiter=rate_limiter,
        )

    if pipeline is None:
        pipeline = ScorePipeline(
            stage2_weight=cfg.stage2_weight,
            dedup_window=float(cfg.dedup_window),
        )

    s1_model_id = stage1_evaluator.model_id
    if cfg.scorer_type == "local":
        s1_label = f"local model '{COLOR_BOLD}{s1_model_id}{COLOR_RESET}'"
        stage1_type_str = "Coarse Quality Filtering"
    elif cfg.scorer_type == "gemini":
        s1_label = f"Gemini API model '{COLOR_BOLD}{s1_model_id}{COLOR_RESET}'"
        stage1_type_str = "Aesthetics Evaluation"
    else:
        s1_label = f"OpenAI API model '{COLOR_BOLD}{s1_model_id}{COLOR_RESET}'"
        stage1_type_str = "Aesthetics Evaluation"

    print(f"\n--- {COLOR_CYAN}Stage 1: {stage1_type_str}{COLOR_RESET} (using {s1_label}) ---")

    result = pipeline.run(
        assets=assets,
        store=store,
        stage1_evaluator=stage1_evaluator,
        thumbnail_loader=gateway.download_thumbnail,
        stage2_evaluator=stage2_evaluator if cfg.two_stage else None,
        stage2_top_pct=cfg.stage2_top_pct,
        concurrency=cfg.concurrency,
        force_score=cfg.force_score,
        use_cache_only=cfg.use_cache_only,
        delay=cfg.delay,
        scorer_type=cfg.scorer_type,
    )

    if cfg.write_ratings:
        if cfg.dry_run:
            print("\n[Dry Run] skipping star ratings sync to Immich.")
        else:
            _sync_star_ratings(result.scored_assets, gateway, cfg.concurrency)

    deduplicated_assets = result.deduplicated_assets

    print("\n--- TOP 10 HIGHEST SCORING IMAGES ---")
    for idx, item in enumerate(deduplicated_assets[:10], 1):
        asset_info = item["asset"]
        orig_name = asset_info.get("originalFileName", "Unknown")
        created_at = str(asset_info.get("fileCreatedAt", "Unknown Date"))[:10]
        print(
            f"{idx:2d}. Score: {item['score']:3d} | File: {orig_name} ({created_at}) | Reason: {item['reason']}"
        )

    limit = min(cfg.limit, len(deduplicated_assets))
    top_assets = deduplicated_assets[:limit]
    print(
        f"\nSelecting the {limit} highest scoring images (Minimum score in top selection: "
        f"{top_assets[-1]['score'] if top_assets else 0})."
    )

    if cfg.dry_run:
        print("\n[Dry Run] skipping album creation and asset addition.")
        return top_assets

    if cfg.target_album_name:
        top_ids = [str(item["id"]) for item in top_assets]
        gateway.ensure_album_with_assets(cfg.target_album_name, top_ids)
        print(
            f"\nSuccess! Successfully added the top {len(top_ids)} photos to album '{cfg.target_album_name}'."
        )

    return top_assets


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint coordinating interactive credential resolution and the scoring workflow."""
    cfg = resolve_config(argv=argv)

    print("Welcome to Immich Aesthetic Scorer!")
    if not cfg.immich_url:
        cfg.immich_url = get_or_prompt(
            "IMMICH_URL", "Enter Immich Server URL", default="http://localhost:2283"
        ).rstrip("/")
    if not cfg.api_key:
        cfg.api_key = get_or_prompt("IMMICH_API_KEY", "Enter Immich API Key", secret=True)

    gateway = ImmichGateway(
        cfg.immich_url,
        cfg.api_key,
        pool_maxsize=max(20, cfg.concurrency * 2),
    )

    print("Verifying connection to Immich...")
    try:
        version = gateway.verify_connection()
        print(f"Connected successfully! Immich Server Version: {version.major}.{version.minor}")
    except ImmichGatewayError as exc:
        print(f"Error: Could not connect to Immich at {cfg.immich_url}. Error: {exc}")
        sys.exit(1)

    person_name: Optional[str] = None
    album_name_source: Optional[str] = None

    if cfg.person_id:
        try:
            person_details = gateway.get_person_details(cfg.person_id)
            person_name = person_details.get("name")
            print(f"Selected Person: {person_name or 'Unnamed'} (ID: {cfg.person_id})")
        except ImmichGatewayError:
            print(
                f"Warning: Person ID {cfg.person_id} provided but could not fetch details from server."
            )
    elif cfg.album_id:
        try:
            album_details = gateway.get_album_details(cfg.album_id)
            album_name_source = album_details.get("albumName")
            print(f"Selected Source Album: {album_name_source} (ID: {cfg.album_id})")
        except ImmichGatewayError:
            print(
                f"Warning: Album ID {cfg.album_id} provided but could not fetch details from server."
            )
    else:
        print("\n--- Target Selection ---")
        print(" [1] Process photos of a Person (from facial recognition)")
        print(" [2] Process photos from an existing Album")
        target_choice = input("Select option [1-2, default: 1]: ").strip()
        if target_choice == "2":
            cfg.album_id, album_name_source = select_album_interactive(cfg.immich_url, cfg.api_key)
        else:
            cfg.person_id, person_name = select_person_interactive(cfg.immich_url, cfg.api_key)

    if not cfg.person_id and not cfg.album_id:
        print("Error: No person or album selected. Exiting.")
        sys.exit(1)

    s2_provider = _classify_stage2_provider(cfg.stage2_model) if cfg.two_stage else "none"
    need_gemini = (cfg.scorer_type == "gemini") or (s2_provider == "gemini")
    need_openai = (cfg.scorer_type == "openai") or (s2_provider == "openai")
    if need_gemini and not cfg.gemini_key:
        cfg.gemini_key = get_or_prompt(
            "GEMINI_API_KEY",
            "Enter Google Gemini API Key",
            secret=True,
            default=DEFAULT_GEMINI_KEY,
        )
    if need_openai and not cfg.openai_key:
        cfg.openai_key = get_or_prompt(
            "OPENAI_API_KEY", "Enter OpenAI (or compatible) API Key", secret=True
        )

    if not cfg.target_album_name:
        if cfg.person_id:
            default_album = (
                f"Best of {person_name}" if person_name else f"Best of Person {cfg.person_id[:8]}"
            )
        else:
            default_album = (
                f"Best of Album {album_name_source}"
                if album_name_source
                else f"Best of Album {cfg.album_id[:8]}"
                if cfg.album_id
                else "Best of Album"
            )
        try:
            user_album = input(f"Enter target album name [{default_album}]: ").strip()
        except (EOFError, OSError):
            user_album = ""
        cfg.target_album_name = user_album or default_album

    try:
        run_scoring_workflow(cfg, gateway=gateway)
    except ImmichGatewayError as exc:
        print(f"\nError communicating with Immich: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cache has been saved. Exiting.")
        sys.exit(1)
