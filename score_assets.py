#!/usr/bin/env python3
"""Immich Aesthetic Scorer & Highlight Album Creator.

CLI driver and backward-compatible facade coordinating `ScorePipeline`,
`EvaluationStore`, `ImageEvaluator`, and `ImmichGateway`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from getpass import getpass
from typing import Any, Callable, Mapping, Optional, Sequence

import requests
from tqdm import tqdm

import immich_aesthetic_scorer.evaluators as _evaluators_mod
from immich_aesthetic_scorer.evaluators import (  # noqa: F401
    AESTHETIC_PROMPT,
    EvaluationResult,
    FakeEvaluator,
    GeminiEvaluator,
    ImageEvaluator,
    LocalClipEvaluator,
    OpenAIEvaluator,
    PyIQAEvaluator,
    RateLimiter,
    RsinemaAestheticScorer,
    create_evaluator,
    negotiate_torch_device,
    resize_image_bytes,
)
from immich_aesthetic_scorer.gateway import (  # noqa: F401
    ImmichGateway,
    ImmichGatewayError,
    ServerVersion,
)
from immich_aesthetic_scorer.pipeline import (  # noqa: F401
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
from immich_aesthetic_scorer.store import (  # noqa: F401
    EvaluationStore,
    cache_lock,
    extract_raw_score_from_reason,
    get_asset_mod_time,
    load_cache,
    save_cache,
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

# Optional ML modules exposed at module scope for backward compatibility with existing tests
torch: Any = None
Image: Any = None
CLIPProcessor: Any = None
pyiqa: Any = None

try:
    import torch

    # Restrict PyTorch CPU intra-op threads to prevent thread contention across parallel workers
    torch.set_num_threads(1)
    from PIL import Image
    from transformers import CLIPProcessor
except ImportError:
    from PIL import Image

try:
    import pyiqa
except ImportError:
    pass

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

# Legacy module-level state preserved for backward-compatible test assertions
local_model: Any = None
local_processor: Any = None
model_lock = threading.Lock()
local_stage2_model: Any = None
stage2_model_lock = threading.Lock()
_cached_local_evaluator: Optional[LocalClipEvaluator] = None
_cached_stage2_evaluator: Optional[PyIQAEvaluator] = None


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


def check_immich_connection(immich_url: str, api_key: str) -> tuple[bool, int | str, Optional[int]]:
    """Verifies connection and returns `(True, major, minor)` or `(False, error_str, None)`."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    try:
        version = gateway.verify_connection()
        return True, version.major, version.minor
    except ImmichGatewayError as exc:
        if exc.status_code is not None:
            return False, f"Status code {exc.status_code}", None
        return False, str(exc), None


def get_person_details(immich_url: str, api_key: str, person_id: str) -> Optional[dict[str, Any]]:
    """Retrieves metadata of a person from Immich, or returns `None` on failure."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    try:
        return gateway.get_person_details(person_id)
    except ImmichGatewayError:
        return None


def search_people_by_name(immich_url: str, api_key: str, query: str) -> list[dict[str, Any]]:
    """Searches Immich people by name substring, returning `[]` on error."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    try:
        return gateway.search_people(query)
    except ImmichGatewayError as exc:
        print(f"Error searching people: {exc}")
        return []


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


def get_album_details(immich_url: str, api_key: str, album_id: str) -> Optional[dict[str, Any]]:
    """Retrieves metadata of an album from Immich, or returns `None` on failure."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    try:
        return gateway.get_album_details(album_id)
    except ImmichGatewayError:
        return None


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


def fetch_all_image_assets(
    immich_url: str,
    api_key: str,
    person_id: Optional[str] = None,
    album_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fetches all `IMAGE` assets for a person or album via `ImmichGateway`."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    return gateway.fetch_image_assets(person_id=person_id, album_id=album_id)


def download_thumbnail(
    immich_url: str,
    api_key: str,
    asset_id: str,
    max_dim: Optional[int] = 512,
) -> bytes:
    """Downloads an uncropped preview thumbnail from Immich and optionally downscales it."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    raw_bytes = gateway.download_thumbnail(asset_id)
    return resize_image_bytes(raw_bytes, max_dim=max_dim)


def call_gemini_api(
    api_key: str,
    image_bytes: bytes,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 5,
    initial_backoff: float = 4.0,
) -> dict[str, Any]:
    """Evaluates image aesthetic quality via `GeminiEvaluator` (0–100 integer score)."""
    evaluator = GeminiEvaluator(
        api_key=api_key,
        model_id=model_name,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        raw_scale=1.0,
        session=requests,
    )
    res = evaluator.evaluate(image_bytes)
    return {"score": res.score, "reason": res.reason}


def call_openai_api(
    api_key: str,
    base_url: str,
    model_name: str,
    image_bytes: bytes,
    max_retries: int = 5,
    initial_backoff: float = 4.0,
) -> dict[str, Any]:
    """Evaluates image aesthetic quality via `OpenAIEvaluator` (0–100 integer score)."""
    evaluator = OpenAIEvaluator(
        api_key=api_key,
        base_url=base_url,
        model_id=model_name,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        raw_scale=1.0,
        session=requests,
    )
    res = evaluator.evaluate(image_bytes)
    return {"score": res.score, "reason": res.reason}


def score_image_local(image_bytes: bytes, model_id: str) -> dict[str, Any]:
    """Evaluates image aesthetics locally using `LocalClipEvaluator` (0.0–10.0 raw scale).

    Synchronizes module-level ML bindings (`torch`, `CLIPProcessor`, `RsinemaAestheticScorer`,
    `Image`) with `immich_aesthetic_scorer.evaluators` so unit tests patching `score_assets.*`
    intercept model loading and inference.
    """
    global local_model, local_processor, _cached_local_evaluator

    with model_lock:
        _evaluators_mod.torch = torch
        _evaluators_mod.CLIPProcessor = CLIPProcessor
        _evaluators_mod.RsinemaAestheticScorer = RsinemaAestheticScorer
        _evaluators_mod.Image = Image

        if (
            local_model is None
            or _cached_local_evaluator is None
            or _cached_local_evaluator.model_id != model_id
        ):
            _cached_local_evaluator = LocalClipEvaluator(model_id=model_id)

        result = _cached_local_evaluator.evaluate(image_bytes)
        local_model = getattr(_cached_local_evaluator, "_model", None)
        local_processor = getattr(_cached_local_evaluator, "_processor", None)
        return result.to_dict()


def call_gemini_api_stage2(
    api_key: str,
    image_bytes: bytes,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 5,
    initial_backoff: float = 4.0,
) -> dict[str, Any]:
    """Evaluates Stage 2 aesthetics via `GeminiEvaluator` (0.0–100.0 raw score)."""
    evaluator = GeminiEvaluator(
        api_key=api_key,
        model_id=model_name,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        raw_scale=1.0,
        session=requests,
    )
    res = evaluator.evaluate(image_bytes)
    return {"raw_score": res.raw_score, "reason": res.reason}


def call_openai_api_stage2(
    api_key: str,
    base_url: str,
    model_name: str,
    image_bytes: bytes,
    max_retries: int = 5,
    initial_backoff: float = 4.0,
) -> dict[str, Any]:
    """Evaluates Stage 2 aesthetics via `OpenAIEvaluator` (0.0–100.0 raw score)."""
    evaluator = OpenAIEvaluator(
        api_key=api_key,
        base_url=base_url,
        model_id=model_name,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        raw_scale=1.0,
        session=requests,
    )
    res = evaluator.evaluate(image_bytes)
    return {"raw_score": res.raw_score, "reason": res.reason}


def score_image_stage2(
    image_bytes: bytes,
    model_name: str = "musiq-spaq",
    gemini_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_url: Optional[str] = None,
    openai_model: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluates Stage 2 technical or aesthetic quality via `create_evaluator`."""
    global local_stage2_model, _cached_stage2_evaluator

    _evaluators_mod.pyiqa = pyiqa
    provider = _classify_stage2_provider(model_name)
    if provider == "local":
        with stage2_model_lock:
            if (
                local_stage2_model is None
                or _cached_stage2_evaluator is None
                or _cached_stage2_evaluator.model_id != model_name
            ):
                _cached_stage2_evaluator = PyIQAEvaluator(model_id=model_name)
            res = _cached_stage2_evaluator.evaluate(image_bytes)
            local_stage2_model = getattr(_cached_stage2_evaluator, "_metric", None)
            return {"raw_score": res.raw_score, "reason": res.reason}

    evaluator = create_evaluator(
        stage=2,
        stage2_model=model_name,
        gemini_key=gemini_key,
        gemini_model=gemini_model or "gemini-2.5-flash",
        openai_key=openai_key,
        openai_url=openai_url or "https://api.openai.com/v1",
        openai_model=openai_model or "gpt-4o-mini",
        session=requests,
    )
    res = evaluator.evaluate(image_bytes)
    return {"raw_score": res.raw_score, "reason": res.reason}


def get_or_create_album(immich_url: str, api_key: str, album_name: str) -> str:
    """Retrieves an existing Immich album by name or creates a new one if missing."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    return gateway.ensure_album_with_assets(album_name, [])


def add_assets_to_album(
    immich_url: str, api_key: str, album_id: str, asset_ids: Sequence[str]
) -> None:
    """Registers a sequence of asset IDs into an existing Immich album in 100-item batches."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    gateway.add_assets_to_album(album_id, asset_ids)


def update_asset_rating(immich_url: str, api_key: str, asset_id: str, rating: int) -> None:
    """Updates the native star rating (`1–5`) of an asset on the Immich server."""
    gateway = ImmichGateway(immich_url, api_key, session=requests)
    try:
        gateway.update_asset_rating(asset_id, rating)
    except Exception as exc:
        print(f"\nWarning: Failed to update rating in Immich for asset {asset_id}: {exc}")


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
        action="store_true",
        default=None,
        help="Download and score assets but do not make changes to albums in Immich",
    )
    parser.add_argument(
        "--force-score",
        action="store_true",
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
        action="store_true",
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
        action="store_true",
        default=None,
        help="Only compile the album using already-cached images, skipping scoring of uncached ones",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
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


def _evaluate_stage1_batch(
    assets: Sequence[dict[str, Any]],
    cfg: ScorerConfig,
    gateway: ImmichGateway,
    store: EvaluationStore,
    evaluator: ImageEvaluator,
) -> list[dict[str, Any]]:
    stage1_results: list[dict[str, Any]] = []
    assets_to_score: list[dict[str, Any]] = []
    s1_model_name = evaluator.model_id

    for asset in assets:
        raw_s1 = store.get_stage1(asset, model_id=s1_model_name, force=cfg.force_score)
        if raw_s1 is not None:
            stage1_results.append(
                {
                    "id": asset["id"],
                    "asset": asset,
                    "raw_score_stage1": float(raw_s1),
                }
            )
        else:
            assets_to_score.append(asset)

    print(f"Stage 1 cached: {len(stage1_results)}")
    if cfg.use_cache_only:
        if assets_to_score:
            print("Running in --use-cache-only mode. Skipping Stage 1 scoring of uncached assets.")
        return stage1_results

    print(f"Stage 1 needing scoring: {len(assets_to_score)}")
    if not assets_to_score:
        return stage1_results

    # Remote vision APIs enforce strict RPM limits; serialize worker threads for non-local scorers
    concurrency_s1 = 1 if cfg.scorer_type not in ("local", "fake") else max(1, cfg.concurrency)
    print(
        f"Starting parallel Stage 1 scoring (using '{s1_model_name}') with {concurrency_s1} worker threads."
    )
    if cfg.scorer_type not in ("local", "fake"):
        print(
            f"Injecting a {cfg.delay}s delay between requests per thread to respect API rate limits."
        )

    def _score_one(asset_item: dict[str, Any]) -> dict[str, Any]:
        asset_id = str(asset_item["id"])
        try:
            raw_bytes = gateway.download_thumbnail(asset_id)
            img_bytes = resize_image_bytes(raw_bytes, max_dim=evaluator.required_max_dimension)
            eval_res = evaluator.evaluate(img_bytes)
            store.record_stage1(asset_item, eval_res.raw_score, s1_model_name)
            if cfg.scorer_type == "local" and cfg.delay > 0:
                time.sleep(cfg.delay)
            return {
                "id": asset_id,
                "raw_score_stage1": eval_res.raw_score,
                "asset": asset_item,
                "status": "success",
            }
        except Exception as exc:
            return {"id": asset_id, "status": "error", "error": str(exc), "asset": asset_item}

    def _on_s1_success(item_res: dict[str, Any], pbar: Any) -> None:
        stage1_results.append(item_res)
        filename = item_res["asset"].get("originalFileName") or item_res["id"]
        pbar.write(
            f"Stage 1 Scored {COLOR_BLUE}{filename}{COLOR_RESET} using "
            f"'{COLOR_BOLD}{s1_model_name}{COLOR_RESET}': "
            f"Raw Score {COLOR_GREEN}{item_res['raw_score_stage1']:.2f}{COLOR_RESET}"
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
    candidates: Sequence[dict[str, Any]],
    cfg: ScorerConfig,
    gateway: ImmichGateway,
    store: EvaluationStore,
    evaluator: ImageEvaluator,
) -> None:
    s2_model_name = evaluator.model_id
    is_local_s2 = _classify_stage2_provider(s2_model_name) == "local"
    stage2_type_str = "Technical Quality Evaluation" if is_local_s2 else "Aesthetics Evaluation"
    print(
        f"\n--- {COLOR_MAGENTA}Stage 2: {stage2_type_str}{COLOR_RESET} "
        f"(selecting top {len(candidates)} candidates ({cfg.stage2_top_pct}%) using model '{COLOR_BOLD}{s2_model_name}{COLOR_RESET}') ---"
    )

    cached_count = 0
    assets_to_score: list[dict[str, Any]] = []
    for item in candidates:
        asset_obj = item.get("asset") or item["id"]
        raw_s2 = store.get_stage2(asset_obj, model_id=s2_model_name, force=cfg.force_score)
        if raw_s2 is not None:
            item["raw_score_stage2"] = float(raw_s2)
            reason_s2 = store.get_stage2_reason(
                asset_obj, model_id=s2_model_name, force=cfg.force_score
            )
            if reason_s2 is not None:
                item["reason_stage2"] = reason_s2
            cached_count += 1
        else:
            assets_to_score.append(item)

    print(f"Stage 2 cached: {cached_count}")
    if cfg.use_cache_only:
        if assets_to_score:
            print("Running in --use-cache-only mode. Skipping Stage 2 scoring of uncached assets.")
        return

    print(f"Stage 2 needing scoring: {len(assets_to_score)}")
    if not assets_to_score:
        return

    concurrency_s2 = 1 if not is_local_s2 else max(1, cfg.concurrency)
    print(
        f"Starting parallel Stage 2 scoring (using model '{s2_model_name}') with {concurrency_s2} worker threads."
    )

    def _score_one_s2(cand_item: dict[str, Any]) -> dict[str, Any]:
        asset_id = str(cand_item["id"])
        try:
            raw_bytes = gateway.download_thumbnail(asset_id)
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
            f"Stage 2 Scored {COLOR_BLUE}{filename}{COLOR_RESET} using "
            f"'{COLOR_BOLD}{s2_model_name}{COLOR_RESET}': "
            f"Raw Score {COLOR_GREEN}{res['raw_score_stage2']:.2f}{COLOR_RESET} | "
            f"Stage 1 Raw: {COLOR_YELLOW}{cand['raw_score_stage1']:.2f}{COLOR_RESET}"
        )

    _run_parallel_stage(
        stage_num=2,
        items_to_score=assets_to_score,
        concurrency=concurrency_s2,
        store=store,
        worker_fn=_score_one_s2,
        on_success=_on_s2_success,
    )


def _sync_star_ratings(
    scored_assets: Sequence[dict[str, Any]],
    gateway: ImmichGateway,
    concurrency: int,
) -> None:
    print(
        f"Syncing calibrated star ratings (1-5) to Immich in parallel using {concurrency} threads..."
    )

    def _update_worker(asset_res: dict[str, Any]) -> bool:
        try:
            stars = int(asset_res.get("rating") or score_to_stars(asset_res["score"]))
            gateway.update_asset_rating(str(asset_res["id"]), stars)
            return True
        except Exception as exc:
            print(f"Warning: Failed to update rating for asset {asset_res['id']}: {exc}")
            return False

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as rating_executor:
        rating_futures = [rating_executor.submit(_update_worker, res) for res in scored_assets]
        for f in tqdm(
            as_completed(rating_futures),
            total=len(rating_futures),
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

    Executes Stage 1 (and optional Stage 2) evaluations across a thread pool, atomically
    flushes intermediate and final calibrated scores to `store`, optionally synchronizes
    `1–5` star ratings back to Immich (`cfg.write_ratings`), and updates the target
    highlights album unless `cfg.dry_run` is enabled.

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

    stage1_results = _evaluate_stage1_batch(assets, cfg, gateway, store, stage1_evaluator)
    s1_stats = pipeline.calibrate_stage1(stage1_results)

    if not cfg.two_stage:
        print("\nCalibrating aesthetic scores based on library distribution...")
        print("Library Statistics:")
        print(f"  - Total Scored Photos: {len(stage1_results)}")
        print(f"  - Raw Score Mean: {s1_stats.mean:.3f}")
        print(f"  - Raw Score Std Dev: {s1_stats.std:.3f}")
        print("Applying sigmoid normalization and updating cache...")

        scored_assets = pipeline.single_stage_finalize(stage1_results)
        for item in scored_assets:
            store.record_final(
                asset_id=str(item["id"]),
                score=int(item["score"]),
                reason=str(item["reason"]),
                raw_score_stage1=float(item["raw_score_stage1"]),
                model_id_stage1=stage1_evaluator.model_id,
                updated_at=get_asset_mod_time(item["asset"]),
            )
        store.flush()
        print("Cache updated successfully!")
    else:
        if stage2_evaluator is None:
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

        candidates_for_stage2, _ = pipeline.split_candidates(
            stage1_results, top_pct=cfg.stage2_top_pct
        )
        _evaluate_stage2_batch(candidates_for_stage2, cfg, gateway, store, stage2_evaluator)

        print(
            f"\nCombining Stage 1 ({stage1_evaluator.model_id}) and "
            f"Stage 2 ({stage2_evaluator.model_id}) scores..."
        )
        scored_assets = pipeline.fuse_stage2(stage1_results)
        s2_stats = pipeline.s2_stats
        if s2_stats is not None:
            print("Stage 2 Statistics:")
            print(f"  - Total Scored Candidates: {len(scored_assets)}")
            print(f"  - Raw Score Mean: {s2_stats.mean:.3f}")
            print(f"  - Raw Score Std Dev: {s2_stats.std:.3f}")

        for item in scored_assets:
            raw_s2_val = item.get("raw_score_stage2")
            store.record_final(
                asset_id=str(item["id"]),
                score=int(item["score"]),
                reason=str(item["reason"]),
                raw_score_stage1=float(item["raw_score_stage1"]),
                model_id_stage1=stage1_evaluator.model_id,
                raw_score_stage2=float(raw_s2_val) if raw_s2_val is not None else None,
                model_id_stage2=stage2_evaluator.model_id if raw_s2_val is not None else None,
                reason_stage2=item.get("reason_stage2"),
                updated_at=get_asset_mod_time(item["asset"]),
            )
        store.flush()
        print("Cache updated successfully with combined scores!")

    if cfg.write_ratings:
        _sync_star_ratings(scored_assets, gateway, cfg.concurrency)

    if cfg.dedup_window > 0:
        print(f"\nDeduplicating burst photos within a {cfg.dedup_window}-second window...")
        original_count = len(scored_assets)
        scored_assets = pipeline.deduplicate(scored_assets, dedup_window=float(cfg.dedup_window))
        print(
            f"Deduplicated burst photos: Kept {len(scored_assets)} unique highlights out of "
            f"{original_count} scored photos (Filtered out {original_count - len(scored_assets)} burst duplicates)."
        )

    scored_assets.sort(key=lambda x: x["score"], reverse=True)

    print("\n--- TOP 10 HIGHEST SCORING IMAGES ---")
    for idx, item in enumerate(scored_assets[:10], 1):
        asset_info = item["asset"]
        orig_name = asset_info.get("originalFileName", "Unknown")
        created_at = str(asset_info.get("fileCreatedAt", "Unknown Date"))[:10]
        print(
            f"{idx:2d}. Score: {item['score']:3d} | File: {orig_name} ({created_at}) | Reason: {item['reason']}"
        )

    limit = min(cfg.limit, len(scored_assets))
    top_assets = scored_assets[:limit]
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
