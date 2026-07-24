#!/usr/bin/env python3
"""
Immich Aesthetic Scorer & Highlight Album Creator
Uses Gemini Flash or local models to score assets of a specific person in Immich and compiles the best into a highlight album.
"""

import argparse
import base64
import io
import json
import logging
import math
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from getpass import getpass
from typing import Any, Optional

import requests
from tqdm import tqdm

# Disable CUDA globally to prevent PyTorch initialization warnings on systems with unsupported GPU hardware
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Suppress Hugging Face hub telemetry/symlinks warnings and other log noises
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Silence optional libraries warnings (like unauthenticated requests) using the warnings filter


warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Set up logging levels to silence verbose warnings/report tables from Hugging Face & Transformers
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Define optional ML modules as Any to satisfy static type checking for conditional imports
torch: Any = None
nn: Any = None
Image: Any = None
CLIPProcessor: Any = None
CLIPVisionModel: Any = None
pyiqa: Any = None

# Import local scoring libraries globally at startup to prevent thread-safety races during parallel execution
try:
    import torch

    torch.set_num_threads(1)
    import torch.nn as nn
    from PIL import Image
    from transformers import CLIPProcessor, CLIPVisionModel
except ImportError:
    pass

try:
    import pyiqa
except ImportError:
    pass

# Determine dynamic base class for type safety
BaseModule: Any = nn.Module if nn is not None else object


class RsinemaAestheticScorer(BaseModule):
    """PyTorch Module wrapping CLIP Vision model with multiple aesthetic classification heads.

    Extracts 768-dimensional visual features using a pre-trained CLIP vision backbone
    and evaluates them across seven aesthetic and technical attributes: overall aesthetics,
    technical quality, composition, lighting, color harmony, depth of field, and semantic content.
    """

    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=False
        ).vision_model
        self.aesthetic_head = nn.Sequential(nn.Linear(768, 1))
        self.quality_head = nn.Sequential(nn.Linear(768, 1))
        self.composition_head = nn.Sequential(nn.Linear(768, 1))
        self.light_head = nn.Sequential(nn.Linear(768, 1))
        self.color_head = nn.Sequential(nn.Linear(768, 1))
        self.dof_head = nn.Sequential(nn.Linear(768, 1))
        self.content_head = nn.Sequential(nn.Linear(768, 1))

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs[1]
        aesthetic = self.aesthetic_head(pooled)
        quality = self.quality_head(pooled)
        composition = self.composition_head(pooled)
        light = self.light_head(pooled)
        color = self.color_head(pooled)
        dof = self.dof_head(pooled)
        content = self.content_head(pooled)
        return torch.cat([aesthetic, quality, composition, light, color, dof, content], dim=-1)


# Constants
DEFAULT_GEMINI_KEY = None
DEFAULT_LIMIT = 100
DEFAULT_CONCURRENCY = 5
DEFAULT_DELAY = 4.0  # 4 seconds delay = max 15 RPM for free tier
DEFAULT_CACHE_FILE = ".immich_aesthetic_cache.json"

AESTHETIC_PROMPT = (
    "You are an expert photography judge. Analyze this photo and score its overall aesthetic "
    "quality and suitability for a high-quality highlight album.\n\n"
    "IMPORTANT: You are analyzing a low-resolution thumbnail of the photo (512x512 pixels max). "
    "It has been resized preserving its original aspect ratio and composition (it is NOT cropped). "
    "Assume that basic image sharpness and technical focus have already been verified by a technical filter. "
    "Do not penalize the score for lack of fine texture or low image resolution itself. "
    "Instead, focus on evaluating the overall composition, lighting, subject flattery, and highlight "
    "suitability as if it were a high-resolution photo.\n\n"
    "Grade the photo from 0 to 100 based on the following criteria:\n"
    "1. Composition: Is the framing clean and aesthetic? (Lower score for awkward crops, clutter, bad angles, distracting background elements).\n"
    "2. Lighting & Exposure: Is it well-exposed? (Lower score for heavy underexposure, overexposure, harsh unflattering shadows/highlights).\n"
    "3. Subject/Expression: If the photo features people, do they look natural, expressive, and flattering (no eyes closed, mid-speech, or awkward/unflattering faces)? If it features other subjects (landscapes, objects, animals), are they compelling and well-captured?\n"
    "4. Highlight Suitability: Is this a photo someone would want in a showcase album?\n\n"
    "Scoring Guide:\n"
    "- 90-100: Professional/stunning quality, perfect composition, lighting, and subject framing/expression.\n"
    "- 75-89: Very good quality, nice composition/lighting, expressive subject, minor imperfections.\n"
    "- 50-74: Decent snapshot, acceptable but lacks professional aesthetic, or has minor composition/lighting issues.\n"
    "- 20-49: Poor quality, bad lighting, or awkward framing/expression.\n"
    "- 0-19: Accidental, blank, or completely ruined photo.\n\n"
    "You must return a JSON object with 'score' (integer 0-100) and 'reason' (brief 1-sentence explanation of the score)."
)

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

# Thread safety lock for cache writes
cache_lock = threading.RLock()

# Global lock and timestamp to serialize Gemini API calls across all threads (prevents 429 quota bursts)
gemini_call_lock = threading.Lock()
last_gemini_call_time = 0.0


def load_cache(cache_path):
    """Loads the local JSON cache file from disk.

    The cache maintains historical score evaluations mapped to asset UUIDs,
    along with the model IDs used during evaluation to enable smart cache invalidation.

    Args:
        cache_path (str): File path of the cache file.

    Returns:
        dict: Cache content dictionary (empty dict if cache does not exist or is corrupted).
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}. Starting fresh.")
    return {}


def save_cache(cache, cache_path):
    """Saves cache entries atomically and thread-safely to disk.

    Writes to a temporary file first and renames it to prevent file corruption
    during write interrupts. Synchronized via global thread RLock.

    Args:
        cache (dict): Cache content to serialize.
        cache_path (str): Destination file path.
    """
    with cache_lock:
        try:
            # Write to a temporary file first, then rename to prevent corruption
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f"Error saving cache: {e}")


def get_asset_mod_time(asset):
    """Resolves a sensible modification/creation timestamp for cache invalidation."""
    mod_time = asset.get("fileModifiedAt")
    # If the modification time is missing or defaulted to the MS-DOS epoch (1980-01-01), fall back to fileCreatedAt
    if not mod_time or "1980-01-01" in mod_time:
        mod_time = asset.get("fileCreatedAt") or asset.get("createdAt")
    return mod_time


def get_or_prompt(env_name, prompt_text, secret=False, default=None):
    """Retrieves a configuration value from environment variables or prompts the user.

    Provides support for hidden input (e.g. passwords/API keys) using standard `getpass`.

    Args:
        env_name (str): The name of the environment variable to look up.
        prompt_text (str): Interactive prompt text displayed to the user.
        secret (bool): If True, hides character input during terminal typing.
        default (str, optional): Default value returned if the input is skipped.

    Returns:
        str: Resolved configuration value.
    """
    val = os.environ.get(env_name)
    if val:
        return val

    display_prompt = prompt_text
    if default:
        display_prompt += f" [{default}]: "
    else:
        display_prompt += ": "

    if secret:
        val = getpass(display_prompt)
    else:
        val = input(display_prompt)

    val = val.strip()
    if not val and default:
        return default
    return val


def check_immich_connection(immich_url, api_key):
    """Verifies connection and retrieves the Immich server's major/minor version.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.

    Returns:
        tuple: (bool, major_version/error_str, minor_version/None)
            - bool: True if connection succeeded, False otherwise.
            - major_version/error_str: Major version integer if successful, error message string if failed.
            - minor_version/None: Minor version integer if successful, None if failed.
    """
    url = f"{immich_url}/api/server/version"
    headers = {"x-api-key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            version_data = r.json()
            return True, version_data.get("major", 1), version_data.get("minor", 0)
    except Exception as e:
        return False, str(e), None
    return False, f"Status code {r.status_code}", None


def get_person_details(immich_url, api_key, person_id):
    """Retrieves metadata of a person from the Immich facial recognition database.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        person_id (str): UUID of the target person.

    Returns:
        dict: Person details (dictionary from response JSON), or None if request fails.
    """
    url = f"{immich_url}/api/people/{person_id}"
    headers = {"x-api-key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def search_people_by_name(immich_url, api_key, query):
    """Performs a search query against Immich people to find matches by name.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        query (str): Search term (e.g. name of the person).

    Returns:
        list: List of matching person records, or an empty list if search fails.
    """
    url = f"{immich_url}/api/search/person"
    headers = {"x-api-key": api_key}
    params = {"name": query}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Error searching people: {e}")
    return []


def select_person_interactive(immich_url, api_key):
    """Guides the user through an interactive terminal search and selection for an Immich person.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.

    Returns:
        tuple: (person_id, person_name)
            - person_id (str): Resolved UUID of the selected person.
            - person_name (str): The name of the selected person (or None if manually inputted).
    """
    print("\n--- Person Selection ---")
    while True:
        query = input("Search for a person by name (or press Enter to list all): ").strip()

        if query:
            results = search_people_by_name(immich_url, api_key, query)
        else:
            # Fetch all people
            url = f"{immich_url}/api/people"
            headers = {"x-api-key": api_key}
            try:
                r = requests.get(url, headers=headers, params={"size": 100}, timeout=10)
                results = r.json() if r.status_code == 200 else []
            except Exception as e:
                print(f"Error fetching people: {e}")
                results = []

        # Filter hidden/unnamed people out of direct listings if they don't match query
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
                return selected["id"], selected.get("name")
            elif choice == len(valid_people) + 1:
                continue
            elif choice == len(valid_people) + 2:
                manual_uuid = input("Enter Person UUID: ").strip()
                if manual_uuid:
                    return manual_uuid, None
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_album_details(immich_url, api_key, album_id):
    """Retrieves details/metadata of an album from the Immich server.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        album_id (str): UUID of the target album.

    Returns:
        dict: Album details dictionary, or None if request fails.
    """
    url = f"{immich_url}/api/albums/{album_id}"
    headers = {"x-api-key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def select_album_interactive(immich_url, api_key):
    """Guides the user through an interactive terminal search and selection for an Immich album.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.

    Returns:
        tuple: (album_id, album_name)
            - album_id (str): Resolved UUID of the selected album.
            - album_name (str): The name of the selected album (or None if manually inputted).
    """
    print("\n--- Album Selection ---")
    url = f"{immich_url}/api/albums"
    headers = {"x-api-key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print("Error: Could not retrieve albums list from server.")
            return None, None
        albums = r.json()
    except Exception as e:
        print(f"Error fetching albums: {e}")
        return None, None

    if not albums:
        print("No albums found on the server.")
        return None, None

    while True:
        query = input("Search for an album by name (or press Enter to list all): ").strip().lower()
        filtered = [a for a in albums if query in a.get("albumName", "").lower()]

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
                return selected["id"], selected.get("albumName")
            elif choice == len(filtered) + 1:
                continue
            elif choice == len(filtered) + 2:
                manual_uuid = input("Enter Album UUID: ").strip()
                if manual_uuid:
                    return manual_uuid, None
        except ValueError:
            print("Invalid input. Please enter a number.")


def fetch_all_image_assets(immich_url, api_key, person_id=None, album_id=None):
    """Paginates through search/metadata endpoint to fetch all assets for a person or album.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        person_id (str, optional): Target person ID to query.
        album_id (str, optional): Target album ID to query.

    Returns:
        list: Filtered list of asset dicts matching the target where type is "IMAGE".
    """
    url = f"{immich_url}/api/search/metadata"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    assets = []
    page = 1
    size = 250

    if person_id:
        payload_key = "personIds"
        payload_val = [person_id]
        print(f"Fetching asset list for person ID: {person_id}...")
    elif album_id:
        payload_key = "albumIds"
        payload_val = [album_id]
        print(f"Fetching asset list for album ID: {album_id}...")
    else:
        raise ValueError("Either person_id or album_id must be provided")

    while True:
        payload = {payload_key: payload_val, "page": page, "size": size}
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        items = data.get("assets", {}).get("items", [])
        if not items:
            break

        # Filter for images
        images = [item for item in items if item.get("type") == "IMAGE"]
        assets.extend(images)
        print(f"Fetched page {page} ({len(images)} images, cumulative: {len(assets)})")

        if len(items) < size:
            break

        page += 1

    return assets


def download_thumbnail(immich_url, api_key, asset_id, max_dim: Optional[int] = 512):
    """Downloads uncropped preview thumbnail from Immich and resizes it client-side if specified.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        asset_id (str): UUID of the target asset.
        max_dim (int, optional): Maximum dimension to resize the image to.
            If None or <= 0, returns the raw unscaled preview image.

    Returns:
        bytes: Downloaded image bytes (potentially resized/re-compressed).
    """
    headers = {"x-api-key": api_key}
    # Download the uncropped preview image
    url = f"{immich_url}/api/assets/{asset_id}/thumbnail?size=preview"
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    image_bytes = r.content

    if max_dim is None or max_dim <= 0:
        return image_bytes

    # Resize client-side to fit under max_dim limit (for 258-token Low-Res Mode)
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        orig_format = img.format or "JPEG"

        # Check current dimensions
        width, height = img.size
        if width > max_dim or height > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))

            if hasattr(Image, "Resampling"):
                resample_method = getattr(Image.Resampling, "LANCZOS")
            elif hasattr(Image, "LANCZOS"):
                resample_method = getattr(Image, "LANCZOS")
            else:
                resample_method = getattr(Image, "ANTIALIAS")

            img = img.resize((new_width, new_height), resample_method)

            # Save back to bytes
            out_bytes = io.BytesIO()
            img.save(out_bytes, format=orig_format, quality=85)
            return out_bytes.getvalue()
    except Exception as e:
        print(f"Warning: Failed to resize thumbnail client-side: {e}")

    return image_bytes


def call_gemini_api(
    api_key, image_bytes, model_name="gemini-2.5-flash", max_retries=5, initial_backoff=4.0
):
    """Evaluates the aesthetic quality and composition of an image using Google Gemini API.

    Performs image-to-text prompt scoring, applying rate limit backing off.

    Args:
        api_key (str): Google Gemini API Key.
        image_bytes (bytes): Binary data of the image.
        model_name (str): Gemini model identifier.
        max_retries (int): Maximum retry attempts for rate limits (429) or server errors.
        initial_backoff (float): Initial wait time in seconds for exponential backoff.

    Returns:
        dict: A dictionary containing 'score' (int 0-100) and 'reason' (str).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": AESTHETIC_PROMPT},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "score": {
                        "type": "INTEGER",
                        "description": "A quality and aesthetic appeal score for the image from 0 to 100.",
                    },
                    "reason": {
                        "type": "STRING",
                        "description": "A brief one-sentence reason explaining the score.",
                    },
                },
                "required": ["score", "reason"],
            },
        },
    }

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                res_json = r.json()
                # Parse JSON output string from Gemini
                text_out = res_json["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text_out)
            elif r.status_code == 429:
                print(
                    f"\n[Gemini API] Rate limit hit (429). Details: {r.text[:300]}\nRetrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(backoff)
                backoff *= 2.0
            elif r.status_code >= 500:
                print(
                    f"\n[Gemini API] Server error ({r.status_code}). Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(backoff)
                backoff *= 1.5
            else:
                raise Exception(f"Gemini API error {r.status_code}: {r.text}")
        except Exception as e:
            # Do not retry on permanent API errors (like 400, 403, 404, etc.)
            if "Gemini API error" in str(e):
                raise e
            if attempt == max_retries - 1:
                raise e
            print(
                f"\n[Gemini API] Connection error: {e}. Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
            )
            time.sleep(backoff)
            backoff *= 1.5

    raise Exception("Max retries exceeded for Gemini API")


def call_openai_api(api_key, base_url, model_name, image_bytes, max_retries=5, initial_backoff=4.0):
    """Evaluates the aesthetic quality and composition of an image using an OpenAI-compatible API.

    Args:
        api_key (str): Authentication API Key.
        base_url (str): Target base URL of the API endpoint.
        model_name (str): Model identifier.
        image_bytes (bytes): Binary data of the image.
        max_retries (int): Maximum retry attempts.
        initial_backoff (float): Initial wait time in seconds for backoff.

    Returns:
        dict: A dictionary containing 'score' (int 0-100) and 'reason' (str).
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = AESTHETIC_PROMPT

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                res_json = r.json()
                text_out = res_json["choices"][0]["message"]["content"]
                return json.loads(text_out)
            elif r.status_code == 429:
                print(
                    f"\n[OpenAI API] Rate limit hit (429). Details: {r.text[:300]}\nRetrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(backoff)
                backoff *= 2.0
            elif r.status_code >= 500:
                print(
                    f"\n[OpenAI API] Server error ({r.status_code}). Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(backoff)
                backoff *= 1.5
            else:
                raise Exception(f"OpenAI API error {r.status_code}: {r.text}")
        except Exception as e:
            if "OpenAI API error" in str(e):
                raise e
            if attempt == max_retries - 1:
                raise e
            print(
                f"\n[OpenAI API] Connection error: {e}. Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
            )
            time.sleep(backoff)
            backoff *= 1.5

    raise Exception("Max retries exceeded for OpenAI API")


# Local model state
local_model = None
local_processor = None
model_lock = threading.Lock()


def score_image_local(image_bytes, model_id):
    """Evaluates the aesthetic quality and composition of an image locally using a CLIP-based module.

    Pre-processes the image using CLIPProcessor, feeds it through RsinemaAestheticScorer,
    and returns a raw aesthetic score (typically between 0.0 and 10.0).

    Args:
        image_bytes (bytes): Binary JPEG image content.
        model_id (str): Hugging Face model identifier (e.g. 'rsinema/aesthetic-scorer').

    Returns:
        float: Aesthetic score.
    """
    global local_model, local_processor

    if torch is None or CLIPProcessor is None or RsinemaAestheticScorer is None:
        raise ImportError(
            "\n[Error] Local aesthetic scoring requires torch, torchvision, and transformers.\n"
            "Please install them inside the virtual environment by running:\n"
            "    .venv/bin/pip install torch torchvision transformers\n"
        )

    # Determine device (fallback to CPU if GPU compute capability is < 7.0 for PyTorch cu130 compatibility)
    device = "cpu"
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 7:
                device = "cuda"
        except Exception:
            pass

    with model_lock:
        if local_model is None:
            print(f"\nLoading local aesthetic model '{model_id}'...")
            if model_id == "rsinema/aesthetic-scorer":
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(repo_id=model_id, filename="model.pt")
                local_model = RsinemaAestheticScorer()
                local_model.load_state_dict(torch.load(model_path, map_location=device))
                local_model.to(device)
                local_model.eval()
                local_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            else:
                try:
                    from aesthetics_predictor import (
                        AestheticsPredictorV1,
                        AestheticsPredictorV2Linear,
                        AestheticsPredictorV2ReLU,
                    )
                except ImportError:
                    raise ImportError(
                        "\n[Error] This model requires simple-aesthetics-predictor.\n"
                        "Please install it inside the virtual environment by running:\n"
                        "    .venv/bin/pip install simple-aesthetics-predictor\n"
                    )

                if "v2" in model_id.lower() or "improved" in model_id.lower():
                    if "relu" in model_id.lower():
                        PredictorClass = AestheticsPredictorV2ReLU
                    else:
                        PredictorClass = AestheticsPredictorV2Linear
                else:
                    PredictorClass = AestheticsPredictorV1

                model_instance: Any = PredictorClass.from_pretrained(model_id)
                local_model = model_instance.to(device)
                local_processor = CLIPProcessor.from_pretrained(model_id)
                local_model.eval()
            print(f"Model loaded successfully on {device}!")

        # Load and process image inside the lock to ensure thread-safety on CPU
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            assert local_processor is not None, "Local processor is not initialized"
            assert local_model is not None, "Local model is not initialized"

            inputs = local_processor(images=image, return_tensors="pt")

            if model_id == "rsinema/aesthetic-scorer":
                pixel_values = inputs["pixel_values"].to(device)
                with torch.no_grad():
                    outputs = local_model(pixel_values)
                    # Extract Overall score (first dimension of output) and scale 0-5 -> 0-10
                    if len(outputs.shape) > 1:
                        raw_score = outputs[0][0].item()
                    else:
                        raw_score = outputs[0].item()
                    raw_score = raw_score * 2.0
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = local_model(**inputs)
                    raw_score = outputs.logits.item()

            # Scale to 0-100 using temporary linear math.
            # Scores will be dynamically calibrated at the end of the run
            # based on the empirical mean and std dev of the entire library.
            temp_score = int(round(raw_score * 10.0))

            return {
                "score": temp_score,
                "raw_score": raw_score,
                "reason": f"Local CLIP score: {raw_score:.2f}/10.0",
            }
        except Exception as e:
            print(f"Error in local scoring: {e}")
            raise e


def call_gemini_api_stage2(
    api_key, image_bytes, model_name="gemini-2.5-flash", max_retries=5, initial_backoff=4.0
):
    """Evaluates the aesthetic quality and composition of an image using Google Gemini API.

    Applies a prompt targeting overall aesthetics, composition, lighting, and highlight suitability.

    Args:
        api_key (str): Gemini API Key.
        image_bytes (bytes): Binary image content.
        model_name (str): Gemini model identifier.
        max_retries (int): Maximum retry attempts.
        initial_backoff (float): Initial wait time in seconds for exponential backoff.

    Returns:
        float: Aesthetic quality score (0.0 to 100.0).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": AESTHETIC_PROMPT},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "score": {
                        "type": "INTEGER",
                        "description": "A quality and aesthetic appeal score for the image from 0 to 100.",
                    },
                    "reason": {
                        "type": "STRING",
                        "description": "A brief one-sentence reason explaining the score.",
                    },
                },
                "required": ["score", "reason"],
            },
        },
    }

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                res_json = r.json()
                text_out = res_json["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(text_out)
                score = int(data["score"])
                reason = data.get("reason")
                return {"raw_score": float(score), "reason": reason}
            elif r.status_code == 429:
                print(
                    f"\n[Gemini API Stage 2] Rate limit hit (429). Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(backoff)
                backoff *= 2.0
            elif r.status_code >= 500:
                print(
                    f"\n[Gemini API Stage 2] Server error ({r.status_code}). Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(backoff)
                backoff *= 2.0
            else:
                raise Exception(f"Gemini API error {r.status_code}: {r.text}")
        except Exception as e:
            # Do not retry on permanent API errors (like 400, 403, 404, etc.)
            if "Gemini API error" in str(e):
                raise e
            if attempt == max_retries - 1:
                raise e
            print(
                f"\n[Gemini API Stage 2] Connection error: {e}. Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})..."
            )
            time.sleep(backoff)
            backoff *= 2.0

    raise RuntimeError("Failed to evaluate Stage 2 with Gemini API after multiple attempts.")


def call_openai_api_stage2(
    api_key, base_url, model_name, image_bytes, max_retries=5, initial_backoff=4.0
):
    """Evaluates the aesthetic quality and composition of an image using an OpenAI-compatible API.

    Applies a prompt targeting overall aesthetics, composition, lighting, and highlight suitability.

    Args:
        api_key (str): Authentication API Key.
        base_url (str): Target base URL of the API endpoint.
        model_name (str): Model identifier.
        image_bytes (bytes): Binary image content.
        max_retries (int): Maximum retry attempts.
        initial_backoff (float): Initial wait time in seconds for backoff.

    Returns:
        float: Aesthetic quality score (0.0 to 100.0).
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = AESTHETIC_PROMPT

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=45)
            if r.status_code == 200:
                res_json = r.json()
                text_out = res_json["choices"][0]["message"]["content"]
                data = json.loads(text_out)
                score = int(data["score"])
                reason = data.get("reason")
                return {"raw_score": float(score), "reason": reason}
            else:
                time.sleep(backoff)
                backoff *= 2.0
        except Exception:
            time.sleep(backoff)
            backoff *= 2.0

    raise RuntimeError("Failed to evaluate Stage 2 with OpenAI API after multiple attempts.")


# Stage 2 model state
local_stage2_model = None
stage2_model_lock = threading.Lock()


def score_image_stage2(
    image_bytes,
    model_name="musiq-spaq",
    gemini_key=None,
    gemini_model=None,
    openai_key=None,
    openai_url=None,
    openai_model=None,
):
    """Evaluates technical quality and sharpness of an image using a local MUSIQ model or a remote API.

    Args:
        image_bytes (bytes): Binary image content.
        model_name (str): Quality model identifier (e.g. 'musiq-spaq' or 'gemini').
        gemini_key (str, optional): Gemini API Key.
        gemini_model (str, optional): Gemini model configuration ID.
        openai_key (str, optional): OpenAI API Key.
        openai_url (str, optional): OpenAI base URL.
        openai_model (str, optional): OpenAI model configuration ID.

    Returns:
        float: Raw quality score.
    """
    global local_stage2_model
    import os
    import tempfile

    model_name_lower = model_name.lower()
    if "gemini" in model_name_lower:
        api_model = model_name if model_name != "gemini" else (gemini_model or "gemini-2.5-flash")
        if not gemini_key:
            raise ValueError(
                "Google Gemini API Key is required for Stage 2 scoring when using gemini model."
            )
        return call_gemini_api_stage2(gemini_key, image_bytes, api_model)
    elif "openai" in model_name_lower or "gpt" in model_name_lower:
        api_model = model_name if model_name != "openai" else (openai_model or "gpt-4o-mini")
        if not openai_key:
            raise ValueError(
                "OpenAI API Key is required for Stage 2 scoring when using openai model."
            )
        return call_openai_api_stage2(openai_key, openai_url, api_model, image_bytes)

    if pyiqa is None or torch is None:
        raise ImportError(
            "\n[Error] Stage 2 scoring requires the pyiqa library.\n"
            "Please install it inside the virtual environment by running:\n"
            "    .venv/bin/pip install pyiqa\n"
        )

    device = "cpu"
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 7:
                device = "cuda"
        except Exception:
            pass

    with stage2_model_lock:
        if local_stage2_model is None:
            print(f"\nLoading Stage 2 model '{model_name}' on {device}...")
            local_stage2_model = pyiqa.create_metric(model_name, device=device)
            print("Stage 2 model loaded successfully!")

        # Create temporary file to pass to pyiqa
        temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        try:
            with os.fdopen(temp_fd, "wb") as tmp:
                tmp.write(image_bytes)

            with torch.no_grad():
                score = local_stage2_model(temp_path)
                val = float(score.item())
                return {"raw_score": val, "reason": f"Local technical score: {val:.2f}"}
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass


def get_or_create_album(immich_url, api_key, album_name):
    """Retrieves an existing Immich album by name or creates a new one if missing.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        album_name (str): The target album name to find or create.

    Returns:
        str: UUID of the matched or newly created album.
    """
    headers = {"x-api-key": api_key}

    # List albums
    url = f"{immich_url}/api/albums"
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    albums = r.json()

    for album in albums:
        if album.get("albumName") == album_name:
            print(f"Using existing album '{album_name}' (ID: {album['id']})")
            return album["id"]

    # Create new
    print(f"Creating new album '{album_name}'...")
    payload = {"albumName": album_name}
    r = requests.post(url, json=payload, headers=headers, timeout=20)
    r.raise_for_status()
    new_album = r.json()
    print(f"Created album (ID: {new_album['id']})")
    return new_album["id"]


def add_assets_to_album(immich_url, api_key, album_id, asset_ids):
    """Registers a list of assets into an existing Immich album in chunked batches.

    Splits the assets into batches of 100 to prevent large request failures,
    deduplicating inputs within each batch.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        album_id (str): UUID of the target highlights album.
        asset_ids (list): List of asset UUIDs to add.
    """
    url = f"{immich_url}/api/albums/{album_id}/assets"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    chunk_size = 100
    for i in range(0, len(asset_ids), chunk_size):
        chunk = list(set(asset_ids[i : i + chunk_size]))  # deduplicate
        payload = {"ids": chunk}
        print(f"Adding chunk {i//chunk_size + 1} ({len(chunk)} items) to album...")
        r = requests.put(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()


def extract_raw_score_from_reason(reason):
    """Parses a local aesthetic CLIP score out of a cached reason explanation string.

    Enables reading legacy aesthetic scores saved in older cache versions
    without repeating local model evaluation.

    Args:
        reason (str): Reason explanation string from cache.

    Returns:
        float: Aesthetic score extracted (0.0 to 10.0), or None if parsing fails.
    """
    if not reason or "Local CLIP score:" not in reason:
        return None
    try:
        parts = reason.split("Local CLIP score:")
        if len(parts) > 1:
            sub = parts[1].split("/10.0")[0].strip()
            return float(sub)
    except Exception:
        pass
    return None


def score_to_stars(score):
    """Maps a standardized 0-100 composite aesthetic score into a 1-5 star rating.

    Args:
        score (float): Composite standard quality score.

    Returns:
        int: Star rating value (1, 2, 3, 4, or 5).
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


def update_asset_rating(immich_url, api_key, asset_id, rating):
    """Updates the native star rating metadata value of a specific asset on the Immich server.

    Args:
        immich_url (str): Immich server base URL.
        api_key (str): Immich API Key.
        asset_id (str): UUID of the target asset.
        rating (int): Rating value to write (1-5 stars).
    """
    url = f"{immich_url}/api/assets"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {"ids": [asset_id], "rating": rating}
    try:
        r = requests.put(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"\nWarning: Failed to update rating in Immich for asset {asset_id}: {e}")


def parse_asset_time(asset_item):
    """Parses chronological datetime from Immich asset metadata."""
    asset_info = asset_item.get("asset") or {}
    time_str = (
        asset_info.get("localDateTime")
        or asset_info.get("fileCreatedAt")
        or asset_info.get("createdAt")
    )
    if not time_str:
        return datetime.min
    try:
        cleaned_str = time_str.replace("Z", "")
        if "." in cleaned_str:
            base_part, ms_part = cleaned_str.split(".", 1)
            ms_part = (ms_part + "000000")[:6]
            cleaned_str = f"{base_part}.{ms_part}"
            return datetime.strptime(cleaned_str, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            return datetime.strptime(cleaned_str, "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return datetime.min


def deduplicate_bursts(scored_assets, dedup_window):
    """Filters out burst photos captured within a specific time window.

    Sorts the assets chronologically, groups them into time-based clusters,
    and preserves only the highest scoring asset from each cluster.
    """
    if dedup_window <= 0 or not scored_assets:
        return scored_assets

    # Sort assets chronologically to group them
    chrono_assets = []
    for item in scored_assets:
        chrono_assets.append((parse_asset_time(item), item))
    chrono_assets.sort(key=lambda x: x[0])

    deduped_assets = []
    current_group = []

    for dt, item in chrono_assets:
        if dt == datetime.min:
            # If timestamp parsing fails, keep it individually
            deduped_assets.append(item)
            continue

        if not current_group:
            current_group.append((dt, item))
        else:
            diff = (dt - current_group[0][0]).total_seconds()
            if diff <= dedup_window:
                current_group.append((dt, item))
            else:
                # Keep the highest scoring item from the group
                best_item = max(current_group, key=lambda x: x[1]["score"])[1]
                deduped_assets.append(best_item)
                current_group = [(dt, item)]

    if current_group:
        best_item = max(current_group, key=lambda x: x[1]["score"])[1]
        deduped_assets.append(best_item)

    return deduped_assets


def main():
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
        help="Hugging Face model ID for local scoring (default: shunk031/aesthetics-predictor-v1-vit-large-patch14)",
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
        help="Model ID for Stage 2 scoring (default: musiq-spaq)",
    )
    parser.add_argument(
        "--stage2-weight",
        type=float,
        help="Weight of Stage 2 score in the combined score (0.0 to 1.0, default: 0.5)",
    )

    args = parser.parse_args()

    # Load config file if it exists
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"Loaded configuration settings from '{args.config}'")
        except Exception as e:
            print(f"Warning: Failed to load config from {args.config}: {e}")

    def resolve(cli_val, env_name, config_key, default):
        # 1. CLI Argument
        if cli_val is not None:
            return cli_val
        # 2. Config JSON
        if config and config_key in config:
            cfg_val = config[config_key]
            if cfg_val is not None and (not isinstance(cfg_val, str) or str(cfg_val).strip() != ""):
                return cfg_val
        # 3. Environment Variable
        env_val = os.environ.get(env_name)
        if env_val is not None and env_val.strip() != "":
            return env_val
        # 4. Default
        return default

    # Resolve settings with CLI > Env > Config > Default precedence
    immich_url = resolve(args.immich_url, "IMMICH_URL", "immich_url", None)
    api_key = resolve(args.api_key, "IMMICH_API_KEY", "immich_api_key", None)
    person_id = resolve(args.person_id, "PERSON_ID", "person_id", None)
    album_id = resolve(args.album_id, "ALBUM_ID", "album_id", None)

    # CLI arguments override config file target modes
    if args.album_id is not None:
        person_id = None
    elif args.person_id is not None:
        album_id = None
    gemini_key = resolve(args.gemini_key, "GEMINI_API_KEY", "gemini_api_key", None)
    target_album_name = resolve(
        args.target_album_name, "TARGET_ALBUM_NAME", "target_album_name", None
    )
    scorer_type = resolve(args.scorer_type, "SCORER_TYPE", "scorer_type", "gemini").lower()
    local_model_id = resolve(
        args.local_model,
        "LOCAL_MODEL_ID",
        "local_model_id",
        "shunk031/aesthetics-predictor-v1-vit-large-patch14",
    )
    gemini_model = resolve(args.gemini_model, "GEMINI_MODEL", "gemini_model", "gemini-2.5-flash")
    openai_key = resolve(args.openai_key, "OPENAI_API_KEY", "openai_api_key", None)
    openai_url = resolve(
        args.openai_url, "OPENAI_BASE_URL", "openai_base_url", "https://api.openai.com/v1"
    )
    openai_model = resolve(args.openai_model, "OPENAI_MODEL", "openai_model", "gpt-4o-mini")
    write_ratings = (
        args.write_ratings
        if args.write_ratings is not None
        else (config.get("write_ratings", False) if config else False)
    )

    # Numeric values resolution
    limit_val = resolve(args.limit, "LIMIT", "limit", DEFAULT_LIMIT)
    try:
        limit = int(limit_val)
    except (ValueError, TypeError):
        limit = DEFAULT_LIMIT

    concurrency_val = resolve(args.concurrency, "CONCURRENCY", "concurrency", DEFAULT_CONCURRENCY)
    try:
        concurrency = int(concurrency_val)
    except (ValueError, TypeError):
        concurrency = DEFAULT_CONCURRENCY

    # Default local delay to 0.0 unless user explicitly requested a delay
    default_delay = 0.0 if scorer_type == "local" else DEFAULT_DELAY
    delay_val = resolve(args.delay, "DELAY", "delay", default_delay)
    try:
        delay = float(delay_val)
    except (ValueError, TypeError):
        delay = default_delay

    cache_file = resolve(args.cache_file, "CACHE_FILE", "cache_file", DEFAULT_CACHE_FILE)

    dedup_window_val = resolve(args.dedup_window, "DEDUP_WINDOW", "dedup_window", 0)
    try:
        dedup_window = int(dedup_window_val)
    except (ValueError, TypeError):
        dedup_window = 0

    # Flag values resolution
    dry_run = (
        args.dry_run
        if args.dry_run is not None
        else (config.get("dry_run", False) if config else False)
    )
    force_score = (
        args.force_score
        if args.force_score is not None
        else (config.get("force_score", False) if config else False)
    )
    use_cache_only = (
        args.use_cache_only
        if args.use_cache_only is not None
        else (config.get("use_cache_only", False) if config else False)
    )
    two_stage = (
        args.two_stage
        if args.two_stage is not None
        else (config.get("two_stage", False) if config else False)
    )

    stage2_top_pct_val = resolve(args.stage2_top_pct, "STAGE2_TOP_PCT", "stage2_top_pct", 15.0)
    try:
        stage2_top_pct = float(stage2_top_pct_val)
    except (ValueError, TypeError):
        stage2_top_pct = 15.0

    stage2_model = resolve(args.stage2_model, "STAGE2_MODEL", "stage2_model", "musiq-spaq")

    stage2_weight_val = resolve(args.stage2_weight, "STAGE2_WEIGHT", "stage2_weight", 0.5)
    try:
        stage2_weight = float(stage2_weight_val)
    except (ValueError, TypeError):
        stage2_weight = 0.5

    # Prompt user interactively if critical credentials are still missing
    print("Welcome to Immich Aesthetic Scorer!")
    if not immich_url:
        immich_url = get_or_prompt(
            "IMMICH_URL", "Enter Immich Server URL", default="http://localhost:2283"
        )
    immich_url = immich_url.rstrip("/")

    if not api_key:
        api_key = get_or_prompt("IMMICH_API_KEY", "Enter Immich API Key", secret=True)

    # Verify Immich Connection
    print("Verifying connection to Immich...")
    connected, major, minor = check_immich_connection(immich_url, api_key)
    if not connected:
        print(f"Error: Could not connect to Immich at {immich_url}. Error: {major}")
        sys.exit(1)
    print(f"Connected successfully! Immich Server Version: {major}.{minor}")

    # Resolve Person / Album Selection
    person_name = None
    album_name_source = None

    if person_id:
        person_details = get_person_details(immich_url, api_key, person_id)
        if person_details:
            person_name = person_details.get("name")
            print(f"Selected Person: {person_name or 'Unnamed'} (ID: {person_id})")
        else:
            print(
                f"Warning: Person ID {person_id} provided but could not fetch details from server."
            )
    elif album_id:
        album_details = get_album_details(immich_url, api_key, album_id)
        if album_details:
            album_name_source = album_details.get("albumName")
            print(f"Selected Source Album: {album_name_source} (ID: {album_id})")
        else:
            print(f"Warning: Album ID {album_id} provided but could not fetch details from server.")
    else:
        # Interactive mode: ask user if they want to process a Person or an existing Album
        print("\n--- Target Selection ---")
        print(" [1] Process photos of a Person (from facial recognition)")
        print(" [2] Process photos from an existing Album")
        target_choice = input("Select option [1-2, default: 1]: ").strip()
        if target_choice == "2":
            album_id, album_name_source = select_album_interactive(immich_url, api_key)
        else:
            person_id, person_name = select_person_interactive(immich_url, api_key)

    if not person_id and not album_id:
        print("Error: No person or album selected. Exiting.")
        sys.exit(1)

    # Resolve Remote Keys depending on scorer_type or stage2_model
    need_gemini = (scorer_type == "gemini") or ("gemini" in stage2_model.lower())
    need_openai = (scorer_type == "openai") or (
        "openai" in stage2_model.lower() or "gpt" in stage2_model.lower()
    )

    if need_gemini:
        if not gemini_key:
            # Fallback to default developer key from system memory if available, otherwise ask user
            gemini_key = get_or_prompt(
                "GEMINI_API_KEY",
                "Enter Google Gemini API Key",
                secret=True,
                default=DEFAULT_GEMINI_KEY,
            )
    if need_openai:
        if not openai_key:
            openai_key = get_or_prompt(
                "OPENAI_API_KEY", "Enter OpenAI (or compatible) API Key", secret=True
            )

    # Resolve Target Album Name
    if not target_album_name:
        if person_id:
            default_album = (
                f"Best of {person_name}" if person_name else f"Best of Person {person_id[:8]}"
            )
        else:
            default_album = (
                f"Best of Album {album_name_source}"
                if album_name_source
                else f"Best of Album {album_id[:8]}" if album_id else "Best of Album"
            )
        target_album_name = input(f"Enter target album name [{default_album}]: ").strip()
        if not target_album_name:
            target_album_name = default_album

    # Load Cache
    cache = load_cache(cache_file)
    print(f"Loaded {len(cache)} entries from cache file: {cache_file}")

    # Fetch all assets
    assets = fetch_all_image_assets(immich_url, api_key, person_id=person_id, album_id=album_id)
    target_label = "person" if person_id else "album"
    if not assets:
        print(f"No image assets found for the specified {target_label}.")
        sys.exit(0)
    print(f"Found {len(assets)} total image assets for the {target_label}.")

    # Identify Stage 1 model representation string
    if scorer_type == "local":
        s1_model_str = f"local model '{local_model_id}'"
        s1_model_name = local_model_id
    elif scorer_type == "gemini":
        s1_model_str = f"Gemini API model '{gemini_model}'"
        s1_model_name = gemini_model
    else:
        s1_model_str = f"OpenAI API model '{openai_model}'"
        s1_model_name = openai_model

    print(
        f"\n--- {COLOR_CYAN}Stage 1: Aesthetics Evaluation{COLOR_RESET} (using {COLOR_BOLD}{s1_model_str}{COLOR_RESET}) ---"
    )

    # Phase 1: Retrieve/Compute Stage 1 scores for all assets
    # We will build a list of dicts: {'id': asset_id, 'asset': asset, 'raw_score_stage1': ...}
    stage1_results = []
    assets_to_score_s1 = []

    for asset in assets:
        asset_id = asset["id"]
        cached_entry = cache.get(asset_id)
        mod_time = get_asset_mod_time(asset)

        # Check if the cached model matches the current model
        cached_model_s1 = cached_entry.get("model_id_stage1") if cached_entry else None
        model_mismatch = cached_model_s1 is not None and cached_model_s1 != s1_model_name

        if (
            not force_score
            and cached_entry
            and not model_mismatch
            and (
                "raw_score_stage1" in cached_entry
                or "raw_score" in cached_entry
                or "score" in cached_entry
            )
        ):
            cached_mod = cached_entry.get("updatedAt")
            if not cached_mod or cached_mod == mod_time:
                raw_s1 = cached_entry.get("raw_score_stage1")
                if raw_s1 is None:
                    raw_s1 = cached_entry.get("raw_score")
                if raw_s1 is None:
                    raw_s1 = extract_raw_score_from_reason(cached_entry.get("reason", ""))

                if raw_s1 is not None:
                    stage1_results.append(
                        {
                            "id": asset_id,
                            "asset": asset,
                            "raw_score_stage1": float(raw_s1),
                        }
                    )
                    continue

        assets_to_score_s1.append(asset)

    print(f"Stage 1 cached: {len(stage1_results)}")
    if use_cache_only:
        if assets_to_score_s1:
            print("Running in --use-cache-only mode. Skipping Stage 1 scoring of uncached assets.")
            assets_to_score_s1 = []
    else:
        print(f"Stage 1 needing scoring: {len(assets_to_score_s1)}")

    if assets_to_score_s1:
        print(
            f"Starting parallel Stage 1 scoring (using {s1_model_str}) with {concurrency} worker threads."
        )
        if scorer_type != "local":
            print(
                f"Injecting a {delay}s delay between requests per thread to respect API rate limits."
            )

        def process_s1_asset(asset_item):
            global last_gemini_call_time
            asset_id = asset_item["id"]
            try:
                # Download thumbnail (regular size: 512px)
                img_bytes = download_thumbnail(immich_url, api_key, asset_id, max_dim=512)

                if scorer_type == "local":
                    score_res = score_image_local(img_bytes, local_model_id)
                else:
                    with gemini_call_lock:
                        elapsed = time.time() - last_gemini_call_time
                        needed_delay = delay if delay is not None else 4.0
                        sleep_time = max(0.0, needed_delay - elapsed)
                        last_gemini_call_time = time.time() + sleep_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    if scorer_type == "gemini":
                        score_res = call_gemini_api(gemini_key, img_bytes, model_name=gemini_model)
                    else:
                        score_res = call_openai_api(openai_key, openai_url, openai_model, img_bytes)

                raw_s1 = score_res.get("raw_score")
                if raw_s1 is None:
                    raw_s1 = float(score_res.get("score", 0)) / 10.0

                # Save to cache
                mod_time = get_asset_mod_time(asset_item)
                cache_entry = cache.get(asset_id, {})
                cache_entry.update(
                    {
                        "raw_score_stage1": raw_s1,
                        "model_id_stage1": s1_model_name,
                        "updatedAt": mod_time,
                    }
                )
                update_cache_entry_threadsafe(asset_id, cache_entry)

                if scorer_type == "local" and delay > 0:
                    time.sleep(delay)

                return {
                    "id": asset_id,
                    "raw_score_stage1": raw_s1,
                    "asset": asset_item,
                    "status": "success",
                }
            except Exception as e:
                return {"id": asset_id, "status": "error", "error": str(e), "asset": asset_item}

        def update_cache_entry_threadsafe(key, val):
            with cache_lock:
                cache[key] = val

        failed_count_s1 = 0
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(process_s1_asset, item): item for item in assets_to_score_s1
                }
                pbar = tqdm(as_completed(futures), total=len(futures), desc="Stage 1 Scoring")
                for future in pbar:
                    item_res = future.result()
                    if item_res["status"] == "success":
                        stage1_results.append(item_res)
                        filename = item_res["asset"].get("originalFileName") or item_res["id"]
                        pbar.write(
                            f"Stage 1 Scored {COLOR_BLUE}{filename}{COLOR_RESET} using '{COLOR_BOLD}{s1_model_name}{COLOR_RESET}': Raw Score {COLOR_GREEN}{item_res['raw_score_stage1']:.2f}{COLOR_RESET}"
                        )
                    else:
                        failed_count_s1 += 1
                        pbar.write(
                            f"Warning: Failed to score asset {item_res['id']} in Stage 1: {item_res['error']}"
                        )
        finally:
            save_cache(cache, cache_file)

        if failed_count_s1 > 0:
            print(f"\nCompleted Stage 1 with {failed_count_s1} failures.")

    # Calculate normalized Stage 1 scores
    raw_s1_list = [
        item["raw_score_stage1"]
        for item in stage1_results
        if item.get("raw_score_stage1") is not None
    ]
    if len(raw_s1_list) > 1:
        mean_s1 = sum(raw_s1_list) / len(raw_s1_list)
        variance_s1 = sum((x - mean_s1) ** 2 for x in raw_s1_list) / len(raw_s1_list)
        std_s1 = math.sqrt(variance_s1)
        if std_s1 < 0.01:
            std_s1 = 0.6
    else:
        mean_s1 = 6.0
        std_s1 = 1.0

    for item in stage1_results:
        raw_s1 = item["raw_score_stage1"]
        z1 = (raw_s1 - mean_s1) / std_s1
        s1_norm = 100.0 / (1.0 + math.exp(-1.5 * z1))
        item["s1_norm"] = min(100.0, max(0.0, s1_norm))

    scored_assets = []

    if not two_stage:
        # Single-stage path: use Stage 1 normalized score as final score
        print("\nCalibrating aesthetic scores based on library distribution...")
        print("Library Statistics:")
        print(f"  - Total Scored Photos: {len(raw_s1_list)}")
        print(f"  - Raw Score Mean: {mean_s1:.3f}")
        print(f"  - Raw Score Std Dev: {std_s1:.3f}")
        print("Applying sigmoid normalization and updating cache...")

        # Re-read cache to update it on disk
        cache = load_cache(cache_file)
        for item in stage1_results:
            raw_s1 = item["raw_score_stage1"]
            z1 = (raw_s1 - mean_s1) / std_s1
            final_score = int(round(item["s1_norm"]))
            reason = f"Local CLIP score: {raw_s1:.2f}/10.0 (z-score: {z1:.2f})"

            # Update cache entry
            asset_id = item["id"]
            cache_entry = cache.get(asset_id, {})
            cache_entry.update(
                {
                    "score": final_score,
                    "raw_score": raw_s1,
                    "raw_score_stage1": raw_s1,
                    "model_id_stage1": s1_model_name,
                    "reason": reason,
                }
            )
            cache[asset_id] = cache_entry

            scored_assets.append(
                {
                    "id": asset_id,
                    "score": final_score,
                    "raw_score": raw_s1,
                    "reason": reason,
                    "asset": item["asset"],
                }
            )
        save_cache(cache, cache_file)
        print("Cache updated successfully!")

    else:
        # Compute stage2_top_n dynamically as a percentage of the total Stage 1 results
        stage2_top_n = max(1, int(len(stage1_results) * (stage2_top_pct / 100.0)))

        # Two-stage path: score top candidates with Stage 2 model, then combine
        print(
            f"\n--- {COLOR_MAGENTA}Stage 2: Technical Re-ranking{COLOR_RESET} (selecting top {stage2_top_n} candidates ({stage2_top_pct}%) using model '{COLOR_BOLD}{stage2_model}{COLOR_RESET}') ---"
        )
        stage1_results.sort(key=lambda x: x["s1_norm"], reverse=True)
        candidates_for_stage2 = stage1_results[:stage2_top_n]

        stage2_results = []
        assets_to_score_s2 = []

        cache = load_cache(cache_file)
        for item in candidates_for_stage2:
            asset_id = item["id"]
            cached_entry = cache.get(asset_id)

            # Check if cached Stage 2 model matches the current configured model
            cached_model_s2 = cached_entry.get("model_id_stage2") if cached_entry else None
            model_mismatch_s2 = cached_model_s2 is not None and cached_model_s2 != stage2_model

            if (
                not force_score
                and cached_entry
                and not model_mismatch_s2
                and "raw_score_stage2" in cached_entry
            ):
                raw_s2 = cached_entry["raw_score_stage2"]
                if raw_s2 is not None:
                    item["raw_score_stage2"] = float(raw_s2)
                    if "reason_stage2" in cached_entry:
                        item["reason_stage2"] = cached_entry["reason_stage2"]
                    stage2_results.append(item)
                    continue

            assets_to_score_s2.append(item)

        print(f"Stage 2 cached: {len(stage2_results)}")
        if use_cache_only:
            if assets_to_score_s2:
                print(
                    "Running in --use-cache-only mode. Skipping Stage 2 scoring of uncached assets."
                )
                assets_to_score_s2 = []
        else:
            print(f"Stage 2 needing scoring: {len(assets_to_score_s2)}")

        if assets_to_score_s2:
            print(
                f"Starting parallel Stage 2 scoring (using model '{stage2_model}') with {concurrency} worker threads."
            )

            def process_s2_asset(item):
                global last_gemini_call_time
                asset_id = item["id"]
                try:
                    # For non-local models (Gemini/OpenAI), download/downscale to 512px thumbnail to save costs.
                    # For local models, download full high-res preview thumbnail (max_dim=None).
                    is_local_s2 = not (
                        "gemini" in stage2_model.lower()
                        or "openai" in stage2_model.lower()
                        or "gpt" in stage2_model.lower()
                    )
                    max_dim_s2 = None if is_local_s2 else 512
                    img_bytes = download_thumbnail(
                        immich_url, api_key, asset_id, max_dim=max_dim_s2
                    )

                    if not is_local_s2:
                        with gemini_call_lock:
                            elapsed = time.time() - last_gemini_call_time
                            needed_delay = delay if delay is not None else 4.0
                            sleep_time = max(0.0, needed_delay - elapsed)
                            last_gemini_call_time = time.time() + sleep_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                    # Score using Stage 2 model (like MUSIQ)
                    score_res = score_image_stage2(
                        img_bytes,
                        stage2_model,
                        gemini_key=gemini_key,
                        gemini_model=gemini_model,
                        openai_key=openai_key,
                        openai_url=openai_url,
                        openai_model=openai_model,
                    )
                    raw_s2 = score_res["raw_score"]
                    reason_s2 = score_res.get("reason")

                    # Save to cache
                    cache_entry = cache.get(asset_id, {})
                    cache_entry.update(
                        {
                            "raw_score_stage2": raw_s2,
                            "model_id_stage2": stage2_model,
                            "reason_stage2": reason_s2,
                        }
                    )
                    with cache_lock:
                        cache[asset_id] = cache_entry

                    return {
                        "id": asset_id,
                        "raw_score_stage1": item["raw_score_stage1"],
                        "raw_score_stage2": raw_s2,
                        "reason_stage2": reason_s2,
                        "status": "success",
                    }
                except Exception as e:
                    return {"id": asset_id, "status": "error", "error": str(e)}

            failed_count_s2 = 0
            try:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {
                        executor.submit(process_s2_asset, item): item for item in assets_to_score_s2
                    }
                    pbar = tqdm(as_completed(futures), total=len(futures), desc="Stage 2 Scoring")
                    for future in pbar:
                        item_res = future.result()
                        if item_res["status"] == "success":
                            # Update the item in candidates_for_stage2
                            for c in candidates_for_stage2:
                                if c["id"] == item_res["id"]:
                                    c["raw_score_stage2"] = item_res["raw_score_stage2"]
                                    c["reason_stage2"] = item_res.get("reason_stage2")
                                    stage2_results.append(c)
                                    break
                            filename = c["asset"].get("originalFileName") or c["id"]
                            pbar.write(
                                f"Stage 2 Scored {COLOR_BLUE}{filename}{COLOR_RESET} using '{COLOR_BOLD}{stage2_model}{COLOR_RESET}': Raw Score {COLOR_GREEN}{item_res['raw_score_stage2']:.2f}{COLOR_RESET} | Stage 1 Raw: {COLOR_YELLOW}{item_res['raw_score_stage1']:.2f}{COLOR_RESET}"
                            )
                        else:
                            failed_count_s2 += 1
                            pbar.write(
                                f"Warning: Failed to score asset {item_res['id']} in Stage 2: {item_res['error']}"
                            )
            finally:
                save_cache(cache, cache_file)

            if failed_count_s2 > 0:
                print(f"\nCompleted Stage 2 with {failed_count_s2} failures.")

        # Calculate normalized Stage 2 scores over all successfully scored candidates
        raw_s2_list = [
            item["raw_score_stage2"]
            for item in stage2_results
            if item.get("raw_score_stage2") is not None
        ]
        if len(raw_s2_list) > 1:
            mean_s2 = sum(raw_s2_list) / len(raw_s2_list)
            variance_s2 = sum((x - mean_s2) ** 2 for x in raw_s2_list) / len(raw_s2_list)
            std_s2 = math.sqrt(variance_s2)
            if std_s2 < 0.01:
                std_s2 = 1.0
        else:
            mean_s2 = 50.0
            std_s2 = 10.0

        print("Stage 2 Statistics:")
        print(f"  - Total Scored Candidates: {len(raw_s2_list)}")
        print(f"  - Raw Score Mean: {mean_s2:.3f}")
        print(f"  - Raw Score Std Dev: {std_s2:.3f}")

        # Combine Stage 1 & Stage 2 scores
        print("Combining Stage 1 (Aesthetic) and Stage 2 (Technical) scores...")
        cache = load_cache(cache_file)

        for item in stage1_results:
            asset_id = item["id"]
            raw_s1 = item["raw_score_stage1"]
            s1_norm = item["s1_norm"]

            if "raw_score_stage2" in item:
                raw_s2 = item["raw_score_stage2"]
                z2 = (raw_s2 - mean_s2) / std_s2
                s2_norm = 100.0 / (1.0 + math.exp(-1.5 * z2))
                s2_norm = min(100.0, max(0.0, s2_norm))

                combined_score = (1.0 - stage2_weight) * s1_norm + stage2_weight * s2_norm
                final_score = int(round(combined_score))

                z1 = (raw_s1 - mean_s1) / std_s1
                reason = (
                    f"Two-stage: S1={s1_norm:.1f} (raw: {raw_s1:.2f}, z: {z1:.2f}), "
                    f"S2={s2_norm:.1f} (raw: {raw_s2:.2f}, z: {z2:.2f})"
                )
            else:
                # Assume average technical quality (S2 = 50.0) for non-candidates
                combined_score = (1.0 - stage2_weight) * s1_norm + stage2_weight * 50.0
                final_score = int(round(combined_score))
                reason = f"Stage 1 only: Aesthetics={s1_norm:.1f} (raw: {raw_s1:.2f}, assumed average S2)"
                raw_s2 = None

            # Update cache entry
            cache_entry = cache.get(asset_id, {})
            reason_s2 = item.get("reason_stage2") or cache_entry.get("reason_stage2")
            cache_entry.update(
                {
                    "score": final_score,
                    "raw_score": raw_s1,
                    "raw_score_stage1": raw_s1,
                    "model_id_stage1": s1_model_name,
                    "raw_score_stage2": raw_s2,
                    "model_id_stage2": stage2_model if raw_s2 is not None else None,
                    "reason_stage2": reason_s2,
                    "reason": reason,
                }
            )
            cache[asset_id] = cache_entry

            scored_assets.append(
                {
                    "id": asset_id,
                    "score": final_score,
                    "raw_score": raw_s1,
                    "reason": reason,
                    "asset": item["asset"],
                }
            )

        save_cache(cache, cache_file)
        print("Cache updated successfully with combined scores!")

    # Update star ratings in Immich in parallel if requested
    if write_ratings:
        print(
            f"Syncing calibrated star ratings (1-5) to Immich in parallel using {concurrency} threads..."
        )

        def update_rating_worker(asset_res):
            try:
                stars = score_to_stars(asset_res["score"])
                update_asset_rating(immich_url, api_key, asset_res["id"], stars)
                return True
            except Exception as e:
                print(f"Warning: Failed to update rating for asset {asset_res['id']}: {e}")
                return False

        with ThreadPoolExecutor(max_workers=concurrency) as rating_executor:
            rating_futures = [
                rating_executor.submit(update_rating_worker, res) for res in scored_assets
            ]
            rating_pbar = tqdm(
                as_completed(rating_futures),
                total=len(rating_futures),
                desc="Syncing Ratings",
            )
            for f in rating_pbar:
                f.result()
        print("Star ratings sync completed!")

    # Deduplicate burst photos if window is configured
    if dedup_window > 0:
        print(f"\nDeduplicating burst photos within a {dedup_window}-second window...")
        original_count = len(scored_assets)
        scored_assets = deduplicate_bursts(scored_assets, dedup_window)
        print(
            f"Deduplicated burst photos: Kept {len(scored_assets)} unique highlights out of {original_count} scored photos (Filtered out {original_count - len(scored_assets)} burst duplicates)."
        )

    # Sort assets by score descending
    scored_assets.sort(key=lambda x: x["score"], reverse=True)

    # Display top 10 scoring images
    print("\n--- TOP 10 HIGHEST SCORING IMAGES ---")
    for idx, item in enumerate(scored_assets[:10], 1):
        asset_info = item["asset"]
        orig_name = asset_info.get("originalFileName", "Unknown")
        created_at = asset_info.get("fileCreatedAt", "Unknown Date")[:10]
        print(
            f"{idx:2d}. Score: {item['score']:3d} | File: {orig_name} ({created_at}) | Reason: {item['reason']}"
        )

    # Get top 100 (or specified limit)
    limit = min(limit, len(scored_assets))
    top_assets = scored_assets[:limit]

    print(
        f"\nSelecting the {limit} highest scoring images (Minimum score in top selection: {top_assets[-1]['score'] if top_assets else 0})."
    )

    if dry_run:
        print("\n[Dry Run] skipping album creation and asset addition.")
        sys.exit(0)

    # Add to album
    try:
        album_id_target = get_or_create_album(immich_url, api_key, target_album_name)
        top_ids = [item["id"] for item in top_assets]
        add_assets_to_album(immich_url, api_key, album_id_target, top_ids)
        print(
            f"\nSuccess! Successfully added the top {len(top_ids)} photos to album '{target_album_name}'."
        )
    except Exception as e:
        print(f"\nError managing album/assets: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cache has been saved. Exiting.")
        sys.exit(1)
