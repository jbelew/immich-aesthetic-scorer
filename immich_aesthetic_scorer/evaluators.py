"""Unified image evaluation engine and model adapters."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Union, runtime_checkable

import requests
from PIL import Image

torch: Any = None
nn: Any = None
CLIPProcessor: Any = None
CLIPVisionModel: Any = None
pyiqa: Any = None

try:
    import torch

    torch.set_num_threads(1)
    import torch.nn as nn
    from transformers import CLIPProcessor, CLIPVisionModel
except ImportError:
    pass

BaseModule: Any = nn.Module if nn is not None else object


class RsinemaAestheticScorer(BaseModule):
    """PyTorch Module wrapping CLIP Vision model with multiple aesthetic classification heads."""

    def __init__(self) -> None:
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

    def forward(self, pixel_values: Any) -> Any:
        """Computes 7-attribute aesthetic and quality predictions from CLIP vision features.

        Args:
            pixel_values: Preprocessed tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Tensor of shape (batch_size, 7) where index 0 is overall aesthetics (0.0–5.0).
        """
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


def negotiate_torch_device(torch_module: Any = None) -> str:
    """Negotiates PyTorch compute device with CUDA capability floor.

    PyTorch cu130 binaries require CUDA compute capability >= 7.0 (Volta sm_70+).
    Older GPUs or driver query errors gracefully fall back to CPU.

    Args:
        torch_module: Optional injected torch module for testing.

    Returns:
        Device identifier string ('cuda' or 'cpu').
    """
    t = torch_module if torch_module is not None else torch
    if t is None:
        return "cpu"
    try:
        if t.cuda.is_available():
            major, _ = t.cuda.get_device_capability(0)
            if major >= 7:
                return "cuda"
    except Exception:
        pass
    return "cpu"


def resize_image_bytes(image_bytes: bytes, max_dim: Optional[int] = 512) -> bytes:
    """Resizes image bytes proportionally so the longest edge does not exceed `max_dim`.

    Args:
        image_bytes: Raw encoded image bytes (JPEG/PNG).
        max_dim: Maximum pixel dimension on the longest edge (`512` for coarse/remote LLM
            scoring; `None` or `<= 0` returns `image_bytes` unchanged).

    Returns:
        Encoded image bytes (re-compressed at quality 85 if resized).
    """
    if max_dim is None or max_dim <= 0:
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
        orig_format = img.format or "JPEG"

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

            out_bytes = io.BytesIO()
            img.save(out_bytes, format=orig_format, quality=85)
            return out_bytes.getvalue()
    except Exception as exc:
        print(f"Warning: Failed to resize thumbnail client-side: {exc}")

    return image_bytes


@dataclass(frozen=True)
class EvaluationResult:
    """Result of an image aesthetic or technical quality evaluation.

    Attributes:
        raw_score: Uncalibrated model output (0.0–10.0 for Stage 1, 0.0–100.0 for Stage 2).
        reason: Human-readable explanation or formatted raw score string.
        score: Optional 0–100 integer score for single-stage compatibility.
    """

    raw_score: float
    reason: str | None = None
    score: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts result to legacy dictionary shape."""
        result: dict[str, Any] = {
            "raw_score": self.raw_score,
            "reason": self.reason,
        }
        if self.score is not None:
            result["score"] = self.score
        return result


@runtime_checkable
class ImageEvaluator(Protocol):
    """Unified evaluation seam for Stage 1 and Stage 2 image quality evaluators."""

    @property
    def model_id(self) -> str:
        """Configured model identifier used for cache invalidation."""
        ...

    @property
    def required_max_dimension(self) -> int | None:
        """Maximum thumbnail dimension in pixels (512 for coarse/remote, None for full-res IQA)."""
        ...

    def evaluate(self, image_bytes: bytes) -> EvaluationResult:
        """Evaluates raw image bytes and returns an EvaluationResult."""
        ...


FakeScoreSource = Optional[
    Union[
        Mapping[bytes, Union[float, EvaluationResult]],
        Callable[[bytes], Union[float, EvaluationResult]],
    ]
]


class FakeEvaluator:
    """Deterministic in-memory evaluator adapter for unit and integration testing."""

    def __init__(
        self,
        scores: FakeScoreSource = None,
        default_score: float = 7.0,
        model_id: str = "fake-evaluator",
        required_max_dimension: int | None = 512,
    ) -> None:
        self._scores = scores
        self._default_score = float(default_score)
        self._model_id = model_id
        self._required_max_dimension = required_max_dimension
        self._lock = threading.Lock()
        self.evaluated_images: list[bytes] = []

    @property
    def model_id(self) -> str:
        """Returns the fake model identifier."""
        return self._model_id

    @property
    def required_max_dimension(self) -> int | None:
        """Returns the configured maximum thumbnail dimension."""
        return self._required_max_dimension

    def evaluate(self, image_bytes: bytes) -> EvaluationResult:
        """Records image_bytes thread-safely and returns a deterministic EvaluationResult."""
        with self._lock:
            self.evaluated_images.append(image_bytes)

        val: float | EvaluationResult = self._default_score
        if callable(self._scores):
            val = self._scores(image_bytes)
        elif isinstance(self._scores, Mapping) and image_bytes in self._scores:
            val = self._scores[image_bytes]

        if isinstance(val, EvaluationResult):
            return val

        raw = float(val)
        # Scale 0-10 scores to 0-100 integer range to mirror local CLIP temporary scaling
        scaled = int(round(raw * 10.0)) if raw <= 10.0 else int(round(raw))
        return EvaluationResult(
            raw_score=raw,
            reason=f"Fake score: {raw:.2f}",
            score=scaled,
        )


class LocalClipEvaluator:
    """Local PyTorch adapter for CLIP, SigLIP, and simple-aesthetics-predictor backbones."""

    def __init__(
        self,
        model_id: str = "somepago/AestheticSigLIP",
        required_max_dimension: int | None = 512,
    ) -> None:
        self._model_id = model_id
        self._required_max_dimension = required_max_dimension
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cpu"
        self._infer_fn: Callable[[Any], float] | None = None

    @property
    def model_id(self) -> str:
        """Returns the Hugging Face model repository identifier."""
        return self._model_id

    @property
    def required_max_dimension(self) -> int | None:
        """Returns 512px default thumbnail ceiling for CLIP/SigLIP vision backbones."""
        return self._required_max_dimension

    def _infer_rsinema(self, image: Any) -> float:
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        outputs = self._model(pixel_values)
        if len(outputs.shape) > 1:
            raw_score = float(outputs[0][0].item())
        else:
            raw_score = float(outputs[0].item())
        # Rsinema overall head outputs 0.0-5.0; double to align with the 0.0-10.0 Stage 1 scale
        return raw_score * 2.0

    def _infer_siglip(self, image: Any) -> float:
        return float(self._model.rate(image))

    def _infer_predictor(self, image: Any) -> float:
        inputs = self._processor(images=image, return_tensors="pt")
        device_inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**device_inputs)
        return float(outputs.logits.item())

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._infer_fn is not None:
            return

        if torch is None or CLIPProcessor is None or RsinemaAestheticScorer is None:
            raise ImportError(
                "\n[Error] Local aesthetic scoring requires torch, torchvision, and transformers.\n"
                "Please install them inside the virtual environment by running:\n"
                "    .venv/bin/pip install torch torchvision transformers\n"
            )

        self._device = negotiate_torch_device(torch)
        model_id_lower = self._model_id.lower()

        if self._model_id == "rsinema/aesthetic-scorer":
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo_id=self._model_id, filename="model.pt")
            model = RsinemaAestheticScorer()
            model.load_state_dict(torch.load(model_path, map_location=self._device))
            model.to(self._device)
            model.eval()
            self._model = model
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._infer_fn = self._infer_rsinema
        elif "aestheticsiglip" in model_id_lower or "siglip" in model_id_lower:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(repo_id=self._model_id)
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)

            from predict import AestheticScorer  # type: ignore

            self._model = AestheticScorer.from_local(
                os.path.join(model_dir, "best.pt"), device=self._device
            )
            self._infer_fn = self._infer_siglip
        else:
            try:
                from aesthetics_predictor import (  # type: ignore
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

            if "v2" in model_id_lower or "improved" in model_id_lower:
                if "relu" in model_id_lower:
                    predictor_cls = AestheticsPredictorV2ReLU
                else:
                    predictor_cls = AestheticsPredictorV2Linear
            else:
                predictor_cls = AestheticsPredictorV1

            model_instance: Any = predictor_cls.from_pretrained(self._model_id)
            self._model = model_instance.to(self._device)
            self._processor = CLIPProcessor.from_pretrained(self._model_id)
            self._model.eval()
            self._infer_fn = self._infer_predictor

    def evaluate(self, image_bytes: bytes) -> EvaluationResult:
        """Evaluates image_bytes locally under a thread lock and torch.no_grad().

        Args:
            image_bytes: Encoded image bytes (JPEG/PNG).

        Returns:
            EvaluationResult with raw_score on 0.0–10.0 scale and preliminary 0–100 score.
        """
        with self._lock:
            self._ensure_loaded()
            assert self._infer_fn is not None
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            with torch.no_grad():
                raw_score = self._infer_fn(image)

            temp_score = int(round(raw_score * 10.0))
            return EvaluationResult(
                raw_score=raw_score,
                reason=f"Local CLIP score: {raw_score:.2f}/10.0",
                score=temp_score,
            )


class PyIQAEvaluator:
    """Local PyIQA technical quality adapter evaluating PIL images in-memory without temp file leaks."""

    def __init__(
        self,
        model_id: str = "musiq-spaq",
        required_max_dimension: int | None = None,
    ) -> None:
        self._model_id = model_id
        self._required_max_dimension = required_max_dimension
        self._lock = threading.Lock()
        self._metric: Any = None
        self._device: str = "cpu"

    @property
    def model_id(self) -> str:
        """Returns the PyIQA metric name (e.g. 'musiq-spaq')."""
        return self._model_id

    @property
    def required_max_dimension(self) -> int | None:
        """Returns None so full-resolution preview thumbnails are preserved for sharpness analysis."""
        return self._required_max_dimension

    def _ensure_loaded(self) -> None:
        global pyiqa
        if self._metric is not None:
            return

        if pyiqa is None:
            try:
                import pyiqa as _pyiqa

                pyiqa = _pyiqa
            except ImportError:
                pass

        if pyiqa is None or torch is None:
            raise ImportError(
                "\n[Error] Stage 2 scoring requires the pyiqa library.\n"
                "Please install it inside the virtual environment by running:\n"
                "    .venv/bin/pip install pyiqa\n"
            )

        self._device = negotiate_torch_device(torch)
        self._metric = pyiqa.create_metric(self._model_id, device=self._device)

    def evaluate(self, image_bytes: bytes) -> EvaluationResult:
        """Evaluates technical image quality in-memory using a PIL RGB Image under torch.no_grad().

        Args:
            image_bytes: Unscaled preview image bytes.

        Returns:
            EvaluationResult containing raw technical quality score.
        """
        with self._lock:
            self._ensure_loaded()
            # Decode directly into an in-memory PIL Image to avoid filesystem temp file I/O
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            with torch.no_grad():
                score = self._metric(image)
                val = float(score.item() if hasattr(score, "item") else score)
                return EvaluationResult(
                    raw_score=val,
                    reason=f"Local technical score: {val:.2f}",
                )


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


class RateLimiter:
    """Thread-safe token pacing rate limiter for remote API calls."""

    def __init__(
        self,
        delay: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.delay = max(0.0, float(delay))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_call_time: float | None = None

    def wait(self) -> None:
        """Reserves the next time slot thread-safely and sleeps outside the lock if needed."""
        if self.delay <= 0.0:
            return
        with self._lock:
            now = self._clock()
            if self._last_call_time is None:
                self._last_call_time = now
                sleep_time = 0.0
            else:
                # Advance _last_call_time inside the lock so concurrent callers queue into distinct slots
                elapsed = now - self._last_call_time
                sleep_time = max(0.0, self.delay - elapsed)
                self._last_call_time = now + sleep_time
        if sleep_time > 0.0:
            self._sleep(sleep_time)


class _NonRetryableAPIError(RuntimeError):
    """Internal exception raised on permanent 4xx remote API errors."""


class _BaseRemoteVisionEvaluator:
    """Shared transport, rate pacing, and exponential retry backoff for remote vision endpoints."""

    provider_name: str = "Remote API"

    def __init__(
        self,
        api_key: str,
        model_id: str,
        delay: float = 0.0,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        initial_backoff: float = 4.0,
        required_max_dimension: int | None = 512,
        raw_scale: float = 1.0,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        prompt: str = AESTHETIC_PROMPT,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._rate_limiter = rate_limiter or RateLimiter(delay=delay, sleep=sleep)
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._required_max_dimension = required_max_dimension
        self._raw_scale = raw_scale
        self._session = session if session is not None else requests
        self._sleep = sleep
        self._prompt = prompt

    @property
    def model_id(self) -> str:
        """Returns the remote vision model identifier."""
        return self._model_id

    @property
    def required_max_dimension(self) -> int | None:
        """Returns 512px thumbnail limit to minimize vision token usage."""
        return self._required_max_dimension

    def _build_request(self, image_b64: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        raise NotImplementedError

    def _extract_json_text(self, response_json: dict[str, Any]) -> str:
        raise NotImplementedError

    def evaluate(self, image_bytes: bytes) -> EvaluationResult:
        """Paces the request, posts base64 JPEG payload with exponential backoff, and parses JSON score.

        Args:
            image_bytes: JPEG image bytes (typically downscaled to <= 512px).

        Returns:
            EvaluationResult with raw_score scaled by raw_scale (0.1 for Stage 1, 1.0 for Stage 2).

        Raises:
            RuntimeError: On permanent 4xx errors or when max_retries are exhausted on 429/5xx.
        """
        self._rate_limiter.wait()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        url, headers, payload = self._build_request(image_b64)

        backoff = float(self._initial_backoff)
        for attempt in range(self._max_retries):
            try:
                r = self._session.post(url, json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    text_out = self._extract_json_text(r.json())
                    data = json.loads(text_out)
                    score = int(data["score"])
                    reason = data.get("reason")
                    return EvaluationResult(
                        raw_score=float(score) * self._raw_scale,
                        reason=str(reason) if reason is not None else None,
                        score=score,
                    )
                if r.status_code == 429 or r.status_code >= 500:
                    if attempt == self._max_retries - 1:
                        raise RuntimeError(
                            f"{self.provider_name} error {r.status_code} after {self._max_retries} attempts: {r.text}"
                        )
                    self._sleep(backoff)
                    backoff *= 2.0
                else:
                    raise _NonRetryableAPIError(
                        f"{self.provider_name} error {r.status_code}: {r.text}"
                    )
            except _NonRetryableAPIError:
                raise
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                self._sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"Max retries exceeded for {self.provider_name}")


class GeminiEvaluator(_BaseRemoteVisionEvaluator):
    """Remote Google Gemini vision API adapter with rate pacing and exponential retry backoff."""

    provider_name = "Gemini API"

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.5-flash",
        delay: float = 0.0,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        initial_backoff: float = 4.0,
        required_max_dimension: int | None = 512,
        raw_scale: float = 1.0,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        prompt: str = AESTHETIC_PROMPT,
    ) -> None:
        if not api_key:
            raise ValueError("Google Gemini API Key is required for GeminiEvaluator.")
        super().__init__(
            api_key=api_key,
            model_id=model_id,
            delay=delay,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            required_max_dimension=required_max_dimension,
            raw_scale=raw_scale,
            session=session,
            sleep=sleep,
            prompt=prompt,
        )

    def _build_request(self, image_b64: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model_id}:generateContent?key={self._api_key}"
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self._prompt},
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
        return url, headers, payload

    def _extract_json_text(self, response_json: dict[str, Any]) -> str:
        return str(response_json["candidates"][0]["content"]["parts"][0]["text"])


class OpenAIEvaluator(_BaseRemoteVisionEvaluator):
    """Remote OpenAI-compatible vision API adapter with rate pacing and exponential retry backoff."""

    provider_name = "OpenAI API"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_id: str = "gpt-4o-mini",
        delay: float = 0.0,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        initial_backoff: float = 4.0,
        required_max_dimension: int | None = 512,
        raw_scale: float = 1.0,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        prompt: str = AESTHETIC_PROMPT,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAIEvaluator.")
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        super().__init__(
            api_key=api_key,
            model_id=model_id,
            delay=delay,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            required_max_dimension=required_max_dimension,
            raw_scale=raw_scale,
            session=session,
            sleep=sleep,
            prompt=prompt,
        )

    def _build_request(self, image_b64: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        return url, headers, payload

    def _extract_json_text(self, response_json: dict[str, Any]) -> str:
        return str(response_json["choices"][0]["message"]["content"])


def create_evaluator(
    scorer_type: str = "gemini",
    *,
    stage: int = 1,
    model_id: str | None = None,
    local_model_id: str = "somepago/AestheticSigLIP",
    gemini_key: str | None = None,
    gemini_model: str = "gemini-2.5-flash",
    openai_key: str | None = None,
    openai_url: str = "https://api.openai.com/v1",
    openai_model: str = "gpt-4o-mini",
    stage2_model: str = "gemini-2.5-flash",
    delay: float = 0.0,
    rate_limiter: RateLimiter | None = None,
    session: Any = None,
    fake_scores: FakeScoreSource = None,
) -> ImageEvaluator:
    """Factory function instantiating the appropriate ImageEvaluator adapter."""
    scorer_lower = scorer_type.lower().strip()

    if scorer_lower == "fake" or (model_id and model_id.lower().startswith("fake")):
        return FakeEvaluator(
            scores=fake_scores,
            model_id=model_id or "fake-evaluator",
            required_max_dimension=None if stage == 2 else 512,
        )

    def _make_gemini(resolved_model: str, scale: float) -> GeminiEvaluator:
        if not gemini_key:
            raise ValueError("Google Gemini API Key is required for Gemini evaluation.")
        return GeminiEvaluator(
            api_key=gemini_key,
            model_id=resolved_model,
            delay=delay,
            rate_limiter=rate_limiter,
            required_max_dimension=512,
            raw_scale=scale,
            session=session,
        )

    def _make_openai(resolved_model: str, scale: float) -> OpenAIEvaluator:
        if not openai_key:
            raise ValueError("OpenAI API Key is required for OpenAI evaluation.")
        return OpenAIEvaluator(
            api_key=openai_key,
            base_url=openai_url,
            model_id=resolved_model,
            delay=delay,
            rate_limiter=rate_limiter,
            required_max_dimension=512,
            raw_scale=scale,
            session=session,
        )

    if stage == 2:
        target = model_id or stage2_model or "gemini-2.5-flash"
        target_lower = target.lower()
        if "gemini" in target_lower:
            api_model = target if target != "gemini" else (gemini_model or "gemini-2.5-flash")
            return _make_gemini(api_model, scale=1.0)
        if "openai" in target_lower or "gpt" in target_lower:
            api_model = target if target != "openai" else (openai_model or "gpt-4o-mini")
            return _make_openai(api_model, scale=1.0)
        return PyIQAEvaluator(model_id=target, required_max_dimension=None)

    if scorer_lower == "local":
        resolved_local = model_id or local_model_id or "somepago/AestheticSigLIP"
        return LocalClipEvaluator(model_id=resolved_local, required_max_dimension=512)
    if scorer_lower == "gemini":
        return _make_gemini(model_id or gemini_model or "gemini-2.5-flash", scale=0.1)
    if scorer_lower == "openai":
        return _make_openai(model_id or openai_model or "gpt-4o-mini", scale=0.1)

    raise ValueError(f"Unsupported scorer_type: {scorer_type}")
