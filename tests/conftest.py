"""Pytest configuration and shared test isolation fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

# Default CUDA_VISIBLE_DEVICES to empty before PyTorch is imported so unit tests
# do not pay ~15s of GPU driver initialization overhead.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(autouse=True)
def _isolate_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolates tests from any untracked local `config.json` or `.immich_aesthetic_cache.json` in the repo root."""
    monkeypatch.chdir(tmp_path)
    yield
