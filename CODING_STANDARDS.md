# Coding Standards

Consulted during `/code-review`. Mechanical rules (formatting, import sorting, type checking, docstring presence) are enforced by `ruff` and `pyright` in `pyproject.toml` and `.pre-commit-config.yaml`. This document covers judgement calls.

## Two-Tier Comment & Documentation Standard

### Tier 1: Production Code (`immich_aesthetic_scorer/`, `score_assets.py`)

- **Public interface contracts**: Public modules, classes, and functions carry Google-style docstrings that state:
  - **Domain units and ranges**: Specify numeric scales explicitly (e.g., raw CLIP `0.0–10.0`, calibrated/LLM `0–100`, star ratings `1–5`, `stage2_weight` `0.0–1.0`, `dedup_window` in seconds).
  - **Invariants and fallbacks**: Document boundary behavior (e.g., `min_std` floor, `50.0` Stage 2 non-candidate fallback, `1980-01-01` MS-DOS epoch timestamp fallback, `CUDA major >= 7` requirement).
  - **Concurrency and side effects**: State thread-safety guarantees (e.g., internal `RLock`, atomic `.tmp` replacement, cross-thread rate-limit slot reservation) and raised exceptions.
- **Explanatory inline comments**: Write inline comments where the rationale or constraint is not visible from the code itself:
  - Hardware, driver, or third-party library quirks (e.g., disabling PyTorch thread contention via `set_num_threads(1)`, passing in-memory `PIL.Image` to `pyiqa` to avoid temp files).
  - Mathematical formulas, scaling factors (`* 2.0` for 0–5 to 0–10 heads, `/ 10.0` for Stage 1 remote score normalization), and retry/backoff rules.
  - Keep inline comments focused on *why* a decision or invariant exists rather than restating the syntax of the next line.

### Tier 2: Test Code (`tests/`, `test_*.py`)

- **Behavioral test names**: Name test functions as observable specifications at the module's public seam (e.g., `test_evaluates_in_memory_without_leaking_temp_files`, `test_invalidates_only_stage_with_changed_model`).
- **Brief setup and math comments**: Use short, single-line comments in tests only to clarify:
  - Worked-example expected values (`# 4.25 * 2.0 = 8.5 on 0-10 scale`).
  - Non-obvious concurrency or timeline setups (`# First call takes 0s slot; concurrent calls reserve 4s and 8s`).

## Module Design

- **Deep modules at explicit seams**: Keep module interfaces small (`pipeline`, `store`, `evaluators`, `gateway`) and hide protocol, persistence, hardware negotiation, and retry complexity inside the implementation.
- **Test through public seams**: Exercise behavior through public functions, classes, and injected adapters (`FakeEvaluator`, `EvaluationStore.in_memory()`) rather than reaching into private attributes.
