# Spec: Refactor Monolith into Deep Modules

**Issue:** [#3](https://github.com/jbelew/immich-aesthetic-scorer/issues/3)
**Status:** `ready-for-agent`

---

## Problem Statement

Currently, the entire Immich Aesthetic Scorer exists as a single monolithic script of over 2,100 lines (`score_assets.py`). All responsibilities—network calls, authentication, pagination, image downscaling, GPU/CPU model loading, prompt generation, remote API retries, rate limiting, cache validation, atomic disk I/O, mathematical calibration, candidate selection, burst deduplication, star rating updates, and terminal interactions—are intertwined within procedural loops in the main driver. This makes the system fragile to change, difficult to test without extensive mocks, and hard for developers and AI agents to navigate or extend with new vision models or features.

## Solution

Refactor the monolithic application into four deep modules, each exposing a minimal public interface at a clean architectural seam and hiding its complex implementation. The monolithic entrypoint becomes a lean CLI driver (~100 lines) coordinating these modules. Existing CLI arguments, config options, and legacy cache entries remain fully backward-compatible.

## User Stories

1. As a user scoring photos, I want the system to calculate calibrated aesthetic scores using statistical z-scores and sigmoid normalization, so that photos from different models are mapped onto an equitable 0–100 scale.
2. As a user scoring photos, I want the scoring pipeline to enforce a minimum standard deviation floor, so that narrow distributions of raw scores do not produce wildly exaggerated outlier scores.
3. As a user evaluating a large library, I want a two-stage evaluation option that filters the top candidates from Stage 1 for high-precision Stage 2 evaluation, so that I get the highest quality album without wasting API tokens or local compute on unpromising photos.
4. As a user with photos scored only in Stage 1, I want non-candidates to receive a mathematically continuous fallback score assuming average technical quality, so that they are not unfairly penalized compared to candidate photos.
5. As a photographer taking continuous burst shots, I want sliding-window burst deduplication based on photo timestamps, so that my highlights album showcases distinct moments rather than multiple identical near-duplicates.
6. As an Immich user, I want composite scores mapped cleanly to 1–5 star ratings, so that my Immich photo library reflects photo quality natively.
7. As a user running multiple scoring sessions, I want an evaluation cache that avoids re-scoring photos whose models and timestamps have not changed, so that scoring is fast and avoids unnecessary compute or API costs.
8. As a user updating a model in configuration, I want the cache to invalidate only the stage matching the changed model, so that assets evaluated by the unchanged stage do not need expensive re-evaluation.
9. As a user modifying or re-uploading an image, I want the cache to detect timestamp changes and re-evaluate that photo, so that edits are reflected in the aesthetic score.
10. As a user with an older cache file, I want legacy reason strings parsed automatically for raw scores, so that I do not lose previously computed evaluations when upgrading.
11. As a user running multi-threaded scoring, I want cache updates to be thread-safe and persisted atomically via temporary files, so that unexpected crashes or power loss cannot corrupt my cache file.
12. As a user using local CLIP or SigLIP models, I want the evaluation module to automatically negotiate CUDA hardware capability (falling back to CPU if capability < 7.0), so that execution never crashes on unsupported GPU drivers.
13. As a user evaluating images locally with PyIQA (such as MUSIQ), I want the model evaluation to happen in-memory without polluting my filesystem with temporary files, so that disk I/O does not bottleneck scoring.
14. As a user utilizing remote vision APIs (Gemini or OpenAI), I want the system to handle rate limiting, token pacing, and exponential backoff transparently, so that I do not encounter unhandled 429 quota errors.
15. As a developer writing tests, I want a fake evaluator adapter, so that I can test the full scoring workflow without connecting to real models or external APIs.
16. As an Immich user, I want the Immich gateway to use pooled HTTP connections, so that repeated asset downloads and metadata searches avoid TCP handshake overhead.
17. As an Immich user with large libraries, I want the gateway to handle metadata search pagination transparently, so that all image assets are retrieved without manual offset management.
18. As an Immich user adding assets to highlights albums, I want the gateway to automatically deduplicate IDs and batch requests into 100-item chunks, so that server payload limits are respected.
19. As a developer or CLI user, I want all existing CLI arguments and configuration JSON keys preserved, so that existing automated scripts or user habits continue to work unchanged.
20. As a developer extending the codebase, I want each module to have a single, well-defined seam and test surface, so that changes to one subsystem (e.g. Immich API endpoints) never break another (e.g. score calibration math).

## Implementation Decisions

- The monolithic code will be structured into a dedicated package:
  - **Score Pipeline & Calibration Module**: Pure in-process computation module responsible for z-score standardization with variance floors, logistic sigmoid conversion, candidate filtering, fallback score fusion, star rating mapping, and sliding-window burst deduplication.
  - **Evaluation Store Module**: Local-substitutable persistence module holding score records. Encapsulates timestamp validation, model ID validation, backward-compatibility regex parsing for legacy scores, thread safety, dirty checking, and atomic disk replacement via temporary files. Supports an in-memory storage adapter for fast unit testing.
  - **Image Evaluation Module**: Unified evaluation seam with concrete adapters for local PyTorch CLIP/SigLIP models, PyIQA models, remote Gemini/OpenAI vision endpoints, and in-memory test doubles. Hides hardware capability negotiation (CUDA sm_70+ requirement), rate-limit pacing, exponential backoff, and thumbnail resolution requirements.
  - **Immich Gateway Adapter**: Ports & adapters module encapsulating the Immich REST protocol. Maintains a persistent session with connection pooling. Encapsulates pagination loops for asset searches, preview thumbnail downloads, album creation/retrieval, 100-item batch slicing for album asset additions, and native star rating updates.
  - **CLI & Backward Compatibility Layer**: Contains the CLI parser and a concise driver coordinating the four modules. Re-exports public functions (e.g., `check_immich_connection`, `fetch_all_image_assets`, `load_cache`, `save_cache`, `score_to_stars`, `deduplicate_bursts`, `score_image_local`) so that all existing unit tests in `test_score_assets.py` and downstream integrations continue to function without disruption.
- Architecture vocabulary strictly adheres to codebase design: modules present small interfaces at explicit seams; callers receive high leverage while maintainers gain locality.
- No new external runtime dependencies are introduced. Standard library `dataclasses`, `typing`, `math`, `json`, and existing project dependencies (`requests`, `pillow`, `tqdm`) are used.

## Testing Decisions

- A good test verifies external observable behavior through the module's public interface at pre-agreed seams, without testing internal private helpers or relying on implementation side channels.
- Four focused test suites will be introduced:
  - Pure mathematical pipeline tests: Tests the mathematical calibration formulas, minimum variance floors, two-stage fallback blending, and burst deduplication using known-good test data and edge cases (empty lists, uniform scores, invalid dates) without any mocks or I/O.
  - Evaluation cache tests: Tests cache invalidation invariants (timestamp drift, model mismatch), backward-compatibility extraction of legacy scores, dirty-state tracking, and thread-safe persistence using in-memory and temporary file adapters.
  - Image evaluation tests: Tests evaluator factory dispatch, fake evaluation, rate limit timing, and retry logic without live network or GPU calls.
  - Immich gateway tests: Tests HTTP session reuse, pagination completion, 100-chunk album additions, and error translation.
- Existing test suite (15 tests in `test_score_assets.py`) is maintained and must continue to pass 100%.

## Out of Scope

- Modifying the underlying statistical formulas (Z-score mean/std standardization, logistic sigmoid coefficient of 1.5, candidate fallback value of 50.0).
- Adding new vision model backbones beyond those already supported (SigLIP, Rsinema, simple-aesthetics-predictor, PyIQA MUSIQ, Gemini, OpenAI).
- Changing Immich REST API payload schemas or Immich server compatibility requirements.
- Adding a GUI or web interface.

## Further Notes

- This refactoring implements Candidates 1 through 4 from the architectural review report.
- Once this spec is published, `/to-tickets` can decompose it into five tracer-bullet tickets with dependency edges for implementation via `/tdd`.
