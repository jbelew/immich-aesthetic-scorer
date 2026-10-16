# Immich Aesthetic Scorer

Evaluates photos from an Immich library using local vision models (SigLIP/CLIP/MUSIQ) or remote multimodal LLMs (Gemini/OpenAI), calibrates their scores across the library population, deduplicates burst shots, and compiles top highlights into an Immich album.

## Language

**Asset**:
An image record retrieved from the Immich server (`type == "IMAGE"`), identified by its UUID and accompanied by capture/modification timestamps.
_Avoid_: Photo file, media item

**Stage 1 Evaluation**:
Coarse aesthetic or quality scoring executed across all target assets on `512px` downscaled thumbnails, producing an uncalibrated `raw_score_stage1` on a `0.0–10.0` scale.
_Avoid_: Pre-filter, initial pass

**Stage 2 Evaluation**:
Deep aesthetic or technical quality scoring executed only on the top `stage2_top_pct` percentage of Stage 1 candidates, producing an uncalibrated `raw_score_stage2` on a `0.0–100.0` scale.
_Avoid_: Reranker, second pass

**Z-Score Sigmoid Calibration**:
Population normalization that standardizes raw scores ($z = (x - \mu) / \sigma$, with a `min_std=1.0` floor in Stage 1) and maps them through a logistic sigmoid ($100 / (1 + e^{-1.5z})$) onto the `0–100` score scale where the population mean maps to `50.0`.
_Avoid_: Min-max scaling, percentile ranking

**Score Fusion**:
Weighted combination of calibrated Stage 1 (`0–100`) and Stage 2 (`0–100`) scores using `stage2_weight` ($w \in [0.0, 1.0]$), assigning non-candidates the neutral population-mean Stage 2 fallback of `50.0` to preserve mathematical continuity across the candidate boundary.
_Avoid_: Score averaging, ensemble score

**Burst Deduplication**:
Greedy chronological clustering (`dedup_window` seconds) where consecutive assets captured within `dedup_window` of the previous asset belong to the same burst group and only the single highest-scoring asset per group is retained.
_Avoid_: Duplicate removal, near-duplicate hashing

**Star Rating**:
Immich's native `1–5` integer asset rating derived deterministically from the calibrated `0–100` score (`>=90` → 5, `>=75` → 4, `>=50` → 3, `>=20` → 2, `<20` → 1).
_Avoid_: Favorite flag, rank

**Highlights Album**:
The target Immich album (`target_album_name`) created or updated with the top `limit` highest-scoring assets.
_Avoid_: Output folder, destination collection

**Evaluation Store**:
The thread-safe, model-aware local cache (`.immich_aesthetic_cache.json`) that persists Stage 1, Stage 2, and final calibrated scores per asset UUID and invalidates only the stage whose configured model ID or asset `updatedAt` timestamp changed.
_Avoid_: Score database, state file
