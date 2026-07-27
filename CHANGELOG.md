# CHANGELOG

<!-- version list -->

## v1.3.0 (2026-07-27)

### Build System

- Update pyright pre-commit hook to v1.1.411
  ([`863f06d`](https://github.com/jbelew/immich-aesthetic-scorer/commit/863f06dbc984487c22437e6dbf16cc9c6db8216d))

### Documentation

- Add list of tested local models to README
  ([`b4a069f`](https://github.com/jbelew/immich-aesthetic-scorer/commit/b4a069f8c696f5c9a2a4d57ad70ec0f813ce8e0a))

- Align Stage 2 diagram nodes to refer to LLM Aesthetics Evaluation
  ([`b6f8f5d`](https://github.com/jbelew/immich-aesthetic-scorer/commit/b6f8f5d2a269112bb34dcd02057edff4420d9657))

- Align Stage 2 text descriptions to refer strictly to LLM Aesthetics Evaluation
  ([`9e5ab1f`](https://github.com/jbelew/immich-aesthetic-scorer/commit/9e5ab1fe8f38931c2a15ec1769f7e87ba0c24059))

- Clarify thumbnail downscaling in flow diagrams
  ([`d75c87e`](https://github.com/jbelew/immich-aesthetic-scorer/commit/d75c87e9ea0bb2d8522888c55d4b3b474a8471a2))

- Correct description of musiq-spaq as a photographic quality evaluator
  ([`1ddb816`](https://github.com/jbelew/immich-aesthetic-scorer/commit/1ddb8161cac38888555533edf1fbec45be4f533e))

- Document local model processing speeds in README
  ([`a2eb7ee`](https://github.com/jbelew/immich-aesthetic-scorer/commit/a2eb7ee7f36ccd9117d113c1fa5f4fbce1522755))

- Position cache check before evaluation in diagrams
  ([`e620dc3`](https://github.com/jbelew/immich-aesthetic-scorer/commit/e620dc35a7aca91f53ede8e8035154b1157e9ccb))

- Quote Mermaid labels containing parentheses to fix syntax error
  ([`a3ea681`](https://github.com/jbelew/immich-aesthetic-scorer/commit/a3ea6816939469c82f736a81a703bf7dfa1bcf76))

- Refine final steps of flow diagrams to show conditional write-ratings sync
  ([`b561fdd`](https://github.com/jbelew/immich-aesthetic-scorer/commit/b561fddf0950487e8e08a309858fa52047c7d6f4))

- Specify Stage 2 remote LLM recommendation in Mermaid diagram
  ([`6e8dcf2`](https://github.com/jbelew/immich-aesthetic-scorer/commit/6e8dcf274f87ec005f5023d1c62d3d35949e6f6d))

- Update architecture document to reflect SigLIP default and std dev floor
  ([`2e0681f`](https://github.com/jbelew/immich-aesthetic-scorer/commit/2e0681f9d84142625bf93cefa6eaa74fc6c498ac))

- Update references from CLIP to SigLIP default model in README
  ([`e8052ee`](https://github.com/jbelew/immich-aesthetic-scorer/commit/e8052ee3d72982cf24b1a450cd1d788137458972))

- Update system architecture flow diagram in README
  ([`a4228c7`](https://github.com/jbelew/immich-aesthetic-scorer/commit/a4228c74014c7c0b8bd931b8d421d5aa207f7b4d))

### Features

- **cli**: Make gemini-2.5-flash the default Stage 2 model
  ([`ad79e33`](https://github.com/jbelew/immich-aesthetic-scorer/commit/ad79e33bcd00bc515a3ad3f91e31b05441a3dd36))

- **cli**: Set default local-model to somepago/AestheticSigLIP
  ([`7c3f142`](https://github.com/jbelew/immich-aesthetic-scorer/commit/7c3f1420ea0ac9f011fade363843f349457bae5e))

### Testing

- Skip local model tests if ML dependencies are missing
  ([`185c747`](https://github.com/jbelew/immich-aesthetic-scorer/commit/185c74743b41a1d855cfb6938a841a514665ffcb))


## v1.2.1 (2026-07-26)

### Bug Fixes

- **dedup**: Improve timestamp parsing and implement sliding window for burst deduplication
  ([`a573eb9`](https://github.com/jbelew/immich-aesthetic-scorer/commit/a573eb994ddd45df136c29b847da54904c73109c))


## v1.2.0 (2026-07-26)

### Features

- **local**: Integrate Aesthetic-SigLIP model and enforce std dev floor
  ([`6ab679f`](https://github.com/jbelew/immich-aesthetic-scorer/commit/6ab679f69529e9eef127e871243904ce38737626))


## v1.1.1 (2026-07-25)

### Bug Fixes

- **cli**: Correct Stage 1 and Stage 2 console print labels to match pipeline roles
  ([`39cd44b`](https://github.com/jbelew/immich-aesthetic-scorer/commit/39cd44b3d0687a258fafa880f24e0e3c8dd22b39))

- **cli**: Make Stage 2 console header and combine labels dynamic
  ([`4f6f992`](https://github.com/jbelew/immich-aesthetic-scorer/commit/4f6f9920ff2632372d87c5649ce8dcd209d62484))

- **cli**: Print actual overridden concurrency instead of configured value
  ([`7a21315`](https://github.com/jbelew/immich-aesthetic-scorer/commit/7a213154d4f35da7ccc13b62d4af0172295b616e))

### Documentation

- Clarify Stage 1 and Stage 2 pipeline stage distinctions in README
  ([`8186b01`](https://github.com/jbelew/immich-aesthetic-scorer/commit/8186b01b707b1e63851136c84d353a8551ac15ac))


## v1.1.0 (2026-07-24)

### Bug Fixes

- **pacing**: Sleep outside lock and fix delay overrides in Stage 2 error handling
  ([`9db2640`](https://github.com/jbelew/immich-aesthetic-scorer/commit/9db2640813dc537db4f4a8e72209c3120a1ea53f))

### Build System

- Configure packaging and resolve pre-commit lint/type errors
  ([`bc2de44`](https://github.com/jbelew/immich-aesthetic-scorer/commit/bc2de441ec6b59876607542a189f9c8862a22a45))

- **ci**: Upgrade actions to v6 to support Node 24 natively and remove env override
  ([`093a846`](https://github.com/jbelew/immich-aesthetic-scorer/commit/093a8462e3128a93263cc89561102a0fe2559bfa))

- **ci**: Upgrade GitHub actions to checkout@v4 and setup-python@v5
  ([`e0d432e`](https://github.com/jbelew/immich-aesthetic-scorer/commit/e0d432ef08185b446b5da7bfb0178ab7463480da))

- **deps**: Bump actions/checkout from 6 to 7
  ([`81766b7`](https://github.com/jbelew/immich-aesthetic-scorer/commit/81766b78a32208d354aa054652cf4fe722acae93))

- **deps**: Bump actions/setup-python from 6 to 7
  ([`8804403`](https://github.com/jbelew/immich-aesthetic-scorer/commit/8804403320d8a8cbdbad7357b90e8cebffe7c051))

- **model**: Unify Gemini model default to gemini-2.5-flash
  ([`4413edc`](https://github.com/jbelew/immich-aesthetic-scorer/commit/4413edca8faa468cbccb6c16dcda4a7031f530e7))

### Continuous Integration

- Configure dependabot for github-actions and pip updates
  ([`7e2c093`](https://github.com/jbelew/immich-aesthetic-scorer/commit/7e2c093a94a754f133649cf4ae672c8ece938e6b))

### Documentation

- Add GNU General Public License v3.0
  ([`4d4a5ee`](https://github.com/jbelew/immich-aesthetic-scorer/commit/4d4a5ee6b65925b7314059c5e4f2da5d0ed5498d))

### Features

- **cache**: Save Stage 2 explanation reason from Gemini/OpenAI in the cache
  ([`38237fc`](https://github.com/jbelew/immich-aesthetic-scorer/commit/38237fcd62c3c8983b06989f6c2b5ada96fdd27a))

### Performance Improvements

- **ci**: Enforce single thread (concurrency=1) for remote LLM API calls
  ([`05587f5`](https://github.com/jbelew/immich-aesthetic-scorer/commit/05587f5e7ab9fe102bae6186e06d9e3649f971d4))

### Refactoring

- Generalize AESTHETIC_PROMPT to handle non-portrait subjects
  ([`f2f84db`](https://github.com/jbelew/immich-aesthetic-scorer/commit/f2f84db255d289f8eaa8ff4715d177f34c1106ba))

- Remove unused STAGE2_TECHNICAL_PROMPT dead code
  ([`f85202b`](https://github.com/jbelew/immich-aesthetic-scorer/commit/f85202b1a15b4ca7e4d1c70e365d3c7ba40c6eec))

- Rename STAGE1_AESTHETIC_PROMPT to AESTHETIC_PROMPT
  ([`33c4072`](https://github.com/jbelew/immich-aesthetic-scorer/commit/33c4072fe886be85cd91ea81b33b825099248373))

- **cache**: Remove redundant raw_score key
  ([`0e5a0b2`](https://github.com/jbelew/immich-aesthetic-scorer/commit/0e5a0b20a10b322a23e21125c5cf9465ccce7714))

- **prompt**: Remove sharpness/focus criteria from AESTHETIC_PROMPT
  ([`e55dc86`](https://github.com/jbelew/immich-aesthetic-scorer/commit/e55dc864b512c000f4f6ba6b438339f97bc0416f))


## v1.0.0 (2026-07-24)

- Initial Release
