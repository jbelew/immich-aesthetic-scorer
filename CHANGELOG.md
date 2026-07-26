# CHANGELOG

<!-- version list -->

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
