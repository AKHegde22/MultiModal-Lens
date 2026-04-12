# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Model preflight system with CLI (`scripts/run_preflight.py`) and UI diagnostics.
- Expanded multimodal family aliasing and auto-resolution.
- Publish/readiness workflows in GitHub Actions.
- Release docs: publishing checklist and tested model matrix.

### Changed

- README reorganized for release clarity.
- Package metadata hardened for build and distribution checks.
- UI family fields now support custom values and preflight reporting.

### Fixed

- CLIP mechanistic probe robustness for logit lens and grounding extraction.

## [0.1.0]

### Added

- Initial release of MultimodalLens with Explore, Compare, and Eval flows.
- Adapterized support for CLIP, BLIP-2, and LLaVA-style models.
- Mechanistic suite: layers/cache/patch/logit-lens/grounding.
