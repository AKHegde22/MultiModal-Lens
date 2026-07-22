# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-22

### Added
- **`HookedVLM` API**: TransformerLens-style entrypoint with `HookedVLM.from_pretrained()`, `run_with_cache()`, and `run_with_hooks()`.
- **`ActivationCache`**: Dictionary-like container for named layer activations with pattern-matching support (`cache["*layer.5*"]`).
- **Custom Exception Hierarchy**: `MultimodalLensError`, `UnsupportedFamilyError`, `UnsupportedDtypeError`, `ModelLoadError`, `AdapterError`, `AnalysisError`.
- **CLI Entrypoint**: `multimodallens` command with subcommands (`ui`, `analyze`, `preflight`, `eval`, `version`).
- **Interactive UI Overhaul**: Accordion controls for model & runtime settings, progressive disclosure, soft theme, upload support for eval JSONL datasets, and user-friendly error wrapping.
- **Root Quickstart & Demos**: Created `Main_Demo.ipynb` notebook and 5 detailed tutorials in `demos/`.
- **Automated PyPI Publish Workflow**: `.github/workflows/publish.yml` for automated PyPI publishing on git tags.
- **Development Makefile**: Added `make lint`, `make test`, `make test-slow`, `make build`, and `make clean`.
- **Machine-Readable Model Matrix**: `src/multimodallens/data/supported_models.json`.

### Fixed
- **Silent LLaVA Fallback Bug**: `infer_family_from_model` now returns `None` instead of silently defaulting to `"llava"`. `resolve_family` raises `UnsupportedFamilyError`.
- **CPU Float16 Crash**: Model adapters running on CPU now auto-promote `float16`/`bfloat16` to `float32` with an explicit user warning.
- **Silent Dtype Fallback**: `ModelAdapter` now validates requested precision strings against supported dtypes, raising `UnsupportedDtypeError` for unknown types.
- **LLaVA Gradient Crash**: Added `hidden_last.grad is not None` guard before computing gradient attribution.
- **LLaVA Image Size Tuple Bug**: `_guess_patch_count()` now handles `image_size` returned as tuple/list.
- **Adapter Cache Key Collision**: `LensPipeline` adapter cache key now includes `trust_remote_code` and `low_cpu_mem_usage`.

---

## [0.1.0] - 2026-07-01

### Added
- Initial release of MultimodalLens with `LensPipeline`, `CLIPAdapter`, `BLIP2Adapter`, and `LlavaAdapter`.
