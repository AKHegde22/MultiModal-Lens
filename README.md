# MultimodalLens

MultimodalLens is a publishing-ready multimodal interpretability toolkit for Hugging Face vision-language models. It combines interactive debugging with mechanistic probes and release-grade evaluation workflows.

## Project Status

- Stage: Beta
- Primary interface: Gradio app (`Explore`, `Compare`, `Eval`)
- Mechanistic focus: TransformerLens-inspired workflows for multimodal models
- Publishing support: preflight checks, tested-model matrix, CI and release build verification

## Core Capabilities

- Attention analysis (rollout, optional gradients, overlays)
- Token-patch alignment and token contribution scoring
- Cross-modal similarity diagnostics
- Faithfulness diagnostics (deletion/insertion/counterfactual/attn-grad agreement)
- Layer activation caching with forward hooks
- Cross-modal activation patching
- Multimodal logit lens decoding
- Grounding-head discovery via visual ablation sensitivity

## Supported Family Labels

Recommended input is `auto`, which resolves to canonical adapters from model config.

- CLIP family: `clip`, `siglip`, `siglip2`, `altclip`, `xclip`
- BLIP family: `blip2`, `instructblip`
- Decoder VLM family: `llava`, `llava_next`, `llava_onevision`, `qwen2_vl`, `qwen2_5_vl`, `idefics2`, `idefics3`, `paligemma`, `mllama`, `internvl`, `minicpmv`, `smolvlm`, `kosmos2`, `florence2`

Custom family labels are also accepted; the registry will infer a canonical adapter path.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional extras:

```bash
pip install -e '.[dev]'
pip install -e '.[publish]'
```

## Run The App

```bash
python app.py
```

Then open the local Gradio URL.

## Quick Start

1. Open `Explore`.
2. Set `Model Family` to `auto`.
3. Enter a model checkpoint (or keep default).
4. Upload image + prompt.
5. Click `Analyze`.
6. Optional: enable `Run mechanistic suite (4 probes)`.

## CLI Workflows

### Batch evaluation

```bash
python scripts/run_eval.py \
  --dataset /absolute/path/dataset.jsonl \
  --family auto \
  --model openai/clip-vit-base-patch32 \
  --output eval_results.csv
```

### Mechanistic probes

```bash
python scripts/run_mechanistic.py grounding \
  --family auto \
  --model llava-hf/llava-1.5-7b-hf \
  --image /abs/path/image.jpg \
  --prompt "Describe this scene" \
  --output outputs/grounding.json
```

Modes: `layers`, `cache`, `patch`, `logit-lens`, `grounding`.

### Model preflight

```bash
python scripts/run_preflight.py \
  --family auto \
  --model openai/clip-vit-base-patch32
```

Optional JSON report:

```bash
python scripts/run_preflight.py \
  --family auto \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --output reports/preflight_qwen2_vl.json
```

## Dataset Format

For `run_eval.py`, each JSONL row must include:

```json
{"id": "sample-1", "image": "/abs/path/image.jpg", "text": "a red bus on a city street"}
```

## Repository Layout

```text
src/multimodallens/
  adapters/      # Family-specific model adapters
  analysis/      # Attribution and mechanistic analysis
  core/          # Registry, pipeline, preflight
  eval/          # Batch evaluation runner
  ui/            # Gradio UI
  utils/         # Tensor/image helpers
  viz/           # Plot builders
scripts/         # CLI utilities
docs/            # Methodology and release documentation
tests/           # Unit and integration-style tests
```

## Documentation

- docs/README.md
- docs/methodology.md
- docs/research_paper_draft.md
- docs/publishing_checklist.md
- docs/tested_model_matrix.md

## Publish Checklist

Before tagging:

1. `ruff check src scripts tests`
2. `pytest -q`
3. `python -m build`
4. `twine check dist/*`
5. Update `docs/tested_model_matrix.md`

## Notes

- Large checkpoints can require significant memory.
- Use `trust_remote_code=True` only for trusted repositories.
- Attention is a diagnostic signal, not a standalone causal proof.

## Next Steps (TODO)

- [ ] Add automated smoke tests against a pinned public checkpoint set per family alias group.
- [ ] Add reproducible benchmark scripts for grounding and hallucination analysis.
- [ ] Add optional notebook exports from Explore artifacts.
- [ ] Add adapter-level telemetry for runtime fallback decisions.
- [ ] Add a versioned changelog process for releases.
