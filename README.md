# MultimodalLens 🔍

*A library for mechanistic interpretability of vision-language models.* Inspired by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

[![PyPI version](https://img.shields.io/pypi/v/multimodallens.svg)](https://pypi.org/project/multimodallens/)
[![Build Status](https://github.com/AKHegde22/Papers-C/actions/workflows/ci.yml/badge.svg)](https://github.com/AKHegde22/Papers-C/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

MultimodalLens lets you inspect internal activations, attention rollouts, token-patch alignments, logit lens predictions, and grounding circuits across 20+ Hugging Face vision-language model architecture families.

---

## Installation

```bash
pip install multimodallens
```

Or install from source:

```bash
git clone https://github.com/AKHegde22/Papers-C.git
cd MultiModal-Lens
pip install -e .
```

---

## Quick Start

### 1. Power API (`HookedVLM`)

```python
from multimodallens import HookedVLM
from PIL import Image

# Load model in 1 line
vlm = HookedVLM.from_pretrained("openai/clip-vit-base-patch32", device="auto")

# Run with cache — get logits, scores, and all internal layer activations
result, cache = vlm.run_with_cache(Image.open("cat.jpg"), "a photo of a cat")

# Access activation by layer name
layer_5_act = cache["vision_encoder.layers.5"]
```

### 2. Intervention Hooks (`run_with_hooks`)

```python
# Zero-ablate a layer on the fly
def zero_ablate(tensor):
    return tensor * 0.0

patched_result = vlm.run_with_hooks(
    image=Image.open("cat.jpg"),
    prompt="a photo of a cat",
    fwd_hooks=[("vision_encoder.layers.5", zero_ablate)]
)
```

### 3. Launch Interactive Web App

```bash
multimodallens ui
```

---

## Core Capabilities

- **`HookedVLM` & `ActivationCache`**: TransformerLens-style stateful interface with dict-like activation lookup.
- **Attention Rollout & Overlays**: Heatmap generation over input images from multi-head attention weights.
- **Token-Patch Alignment**: Cross-modal cosine similarity matrices between text tokens and visual patches.
- **Multimodal Logit Lens**: Decode intermediate hidden states to vocabulary tokens layer by layer.
- **Cross-Modal Activation Patching**: Causal tracing by swapping activations between source and target images.
- **Grounding Head Discovery**: Identify specific attention heads responsible for visual grounding.
- **Faithfulness Diagnostics**: Deletion/insertion curves and counterfactual perturbation drops.

---

## Demo Notebooks & Tutorials

| Notebook | Description | Link |
|---|---|---|
| **Main Demo** | Full overview of all library features | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Main_Demo.ipynb) |
| `01_quickstart` | 5-minute introduction to `HookedVLM` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/01_quickstart.ipynb) |
| `02_attention_deep_dive` | Attention rollout, alignment & gradients | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/02_attention_deep_dive.ipynb) |
| `03_mechanistic_probes` | Logit lens, patching & grounding heads | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/03_mechanistic_probes.ipynb) |
| `04_faithfulness_testing` | Perturbation curves & Spearman rank tests | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/04_faithfulness_testing.ipynb) |
| `05_comparing_models` | Differential prompt & model comparisons | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](demos/05_comparing_models.ipynb) |

---

## Supported Model Families

| Family Label | Canonical Adapter | Example Checkpoints | Status |
|---|---|---|---|
| `auto` | Automatic Inference | *Infers family from HF AutoConfig* | ✅ Supported |
| `clip` | `CLIPAdapter` | `openai/clip-vit-base-patch32`, `google/siglip-base-patch16-224` | ✅ Supported |
| `blip2` | `BLIP2Adapter` | `Salesforce/blip2-opt-2.7b`, `Salesforce/instructblip-vicuna-7b` | ✅ Supported |
| `llava` | `LlavaAdapter` | `llava-hf/llava-1.5-7b-hf`, `Qwen/Qwen2-VL-2B-Instruct`, `HuggingFaceM4/idefics2-8b` | ✅ Supported |

Other supported aliases: `siglip`, `siglip2`, `altclip`, `xclip`, `instructblip`, `llava_next`, `llava_onevision`, `qwen2_vl`, `qwen2_5_vl`, `idefics2`, `idefics3`, `paligemma`, `mllama`, `internvl`, `minicpmv`, `smolvlm`, `kosmos2`, `florence2`.

---

## CLI Usage

```bash
# Launch UI
multimodallens ui --port 7860

# Run single analysis
multimodallens analyze --model openai/clip-vit-base-patch32 --image photo.jpg --prompt "a dog"

# Model compatibility preflight
multimodallens preflight --model Qwen/Qwen2-VL-2B-Instruct

# Batch evaluation
multimodallens eval --dataset dataset.jsonl --model openai/clip-vit-base-patch32 --output results.csv
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
