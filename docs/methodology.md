# MultimodalLens Methodology

## 1. Objective

MultimodalLens is designed as a model-debugging interface, not just a visualization dashboard. The core methodology is to combine three complementary evidence channels for every `(image, prompt)` pair:

1. Structural signal: attention-derived maps (rollout and layer-level views)
2. Representation signal: token-patch similarity in shared embedding space
3. Causal sanity signal: perturbation-based faithfulness tests

This reduces over-reliance on a single explanation primitive.

## 2. System Design

### 2.1 Adapterized model abstraction

All model-specific logic is isolated behind `ModelAdapter`:

- `load()`
- `prepare(image, prompt)`
- `analyze(image, prompt, compute_gradients=False)`
- `score(image, prompt)`

This abstraction ensures UI and evaluation layers stay architecture-agnostic.

### 2.2 Implemented model families

- `CLIPAdapter` (`clip`): dual-encoder extraction from text/vision towers.
- `BLIP2Adapter` (`blip2`): vision + Q-Former + language outputs.
- `LlavaAdapter` (`llava`): decoder-side internals with image token handling.

Adapters normalize outputs into a common `AnalysisResult` schema.

## 3. Attention Processing

### 3.1 Rollout

Given per-layer attention matrices `A_l`, MultimodalLens computes residual-augmented rollout:

`A'_l = normalize(A_l + I)`

`R = A'_L A'_{L-1} ... A'_1`

The image heatmap is derived from selected query-to-image-token entries in `R`, reshaped to patch grid.

### 3.2 Additional maps

- Last-layer mean-head maps for direct inspection.
- Optional gradient attribution map from objective backpropagation to visual hidden states.

## 4. Token-Image Alignment

For extracted text states `T in R^{N x D}` and image patch states `V in R^{M x D}`:

1. L2 normalize each row.
2. Compute `S = T V^T`.

`S` is shown as token-by-patch heatmap. Token salience proxy is `max_j S_ij`.

## 5. Cross-Modal Global Score

Scoring is model-family dependent:

- CLIP: direct `logits_per_image` similarity.
- BLIP-2/LLaVA: sequence-level log-probability proxy from teacher-forced logits.

This score is reused for perturbation tests.

## 6. Faithfulness Diagnostics

### 6.1 Deletion/insertion curves

Using patch importance scores:

- Deletion: progressively mask top-ranked patches and track score drop.
- Insertion (complement masking): progressively preserve top-ranked evidence and track score gain.

### 6.2 Counterfactual drop

Mask top-k% evidence once (default 30%) and measure score delta.

### 6.3 Attention-gradient agreement

Spearman correlation between flattened attention and gradient maps.

## 7. UI and Workflow

### 7.1 Explore

Single-instance deep inspection with synchronized panels:

- Attention overlay
- Alignment matrix
- Token scores
- Optional faithfulness panel

### 7.2 Compare

Two prompts on same model/image for differential debugging.

### 7.3 Eval

Batch runner over JSONL to produce per-sample metrics and aggregate summaries.

## 8. Engineering Decisions

- Lazy model loading with adapter cache.
- Family-specific extraction logic with shared normalized output schema.
- Error-first handling when hidden-state or attention channels are absent.
- Minimal assumptions in LLaVA segmentation with explicit fallback paths.

## 9. Mechanistic Extensions

To move beyond output-only attribution, MMLens now includes four TransformerLens-style
mechanistic probes.

### 9.1 Forward-hook activation caching

- Auto-discovers transformer block paths (e.g. `layers.12`, `encoder.layer.5`).
- Uses `register_forward_hook` to cache per-layer tensors during a regular forward pass.
- Preserves exact tensors for later causal analysis without rewriting model architectures.

### 9.2 Cross-modal activation patching

For `(source_image, target_image, prompt)` and layer `l`:

1. Capture source activation at layer `l`.
2. Re-run target with a forward hook that replaces target activation with source activation.
3. Compare scalar scores before and after replacement.

If replacing only visual-token positions changes output score, the selected layer carries
causal visual evidence used for text generation.

### 9.3 Multimodal logit lens

- Captures hidden states across layers.
- Projects hidden vectors through output embeddings (`lm_head`/`get_output_embeddings`).
- Reads top vocabulary predictions at selected sequence positions per layer.

This yields a layer-by-layer trajectory of what the model "believes" before final decoding.

### 9.4 Grounding-head discovery

1. Build visual ablation (mask top rollout patches).
2. Run baseline and ablated forwards.
3. Compute per-head visual attention mass difference.
4. Rank heads by `|delta_visual_mass| * (1 + max(score_drop, 0))`.

Top-ranked heads represent candidate grounding circuits responsible for image-conditioned
reasoning (especially in BLIP-2 cross-attention and LLaVA decoder attention).

## 10. Limitations

- Output structures vary across checkpoint variants; some fields may be unavailable.
- Sequence log-probability proxy for generative VLMs is useful but not a perfect semantic confidence measure.
- Attention maps are not treated as causal proof; perturbation checks are mandatory for high-confidence claims.

## 11. Recommended Extension Roadmap

1. Add pinned smoke suites for each alias group using public checkpoints.
2. Add grounding benchmarks (Flickr30k-Entities/RefCOCO loaders).
3. Add hallucination benchmark protocol (POPE-style templates).
4. Add exportable experiment manifests for strict reproducibility.
